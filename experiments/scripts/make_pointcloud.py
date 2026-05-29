#!/usr/bin/env python3
"""
VGGT-Omega 深度 → 彩色点云。

两种模式：
  默认（per-sequence）：每条序列单独推理，保存 <seq_dir>/pointcloud.pcd
  --region-mode       ：把整个众包区域所有序列的帧一次性输入 VGGT，
                        保存 <region_dir>/pointcloud_region.pcd

用法:
  conda run -n csm python experiments/scripts/make_pointcloud.py --region-id 0000
  conda run -n csm python experiments/scripts/make_pointcloud.py --region-id 0000 --region-mode
  conda run -n csm python experiments/scripts/make_pointcloud.py --max-regions 3
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import torch

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEVELOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT    = os.path.dirname(_DEVELOP_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.core.dataset import DATA_DIR, META_FILENAME, NuScenesCrowdDataset
from experiments.core.streetview import StreetviewProvider
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

DEFAULT_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "vggt_omega_1b_512.pt")
DEFAULT_WORK_DIR = os.path.join(_DEVELOP_ROOT, "work_dir")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--region-id", default=None)
    p.add_argument("--scene-name", default=None)
    p.add_argument("--max-regions", type=int, default=1)
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--resolution", type=int, default=336)
    p.add_argument("--chunk-size", type=int, default=10,
                   help="per-sequence 模式每次推理帧数")
    p.add_argument("--voxel", type=float, default=0.01,
                   help="体素下采样（VGGT 相对尺度），0=不下采样")
    p.add_argument("--max-depth", type=float, default=60.0,
                   help="最大深度截断（米）")
    # region 模式
    p.add_argument("--region-mode", action="store_true",
                   help="将区域内所有序列帧一起输入 VGGT 做单次推理")
    p.add_argument("--region-max-frames", type=int, default=40,
                   help="region-mode 总帧数上限（均匀采样各序列）")
    p.add_argument("--with-streetview", action="store_true",
                   help="把每帧对应的街景图一起输入 VGGT")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.cuda()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(model: VGGTOmega, paths: list[str], resolution: int):
    """
    返回 (depth [N,H,W,1], conf [N,H,W], extrinsic [N,3,4], intrinsic [N,3,3])。
    """
    images = load_and_preprocess_images(paths, image_resolution=resolution).cuda()
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=amp):
        preds = model(images)
    H, W = images.shape[-2], images.shape[-1]
    ext, intr = encoding_to_camera(preds["pose_enc"], (H, W))
    depth     = preds["depth"][0].float().cpu().numpy()       # [N,H,W,1]
    conf      = preds["depth_conf"][0].float().cpu().numpy()  # [N,H,W]
    extrinsic = ext[0].float().cpu().numpy()                  # [N,3,4]
    intrinsic = intr[0].float().cpu().numpy()                 # [N,3,3]
    return depth, conf, extrinsic, intrinsic


# ---------------------------------------------------------------------------
# Unproject
# ---------------------------------------------------------------------------

def unproject(
    depth: np.ndarray,      # [N,H,W,1]
    conf: np.ndarray,       # [N,H,W]
    extrinsic: np.ndarray,  # [N,3,4]  cam←world (OpenCV)
    intrinsic: np.ndarray,  # [N,3,3]
    rgb_images: np.ndarray, # [N,H,W,3] float32 0~1
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    仅用深度范围过滤（不用置信度百分位），保留动态车辆等低置信度点。
    只去掉：无效值、深度≤0、深度>max_depth。
    返回 (points [M,3], colors [M,3])。
    """
    N, H, W, _ = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y = np.broadcast_to(y[None], (N, H, W))
    x = np.broadcast_to(x[None], (N, H, W))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    d  = depth[..., 0]

    cam_pts = np.stack([
        (x - cx) / fx * d,
        (y - cy) / fy * d,
        d,
    ], axis=-1)  # [N,H,W,3]

    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    world_pts = np.einsum(
        "nij,nhwj->nhwi",
        np.transpose(R, (0, 2, 1)),
        cam_pts - t[:, None, None, :],
    )  # [N,H,W,3]

    mask = (
        np.isfinite(world_pts).all(axis=-1)
        & (d > 0.1)
        & (d < max_depth)
    )
    # 额外：去掉 conf=0 的完全无效点（非百分位过滤）
    mask &= (conf > 0)

    return world_pts[mask].astype(np.float32), rgb_images[mask]


# ---------------------------------------------------------------------------
# open3d helpers
# ---------------------------------------------------------------------------

def make_pcd(pts: np.ndarray, col: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1).astype(np.float64))
    return pcd


def read_rgb(path: str, W: int, H: int) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.resize(img, (W, H))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_pcd(pcd: o3d.geometry.PointCloud, path: str, voxel: float) -> int:
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    o3d.io.write_point_cloud(path, pcd)
    return len(pcd.points)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_inference_safe(model, paths, args):
    """一次性推理，OOM 则降分辨率重试。"""
    try:
        return run_inference(model, paths, args.resolution)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        lo_res = max(args.resolution * 2 // 3, 224)
        print(f"  OOM, retry at resolution={lo_res} ...")
        return run_inference(model, paths, lo_res)


def _unproject_save(model, all_paths, args, out_path, label=""):
    depth, conf, extrinsic, intrinsic = _run_inference_safe(model, all_paths, args)
    H, W = depth.shape[1], depth.shape[2]
    rgb_np = np.stack([read_rgb(p, W, H) for p in all_paths])
    pts, col = unproject(depth, conf, extrinsic, intrinsic, rgb_np, args.max_depth)
    print(f"  {label}raw points: {len(pts):,}")
    pcd = make_pcd(pts, col)
    n = save_pcd(pcd, out_path, args.voxel)
    print(f"  After voxel({args.voxel}m): {n:,} pts  -> {out_path}")


# ---------------------------------------------------------------------------
# Per-sequence mode
# ---------------------------------------------------------------------------

def process_sequence(model, seq, dataset, seq_dir, args, sv_provider=None):
    if sv_provider is not None:
        rgb_list, sv_list = [], []
        for kf in seq.keyframes:
            rgb = dataset.resolve_path(kf.image_path)
            if not os.path.isfile(rgb):
                continue
            sv = sv_provider.get(kf.sample_token)
            if not sv:
                continue
            rgb_list.append(rgb); sv_list.append(sv)
        if len(rgb_list) < 2:
            print(f"  [skip] {seq.scene_name}: 有效帧 < 2 (rgb={len(rgb_list)})")
            return
        all_paths = rgb_list + sv_list
        print(f"  {seq.scene_name}: {len(rgb_list)} rgb + {len(sv_list)} sv = {len(all_paths)} frames -> single VGGT pass ...")
        out = os.path.join(seq_dir, "pointcloud_with_sv.pcd")
        _unproject_save(model, all_paths, args, out)
        torch.cuda.empty_cache()
        return

    image_paths = [dataset.resolve_path(kf.image_path) for kf in seq.keyframes]
    valid = [p for p in image_paths if os.path.isfile(p)]
    if not valid:
        print(f"  [skip] {seq.scene_name}: 图片不可读")
        return

    print(f"  {seq.scene_name}: {len(valid)} frames -> single VGGT pass ...")
    out = os.path.join(seq_dir, "pointcloud.pcd")
    _unproject_save(model, valid, args, out)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Region mode: 所有序列帧一起输入 VGGT
# ---------------------------------------------------------------------------

def process_region(model, region, dataset, region_dir, args, sv_provider=None):
    seqs = region.sequences
    if not seqs:
        return
    n_seqs = len(seqs)
    max_per_seq = max(1, args.region_max_frames // n_seqs)

    rgb_all: list[str] = []
    sv_all: list[str] = []
    for seq in seqs:
        rgb_paths, sv_paths = [], []
        for kf in seq.keyframes:
            rgb = dataset.resolve_path(kf.image_path)
            if not os.path.isfile(rgb):
                continue
            if sv_provider is not None:
                sv = sv_provider.get(kf.sample_token)
                if not sv:
                    continue
                sv_paths.append(sv)
            rgb_paths.append(rgb)
        if not rgb_paths:
            continue
        if len(rgb_paths) > max_per_seq:
            idx = np.linspace(0, len(rgb_paths) - 1, max_per_seq, dtype=int)
            rgb_paths = [rgb_paths[i] for i in idx]
            if sv_paths:
                sv_paths = [sv_paths[i] for i in idx]
        rgb_all.extend(rgb_paths)
        sv_all.extend(sv_paths)

    if not rgb_all:
        return

    all_paths = rgb_all + sv_all
    if sv_provider is not None:
        out = os.path.join(region_dir, "pointcloud_region_with_sv.pcd")
        print(f"  Region {region.region_id}: {len(rgb_all)} rgb + {len(sv_all)} sv = {len(all_paths)} frames -> single VGGT pass ...")
    else:
        out = os.path.join(region_dir, "pointcloud_region.pcd")
        print(f"  Region {region.region_id}: {len(all_paths)} frames ({n_seqs} seqs) -> single VGGT pass ...")
    _unproject_save(model, all_paths, args, out)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    meta_path = os.path.join(DATA_DIR, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到元数据: {meta_path}，请先运行 make_videos.py")

    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt)

    sv_provider = StreetviewProvider() if args.with_streetview else None
    if sv_provider is not None:
        print(f"Streetview indexed: {len(sv_provider._sample_info)} samples")

    dataset = NuScenesCrowdDataset(meta_path)
    regions = list(dataset)
    if args.region_id:
        regions = [r for r in regions if r.region_id == args.region_id]
    else:
        regions = regions[: args.max_regions]

    for region in regions:
        print(f"\nRegion {region.region_id} @ {region.location}  "
              f"center={region.center}  sequences={len(region.sequences)}")
        region_dir = os.path.join(
            args.work_dir, f"region_{region.region_id}_{region.location}"
        )
        os.makedirs(region_dir, exist_ok=True)

        if args.region_mode:
            try:
                process_region(model, region, dataset, region_dir, args, sv_provider)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [error] region mode: {e}")
        else:
            seqs = region.sequences
            if args.scene_name:
                seqs = [s for s in seqs if s.scene_name == args.scene_name]
            for seq in seqs:
                seq_dir = os.path.join(region_dir, seq.scene_name)
                os.makedirs(seq_dir, exist_ok=True)
                try:
                    process_sequence(model, seq, dataset, seq_dir, args, sv_provider)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  [error] {seq.scene_name}: {e}")
                finally:
                    torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
