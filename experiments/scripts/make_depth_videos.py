#!/usr/bin/env python3
"""
对 make_videos.py 导出的序列运行 VGGT-Omega 深度估计。

两种模式：
  默认（per-sequence）：每条序列单独推理，保存 <seq_dir>/front_depth.mp4
  --region-mode      ：区域内所有序列均匀采样后一次性输入 VGGT，
                       保存 <region_dir>/region_depth.mp4

用法:
  python experiments/scripts/make_depth_videos.py --region-id 0000
  python experiments/scripts/make_depth_videos.py --region-id 0000 --scene-name scene-0239
  python experiments/scripts/make_depth_videos.py --region-id 0000 --region-mode
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
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

DEFAULT_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "vggt_omega_1b_512.pt")
DEFAULT_WORK_DIR = os.path.join(_DEVELOP_ROOT, "work_dir")
IMG_SIZE = (640, 360)  # (W, H) 输出视频帧尺寸


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--region-id", default=None)
    p.add_argument("--scene-name", default=None)
    p.add_argument("--max-regions", type=int, default=1)
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--resolution", type=int, default=336,
                   help="VGGT 输入分辨率，越小越快；建议 336 或 512")
    p.add_argument("--chunk-size", type=int, default=10,
                   help="每次推理帧数（减小可降低显存占用）")
    p.add_argument("--colormap", default="plasma",
                   help="深度 colormap 名称（matplotlib 格式）")
    p.add_argument("--region-mode", action="store_true",
                   help="区域内所有序列一起输入 VGGT，输出 region_depth.mp4")
    p.add_argument("--region-max-frames", type=int, default=40,
                   help="region-mode 总帧数上限（按序列均匀采样）")
    p.add_argument("--with-streetview", action="store_true",
                   help="把每帧对应的街景图一起输入 VGGT")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    model = VGGTOmega().eval()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.cuda()


# ---------------------------------------------------------------------------
# Depth colorization
# ---------------------------------------------------------------------------

def depth_to_color(depth_np: np.ndarray, cmap_name: str = "plasma") -> np.ndarray:
    """
    depth_np: [H, W] float，返回 [H, W, 3] uint8 BGR。
    用 log scale 增强近距离细节。
    """
    import matplotlib
    d = depth_np.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d, 1e-3, None)
    d = np.log(d)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi > lo:
        d = (d - lo) / (hi - lo)
    else:
        d = np.zeros_like(d)
    d = np.clip(d, 0, 1)
    cmap = matplotlib.colormaps[cmap_name]
    colored = (cmap(d)[:, :, :3] * 255).astype(np.uint8)  # RGB
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_depth_chunk(model: VGGTOmega, paths: list[str], resolution: int) -> np.ndarray:
    """返回 [N, H, W] float32 深度图。"""
    images = load_and_preprocess_images(paths, image_resolution=resolution).cuda()
    with torch.autocast(device_type="cuda",
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        preds = model(images)
    depth = preds["depth"]           # [1, N, H, W, 1]
    depth = depth[0, :, :, :, 0]    # [N, H, W]
    return depth.float().cpu().numpy()


def run_sequence_depth(
    model: VGGTOmega,
    image_paths: list[str],
    resolution: int,
    chunk_size: int,
) -> np.ndarray:
    """分块推理整条序列，返回 [N, H, W] float32。"""
    all_depths = []
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i: i + chunk_size]
        depths = run_depth_chunk(model, chunk, resolution)
        all_depths.append(depths)
        print(f"    chunk {i//chunk_size + 1}/{(len(image_paths)-1)//chunk_size + 1} done")
    return np.concatenate(all_depths, axis=0)


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def _ffmpeg_encode(frames_dir: str, out_path: str, fps: float) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(frames_dir, "*.jpg"),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def write_depth_video(
    image_paths: list[str],
    depths: np.ndarray,
    out_path: str,
    fps: float,
    cmap_name: str,
) -> bool:
    W, H = IMG_SIZE
    with tempfile.TemporaryDirectory() as tmp:
        for i, (src, depth) in enumerate(zip(image_paths, depths)):
            rgb = cv2.imread(src)
            if rgb is None:
                continue
            rgb = cv2.resize(rgb, (W, H))
            dep_color = depth_to_color(depth, cmap_name)
            dep_color = cv2.resize(dep_color, (W, H))
            frame = np.concatenate([rgb, dep_color], axis=1)  # [H, 2W, 3]
            cv2.imwrite(os.path.join(tmp, f"{i:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        return _ffmpeg_encode(tmp, out_path, fps)


# ---------------------------------------------------------------------------
# Streetview helpers
# ---------------------------------------------------------------------------

def _seq_paths_with_sv(seq, dataset, sv_provider):
    """返回 [rgb..., sv...] 列表（同步过滤掉缺街景的帧）。"""
    rgb_list, sv_list = [], []
    for kf in seq.keyframes:
        rgb = dataset.resolve_path(kf.image_path)
        if not os.path.isfile(rgb):
            continue
        sv = sv_provider.get(kf.sample_token)
        if not sv:
            continue
        rgb_list.append(rgb)
        sv_list.append(sv)
    return rgb_list + sv_list, len(rgb_list)


def _run_depth_single_pass(model, paths, args, label=""):
    """一次性推理所有帧，OOM 则降分辨率重试。"""
    try:
        return run_depth_chunk(model, paths, args.resolution)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        lo_res = max(args.resolution * 2 // 3, 224)
        print(f"  OOM{label}, retry at resolution={lo_res} ...")
        return run_depth_chunk(model, paths, lo_res)


# ---------------------------------------------------------------------------
# Region mode: 所有序列一起跑 VGGT，生成 region_depth.mp4
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
        print(f"  [skip] region {region.region_id}: 无可读图片")
        return

    all_paths = rgb_all + sv_all  # 两段拼接
    out_name = "region_with_sv_depth.mp4" if sv_provider is not None else "region_depth.mp4"
    print(f"  Region {region.region_id}: {len(rgb_all)} rgb + {len(sv_all)} sv = {len(all_paths)} frames -> single VGGT pass ...")

    depths = _run_depth_single_pass(model, all_paths, args)
    out_path = os.path.join(region_dir, out_name)
    ok = write_depth_video(all_paths, depths, out_path, args.fps, args.colormap)
    print(f"  -> {'OK' if ok else 'encode failed'}: {out_path}")


# ---------------------------------------------------------------------------
# Per-sequence + streetview: 每条序列的 [rgb..., sv...] 一次推理
# ---------------------------------------------------------------------------

def process_sequence_with_sv(model, seq, dataset, seq_dir, args, sv_provider):
    paths, n_rgb = _seq_paths_with_sv(seq, dataset, sv_provider)
    if n_rgb < 2:
        print(f"  [skip] {seq.scene_name}: 有效帧 < 2 (rgb={n_rgb})")
        return
    n_sv = len(paths) - n_rgb
    print(f"  {seq.scene_name}: {n_rgb} rgb + {n_sv} sv = {len(paths)} frames -> single VGGT pass ...")
    depths = _run_depth_single_pass(model, paths, args)
    out_path = os.path.join(seq_dir, "front_with_sv_depth.mp4")
    ok = write_depth_video(paths, depths, out_path, args.fps, args.colormap)
    print(f"  -> {'OK' if ok else 'encode failed'}: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    meta_path = os.path.join(DATA_DIR, META_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"元数据不存在: {meta_path}\n请先运行: python experiments/scripts/make_videos.py"
        )

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
              f"sequences={len(region.sequences)}")
        region_dir = os.path.join(
            args.work_dir, f"region_{region.region_id}_{region.location}"
        )
        os.makedirs(region_dir, exist_ok=True)

        if args.region_mode:
            try:
                process_region(model, region, dataset, region_dir, args, sv_provider)
            except Exception as e:
                print(f"  [error] region mode: {e}")
            finally:
                torch.cuda.empty_cache()
            continue

        seqs = region.sequences
        if args.scene_name:
            seqs = [s for s in seqs if s.scene_name == args.scene_name]

        for seq in seqs:
            seq_dir = os.path.join(region_dir, seq.scene_name)
            if not os.path.isdir(seq_dir):
                print(f"  [skip] {seq.scene_name}: 先运行 make_videos.py 导出序列")
                continue

            try:
                if sv_provider is not None:
                    process_sequence_with_sv(model, seq, dataset, seq_dir, args, sv_provider)
                else:
                    out_path = os.path.join(seq_dir, "front_depth.mp4")
                    image_paths = [dataset.resolve_path(kf.image_path) for kf in seq.keyframes]
                    valid = [p for p in image_paths if os.path.isfile(p)]
                    if not valid:
                        print(f"  [skip] {seq.scene_name}: 图片不可读")
                        continue
                    print(f"  {seq.scene_name}: {len(valid)} frames, running VGGT ...")
                    depths = run_sequence_depth(model, valid, args.resolution, args.chunk_size)
                    ok = write_depth_video(valid, depths, out_path, args.fps, args.colormap)
                    print(f"  -> {'OK' if ok else 'encode failed'}: {out_path}")
            except Exception as e:
                print(f"  [error] {seq.scene_name}: {e}")
            finally:
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
