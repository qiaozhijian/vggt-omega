#!/usr/bin/env python3
"""
NuScenes 众包区域 demo：获取 CAM_FRONT 关键帧并生成视频。

首次运行时自动预处理（需要 maptr_traj.pkl），后续直接读取缓存。

用法:
  python experiments/scripts/make_videos.py                            # 第一个区域，生成视频
  python experiments/scripts/make_videos.py --region-id 0000           # 指定区域
  python experiments/scripts/make_videos.py --max-regions 3            # 处理前3个区域
  python experiments/scripts/make_videos.py --no-video                 # 只导出帧，不生成视频
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import subprocess
import tempfile

import cv2
import numpy as np

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEVELOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT    = os.path.dirname(_DEVELOP_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.core.dataset import (
    DATA_DIR,
    META_FILENAME,
    CrowdRegion,
    NuScenesCrowdDataset,
    SceneSequence,
    build_and_save_csm_meta,
)

DEFAULT_NUSC_ROOT = os.environ.get("NUSCENES_ROOT", "/media/qzj/disk-b/nuscenes")
DEFAULT_WORK_DIR = os.path.join(_DEVELOP_ROOT, "work_dir")
IMG_SIZE = (640, 360)   # (W, H) 视频帧尺寸


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nusc-root", default=DEFAULT_NUSC_ROOT)
    p.add_argument("--maptr-traj", default=None,
                   help="maptr_traj.pkl，默认 <nusc-root>/custom/maptrv2/maptr_traj.pkl")
    p.add_argument("--patch-size", type=float, default=50.0)
    p.add_argument("--min-frames", type=int, default=0,
                   help="区域内最少轨迹点数（0=不限）")
    p.add_argument("--min-traj-num", type=int, default=2,
                   help="区域内最少 scene 数")
    p.add_argument("--max-seqs-per-region", type=int, default=5,
                   help="每个众包区域最多保留 N 条序列（按帧数降序）")
    p.add_argument("--region-id", default=None)
    p.add_argument("--max-regions", type=int, default=1,
                   help="处理前 N 个区域，默认 1")
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--no-video", action="store_true", help="只导出帧，不生成视频")
    p.add_argument("--rebuild", action="store_true", help="强制重新预处理")
    return p.parse_args()


def ensure_meta(args) -> str:
    meta_path = os.path.join(DATA_DIR, META_FILENAME)
    if os.path.isfile(meta_path) and not args.rebuild:
        return meta_path
    maptr_traj = args.maptr_traj or os.path.join(
        args.nusc_root, "custom/maptrv2/maptr_traj.pkl"
    )
    if not os.path.isfile(maptr_traj):
        raise FileNotFoundError(
            f"找不到 maptr_traj.pkl: {maptr_traj}\n"
            "请用 --maptr-traj 指定路径，或 --nusc-root 指定 NuScenes 根目录。"
        )
    build_and_save_csm_meta(
        nusc_root=args.nusc_root,
        maptr_traj_pkl=maptr_traj,
        output_dir=DATA_DIR,
        patch_size=args.patch_size,
        min_frames=args.min_frames,
        min_traj_num=args.min_traj_num,
        max_seqs_per_region=args.max_seqs_per_region,
    )
    return meta_path


def read_frame(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, IMG_SIZE)


def _ffmpeg_encode(frames_dir: str, out_path: str, fps: float) -> bool:
    """用 ffmpeg 把 frames_dir/*.jpg 编成兼容 H.264 的 mp4。"""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(frames_dir, "*.jpg"),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out_path,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    return ret.returncode == 0


def write_sequence_video(
    seq: SceneSequence,
    out_path: str,
    dataset: NuScenesCrowdDataset,
    fps: float,
) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for i, kf in enumerate(seq.keyframes):
            src = dataset.resolve_path(kf.image_path)
            frame = read_frame(src)
            if frame is None:
                continue
            cv2.imwrite(os.path.join(tmp, f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        if n == 0:
            return False
        return _ffmpeg_encode(tmp, out_path, fps)


def write_tiled_video(
    region: CrowdRegion,
    out_path: str,
    dataset: NuScenesCrowdDataset,
    fps: float,
    max_cols: int = 3,
) -> bool:
    seqs = region.sequences
    if not seqs:
        return False

    n = len(seqs)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    W, H = IMG_SIZE
    grid_w, grid_h = W * cols, H * rows

    all_paths: List[List[str]] = [
        [dataset.resolve_path(kf.image_path) for kf in seq.keyframes]
        for seq in seqs
    ]
    n_frames = max(len(p) for p in all_paths)

    with tempfile.TemporaryDirectory() as tmp:
        for t in range(n_frames):
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for si, (seq, paths) in enumerate(zip(seqs, all_paths)):
                r, c = divmod(si, cols)
                idx = t % len(paths)
                frame = read_frame(paths[idx])
                if frame is None:
                    frame = np.zeros((H, W, 3), dtype=np.uint8)
                grid[r*H:(r+1)*H, c*W:(c+1)*W] = frame
            cv2.imwrite(os.path.join(tmp, f"{t:06d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return _ffmpeg_encode(tmp, out_path, fps)


def export_region(
    region: CrowdRegion,
    dataset: NuScenesCrowdDataset,
    work_dir: str,
    fps: float,
    make_video: bool,
):
    region_dir = os.path.join(work_dir, f"region_{region.region_id}_{region.location}")
    os.makedirs(region_dir, exist_ok=True)

    seq_metas = []
    for seq in region.sequences:
        seq_dir = os.path.join(region_dir, seq.scene_name)
        os.makedirs(seq_dir, exist_ok=True)

        # 保存帧路径 meta
        frames_meta = [
            {"local_idx": kf.local_idx, "timestamp": kf.timestamp,
             "sample_token": kf.sample_token, "image_path": kf.image_path}
            for kf in seq.keyframes
        ]
        json.dump({"scene_name": seq.scene_name, "scene_token": seq.scene_token,
                   "num_frames": len(frames_meta), "frames": frames_meta},
                  open(os.path.join(seq_dir, "meta.json"), "w"), indent=2)

        vid_path = None
        if make_video:
            vid_path = os.path.join(seq_dir, "front.mp4")
            ok = write_sequence_video(seq, vid_path, dataset, fps)
            if not ok:
                vid_path = None
                print(f"  [warn] {seq.scene_name}: 图片不可读，跳过视频")

        print(f"  {seq.scene_name}: {len(seq.keyframes)} frames"
              + (f" -> {vid_path}" if vid_path else ""))
        seq_metas.append({"scene_name": seq.scene_name, "num_frames": len(seq.keyframes),
                          "video": vid_path, "dir": seq_dir})

    # 整个区域的平铺视频
    if make_video and region.sequences:
        tiled_path = os.path.join(region_dir, "region_tiled.mp4")
        ok = write_tiled_video(region, tiled_path, dataset, fps)
        print(f"  Tiled video ({len(region.sequences)} seqs): {tiled_path}" if ok else "  Tiled video failed")

    summary = {"region_id": region.region_id, "center": region.center,
               "location": region.location, "sequences": seq_metas}
    json.dump(summary, open(os.path.join(region_dir, "summary.json"), "w"), indent=2)
    return region_dir


def main():
    args = parse_args()
    meta_path = ensure_meta(args)
    dataset = NuScenesCrowdDataset(meta_path)
    print(f"Loaded {len(dataset)} regions from {meta_path}")

    regions = list(dataset)
    if args.region_id:
        regions = [r for r in regions if r.region_id == args.region_id]
        if not regions:
            raise ValueError(f"找不到 region_id={args.region_id}")
    else:
        regions = regions[: args.max_regions]

    os.makedirs(args.work_dir, exist_ok=True)
    make_video = not args.no_video

    for region in regions:
        print(f"\nRegion {region.region_id} @ {region.location}  "
              f"center={region.center}  sequences={len(region.sequences)}")
        out = export_region(region, dataset, args.work_dir, args.fps, make_video)
        print(f"  -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
