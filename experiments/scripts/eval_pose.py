#!/usr/bin/env python3
"""
VGGT-Omega 位姿评测：在 NuScenes val 上随机采样若干前视相机短片段，
用真值 cam2global 做参考，Umeyama (sim3) 对齐 VGGT 估计的相机轨迹
后报告 scale、平均旋转/平移误差。

用法:
  conda run -n csm python experiments/scripts/eval_pose.py
  conda run -n csm python experiments/scripts/eval_pose.py --num-clips 20 --frames 17 --hz 10
  conda run -n csm python experiments/scripts/eval_pose.py --nusc-root /path/to/nuscenes --seed 0

输出:
  experiments/work_dir/eval_pose/result.json   每个 clip 的 scale / rot_err / trans_err
  控制台打印聚合统计。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_DEVELOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT    = os.path.dirname(_DEVELOP_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.core.dataset import _quat_to_matrix, _rt2se3
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

DEFAULT_CKPT      = os.path.join(_REPO_ROOT, "checkpoints", "vggt_omega_1b_512.pt")
DEFAULT_NUSC_ROOT = os.environ.get("NUSCENES_ROOT", "/media/qzj/disk-b/nuscenes")
DEFAULT_OUTPUT    = os.path.join(_DEVELOP_ROOT, "work_dir", "eval_pose")

CAM_FRONT = "CAM_FRONT"


# ---------------------------------------------------------------------------
# Lightweight NuScenes index：所有 CAM_FRONT sample_data（含非关键帧）
# ---------------------------------------------------------------------------

@dataclass
class FrontSweep:
    timestamp: int            # 微秒
    sample_data_token: str
    filename: str
    cs_token: str
    ego_pose_token: str


class NuScenesValIndex:
    """索引 val split 中每个 scene 的 CAM_FRONT sweep 列表（按时间排序）。"""

    def __init__(self, dataroot: str, val_scenes: List[str], version: str = "v1.0-trainval"):
        self.dataroot = os.path.abspath(dataroot)
        self.version = version

        def _read(name):
            return json.load(open(os.path.join(self.dataroot, version, f"{name}.json")))

        sensor = {r["token"]: r["channel"] for r in _read("sensor")}
        cs_rows = _read("calibrated_sensor")
        self.calibrated_sensor = {r["token"]: r for r in cs_rows}
        cs_channel = {t: sensor.get(r["sensor_token"], "") for t, r in self.calibrated_sensor.items()}

        self.ego_pose = {r["token"]: r for r in _read("ego_pose")}

        scene_rows = _read("scene")
        name2scene = {r["name"]: r for r in scene_rows}
        val_scene_tokens = {name2scene[n]["token"] for n in val_scenes if n in name2scene}

        sample_rows = _read("sample")
        sample2scene = {r["token"]: r["scene_token"] for r in sample_rows}

        scene_front: dict[str, list[FrontSweep]] = {}
        for row in _read("sample_data"):
            if cs_channel.get(row["calibrated_sensor_token"]) != CAM_FRONT:
                continue
            scene_token = sample2scene.get(row["sample_token"])
            if scene_token not in val_scene_tokens:
                continue
            scene_front.setdefault(scene_token, []).append(FrontSweep(
                timestamp=int(row["timestamp"]),
                sample_data_token=row["token"],
                filename=row["filename"],
                cs_token=row["calibrated_sensor_token"],
                ego_pose_token=row["ego_pose_token"],
            ))

        self.scene_front = {tok: sorted(v, key=lambda s: s.timestamp) for tok, v in scene_front.items()}
        self.name2token = name2scene
        self.token2name = {tok: name for name, tok in {r["name"]: r["token"] for r in scene_rows}.items()}

    def sample_clip(self, scene_token: str, num_frames: int, hz: float, rng: random.Random) -> Optional[List[FrontSweep]]:
        """从一个 scene 的 CAM_FRONT 帧链中按 1/hz 间隔取 num_frames 个最近帧。"""
        sweeps = self.scene_front.get(scene_token, [])
        if len(sweeps) < num_frames:
            return None
        ts = np.array([s.timestamp for s in sweeps], dtype=np.int64)
        clip_us = int(1_000_000 / hz)
        total_span = clip_us * (num_frames - 1)
        if ts[-1] - ts[0] < total_span:
            return None
        max_start = ts[-1] - total_span
        start_t = rng.randint(int(ts[0]), int(max_start))
        targets = start_t + np.arange(num_frames) * clip_us
        idx = np.searchsorted(ts, targets)
        idx = np.clip(idx, 1, len(ts) - 1)
        left = ts[idx - 1]; right = ts[idx]
        choose_right = (np.abs(right - targets) < np.abs(left - targets))
        idx = np.where(choose_right, idx, idx - 1)
        # 去重保证帧间不重复（同一时刻可能取到同 sweep）
        if len(set(idx.tolist())) < num_frames:
            return None
        return [sweeps[i] for i in idx]

    def cam_front_pose(self, sw: FrontSweep) -> np.ndarray:
        """返回 CAM_FRONT 的 cam2global (4x4)。"""
        cs  = self.calibrated_sensor[sw.cs_token]
        ego = self.ego_pose[sw.ego_pose_token]
        c2e = _rt2se3(_quat_to_matrix(cs["rotation"]),  cs["translation"])
        e2g = _rt2se3(_quat_to_matrix(ego["rotation"]), ego["translation"])
        return e2g @ c2e

    def image_path(self, sw: FrontSweep) -> str:
        return os.path.join(self.dataroot, sw.filename)


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """估计将 src 变换到 dst 的 (s, R, t)，使 ||s R src + t - dst||^2 最小。

    src, dst: [N, 3]，对应点。返回 (s, R, t)。
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = (src_c ** 2).sum() / len(src_c)
    cov = dst_c.T @ src_c / len(src_c)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if with_scale and var_src > 1e-12:
        s = (D * np.diag(S)).sum() / var_src
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return float(s), R, t


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def relative_pose(M0_inv: np.ndarray, Mi: np.ndarray) -> np.ndarray:
    return M0_inv @ Mi


# ---------------------------------------------------------------------------
# Val split
# ---------------------------------------------------------------------------

def load_val_scenes() -> List[str]:
    """优先用 nuscenes-devkit 的 splits，否则回退到 hardcoded（这里只装 devkit）。"""
    from nuscenes.utils.splits import val
    return list(val)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str) -> VGGTOmega:
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.cuda()


@torch.inference_mode()
def run_vggt(model: VGGTOmega, paths: List[str], resolution: int) -> np.ndarray:
    """返回 extrinsic [N, 3, 4]，cam_i-from-world (= cam_i-from-cam_0)。"""
    images = load_and_preprocess_images(paths, image_resolution=resolution).cuda()
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=amp):
        preds = model(images)
    H, W = images.shape[-2], images.shape[-1]
    ext, _ = encoding_to_camera(preds["pose_enc"], (H, W))
    return ext[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# 单 clip 评测
# ---------------------------------------------------------------------------

def _to_se3(rt_3x4: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :4] = rt_3x4
    return M


def plot_trajectories(trajectories: list, results: list, out_path: str):
    """
    把每个 clip 的尺度对齐后 pred 轨迹 vs GT 轨迹画成 5×4 网格 (cam_0 系俯视图 X-Z)。
      X 右为 +，Z 前为 +；第 0 帧在 (0,0)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(trajectories)
    if n == 0:
        return
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for i, (tr, res) in enumerate(zip(trajectories, results)):
        ax = axes[i // cols, i % cols]
        gt = tr["gt_t"]; pr = tr["pr_t"]
        ax.plot(gt[:, 0], gt[:, 2], "-o", color="#1f77b4", ms=3, lw=1.5, label="GT")
        ax.plot(pr[:, 0], pr[:, 2], "-x", color="#d62728", ms=4, lw=1.2, label="VGGT (×s)")
        ax.scatter([0], [0], c="black", s=30, marker="s", zorder=5)
        # 等比 + 紧贴数据
        all_pts = np.concatenate([gt[:, [0, 2]], pr[:, [0, 2]]])
        lo = all_pts.min(axis=0); hi = all_pts.max(axis=0)
        ctr = (lo + hi) / 2
        span = max(hi - lo).max() if (hi - lo).max() > 0 else 1.0
        pad = span * 0.15 + 0.1
        ax.set_xlim(ctr[0] - span / 2 - pad, ctr[0] + span / 2 + pad)
        ax.set_ylim(ctr[1] - span / 2 - pad, ctr[1] + span / 2 + pad)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(
            f"#{res['clip_index']:02d} {res['scene_name']}\n"
            f"s={res['scale_pred_to_gt']:.1f}  "
            f"rot={res['rot_err_mean_deg']:.2f}°  "
            f"trans={res['trans_err_mean_m']:.2f}m",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    # 隐藏未用 axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle("Trajectories in cam_0 frame (X-Z top-down, scale-aligned)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_clip(model, paths: List[str], gt_cam2global: List[np.ndarray], resolution: int,
                  return_traj: bool = False):
    """
    约定:
      - GT M_i = cam_i → world (cam2global, 4x4)
      - VGGT extrinsic[i] = cam_i ← world (3x4)
      - 评测位姿 P_i := cam_i 在 cam_0 系中的 pose（4x4）
          GT  : P_gt[i]   = inv(M_0) @ M_i
          Pred: P_pred[i] = inv(ext_4x4[i]) @ ext_4x4[0]   (等价于 M_pred[0] @ inv(M_pred[i]))

    因为两边 cam_0 物理上是同一个相机帧，初始坐标系自动对齐 (P_*[0] = I)。
    剩下唯一未知量是 VGGT 输出的相对尺度，做最小二乘标量解：
        s = argmin sum_i || s * pr_t[i] - gt_t[i] ||^2 = (Σ pr_t·gt_t) / (Σ pr_t·pr_t)
    旋转两边都在同一个 cam_0 系中表达，直接比较相对旋转。
    """
    N = len(paths)
    ext = run_vggt(model, paths, resolution)                       # [N, 3, 4]
    M_pred = [_to_se3(e) for e in ext]                             # cam_i ← world
    P_pred = np.stack([M_pred[0] @ np.linalg.inv(M_pred[i]) for i in range(N)])

    M0_inv = np.linalg.inv(gt_cam2global[0])
    P_gt   = np.stack([M0_inv @ Mi for Mi in gt_cam2global])

    pr_R, pr_t = P_pred[:, :3, :3], P_pred[:, :3, 3]
    gt_R, gt_t = P_gt[:, :3, :3],  P_gt[:, :3, 3]

    denom = float((pr_t * pr_t).sum())
    s = float((pr_t * gt_t).sum() / denom) if denom > 1e-12 else 1.0
    pr_t_aligned = s * pr_t

    rot_errs   = np.array([rotation_angle_deg(pr_R[i].T @ gt_R[i]) for i in range(N)])
    trans_errs = np.linalg.norm(pr_t_aligned - gt_t, axis=1)

    metrics = {
        "scale_pred_to_gt": s,
        "rot_err_mean_deg":  float(rot_errs.mean()),
        "rot_err_max_deg":   float(rot_errs.max()),
        "trans_err_mean_m":  float(trans_errs.mean()),
        "trans_err_max_m":   float(trans_errs.max()),
        "gt_total_disp_m":   float(np.linalg.norm(gt_t[-1] - gt_t[0])),
        "n_frames":          int(len(paths)),
    }
    if return_traj:
        return metrics, gt_t, pr_t_aligned
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default=DEFAULT_CKPT)
    p.add_argument("--nusc-root",  default=DEFAULT_NUSC_ROOT)
    p.add_argument("--num-clips",  type=int,   default=20)
    p.add_argument("--frames",     type=int,   default=17)
    p.add_argument("--hz",         type=float, default=10.0)
    p.add_argument("--resolution", type=int,   default=336)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--min-gt-span", type=float, default=1.0,
                   help="GT 起止平移模长下限（米）；低于此值视为静止 clip 跳过")
    p.add_argument("--max-tries",  type=int,   default=200,
                   help="最多尝试的 scene/起点次数，避免死循环")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    val_scenes = load_val_scenes()
    print(f"val split: {len(val_scenes)} scenes")

    print("Building NuScenes index (val) ...")
    idx = NuScenesValIndex(args.nusc_root, val_scenes)
    valid_tokens = [tok for tok in idx.scene_front if len(idx.scene_front[tok]) >= args.frames]
    rng.shuffle(valid_tokens)
    print(f"{len(valid_tokens)} val scenes have enough CAM_FRONT frames")

    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt)

    results = []
    trajectories = []                                       # 用于画图
    tried = 0
    skipped_static = 0
    while len(results) < args.num_clips and tried < args.max_tries:
        tried += 1
        scene_token = rng.choice(valid_tokens)
        scene_name = idx.token2name[scene_token]
        clip = idx.sample_clip(scene_token, args.frames, args.hz, rng)
        if clip is None:
            continue
        paths = [idx.image_path(sw) for sw in clip]
        if not all(os.path.isfile(p) for p in paths):
            continue
        try:
            gt_poses = [idx.cam_front_pose(sw) for sw in clip]
            gt_span = float(np.linalg.norm(gt_poses[-1][:3, 3] - gt_poses[0][:3, 3]))
            if gt_span < args.min_gt_span:
                skipped_static += 1
                continue
            res, gt_t, pr_t_aligned = evaluate_clip(
                model, paths, gt_poses, args.resolution, return_traj=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip OOM] {scene_name}")
            continue
        except Exception as e:
            print(f"  [skip error] {scene_name}: {e}")
            continue
        res["scene_name"] = scene_name
        res["clip_index"] = len(results)
        results.append(res)
        trajectories.append({"scene_name": scene_name, "gt_t": gt_t, "pr_t": pr_t_aligned})
        print(f"  clip {res['clip_index']:02d}  {scene_name}  "
              f"s={res['scale_pred_to_gt']:.3f}  "
              f"rot={res['rot_err_mean_deg']:.2f}°  "
              f"trans={res['trans_err_mean_m']:.3f}m  "
              f"(GT span {res['gt_total_disp_m']:.1f}m)")
        torch.cuda.empty_cache()

    if not results:
        sys.exit("No valid clips evaluated.")

    scales = np.array([r["scale_pred_to_gt"] for r in results])
    rot    = np.array([r["rot_err_mean_deg"] for r in results])
    tr     = np.array([r["trans_err_mean_m"]  for r in results])
    span   = np.array([r["gt_total_disp_m"]   for r in results])

    summary = {
        "num_clips":          int(len(results)),
        "tries_total":        int(tried),
        "skipped_static":     int(skipped_static),
        "min_gt_span_m":      float(args.min_gt_span),
        "frames_per_clip":    int(args.frames),
        "hz":                 float(args.hz),
        "scale_mean":         float(scales.mean()),
        "scale_median":       float(np.median(scales)),
        "scale_std":          float(scales.std()),
        "rot_err_mean_deg":   float(rot.mean()),
        "rot_err_median_deg": float(np.median(rot)),
        "trans_err_mean_m":   float(tr.mean()),
        "trans_err_median_m": float(np.median(tr)),
        "gt_span_mean_m":     float(span.mean()),
    }

    out = {"summary": summary, "clips": results}
    out_path = os.path.join(args.output_dir, "result.json")
    json.dump(out, open(out_path, "w"), indent=2)

    fig_path = os.path.join(args.output_dir, "trajectories.png")
    plot_trajectories(trajectories, results, fig_path)

    # 保存轨迹原始 numpy（便于复用）
    np.savez(os.path.join(args.output_dir, "trajectories.npz"),
             **{f"gt_{i:02d}":   tr["gt_t"] for i, tr in enumerate(trajectories)},
             **{f"pred_{i:02d}": tr["pr_t"] for i, tr in enumerate(trajectories)})

    print("\n========== SUMMARY ==========")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(f"\nDetails -> {out_path}")
    print(f"Trajectory plot -> {fig_path}")


if __name__ == "__main__":
    main()
