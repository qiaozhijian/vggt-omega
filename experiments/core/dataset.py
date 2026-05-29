"""
NuScenes 众包区域 (CSM) dataloader：仅前向相机关键帧序列。

预处理一次后从 experiments/data/csm_regions.pkl 读取，运行时无需 nuscenes-devkit。
众包区域划分逻辑对齐 csm-devkit/library/pipelines/csm_split_pipeline.py。
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import shapely.geometry as geom
from scipy.spatial import KDTree

CAM_FRONT = "CAM_FRONT"
META_FILENAME = "csm_regions.pkl"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def _quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _rt2se3(r, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(r, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _se3_to_xy(se3: np.ndarray) -> np.ndarray:
    """提取 SE3 的 XY 平移（lidar/ego 2D 位置）。"""
    from scipy.spatial.transform import Rotation as R
    yaw = R.from_matrix(se3[:3, :3]).as_euler("zyx", degrees=True)[0]
    se2 = np.eye(4, dtype=np.float64)
    se2[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix()
    se2[:2, 3] = se3[:2, 3]
    return se2[:2, 3]


# ---------------------------------------------------------------------------
# Lightweight NuScenes JSON index（不用 devkit，直接读 JSON 表）
# ---------------------------------------------------------------------------

class NuScenesJsonIndex:
    """
    从 v1.0-* JSON 表构建索引，不依赖 devkit。
    注意：NuScenes sample.json 不含 data 字段，CAM_FRONT 通过
    sample_data + calibrated_sensor + sensor 三表关联获取。
    """

    def __init__(self, dataroot: str, versions: Sequence[str] = ("v1.0-trainval", "v1.0-test")):
        self.dataroot = os.path.abspath(dataroot)
        self.sample: Dict[str, dict] = {}
        # sample_token -> CAM_FRONT sample_data row
        self.front_sd: Dict[str, dict] = {}
        self.calibrated_sensor: Dict[str, dict] = {}
        self.ego_pose: Dict[str, dict] = {}
        for ver in versions:
            self._load(ver)

    def _load(self, version: str) -> None:
        def _read(name):
            path = os.path.join(self.dataroot, version, f"{name}.json")
            return json.load(open(path)) if os.path.isfile(path) else []

        for row in _read("sample"):
            self.sample[row["token"]] = row
        for row in _read("calibrated_sensor"):
            self.calibrated_sensor[row["token"]] = row
        for row in _read("ego_pose"):
            self.ego_pose[row["token"]] = row

        # sensor_token -> channel
        sensor_channel = {r["token"]: r["channel"] for r in _read("sensor")}
        # calibrated_sensor_token -> channel
        cs_channel = {
            r["token"]: sensor_channel.get(r["sensor_token"], "")
            for r in self.calibrated_sensor.values()
        }
        for row in _read("sample_data"):
            if (row.get("is_key_frame")
                    and cs_channel.get(row["calibrated_sensor_token"]) == CAM_FRONT):
                self.front_sd[row["sample_token"]] = row

    def cam_front_path(self, sample_token: str) -> str:
        sd = self.front_sd[sample_token]
        return os.path.join(self.dataroot, sd["filename"])

    def cam_front_pose(self, sample_token: str) -> np.ndarray:
        """返回 CAM_FRONT 的 cam2global (4x4)。"""
        sd = self.front_sd[sample_token]
        cs = self.calibrated_sensor[sd["calibrated_sensor_token"]]
        ego = self.ego_pose[sd["ego_pose_token"]]
        c2e = _rt2se3(_quat_to_matrix(cs["rotation"]), cs["translation"])
        e2g = _rt2se3(_quat_to_matrix(ego["rotation"]), ego["translation"])
        return e2g @ c2e


# ---------------------------------------------------------------------------
# CSM 区域构建
# ---------------------------------------------------------------------------

def _load_trajectories(maptr_traj_pkl: str) -> Tuple[Dict, dict]:
    """返回 (trajectories, maptr_results)。"""
    data = pickle.load(open(maptr_traj_pkl, "rb"))
    results = data.get("results", data)
    trajectories: Dict[str, Dict[str, np.ndarray]] = {}
    for frames in results.values():
        if not frames:
            continue
        loc = frames[0]["map_location"]
        name = frames[0]["scene_name"]
        traj = np.array([_se3_to_xy(_rt2se3(f["lidar2global_rotation"], f["lidar2global_translation"]))
                         for f in frames])
        trajectories.setdefault(loc, {})[name] = traj
    return trajectories, results


def _build_candidates(
    trajectories: Dict,
    patch_size: float,
    min_frames: int,
    min_traj_num: int,
) -> List[dict]:
    candidates = []
    for loc, trajs_dict in trajectories.items():
        trajs = np.concatenate(list(trajs_dict.values()))
        # 体素下采样候选点
        grid = np.floor(trajs).astype(np.int64)
        _, idx = np.unique(grid, axis=0, return_index=True)
        waypoints = trajs[np.sort(idx)]

        tree = KDTree(trajs)
        # 每个候选点的邻居数（square patch）
        counts = {}
        for i, wp in enumerate(waypoints):
            nbrs = tree.query_ball_point(wp, patch_size * np.sqrt(2))
            nbrs = [j for j in nbrs if np.linalg.norm(wp - trajs[j]) < patch_size]
            counts[i] = len(nbrs)

        # NMS：按邻居数降序，间距 < patch_size 的去掉（保证区域间无几何重叠）
        valid = sorted([i for i in counts if counts[i] >= min_frames],
                       key=lambda x: counts[x], reverse=True)
        keep = np.ones(len(valid), dtype=bool)
        for i in range(len(valid)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(valid)):
                if keep[j] and np.linalg.norm(waypoints[valid[i]] - waypoints[valid[j]]) < patch_size:
                    keep[j] = False
        valid = [valid[i] for i in range(len(valid)) if keep[i]]

        for i in valid:
            center = [int(c) for c in waypoints[i]]
            patch = geom.box(center[0] - patch_size/2, center[1] - patch_size/2,
                             center[0] + patch_size/2, center[1] + patch_size/2)
            scene_names = sorted(
                name for name, traj in trajs_dict.items()
                if patch.intersects(geom.LineString(traj))
            )
            if len(scene_names) >= min_traj_num:
                candidates.append({"center": center, "location": loc, "scene_names": scene_names})
    return candidates


def _scene_frames_in_patch(center, patch_size, scene_name, maptr_results) -> List[Tuple[int, dict]]:
    scene_frames = next((v for v in maptr_results.values()
                         if v and v[0]["scene_name"] == scene_name), None)
    if scene_frames is None:
        return []
    area = geom.box(center[0] - patch_size/2, center[1] - patch_size/2,
                    center[0] + patch_size/2, center[1] + patch_size/2)
    return [
        (i, f) for i, f in enumerate(scene_frames)
        if area.contains(geom.Point(_se3_to_xy(_rt2se3(f["lidar2global_rotation"], f["lidar2global_translation"]))))
    ]


def build_and_save_csm_meta(
    nusc_root: str,
    maptr_traj_pkl: str,
    output_dir: str = DATA_DIR,
    patch_size: float = 50.0,
    min_frames: int = 0,
    min_traj_num: int = 2,
    max_regions: Optional[int] = None,
    max_seqs_per_region: Optional[int] = 5,
    nusc_versions: Sequence[str] = ("v1.0-trainval", "v1.0-test"),
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading trajectories from {maptr_traj_pkl} ...")
    trajectories, maptr_results = _load_trajectories(maptr_traj_pkl)
    candidates = _build_candidates(trajectories, patch_size, min_frames, min_traj_num)
    if max_regions is not None:
        candidates = candidates[:max_regions]
    print(f"Found {len(candidates)} candidate regions, building sequences ...")

    nusc_idx = NuScenesJsonIndex(nusc_root, versions=nusc_versions)
    regions = []
    for idx, cand in enumerate(candidates):
        sequences = []
        for scene_name in cand["scene_names"]:
            patch_frames = _scene_frames_in_patch(cand["center"], patch_size, scene_name, maptr_results)
            keyframes = []
            for local_idx, mf in patch_frames:
                tok = mf.get("sample_token")
                if not tok or tok not in nusc_idx.front_sd:
                    continue
                keyframes.append({
                    "local_idx": local_idx,
                    "timestamp": int(mf["timestamp"]),
                    "sample_token": tok,
                    "image_path": nusc_idx.cam_front_path(tok),
                    "ego2global": _rt2se3(mf["lidar2global_rotation"], mf["lidar2global_translation"]).tolist(),
                    "cam2global": nusc_idx.cam_front_pose(tok).tolist(),
                })
            if keyframes:
                sequences.append({
                    "scene_name": scene_name,
                    "scene_token": patch_frames[0][1].get("scene_token", ""),
                    "keyframes": keyframes,
                })
        if not sequences:
            continue
        # 按帧数降序，保留最多 max_seqs_per_region 条序列
        if max_seqs_per_region and len(sequences) > max_seqs_per_region:
            sequences.sort(key=lambda s: len(s["keyframes"]), reverse=True)
            sequences = sequences[:max_seqs_per_region]
        regions.append({
            "region_id": str(len(regions)).zfill(4),
            "center": cand["center"],
            "location": cand["location"],
            "scene_names": cand["scene_names"],
            "patch_size": patch_size,
            "sequences": sequences,
        })

    meta = {
        "nusc_root": os.path.abspath(nusc_root),
        "patch_size": patch_size,
        "regions": regions,
    }
    meta_path = os.path.join(output_dir, META_FILENAME)
    pickle.dump(meta, open(meta_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(regions)} regions -> {meta_path}")
    return meta_path


# ---------------------------------------------------------------------------
# Runtime dataloader
# ---------------------------------------------------------------------------

@dataclass
class FrontKeyframe:
    local_idx: int
    timestamp: int
    sample_token: str
    image_path: str
    ego2global: np.ndarray
    cam2global: np.ndarray


@dataclass
class SceneSequence:
    scene_name: str
    scene_token: str
    keyframes: List[FrontKeyframe]

    def image_paths(self) -> List[str]:
        return [kf.image_path for kf in self.keyframes]


@dataclass
class CrowdRegion:
    region_id: str
    center: List[int]
    location: str
    scene_names: List[str]
    patch_size: float
    sequences: List[SceneSequence] = field(default_factory=list)


class NuScenesCrowdDataset:
    """迭代众包区域，每个区域包含多条历史 scene 的 CAM_FRONT 关键帧序列。"""

    def __init__(self, meta_path: str = os.path.join(DATA_DIR, META_FILENAME)):
        self.meta = pickle.load(open(meta_path, "rb"))
        self.nusc_root = self.meta["nusc_root"]
        self.regions = [self._parse(r) for r in self.meta["regions"]]

    @staticmethod
    def _parse(raw: dict) -> CrowdRegion:
        seqs = []
        for s in raw["sequences"]:
            kfs = [FrontKeyframe(
                local_idx=k["local_idx"],
                timestamp=k["timestamp"],
                sample_token=k["sample_token"],
                image_path=k["image_path"],
                ego2global=np.array(k["ego2global"]),
                cam2global=np.array(k["cam2global"]),
            ) for k in s["keyframes"]]
            seqs.append(SceneSequence(s["scene_name"], s.get("scene_token", ""), kfs))
        return CrowdRegion(raw["region_id"], raw["center"], raw["location"],
                           raw["scene_names"], raw["patch_size"], seqs)

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, idx: int) -> CrowdRegion:
        return self.regions[idx]

    def __iter__(self) -> Iterator[CrowdRegion]:
        yield from self.regions

    def resolve_path(self, path: str) -> str:
        if os.path.isfile(path):
            return path
        alt = os.path.join(self.nusc_root, path)
        return alt if os.path.isfile(alt) else path
