"""
Streetview provider: 把 nuScenes sample_token 映射到对齐 CAM_FRONT 的街景图 JPG。

数据来源: nuScenes-Geography-Data
  https://huggingface.co/datasets/SpatialRetrievalAD/nuScenes-Geography-Data

流程:
  sample_token -> LIDAR_TOP ego_pose_token == frame_id
  frame_id     -> frame_metadata.json   -> pano_id
  pano_id      -> pano_metadata.json    -> pano.jpg
  pano.jpg + yaw(CAM_FRONT) -> 裁切一张对齐 CAM_FRONT 的视图

公式参考: SpatialRetrievalAD-Dataset-Devkit (extract_view_from_pano + get_camera_yaw_pitch)
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

DEFAULT_NUSC_ROOT = os.environ.get("NUSCENES_ROOT", "/media/qzj/disk-b/nuscenes")
DEFAULT_GEO_ROOT = os.path.join(DEFAULT_NUSC_ROOT, "custom/nuScenes-Geography-Data")
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "streetview_cache")

FOV_H_DEG = 70.0            # NuScenes CAM_FRONT ≈ 70°
OUT_SIZE = (640, 360)       # 与 front.mp4 相同 (W, H)
CAM_FRONT = "CAM_FRONT"
LIDAR_TOP = "LIDAR_TOP"


# ---------------------------------------------------------------------------
# 等距全景 -> 透视视图 (从 devkit pano.py 移植)
# ---------------------------------------------------------------------------

def extract_view_from_pano(pano: np.ndarray, yaw_deg: float, fov_h_deg: float,
                           out_size: tuple = (640, 640), pitch_deg: float = 0.0) -> np.ndarray:
    """
    pano: BGR 等距投影全景 [H, W, 3]
    返回与 Google StreetView API 语义一致的透视视图 (h, w, 3) BGR uint8。
    """
    pano_h, pano_w = pano.shape[:2]
    out_w, out_h = out_size

    if pano_h < pano_w // 2:
        bg = np.zeros((pano_w // 2, pano_w, 3), dtype=np.uint8)
        bg[pano_w // 4 - pano_h // 2: pano_w // 4 + pano_h // 2, :] = pano
        pano = bg
        pano_h = pano_w // 2

    fov_h_rad = np.deg2rad(fov_h_deg)
    f_x = out_w / (2 * np.tan(fov_h_rad / 2))
    f_y = f_x
    c_x, c_y = out_w / 2, out_h / 2

    xx, yy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    X = (xx - c_x) / f_x
    Y = (yy - c_y) / f_y
    Z = np.ones_like(X)
    norm = np.sqrt(X * X + Y * Y + Z * Z)
    X /= norm; Y /= norm; Z /= norm

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])
    R = R_pitch @ R_yaw

    dirs = np.stack([X, Y, Z], axis=-1) @ R.T
    Xr, Yr, Zr = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = np.arctan2(Xr, Zr)
    phi = np.arcsin(np.clip(Yr, -1, 1))
    u = ((theta + np.pi) / (2 * np.pi) * pano_w).astype(np.float32)
    v = ((np.pi / 2 - phi) / np.pi * pano_h).astype(np.float32)

    view = cv2.remap(pano, u, v, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_WRAP)
    return cv2.flip(view, 0)


# ---------------------------------------------------------------------------
# Yaw 计算 (从 devkit transform.py 移植)
# ---------------------------------------------------------------------------

def _quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """单位四元数 q=(w,x,y,z) 旋转向量 v。"""
    return _quat_to_matrix(q) @ np.asarray(v, dtype=np.float64)


def _compute_yaw(cam_rotation: Sequence[float], ego_rotation: Sequence[float]) -> float:
    """根据 CAM_FRONT 的相机标定旋转和 ego_pose 旋转，得到与全景图对齐用的 yaw（度）。"""
    cam_forward = _quat_rotate(cam_rotation, [0.0, 0.0, 1.0])
    # devkit: cam_forward[0], cam_forward[1] = cam_forward[1], -cam_forward[0]
    cam_forward = np.array([cam_forward[1], -cam_forward[0], cam_forward[2]])
    world_forward = _quat_rotate(ego_rotation, cam_forward)
    yaw = -np.degrees(np.arctan2(world_forward[1], world_forward[0]))
    return float(yaw)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class StreetviewProvider:
    """
    按需为 sample_token 生成对齐 CAM_FRONT 的街景视图。
    第一次访问会裁切并缓存到 experiments/data/streetview_cache/<sample_token>.jpg。
    """

    def __init__(self,
                 nusc_root: str = DEFAULT_NUSC_ROOT,
                 geo_root: str = DEFAULT_GEO_ROOT,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 versions: Sequence[str] = ("v1.0-trainval", "v1.0-test"),
                 pano_lru: int = 16):
        self.nusc_root = nusc_root
        self.geo_root = geo_root
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.frame_meta: Dict[str, dict] = self._read_json(os.path.join(geo_root, "frame_metadata.json"))
        self.pano_meta:  Dict[str, dict] = self._read_json(os.path.join(geo_root, "pano_metadata.json"))
        self.unavailable: set = set(self._read_json(os.path.join(geo_root, "unavailable_metadata.json")).keys())

        # sample_token -> {frame_id, cam_rot, ego_rot}
        self._sample_info: Dict[str, dict] = {}
        for ver in versions:
            self._index_version(ver)

        self._pano_cache: Dict[str, np.ndarray] = {}
        self._pano_lru = pano_lru
        self._lock = threading.Lock()

    @staticmethod
    def _read_json(path: str):
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _index_version(self, version: str) -> None:
        ver_dir = os.path.join(self.nusc_root, version)
        if not os.path.isdir(ver_dir):
            return

        def _read(name):
            p = os.path.join(ver_dir, f"{name}.json")
            return self._read_json(p) if os.path.isfile(p) else []

        sensor = _read("sensor")
        calib = _read("calibrated_sensor")
        ego = _read("ego_pose")
        sd = _read("sample_data")

        sensor_chan = {s["token"]: s["channel"] for s in sensor}
        cs_chan = {c["token"]: sensor_chan.get(c["sensor_token"], "") for c in calib}
        cs_by_tok = {c["token"]: c for c in calib}
        ego_by_tok = {e["token"]: e for e in ego}

        # sample_token -> CAM_FRONT sd, LIDAR_TOP sd
        cam_front_sd: Dict[str, dict] = {}
        lidar_sd: Dict[str, dict] = {}
        for r in sd:
            if not r.get("is_key_frame"):
                continue
            ch = cs_chan.get(r["calibrated_sensor_token"], "")
            if ch == CAM_FRONT:
                cam_front_sd[r["sample_token"]] = r
            elif ch == LIDAR_TOP:
                lidar_sd[r["sample_token"]] = r

        for stok, cf in cam_front_sd.items():
            ld = lidar_sd.get(stok)
            if ld is None:
                continue
            try:
                cs = cs_by_tok[cf["calibrated_sensor_token"]]
                eg = ego_by_tok[cf["ego_pose_token"]]
            except KeyError:
                continue
            self._sample_info[stok] = {
                "frame_id": ld["ego_pose_token"],
                "cam_rot": cs["rotation"],
                "ego_rot": eg["rotation"],
            }

    def _load_pano(self, pano_id: str) -> Optional[np.ndarray]:
        with self._lock:
            cached = self._pano_cache.get(pano_id)
            if cached is not None:
                return cached
        info = self.pano_meta.get(pano_id)
        if not info:
            return None
        path = os.path.join(self.geo_root, info["pano_path"])
        if not os.path.isfile(path):
            return None
        pano = cv2.imread(path)
        if pano is None:
            return None
        with self._lock:
            if len(self._pano_cache) >= self._pano_lru:
                # 简单丢一项
                self._pano_cache.pop(next(iter(self._pano_cache)))
            self._pano_cache[pano_id] = pano
        return pano

    def get(self, sample_token: str) -> Optional[str]:
        cached_path = os.path.join(self.cache_dir, f"{sample_token}.jpg")
        if os.path.isfile(cached_path):
            return cached_path

        info = self._sample_info.get(sample_token)
        if info is None:
            return None
        frame_id = info["frame_id"]
        if frame_id in self.unavailable:
            return None
        pano_id = (self.frame_meta.get(frame_id) or {}).get("pano_id")
        if not pano_id:
            return None
        pano = self._load_pano(pano_id)
        if pano is None:
            return None

        yaw = _compute_yaw(info["cam_rot"], info["ego_rot"])
        view = extract_view_from_pano(pano, yaw_deg=yaw, fov_h_deg=FOV_H_DEG, out_size=OUT_SIZE)
        cv2.imwrite(cached_path, view, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return cached_path

    def get_many(self, sample_tokens: Sequence[str]) -> Dict[str, Optional[str]]:
        return {t: self.get(t) for t in sample_tokens}
