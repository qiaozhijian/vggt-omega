#!/usr/bin/env python3
"""
Open3D 点云可视化。

用法:
  conda run -n csm python experiments/scripts/view_pcd.py <pcd1> [pcd2 ...]
  conda run -n csm python experiments/scripts/view_pcd.py experiments/work_dir/region_0038_boston-seaport/scene-0080/pointcloud.pcd
  conda run -n csm python experiments/scripts/view_pcd.py experiments/work_dir/region_0038_boston-seaport/pointcloud_region.pcd
  conda run -n csm python experiments/scripts/view_pcd.py --no-axis experiments/work_dir/.../*.pcd
  conda run -n csm python experiments/scripts/view_pcd.py --no-sky experiments/work_dir/.../pointcloud.pcd

窗口快捷键: 左键旋转/滚轮缩放/右键平移；R 重置视角；+ - 调点大小；B 切换背景；H 帮助。
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="一个或多个 .pcd / .ply 文件路径")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--axis-size", type=float, default=0.5,
                   help="世界坐标轴大小，默认 0.5（VGGT 输出尺度被压缩）")
    p.add_argument("--no-axis", action="store_true", help="不显示世界坐标轴")
    p.add_argument("--bg", choices=["dark", "light"], default="dark")
    p.add_argument("--no-sky", action="store_true",
                   help="按颜色启发式过滤天空点（蓝天 / 亮灰白云）")
    p.add_argument("--sky-bright", type=float, default=0.75,
                   help="亮度阈值，亮度高于此且饱和度低视为云/雾天")
    p.add_argument("--sky-sat", type=float, default=0.20,
                   help="灰白云的最大饱和度")
    return p.parse_args()


def _filter_sky(pcd, v_thresh: float, s_thresh: float):
    """按 HSV 启发式去除天空点。

    判定规则（命中任一即视为天空）：
      1) 蓝色：H ∈ [185°, 255°] 且 S > 0.15 且 V > 0.4
      2) 亮灰白云：V > v_thresh 且 S < s_thresh
    """
    import numpy as np
    import open3d as o3d

    cols = np.asarray(pcd.colors)
    if cols.size == 0:
        return pcd, 0

    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    mx = np.max(cols, axis=1)
    mn = np.min(cols, axis=1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    h = np.zeros_like(v)
    rg_mask = (mx == r) & (mx > mn)
    gb_mask = (mx == g) & (mx > mn)
    bb_mask = (mx == b) & (mx > mn)
    h[rg_mask] = ((g[rg_mask] - b[rg_mask]) / (mx[rg_mask] - mn[rg_mask])) % 6
    h[gb_mask] = (b[gb_mask] - r[gb_mask]) / (mx[gb_mask] - mn[gb_mask]) + 2
    h[bb_mask] = (r[bb_mask] - g[bb_mask]) / (mx[bb_mask] - mn[bb_mask]) + 4
    h = (h * 60.0) % 360.0

    blue_sky = (h >= 185) & (h <= 255) & (s > 0.15) & (v > 0.4)
    bright_cloud = (v > v_thresh) & (s < s_thresh)
    sky_mask = blue_sky | bright_cloud

    if not sky_mask.any():
        return pcd, 0

    pts = np.asarray(pcd.points)[~sky_mask]
    cols = cols[~sky_mask]
    new = o3d.geometry.PointCloud()
    new.points = o3d.utility.Vector3dVector(pts)
    new.colors = o3d.utility.Vector3dVector(cols)
    return new, int(sky_mask.sum())


def main():
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError:
        sys.exit("需要 open3d，请用 `conda run -n csm python experiments/scripts/view_pcd.py ...`")

    geoms = []
    total_pts = 0
    for path in args.paths:
        if not os.path.isfile(path):
            print(f"[skip] {path}: not found")
            continue
        pcd = o3d.io.read_point_cloud(path)
        n = len(pcd.points)
        if n == 0:
            print(f"[skip] {path}: empty")
            continue
        if args.no_sky:
            pcd, removed = _filter_sky(pcd, args.sky_bright, args.sky_sat)
            n_after = len(pcd.points)
            print(f"  {path}  ({n:,} -> {n_after:,} pts, -{removed:,} sky)")
            n = n_after
        else:
            print(f"  {path}  ({n:,} pts)")
        geoms.append(pcd)
        total_pts += n

    if not geoms:
        sys.exit("没有可读取的点云")

    if not args.no_axis:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size))

    print(f"Total: {total_pts:,} points in {len(args.paths)} file(s)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=" | ".join(os.path.basename(p) for p in args.paths),
                      width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = [0.05, 0.05, 0.05] if args.bg == "dark" else [1.0, 1.0, 1.0]
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
