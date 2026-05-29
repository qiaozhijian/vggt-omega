# experiments/ — NuScenes 众包区域 + VGGT-Omega 重建

围绕"众包区域" (Crowd-sourced Mapping, CSM) 跑 VGGT-Omega 推理的小工具集。
区域划分逻辑对齐 `csm-devkit/library/pipelines/csm_split_pipeline.py`，但本目录的运行时**不依赖 nuscenes-devkit**，预处理一次后直接读 pkl 缓存。

## 目录

```
experiments/
├── core/        # 库代码（被 import）
│   ├── dataset.py        CSM dataloader（直接读 v1.0-*/sample*.json，不依赖 devkit）
│   └── streetview.py     按需裁切 CAM_FRONT 对齐的街景 JPG，缓存到 data/streetview_cache/
├── scripts/     # CLI 入口
│   ├── make_videos.py        步骤 1：元数据 + RGB 视频
│   ├── make_depth_videos.py  步骤 2：RGB+Depth 并排视频（per-seq / region-mode）
│   ├── make_pointcloud.py    步骤 3：反投影彩色点云（per-seq / region-mode）
│   ├── select_regions.py     从 488 候选区域里挑时间差异最大的 N 个
│   ├── view_pcd.py           Open3D 点云可视化（含天空启发式过滤）
│   └── eval_pose.py          在 val 上跑 VGGT 位姿评测（scale / 旋转 / 平移）
├── data/        # csm_regions.pkl + streetview_cache/（gitignored）
└── work_dir/    # 视频/点云/评测输出（gitignored）
```

所有脚本入口路径都是 `experiments/scripts/<name>.py`。

## 前置条件

- NuScenes 完整数据（`v1.0-trainval` + `v1.0-test`，含 `samples/CAM_FRONT/`）
- `maptr_traj.pkl`（默认路径 `<nusc_root>/custom/maptrv2/maptr_traj.pkl`）
- VGGT-Omega checkpoint：`checkpoints/vggt_omega_1b_512.pt`
- conda env：`csm`（torch + open3d 0.19 + matplotlib + ffmpeg）
- 【街景实验需要】nuScenes-Geography-Data，默认 `<nusc_root>/custom/nuScenes-Geography-Data/`（含 `streetview/panos/`、`frame_metadata.json`、`pano_metadata.json`、`unavailable_metadata.json`）

可通过 `--nusc-root` 或 `NUSCENES_ROOT` 环境变量覆盖默认根目录 `/media/qzj/disk-b/nuscenes`。

## 三步流水线

```bash
# 步骤 1: 元数据 + RGB 视频（首次自动预处理 488 个区域）
conda run -n csm python experiments/scripts/make_videos.py --max-regions 5

# 可选：按"时间跨度最大"挑出差异最大的 N 个区域
python experiments/scripts/select_regions.py 5
# 输出例如: REGION_IDS=0038 0050 0325 0318 0305

# 步骤 2: 深度视频（基于步骤 1 的序列）
for rid in 0038 0050 0325 0318 0305; do
  # 2a) per-sequence：每条序列各自的 RGB+Depth
  conda run -n csm python experiments/scripts/make_depth_videos.py --region-id $rid
  # 2b) region-mode：所有序列一起输入 VGGT，区域级 RGB+Depth
  conda run -n csm python experiments/scripts/make_depth_videos.py --region-id $rid --region-mode
done

# 步骤 3: 点云
for rid in 0038 0050 0325 0318 0305; do
  # 3a) per-sequence：每条序列独立点云
  conda run -n csm python experiments/scripts/make_pointcloud.py --region-id $rid
  # 3b) region-mode：区域内所有序列帧一起输入 VGGT 做单次推理
  conda run -n csm python experiments/scripts/make_pointcloud.py --region-id $rid --region-mode
done
```

## 加入街景的两个实验

加 `--with-streetview`：脚本会按需裁切 CAM_FRONT 对齐的街景图（首次访问会写到 `data/streetview_cache/<sample_token>.jpg`），与真实序列**两段拼接** `[rgb_0..rgb_N, sv_0..sv_N]` 一次输入 VGGT。

```bash
for rid in 0038 0050 0325 0318 0305; do
  # 实验 3：每条序列的真实序列 + 街景序列一起跑
  conda run -n csm python experiments/scripts/make_depth_videos.py --region-id $rid --with-streetview
  conda run -n csm python experiments/scripts/make_pointcloud.py   --region-id $rid --with-streetview

  # 实验 4：区域内所有序列 + 街景一起跑（一次大 batch）
  conda run -n csm python experiments/scripts/make_depth_videos.py --region-id $rid --region-mode --with-streetview
  conda run -n csm python experiments/scripts/make_pointcloud.py   --region-id $rid --region-mode --with-streetview
done
```

## VGGT 位姿评测（val split）

在 NuScenes val 上随机采样若干 CAM_FRONT 短片段（17 帧 @ 10 Hz，含非关键帧），用真值 `cam2global` 做参考，强制 `cam_0` 对齐后只解一个**尺度比** `s`，再报告平均旋转/平移误差。

```bash
# 默认：20 clips × 17 帧 × 10 Hz，过滤 GT 位移 < 1m 的静止 clip
conda run -n csm python experiments/scripts/eval_pose.py

# 可自定义
conda run -n csm python experiments/scripts/eval_pose.py \
  --num-clips 20 --frames 17 --hz 10 --seed 42 --min-gt-span 1.0
```

输出：
- `work_dir/eval_pose/result.json` 每 clip 的 `scale_pred_to_gt`、`rot_err_mean_deg`、`trans_err_mean_m` + 聚合统计
- `work_dir/eval_pose/trajectories.png` 5×4 网格俯视图（cam_0 系 X-Z），蓝 GT vs 红 VGGT(×s)
- `work_dir/eval_pose/trajectories.npz` 原始轨迹（每 clip 的 `gt_{i:02d}` / `pred_{i:02d}`）

> 公式：把 VGGT `extrinsic[i]` (cam_i ← world) 转成 `P_pred[i] = M_pred[0] @ inv(M_pred[i])`（cam_i 在 cam_0 系下的 pose），与 GT 同样定义对齐。直线运动下 Umeyama sim3 退化，所以这里**只估 scale**：`s = (Σ pr_t·gt_t) / (Σ pr_t·pr_t)`。

> 缺街景的帧（落在 `unavailable_metadata.json` 里）会同时从真实和街景两个列表中丢弃，避免对齐错位。
> 街景默认 FOV=70°，输出尺寸 640×360（与 `front.mp4` 一致）。

## 输出结构

```
work_dir/region_<id>_<location>/
├── region_tiled.mp4               # 区域内多序列 RGB 并排预览
├── region_depth.mp4               # region-mode：所有序列帧→VGGT 单次推理→RGB+Depth 视频
├── region_with_sv_depth.mp4       # 实验 4：rgb + 街景一起跑的 RGB+Depth 视频
├── pointcloud_region.pcd          # region-mode：整体点云
├── pointcloud_region_with_sv.pcd  # 实验 4：rgb + 街景一起跑的整体点云
├── summary.json
└── scene-XXXX/
    ├── meta.json
    ├── front.mp4                  # RGB 序列视频
    ├── front_depth.mp4            # per-sequence: RGB + 深度并排
    ├── front_with_sv_depth.mp4    # 实验 3：每条序列 rgb + 街景一起跑
    ├── pointcloud.pcd             # per-sequence 点云
    └── pointcloud_with_sv.pcd     # 实验 3：rgb + 街景一起跑的点云
```

## 点云可视化（Open3D）

`view_pcd.py` 一键打开，base 或 csm 环境都可用：

```bash
# 单个序列点云
python experiments/scripts/view_pcd.py \
  experiments/work_dir/region_0038_boston-seaport/scene-0080/pointcloud.pcd

# 区域整体点云
python experiments/scripts/view_pcd.py \
  experiments/work_dir/region_0038_boston-seaport/pointcloud_region.pcd

# 多文件同时显示（支持 shell 通配符）
python experiments/scripts/view_pcd.py \
  experiments/work_dir/region_0038_boston-seaport/scene-*/pointcloud.pcd

# 调点大小 / 背景 / 隐藏坐标轴 / 改坐标轴尺寸
python experiments/scripts/view_pcd.py --point-size 3 --bg light --no-axis <pcd>
python experiments/scripts/view_pcd.py --axis-size 0.2 <pcd>     # 坐标轴更小

# 启发式去除天空（蓝天 / 亮灰白云，按 HSV 颜色过滤）
python experiments/scripts/view_pcd.py --no-sky <pcd>
python experiments/scripts/view_pcd.py --no-sky --sky-bright 0.7 --sky-sat 0.25 <pcd>
```

窗口快捷键：左键旋转 / 滚轮缩放 / 右键平移；`R` 重置视角；`+` `-` 调点大小；`B` 切背景；`H` 打开帮助。

> base 环境（Python 3.13）需手动装一次 Open3D（官方 PyPI 尚无 cp313 wheel）：
> ```bash
> pip install "https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-0.19.0-cp313-cp313-manylinux_2_35_x86_64.whl"
> ```
> SSH/远程需要 X11 转发（`ssh -X`），无 GUI 时改用 [CloudCompare](https://www.cloudcompare.org/) 或 `pcl_viewer` 离线查看。

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--patch-size`            | `50.0`  | 区域尺寸（米）。NMS 距离 = patch_size，确保区域间无几何重叠 |
| `--max-seqs-per-region`   | `5`     | 每个区域最多保留 N 条序列（按帧数降序截断） |
| `--min-traj-num`          | `2`     | 区域内最少 scene 数 |
| `--max-depth`             | `60.0`  | 点云深度截断（米）。**不**做置信度百分位过滤，保留动态车辆 |
| `--voxel`                 | `0.05`  | 点云体素下采样（米），`0` 关闭 |
| `--resolution`            | `336`   | VGGT 输入分辨率 |
| `--chunk-size`            | `10`    | per-sequence 推理批大小 |
| `--region-max-frames`     | `40`    | region-mode 总帧数上限（按序列均匀采样） |

## 已知注意点

- **不过滤动态车辆**：之前用 `conf_percentile=15` 会把动态车辆当低置信点滤掉，本目录改用纯深度截断（`max_depth`）+ `conf > 0`（去无效点）。
- **`matplotlib` 兼容**：`get_cmap` 在 3.5+ 已移除，统一用 `matplotlib.colormaps[name]` 下标访问。
- **MP4 编码**：用 `ffmpeg + libx264 + yuv420p + +faststart`，VS Code / 浏览器都能直接播放。
