#!/usr/bin/env python3
"""
从 csm_regions.pkl 中挑选时间跨度最大的 N 个区域。
评分：区域内各序列时间戳的跨度（秒），跨度越大说明覆盖时间越长，天气/季节差异越大。
还保证5个选出区域中心互相距离 >= 300m（来自不同地理位置）。
"""
import datetime, os, pickle, sys
import numpy as np

DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
META_FILE  = os.path.join(DATA_DIR, "csm_regions.pkl")
TOP_N      = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MIN_DIST   = 300.0   # 选出区域中心最小间距（米/地图坐标）


def region_timestamps(region):
    """取区域内所有序列第一帧的 timestamp（微秒）"""
    ts = []
    for seq in region["sequences"]:
        kfs = seq.get("keyframes", [])
        if kfs:
            ts.append(kfs[0]["timestamp"])
    return ts


def main():
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
    regions = data["regions"]
    print(f"Total regions: {len(regions)}")

    scored = []
    for r in regions:
        ts = region_timestamps(r)
        if len(ts) < 2:
            continue
        time_range_s = (max(ts) - min(ts)) / 1e6
        center = np.array(r["center"], dtype=float)
        scored.append((time_range_s, r["region_id"], center, r))

    scored.sort(key=lambda x: -x[0])

    # 贪心选 TOP_N，保证中心间距 >= MIN_DIST
    selected = []
    for score, rid, center, r in scored:
        if any(np.linalg.norm(center - c2) < MIN_DIST for _, _, c2, _ in selected):
            continue
        selected.append((score, rid, center, r))
        if len(selected) >= TOP_N:
            break

    print(f"\nSelected {len(selected)} most temporally diverse regions:")
    for score, rid, center, r in selected:
        ts = region_timestamps(r)
        dates = sorted(set(
            datetime.datetime.utcfromtimestamp(t / 1e6).strftime("%Y-%m-%d")
            for t in ts
        ))
        names = [s["scene_name"] for s in r["sequences"]]
        print(f"  region {rid}  span={score/3600:.1f}h  dates={dates}  seqs={names}")

    ids = " ".join(s[1] for s in selected)
    print(f"\nREGION_IDS={ids}")


if __name__ == "__main__":
    main()
