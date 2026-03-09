#!/usr/bin/env python3
"""
验证 2_cluster_qc_stations.py 的聚类结果是否正确。

检查项：
  1) 同一 cluster 内：多站时在“距离<阈值”的边下应连通（图连通性）。
  2) 不同 cluster 之间：抽样跨簇站对，不应出现距离 < 阈值的对（否则应被合并）。
  3) 输出简要统计与违规列表（若有）。

用法：
  python verify_cluster_qc_stations.py --clustered clustered_stations.csv [--threshold 0.05]
  python verify_cluster_qc_stations.py -c output/02_cluster/clustered_stations.csv -t 0.05 --sample-cross 5000
"""

import argparse
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"


def haversine_deg(lat1, lon1, lat2, lon2):
    """与 2_cluster_qc_stations 一致：近似距离（度），约 0.01°≈1.1km。"""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    d_km = 2 * R * atan2(sqrt(a), sqrt(1 - a))
    return d_km / 111.0


def haversine_km(lat1, lon1, lat2, lon2):
    """返回公里。"""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def main():
    ap = argparse.ArgumentParser(description="Verify distance clustering result")
    ap.add_argument("--clustered", "-c", default=str(OUTPUT_R_ROOT / "output/02_cluster/clustered_stations.csv"),
                   help="clustered_stations.csv 路径")
    ap.add_argument("--threshold", "-t", type=float, default=0.05,
                   help="Same threshold (degrees) used when clustering")
    ap.add_argument("--sample-cross", type=int, default=0,
                   help="Max random cross-cluster pairs to check (0=auto, ~100k)")
    ap.add_argument("--quick", action="store_true",
                   help="Skip connectivity check for clusters with >200 stations; use 5k cross sample")
    args = ap.parse_args()

    path = Path(args.clustered)
    if not path.is_absolute():
        path = OUTPUT_R_ROOT / path
    path = path.resolve()
    if not path.is_file():
        print("Error: not found: {}".format(path))
        return

    df = pd.read_csv(path)
    for col in ["lat", "lon", "cluster_id"]:
        if col not in df.columns:
            print("Error: CSV must have columns: lat, lon, cluster_id")
            return

    threshold = args.threshold
    n = len(df)
    cids = df["cluster_id"].values
    lats = df["lat"].values.astype(float)
    lons = df["lon"].values.astype(float)

    # 1) 同一 cluster 内：图连通性（边 = 距离 < 阈值的对）
    print("Check 1: Within-cluster connectivity (each cluster = one connected component under threshold)")
    violations_connect = []
    clusters = defaultdict(list)
    for i in range(n):
        clusters[cids[i]].append(i)
    n_multi = 0
    max_connect_size = 200 if args.quick else 999999
    for cid, inds in clusters.items():
        if len(inds) <= 1:
            continue
        if len(inds) > max_connect_size:
            continue  # --quick: skip very large clusters
        n_multi += 1
        # 建图：节点 inds，边 (i,j) 若 d(i,j) < threshold
        parent = {idx: idx for idx in inds}

        def find(x):
            stack = []
            while parent[x] != x:
                stack.append(x)
                x = parent[x]
            for i in stack:
                parent[i] = x
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for ii in range(len(inds)):
            for jj in range(ii + 1, len(inds)):
                i, j = inds[ii], inds[jj]
                d = haversine_deg(lats[i], lons[i], lats[j], lons[j])
                if d < threshold:
                    union(i, j)
        roots = set(find(i) for i in inds)
        if len(roots) > 1:
            violations_connect.append((cid, len(inds), len(roots)))

    if violations_connect:
        print("  FAIL: {} cluster(s) are not connected under threshold:".format(len(violations_connect)))
        for cid, size, n_comp in violations_connect[:20]:
            print("    cluster_id={} n_stations={} n_components={}".format(cid, size, n_comp))
        if len(violations_connect) > 20:
            print("    ... and {} more".format(len(violations_connect) - 20))
    else:
        print("  OK: All multi-station clusters are connected under threshold.")

    # 2) 不同 cluster 之间：抽样检查跨簇站对，不应存在距离 < 阈值的对
    print("\nCheck 2: No pair from different clusters with distance < threshold (sampled)")
    idx_by_cid = clusters  # cid -> list of row indices
    # 随机抽若干跨簇站对做检查（全量 O(n^2) 太慢）
    if args.quick:
        sample_size = 5000
    else:
        sample_size = args.sample_cross if args.sample_cross > 0 else min(100000, n * (n - 1) // 2)
    violations_cross = []
    np.random.seed(42)
    checked = 0
    for _ in range(sample_size):
        i, j = np.random.randint(0, n, 2)
        if i >= j or cids[i] == cids[j]:
            continue
        checked += 1
        d = haversine_deg(lats[i], lons[i], lats[j], lons[j])
        if d < threshold:
            violations_cross.append((cids[i], cids[j], i, j, d, df["path"].iloc[i], df["path"].iloc[j]))
    print("  Sampled {} cross-cluster pairs.".format(checked))

    if violations_cross:
        print("  FAIL: {} pair(s) from different clusters with distance < threshold:".format(len(violations_cross)))
        for c1, c2, i, j, d, p1, p2 in violations_cross[:10]:
            print("    cluster {} vs {} d_deg={:.5f} ({} km)".format(c1, c2, d, d * 111))
            print("      {}".format(Path(p1).name))
            print("      {}".format(Path(p2).name))
        if len(violations_cross) > 10:
            print("    ... and {} more".format(len(violations_cross) - 10))
    else:
        print("  OK: No cross-cluster pairs within threshold (sampled or full).")

    # 3) 统计摘要
    print("\nSummary:")
    print("  Total stations: {}".format(n))
    print("  Total clusters: {}".format(len(clusters)))
    sizes = [len(inds) for inds in clusters.values()]
    print("  Clusters with 1 station: {}".format(sum(1 for s in sizes if s == 1)))
    print("  Clusters with 2+ stations: {}".format(sum(1 for s in sizes if s >= 2)))
    if sizes:
        print("  Max cluster size: {}".format(max(sizes)))
    print("  Threshold: {} deg (~{:.2f} km)".format(threshold, threshold * 111))

    # 多站 cluster 内最大成对距离（公里）
    max_diam_km = 0
    for cid, inds in clusters.items():
        if len(inds) < 2:
            continue
        for ii in range(len(inds)):
            for jj in range(ii + 1, len(inds)):
                i, j = inds[ii], inds[jj]
                d_km = haversine_km(lats[i], lons[i], lats[j], lons[j])
                max_diam_km = max(max_diam_km, d_km)
    print("  Max pairwise distance within any cluster: {:.2f} km".format(max_diam_km))

    if violations_connect or violations_cross:
        print("\nResult: VERIFICATION FAILED")
    else:
        print("\nResult: VERIFICATION PASSED (distance merging logic is consistent)")


if __name__ == "__main__":
    main()
