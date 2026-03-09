#!/usr/bin/env python3
"""
第二步：对已收集的站点（collected_stations.csv）按经纬度距离聚类，输出带 cluster_id 的 CSV。

输入：collected_stations.csv（列 path, source, lat, lon, resolution；resolution 来自 s1）
输出：clustered_stations.csv（多一列 cluster_id，保留 resolution）、merge_qc_nc_report.csv
供 merge_qc_nc_by_location.py 使用 --clustered-stations 跳过第一步和第二步。

用法：
  python 2_cluster_qc_stations.py [--out-dir output/02_cluster] [--threshold 0.05] [-j 32]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"  # 数据根目录 Output_r


def get_memory_mb():
    """当前进程内存占用（MB），失败返回 None。"""
    if _HAS_PSUTIL:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None


def print_memory(label=""):
    m = get_memory_mb()
    if m is not None:
        print("  [memory] {} {:.1f} MB".format(label, m).strip())


def haversine_deg(lat1, lon1, lat2, lon2):
    """近似距离（度）。约 0.01° ≈ 1.1 km。"""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    d_km = 2 * R * atan2(sqrt(a), sqrt(1 - a))
    return d_km / 111.0


_cluster_coords = None
_cluster_threshold = None


def _cluster_init(coords_threshold):
    global _cluster_coords, _cluster_threshold
    _cluster_coords, _cluster_threshold = coords_threshold


def _cluster_worker(args):
    chunk = args
    pairs = []
    for inds, neighbor_lists in chunk:
        for neig in neighbor_lists:
            for ii in inds:
                for jj in neig:
                    if ii >= jj:
                        continue
                    d = haversine_deg(
                        _cluster_coords[ii, 0], _cluster_coords[ii, 1],
                        _cluster_coords[jj, 0], _cluster_coords[jj, 1],
                    )
                    if d < _cluster_threshold:
                        pairs.append((ii, jj))
    return pairs


def cluster_stations(stations_df, threshold_deg, workers=1, memory_print=False, max_memory_mb=0, check_memory=None):
    """按经纬度距离聚类，返回 cluster_id 数组。"""
    n = len(stations_df)
    parent = list(range(n))
    coords = np.asarray(stations_df[["lat", "lon"]].values, dtype=np.float64)
    if check_memory:
        check_memory()

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

    cell_size = max(threshold_deg * 1.5, 0.001)
    grid = defaultdict(list)
    for i in range(n):
        ki = (int(coords[i, 0] / cell_size), int(coords[i, 1] / cell_size))
        grid[ki].append(i)
    if memory_print:
        print_memory("after grid:")
    if check_memory:
        check_memory()

    if workers <= 1:
        for (ci, cj), inds in grid.items():
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    neigs = grid.get((ci + di, cj + dj), [])
                    for ii in inds:
                        for jj in neigs:
                            if ii == jj or find(ii) == find(jj):
                                continue
                            d = haversine_deg(
                                coords[ii, 0], coords[ii, 1],
                                coords[jj, 0], coords[jj, 1],
                            )
                            if d < threshold_deg:
                                union(ii, jj)
    else:
        cell_tasks = []
        max_inds_per_task = 2000
        for (ci, cj), inds in grid.items():
            neighbor_lists = [grid.get((ci + di, cj + dj), []) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
            if len(inds) <= max_inds_per_task:
                cell_tasks.append((inds, neighbor_lists))
            else:
                # 大格按 workers 数分块，保证至少 workers 个 task，用满多核
                n_parts = min(workers, max(1, (len(inds) + max_inds_per_task - 1) // max_inds_per_task))
                if n_parts < workers and len(inds) >= workers:
                    n_parts = workers
                for i in range(n_parts):
                    start = (i * len(inds)) // n_parts
                    end = ((i + 1) * len(inds)) // n_parts
                    if start < end:
                        cell_tasks.append((inds[start:end], neighbor_lists))
        n_chunks = min(workers, max(1, len(cell_tasks)))
        chunk_size = (len(cell_tasks) + n_chunks - 1) // n_chunks
        tasks = [cell_tasks[i:i + chunk_size] for i in range(0, len(cell_tasks), chunk_size)]
        print("  Phase 1: computing distance pairs in parallel ({} tasks)...".format(len(tasks)))
        with Pool(
            len(tasks),
            initializer=_cluster_init,
            initargs=((coords, threshold_deg),),
        ) as pool:
            results = pool.map(_cluster_worker, tasks)
        if memory_print:
            print_memory("after Phase 1 (distance pairs done):")
        if check_memory:
            check_memory()
        total_pairs = sum(len(p) for p in results)
        print("  Phase 2: merging {} pairs (single-threaded union)...".format(total_pairs))
        done = 0
        step = max(1, total_pairs // 20)
        for pairs in results:
            for i, j in pairs:
                union(i, j)
                done += 1
                if done % step == 0 and done > 0:
                    print("    union progress: {}/{} ({:.0f}%)".format(done, total_pairs, 100.0 * done / total_pairs))
                    if memory_print:
                        print_memory("  ")
                    if check_memory:
                        check_memory()

    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    cluster_ids = np.zeros(n, dtype=int)
    for cid, inds in enumerate(comp.values()):
        for i in inds:
            cluster_ids[i] = cid
    return cluster_ids


def main():
    ap = argparse.ArgumentParser(description="Step 2: cluster stations by distance, add cluster_id")
    ap.add_argument("--input", "-i", metavar="CSV",
                    default=str(OUTPUT_R_ROOT / "output/01_collect/collected_stations.csv"),
                    help="Input: collected_stations.csv (path, source, lat, lon)")
    ap.add_argument("--out-dir", "-o",
                    default=str(OUTPUT_R_ROOT / "output/02_cluster"),
                    help="Output directory for clustered_stations.csv and report")
    ap.add_argument("--threshold", "-t", type=float, default=0.05,
                    help="Distance threshold in degrees (~0.01°≈1.1km)")
    ap.add_argument("--workers", "-j", type=int, default=0,
                    help="Parallel workers; 0=auto (cpu_count-1, max 32)")
    ap.add_argument("--max-memory", "-M", type=float, default=0, metavar="MB",
                    help="Abort if process memory exceeds this (MB); 0=no limit")
    ap.add_argument("--no-memory-print", action="store_true",
                    help="Do not print memory at each stage")
    args = ap.parse_args()

    t0 = time.perf_counter()
    inp = Path(args.input)
    if not inp.is_file():
        print("Error: input file not found: {}".format(inp))
        return

    workers = args.workers if args.workers > 0 else min(32, max(1, (cpu_count() or 2) - 1))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def check_memory():
        if args.max_memory <= 0:
            return
        m = get_memory_mb()
        if m is not None and m > args.max_memory:
            print("Error: memory {:.1f} MB exceeds --max-memory {:.0f} MB. Aborting.".format(m, args.max_memory))
            sys.exit(1)

    print("Input:  {}".format(inp))
    print("Output: {}".format(out_dir))
    print("Loading {} ...".format(inp))
    stations = pd.read_csv(inp)
    if not args.no_memory_print:
        print_memory("after load CSV:")
    check_memory()
    for col in ["path", "source", "lat", "lon"]:
        if col not in stations.columns:
            print("Error: input CSV must have columns: path, source, lat, lon.")
            return
    if "resolution" not in stations.columns:
        stations["resolution"] = "unknown"
    print("Loaded {} stations.".format(len(stations)))

    print("Clustering by distance (threshold_deg = {}, workers={}) ...".format(args.threshold, workers))
    stations["cluster_id"] = cluster_stations(
        stations, args.threshold, workers=workers,
        memory_print=not args.no_memory_print,
        max_memory_mb=args.max_memory,
        check_memory=check_memory,
    )
    if not args.no_memory_print:
        print_memory("after clustering:")
    check_memory()
    n_clusters = stations["cluster_id"].nunique()
    print("Clusters: {}.".format(n_clusters))

    clustered_path = out_dir / "clustered_stations.csv"
    stations.to_csv(clustered_path, index=False)
    print("Saved {}.".format(clustered_path))

    report_rows = []
    for cid in range(n_clusters):
        members = stations[stations["cluster_id"] == cid]
        rep = members.iloc[0]
        report_rows.append({
            "cluster_id": cid,
            "lat": rep["lat"],
            "lon": rep["lon"],
            "n_stations": len(members),
            "resolutions": ",".join(sorted(members["resolution"].unique())),
            "sources": ",".join(sorted(members["source"].unique())),
            "paths": "; ".join(members["path"].tolist()[:3]) + (" ..." if len(members) > 3 else ""),
        })
    report_path = out_dir / "merge_qc_nc_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print("Saved {}.".format(report_path))

    elapsed = time.perf_counter() - t0
    print("Run time: {:.2f} s ({:.2f} min)".format(elapsed, elapsed / 60.0))
    time_log_path = out_dir / "step2_run_time.txt"
    with open(time_log_path, "a") as f:
        f.write("{}\tthreshold={}\tworkers={}\telapsed_s={:.2f}\tn_stations={}\tn_clusters={}\n".format(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            args.threshold,
            workers,
            elapsed,
            len(stations),
            n_clusters,
        ))
    print("Run time logged to {}.".format(time_log_path))
    print("Next: run s3_merge_timeseries_by_cluster.py with --clustered {}".format(clustered_path))


if __name__ == "__main__":
    main()
