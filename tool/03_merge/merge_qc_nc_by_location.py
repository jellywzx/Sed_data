#!/usr/bin/env python3
"""
从所有 qc 文件夹中的 nc 数据按“经纬度邻近”合并站点；
时间有重叠时保留多源，供人工挑选可信数据。

规则：
  1. 经纬度相邻很近的站点合并为同一虚拟站点（由 --threshold 控制，单位度）
  2. 同一虚拟站内，时间重叠的日期会输出多行（每行一个数据源），并标记 is_overlap=True，
     便于在 Excel/编辑器中手动保留你认为可信的一行、删除其余

并行核数：
  --workers N 或 -j N：全程共用。① 收集站点 ② 按距离聚类（并行算距离对）③ 合并时间序列（按 cluster 并行）；N=0 表示自动（cpu_count-1，且不超过 32）

最终输出：
  1) merge_qc_nc_report.csv
     每个 cluster 一行：cluster_id, lat, lon, n_stations, sources, paths
  2) merged_qc_cluster_{id}_lat{lat}_lon{lon}.csv（每个有数据的 cluster 一个）
     列：date, Q, SSC, SSL, source, is_overlap
     is_overlap=True 的行表示该日期有多源，需人工保留一条

用法（内置配置，直接运行即可）：
  直接运行：python 3_merge_qc_nc_by_location.py
  所有参数已在脚本顶部 BUILDIN_CONFIG 中配置，修改该字典即可，无需命令行。
  可选：仍可通过命令行覆盖，如 --threshold 0.02 --no-output-nc
"""

import os
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc4
    from netCDF4 import num2date
    HAS_NC = True
except ImportError:
    HAS_NC = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
    def _progress_write(*args, **kwargs):
        tqdm.write(*args, **kwargs)
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable
    def _progress_write(*args, **kwargs):
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = _SCRIPT_DIR.parent.parent.parent / "Output_r"
_DEFAULT_MERGED = OUTPUT_R_ROOT / "output"
BUILDIN_CONFIG = {
    "root": str(OUTPUT_R_ROOT),
    "out_dir": "output/03_merge/clusterd_stations",
    "threshold": 0.05,              # 经纬度聚类阈值（度），约 0.01°≈1.1km
    "workers": 32,                  # 0=自动
    "output_nc": True,              # True=每个 cluster 同时输出 .nc；False 或命令行 --no-output-nc 则不输出
    "out_nc_dir": "",               # 空=用 out_dir/merged_nc；也可指定绝对路径
    "dry_run": False,
    "only_collect": False,
    "stations_cache": str(OUTPUT_R_ROOT / "output/01_collect" / "collected_stations.csv"),
    "clustered_stations": str(OUTPUT_R_ROOT / "output/02_cluster" / "clustered_stations.csv"),
}

LAT_NAMES = ["lat", "latitude", "Latitude"]
LON_NAMES = ["lon", "longitude", "Longitude"]
TIME_NAMES = ["time", "Time", "t", "sample"]
Q_NAMES = ["Q", "discharge", "Discharge_m3_s", "Discharge"]
SSC_NAMES = ["SSC", "ssc", "TSS_mg_L", "TSS"]
SSL_NAMES = ["SSL", "sediment_load", "Sediment_load"]
FILL = -9999.0


def _get_scalar(var):
    if var is None:
        return None
    a = np.asarray(var).flatten()
    if a.size == 0:
        return None
    v = float(a.flat[0])
    if np.isnan(v) or v == FILL or v == -9999:
        return np.nan
    return v


def _get_var(ds, names, default=np.nan):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds.variables[n][:]).flatten()
    return np.full(1, default)


def get_lat_lon_from_nc(path):
    """从 nc 文件读取标量 lat, lon。失败返回 (None, None)。"""
    if not HAS_NC:
        return None, None
    try:
        with nc4.Dataset(path, "r") as nc:
            lat_var = next((x for x in LAT_NAMES if x in nc.variables), None)
            lon_var = next((x for x in LON_NAMES if x in nc.variables), None)
            if lat_var is None or lon_var is None:
                return None, None
            lat = _get_scalar(nc.variables[lat_var][:])
            lon = _get_scalar(nc.variables[lon_var][:])
            if lat is None or lon is None or (np.isnan(lat) or np.isnan(lon)):
                return None, None
            return float(lat), float(lon)
    except Exception:
        return None, None


def get_source_from_path(path, root_dir):
    """从相对路径解析数据源名，如 daily/GloRiSe/SS/qc/xxx.nc -> GloRiSe_SS。"""
    try:
        rel = Path(path).relative_to(Path(root_dir))
        parts = rel.parts
        if "qc" in parts:
            idx = parts.index("qc")
            before = parts[:idx]
            if len(before) >= 2:
                source = "_".join(before[1:])
            else:
                source = before[0] if before else "unknown"
        else:
            source = parts[0] if parts else "unknown"
        return re.sub(r"[^\w\-]", "_", source)
    except Exception:
        return "unknown"


def _collect_one_nc(path, root_dir):
    """Worker: 读单个 nc 的 path/source/lat/lon。用于多进程，必须是模块级函数才能被 pickle。"""
    try:
        lat, lon = get_lat_lon_from_nc(path)
        if lat is None or lon is None:
            return None
        source = get_source_from_path(path, root_dir)
        return {"path": path, "source": source, "lat": lat, "lon": lon}
    except (ValueError, OSError):
        return None


def collect_qc_nc_stations(root_dir, workers=1):
    """收集所有 qc 目录下的 .nc 文件，返回 DataFrame: path, source, lat, lon。"""
    root = Path(root_dir).resolve()
    paths = []
    for p in root.rglob("*.nc"):
        try:
            if "qc" in p.relative_to(root).parts:
                paths.append(str(p))
        except ValueError:
            continue
    if not paths:
        return pd.DataFrame()

    root_str = str(root)
    if workers <= 1:
        rows = [_collect_one_nc(p, root_str) for p in tqdm(paths, desc="收集站点", unit="nc")]
    else:
        with Pool(min(workers, len(paths), cpu_count() or 1)) as pool:
            rows = list(tqdm(
                pool.starmap(_collect_one_nc, [(p, root_str) for p in paths]),
                total=len(paths), desc="收集站点", unit="nc",
            ))
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def haversine_deg(lat1, lon1, lat2, lon2):
    """近似距离（度）。约 0.01° ≈ 1.1 km。"""
    R = 6371.0  # km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    d_km = 2 * R * atan2(sqrt(a), sqrt(1 - a))
    return d_km / 111.0


# 多进程聚类 worker 用到的全局变量（由 initializer 设置）
_cluster_coords = None
_cluster_threshold = None


def _cluster_init(coords_threshold):
    """Pool initializer：传入 (coords, threshold) 供 _cluster_worker 使用。"""
    global _cluster_coords, _cluster_threshold
    _cluster_coords, _cluster_threshold = coords_threshold


def _cluster_worker(args):
    """Worker：处理一批 (inds, neighbor_lists)，返回距离 < threshold 的 (i,j)，i<j。"""
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


def cluster_stations(stations_df, threshold_deg, workers=1):
    """按经纬度距离聚类，返回 cluster_id 数组。workers>1 时并行计算距离对，主进程做 union。"""
    n = len(stations_df)
    parent = list(range(n))
    coords = np.asarray(stations_df[["lat", "lon"]].values, dtype=np.float64)

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
        # 每个格一个 (inds, neighbor_lists)；若某格 inds 很大则按 inds 分块，以增加 task 数、用满 workers
        cell_tasks = []
        max_inds_per_task = 2000
        for (ci, cj), inds in grid.items():
            neighbor_lists = [grid.get((ci + di, cj + dj), []) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
            if len(inds) <= max_inds_per_task:
                cell_tasks.append((inds, neighbor_lists))
            else:
                for start in range(0, len(inds), max_inds_per_task):
                    chunk = inds[start:start + max_inds_per_task]
                    cell_tasks.append((chunk, neighbor_lists))
        n_chunks = min(workers, max(1, len(cell_tasks)))
        chunk_size = (len(cell_tasks) + n_chunks - 1) // n_chunks
        tasks = [cell_tasks[i:i + chunk_size] for i in range(0, len(cell_tasks), chunk_size)]
        with Pool(
            len(tasks),
            initializer=_cluster_init,
            initargs=((coords, threshold_deg),),
        ) as pool:
            results = pool.map(_cluster_worker, tasks)
        for pairs in results:
            for i, j in pairs:
                union(i, j)

    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    cluster_ids = np.zeros(n, dtype=int)
    for cid, inds in enumerate(comp.values()):
        for i in inds:
            cluster_ids[i] = cid
    return cluster_ids


def load_nc_series(path):
    """
    从 nc 读取时间序列，返回 DataFrame：date, Q, SSC, SSL。
    变量名做统一映射；缺失为 np.nan。
    """
    if not HAS_NC:
        return None
    try:
        with nc4.Dataset(path, "r") as nc:
            time_var = next((x for x in TIME_NAMES if x in nc.variables), None)
            if time_var is None:
                return None
            t = nc.variables[time_var]
            t_vals = np.asarray(t[:]).flatten()
            units = getattr(t, "units", "days since 1970-01-01")
            calendar = getattr(t, "calendar", "gregorian")
            try:
                times = num2date(t_vals, units, calendar=calendar)
            except TypeError:
                try:
                    times = num2date(t_vals, units, calendar=calendar, only_use_cftime_datetimes=False)
                except Exception:
                    times = pd.to_datetime(t_vals, unit="D", origin="1970-01-01")
            except Exception:
                times = pd.to_datetime(t_vals, unit="D", origin="1970-01-01")
            # num2date 可能返回 cftime 对象，pd.to_datetime 无法直接转换，先转成 date 列表
            try:
                times = pd.to_datetime(times)
                if hasattr(times, "date"):
                    dates = [pd.Timestamp(tt).date() for tt in times]
                else:
                    dates = [pd.Timestamp(tt).date() for tt in times.tolist()]
            except (TypeError, ValueError):
                # cftime.DatetimeGregorian 等：用 isoformat 或 (year,month,day)
                dates = []
                for tt in times:
                    if hasattr(tt, "isoformat"):
                        dates.append(pd.Timestamp(tt.isoformat()).date())
                    elif hasattr(tt, "year") and hasattr(tt, "month") and hasattr(tt, "day"):
                        dates.append(pd.Timestamp(tt.year, tt.month, tt.day).date())
                    else:
                        dates.append(pd.Timestamp(str(tt)).date())

            q = _get_var(nc, Q_NAMES)
            ssc = _get_var(nc, SSC_NAMES)
            ssl = _get_var(nc, SSL_NAMES)
            n = len(dates)
            if n == 0:
                return None
            def pad(a, size, fill=np.nan):
                a = np.asarray(a).flatten()
                if len(a) >= size:
                    return a[:size]
                return np.concatenate([a, np.full(size - len(a), fill)])
            q = pad(q, n)
            ssc = pad(ssc, n)
            ssl = pad(ssl, n)
            df = pd.DataFrame({
                "date": dates,
                "Q": q,
                "SSC": ssc,
                "SSL": ssl,
            })
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for col in ["Q", "SSC", "SSL"]:
                df.loc[df[col] == FILL, col] = np.nan
                df.loc[df[col] == -9999, col] = np.nan
            return df
    except Exception:
        return None


def merge_series_keep_overlaps(records):
    """
    records: list of (source, DataFrame with date, Q, SSC, SSL)
    合并为一张表：所有日期的并集；同一日期多源则多行，并标记 is_overlap。
    列：date, Q, SSC, SSL, source, is_overlap
    """
    all_dates = set()
    series_list = []
    for source, df in records:
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["_source"] = source
        series_list.append(df)
        all_dates.update(df["date"].tolist())
    if not series_list:
        return None
    all_dates = sorted(all_dates)

    rows = []
    for d in all_dates:
        candidates = []
        for s in series_list:
            r = s[s["date"] == d]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            q = r.get("Q", np.nan)
            ssc = r.get("SSC", np.nan)
            ssl = r.get("SSL", np.nan)
            if pd.isna(q) and pd.isna(ssc) and pd.isna(ssl):
                continue
            candidates.append((q, ssc, ssl, s["_source"].iloc[0]))
        if not candidates:
            rows.append({"date": d, "Q": np.nan, "SSC": np.nan, "SSL": np.nan, "source": "", "is_overlap": False})
            continue
        is_overlap = len(candidates) > 1
        for q, ssc, ssl, src in candidates:
            rows.append({
                "date": d, "Q": q, "SSC": ssc, "SSL": ssl,
                "source": src, "is_overlap": is_overlap,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["date", "source"]).reset_index(drop=True)


def merge_series_single_per_date(records):
    """
    合并多源时间序列：每个日期只保留一个值（优先取第一个非缺失源）。
    返回 DataFrame：date, Q, SSC, SSL，用于写入合并后的 NetCDF。
    """
    all_dates = set()
    series_list = []
    for source, df in records:
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["_source"] = source
        series_list.append(df)
        all_dates.update(df["date"].tolist())
    if not series_list:
        return None
    all_dates = sorted(all_dates)
    rows = []
    for d in all_dates:
        q, ssc, ssl = np.nan, np.nan, np.nan
        for s in series_list:
            r = s[s["date"] == d]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            q = r.get("Q", np.nan) if pd.isna(q) else q
            ssc = r.get("SSC", np.nan) if pd.isna(ssc) else ssc
            ssl = r.get("SSL", np.nan) if pd.isna(ssl) else ssl
            if not (pd.isna(q) and pd.isna(ssc) and pd.isna(ssl)):
                break
        rows.append({"date": d, "Q": q, "SSC": ssc, "SSL": ssl})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def write_merged_nc(path, df, lat, lon, fill_value=FILL):
    """将合并后的时间序列写入单个 NetCDF 文件。"""
    if not HAS_NC or df is None or len(df) == 0:
        return
    try:
        dates = pd.to_datetime(df["date"])
        ref = pd.Timestamp("1970-01-01")
        time_num = (dates - ref).astype("timedelta64[D]").astype(np.float64)
        with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as nc:
            nc.createDimension("time", len(time_num))
            t = nc.createVariable("time", "f8", ("time",))
            t.units = "days since 1970-01-01"
            t.calendar = "gregorian"
            t[:] = time_num
            lat_v = nc.createVariable("lat", "f8", ())
            lat_v[:] = float(lat)
            lon_v = nc.createVariable("lon", "f8", ())
            lon_v[:] = float(lon)
            for name, col in [("Q", "Q"), ("SSC", "SSC"), ("SSL", "SSL")]:
                v = nc.createVariable(name, "f4", ("time",), fill_value=fill_value)
                arr = np.asarray(df[col], dtype=np.float32)
                arr = np.where(np.isnan(arr), fill_value, arr)
                v[:] = arr
        return True
    except Exception:
        return False


def _merge_one_cluster(args):
    """
    多进程 worker：处理一个 cluster 的合并并写入 CSV，可选写入合并 NC。
    参数需可 pickle，out_dir/out_nc_dir 传字符串。
    返回 (cid, n_rows, n_overlap) 或 (cid, 0, 0) 表示跳过/失败。
    """
    cid, recs, out_dir_str, rep_lat, rep_lon, out_nc_dir_str = args
    records_with_df = []
    for source, path in recs:
        df = load_nc_series(path)
        if df is not None and len(df) > 0:
            records_with_df.append((source, df))
    if not records_with_df:
        return (cid, 0, 0)
    merged = merge_series_keep_overlaps(records_with_df)
    if merged is None or len(merged) == 0:
        return (cid, 0, 0)
    base_name = "merged_qc_cluster_{:04d}_lat{:.4f}_lon{:.4f}".format(cid, rep_lat, rep_lon)
    out_path = Path(out_dir_str) / (base_name + ".csv")
    merged.to_csv(out_path, index=False)
    if out_nc_dir_str:
        single_df = merge_series_single_per_date(records_with_df)
        if single_df is not None and len(single_df) > 0:
            nc_path = Path(out_nc_dir_str) / (base_name + ".nc")
            write_merged_nc(nc_path, single_df, rep_lat, rep_lon)
    n_overlap = int(merged["is_overlap"].sum())
    return (cid, len(merged), n_overlap)


def _log_run_time(out_dir, elapsed_s, threshold, workers, n_stations=None, n_clusters=None, written=None, mode="full"):
    """打印运行时间并追加到 out_dir/step3_run_time.txt"""
    print("Run time: {:.2f} s ({:.2f} min)".format(elapsed_s, elapsed_s / 60.0))
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "step3_run_time.txt"
        parts = [time.strftime("%Y-%m-%d %H:%M:%S"), "threshold={}".format(threshold), "workers={}".format(workers), "elapsed_s={:.2f}".format(elapsed_s)]
        if n_stations is not None:
            parts.append("n_stations={}".format(n_stations))
        if n_clusters is not None:
            parts.append("n_clusters={}".format(n_clusters))
        if written is not None:
            parts.append("written={}".format(written))
        parts.append("mode={}".format(mode))
        with open(log_path, "a") as f:
            f.write("\t".join(parts) + "\n")
        print("Run time logged to {}.".format(log_path))
    except Exception:
        pass


def main():
    c = BUILDIN_CONFIG
    ap = argparse.ArgumentParser(description="Merge qc nc by location; keep overlapping times for manual choice")

    # 路径与输出
    ap.add_argument("--root", default=c["root"], help="Output_r 根目录")
    ap.add_argument("--out-dir", default=c["out_dir"], help="合并 CSV 输出目录（相对 root）")
    ap.add_argument("--out-nc-dir", default=c["out_nc_dir"], help="合并 NC 输出目录，空则用 out_dir/merged_nc")

    # 是否输出 .nc：默认由 BUILDIN_CONFIG['output_nc'] 决定（True=输出）；加 --no-output-nc 则不输出
    ap.add_argument("--no-output-nc", action="store_true", help="不输出合并后的 .nc 文件")

    # 聚类与运行
    ap.add_argument("--threshold", type=float, default=c["threshold"], help="经纬度聚类距离阈值（度）")
    ap.add_argument("--dry-run", action="store_true", default=c["dry_run"], help="仅报告聚类结果，不写文件")
    ap.add_argument("--workers", "-j", type=int, default=c["workers"], help="并行进程数，0=自动")
    ap.add_argument("--only-collect", action="store_true", default=c["only_collect"], help="仅收集站点并保存 CSV 后退出")

    # 输入：可跳过“收集/聚类”，直接读已有 CSV
    ap.add_argument("--stations-cache", default=c["stations_cache"], metavar="FILE", help="从已有 CSV 加载站点，跳过收集")
    ap.add_argument("--clustered-stations", default=c["clustered_stations"], metavar="FILE", help="从 step2 的 clustered_stations.csv 加载，跳过收集与聚类")

    args = ap.parse_args()

    # 是否输出 NC：未传 --no-output-nc 时用配置里的 output_nc，传了则强制不输出
    args.output_nc = False if args.no_output_nc else c["output_nc"]

    if not HAS_NC:
        print("Error: netCDF4 is required. Install with: pip install netCDF4")
        return

    root_dir = Path(args.root).resolve()
    out_dir = root_dir / args.out_dir
    threshold = args.threshold
    workers = args.workers if args.workers > 0 else min(32, max(1, (cpu_count() or 2) - 1))

    print("Root:   {}".format(root_dir))
    print("Output: {}".format(out_dir))

    t0 = time.perf_counter()

    if args.clustered_stations and Path(args.clustered_stations).is_file():
        print("Loading clustered stations from: {} ...".format(args.clustered_stations))
        stations = pd.read_csv(args.clustered_stations)
        for col in ["path", "source", "lat", "lon", "cluster_id"]:
            if col not in stations.columns:
                print("Error: clustered CSV must have columns: path, source, lat, lon, cluster_id.")
                return
        n_clusters = int(stations["cluster_id"].nunique())
        print("Loaded {} stations, {} clusters (skip step 1 & 2).".format(len(stations), n_clusters))
    elif args.stations_cache and Path(args.stations_cache).is_file():
        print("Loading stations from cache: {} ...".format(args.stations_cache))
        stations = pd.read_csv(args.stations_cache)
        for col in ["path", "source", "lat", "lon"]:
            if col not in stations.columns:
                print("Error: cached CSV must have columns: path, source, lat, lon.")
                return
        print("Loaded {} stations.".format(len(stations)))
        if args.only_collect:
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path = out_dir / "collected_stations.csv"
            stations.to_csv(cache_path, index=False)
            print("Saved to {}. Run later with: --stations-cache {}".format(cache_path, cache_path))
            _log_run_time(out_dir, time.perf_counter() - t0, threshold, workers, n_stations=len(stations), mode="only_collect")
            return
        print("Clustering by distance (threshold_deg = {}, workers={}) ...".format(threshold, workers))
        stations["cluster_id"] = cluster_stations(stations, threshold, workers=workers)
        n_clusters = stations["cluster_id"].nunique()
        print("Clusters: {}.".format(n_clusters))
    else:
        print("Collecting qc .nc stations (workers={}) ...".format(workers))
        stations = collect_qc_nc_stations(root_dir, workers=workers)
        if len(stations) == 0:
            print("No qc .nc files found with valid lat/lon.")
            _log_run_time(out_dir, time.perf_counter() - t0, threshold, workers, mode="no_files")
            return
        print("Found {} qc .nc files.".format(len(stations)))
        if args.only_collect:
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path = out_dir / "collected_stations.csv"
            stations.to_csv(cache_path, index=False)
            print("Saved to {}. Run later with: --stations-cache {}".format(cache_path, cache_path))
            _log_run_time(out_dir, time.perf_counter() - t0, threshold, workers, n_stations=len(stations), mode="only_collect")
            return
        print("Clustering by distance (threshold_deg = {}, workers={}) ...".format(threshold, workers))
        stations["cluster_id"] = cluster_stations(stations, threshold, workers=workers)
        n_clusters = stations["cluster_id"].nunique()
        print("Clusters: {}.".format(n_clusters))

    by_cluster = defaultdict(list)
    for _, row in stations.iterrows():
        by_cluster[row["cluster_id"]].append((row["source"], row["path"]))

    report_rows = []
    for cid in range(n_clusters):
        members = stations[stations["cluster_id"] == cid]
        rep = members.iloc[0]
        report_rows.append({
            "cluster_id": cid,
            "lat": rep["lat"],
            "lon": rep["lon"],
            "n_stations": len(members),
            "sources": ",".join(sorted(members["source"].unique())),
            "paths": "; ".join(members["path"].tolist()[:3]) + (" ..." if len(members) > 3 else ""),
        })
    report_df = pd.DataFrame(report_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "merge_qc_nc_report.csv"
    report_df.to_csv(report_path, index=False)
    print("Wrote {}.".format(report_path))

    if args.dry_run:
        print("Dry run: skipping merged time series.")
        _log_run_time(out_dir, time.perf_counter() - t0, threshold, workers, n_stations=len(stations), n_clusters=n_clusters, mode="dry_run")
        return

    out_nc_dir_str = None
    if args.output_nc:
        out_nc_dir = Path(args.out_nc_dir).resolve() if args.out_nc_dir else (out_dir / "merged_nc").resolve()
        out_nc_dir.mkdir(parents=True, exist_ok=True)
        out_nc_dir_str = str(out_nc_dir)
        print("Merged NetCDF will be written to: {} (absolute path)".format(out_nc_dir))

    print("Merging time series (overlaps kept for manual choice, workers={}) ...".format(workers))
    tasks = []
    for cid in range(n_clusters):
        recs = by_cluster.get(cid, [])
        if not recs:
            continue
        members = stations[stations["cluster_id"] == cid]
        rep = members.iloc[0]
        tasks.append((cid, recs, str(out_dir), float(rep["lat"]), float(rep["lon"]), out_nc_dir_str))

    n_tasks = len(tasks)
    use_pool = workers > 1 and n_tasks > 1
    if use_pool:
        n_workers = min(workers, n_tasks)
        print("  {} clusters to merge, using Pool with {} processes.".format(n_tasks, n_workers))
    else:
        reason = "workers=1" if workers <= 1 else "only 1 cluster to merge"
        print("  {} cluster(s) to merge, single process ({}).".format(n_tasks, reason))

    written = 0
    if not use_pool:
        for t in tqdm(tasks, desc="合并时间序列", unit="cluster"):
            cid, n_rows, n_overlap = _merge_one_cluster(t)
            if n_rows > 0:
                written += 1
                if n_overlap > 0:
                    fname = "merged_qc_cluster_{:04d}_lat{:.4f}_lon{:.4f}.csv".format(cid, t[3], t[4])
                    _progress_write("  {}: {} rows, {} overlap rows (manual choice)".format(fname, n_rows, n_overlap))
    else:
        chunksize = max(1, n_tasks // (n_workers * 4))  # 减少 IPC 次数，提高 CPU 占用
        cid_to_task = {t[0]: t for t in tasks}  # 用于 imap_unordered 后按 cid 取 lat,lon
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_merge_one_cluster, tasks, chunksize=chunksize),
                total=len(tasks), desc="合并时间序列", unit="cluster",
            ))
        for (cid, n_rows, n_overlap) in results:
            t = cid_to_task[cid]
            if n_rows > 0:
                written += 1
                if n_overlap > 0:
                    fname = "merged_qc_cluster_{:04d}_lat{:.4f}_lon{:.4f}.csv".format(cid, t[3], t[4])
                    print("  {}: {} rows, {} overlap rows (manual choice)".format(fname, n_rows, n_overlap))
    print("Wrote {} merged CSV(s) to {}.".format(written, out_dir))
    if out_nc_dir_str:
        print("Wrote merged NetCDF(s) to {}.".format(out_nc_dir_str))
    print("\nFor dates with is_overlap=True, keep one row per date (the source you trust) and delete the others.")
    _log_run_time(out_dir, time.perf_counter() - t0, threshold, workers, n_stations=len(stations), n_clusters=n_clusters, written=written, mode="full")


if __name__ == "__main__":
    main()
