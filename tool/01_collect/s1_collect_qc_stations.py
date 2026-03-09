#!/usr/bin/env python3
"""
第一步：从 Output_r 下所有 qc 文件夹中扫描 .nc 文件，读取经纬度与数据源，输出 CSV。

输出：collected_stations.csv，列 path, source, lat, lon, resolution。
resolution 来自路径第一级：daily, monthly, annually_climatology（同一站点可有多行不同分辨率）。
供 s2 聚类时保留分辨率信息。

用法：
  python collect_qc_stations.py [--root .] [--out output/01_collect/collected_stations.csv] [-j 32]
"""

import re
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc4
    HAS_NC = True
except ImportError:
    HAS_NC = False

# 数据根目录：Script/tool 与 Output_r 同位于 sediment_wzx_1111 下
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"

LAT_NAMES = ["lat", "latitude", "Latitude"]
LON_NAMES = ["lon", "longitude", "Longitude"]
FILL = -9999.0


def _get_scalar(var):
    if var is None:
        return None
    # 先按 masked 处理，再 asarray，否则 np.asarray(masked) 会变成 0
    if np.ma.isMaskedArray(var):
        v = var.flatten()
        if v.size == 0:
            return None
        v = v.flat[0]
        if np.ma.is_masked(v):
            return np.nan
        v = float(v)
    else:
        arr = np.asarray(var).flatten()
        if arr.size == 0:
            return None
        v = float(arr.flat[0])
    if np.isnan(v) or v == FILL or v == -9999:
        return np.nan
    return v


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


def get_resolution_from_path(path, root_dir):
    """从路径第一级目录解析时间分辨率：daily, monthly, annually_climatology。"""
    try:
        rel = Path(path).relative_to(Path(root_dir))
        parts = rel.parts
        if parts:
            res = parts[0].strip().lower()
            if res in ("daily", "monthly"):
                return res
            if "annually" in res or "climatology" in res:
                return "annually_climatology"
            return res
    except Exception:
        pass
    return "unknown"


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
    """Worker: 读单个 nc 的 path/source/lat/lon/resolution。用于多进程，须为模块级函数。"""
    try:
        lat, lon = get_lat_lon_from_nc(path)
        if lat is None or lon is None:
            return None
        source = get_source_from_path(path, root_dir)
        resolution = get_resolution_from_path(path, root_dir)
        return {"path": path, "source": source, "lat": lat, "lon": lon, "resolution": resolution}
    except (ValueError, OSError):
        return None


def collect_qc_nc_stations(root_dir, workers=1):
    """收集所有 qc 目录下的 .nc 文件，返回 DataFrame: path, source, lat, lon, resolution。"""
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
        rows = [_collect_one_nc(p, root_str) for p in paths]
    else:
        with Pool(min(workers, len(paths), cpu_count() or 1)) as pool:
            rows = pool.starmap(_collect_one_nc, [(p, root_str) for p in paths])
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Step 1: collect qc nc stations (path, source, lat, lon, resolution) to CSV")
    ap.add_argument("--root", default=str(OUTPUT_R_ROOT), help="Output_r 数据根目录")
    ap.add_argument("--out", default=str(OUTPUT_R_ROOT / "output/01_collect/collected_stations.csv"), help="Output CSV 路径")
    ap.add_argument("--workers", "-j", type=int, default=0,
                    help="Parallel workers; 0=auto (cpu_count-1, max 32)")
    args = ap.parse_args()

    if not HAS_NC:
        print("Error: netCDF4 is required. Install with: pip install netCDF4")
        return

    root_dir = Path(args.root).resolve()
    workers = args.workers if args.workers > 0 else min(32, max(1, (cpu_count() or 2) - 1))

    print("Collecting qc .nc stations (workers={}) ...".format(workers))
    stations = collect_qc_nc_stations(root_dir, workers=workers)
    if len(stations) == 0:
        print("No qc .nc files found with valid lat/lon.")
        return
    print("Found {} qc .nc files.".format(len(stations)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(out_path, index=False)
    print("Saved to {}.".format(out_path))
    print("Next: run s2_cluster_qc_stations.py with input {}".format(out_path))


if __name__ == "__main__":
    main()
