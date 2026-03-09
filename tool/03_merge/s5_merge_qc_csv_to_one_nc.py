#!/usr/bin/env python3
"""
第五步：将各 cluster 的时间序列按分辨率分别合并到单个 NetCDF 文件。

  - 同一站点（cluster）可同时保留 daily / monthly / annually_climatology 多条序列；
  - 重叠日期用 overlap_resolved.csv（由 s4 生成，键为 cluster_id, resolution, date）；
  - 输出一个大 NC：每条记录带 station_index, time, resolution（0=daily,1=monthly,2=annually_climatology）, Q, SSC, SSL, is_overlap。

输出 NC 结构：
  - 维度：n_stations, n_records
  - lat(n_stations), lon(n_stations)
  - station_index(n_records), time(n_records), resolution(n_records), Q, SSC, SSL, is_overlap
  - time 单位：days since 1970-01-01；resolution 0=daily, 1=monthly, 2=annually_climatology
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

try:
    import netCDF4 as nc4
    from netCDF4 import num2date
    HAS_NC = True
except ImportError:
    HAS_NC = False

FILL = -9999.0
TIME_NAMES = ["time", "Time", "t", "sample"]
Q_NAMES = ["Q", "discharge", "Discharge_m3_s", "Discharge"]
SSC_NAMES = ["SSC", "ssc", "TSS_mg_L", "TSS"]
SSL_NAMES = ["SSL", "sediment_load", "Sediment_load"]

# 多核并行：进程数，None 表示自动取 CPU 核数，设为 1 则关闭并行
N_WORKERS = 32

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"


def _get_var(ds, names, default=np.nan):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds.variables[n][:]).flatten()
    return np.full(1, default)


def load_nc_series(path):
    """从 nc 读取时间序列，返回 DataFrame：date, Q, SSC, SSL。"""
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
            try:
                times = pd.to_datetime(times)
                if hasattr(times, "date"):
                    dates = [pd.Timestamp(tt).date() for tt in times]
                else:
                    dates = [pd.Timestamp(tt).date() for tt in times.tolist()]
            except (TypeError, ValueError):
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
            df = pd.DataFrame({"date": dates, "Q": q, "SSC": ssc, "SSL": ssl})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for col in ["Q", "SSC", "SSL"]:
                df.loc[df[col] == FILL, col] = np.nan
                df.loc[df[col] == -9999, col] = np.nan
            return df
    except Exception:
        return None


# resolution 编码：与 s1 输出的 resolution 一致
RESOLUTION_CODES = {"daily": 0, "monthly": 1, "annually_climatology": 2}


def _resolution_to_code(res):
    return RESOLUTION_CODES.get(str(res).strip().lower(), 0)


def build_cluster_series(cid, resolution, recs, overlap_lookup):
    """
    为单个 (cluster_id, resolution) 构建时间序列。
    overlap_lookup: (cluster_id, resolution, date) -> (Q, SSC, SSL)
    返回 (dates, q, ssc, ssl, is_overlap) 各为 list，或 None
    """
    series_list = []
    all_dates = set()
    for source, path in recs:
        df = load_nc_series(path)
        if df is not None and len(df) > 0:
            df = df.copy()
            df["_source"] = source
            series_list.append(df)
            all_dates.update(df["date"].tolist())
    if not series_list:
        return None
    all_dates = sorted(all_dates)

    dates_out = []
    q_out = []
    ssc_out = []
    ssl_out = []
    is_overlap_out = []

    for d in all_dates:
        key = (cid, resolution, d)
        if key in overlap_lookup:
            q, ssc, ssl = overlap_lookup[key]
            dates_out.append(d)
            q_out.append(q if pd.notna(q) else FILL)
            ssc_out.append(ssc if pd.notna(ssc) else FILL)
            ssl_out.append(ssl if pd.notna(ssl) else FILL)
            is_overlap_out.append(1)
        else:
            q, ssc, ssl = np.nan, np.nan, np.nan
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
                break
            dates_out.append(d)
            q_out.append(q if pd.notna(q) else FILL)
            ssc_out.append(ssc if pd.notna(ssc) else FILL)
            ssl_out.append(ssl if pd.notna(ssl) else FILL)
            is_overlap_out.append(0)

    if not dates_out:
        return None
    return (dates_out, q_out, ssc_out, ssl_out, is_overlap_out)


# 多进程 worker：子进程通过 initializer 注入 overlap_lookup，避免每个任务重复 pickle
_OVERLAP_LOOKUP = None


def _init_worker(overlap_lookup):
    global _OVERLAP_LOOKUP
    _OVERLAP_LOOKUP = overlap_lookup


# 多进程 worker：任务为 (cid, resolution, recs)，返回 (cid, resolution, result)
def _worker_build_cluster(args):
    cid, resolution, recs = args
    res = build_cluster_series(cid, resolution, recs, _OVERLAP_LOOKUP)
    return (cid, resolution, res)


def main():
    ap = argparse.ArgumentParser(
        description="Step 5: merge all clusters into one NC from clustered_stations + NCs + overlap_resolved (from s4)"
    )
    ap.add_argument("--input-dir", "-i", default=str(OUTPUT_R_ROOT / "output/03_merge"), help="Directory with report, overlap_resolved (and optional clustered_stations if not using --clustered)")
    ap.add_argument("--clustered", "-c", default=str(OUTPUT_R_ROOT / "output/02_cluster/clustered_stations.csv"), help="clustered_stations.csv path (from step 2)")
    ap.add_argument("--output", "-o", default=str(OUTPUT_R_ROOT / "output/03_merge/merged_all.nc"), help="Output NetCDF path")
    args = ap.parse_args()

    if not HAS_NC:
        print("Error: netCDF4 is required. pip install netCDF4")
        return

    indir = Path(args.input_dir)
    report_path = indir / "merge_qc_nc_report.csv"
    clustered_path = Path(args.clustered)
    overlap_path = indir / "overlap_resolved.csv"

    if not report_path.is_file():
        print("Error: not found: {}".format(report_path))
        return
    if not clustered_path.is_file():
        print("Error: not found: {}".format(clustered_path))
        return

    report = pd.read_csv(report_path)
    stations = pd.read_csv(clustered_path)
    n_stations = len(report)
    lats = np.asarray(report["lat"], dtype=np.float64)
    lons = np.asarray(report["lon"], dtype=np.float64)

    for col in ["path", "source", "cluster_id"]:
        if col not in stations.columns:
            print("Error: clustered_stations.csv must have columns: path, source, cluster_id")
            return

    for col in ["path", "source", "cluster_id"]:
        if col not in stations.columns:
            print("Error: clustered_stations.csv must have columns: path, source, cluster_id")
            return
    if "resolution" not in stations.columns:
        stations["resolution"] = "unknown"

    # 重叠日期：(cluster_id, resolution, date) -> (Q, SSC, SSL)；若 CSV 无 resolution 列则用 (cluster_id, date) 兼容旧版
    overlap_lookup = {}
    if overlap_path.is_file():
        ov = pd.read_csv(overlap_path)
        if len(ov) > 0 and "cluster_id" in ov.columns and "date" in ov.columns:
            ov["date"] = pd.to_datetime(ov["date"], errors="coerce").dt.date
            ov = ov.dropna(subset=["date"])
            has_res = "resolution" in ov.columns
            for _, row in ov.iterrows():
                cid = int(row["cluster_id"])
                d = row["date"]
                q = row.get("Q", np.nan)
                ssc = row.get("SSC", np.nan)
                ssl = row.get("SSL", np.nan)
                if pd.isna(q):
                    q = np.nan
                if pd.isna(ssc):
                    ssc = np.nan
                if pd.isna(ssl):
                    ssl = np.nan
                if has_res:
                    res = str(row["resolution"]).strip()
                    overlap_lookup[(cid, res, d)] = (q, ssc, ssl)
                else:
                    # 旧版无 resolution 列：同一 (cid, date) 对三种分辨率均生效
                    for res in ("daily", "monthly", "annually_climatology"):
                        overlap_lookup[(cid, res, d)] = (q, ssc, ssl)
            print("Loaded {} overlap-resolved keys from {}.".format(len(overlap_lookup), overlap_path.name))
    else:
        print("No overlap_resolved.csv; using first available source for all dates.")

    # 按 (cluster_id, resolution) 分组
    by_cluster_resolution = defaultdict(list)
    for _, row in stations.iterrows():
        cid = row["cluster_id"]
        res = str(row.get("resolution", "unknown")).strip()
        by_cluster_resolution[(cid, res)].append((row["source"], row["path"]))

    ref = pd.Timestamp("1970-01-01")
    tasks = []
    for (cid, res), recs in by_cluster_resolution.items():
        if recs:
            tasks.append((int(cid), res, recs))
    n_tasks = len(tasks)
    n_workers = N_WORKERS if N_WORKERS is not None else (os.cpu_count() or 4)
    n_workers = max(1, min(n_workers, n_tasks))
    use_parallel = n_workers > 1 and n_tasks >= 2

    if use_parallel:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(overlap_lookup,)) as ex:
            raw_results = ex.map(_worker_build_cluster, tasks)
        results = [(c, r, x) for c, r, x in raw_results if x is not None]
    else:
        results = []
        for (cid, res, recs) in tasks:
            x = build_cluster_series(cid, res, recs, overlap_lookup)
            if x is not None:
                results.append((cid, res, x))

    station_index = []
    time_vals = []
    resolution_vals = []
    q_vals = []
    ssc_vals = []
    ssl_vals = []
    is_overlap_vals = []
    for cid, resolution, res in results:
        dates_out, q_out, ssc_out, ssl_out, is_overlap_out = res
        n = len(dates_out)
        delta = pd.to_datetime(dates_out) - ref
        days = delta.total_seconds().values / 86400.0
        days = np.where(np.isnan(days), FILL, days).astype(np.float64)
        rc = _resolution_to_code(resolution)
        station_index.extend([cid] * n)
        time_vals.extend(days.tolist())
        resolution_vals.extend([rc] * n)
        q_vals.extend(q_out)
        ssc_vals.extend(ssc_out)
        ssl_vals.extend(ssl_out)
        is_overlap_vals.extend(is_overlap_out)

    if use_parallel:
        print("Parallel: used {} workers for {} (cluster, resolution) tasks.".format(n_workers, n_tasks))

    n_records = len(time_vals)
    if n_records == 0:
        print("Error: no records collected.")
        return

    station_index = np.array(station_index, dtype=np.int32)
    time_vals = np.array(time_vals, dtype=np.float64)
    resolution_vals = np.array(resolution_vals, dtype=np.int8)
    q_vals = np.array(q_vals, dtype=np.float32)
    ssc_vals = np.array(ssc_vals, dtype=np.float32)
    ssl_vals = np.array(ssl_vals, dtype=np.float32)
    is_overlap_vals = np.array(is_overlap_vals, dtype=np.int8)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing {} (n_stations={}, n_records={}) ...".format(out_path, n_stations, n_records))

    with nc4.Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("n_stations", n_stations)
        nc.createDimension("n_records", n_records)

        lat_v = nc.createVariable("lat", "f8", ("n_stations",))
        lat_v.long_name = "latitude of virtual station"
        lat_v.units = "degrees_north"
        lat_v[:] = lats

        lon_v = nc.createVariable("lon", "f8", ("n_stations",))
        lon_v.long_name = "longitude of virtual station"
        lon_v.units = "degrees_east"
        lon_v[:] = lons

        sid_v = nc.createVariable("station_index", "i4", ("n_records",))
        sid_v.long_name = "index of virtual station (0-based)"
        sid_v[:] = station_index

        res_v = nc.createVariable("resolution", "i1", ("n_records",))
        res_v.long_name = "time resolution: 0=daily, 1=monthly, 2=annually_climatology"
        res_v[:] = resolution_vals

        t_v = nc.createVariable("time", "f8", ("n_records",))
        t_v.long_name = "time"
        t_v.units = "days since 1970-01-01"
        t_v.calendar = "gregorian"
        t_v[:] = time_vals

        q_v = nc.createVariable("Q", "f4", ("n_records",), fill_value=FILL)
        q_v.long_name = "river discharge"
        q_v.units = "m3 s-1"
        q_v[:] = q_vals

        ssc_v = nc.createVariable("SSC", "f4", ("n_records",), fill_value=FILL)
        ssc_v.long_name = "suspended sediment concentration"
        ssc_v.units = "mg L-1"
        ssc_v[:] = ssc_vals

        ssl_v = nc.createVariable("SSL", "f4", ("n_records",), fill_value=FILL)
        ssl_v.long_name = "suspended sediment load"
        ssl_v.units = "ton day-1"
        ssl_v[:] = ssl_vals

        io_v = nc.createVariable("is_overlap", "i1", ("n_records",))
        io_v.long_name = "1 if this date used manual choice from overlap_resolved"
        io_v[:] = is_overlap_vals

        nc.source = "Merged from NCs + clustered_stations + overlap_resolved (s4/s5)"
        nc.conventions = "CF-1.8"

    print("Wrote {}.".format(out_path))


if __name__ == "__main__":
    main()
