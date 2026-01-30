#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Dethier (2022) river sediment dataset:
- 读取已有 NetCDF 文件（含 Discharge, sediment_flux, SSC）
- 进行质量控制（QC），生成 CF-1.8 / ACDD-1.3 规范的 NetCDF
- 自动检查和统一变量名：Q, SSC, SSL
- 自动时间转换为 pandas.DatetimeIndex
- 自动检查单位（若 sediment_flux 为 kg/s，则转换为 ton/day）
- 生成站点汇总 CSV，包含时间范围、完整率、均值、中位数、范围等指标
- 全程写入日志文件 + 控制台

Author: Zhongwang Wei (改进版 by ChatGPT)
"""

import os
import glob
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl,
    apply_quality_flag_array,        
    apply_hydro_qc_with_provenance, 
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)

# =========================
# 全局 QC / FLAG 设置
# =========================

FLAG_GOOD = 0       # good_data
FLAG_EST = 1        # estimated_data（当前未使用，预留）
FLAG_SUSPECT = 2    # suspect_data
FLAG_BAD = 3        # bad_data
FLAG_MISS = 9       # missing_data

# 物理阈值（可以根据需要调整）
# Q_EXTREME_HIGH = 1_000_000.0   # m3/s，极端大值，标记为 suspect
# SSC_EXTREME_HIGH = 3000.0      # mg/L，极端高浓度
# SSL_EXTREME_HIGH = 1_000_000.0 # ton/day，可按需要调整

FILL_FLOAT = -9999.0
FILL_FLAG = np.int8(-127)


# =========================
# 日志系统
# =========================

LOGGER = logging.getLogger("process_dethier")


def setup_logging(output_dir: str, log_name: str = "process_dethier.log"):
    """
    初始化日志系统，同时输出到文件和控制台
    """
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.handlers.clear()

    log_path = os.path.join(output_dir, log_name)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)

    LOGGER.info("Logging initialized")
    LOGGER.info("Log file: %s", log_path)


# =========================
# 工具函数
# =========================

def detect_var(ds_raw: xr.Dataset, candidates):
    """
    在 Dataset 中按候选名列表查找第一个存在的变量名
    """
    for name in candidates:
        if name in ds_raw.variables:
            return name
    return None


def get_lat_lon(ds_raw: xr.Dataset):
    """
    尽可能稳健地从 Dataset 中获取 latitude / longitude
    - 优先从全局属性
    - 然后从变量 lat / latitude / lon / longitude
    - 获取失败则返回 NaN
    """
    # 先看属性
    lat = ds_raw.attrs.get("latitude", None)
    lon = ds_raw.attrs.get("longitude", None)

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except Exception:
            pass

    # 再看变量
    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]

    lat_val = np.nan
    lon_val = np.nan

    for name in lat_candidates:
        if name in ds_raw.variables:
            try:
                lat_val = float(ds_raw[name].values.squeeze())
                break
            except Exception:
                pass

    for name in lon_candidates:
        if name in ds_raw.variables:
            try:
                lon_val = float(ds_raw[name].values.squeeze())
                break
            except Exception:
                pass

    return lat_val, lon_val


def compute_summary_stats(df: pd.DataFrame, var: str, flag_col: str):
    """
    计算某变量的统计量：
    - 起止时间（good data）
    - 完整率（good / 全部）
    - 均值 / 中位数 / 最小值 / 最大值（仅 good）
    """
    if var not in df.columns or flag_col not in df.columns:
        return {
            f"{var}_start_date": "N/A",
            f"{var}_end_date": "N/A",
            f"{var}_percent_complete": 0.0,
            f"{var}_mean": np.nan,
            f"{var}_median": np.nan,
            f"{var}_min": np.nan,
            f"{var}_max": np.nan,
        }

    good_mask = df[flag_col] == FLAG_GOOD
    good_series = df.loc[good_mask, var]

    if good_series.empty:
        return {
            f"{var}_start_date": "N/A",
            f"{var}_end_date": "N/A",
            f"{var}_percent_complete": 0.0,
            f"{var}_mean": np.nan,
            f"{var}_median": np.nan,
            f"{var}_min": np.nan,
            f"{var}_max": np.nan,
        }

    total_len = len(df)
    percent_complete = len(good_series) / total_len * 100.0 if total_len > 0 else 0.0

    return {
        f"{var}_start_date": good_series.index.min().strftime("%Y-%m-%d"),
        f"{var}_end_date": good_series.index.max().strftime("%Y-%m-%d"),
        f"{var}_percent_complete": percent_complete,
        f"{var}_mean": good_series.mean(),
        f"{var}_median": good_series.median(),
        f"{var}_min": good_series.min(),
        f"{var}_max": good_series.max(),
    }
def print_station_qc_summary(
    station_id,
    station_name,
    n_samples,
    skipped_log_iqr,
    skipped_ssc_q,
    q_value, q_flag,
    ssc_value, ssc_flag,
    ssl_value, ssl_flag,
    created_nc_path
):
  
    print(f"\nProcessing: {station_name} ({station_id})")
    print(f"  Sample size = {n_samples}")

    if skipped_log_iqr:
        print("  ⚠ log-IQR statistical QC skipped (sample size < 5).")
    if skipped_ssc_q:
        print("  ⚠ SSC–Q consistency check skipped (sample size < 5).")

    print(f"  ✓ Created: {os.path.basename(created_nc_path)}")
    print(f"    Q   : {q_value:.2f} m3/s (flag={q_flag})")
    print(f"    SSC : {ssc_value:.2f} mg/L (flag={ssc_flag})")
    print(f"    SSL : {ssl_value:.2f} ton/day (flag={ssl_flag})")

# =========================
# 核心处理函数
# =========================

def process_single_netcdf(nc_path: str, output_dir: str) -> dict | None:
    """
    处理单个 Dethier 原始 NetCDF 文件，输出标准化 NetCDF 和统计信息字典
    返回用于 summary CSV 的 dict，如果无有效数据则返回 None
    """
    basename = os.path.basename(nc_path)
    LOGGER.info("Processing file: %s", basename)

    try:
        ds_raw = xr.open_dataset(nc_path)
    except Exception as e:
        LOGGER.error("Failed to open %s: %s", basename, e)
        return None

    # ---- 站点元信息 ----
    station_id = ds_raw.attrs.get("site_no", os.path.splitext(basename)[0])
    river_name = ds_raw.attrs.get("river_name", "UnknownRiver")

    lat, lon = get_lat_lon(ds_raw)

    # ---- 自动检测变量名 ----
    q_name = detect_var(ds_raw, ["Discharge", "discharge", "Q", "q"])
    ssl_name = detect_var(ds_raw, ["sediment_flux", "SSL", "ssl", "sediment_load"])
    ssc_name = detect_var(ds_raw, ["SSC", "ssc", "TSS", "tss"])

    if q_name is None and ssl_name is None and ssc_name is None:
        LOGGER.warning("No Q/SSC/SSL variables found in %s, skip.", basename)
        return None

    # 构建新 Dataset
    ds = xr.Dataset()

    # 时间坐标
    if "time" not in ds_raw.coords and "time" not in ds_raw.dims:
        LOGGER.error("No 'time' coordinate in %s, skip.", basename)
        return None

    time_index = ds_raw.indexes.get("time", None)
    if time_index is None:
        # 利用 to_index
        try:
            time_index = ds_raw["time"].to_index()
        except Exception:
            LOGGER.error("Failed to parse 'time' in %s", basename)
            return None

    # 转为 pandas DatetimeIndex
    time_pd = pd.to_datetime(time_index, errors="coerce")
    valid_time_mask = ~time_pd.isna()
    if not valid_time_mask.any():
        LOGGER.error("All time values are invalid in %s, skip.", basename)
        return None

    # 只保留有效时间行
    ds_raw = ds_raw.isel(time=valid_time_mask)
    time_pd = time_pd[valid_time_mask]

    ds = ds.assign_coords(time=("time", time_pd))

    # ---- Q ----
    if q_name is not None:
        q_da = ds_raw[q_name].squeeze()
        q_val = q_da.values.astype(float)
        ds["Q"] = ("time", q_val)
        ds["Q"].attrs.update({
            "long_name": "River Discharge",
            "standard_name": "river_discharge",
            "units": getattr(q_da, "units", "m3 s-1"),
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "Q_flag",
        })
    else:
        q_val = np.full(time_pd.shape, np.nan)
        ds["Q"] = ("time", q_val)
        ds["Q"].attrs.update({
            "long_name": "River Discharge (not available)",
            "standard_name": "river_discharge",
            "units": "m3 s-1",
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "Q_flag",
        })

    # ---- SSC ----
    if ssc_name is not None:
        ssc_da = ds_raw[ssc_name].squeeze()
        ssc_val = ssc_da.values.astype(float)
        ds["SSC"] = ("time", ssc_val)
        ds["SSC"].attrs.update({
            "long_name": "Suspended Sediment Concentration",
            "standard_name": "mass_concentration_of_suspended_matter_in_water",
            "units": getattr(ssc_da, "units", "mg L-1"),
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "SSC_flag",
        })
    else:
        ssc_val = np.full(time_pd.shape, np.nan)
        ds["SSC"] = ("time", ssc_val)
        ds["SSC"].attrs.update({
            "long_name": "Suspended Sediment Concentration (not available)",
            "standard_name": "mass_concentration_of_suspended_matter_in_water",
            "units": "mg L-1",
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "SSC_flag",
        })

    # ---- SSL ----
    if ssl_name is not None:
        ssl_da_raw = ds_raw[ssl_name].squeeze()
        ssl_da = convert_ssl_units_if_needed(ssl_da_raw)
        ssl_val = ssl_da.values.astype(float)
        ds["SSL"] = ("time", ssl_val)
        ds["SSL"].attrs.update({
            "long_name": "Suspended Sediment Load",
            "standard_name": "sediment_transport_in_river",
            "units": ssl_da.attrs.get("units", "ton day-1"),
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "SSL_flag",
        })
    else:
        ssl_val = np.full(time_pd.shape, np.nan)
        ds["SSL"] = ("time", ssl_val)
        ds["SSL"].attrs.update({
            "long_name": "Suspended Sediment Load (not available)",
            "standard_name": "sediment_transport_in_river",
            "units": "ton day-1",
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": "SSL_flag",
        })

    # ---- 加入坐标 lat/lon ----
    ds = ds.assign_coords(latitude=("latitude", [lat]))
    ds = ds.assign_coords(longitude=("longitude", [lon]))

    # 转 DataFrame 做 QC
    df = ds[["Q", "SSC", "SSL"]].to_dataframe().reset_index().set_index("time")
    df.index = pd.to_datetime(df.index)  # 再保险
    df.sort_index(inplace=True)

    # ---- QC Flag ----
    q_arr = df["Q"].values.astype(float)
    ssc_arr = df["SSC"].values.astype(float)
    ssl_arr = df["SSL"].values.astype(float)

    # --------------------------------------------------
    # Build station-level SSC–Q envelope (for diagnostic plot)
    # --------------------------------------------------
    ssc_q_bounds = build_ssc_q_envelope(Q_m3s=q_arr, SSC_mgL=ssc_arr, k=1.5)
    # 1) QC1-array
    Q_flag_qc1   = apply_quality_flag_array(q_arr,  "Q")
    SSC_flag_qc1 = apply_quality_flag_array(ssc_arr,"SSC")
    SSL_flag_qc1 = apply_quality_flag_array(ssl_arr,"SSL")

    # 2) QC1 的 missing(9) 做一次 trim
    valid_time = (Q_flag_qc1 != 9) | (SSC_flag_qc1 != 9) | (SSL_flag_qc1 != 9)
    if not valid_time.any():
        LOGGER.warning("No valid data (QC1 all missing) for station %s, skip.", station_id)
        return None

    df_qc = df.loc[valid_time].copy()

    Q_qc   = df_qc["Q"].to_numpy(dtype=float)
    SSC_qc = df_qc["SSC"].to_numpy(dtype=float)
    SSL_qc = df_qc["SSL"].to_numpy(dtype=float)

    # time:days since 1970-01-01
    base = np.datetime64("1970-01-01T00:00:00")
    time_qc = ((df_qc.index.values.astype("datetime64[ns]") - base) / np.timedelta64(1, "D")).astype(float)

    # 3) End-to-end QC
    qc = apply_hydro_qc_with_provenance(
        time=time_qc,
        Q=Q_qc,
        SSC=SSC_qc,
        SSL=SSL_qc,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5, 
        qc3_min_samples=5,
    )

    if qc is None:
        LOGGER.warning("No valid time remains after hydro QC for station %s, skip.", station_id)
        return None
    # === align qc outputs back to df by time (robust) ===
    qc_time_dt = pd.to_datetime(qc["time"], unit="D", origin="1970-01-01")
    qc_time_dt = pd.DatetimeIndex(qc_time_dt)

    # 先全表填 missing
    df["Q_flag"]   = np.full(df.shape[0], FLAG_MISS, dtype=np.int8)
    df["SSC_flag"] = np.full(df.shape[0], FLAG_MISS, dtype=np.int8)
    df["SSL_flag"] = np.full(df.shape[0], FLAG_MISS, dtype=np.int8)

    # ✅ final flags 回填（按时间对齐）
    df.loc[qc_time_dt, "Q_flag"]   = qc["Q_flag"].astype(np.int8)
    df.loc[qc_time_dt, "SSC_flag"] = qc["SSC_flag"].astype(np.int8)
    df.loc[qc_time_dt, "SSL_flag"] = qc["SSL_flag"].astype(np.int8)

    # ✅ step/provenance flags：先创建列(默认9)，再回填
    step_cols = [
        "Q_flag_qc1_physical",
        "SSC_flag_qc1_physical",
        "SSL_flag_qc1_physical",
        "Q_flag_qc2_log_iqr",
        "SSC_flag_qc2_log_iqr",
        "SSL_flag_qc2_log_iqr",
        "SSC_flag_qc3_ssc_q",
        "SSL_flag_qc3_from_ssc_q",
    ]
    for c in step_cols:
        df[c] = np.full(df.shape[0], FLAG_MISS, dtype=np.int8)

    for c in step_cols:
        if c in qc:
            df.loc[qc_time_dt, c] = qc[c].astype(np.int8)

    df["Q_flag"] = np.full(len(df), FLAG_MISS, dtype=np.int8)
    df["SSC_flag"] = np.full(len(df), FLAG_MISS, dtype=np.int8)
    df["SSL_flag"] = np.full(len(df), FLAG_MISS, dtype=np.int8)

    df.loc[df_qc.index, "Q_flag"] = qc["Q_flag"].astype(np.int8)
    df.loc[df_qc.index, "SSC_flag"] = qc["SSC_flag"].astype(np.int8)
    df.loc[df_qc.index, "SSL_flag"] = qc["SSL_flag"].astype(np.int8)

    print(
        f"[QC] {station_id} QC1(good cnt): Q={int(np.sum(Q_flag_qc1==0))}, "
        f"SSC={int(np.sum(SSC_flag_qc1==0))}, SSL={int(np.sum(SSL_flag_qc1==0))} | "
        f"Final(good cnt): Q={int(np.sum(df['Q_flag'].to_numpy()==0))}, "
        f"SSC={int(np.sum(df['SSC_flag'].to_numpy()==0))}, SSL={int(np.sum(df['SSL_flag'].to_numpy()==0))}"
    )

    # ---- 时间切片：至少有一个变量非 NaN 且 flag 不是 missing/bad ----
    valid_mask = (
        ((df["Q_flag"] != FLAG_MISS) & (df["Q_flag"] != FLAG_BAD)) |
        ((df["SSC_flag"] != FLAG_MISS) & (df["SSC_flag"] != FLAG_BAD)) |
        ((df["SSL_flag"] != FLAG_MISS) & (df["SSL_flag"] != FLAG_BAD))
    )

    if not valid_mask.any():
        LOGGER.warning("No valid data (after QC) for station %s, skip.", station_id)
        return None

    df_valid = df.loc[valid_mask].copy()
    start_date = df_valid.index.min()
    end_date = df_valid.index.max()

    # 最终时间序列：从第一条有效到最后一条有效，中间允许 NaN 和非 good flag
    df_final = df.loc[start_date:end_date].copy()
    q_flags   = df_final["Q_flag"].to_numpy(np.int8)
    ssc_flags = df_final["SSC_flag"].to_numpy(np.int8)
    ssl_flags = df_final["SSL_flag"].to_numpy(np.int8)

    LOGGER.debug(
        "QC flags count: Q=%d, SSC=%d, SSL=%d",
            np.sum(q_flags == FLAG_GOOD),
            np.sum(ssc_flags == FLAG_GOOD),
            np.sum(ssl_flags == FLAG_GOOD),
    )

    # ---- 构建最终 Dataset ----
    ds_out = xr.Dataset()
    ds_out = ds_out.assign_coords(time=("time", df_final.index))
    ds_out = ds_out.assign_coords(latitude=("latitude", [lat]))
    ds_out = ds_out.assign_coords(longitude=("longitude", [lon]))
    # =========================
    # Step / provenance flags
    # =========================
    def _add_step_flag(name, values, flag_values, flag_meanings, long_name):
        ds_out[name] = ("time", np.asarray(values, dtype=np.int8))
        ds_out[name].attrs.update({
            "long_name": long_name,
            "_FillValue": FILL_FLAG,
            "flag_values": np.asarray(flag_values, dtype=np.int8),
            "flag_meanings": flag_meanings,
            "standard_name": "status_flag",
        })

    # QC1 physical: 0 pass, 3 bad, 9 missing
    qc1_vals = [0, 3, 9]
    qc1_mean = "pass bad missing"
    # QC2 log-IQR: 0 pass, 2 suspect, 8 not_checked, 9 missing
    qc2_vals = [0, 2, 8, 9]
    qc2_mean = "pass suspect not_checked missing"
    # QC3 SSC-Q: same as qc2
    qc3_vals = [0, 2, 8, 9]
    qc3_mean = "pass suspect not_checked missing"
    # QC3 SSL propagation: 0 not_propagated, 2 propagated, 8 not_checked, 9 missing
    ssl3_vals = [0, 2, 8, 9]
    ssl3_mean = "not_propagated propagated not_checked missing"
    # Q step flags
    if "Q_flag_qc1_physical" in df_final.columns:
        _add_step_flag("Q_flag_qc1_physical", df_final["Q_flag_qc1_physical"].values,
                    qc1_vals, qc1_mean, "QC1 physical flag for river discharge")
    if "Q_flag_qc2_log_iqr" in df_final.columns:
        _add_step_flag("Q_flag_qc2_log_iqr", df_final["Q_flag_qc2_log_iqr"].values,
                    qc2_vals, qc2_mean, "QC2 log-IQR flag for river discharge")

    # SSC step flags
    if "SSC_flag_qc1_physical" in df_final.columns:
        _add_step_flag("SSC_flag_qc1_physical", df_final["SSC_flag_qc1_physical"].values,
                    qc1_vals, qc1_mean, "QC1 physical flag for SSC")
    if "SSC_flag_qc2_log_iqr" in df_final.columns:
        _add_step_flag("SSC_flag_qc2_log_iqr", df_final["SSC_flag_qc2_log_iqr"].values,
                    qc2_vals, qc2_mean, "QC2 log-IQR flag for SSC")
    if "SSC_flag_qc3_ssc_q" in df_final.columns:
        _add_step_flag("SSC_flag_qc3_ssc_q", df_final["SSC_flag_qc3_ssc_q"].values,
                    qc3_vals, qc3_mean, "QC3 SSC-Q consistency flag for SSC")

    # SSL step flags
    if "SSL_flag_qc1_physical" in df_final.columns:
        _add_step_flag("SSL_flag_qc1_physical", df_final["SSL_flag_qc1_physical"].values,
                    qc1_vals, qc1_mean, "QC1 physical flag for SSL")
    if "SSL_flag_qc2_log_iqr" in df_final.columns:
        _add_step_flag("SSL_flag_qc2_log_iqr", df_final["SSL_flag_qc2_log_iqr"].values,
                    qc2_vals, qc2_mean, "QC2 log-IQR flag for SSL")
    if "SSL_flag_qc3_from_ssc_q" in df_final.columns:
        _add_step_flag("SSL_flag_qc3_from_ssc_q", df_final["SSL_flag_qc3_from_ssc_q"].values,
                    ssl3_vals, ssl3_mean, "QC3 SSL flag propagated from SSC-Q inconsistency")

    # Q / SSC / SSL
    for var_name, long_name, std_name, units, arr in [
        ("Q",   "River Discharge", "river_discharge", "m3 s-1", df_final["Q"].values),
        ("SSC", "Suspended Sediment Concentration",
         "mass_concentration_of_suspended_matter_in_water", "mg L-1", df_final["SSC"].values),
        ("SSL", "Suspended Sediment Load", "sediment_transport_in_river",
         "ton day-1", df_final["SSL"].values),
    ]:
        ds_out[var_name] = ("time", arr)
        ds_out[var_name].attrs.update({
            "long_name": long_name,
            "standard_name": std_name,
            "units": units,
            "_FillValue": FILL_FLOAT,
            "ancillary_variables": f"{var_name}_flag",
        })

    # Flag 变量（CF-1.8 样式）
    flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
    flag_values = np.array([FLAG_GOOD, FLAG_EST, FLAG_SUSPECT, FLAG_BAD, FLAG_MISS], dtype=np.int8)

    ds_out["Q_flag"] = ("time", df_final["Q_flag"].values.astype(np.int8))
    ds_out["Q_flag"].attrs.update({
        "long_name": "Quality flag for River Discharge",
        "_FillValue": FILL_FLAG,
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
        "standard_name": "status_flag",
        "comment": "0=good,1=estimated,2=suspect,3=bad,9=missing",
    })

    ds_out["SSC_flag"] = ("time", df_final["SSC_flag"].values.astype(np.int8))
    ds_out["SSC_flag"].attrs.update({
        "long_name": "Quality flag for Suspended Sediment Concentration",
        "_FillValue": FILL_FLAG,
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
        "standard_name": "status_flag",
        "comment": "0=good,1=estimated,2=suspect,3=bad,9=missing",
    })

    ds_out["SSL_flag"] = ("time", df_final["SSL_flag"].values.astype(np.int8))
    ds_out["SSL_flag"].attrs.update({
        "long_name": "Quality flag for Suspended Sediment Load",
        "_FillValue": FILL_FLAG,
        "flag_values": flag_values,
        "flag_meanings": flag_meanings,
        "standard_name": "status_flag",
        "comment": "0=good,1=estimated,2=suspect,3=bad,9=missing",
    })

    # ---- 全局属性 ----
    raw_attrs = ds_raw.attrs

    ds_out.attrs.update({
        "title": raw_attrs.get("title", "Harmonized Global River Discharge and Sediment"),
        "Data_Source_Name": raw_attrs.get("Data_Source_Name", "Dethier Dataset"),
        "station_name": raw_attrs.get("station_name", station_id),
        "river_name": raw_attrs.get("river_name", river_name),
        "Source_ID": raw_attrs.get("site_no", station_id),
        "Type": raw_attrs.get("Type", "Satellite station"),
        "Temporal_Resolution": raw_attrs.get("Temporal_Resolution", "monthly"),
        "Temporal_Span": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        "Geographic_Coverage": raw_attrs.get(
            "Geographic_Coverage",
            f"River reach near ({lat:.4f}N, {lon:.4f}E)"
        ),
        "Variables_Provided": "Q, SSC, SSL",
        "Reference1": "Dethier, E. N., et al. (2022), Science, DOI:10.1126/science.abn7980",
        "summary": "This dataset provides monthly river discharge and sediment data, processed and QC-filtered.",
        "creator_name": "Zhongwang Wei",
        "creator_email": "weizhw6@mail.sysu.edu.cn",
        "creator_institution": "Sun Yat-sen University, China",
        "Conventions": "CF-1.8, ACDD-1.3",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                   f"from {basename} using process_dethier_cf18.py",
    })


    # --------------------------------------------------
    # Station-level SSC–Q diagnostic plot
    # --------------------------------------------------
    diag_dir = os.path.join(output_dir, "diagnostic")
    os.makedirs(diag_dir, exist_ok=True)

    diag_png = os.path.join(
        diag_dir,
        f"SSC_Q_{station_id}.png"
    )
    plot_ssc_q_diagnostic(
        time=df_final.index,
        Q=df_final["Q"].values,
        SSC=df_final["SSC"].values,
        Q_flag=df_final["Q_flag"].values,
        SSC_flag=df_final["SSC_flag"].values,
        ssc_q_bounds=ssc_q_bounds,
        station_id=station_id,
        station_name=ds_out.attrs["station_name"],
        out_png=diag_png
    )

    # ---- 写出 NetCDF ----
    safe_river_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in river_name)
    out_name = f"Dethier_{safe_river_name}_{station_id}.nc"
    out_path = os.path.join(output_dir, out_name)

    encoding = {
        "time": {"units": "days since 1970-01-01 00:00:00"},
    }

    ds_out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    LOGGER.info("Saved standardized NetCDF: %s", out_path)
    # =========================
    # Print QC summary (console)
    # =========================
    print_station_qc_summary(
        station_id=station_id,
        station_name=ds_out.attrs["station_name"],
        n_samples=len(df),
        skipped_log_iqr=(len(df) < 5),
        skipped_ssc_q=(len(df) < 5),
        q_value=np.nanmean(df_final["Q"].values),
        q_flag=int(np.nanmax(df_final["Q_flag"].values)),
        ssc_value=np.nanmean(df_final["SSC"].values),
        ssc_flag=int(np.nanmax(df_final["SSC_flag"].values)),
        ssl_value=np.nanmean(df_final["SSL"].values),
        ssl_flag=int(np.nanmax(df_final["SSL_flag"].values)),
        created_nc_path=out_path
    )

    # ---- 汇总信息用于 CSV ----
    stats_q = compute_summary_stats(df_final, "Q", "Q_flag")
    stats_ssc = compute_summary_stats(df_final, "SSC", "SSC_flag")
    stats_ssl = compute_summary_stats(df_final, "SSL", "SSL_flag")

    summary = {
        "Source_ID": station_id,
        "station_name": ds_out.attrs["station_name"],
        "river_name": ds_out.attrs["river_name"],
        "longitude": float(lon) if not np.isnan(lon) else np.nan,
        "latitude": float(lat) if not np.isnan(lat) else np.nan,
        "Data Source Name": ds_out.attrs["Data_Source_Name"],
        "Type": ds_out.attrs["Type"],
        "Temporal Resolution": ds_out.attrs["Temporal_Resolution"],
        "Temporal Span": ds_out.attrs["Temporal_Span"],
        "Variables Provided": ds_out.attrs["Variables_Provided"],
        "Reference/DOI": "10.1126/science.abn7980",
    }
    summary.update(stats_q)
    summary.update(stats_ssc)
    summary.update(stats_ssl)

    def _count(arr, mapping):
        a = np.asarray(arr, dtype=np.int8)
        return {k: int(np.sum(a == np.int8(v))) for k, v in mapping.items()}

    # final flag counts
    final_map = {
        "good": 0, "estimated": 1, "suspect": 2, "bad": 3, "missing": 9
    }

    summary["QC_n_days"] = int(len(df_final))

    cQ   = _count(df_final["Q_flag"].values, final_map)
    cSSC = _count(df_final["SSC_flag"].values, final_map)
    cSSL = _count(df_final["SSL_flag"].values, final_map)

    summary.update({
        "Q_final_good": cQ["good"], "Q_final_estimated": cQ["estimated"], "Q_final_suspect": cQ["suspect"],
        "Q_final_bad": cQ["bad"], "Q_final_missing": cQ["missing"],

        "SSC_final_good": cSSC["good"], "SSC_final_estimated": cSSC["estimated"], "SSC_final_suspect": cSSC["suspect"],
        "SSC_final_bad": cSSC["bad"], "SSC_final_missing": cSSC["missing"],

        "SSL_final_good": cSSL["good"], "SSL_final_estimated": cSSL["estimated"], "SSL_final_suspect": cSSL["suspect"],
        "SSL_final_bad": cSSL["bad"], "SSL_final_missing": cSSL["missing"],
    })

    # QC1 physical
    qc1_map = {"pass": 0, "bad": 3, "missing": 9}
    # QC2 / QC3
    qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}
    # SSL propagation
    ssl3_map = {"not_propagated": 0, "propagated": 2, "not_checked": 8, "missing": 9}

    if "Q_flag_qc1_physical" in df_final:
        c = _count(df_final["Q_flag_qc1_physical"].values, qc1_map)
        summary.update({"Q_qc1_pass": c["pass"], "Q_qc1_bad": c["bad"], "Q_qc1_missing": c["missing"]})

    if "SSC_flag_qc1_physical" in df_final:
        c = _count(df_final["SSC_flag_qc1_physical"].values, qc1_map)
        summary.update({"SSC_qc1_pass": c["pass"], "SSC_qc1_bad": c["bad"], "SSC_qc1_missing": c["missing"]})

    if "SSL_flag_qc1_physical" in df_final:
        c = _count(df_final["SSL_flag_qc1_physical"].values, qc1_map)
        summary.update({"SSL_qc1_pass": c["pass"], "SSL_qc1_bad": c["bad"], "SSL_qc1_missing": c["missing"]})

    if "Q_flag_qc2_log_iqr" in df_final:
        c = _count(df_final["Q_flag_qc2_log_iqr"].values, qc2_map)
        summary.update({"Q_qc2_pass": c["pass"], "Q_qc2_suspect": c["suspect"], "Q_qc2_not_checked": c["not_checked"], "Q_qc2_missing": c["missing"]})

    if "SSC_flag_qc2_log_iqr" in df_final:
        c = _count(df_final["SSC_flag_qc2_log_iqr"].values, qc2_map)
        summary.update({"SSC_qc2_pass": c["pass"], "SSC_qc2_suspect": c["suspect"], "SSC_qc2_not_checked": c["not_checked"], "SSC_qc2_missing": c["missing"]})

    if "SSL_flag_qc2_log_iqr" in df_final:
        c = _count(df_final["SSL_flag_qc2_log_iqr"].values, qc2_map)
        summary.update({"SSL_qc2_pass": c["pass"], "SSL_qc2_suspect": c["suspect"], "SSL_qc2_not_checked": c["not_checked"], "SSL_qc2_missing": c["missing"]})

    if "SSC_flag_qc3_ssc_q" in df_final:
        c = _count(df_final["SSC_flag_qc3_ssc_q"].values, qc2_map)
        summary.update({"SSC_qc3_pass": c["pass"], "SSC_qc3_suspect": c["suspect"], "SSC_qc3_not_checked": c["not_checked"], "SSC_qc3_missing": c["missing"]})

    if "SSL_flag_qc3_from_ssc_q" in df_final:
        c = _count(df_final["SSL_flag_qc3_from_ssc_q"].values, ssl3_map)
        summary.update({"SSL_qc3_not_propagated": c["not_propagated"], "SSL_qc3_propagated": c["propagated"], "SSL_qc3_not_checked": c["not_checked"], "SSL_qc3_missing": c["missing"]})

    return summary


def process_dethier_data_from_nc(input_nc_dir: str, output_dir: str, summary_csv_path: str):
    """
    批量处理 Dethier 源数据文件夹中的 NetCDF
    """
    setup_logging(output_dir)

    LOGGER.info("Input directory: %s", input_nc_dir)
    LOGGER.info("Output directory: %s", output_dir)

    if not os.path.exists(input_nc_dir):
        LOGGER.error("Input directory does not exist: %s", input_nc_dir)
        return

    os.makedirs(output_dir, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(input_nc_dir, "*.nc")))
    LOGGER.info("Found %d NetCDF files", len(nc_files))

    if not nc_files:
        LOGGER.warning("No .nc files found in %s", input_nc_dir)
        return

    summaries = []
    success = 0
    skipped = 0

    for nc_file in nc_files:
        try:
            summary = process_single_netcdf(nc_file, output_dir)
            if summary is not None:
                summaries.append(summary)
                success += 1
            else:
                skipped += 1
        except Exception as e:
            LOGGER.exception("Unexpected error while processing %s: %s", nc_file, e)
            skipped += 1

    # 写 summary CSV
    if summaries:
        generate_csv_summary_tool(summaries,summary_csv_path)
        # 写 2) QC results CSV（同目录下另存一份）
        qc_csv_path = os.path.join(output_dir, "Dethier_qc_results.csv")
        generate_qc_results_csv_tool(summaries,qc_csv_path)
        LOGGER.info("Summary CSV saved to: %s", summary_csv_path)
    else:
        LOGGER.warning("No valid stations processed, summary CSV not created.")

    LOGGER.info("Processing finished. Success: %d, Skipped/Failed: %d", success, skipped)

# =========================
# main 入口
# =========================

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

    INPUT_NC_DIR = os.path.join(PROJECT_ROOT, "Source", "Dethier", "nc_convert")
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "Output_r", "monthly","Dethier", "qc")
    SUMMARY_CSV  = os.path.join(OUTPUT_DIR, "Dethier_station_summary.csv")

    process_dethier_data_from_nc(INPUT_NC_DIR, OUTPUT_DIR, SUMMARY_CSV)
