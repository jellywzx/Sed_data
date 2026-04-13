#!/usr/bin/env python3
"""
Convert GFQA GEMStat data to NetCDF format (Observed daily data only)
with Dual Quality Control:
---------------------------------------------------------------------
功能：
- 从 Flux.csv / Water.csv / GEMStat_station_metadata.csv 读取原始数据
- 提取流量(Q-Inst)与悬浮泥沙浓度(TSS)数据
- 输出含两类质量信息：
  1. Data.Quality（来自原始CSV）
  2. QC Flags（自动判断）
- 仅保留“流量与泥沙在同一天都有观测”的日期
- 不插值、不补齐日期
- 输出 CF-1.8 兼容的 NetCDF 文件
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
def find_project_root(start_dir, max_up=6):
    p = Path(start_dir).resolve()
    for _ in range(max_up):
        if (p / "Source").exists() and (p / "Output_r").exists():
            return p
        p = p.parent
    return Path(start_dir).resolve().parent

if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.output import (
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_hydro_qc_with_provenance,
    apply_quality_flag,
    apply_quality_flag_array,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.runtime import resolve_output_root, resolve_source_root
from code.units import convert_ssl_units_if_needed

SOURCE_DIR = resolve_source_root(start=__file__) / "GFQA_v2" / "sed"


# ==========================================================
# 通用函数
# ==========================================================

def get_flag(value, thresholds, meanings):
    """根据阈值判断数据质量等级"""
    if pd.isna(value) or value == -9999.0:
        return meanings.split().index('missing_data')
    if value < thresholds.get('negative', -float('inf')):
        return meanings.split().index('bad_data')
    if value == thresholds.get('zero', -1):
        return meanings.split().index('suspect_data')
    if value > thresholds.get('extreme', float('inf')):
        return meanings.split().index('suspect_data')
    return meanings.split().index('good_data')


def clean_value(value):
    """清洗数值"""
    try:
        val = float(str(value).replace(',', '.'))
        if np.isnan(val) or val < 0:
            return -9999.0
        return val
    except Exception:
        return -9999.0

def log_station_qc(station_id, station_name, n_samples,
                   skipped_log_iqr, skipped_ssc_q,
                   q_value, q_flag, ssc_value, ssc_flag, ssl_value, ssl_flag,
                   created_path):
    print(f"\nProcessing: {station_name} ({station_id}) +")
    if skipped_log_iqr:
        print(f"  [{station_name} ({station_id})] Sample size = {n_samples} < 5, log-IQR statistical QC skipped.")
    if skipped_ssc_q:
        print(f"  [{station_name} ({station_id})] Sample size = {n_samples} < 5, SSC-Q consistency check and diagnostic plot skipped.")
    print(f"✓ Created: {created_path}")
    print(f"  Q:   {q_value:.2f} m3/s (flag={int(q_flag)})")
    print(f"  SSC: {ssc_value:.2f} mg/L (flag={int(ssc_flag)})")
    print(f"  SSL: {ssl_value:.2f} ton/day (flag={int(ssl_flag)})")

def parse_float(value):
    """解析浮点元数据"""
    if pd.isna(value):
        return -9999.0
    try:
        return float(str(value).replace(',', '.'))
    except Exception:
        return -9999.0


# ==========================================================
# 数据读取与预处理
# ==========================================================

def read_csv_files():
    """读取 CSV 文件"""
    print("Reading CSV files...")
    base_dir = SOURCE_DIR
    flux_df = pd.read_csv(base_dir / "Flux.csv", delimiter=';', parse_dates=['Sample.Date'], encoding='iso-8859-1')
    water_df = pd.read_csv(base_dir / "Water.csv", delimiter=';', parse_dates=['Sample.Date'], encoding='iso-8859-1')
    station_df = pd.read_excel(base_dir / "GEMStat_station_metadata.xlsx")

    # print(flux_df['Sample.Date'].head())
    # print(water_df['Sample.Date'].head())
    flux_df['GEMS.Station.Number'] = flux_df['GEMS.Station.Number'].astype(str).str.strip()
    water_df['GEMS.Station.Number'] = water_df['GEMS.Station.Number'].astype(str).str.strip()
    station_df['GEMS Station Number'] = station_df['GEMS Station Number'].astype(str).str.strip()
    flux_df['Parameter.Code'] = flux_df['Parameter.Code'].astype(str).str.strip()
    water_df['Parameter.Code'] = water_df['Parameter.Code'].astype(str).str.strip()
    # print("Flux station sample:", list(flux_stations)[:5])
    # print("Water station sample:", list(water_stations)[:5])
    # print("Intersection size:", len(common_stations))


    print(f"Flux records: {len(flux_df)}")
    print(f"Water records: {len(water_df)}")
    print(f"Stations: {len(station_df)}")
    return flux_df, water_df, station_df


def extract_station_data(station_id, flux_df, water_df):
    """提取指定测站的流量与TSS数据"""
    discharge_data = flux_df[
        (flux_df['GEMS.Station.Number'] == station_id) &
        (flux_df['Parameter.Code'] == 'Q-Inst')
    ].copy()

    sediment_data = water_df[
        (water_df['GEMS.Station.Number'] == station_id) &
        (water_df['Parameter.Code'] == 'TSS')
    ].copy()

    return discharge_data, sediment_data


def find_overlapping_period(discharge_data, sediment_data):
    """找到两个数据集的重叠时间段"""
    if len(discharge_data) == 0 or len(sediment_data) == 0:
        return None, None
    start = max(discharge_data['Sample.Date'].min(), sediment_data['Sample.Date'].min())
    end = min(discharge_data['Sample.Date'].max(), sediment_data['Sample.Date'].max())
    if start > end:
        return None, None
    return start, end


def aggregate_to_daily(data, date_col='Sample.Date', value_col='Value', quality_col='Data.Quality'):
    """按日聚合（取同日平均）并附加原始Data.Quality"""
    data = data.copy()
    data['Date'] = data[date_col].dt.floor('D')
    data['Clean_Value'] = data[value_col].apply(clean_value)

    daily = (
        data.groupby('Date')
        .agg({
            'Clean_Value': 'mean',
            quality_col: lambda x: x.mode().iat[0] if not x.mode().empty else 'unknown'
        })
        .reset_index()
        .rename(columns={quality_col: 'Quality'})
    )
    return daily

def parse_lat_lon(station_row):
    lat = float(str(station_row['Latitude']).replace(',', '.'))
    lon = float(str(station_row['Longitude']).replace(',', '.'))
    return lat, lon

# ==========================================================
# 计算与文件输出
# ==========================================================
def calculate_sediment_load(q, ssc):
    """计算每日泥沙通量 (ton/day)"""
    if q == -9999.0 or ssc == -9999.0:
        return -9999.0
    return q * ssc * 0.0864

def create_netcdf_file(station_id, station_row, qc, q_quality, ssc_quality, output_dir):
    """创建 NetCDF 文件（含最终QC + 分步QC flags + 原始Data.Quality）"""

    filename = f"GFQA_{station_id}.nc"
    filepath = os.path.join(output_dir, filename)
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # --------------------------
    # unpack qc dict
    # --------------------------
    dates = qc["time"]
    discharge = qc["Q"]
    ssc = qc["SSC"]
    ssl = qc["SSL"]

    Q_flag   = qc["Q_flag"].astype(np.int8)
    SSC_flag = qc["SSC_flag"].astype(np.int8)
    SSL_flag = qc["SSL_flag"].astype(np.int8)

    # step/provenance flags
    Q_flag_qc1   = qc.get("Q_flag_qc1_physical")
    SSC_flag_qc1 = qc.get("SSC_flag_qc1_physical")
    SSL_flag_qc1 = qc.get("SSL_flag_qc1_physical")

    Q_flag_qc2   = qc.get("Q_flag_qc2_log_iqr")
    SSC_flag_qc2 = qc.get("SSC_flag_qc2_log_iqr")
    SSL_flag_qc2 = qc.get("SSL_flag_qc2_log_iqr")

    SSC_flag_qc3 = qc.get("SSC_flag_qc3_ssc_q")
    SSL_flag_qc3 = qc.get("SSL_flag_qc3_from_ssc_q")

    # --------------------------
    # dimensions / time
    # --------------------------
    ds.createDimension('time', len(dates))
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.standard_name = 'time'
    time_var.calendar = 'gregorian'
    time_var[:] = [(pd.Timestamp(d) - pd.Timestamp('1970-01-01')).days for d in dates]

    # --------------------------
    # coords
    # --------------------------
    lat, lon = parse_lat_lon(station_row)

    lat_var = ds.createVariable('latitude', 'f4')
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var[:] = lat

    lon_var = ds.createVariable('longitude', 'f4')
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var[:] = lon

    # --------------------------
    # helper: add flag var
    # --------------------------
    def _add_flag_var(name, values, long_name, flag_values, flag_meanings, comment=""):
        v = ds.createVariable(name, 'b', ('time',), fill_value=-127)
        v.long_name = long_name
        v.flag_values = np.array(flag_values, dtype=np.byte)
        v.flag_meanings = flag_meanings
        if comment:
            v.comment = comment
        v[:] = np.asarray(values, dtype=np.int8)
        return v

    # --------------------------
    # main vars
    # --------------------------
    q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
    q_var.units = 'm3 s-1'
    q_var.long_name = 'river discharge'
    q_var.coordinates = "latitude longitude"

    ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    ssc_var.units = 'mg L-1'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.coordinates = "latitude longitude"

    ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    ssl_var.units = 'ton day-1'
    ssl_var.long_name = 'suspended sediment load'
    ssl_var.coordinates = "latitude longitude"

    q_var[:] = discharge
    ssc_var[:] = ssc
    ssl_var[:] = ssl

    # --------------------------
    # flags: final
    # final flag convention: 0 good, 1 estimated, 2 suspect, 3 bad, 9 missing
    # --------------------------
    final_meanings = "good_data estimated_data suspect_data bad_data missing_data"
    _add_flag_var(
        "Q_flag", Q_flag, "final quality flag for discharge",
        flag_values=[0, 1, 2, 3, 9],
        flag_meanings=final_meanings
    )

    _add_flag_var(
        "SSC_flag", SSC_flag, "final quality flag for SSC",
        flag_values=[0, 1, 2, 3, 9],
        flag_meanings=final_meanings
    )
    _add_flag_var(
        "SSL_flag", SSL_flag, "final quality flag for sediment load",
        flag_values=[0, 1, 2, 3, 9],
        flag_meanings=final_meanings
    )

    # --------------------------
    # flags: step/provenance
    # QC1: 0 pass, 3 bad, 9 missing
    # QC2: 0 pass, 2 suspect, 8 not_checked, 9 missing
    # QC3 SSC–Q: 0 pass, 2 suspect, 8 not_checked, 9 missing
    # QC3 SSL propagation: 0 not_propagated, 2 propagated, 8 not_checked, 9 missing
    # --------------------------
    if Q_flag_qc1 is not None:
        _add_flag_var("Q_flag_qc1_physical", Q_flag_qc1, "QC1 physical check flag for discharge",
                      flag_values=[0, 3, 9],
                      flag_meanings="pass bad missing")
    if SSC_flag_qc1 is not None:
        _add_flag_var("SSC_flag_qc1_physical", SSC_flag_qc1, "QC1 physical check flag for SSC",
                      flag_values=[0, 3, 9],
                      flag_meanings="pass bad missing")
    if SSL_flag_qc1 is not None:
        _add_flag_var("SSL_flag_qc1_physical", SSL_flag_qc1, "QC1 physical check flag for SSL",
                      flag_values=[0, 3, 9],
                      flag_meanings="pass bad missing")

    if Q_flag_qc2 is not None:
        _add_flag_var("Q_flag_qc2_log_iqr", Q_flag_qc2, "QC2 log-IQR screening flag for discharge",
                      flag_values=[0, 2, 8, 9],
                      flag_meanings="pass suspect not_checked missing")
    if SSC_flag_qc2 is not None:
        _add_flag_var("SSC_flag_qc2_log_iqr", SSC_flag_qc2, "QC2 log-IQR screening flag for SSC",
                      flag_values=[0, 2, 8, 9],
                      flag_meanings="pass suspect not_checked missing")
    if SSL_flag_qc2 is not None:
        _add_flag_var("SSL_flag_qc2_log_iqr", SSL_flag_qc2, "QC2 log-IQR screening flag for SSL",
                      flag_values=[0, 2, 8, 9],
                      flag_meanings="pass suspect not_checked missing")

    if SSC_flag_qc3 is not None:
        _add_flag_var("SSC_flag_qc3_ssc_q", SSC_flag_qc3, "QC3 SSC–Q consistency flag for SSC",
                      flag_values=[0, 2, 8, 9],
                      flag_meanings="pass suspect not_checked missing")

    if SSL_flag_qc3 is not None:
        _add_flag_var("SSL_flag_qc3_from_ssc_q", SSL_flag_qc3, "QC3 propagation flag to SSL from SSC–Q",
                      flag_values=[0, 2, 8, 9],
                      flag_meanings="not_propagated propagated not_checked missing")

    # --------------------------
    # attach ancillary_variables (关键：分步写入靠这个关联)
    # --------------------------
    q_anc = ["Q_flag"]
    if Q_flag_qc1 is not None: q_anc.append("Q_flag_qc1_physical")
    if Q_flag_qc2 is not None: q_anc.append("Q_flag_qc2_log_iqr")
    q_var.ancillary_variables = " ".join(q_anc)

    ssc_anc = ["SSC_flag"]
    if SSC_flag_qc1 is not None: ssc_anc.append("SSC_flag_qc1_physical")
    if SSC_flag_qc2 is not None: ssc_anc.append("SSC_flag_qc2_log_iqr")
    if SSC_flag_qc3 is not None: ssc_anc.append("SSC_flag_qc3_ssc_q")
    ssc_var.ancillary_variables = " ".join(ssc_anc)

    ssl_anc = ["SSL_flag"]
    if SSL_flag_qc1 is not None: ssl_anc.append("SSL_flag_qc1_physical")
    if SSL_flag_qc2 is not None: ssl_anc.append("SSL_flag_qc2_log_iqr")
    if SSL_flag_qc3 is not None: ssl_anc.append("SSL_flag_qc3_from_ssc_q")
    ssl_var.ancillary_variables = " ".join(ssl_anc)

    # --------------------------
    # original Data.Quality text vars
    # --------------------------
    q_quality_var = ds.createVariable('Q_quality', str, ('time',))
    q_quality_var.long_name = 'data quality label for discharge'
    q_quality_var.comment = 'Original Data.Quality from Flux.csv'
    q_quality_var[:] = np.array(q_quality, dtype='object')

    ssc_quality_var = ds.createVariable('SSC_quality', str, ('time',))
    ssc_quality_var.long_name = 'data quality label for SSC'
    ssc_quality_var.comment = 'Original Data.Quality from Water.csv'
    ssc_quality_var[:] = np.array(ssc_quality, dtype='object')

    # --------------------------
    # scalar metadata
    # --------------------------
    ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
    ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))

    ds.Conventions = 'CF-1.8'
    ds.title = f'GFQA Daily Observed Sediment and Discharge Data for Station {station_id}'
    ds.comment = (
        'Includes: (1) final QC flags and (2) step-level QC provenance flags, '
        'plus original Data.Quality labels from source CSV.'
    )
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by gfqa_to_netcdf_daily_dualqc.py'
    ds.close()

    print(f"✅ Created file: {filename}")
    log_station_qc(
        station_id=station_id,
        station_name=str(station_row.get('Station Name', station_id)),
        n_samples=len(discharge),
        skipped_log_iqr=False,
        skipped_ssc_q=False,
        q_value=float(np.nanmedian(np.asarray(discharge, dtype=float))), q_flag=int(np.min(Q_flag)),
        ssc_value=float(np.nanmedian(np.asarray(ssc, dtype=float))), ssc_flag=int(np.min(SSC_flag)),
        ssl_value=float(np.nanmedian(np.asarray(ssl, dtype=float))), ssl_flag=int(np.min(SSL_flag)),
        created_path=filepath
    )


# ==========================================================
# main processing function
# ==========================================================

def process_all_stations(flux_df, water_df, station_df, output_dir):
    all_records = []
    stations_info = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    flux_stations = set(flux_df['GEMS.Station.Number'].unique())
    water_stations = set(water_df['GEMS.Station.Number'].unique())
    common_stations = flux_stations & water_stations

    # print("Flux station sample:", list(flux_stations)[:5])
    # print("Water station sample:", list(water_stations)[:5])
    # print("Intersection size:", len(common_stations))


    for station_id in sorted(common_stations):
        print(f"\nProcessing station {station_id}")
        station_row = station_df[station_df['GEMS Station Number'] == station_id].iloc[0]


        discharge_data, sediment_data = extract_station_data(station_id, flux_df, water_df)
        start, end = find_overlapping_period(discharge_data, sediment_data)
        if start is None:
            print("  ⚠️ Skipped: no overlapping period")
            continue

        discharge_daily = aggregate_to_daily(discharge_data)
        sediment_daily = aggregate_to_daily(sediment_data)
        merged = pd.merge(discharge_daily, sediment_daily, on='Date', how='inner', suffixes=('_Q', '_SSC'))
        if merged.empty:
            print("  ⚠️ Skipped: no same-day data")
            continue

        merged['SSL'] = merged['Clean_Value_Q'] * merged['Clean_Value_SSC'] * 0.0864
        
        # ==================================================
        # ✅ 用 tool.py 的一键QC（含分步provenance flags）
        # ==================================================
        time_arr = pd.to_datetime(merged['Date']).values
        Q_arr = merged['Clean_Value_Q'].to_numpy(dtype=float)
        SSC_arr = merged['Clean_Value_SSC'].to_numpy(dtype=float)
        SSL_arr = merged['SSL'].to_numpy(dtype=float)

        qc = apply_hydro_qc_with_provenance(
            time=time_arr,
            Q=Q_arr,
            SSC=SSC_arr,
            SSL=SSL_arr,
            Q_is_independent=True,
            SSC_is_independent=True,
            SSL_is_independent=False,          # SSL = Q*SSC 推导
            ssl_is_derived_from_q_ssc=True,
            qc2_k=1.5,
            qc2_min_samples=5,
            qc3_k=1.5,
            qc3_min_samples=5,
        )
        if qc is None:
            print("  ⚠️ Skipped: QC produced no valid data")
            continue

        # 写回 merged（用于导出/诊断图）
        merged['Q_flag'] = qc['Q_flag']
        merged['SSC_flag'] = qc['SSC_flag']
        merged['SSL_flag'] = qc['SSL_flag']

        merged['Q_flag_qc1_physical'] = qc.get('Q_flag_qc1_physical', np.full(len(merged), FILL_VALUE_INT, dtype=np.int8))
        merged['SSC_flag_qc1_physical'] = qc.get('SSC_flag_qc1_physical', np.full(len(merged), FILL_VALUE_INT, dtype=np.int8))
        merged['SSL_flag_qc1_physical'] = qc.get('SSL_flag_qc1_physical', np.full(len(merged), FILL_VALUE_INT, dtype=np.int8))

        merged['Q_flag_qc2_log_iqr'] = qc.get('Q_flag_qc2_log_iqr', np.full(len(merged), 8, dtype=np.int8))
        merged['SSC_flag_qc2_log_iqr'] = qc.get('SSC_flag_qc2_log_iqr', np.full(len(merged), 8, dtype=np.int8))
        merged['SSL_flag_qc2_log_iqr'] = qc.get('SSL_flag_qc2_log_iqr', np.full(len(merged), 8, dtype=np.int8))

        merged['SSC_flag_qc3_ssc_q'] = qc.get('SSC_flag_qc3_ssc_q', np.full(len(merged), 8, dtype=np.int8))
        merged['SSL_flag_qc3_from_ssc_q'] = qc.get('SSL_flag_qc3_from_ssc_q', np.full(len(merged), 8, dtype=np.int8))

        ssc_q_bounds = qc.get("ssc_q_bounds", None)
            
        # --------------------------------------------------
        # SSC–Q diagnostic plot (station-level)
        # --------------------------------------------------
        if ssc_q_bounds is not None:
            plot_dir = Path(output_dir) / "diagnostic"
            plot_dir.mkdir(exist_ok=True)

            out_png = plot_dir / f"GFQA_{station_id}_ssc_q_diagnostic.png"

            plot_ssc_q_diagnostic(
                time=pd.to_datetime(merged['Date']).values,
                Q=merged['Clean_Value_Q'].values,
                SSC=merged['Clean_Value_SSC'].values,
                Q_flag=merged['Q_flag'].values,
                SSC_flag=merged['SSC_flag'].values,
                ssc_q_bounds=ssc_q_bounds,
                station_id=station_id,
                station_name=str(station_row.get('Station Name', station_id)),
                out_png=str(out_png),
            )

        # === 收集所有站点的合并数据 ===
        export_df = merged.copy()
        export_df['Station_ID'] = station_id     # 加入站点号
        all_records.append(export_df)
        # ==========================
        # 站点QC统计（用于CSV汇总）
        # ==========================
        lat, lon = parse_lat_lon(station_row)

        def _count_final(f):
            f = np.asarray(f, dtype=np.int8)
            return {
                "good": int(np.sum(f == 0)),
                "estimated": int(np.sum(f == 1)),
                "suspect": int(np.sum(f == 2)),
                "bad": int(np.sum(f == 3)),
                "missing": int(np.sum(f == FILL_VALUE_INT)),  # 9
            }

        def _count_step(f, mapping):
            f = np.asarray(f, dtype=np.int8)
            return {k: int(np.sum(f == np.int8(v))) for k, v in mapping.items()}

        station_info = {
            "station_name": str(station_row.get("Station Name", station_id)),
            "Source_ID": station_id,
            "longitude": lon,
            "latitude": lat,
            "QC_n_days": int(len(merged)),
        }

        # final flags（你现在 merged 里已有 Q_flag/SSC_flag/SSL_flag）
        c = _count_final(merged["Q_flag"].to_numpy());   station_info.update({f"Q_final_{k}": v for k, v in c.items()})
        c = _count_final(merged["SSC_flag"].to_numpy()); station_info.update({f"SSC_final_{k}": v for k, v in c.items()})
        c = _count_final(merged["SSL_flag"].to_numpy()); station_info.update({f"SSL_final_{k}": v for k, v in c.items()})

        # 如果你已经把分步 flags 写回 merged（比如 Q_flag_qc1_physical 等），这里再统计：
        qc1_map = {"pass": 0, "bad": 3, "missing": 9}
        qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}

        if "Q_flag_qc1_physical" in merged.columns:
            c = _count_step(merged["Q_flag_qc1_physical"].to_numpy(), qc1_map)
            station_info.update({f"Q_qc1_{k}": v for k, v in c.items()})
        if "Q_flag_qc2_log_iqr" in merged.columns:
            c = _count_step(merged["Q_flag_qc2_log_iqr"].to_numpy(), qc2_map)
            station_info.update({f"Q_qc2_{k}": v for k, v in c.items()})

        # SSC/SSL 分步同理（你有这些列就统计，没有就自动跳过）
        if "SSC_flag_qc1_physical" in merged.columns:
            c = _count_step(merged["SSC_flag_qc1_physical"].to_numpy(), qc1_map)
            station_info.update({f"SSC_qc1_{k}": v for k, v in c.items()})
        if "SSC_flag_qc2_log_iqr" in merged.columns:
            c = _count_step(merged["SSC_flag_qc2_log_iqr"].to_numpy(), qc2_map)
            station_info.update({f"SSC_qc2_{k}": v for k, v in c.items()})
        if "SSL_flag_qc1_physical" in merged.columns:
            c = _count_step(merged["SSL_flag_qc1_physical"].to_numpy(), qc1_map)
            station_info.update({f"SSL_qc1_{k}": v for k, v in c.items()})
        if "SSL_flag_qc2_log_iqr" in merged.columns:
            c = _count_step(merged["SSL_flag_qc2_log_iqr"].to_numpy(), qc2_map)
            station_info.update({f"SSL_qc2_{k}": v for k, v in c.items()})

        stations_info.append(station_info)

        create_netcdf_file(
            station_id=station_id,
            station_row=station_row,
            qc=qc,
            q_quality=merged['Quality_Q'].fillna('unknown').to_numpy(),
            ssc_quality=merged['Quality_SSC'].fillna('unknown').to_numpy(),
            output_dir=output_dir,
        )


        # errors, warnings = check_nc_completeness(filepath, strict=False)

        # if errors:
        #     print("  ❌ NetCDF CF/ACDD compliance errors:")
        #     for e in errors:
        #         print(f"     - {e}")
        #     raise RuntimeError("NetCDF completeness check failed")

        # if warnings:
        #     print("  ⚠️ NetCDF CF/ACDD compliance warnings:")
        #     for w in warnings:
        #         print(f"     - {w}")

    # === 所有站点合并输出 Excel ===
    if all_records:
        big_df = pd.concat(all_records, ignore_index=True)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "GFQA_all_stations.xlsx"
        big_df.to_excel(out_path, index=False)

        print(f"\n📘 Saved merged Excel for all stations: {out_path}")
    # === 输出两个CSV汇总 ===
    if stations_info:
        out_dir = Path(output_dir)
        generate_csv_summary_tool(
            stations_info,
            str(out_dir / "GFQA_station_summary.csv")
        )
        generate_qc_results_csv_tool(
            stations_info,
            str(out_dir / "GFQA_station_qc_results.csv")
        )




def main():
    print("=" * 60)
    print("GFQA Observed Daily Data → NetCDF Conversion with Dual QC")
    print("=" * 60)

    flux_df, water_df, station_df = read_csv_files()
    
    output_dir = str(resolve_output_root(start=__file__) / "daily" / "GFQA_v2" / "qc")
    process_all_stations(flux_df, water_df, station_df, output_dir=output_dir)
    print("\nConversion complete with Data.Quality and QC Flags!")


if __name__ == '__main__':
    main()
