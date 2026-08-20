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
- 站点准入基于实际存在有效 TSS/SSC 的 water-quality records（不要求 Q）
- Q 和 SSC daily series 独立聚合 → outer merge（保留 SSC-only 日期）
- SSL = Q * SSC * 0.0864 仅当 Q 和 SSC 同时有效时计算
- QC3 SSC-Q consistency 仅对有效 Q-SSC pairs 执行
- 不插值、不补齐日期
- 输出 CF-1.8 兼容的 NetCDF 文件

统一规则 (manuscript-consistent):
1. Q 不是站点或记录的准入条件
2. 只要一个站点存在有效 SSC/TSS 或 SSL，就应保留
3. 一个时间点只要 SSC 或 SSL 至少一个有值，就应视为 sediment-eligible
4. 如果只有 SSC 而没有 Q，保留 SSC，Q 和 SSL 设为 missing
5. 只有 Q 和 SSC 同时存在时才计算 SSL = Q * SSC * 0.0864
6. QC3 SSC-Q consistency 只对有效 Q-SSC pairs 执行
7. source-reported values 优先于 derived values
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
from pathlib import Path
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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
try:
    from code.plot import plot_ssc_q_diagnostic
except Exception:
    plot_ssc_q_diagnostic = None
from code.qc import (
    apply_hydro_qc_with_provenance,
    apply_quality_flag,
    apply_quality_flag_array,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.global_attrs import COUNTRY_METADATA
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
            return np.nan
        return val
    except Exception:
        return np.nan

def clean_metadata_text(value):
    """Return a clean scalar metadata string, or empty text for missing values."""
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {'', 'nan', 'none', 'null', '<na>'}:
        return ''
    return text

def first_metadata_text(station_row, columns, default=''):
    for column in columns:
        if column in station_row.index:
            text = clean_metadata_text(station_row.get(column))
            if text:
                return text
    return default

def get_station_display_name(station_row, station_id):
    return first_metadata_text(
        station_row,
        ['Station Identifier', 'Station Narrative', 'Water Body Name'],
        default=str(station_id),
    )

def get_river_name(station_row):
    return first_metadata_text(station_row, ['Water Body Name', 'Main Basin'])

def _safe_fmt(val):
    """Safely format a float for logging; returns 'N/A' for NaN."""
    try:
        v = float(val)
        if np.isfinite(v):
            return f"{v:.2f}"
        return "N/A"
    except (TypeError, ValueError):
        return "N/A"

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
    print(f"  Q:   {_safe_fmt(q_value)} m3/s (flag={int(q_flag)})")
    print(f"  SSC: {_safe_fmt(ssc_value)} mg/L (flag={int(ssc_flag)})")
    print(f"  SSL: {_safe_fmt(ssl_value)} ton/day (flag={int(ssl_flag)})")

def parse_float(value):
    """解析浮点元数据"""
    if pd.isna(value):
        return -9999.0
    try:
        return float(str(value).replace(',', '.'))
    except Exception:
        return -9999.0


def get_station_altitude(station_row, fill_value=-9999.0):
    """
    Return station altitude/elevation from GEMStat metadata.

    GFQA station metadata usually stores this as ``Elevation`` rather than
    ``altitude``. Several possible names are checked so the converter keeps
    working if the source metadata uses a slightly different header.
    """
    altitude_columns = [
        'Elevation',
        'Elevation (m)',
        'Elevation_m',
        'Altitude',
        'Altitude (m)',
        'Altitude_m',
        'altitude',
        'elevation',
    ]

    for col in altitude_columns:
        if col in station_row.index:
            return parse_float(station_row.get(col, fill_value))
    return fill_value


def report_station_metadata_altitude(station_df):
    """Print whether the source station metadata contains altitude/elevation."""
    altitude_columns = [
        'Elevation', 'Elevation (m)', 'Elevation_m',
        'Altitude', 'Altitude (m)', 'Altitude_m',
        'altitude', 'elevation',
    ]
    matched = [c for c in altitude_columns if c in station_df.columns]
    if not matched:
        print('Altitude/Elevation column not found in GEMStat_station_metadata.xlsx')
        print('Available station metadata columns:', list(station_df.columns))
        return

    col = matched[0]
    parsed = station_df[col].apply(parse_float)
    valid = parsed != -9999.0
    print(f"Altitude/Elevation source column: {col} ({int(valid.sum())}/{len(station_df)} valid values)")


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

    flux_df['GEMS.Station.Number'] = flux_df['GEMS.Station.Number'].astype(str).str.strip()
    water_df['GEMS.Station.Number'] = water_df['GEMS.Station.Number'].astype(str).str.strip()
    station_df['GEMS Station Number'] = station_df['GEMS Station Number'].astype(str).str.strip()
    report_station_metadata_altitude(station_df)
    flux_df['Parameter.Code'] = flux_df['Parameter.Code'].astype(str).str.strip()
    water_df['Parameter.Code'] = water_df['Parameter.Code'].astype(str).str.strip()

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
    """找到两个数据集的重叠时间段（保留用于信息输出，不再作为 gate）"""
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
    """计算每日泥沙通量 (ton/day) — 仅当 Q 和 SSC 同时有效"""
    if pd.isna(q) or pd.isna(ssc) or q < 0 or ssc < 0:
        return np.nan
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

    altitude = get_station_altitude(station_row)
    alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
    alt_var.units = 'm'
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station altitude above mean sea level'
    alt_var.positive = 'up'
    alt_var[:] = altitude

    # --------------------------
    # helper: add flag var
    # --------------------------
    def _add_flag_var(name, values, long_name, flag_values, flag_meanings, comment=""):
        v = ds.createVariable(name, 'b', ('time',), fill_value=FILL_VALUE_INT)
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
    q_var.coordinates = "latitude longitude altitude"

    ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    ssc_var.units = 'mg L-1'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.coordinates = "latitude longitude altitude"

    ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    ssl_var.units = 'ton day-1'
    ssl_var.long_name = 'suspended sediment load'
    ssl_var.coordinates = "latitude longitude altitude"

    q_var[:] = np.where(np.isfinite(discharge), discharge, -9999.0)
    ssc_var[:] = np.where(np.isfinite(ssc), ssc, -9999.0)
    ssl_var[:] = np.where(np.isfinite(ssl), ssl, -9999.0)

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
    ds.altitude = altitude
    ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))
    ds.station_id = str(station_id)
    ds.Source_ID = str(station_id)
    ds.source_station_id = str(station_id)
    ds.station_name = get_station_display_name(station_row, station_id)
    ds.river_name = get_river_name(station_row)

    # Geographic metadata
    country_name = str(station_row.get('Country Name', '')).strip()
    country_meta = COUNTRY_METADATA.get(country_name, {})
    ds.country = country_name
    ds.continent_region = country_meta.get('continent_region', '')
    ds.iso_a3 = country_meta.get('iso_a3', '')

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
        station_name=get_station_display_name(station_row, station_id),
        n_samples=len(dates),
        skipped_log_iqr=False,
        skipped_ssc_q=False,
        q_value=float(np.nanmedian(np.asarray(discharge, dtype=float))),
        q_flag=int(np.min(Q_flag)),
        ssc_value=float(np.nanmedian(np.asarray(ssc, dtype=float))),
        ssc_flag=int(np.min(SSC_flag)),
        ssl_value=float(np.nanmedian(np.asarray(ssl, dtype=float))),
        ssl_flag=int(np.min(SSL_flag)),
        created_path=filepath
    )


# ==========================================================
# main processing function
# ==========================================================
def process_one_station(args):
    """
    Process one station with manuscript-consistent rules:
    - Candidate stations from water-quality records (TSS in Water.csv)
    - Q and SSC daily series aggregated independently, then outer-merged
    - SSC-only records preserved; SSL computed only when Q+SSC both valid
    - QC3 SSC-Q consistency only on valid Q-SSC pairs
    - Station skipped only if no valid SSC or SSL remains after QC
    """
    station_id, flux_df, water_df, station_df, output_dir = args

    try:
        print(f"\nProcessing station {station_id}")

        station_match = station_df[station_df['GEMS Station Number'] == station_id]
        if station_match.empty:
            return None, None, None, f"Skipped {station_id}: station metadata not found"

        station_row = station_match.iloc[0]

        # --- 1. Extract raw data ---
        discharge_data, sediment_data = extract_station_data(station_id, flux_df, water_df)

        has_q = len(discharge_data) > 0
        has_ssc = len(sediment_data) > 0

        # --- 2. Aggregate to daily independently ---
        if has_q:
            discharge_daily = aggregate_to_daily(discharge_data)
        else:
            discharge_daily = pd.DataFrame(columns=['Date', 'Clean_Value', 'Quality'])

        if has_ssc:
            sediment_daily = aggregate_to_daily(sediment_data)
        else:
            sediment_daily = pd.DataFrame(columns=['Date', 'Clean_Value', 'Quality'])

        # --- 3. Outer merge: keep all dates where either Q or SSC exists ---
        if has_q and has_ssc:
            merged = pd.merge(
                discharge_daily,
                sediment_daily,
                on='Date',
                how='outer',
                suffixes=('_Q', '_SSC')
            )
        elif has_q and not has_ssc:
            merged = discharge_daily.rename(columns={'Clean_Value': 'Clean_Value_Q', 'Quality': 'Quality_Q'})
            merged['Clean_Value_SSC'] = np.nan
            merged['Quality_SSC'] = 'unknown'
        elif not has_q and has_ssc:
            merged = sediment_daily.rename(columns={'Clean_Value': 'Clean_Value_SSC', 'Quality': 'Quality_SSC'})
            merged['Clean_Value_Q'] = np.nan
            merged['Quality_Q'] = 'unknown'
        else:
            return None, None, None, f"Skipped {station_id}: no Q and no SSC data"

        # --- 4. Compute SSL only where Q AND SSC both valid (Rule 5) ---
        Q_arr = merged["Clean_Value_Q"].to_numpy(dtype=float)
        SSC_arr = merged["Clean_Value_SSC"].to_numpy(dtype=float)
        SSL_arr = np.full(len(merged), np.nan, dtype=float)
        valid_ssl = (
            np.isfinite(Q_arr)
            & np.isfinite(SSC_arr)
            & (Q_arr >= 0)
            & (SSC_arr >= 0)
        )
        SSL_arr[valid_ssl] = Q_arr[valid_ssl] * SSC_arr[valid_ssl] * 0.0864
        merged["SSL"] = SSL_arr

        # Count paired Q-SSC records (for audit)
        n_paired = int(valid_ssl.sum())

        time_arr = pd.to_datetime(merged["Date"]).values

        # --- 5. Run QC pipeline ---
        qc = apply_hydro_qc_with_provenance(
            time=time_arr,
            Q=Q_arr,
            SSC=SSC_arr,
            SSL=SSL_arr,
            Q_is_independent=True,
            SSC_is_independent=True,
            SSL_is_independent=False,
            ssl_is_derived_from_q_ssc=True,
            qc2_k=1.5,
            qc2_min_samples=5,
            qc3_k=1.5,
            qc3_min_samples=5,
        )

        if qc is None:
            return None, None, None, f"Skipped {station_id}: QC produced no valid data"

        # --- 6. Final station skip: based on SSC or SSL at least one valid (Rule 2) ---
        ssc_valid = (qc["SSC_flag"] != FILL_VALUE_INT)
        ssl_valid = (qc["SSL_flag"] != FILL_VALUE_INT)
        if not np.any(ssc_valid | ssl_valid):
            return None, None, None, f"Skipped {station_id}: no valid SSC or SSL after QC"

        # Track final SSC record count (for audit)
        n_final_ssc = int(np.sum(ssc_valid))

        merged['Q_flag'] = qc['Q_flag']
        merged['SSC_flag'] = qc['SSC_flag']
        merged['SSL_flag'] = qc['SSL_flag']

        merged['Q_flag_qc1_physical'] = qc.get(
            'Q_flag_qc1_physical',
            np.full(len(merged), FILL_VALUE_INT, dtype=np.int8)
        )
        merged['SSC_flag_qc1_physical'] = qc.get(
            'SSC_flag_qc1_physical',
            np.full(len(merged), FILL_VALUE_INT, dtype=np.int8)
        )
        merged['SSL_flag_qc1_physical'] = qc.get(
            'SSL_flag_qc1_physical',
            np.full(len(merged), FILL_VALUE_INT, dtype=np.int8)
        )

        merged['Q_flag_qc2_log_iqr'] = qc.get(
            'Q_flag_qc2_log_iqr',
            np.full(len(merged), 8, dtype=np.int8)
        )
        merged['SSC_flag_qc2_log_iqr'] = qc.get(
            'SSC_flag_qc2_log_iqr',
            np.full(len(merged), 8, dtype=np.int8)
        )
        merged['SSL_flag_qc2_log_iqr'] = qc.get(
            'SSL_flag_qc2_log_iqr',
            np.full(len(merged), 8, dtype=np.int8)
        )

        merged['SSC_flag_qc3_ssc_q'] = qc.get(
            'SSC_flag_qc3_ssc_q',
            np.full(len(merged), 8, dtype=np.int8)
        )
        merged['SSL_flag_qc3_from_ssc_q'] = qc.get(
            'SSL_flag_qc3_from_ssc_q',
            np.full(len(merged), 8, dtype=np.int8)
        )

        ssc_q_bounds = qc.get("ssc_q_bounds", None)

        # Diagnostic plot: only when ssc_q_bounds is available (requires Q-SSC pairs)
        if ssc_q_bounds is not None and plot_ssc_q_diagnostic is not None:
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
                station_name=get_station_display_name(station_row, station_id),
                out_png=str(out_png),
            )

        export_df = merged.copy()
        export_df['Station_ID'] = station_id

        lat, lon = parse_lat_lon(station_row)

        def _count_final(f):
            f = np.asarray(f, dtype=np.int8)
            return {
                "good": int(np.sum(f == 0)),
                "estimated": int(np.sum(f == 1)),
                "suspect": int(np.sum(f == 2)),
                "bad": int(np.sum(f == 3)),
                "missing": int(np.sum(f == FILL_VALUE_INT)),
            }

        def _count_step(f, mapping):
            f = np.asarray(f, dtype=np.int8)
            return {k: int(np.sum(f == np.int8(v))) for k, v in mapping.items()}

        station_info = {
            "station_name": get_station_display_name(station_row, station_id),
            "Source_ID": station_id,
            "longitude": lon,
            "latitude": lat,
            "QC_n_days": int(len(merged)),
        }

        c = _count_final(merged["Q_flag"].to_numpy())
        station_info.update({f"Q_final_{k}": v for k, v in c.items()})

        c = _count_final(merged["SSC_flag"].to_numpy())
        station_info.update({f"SSC_final_{k}": v for k, v in c.items()})

        c = _count_final(merged["SSL_flag"].to_numpy())
        station_info.update({f"SSL_final_{k}": v for k, v in c.items()})

        qc1_map = {"pass": 0, "bad": 3, "missing": 9}
        qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}

        c = _count_step(merged["Q_flag_qc1_physical"].to_numpy(), qc1_map)
        station_info.update({f"Q_qc1_{k}": v for k, v in c.items()})

        c = _count_step(merged["Q_flag_qc2_log_iqr"].to_numpy(), qc2_map)
        station_info.update({f"Q_qc2_{k}": v for k, v in c.items()})

        c = _count_step(merged["SSC_flag_qc1_physical"].to_numpy(), qc1_map)
        station_info.update({f"SSC_qc1_{k}": v for k, v in c.items()})

        c = _count_step(merged["SSC_flag_qc2_log_iqr"].to_numpy(), qc2_map)
        station_info.update({f"SSC_qc2_{k}": v for k, v in c.items()})

        c = _count_step(merged["SSL_flag_qc1_physical"].to_numpy(), qc1_map)
        station_info.update({f"SSL_qc1_{k}": v for k, v in c.items()})

        c = _count_step(merged["SSL_flag_qc2_log_iqr"].to_numpy(), qc2_map)
        station_info.update({f"SSL_qc2_{k}": v for k, v in c.items()})

        create_netcdf_file(
            station_id=station_id,
            station_row=station_row,
            qc=qc,
            q_quality=merged['Quality_Q'].fillna('unknown').to_numpy(),
            ssc_quality=merged['Quality_SSC'].fillna('unknown').to_numpy(),
            output_dir=output_dir,
        )

        # Attach audit info to the return
        audit = {
            "has_q": has_q,
            "has_ssc": has_ssc,
            "n_raw_ssc": len(sediment_data),
            "n_paired": n_paired,
            "n_final_ssc": n_final_ssc,
        }

        return export_df, station_info, audit, f"Finished {station_id}"

    except Exception as e:
        return None, None, None, f"Failed {station_id}: {repr(e)}"


def process_all_stations(flux_df, water_df, station_df, output_dir):
    all_records = []
    stations_info = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ================================================================
    # CANDIDATE STATIONS: from water-quality records with TSS (Rule 1-2)
    # 不再使用 flux_stations & water_stations 的交集
    # ================================================================
    tss_water = water_df[water_df['Parameter.Code'] == 'TSS']
    ssc_stations = set(tss_water['GEMS.Station.Number'].unique())
    flux_stations = set(flux_df['GEMS.Station.Number'].unique())

    stations_with_q = ssc_stations & flux_stations
    ssc_only_stations = ssc_stations - flux_stations

    print(f"\n{'='*60}")
    print(f"AUDIT: Candidate station summary")
    print(f"{'='*60}")
    print(f"  Raw SSC-bearing stations (TSS in Water.csv):  {len(ssc_stations)}")
    print(f"  Stations with Q+SSC:                           {len(stations_with_q)}")
    print(f"  SSC-only stations (no Q in Flux.csv):          {len(ssc_only_stations)}")

    tasks = [
        (station_id, flux_df, water_df, station_df, output_dir)
        for station_id in sorted(ssc_stations)
    ]

    max_workers = min(24, max(1, mp.cpu_count() - 1))

    # Audit accumulators
    total_raw_ssc_records = 0
    total_paired_records = 0
    total_final_ssc_records = 0
    final_station_ids = []
    final_q_ssc_stations = []
    final_ssc_only_stations = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_station, task) for task in tasks]

        for future in as_completed(futures):
            export_df, station_info, audit, message = future.result()
            print(message)

            if export_df is not None:
                all_records.append(export_df)

            if station_info is not None:
                stations_info.append(station_info)
                sid = station_info["Source_ID"]
                final_station_ids.append(sid)
                if sid in ssc_only_stations:
                    final_ssc_only_stations.append(sid)
                else:
                    final_q_ssc_stations.append(sid)

            if audit is not None:
                total_raw_ssc_records += audit["n_raw_ssc"]
                total_paired_records += audit["n_paired"]
                total_final_ssc_records += audit["n_final_ssc"]

    # === AUDIT REPORT ===
    print(f"\n{'='*60}")
    print(f"AUDIT: Final output summary")
    print(f"{'='*60}")
    print(f"  Final output stations:                         {len(final_station_ids)}")
    print(f"    with Q+SSC:                                  {len(final_q_ssc_stations)}")
    print(f"    SSC-only (no Q, still output):               {len(final_ssc_only_stations)}")
    print(f"  SSC raw records (from Water.csv TSS rows):     {total_raw_ssc_records}")
    print(f"  Paired Q-SSC records (same-day Q & SSC):       {total_paired_records}")
    print(f"  Final retained SSC records (after QC):         {total_final_ssc_records}")
    print(f"{'='*60}")

    # === 所有站点合并输出 Excel ===
    if all_records:
        big_df = pd.concat(all_records, ignore_index=True)
        big_df = big_df.sort_values(["Station_ID", "Date"]).reset_index(drop=True)

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
