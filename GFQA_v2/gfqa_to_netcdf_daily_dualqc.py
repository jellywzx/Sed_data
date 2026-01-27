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
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
def find_project_root(start_dir, max_up=6):
    p = Path(start_dir).resolve()
    for _ in range(max_up):
        if (p / "Source").exists() and (p / "Output_r").exists():
            return p
        p = p.parent
    return Path(start_dir).resolve().parent

PROJECT_ROOT = find_project_root(CURRENT_DIR)
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
)


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
    base_dir = PROJECT_ROOT / "Source" / "GFQA_v2" / "sed"
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


def create_netcdf_file(station_id, station_row, dates,
                       discharge, ssc, ssl,
                       q_flag, ssc_flag, ssl_flag,
                       q_quality, ssc_quality,
                       output_dir):
    """创建 NetCDF 文件（含自动QC与Data.Quality）"""

    filename = f"GFQA_{station_id}.nc"
    filepath = os.path.join(output_dir, filename)
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # 时间维度
    ds.createDimension('time', len(dates))
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.standard_name = 'time'
    time_var.calendar = 'gregorian'
    time_var[:] = [(pd.Timestamp(d) - pd.Timestamp('1970-01-01')).days for d in dates]

    # 主变量
    q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
    q_var.units = 'm3 s-1'
    q_var.long_name = 'river discharge'
    q_var.ancillary_variables = 'Q_flag'
    q_var[:] = discharge

    ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    ssc_var.units = 'mg L-1'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.ancillary_variables = 'SSC_flag'
    ssc_var[:] = ssc

    ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    ssl_var.units = 'ton day-1'
    ssl_var.long_name = 'suspended sediment load'
    ssl_var.ancillary_variables = 'SSL_flag'
    ssl_var[:] = ssl

    q_var.coordinates = "latitude longitude"
    ssc_var.coordinates = "latitude longitude"
    ssl_var.coordinates = "latitude longitude"

    
    lat, lon = parse_lat_lon(station_row)

    # ds.latitude = lat
    # ds.longitude = lon
    # ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
    # ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))

    lat_var = ds.createVariable('latitude', 'f4')
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var[:] = lat

    lon_var = ds.createVariable('longitude', 'f4')
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var[:] = lon

    
    # 自动QC标志
    flag_meanings = "good_data suspect_data bad_data missing_data"
    for name, values, desc in zip(
        ['Q_flag', 'SSC_flag', 'SSL_flag'],
        [q_flag, ssc_flag, ssl_flag],
        ['discharge', 'SSC', 'sediment load']
    ):
        var = ds.createVariable(name, 'b', ('time',), fill_value=-127)
        var.long_name = f'quality flag for {desc}'
        var.flag_values = np.array([0, 2, 3, 9], dtype=np.byte)
        var.flag_meanings = flag_meanings
        var.comment = "good_data suspect_data bad_data missing_data"
        var[:] = values

    # 原始Data.Quality字符串变量
    q_quality_var = ds.createVariable('Q_quality', str, ('time',))
    q_quality_var.long_name = 'data quality label for discharge'
    q_quality_var.comment = 'Original Data.Quality from Flux.csv'
    q_quality_var[:] = np.array(q_quality, dtype='object')

    ssc_quality_var = ds.createVariable('SSC_quality', str, ('time',))
    ssc_quality_var.long_name = 'data quality label for SSC'
    ssc_quality_var.comment = 'Original Data.Quality from Water.csv'
    ssc_quality_var[:] = np.array(ssc_quality, dtype='object')

    # 元数据
    # ds.latitude = parse_float(station_row.get('Latitude', -9999.0))
    # ds.longitude = parse_float(station_row.get('Longitude', -9999.0))
    ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
    ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))

    ds.Conventions = 'CF-1.8'
    ds.title = f'GFQA Daily Observed Sediment and Discharge Data for Station {station_id}'
    ds.comment = (
        'Includes both automatic QC flags and original Data.Quality labels. '
        'Flags: 0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data. '
        'Data.Quality is a text label from the GEMS/Water CSV source.'
    )
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by gfqa_to_netcdf_daily_dualqc.py'
    ds.close()
    print(f"✅ Created file: {filename}")
    n = len(discharge)
    skipped_log_iqr = (n < 5)
    skipped_ssc_q = (n < 5)

    def _repr(v, f):
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int8)
        ok = np.isfinite(v) & (v > 0)
        ok_good = ok & (f == 0)
        if np.any(ok_good):
            return float(np.nanmedian(v[ok_good])), 0
        if np.any(ok):
            return float(np.nanmedian(v[ok])), int(np.min(f[ok]))
        return float("nan"), 9

    qv, qf = _repr(discharge, q_flag)
    sscv, sscf = _repr(ssc, ssc_flag)
    sslv, sslf = _repr(ssl, ssl_flag)

    log_station_qc(
        station_id=station_id,
        station_name=str(station_row.get('Station Name', station_id)),
        n_samples=n,
        skipped_log_iqr=skipped_log_iqr,
        skipped_ssc_q=skipped_ssc_q,
        q_value=qv, q_flag=qf,
        ssc_value=sscv, ssc_flag=sscf,
        ssl_value=sslv, ssl_flag=sslf,
        created_path=filepath
    )



# ==========================================================
# main processing function
# ==========================================================

def process_all_stations(flux_df, water_df, station_df, output_dir):
    all_records = []

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
        
        #apply quality flags
        merged['Q_flag'] = merged['Clean_Value_Q'].apply(
            lambda x: apply_quality_flag(x, "Q")
        )

        merged['SSC_flag'] = merged['Clean_Value_SSC'].apply(
            lambda x: apply_quality_flag(x, "SSC")
        )

        merged['SSL_flag'] = merged['SSL'].apply(
            lambda x: apply_quality_flag(x, "SSL")
        )

        # === SSC-Q check ===
        lower, upper = compute_log_iqr_bounds(merged['SSL'].values)

        if lower is not None:
            is_outlier = (
                (merged['SSL'] < lower) |
                (merged['SSL'] > upper)
            ) & (merged['SSL_flag'] == 0)

            merged.loc[is_outlier, 'SSL_flag'] = 2  # suspect

        ssc_q_bounds = build_ssc_q_envelope(
        Q_m3s=merged['Clean_Value_Q'].values,
        SSC_mgL=merged['Clean_Value_SSC'].values,
        k=1.5,
        min_samples=5
        )

        for i, row in merged.iterrows():
            is_bad, _ = check_ssc_q_consistency(
                Q=row['Clean_Value_Q'],
                SSC=row['Clean_Value_SSC'],
                Q_flag=row['Q_flag'],
                SSC_flag=row['SSC_flag'],
                ssc_q_bounds=ssc_q_bounds
            )
            if is_bad and row['SSC_flag'] == 0:
                merged.at[i, 'SSC_flag'] = np.int8(2)

                merged.at[i, 'SSL_flag'] = propagate_ssc_q_inconsistency_to_ssl(
                    inconsistent=True,
                    Q=row['Clean_Value_Q'],
                    SSC=row['Clean_Value_SSC'],
                    SSL=merged.at[i, 'SSL'],
                    Q_flag=row['Q_flag'],
                    SSC_flag=merged.at[i, 'SSC_flag'],  
                    SSL_flag=row['SSL_flag'],
                    ssl_is_derived_from_q_ssc=True,  
                )

            
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

        create_netcdf_file(
            station_id, station_row,
            pd.to_datetime(merged['Date']).dt.to_pydatetime(),
            merged['Clean_Value_Q'].to_numpy(),
            merged['Clean_Value_SSC'].to_numpy(),
            merged['SSL'].to_numpy(),
            merged['Q_flag'].to_numpy().astype(np.byte),
            merged['SSC_flag'].to_numpy().astype(np.byte),
            merged['SSL_flag'].to_numpy().astype(np.byte),
            merged['Quality_Q'].fillna('unknown').to_numpy(),
            merged['Quality_SSC'].fillna('unknown').to_numpy(),
            output_dir
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



def main():
    print("=" * 60)
    print("GFQA Observed Daily Data → NetCDF Conversion with Dual QC")
    print("=" * 60)

    flux_df, water_df, station_df = read_csv_files()
    
    output_dir = str(PROJECT_ROOT / "Output_r" / "daily" / "GFQA_v2" / "qc")
    process_all_stations(flux_df, water_df, station_df, output_dir=output_dir)
    print("\nConversion complete with Data.Quality and QC Flags!")


if __name__ == '__main__':
    main()

