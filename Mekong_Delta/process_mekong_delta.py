#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs quality control, CF-1.8 standardization, and metadata
enhancement for the Mekong Delta sediment and discharge dataset.

Original data source:
Darby, S.E.; Hackney, C.R.; Parsons, D.R.; Tri, P.D.V. (2020).
Water and suspended sediment discharges for the Mekong Delta, Vietnam (2005-2015).
NERC Environmental Information Data Centre.
https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595

The script performs the following steps:
1.  Reads original wide-format CSV files for fluxes and ratings.
2.  Merges discharge (Q) and sediment load (SSL) data.
3.  Calculates suspended sediment concentration (SSC) from Q and SSL.
4.  Performs Quality Control (QC) on Q, SSC, and SSL, generating quality flags.
5.  Truncates the data to the period with valid observations.
6.  Writes CF-1.8 compliant NetCDF files for each station.
7.  Generates a summary CSV file with metadata for all stations.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import warnings
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    apply_quality_flag_array,                
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    apply_hydro_qc_with_provenance,           
    generate_csv_summary as generate_csv_summary_tool,          
    generate_qc_results_csv as generate_qc_results_csv_tool,
)


# --- CONFIGURATION ---

# Input and Output directories
# Assumes the script is run from the 'Script/Dataset/Mekong_Delta' directory
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  
SOURCE_DATA_DIR = os.path.join(BASE_DIR, "Source", "Mekong_Delta", "data")
TARGET_NC_DIR = os.path.join(BASE_DIR, "Output_r", "daily", "Mekong_Delta", "qc")
TARGET_CSV_PATH = os.path.join(BASE_DIR, "Output_r", "daily", "Mekong_Delta", "qc")


# Station metadata
STATIONS = {
    'Cantho': {
        'name': 'Can Tho', 'Source_ID': 'Cantho', 'lat': 10.088109, 'lon': 105.736458,
        'river': 'Hau River (Bassac)', 'altitude': np.nan, 'upstream_area': 550000
    },
    'Chaudoc': {
        'name': 'Chau Doc', 'Source_ID': 'Chaudoc', 'lat': 10.708268, 'lon': 105.134606,
        'river': 'Hau River (Bassac)', 'altitude': np.nan, 'upstream_area': np.nan
    },
    'Mythaun': {
        'name': 'My Thuan', 'Source_ID': 'Mythaun', 'lat': 10.272038, 'lon': 105.900920,
        'river': 'Tien River (Mekong)', 'altitude': np.nan, 'upstream_area': np.nan
    },
    'Tanchau': {
        'name': 'Tan Chau', 'Source_ID': 'Tanchau', 'lat': 10.822642, 'lon': 105.227879,
        'river': 'Tien River (Mekong)', 'altitude': np.nan, 'upstream_area': np.nan
    }
}


# --- HELPER FUNCTIONS ---

def read_fluxes_file(filepath):
    """
    Read fluxes CSV file (wide format), convert to long format, and convert
    sediment load from Mt/day to ton/day.
    """
    df = pd.read_csv(filepath)
    df_long = df.melt(id_vars=['Date'], var_name='Year', value_name='Flux_Mt_day')
    df_long['Date_str'] = df_long['Date'] + '-' + df_long['Year']
    df_long['time'] = pd.to_datetime(df_long['Date_str'], format='%d-%b-%Y', errors='coerce')
    # Conversion: 1 Mt = 1e6 tons.
    df_long['SSL'] = df_long['Flux_Mt_day'] * 1e6
    df_long = df_long.dropna(subset=['time', 'SSL'])
    return df_long[['time', 'SSL']].sort_values('time')

def read_ratings_file(filepath):
    """
    Read ratings CSV file, fix year inconsistencies, and aggregate to daily
    discharge (Q) and SSC.
    """
    df = pd.read_csv(filepath)
    # Fix year column where '10' might mean '2010'
    df.loc[df['Year'] == 10, 'Year'] = 2010
    df.loc[df['Year'] < 100, 'Year'] = df.loc[df['Year'] < 100, 'Year'] + 2000
    df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    daily = df.groupby('time').agg({
        'Discharge (m3/s)': 'mean',
        'Section Averaged SSC (mg/l)': 'mean'
    }).reset_index()
    
    daily = daily.rename(columns={
        'Discharge (m3/s)': 'Q',
        'Section Averaged SSC (mg/l)': 'SSC_original'
    })
    return daily

def calculate_ssc_from_ssl(df):
    """
    Calculate SSC from discharge (Q) and sediment load (SSL).
    Formula: SSC (mg/L) = SSL (ton/day) / [Q (m³/s) × 0.0864]
    Derivation:
      - Q (m³/s) × SSC (mg/L) -> Q (m³/s) × SSC (g/m³) = Q × SSC (g/s)
      - Convert to ton/day: Q×SSC (g/s) × 86400 (s/day) / 1e6 (g/ton) = Q×SSC×0.0864
    """
    # Suppress division by zero warnings, they are handled by np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        ssc = df['SSL'] / (df['Q'] * 0.0864)
    # Replace inf/-inf/nan with np.nan
    ssc[~np.isfinite(ssc)] = np.nan
    return ssc


def get_summary_stats(df, var_name):
    """Calculate summary statistics for a variable."""
    flag_name = f"{var_name}_flag"
    valid_data = df[df[flag_name] == 0][var_name]
    if valid_data.empty:
        return np.nan, np.nan, 0.0
    
    start_date = df[df[var_name].notna()]['time'].min()
    end_date = df[df[var_name].notna()]['time'].max()
    
    if pd.isna(start_date):
        return np.nan, np.nan, 0.0

    total_days = (end_date - start_date).days + 1
    good_data_count = len(valid_data)
    percent_complete = (good_data_count / total_days) * 100 if total_days > 0 else 0
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), round(percent_complete, 2)

def apply_tool_qc(
    time,
    Q,
    SSC,
    SSL,
    station_id,
    station_name,
    plot_dir=None,
):
    """
    Use tool.py unified hydro QC WITH provenance(step) flags.
    Returns:
        qc (dict): trimmed arrays + final flags + step flags + ssc_q_bounds
        qc_report (dict): station-level summary counters for CSV
    """

    # -----------------------------
    # 0) force 1D + align length (✅ 防止 len()/shape 异常)
    # -----------------------------
    time = np.atleast_1d(np.asarray(time)).squeeze()
    Q    = np.atleast_1d(np.asarray(Q)).squeeze()
    SSC  = np.atleast_1d(np.asarray(SSC)).squeeze()
    SSL  = np.atleast_1d(np.asarray(SSL)).squeeze()

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None, None

    time = time[:n]
    Q    = Q[:n]
    SSC  = SSC[:n]
    SSL  = SSL[:n]

    # -----------------------------
    # 1) unified QC (final + step flags)
    #    这里按你的规则：
    #      Q: independent True
    #      SSL: Mt/day->ton/day 属于单位转换 => independent True
    #      SSC: 由 SSL/(Q*0.0864) 推导 => independent False
    #      ssl_is_derived_from_q_ssc=False (因为 SSL 是原始给的)
    # -----------------------------
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=False,
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )
    if qc is None:
        return None, None

    # -----------------------------
    # 2) ✅ 更稳的 valid_time（值 + flag）
    # -----------------------------
    def _present(v, f):
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int8)
        return (
            (f != FILL_VALUE_INT)
            & np.isfinite(v)
            & (~np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5))
        )

    present_Q   = _present(qc["Q"],   qc["Q_flag"])
    present_SSC = _present(qc["SSC"], qc["SSC_flag"])
    present_SSL = _present(qc["SSL"], qc["SSL_flag"])

    valid_time = present_Q | present_SSC | present_SSL
    if not np.any(valid_time):
        return None, None

    # 裁剪所有同长度数组（包括 step flags）
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and len(qc[k]) == len(valid_time):
            qc[k] = qc[k][valid_time]

    # -----------------------------
    # 3) station qc_report（给 CSV 汇总用）
    # -----------------------------
    def _count(arr, values):
        arr = np.asarray(arr, dtype=np.int8)
        return {v: int(np.sum(arr == np.int8(v))) for v in values}

    def _pack_final(prefix, flag_arr):
        c = _count(flag_arr, [0, 1, 2, 3, 9])
        return {
            f"{prefix}_final_good": c[0],
            f"{prefix}_final_estimated": c[1],
            f"{prefix}_final_suspect": c[2],
            f"{prefix}_final_bad": c[3],
            f"{prefix}_final_missing": c[9],
        }

    # step flags codes:
    # qc1: 0 pass, 3 bad, 9 missing
    # qc2/qc3: 0 pass, 2 suspect, 8 not_checked, 9 missing
    def _pack_step(prefix, step, flag_arr, values):
        c = _count(flag_arr, values)
        out = {}
        # 映射成你 qc_results_csv 常用字段名
        if step == "qc1":
            out[f"{prefix}_qc1_pass"] = c.get(0, 0)
            out[f"{prefix}_qc1_bad"] = c.get(3, 0)
            out[f"{prefix}_qc1_missing"] = c.get(9, 0)
        elif step == "qc2":
            out[f"{prefix}_qc2_pass"] = c.get(0, 0)
            out[f"{prefix}_qc2_suspect"] = c.get(2, 0)
            out[f"{prefix}_qc2_not_checked"] = c.get(8, 0)
            out[f"{prefix}_qc2_missing"] = c.get(9, 0)
        elif step == "qc3":
            out[f"{prefix}_qc3_pass"] = c.get(0, 0)
            out[f"{prefix}_qc3_suspect"] = c.get(2, 0)
            out[f"{prefix}_qc3_not_checked"] = c.get(8, 0)
            out[f"{prefix}_qc3_missing"] = c.get(9, 0)
        return out

    qc_report = {
        "station_id": station_id,
        "station_name": station_name,
        "QC_n_days": int(len(qc["time"])),
    }
    qc_report.update(_pack_final("Q", qc["Q_flag"]))
    qc_report.update(_pack_final("SSC", qc["SSC_flag"]))
    qc_report.update(_pack_final("SSL", qc["SSL_flag"]))

    # 分步统计（如果 key 存在）
    if "Q_flag_qc1_physical" in qc:
        qc_report.update(_pack_step("Q", "qc1", qc["Q_flag_qc1_physical"], [0, 3, 9]))
    if "SSC_flag_qc1_physical" in qc:
        qc_report.update(_pack_step("SSC", "qc1", qc["SSC_flag_qc1_physical"], [0, 3, 9]))
    if "SSL_flag_qc1_physical" in qc:
        qc_report.update(_pack_step("SSL", "qc1", qc["SSL_flag_qc1_physical"], [0, 3, 9]))

    if "Q_flag_qc2_log_iqr" in qc:
        qc_report.update(_pack_step("Q", "qc2", qc["Q_flag_qc2_log_iqr"], [0, 2, 8, 9]))
    if "SSC_flag_qc2_log_iqr" in qc:
        qc_report.update(_pack_step("SSC", "qc2", qc["SSC_flag_qc2_log_iqr"], [0, 2, 8, 9]))
    if "SSL_flag_qc2_log_iqr" in qc:
        qc_report.update(_pack_step("SSL", "qc2", qc["SSL_flag_qc2_log_iqr"], [0, 2, 8, 9]))

    if "SSC_flag_qc3_ssc_q" in qc:
        qc_report.update(_pack_step("SSC", "qc3", qc["SSC_flag_qc3_ssc_q"], [0, 2, 8, 9]))
    if "SSL_flag_qc3_from_ssc_q" in qc:
        qc_report.update(_pack_step("SSL", "qc3", qc["SSL_flag_qc3_from_ssc_q"], [0, 2, 8, 9]))

    # -----------------------------
    # 4) plot
    # -----------------------------
    if plot_dir is not None and qc.get("ssc_q_bounds") is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_ssc_q_diagnostic(
            time=qc["time"],
            Q=qc["Q"],
            SSC=qc["SSC"],
            Q_flag=qc["Q_flag"],
            SSC_flag=qc["SSC_flag"],
            ssc_q_bounds=qc["ssc_q_bounds"],
            station_id=station_id,
            station_name=station_name,
            out_png=os.path.join(plot_dir, f"{station_id}_ssc_q.png"),
        )

    return qc, qc_report


def _repr_value_and_flag(values, flags):
    v = np.asarray(values, dtype=float)
    f = np.asarray(flags, dtype=np.int8)
    ok = np.isfinite(v) & (v > 0)

    ok_good = ok & (f == 0)
    if np.any(ok_good):
        return float(np.nanmedian(v[ok_good])), int(0)

    if np.any(ok):
        return float(np.nanmedian(v[ok])), int(np.min(f[ok]))
    return float("nan"), 9


def log_station_qc(
    station_id,
    station_name,
    n_samples,
    skipped_log_iqr,
    skipped_ssc_q,
    Q, Q_flag,
    SSC, SSC_flag,
    SSL, SSL_flag,
    created_path=None,
):
    print(f"\nProcessing: {station_name} ({station_id})")
    print(f"  Sample size = {n_samples} {'< 5, ' if n_samples < 5 else ''}"
          f"log-IQR {'skipped' if skipped_log_iqr else 'applied'}, "
          f"SSC-Q {'skipped' if skipped_ssc_q else 'checked'}")

    qv, qf = _repr_value_and_flag(Q, Q_flag)
    sscv, sscf = _repr_value_and_flag(SSC, SSC_flag)
    sslv, sslf = _repr_value_and_flag(SSL, SSL_flag)

    if created_path:
        print(f"✓ Created: {created_path}")
    print(f"  Q:   {qv:.2f} m3/s (flag={qf})")
    print(f"  SSC: {sscv:.2f} mg/L (flag={sscf})")
    print(f"  SSL: {sslv:.2f} ton/day (flag={sslf})")


# --- MAIN PROCESSING ---

def create_netcdf_file(filepath, df, station_meta):
    """Create a CF-1.8 compliant NetCDF file."""
    
    start_date_obj = df['time'].iloc[0]
    time_units = f"days since {start_date_obj.strftime('%Y-%m-%d')} 00:00:00"

    with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
        # === DIMENSIONS ===
        ds.createDimension('time', None)

        # === COORDINATE VARIABLES ===
        # time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = time_units
        time_var.calendar = "gregorian"
        # Avoid using `df['time'].dt.to_pydatetime()` which emits a FutureWarning
        # Build a list of Python datetimes via individual Timestamp.to_pydatetime()
        time_py = [t.to_pydatetime() for t in df['time'].tolist()]
        time_var[:] = nc.date2num(time_py, units=time_units, calendar="gregorian")
        
        # lat
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = station_meta['lat']

        # lon
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = station_meta['lon']

        # === GLOBAL ATTRIBUTES ===
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.dataset_name = "Mekong Delta (Darby et al., 2020)"
        ds.station_name = station_meta['name']
        ds.river_name = station_meta['river']
        ds.Source_ID = station_meta['Source_ID']
        ds.source_url = "https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595"
        ds.reference = "Darby, S.E.; Hackney, C.R.; Parsons, D.R.; Tri, P.D.V. (2020). Water and suspended sediment discharges for the Mekong Delta, Vietnam (2005-2015). NERC Environmental Information Data Centre."
        ds.summary = "This dataset provides in-situ daily time series of river discharge and sediment transport for the Mekong Delta, harmonized and quality-controlled."
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"
        ds.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using process_mekong_delta.py."
        ds.Type = "In-situ station data"
        ds.Temporal_Resolution = "daily"
        ds.Temporal_Span = f"{df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}"
        ds.Geographic_Coverage = "Mekong River Delta, Vietnam"
        ds.Variables_Provided = "altitude, upstream_area, Q, SSC, SSL, station_name, river_name, Source_ID"
        ds.Number_of_data = 1 # This file contains one station
        
        # === DATA VARIABLES ===
        fill_value = -9999.0

        # altitude
        alt_var = ds.createVariable('altitude', 'f4')
        alt_var.long_name = "station altitude"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.missing_value = fill_value
        if not np.isnan(station_meta['altitude']):
            alt_var[:] = station_meta['altitude']
        else:
            alt_var[:] = fill_value

        # upstream_area
        area_var = ds.createVariable('upstream_area', 'f4')
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.missing_value = fill_value
        if not np.isnan(station_meta['upstream_area']):
            area_var[:] = station_meta['upstream_area']
        else:
            area_var[:] = fill_value

        # Q
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=fill_value)
        q_var.long_name = "River Discharge"
        q_var.standard_name = "river_discharge"
        q_var.units = "m3 s-1"
        q_var.coordinates = "lat lon altitude"
        q_var.ancillary_variables = "Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr"
        q_var.comment = "Source: Original data provided by Darby et al. (2020)."
        q_var[:] = df['Q'].fillna(fill_value).values

        # SSC
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=fill_value)
        ssc_var.long_name = "Suspended Sediment Concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water_body"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q"
        ssc_var.comment = "Source: Calculated. Formula: SSC = SSL / (Q * 0.0864)."
        ssc_var[:] = df['SSC'].fillna(fill_value).values

        # SSL
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=fill_value)
        ssl_var.long_name = "Suspended Sediment Load"
        ssl_var.units = "ton day-1"
        ssl_var.coordinates = "lat lon altitude"
        ssl_var.ancillary_variables = "SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q"
        ssl_var.comment = "Source: Original data provided by Darby et al. (2020)."
        ssl_var[:] = df['SSL'].fillna(fill_value).values

        # === FLAG VARIABLES ===
        flag_fill_value = np.int8(-127)
        
        # Q_flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=flag_fill_value)
        q_flag_var.long_name = "Quality flag for River Discharge"
        q_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        q_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
        q_flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source."
        q_flag_var[:] = df['Q_flag'].fillna(flag_fill_value).values

        # SSC_flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=flag_fill_value)
        ssc_flag_var.long_name = "Quality flag for Suspended Sediment Concentration"
        ssc_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        ssc_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., extreme), 3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = df['SSC_flag'].fillna(flag_fill_value).values

        # SSL_flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=flag_fill_value)
        ssl_flag_var.long_name = "Quality flag for Suspended Sediment Load"
        ssl_flag_var.flag_values = np.array([0, 3, 9], dtype='b')
        ssl_flag_var.flag_meanings = "good_data bad_data missing_data"
        ssl_flag_var.comment = "Flag definitions: 0=Good, 3=Bad (e.g., negative), 9=Missing in source."
        ssl_flag_var[:] = df['SSL_flag'].fillna(flag_fill_value).values
        # === STEP/PROVENANCE FLAG VARIABLES ===
        def _add_step_flag(name, values, *, flag_values, flag_meanings, long_name):
            v = ds.createVariable(name, 'b', ('time',), fill_value=flag_fill_value)
            v.long_name = long_name
            v.standard_name = 'status_flag'
            v.flag_values = np.array(flag_values, dtype='b')
            v.flag_meanings = flag_meanings
            v.missing_value = np.int8(flag_fill_value)
            v[:] = np.asarray(values, dtype=np.int8)
            return v

        # QC1 physical: 0 pass, 3 bad, 9 missing
        if 'Q_flag_qc1_physical' in df.columns:
            _add_step_flag(
                "Q_flag_qc1_physical", df['Q_flag_qc1_physical'].fillna(9).values,
                flag_values=[0, 3, 9],
                flag_meanings="pass bad missing",
                long_name="QC1 physical flag for river discharge"
            )
        if 'SSC_flag_qc1_physical' in df.columns:
            _add_step_flag(
                "SSC_flag_qc1_physical", df['SSC_flag_qc1_physical'].fillna(9).values,
                flag_values=[0, 3, 9],
                flag_meanings="pass bad missing",
                long_name="QC1 physical flag for suspended sediment concentration"
            )
        if 'SSL_flag_qc1_physical' in df.columns:
            _add_step_flag(
                "SSL_flag_qc1_physical", df['SSL_flag_qc1_physical'].fillna(9).values,
                flag_values=[0, 3, 9],
                flag_meanings="pass bad missing",
                long_name="QC1 physical flag for suspended sediment load"
            )

        # QC2 log-IQR: 0 pass, 2 suspect, 8 not_checked, 9 missing
        if 'Q_flag_qc2_log_iqr' in df.columns:
            _add_step_flag(
                "Q_flag_qc2_log_iqr", df['Q_flag_qc2_log_iqr'].fillna(9).values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC2 log-IQR flag for river discharge"
            )
        if 'SSC_flag_qc2_log_iqr' in df.columns:
            _add_step_flag(
                "SSC_flag_qc2_log_iqr", df['SSC_flag_qc2_log_iqr'].fillna(9).values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC2 log-IQR flag for suspended sediment concentration"
            )
        if 'SSL_flag_qc2_log_iqr' in df.columns:
            _add_step_flag(
                "SSL_flag_qc2_log_iqr", df['SSL_flag_qc2_log_iqr'].fillna(9).values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC2 log-IQR flag for suspended sediment load"
            )

        # QC3 SSC–Q: SSC_flag_qc3_ssc_q, SSL_flag_qc3_from_ssc_q
        if 'SSC_flag_qc3_ssc_q' in df.columns:
            _add_step_flag(
                "SSC_flag_qc3_ssc_q", df['SSC_flag_qc3_ssc_q'].fillna(9).values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC3 SSC–Q consistency flag for suspended sediment concentration"
            )
        if 'SSL_flag_qc3_from_ssc_q' in df.columns:
            _add_step_flag(
                "SSL_flag_qc3_from_ssc_q", df['SSL_flag_qc3_from_ssc_q'].fillna(9).values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC3 propagated SSC–Q flag for suspended sediment load"
            )



def main():
    """Main function to process all stations."""
    print("Starting Mekong Delta dataset processing...")
    os.makedirs(TARGET_NC_DIR, exist_ok=True)
    
    station_summaries = []

    for station_id, station_meta in STATIONS.items():
        try:
            # Read and merge data
            fluxes_df = read_fluxes_file(os.path.join(SOURCE_DATA_DIR, f'{station_id}fluxes.csv'))
            ratings_df = read_ratings_file(os.path.join(SOURCE_DATA_DIR, f'{station_id}ratings.csv'))
            
            # Use an outer merge to keep all data points
            merged_df = pd.merge(ratings_df, fluxes_df, on='time', how='outer')
            
            # Calculate SSC
            merged_df['SSC'] = calculate_ssc_from_ssl(merged_df)
            
            # Apply QC
            qc, qc_report = apply_tool_qc(
                time=merged_df['time'].values,
                Q=merged_df['Q'].values,
                SSC=merged_df['SSC'].values,
                SSL=merged_df['SSL'].values,
                station_id=station_meta['Source_ID'],
                station_name=station_meta['name'],
                plot_dir=os.path.join(TARGET_NC_DIR, "diagnostic_plots"),
            )

            if qc is None:
                warnings.warn(f"No valid data after QC for station {station_id}. Skipping.")
                continue

            qc_df = pd.DataFrame(qc)


            # Truncate time series
            valid_df = qc_df.dropna(subset=['Q', 'SSC', 'SSL'], how='all')
            if valid_df.empty:
                warnings.warn(f"No valid data for station {station_id} after QC. Skipping.")
                continue
            
            start_year = valid_df['time'].min().year
            end_year = valid_df['time'].max().year
            
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            final_df = pd.DataFrame({'time': date_range})
            final_df = pd.merge(final_df, qc_df, on='time', how='left')
            
            # Re-apply QC flags for the padded time series
            final_df.loc[final_df['Q'].isna(), 'Q_flag'] = 9
            final_df.loc[final_df['SSC'].isna(), 'SSC_flag'] = 9
            final_df.loc[final_df['SSL'].isna(), 'SSL_flag'] = 9
            # --- also pad step/provenance flags (NaN -> 9) ---
            for col in final_df.columns:
                if col.endswith("_flag") or ("flag_qc" in col):
                    # for padded empty days: treat as missing
                    final_df[col] = final_df[col].fillna(9).astype(np.int8)


            # Generate summary stats before creating file
            q_start, q_end, q_perc = get_summary_stats(final_df, 'Q')
            ssc_start, ssc_end, ssc_perc = get_summary_stats(final_df, 'SSC')
            ssl_start, ssl_end, ssl_perc = get_summary_stats(final_df, 'SSL')

            summary = {
                'Source_ID': station_meta['Source_ID'],
                'station_name': station_meta['name'],
                'river_name': station_meta['river'],
                'longitude': station_meta['lon'],
                'latitude': station_meta['lat'],
                'altitude': station_meta['altitude'],
                'upstream_area': station_meta['upstream_area'],
                'Q_start_date': q_start, 'Q_end_date': q_end, 'Q_percent_complete': q_perc,
                'SSC_start_date': ssc_start, 'SSC_end_date': ssc_end, 'SSC_percent_complete': ssc_perc,
                'SSL_start_date': ssl_start, 'SSL_end_date': ssl_end, 'SSL_percent_complete': ssl_perc,
                'Data Source Name': 'Mekong Delta (Darby et al., 2020)',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{final_df['time'].min().strftime('%Y-%m-%d')} to {final_df['time'].max().strftime('%Y-%m-%d')}",
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Mekong River Delta, Vietnam',
                'Reference/DOI': 'https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595'
            }
            summary.update(qc_report)
            station_summaries.append(summary)

            # Create NetCDF file
            output_filename = f"Mekong_Delta_{station_id}.nc"
            output_filepath = os.path.join(TARGET_NC_DIR, output_filename)
            create_netcdf_file(output_filepath, final_df, station_meta)
            # --- QC printing ---
            n_samples = len(qc_df)
            skipped_log_iqr = (n_samples < 5) or (compute_log_iqr_bounds(qc_df["Q"].values)[0] is None)
            skipped_ssc_q = (n_samples < 5) or (build_ssc_q_envelope(qc_df["Q"].values, qc_df["SSC"].values) is None)

            log_station_qc(
                station_id=station_meta["Source_ID"],
                station_name=station_meta["name"],
                n_samples=n_samples,
                skipped_log_iqr=skipped_log_iqr,
                skipped_ssc_q=skipped_ssc_q,
                Q=qc_df["Q"].values, Q_flag=qc_df["Q_flag"].values,
                SSC=qc_df["SSC"].values, SSC_flag=qc_df["SSC_flag"].values,
                SSL=qc_df["SSL"].values, SSL_flag=qc_df["SSL_flag"].values,
                created_path=output_filepath,
            )


            # errors, warnings_nc = check_nc_completeness(output_filepath, strict=True)

            # if errors:
            #     print(f"✗ Completeness check failed for {station_id}")
            #     for e in errors:
            #         print(f"  ERROR: {e}")
            #     os.remove(output_filepath)
            #     continue

            # if warnings_nc:
            #     print(f"⚠ Completeness warnings for {station_id}")
            #     for w in warnings_nc:
            #         print(f"  WARNING: {w}")

            # print(f"  Successfully created {output_filepath}")

        except FileNotFoundError as e:
            warnings.warn(f"Data file not found for station {station_id}: {e}. Skipping.")
        except Exception as e:
            warnings.warn(f"An error occurred while processing station {station_id}: {e}")
            

    # Generate summary CSV
    if station_summaries:
        csv_filepath = os.path.join(TARGET_CSV_PATH, "Mekong_Delta_station_summary.csv")
        qc_csv_path  = os.path.join(TARGET_CSV_PATH, "Mekong_Delta_qc_results.csv")
        generate_csv_summary_tool(station_summaries, csv_filepath)
        generate_qc_results_csv_tool(station_summaries, qc_csv_path)
        print(f"Successfully created summary file: {csv_filepath}")
        print(f"Successfully created QC results  : {qc_csv_path}")
    print("Processing complete.")

if __name__ == "__main__":
    main()
