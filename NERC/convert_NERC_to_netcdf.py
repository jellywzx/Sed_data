#!/usr/bin/env python3
"""
Script to convert NERC Hampshire Avon dataset to CF-1.8 compliant NetCDF format.

This script processes daily discharge and water chemistry data from four tributaries
of the Hampshire Avon (Sem, Nadder, West Avon, Ebble) and converts them to standardized
NetCDF files with quality control flags.

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
import sys
import argparse
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

# project root 
PROJECT_ROOT = os.path.abspath(os.path.join(PARENT_DIR, '..'))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, 'Source', 'NERC', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Output_r', 'daily', 'NERC', 'qc')


# Station metadata based on NERC documentation
# Reference: Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage
# and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble)
STATION_METADATA = {
    'AS': {
        'station_name': 'Ashley',
        'river_name': 'Sem',
        'Source_ID': 'NERC_AS',
        'latitude': 51.0,  # Approximate, need to verify from documentation
        'longitude': -1.8,  # Approximate, need to verify from documentation
        'altitude': -9999.0,  # Not provided in data files
        'upstream_area': -9999.0  # Not provided in data files
    },
    'CE': {
        'station_name': 'Cerne Abbas',
        'river_name': 'Nadder',
        'Source_ID': 'NERC_CE',
        'latitude': 50.9,  # Approximate, need to verify from documentation
        'longitude': -1.9,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GA': {
        'station_name': 'Gale',
        'river_name': 'West Avon',
        'Source_ID': 'NERC_GA',
        'latitude': 51.1,  # Approximate, need to verify from documentation
        'longitude': -1.7,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GN': {
        'station_name': 'Green Lane',
        'river_name': 'Ebble',
        'Source_ID': 'NERC_GN',
        'latitude': 51.0,  # Approximate, need to verify from documentation
        'longitude': -2.0,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    }
}

def parse_date(date_str):
    """Parse date string in DD/MM/YYYY format."""
    try:
        return pd.to_datetime(date_str, format='%d/%m/%Y')
    except:
        return pd.NaT

def apply_tool_qc(time, Q, SSC, SSL, station_id, station_name, plot_dir=None):
    """
    Apply QC using tool.py end-to-end pipeline WITH step-level provenance flags.
    Robust to scalar / mismatched shapes.
    Returns:
        qc (dict): trimmed arrays + final flags + step flags
        qc_report (dict): station-level summary counters (FLAT fields for CSV)
    """

    # --- force strict 1D(time) ---
    time = np.atleast_1d(np.asarray(time)).reshape(-1)
    Q    = np.atleast_1d(np.asarray(Q,   dtype=float)).reshape(-1)
    SSC  = np.atleast_1d(np.asarray(SSC, dtype=float)).reshape(-1)
    SSL  = np.atleast_1d(np.asarray(SSL, dtype=float)).reshape(-1)

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None, None
    time, Q, SSC, SSL = time[:n], Q[:n], SSC[:n], SSL[:n]

    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        qc2_k=1.5, qc2_min_samples=5,
        qc3_k=1.5, qc3_min_samples=5,
    )
    if qc is None:
        return None, None

    # ---- value-based present (更稳) ----
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

    # trim all 1D arrays (incl. step flags)
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and qc[k].shape[0] == valid_time.shape[0]:
            qc[k] = qc[k][valid_time]

    # ✅ IMPORTANT: remove dict item before pd.DataFrame(qc)
    ssc_q_bounds = qc.pop("ssc_q_bounds", None)

    # ---- helper counts ----
    def _cnt(arr, mapping):
        arr = np.asarray(arr, dtype=np.int8)
        return {k: int(np.sum(arr == np.int8(v))) for k, v in mapping.items()}

    # final: 0 good,1 est,2 suspect,3 bad,9 missing
    final_map = {"good":0, "estimated":1, "suspect":2, "bad":3, "missing":9}
    # qc1: 0 pass,3 bad,9 missing
    qc1_map   = {"pass":0, "bad":3, "missing":9}
    # qc2: 0 pass,2 suspect,8 not_checked,9 missing
    qc2_map   = {"pass":0, "suspect":2, "not_checked":8, "missing":9}
    # qc3 SSC-Q: 0 pass,2 suspect,8 not_checked,9 missing
    qc3_map   = {"pass":0, "suspect":2, "not_checked":8, "missing":9}
    # qc3 SSL propagate: 0 not_propagated,1 propagated,8 not_checked,9 missing
    ssl3_map  = {"not_propagated":0, "propagated":1, "not_checked":8, "missing":9}

    qf   = _cnt(qc["Q_flag"],   final_map)
    sscf = _cnt(qc["SSC_flag"], final_map)
    sslf = _cnt(qc["SSL_flag"], final_map)

    qc_report = {
        # 让 generate_qc_results_csv 识别的扁平字段（非常关键）:contentReference[oaicite:4]{index=4}
        "QC_n_days": int(qc["time"].shape[0]),

        "Q_final_good": qf["good"],
        "Q_final_estimated": qf["estimated"],
        "Q_final_suspect": qf["suspect"],
        "Q_final_bad": qf["bad"],
        "Q_final_missing": qf["missing"],

        "SSC_final_good": sscf["good"],
        "SSC_final_estimated": sscf["estimated"],
        "SSC_final_suspect": sscf["suspect"],
        "SSC_final_bad": sscf["bad"],
        "SSC_final_missing": sscf["missing"],

        "SSL_final_good": sslf["good"],
        "SSL_final_estimated": sslf["estimated"],
        "SSL_final_suspect": sslf["suspect"],
        "SSL_final_bad": sslf["bad"],
        "SSL_final_missing": sslf["missing"],
    }

    # step flags（如果存在就统计；不存在就跳过）
    def _maybe_step(key, mapping, out_prefix, names):
        if key in qc:
            c = _cnt(qc[key], mapping)
            for n in names:
                qc_report[f"{out_prefix}_{n}"] = c[n]

    _maybe_step("Q_flag_qc1_physical",   qc1_map,  "Q_qc1",   ["pass","bad","missing"])
    _maybe_step("SSC_flag_qc1_physical", qc1_map,  "SSC_qc1", ["pass","bad","missing"])
    _maybe_step("SSL_flag_qc1_physical", qc1_map,  "SSL_qc1", ["pass","bad","missing"])

    _maybe_step("Q_flag_qc2_log_iqr",    qc2_map,  "Q_qc2",   ["pass","suspect","not_checked","missing"])
    _maybe_step("SSC_flag_qc2_log_iqr",  qc2_map,  "SSC_qc2", ["pass","suspect","not_checked","missing"])
    _maybe_step("SSL_flag_qc2_log_iqr",  qc2_map,  "SSL_qc2", ["pass","suspect","not_checked","missing"])

    _maybe_step("SSC_flag_qc3_ssc_q",        qc3_map,  "SSC_qc3", ["pass","suspect","not_checked","missing"])
    _maybe_step("SSL_flag_qc3_from_ssc_q",   ssl3_map, "SSL_qc3", ["not_propagated","propagated","not_checked","missing"])

    # 如果你还要用 bounds 画图：这里你可以把 ssc_q_bounds 传去 plot 函数
    # （不要放回 qc，否则又会触发 DataFrame 问题）
    return qc, qc_report



def calculate_SSL(Q, SSC):
    """
    Calculate Suspended Sediment Load (SSL) from discharge and concentration.

    Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864

    Derivation:
    Step 1: Convert Q from m³/s to L/day
        Q (L/day) = Q (m³/s) × 86400 (s/day) × 1000 (L/m³)
        Q (L/day) = Q × 86,400,000

    Step 2: Calculate mass per day
        Mass (mg/day) = Q (L/day) × SSC (mg/L)
        Mass (mg/day) = Q × 86,400,000 × SSC

    Step 3: Convert from mg to ton
        SSL (ton/day) = Mass (mg/day) / 10⁶ (mg/ton)
        SSL (ton/day) = Q × 86,400,000 × SSC / 10⁶
        SSL (ton/day) = Q × SSC × 86.4

    Wait, this gives 86.4, but let me verify with an example:
    Q = 1 m³/s, SSC = 1000 mg/L
    - Volume per day: 1 m³/s × 86400 s = 86400 m³ = 86,400,000 L
    - Mass per day: 86,400,000 L × 1000 mg/L = 86,400,000,000 mg = 86,400 kg = 86.4 ton

    So SSL = 1 × 1000 × 0.0864 = 86.4 ton/day ✓

    Therefore: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864

    Parameters:
    -----------
    Q : float or array
        Discharge in m³/s
    SSC : float or array
        Suspended sediment concentration in mg/L

    Returns:
    --------
    SSL : float or array
        Suspended sediment load in ton/day
    """
    if pd.isna(Q) or pd.isna(SSC):
        return np.nan
    return Q * SSC * 0.0864

def process_station(station_code, data_dir='data', output_dir='Output'):
    """
    Process data for a single station and create CF-1.8 compliant NetCDF file.

    Parameters:
    -----------
    station_code : str
        Station code (AS, CE, GA, GN)
    data_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory for output NetCDF files
    """

    print(f"\nProcessing station: {station_code}")

    # Get station metadata
    metadata = STATION_METADATA[station_code]

    # Read discharge data
    discharge_file = f"{data_dir}/{station_code}_Discharge_data.csv"
    print(f"  Reading {discharge_file}...")
    df_q = pd.read_csv(discharge_file)

    # Parse dates and discharge
    df_q['date'] = df_q['Date'].apply(parse_date)
    df_q['Q'] = pd.to_numeric(df_q.iloc[:, 1], errors='coerce')

    # Read chemistry data (includes SSC)
    chemistry_file = f"{data_dir}/{station_code}_surfacewater_chemistry.csv"
    print(f"  Reading {chemistry_file}...")
    df_chem = pd.read_csv(chemistry_file)

    # Parse dates and SSC
    df_chem['date'] = df_chem['Date'].apply(parse_date)
    df_chem['SSC'] = pd.to_numeric(df_chem['SSC (mg L-1)'], errors='coerce')

    # Merge discharge and chemistry data
    print("  Merging discharge and chemistry data...")
    df = pd.merge(df_q[['date', 'Q']], df_chem[['date', 'SSC']],
                  on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)

    # Remove rows with missing dates
    df = df.dropna(subset=['date'])

    # Calculate SSL
    print("  Calculating SSL...")
    df['SSL'] = df.apply(lambda row: calculate_SSL(row['Q'], row['SSC']), axis=1)

    # Apply quality checks
    print("  Applying quality control checks...")
    print("  Applying tool.py quality control...")

    qc, qc_report = apply_tool_qc(
        time=df['date'].values,
        Q=df['Q'].values,
        SSC=df['SSC'].values,
        SSL=df['SSL'].values,
        station_id=metadata['Source_ID'],
        station_name=metadata['station_name'],
        plot_dir=os.path.join(output_dir, "diagnostic_plots"),
    )

    if qc is None:
        warnings.warn(f"No valid data after QC for station {station_code}. Skipping.")
        return None, None, None, None, None
    # -------- print QC summary (station-level) --------
    n_raw = len(df)  # 注意：这里的 df 还是 merge 后的原始表（还没被 df = DataFrame(qc) 覆盖）
    skipped_log_iqr = (n_raw < 5)
    skipped_ssc_q = (n_raw < 5)

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

    qv, qf = _repr(qc["Q"], qc["Q_flag"])
    sscv, sscf = _repr(qc["SSC"], qc["SSC_flag"])
    sslv, sslf = _repr(qc["SSL"], qc["SSL_flag"])

    print(f"  ▣ [{metadata['Source_ID']}] Sample size = {n_raw}"
          + (", log-IQR statistical QC skipped." if skipped_log_iqr else ""))
    print(f"  ▣ [{metadata['Source_ID']}] Sample size = {n_raw}"
          + (", SSC–Q consistency check and diagnostic plot skipped." if skipped_ssc_q else ""))
    print(f"  ✓ QC summary for {metadata['Source_ID']}:")
    print(f"    Q:   {qv:.2f} m3/s (flag={qf})")
    print(f"    SSC: {sscv:.2f} mg/L (flag={sscf})")
    print(f"    SSL: {sslv:.2f} ton/day (flag={sslf})")

    ssc_q_bounds = qc.pop("ssc_q_bounds", None)  # optional: keep for plotting/debug

    qc_for_df = {}
    for k, v in qc.items():
        if isinstance(v, np.ndarray):
            vv = np.asarray(v).squeeze()
            if vv.ndim == 1:
                qc_for_df[k] = vv

    df = pd.DataFrame(qc_for_df)
    # also pad step/provenance flags (NaN -> 9)
    for col in df.columns:
        if col.endswith("_flag") or ("flag_qc" in col):
            df[col] = df[col].fillna(9).astype(np.int8)
    # Convert dates to days since 1970-01-01
    reference_date = datetime(1970, 1, 1)
    # df['time'] = (df['time'] - pd.Timestamp(reference_date)).dt.total_seconds() / 86400
    df["date"] = pd.to_datetime(df["time"])
    df["time"] = (df["date"] - pd.Timestamp(reference_date)).dt.total_seconds() / 86400.0
    # Get temporal span
    valid_dates = df['date'].dropna()
    if len(valid_dates) > 0:
        start_date = valid_dates.min().strftime('%Y-%m-%d')
        end_date = valid_dates.max().strftime('%Y-%m-%d')
        temporal_span = f"{valid_dates.min().year}-{valid_dates.max().year}"
    else:
        start_date = "N/A"
        end_date = "N/A"
        temporal_span = "N/A"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create NetCDF file
    output_file = f"{output_dir}/NERC_{station_code}.nc"
    print(f"  Creating NetCDF file: {output_file}...")

    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:

        # Create dimensions
        time_dim = ncfile.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        lat_var = ncfile.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')

        lon_var = ncfile.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')

        # Create scalar variables
        alt_var = ncfile.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var.comment = 'Source: Not provided in original dataset.'

        area_var = ncfile.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Source: Not provided in original dataset.'

        # Create data variables
        q_var = ncfile.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'time lat lon'
        q_var.ancillary_variables = "Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr"
        q_var.comment = 'Source: Original data provided by Heppell & Binley (2016). Daily average discharge measurements.'

        q_flag_var = ncfile.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        ssc_var = ncfile.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time lat lon'
        ssc_var.ancillary_variables = 'ssc_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'
        ssc_var.comment = 'Source: Original data provided by Heppell & Binley (2016). Laboratory measurements from water samples.'

        ssc_flag_var = ncfile.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        ssl_var = ncfile.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon'
        ssl_var.ancillary_variables = 'ssl_flag SSL_flag_qc1_physical SSL_flag_qc3_from_ssc_q'
        ssl_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864, where 0.0864 = 86400 s/day × 1000 L/m³ / 10⁶ mg/ton.'

        ssl_flag_var = ncfile.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        # Write data
        time_var[:] = df['time'].values
        lat_var[:] = metadata['latitude']
        lon_var[:] = metadata['longitude']
        alt_var[:] = metadata['altitude']
        area_var[:] = metadata['upstream_area']

        # Replace NaN with fill value
        q_data = df['Q'].fillna(-9999.0).values
        ssc_data = df['SSC'].fillna(-9999.0).values
        ssl_data = df['SSL'].fillna(-9999.0).values

        q_var[:] = q_data
        ssc_var[:] = ssc_data
        ssl_var[:] = ssl_data

        q_flag_var[:] = df['Q_flag'].values.astype('i1')
        ssc_flag_var[:] = df['SSC_flag'].values.astype('i1')
        ssl_flag_var[:] = df['SSL_flag'].values.astype('i1')

        # -------------------------------------------------
        # Step-level provenance/QC flags (write into NetCDF)
        # -------------------------------------------------
        def _add_step_flag(name, values, *, flag_values, flag_meanings, long_name):
            v = ncfile.createVariable(name, "i1", ("time",), fill_value=9)
            v.long_name = long_name
            v.standard_name = "status_flag"
            v.flag_values = np.array(flag_values, dtype=np.int8)
            v.flag_meanings = flag_meanings
            v.missing_value = np.int8(9)
            v[:] = np.asarray(values, dtype=np.int8)
            return v

        # 注意：这些列名必须存在于 df.columns（来自 apply_hydro_qc_with_provenance）
        step_specs = [
            # --- Q ---
            ("Q_flag_qc1_physical", [0, 3, 9], "pass bad missing",
             "QC1 physical flag for river discharge"),
            ("Q_flag_qc2_log_iqr",  [0, 2, 8, 9], "pass suspect not_checked missing",
             "QC2 log-IQR flag for river discharge"),

            # --- SSC ---
            ("SSC_flag_qc1_physical", [0, 3, 9], "pass bad missing",
             "QC1 physical flag for suspended sediment concentration"),
            ("SSC_flag_qc2_log_iqr",  [0, 2, 8, 9], "pass suspect not_checked missing",
             "QC2 log-IQR flag for suspended sediment concentration"),
            ("SSC_flag_qc3_ssc_q",    [0, 2, 8, 9], "pass suspect not_checked missing",
             "QC3 SSC–Q consistency flag for suspended sediment concentration"),

            # --- SSL ---
            ("SSL_flag_qc1_physical",   [0, 3, 9], "pass bad missing",
             "QC1 physical flag for suspended sediment load"),
            ("SSL_flag_qc3_from_ssc_q", [0, 2, 8, 9], "not_propagated propagated not_checked missing",
             "QC3 propagated flag for SSL from SSC–Q consistency"),
        ]

        created_step_flags = []
        for col, fvals, fmeans, lname in step_specs:
            if col in df.columns:
                _add_step_flag(
                    col,
                    df[col].fillna(9).values,
                    flag_values=fvals,
                    flag_meanings=fmeans,
                    long_name=lname,
                )
                created_step_flags.append(col)

        # 可选：把 step flags 填到 ancillary_variables（更规范）
        if created_step_flags:
            q_steps   = [c for c in created_step_flags if c.startswith("Q_flag_")]
            ssc_steps = [c for c in created_step_flags if c.startswith("SSC_flag_")]
            ssl_steps = [c for c in created_step_flags if c.startswith("SSL_flag_")]

            if q_steps:
                q_var.ancillary_variables = " ".join(["Q_flag"] + q_steps)
            if ssc_steps:
                ssc_var.ancillary_variables = " ".join(["SSC_flag"] + ssc_steps)
            if ssl_steps:
                ssl_var.ancillary_variables = " ".join(["SSL_flag"] + ssl_steps)
        # -------------------------------------------------
        # Global attributes
        ncfile.Conventions = 'CF-1.8, ACDD-1.3'
        ncfile.title = 'Harmonized Global River Discharge and Sediment'
        ncfile.summary = (f'Daily river discharge and suspended sediment data for {metadata["station_name"]} '
                         f'station on the {metadata["river_name"]} tributary of the Hampshire Avon, UK. '
                         f'This dataset contains time series of discharge, suspended sediment concentration, '
                         f'and calculated sediment load with quality control flags.')
        ncfile.source = 'In-situ station data'
        ncfile.data_source_name = 'NERC Hampshire Avon Dataset'
        ncfile.station_name = metadata['station_name']
        ncfile.river_name = metadata['river_name']
        ncfile.Source_ID = metadata['Source_ID']

        ncfile.geospatial_lat_min = float(metadata['latitude'])
        ncfile.geospatial_lat_max = float(metadata['latitude'])
        ncfile.geospatial_lon_min = float(metadata['longitude'])
        ncfile.geospatial_lon_max = float(metadata['longitude'])

        if metadata['altitude'] != -9999.0:
            ncfile.geospatial_vertical_min = float(metadata['altitude'])
            ncfile.geospatial_vertical_max = float(metadata['altitude'])

        ncfile.geographic_coverage = 'Hampshire Avon Basin, Southern England, UK'
        ncfile.time_coverage_start = start_date
        ncfile.time_coverage_end = end_date
        ncfile.temporal_span = temporal_span
        ncfile.temporal_resolution = 'daily'
        ncfile.variables_provided = 'Q, SSC, SSL'
        ncfile.number_of_data = str(len(df))

        ncfile.reference = ('Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage '
                          'and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble). '
                          'NERC Environmental Information Data Centre. '
                          'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5')
        ncfile.source_data_link = 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5'

        ncfile.creator_name = 'Zhongwang Wei'
        ncfile.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ncfile.creator_institution = 'Sun Yat-sen University, China'

        # Add processing history
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ncfile.history = (f'{current_time}: Converted from CSV to CF-1.8 compliant NetCDF format. '
                         f'Applied quality control checks and calculated SSL. '
                         f'Script: convert_NERC_to_netcdf.py')

        ncfile.date_created = datetime.now().strftime('%Y-%m-%d')
        ncfile.date_modified = datetime.now().strftime('%Y-%m-%d')
        ncfile.processing_level = 'Quality controlled and standardized'

        ncfile.comment = ('Data includes daily discharge measurements and periodic suspended sediment '
                         'concentration measurements. SSL calculated from Q and SSC. Quality flags '
                         'indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.')

    print(f"  Successfully created {output_file}")

    # errors, warnings_nc = check_nc_completeness(output_file, strict=True)

    # if errors:
    #     print(f"  ✗ Completeness check FAILED for {station_code}")
    #     for e in errors:
    #         print(f"    ERROR: {e}")
    #     os.remove(output_file)
    #     return None, None, None, None

    # if warnings_nc:
    #     print(f"  ⚠ Completeness warnings for {station_code}")
    #     for w in warnings_nc:
    #         print(f"    WARNING: {w}")


    return df, metadata, start_date, end_date, qc_report

def generate_summary_csv(station_summaries, output_dir='Output'):
    """
    Generate station summary CSV + QC results CSV using tool.py unified writers.
    station_summaries: list[dict]
    """
    print("\nGenerating station summary CSV...")

    os.makedirs(output_dir, exist_ok=True)

    # 1) 站点摘要
    station_csv = os.path.join(output_dir, "NERC_station_summary.csv")
    generate_csv_summary_tool(station_summaries, station_csv)

    # 2) QC 汇总（依赖 summary 里包含 qc_report 字段）
    qc_csv = os.path.join(output_dir, "NERC_qc_results_summary.csv")
    generate_qc_results_csv_tool(station_summaries, qc_csv)


def calculate_percent_complete(df, var_name, flag_name):
    """Calculate percentage of good data for a variable."""
    if var_name not in df.columns:
        return 0.0

    # Count non-missing values
    total = len(df)
    if total == 0:
        return 0.0

    # Count good data (flag = 0)
    good_data = (df[flag_name] == 0).sum()

    return (good_data / total) * 100.0

def main():
    """Main processing function."""

    print("="*70)
    print("NERC Hampshire Avon Data Processing")
    print("CF-1.8 Compliant NetCDF Generation with Quality Control")
    print("="*70)

    # parse command-line arguments for input/output directories
    parser = argparse.ArgumentParser(description='Convert NERC CSVs to CF-1.8 NetCDF.')
    parser.add_argument('--input-dir', '-i', default=DEFAULT_INPUT_DIR,
                    help='Input data directory containing CSV files')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR,
                    help='Output directory for NetCDF files')

    args = parser.parse_args()

    print(f"Using input directory: {args.input_dir}")
    print(f"Using output directory: {args.output_dir}")

    station_codes = ['AS', 'CE', 'GA', 'GN']
    station_summaries = []

    for station_code in station_codes:
        try:
            df, metadata, start_date, end_date, qc_report  = process_station(station_code, data_dir=args.input_dir, output_dir=args.output_dir)

            # Calculate data completeness
            q_complete = calculate_percent_complete(df, 'Q', 'Q_flag')
            ssc_complete = calculate_percent_complete(df, 'SSC', 'SSC_flag')
            ssl_complete = calculate_percent_complete(df, 'SSL', 'SSL_flag')

            # Get date ranges for each variable
            q_dates = df[df['Q_flag'] == 0]['date']
            ssc_dates = df[df['SSC_flag'] == 0]['date']
            ssl_dates = df[df['SSL_flag'] == 0]['date']

            summary = {
                'station_name': metadata['station_name'],
                'Source_ID': metadata['Source_ID'],
                'river_name': metadata['river_name'],
                'longitude': metadata['longitude'],
                'latitude': metadata['latitude'],
                'altitude': metadata['altitude'] if metadata['altitude'] != -9999.0 else 'N/A',
                'upstream_area': metadata['upstream_area'] if metadata['upstream_area'] != -9999.0 else 'N/A',
                'Data Source Name': 'NERC Hampshire Avon Dataset',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{start_date.split('-')[0]}-{end_date.split('-')[0]}" if start_date != 'N/A' else 'N/A',
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Hampshire Avon Basin, Southern England, UK',
                'Reference/DOI': 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5',
                'Q_start_date': q_dates.min().year if len(q_dates) > 0 else 'N/A',
                'Q_end_date': q_dates.max().year if len(q_dates) > 0 else 'N/A',
                'Q_percent_complete': round(q_complete, 1),
                'SSC_start_date': ssc_dates.min().year if len(ssc_dates) > 0 else 'N/A',
                'SSC_end_date': ssc_dates.max().year if len(ssc_dates) > 0 else 'N/A',
                'SSC_percent_complete': round(ssc_complete, 1),
                'SSL_start_date': ssl_dates.min().year if len(ssl_dates) > 0 else 'N/A',
                'SSL_end_date': ssl_dates.max().year if len(ssl_dates) > 0 else 'N/A',
                'SSL_percent_complete': round(ssl_complete, 1)
            }
            if qc_report is not None:
                summary.update(qc_report)
            station_summaries.append(summary)

        except Exception as e:
            print(f"  Error processing station {station_code}: {str(e)}")
            continue

    # Generate summary CSV
    if station_summaries:
        generate_summary_csv(station_summaries, output_dir=args.output_dir)

    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

if __name__ == '__main__':
    main()