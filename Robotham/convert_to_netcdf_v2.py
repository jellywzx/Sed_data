#!/usr/bin/env python3
"""
Convert Robotham et al. (2022) CSV data to a harmonized, CF-1.8 compliant NetCDF format.

This script performs the following steps:
1.  Reads raw 5-minute resolution CSV data.
2.  Truncates the data to the full temporal span of available measurements.
3.  Averages the data to a daily resolution.
4.  Applies quality control checks and assigns flags for discharge (Q) and
    suspended sediment concentration (SSC).
5.  Converts units to a standard format (Q: m³/s, SSC: mg/L, SSL: ton/day).
6.  Calculates suspended sediment load (SSL).
7.  Writes the processed data to a NetCDF file with comprehensive, CF-1.8 compliant metadata.
8.  Generates a summary CSV file containing key metadata for all processed stations.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
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
    apply_quality_flag_array,            
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl,
    apply_hydro_qc_with_provenance,   
    generate_csv_summary as generate_csv_summary_tool,        
    generate_qc_results_csv as generate_qc_results_csv_tool,   
)



# --- Configuration ---

# WSL format absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..','..'))
DEFAULT_SOURCE_DIR = os.path.join(PROJECT_ROOT, 'Source', 'Robotham', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Output_r', 'daily', 'Robotham')

# Station metadata
STATIONS = {
    'lsb_the_heath': {
        'Source_ID': 'The_Heath',
        'station_name': 'The Heath',
        'river_name': 'Littlestock Brook',
        'latitude': 51.865283,
        'longitude': -1.6180989,
        'altitude': np.nan,
        'upstream_area': np.nan
    },
    'lsb_upstream_the_heath': {
        'Source_ID': 'Upstream_The_Heath',
        'station_name': 'Upstream The Heath',
        'river_name': 'Littlestock Brook',
        'latitude': 51.868396,
        'longitude': -1.6316682,
        'altitude': np.nan,
        'upstream_area': np.nan
    },
    'lsb_church_meadow': {
        'Source_ID': 'Church_Meadow',
        'station_name': 'Church Meadow',
        'river_name': 'Littlestock Brook',
        'latitude': 51.864193,
        'longitude': -1.6187105,
        'altitude': np.nan,
        'upstream_area': np.nan
    }
}

# --- Constants ---
FILL_VALUE = -9999.0
REFERENCE = ("Robotham, J., Old, G., Rameshwaran, P., Trill, E., Bishop, J. (2022). "
             "High-resolution time series of turbidity, suspended sediment concentration, "
             "total phosphorus concentration, and discharge in the Littlestock Brook, England, "
             "2017-2021. NERC EDS Environmental Information Data Centre. (Dataset). "
             "https://doi.org/10.5285/9f80e349-0594-4ae1-bff3-b055638569f8")
DATA_SOURCE_NAME = "Robotham et al. (2022)"
GEOGRAPHIC_COVERAGE = "Littlestock Brook, England"

# --- Helper Functions ---

def _qc_to_dataframe(qc: dict):
    """
    Robustly convert qc dict -> DataFrame.
    Keep ONLY 1D time-series-like arrays with the same length.
    Drop nested dicts / scalars / mismatched-length arrays.
    """
    def _to_1d(x):
        a = np.asarray(x)
        return np.atleast_1d(a).reshape(-1)

    # 1) determine n_time
    if "time" in qc:
        t = _to_1d(qc["time"])
        n_time = t.shape[0]
    else:
        # fallback: pick the first array-like length as n_time
        n_time = None
        for v in qc.values():
            if isinstance(v, dict):
                continue
            a = np.asarray(v)
            if a.ndim >= 1 and a.size > 0:
                n_time = np.atleast_1d(a).reshape(-1).shape[0]
                break
        if n_time is None:
            return pd.DataFrame()

    # 2) collect columns with matching length
    cols = {}
    for k, v in qc.items():
        if isinstance(v, dict):
            continue  # e.g., envelope/bounds/report
        try:
            a = _to_1d(v)
        except Exception:
            continue
        if a.shape[0] == n_time:
            cols[k] = a

    return pd.DataFrame(cols)

def read_and_prepare_data(csv_path):
    """Reads and prepares the raw CSV data."""
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path, parse_dates=['Timestamp'], low_memory=False)
        df.set_index('Timestamp', inplace=True)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return None

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[['SSC', 'Q']].apply(pd.to_numeric, errors='coerce')

    if df.index.duplicated().any():
        print(f"  Found {df.index.duplicated().sum()} duplicate timestamps, averaging values.")
        df = df.groupby(df.index).mean(numeric_only=True)

    return df

def truncate_time_range(df):
    """Truncates data to the first month of the first year and the last month of the last year."""
    valid_data = df.dropna(subset=['Q', 'SSC'], how='all')
    if valid_data.empty:
        print("  No valid data found for Q or SSC.")
        return None

    start_year = valid_data.index.min().year
    end_year = valid_data.index.max().year

    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')

    print(f"  Truncating data from {start_date.date()} to {end_date.date()}")
    return df.loc[start_date:end_date]

def apply_tool_qc(time, Q, SSC, SSL, station_id, station_name, plot_dir=None):
    """
    Unified QC using tool.py end-to-end pipeline WITH step-level provenance flags.

    Returns:
        qc (dict): trimmed arrays + final flags + step flags (+ optional ssc_q_bounds)
        qc_report (dict): station-level summary counters (final flags统计)
    """

    # --- force strict 1D & align length (avoid 0d / mismatched) ---
    time = np.atleast_1d(np.asarray(time)).reshape(-1)
    Q    = np.atleast_1d(np.asarray(Q,   dtype=float)).reshape(-1)
    SSC  = np.atleast_1d(np.asarray(SSC, dtype=float)).reshape(-1)
    SSL  = np.atleast_1d(np.asarray(SSL, dtype=float)).reshape(-1)

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None, None
    time, Q, SSC, SSL = time[:n], Q[:n], SSC[:n], SSL[:n]

    # --- tool pipeline: QC1/QC2/QC3 + provenance ---
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,     # ✅ Q 是源数据
        SSC_is_independent=True,   # ✅ SSC 是源数据
        SSL_is_independent=False,  # ✅ SSL 是推导量
        ssl_is_derived_from_q_ssc=True,
        qc2_k=1.5, qc2_min_samples=5,
        qc3_k=1.5, qc3_min_samples=5,
    )
    if qc is None:
        return None, None

    # --- valid_time: value-based missing detection (更稳) ---
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

    # trim ALL arrays incl. step flags
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and qc[k].shape[0] == valid_time.shape[0]:
            qc[k] = qc[k][valid_time]

    # ---------- build flat qc_report for CSV ----------
    F9 = int(FILL_VALUE_INT)

    def _count_final(f):
        f = np.asarray(f, dtype=np.int8)
        return {
            "good":      int(np.sum(f == 0)),
            "estimated": int(np.sum(f == 1)),
            "suspect":   int(np.sum(f == 2)),
            "bad":       int(np.sum(f == 3)),
            "missing":   int(np.sum(f == F9)),
        }

    def _count_step(f, mapping):
        # mapping: {"pass":0, "bad":3, "missing":9} 之类
        f = np.asarray(f, dtype=np.int8)
        return {name: int(np.sum(f == np.int8(code))) for name, code in mapping.items()}

    qf   = qc["Q_flag"]
    sscf = qc["SSC_flag"]
    sslf = qc["SSL_flag"]

    qc_report = {
        "QC_n_days": int(len(qc["time"])) if "time" in qc else int(len(qf)),

        # final
        "Q_final_good": _count_final(qf)["good"],
        "Q_final_estimated": _count_final(qf)["estimated"],
        "Q_final_suspect": _count_final(qf)["suspect"],
        "Q_final_bad": _count_final(qf)["bad"],
        "Q_final_missing": _count_final(qf)["missing"],

        "SSC_final_good": _count_final(sscf)["good"],
        "SSC_final_estimated": _count_final(sscf)["estimated"],
        "SSC_final_suspect": _count_final(sscf)["suspect"],
        "SSC_final_bad": _count_final(sscf)["bad"],
        "SSC_final_missing": _count_final(sscf)["missing"],

        "SSL_final_good": _count_final(sslf)["good"],
        "SSL_final_estimated": _count_final(sslf)["estimated"],
        "SSL_final_suspect": _count_final(sslf)["suspect"],
        "SSL_final_bad": _count_final(sslf)["bad"],
        "SSL_final_missing": _count_final(sslf)["missing"],
    }

    # step flags（有就统计，没有就跳过）
    if "Q_flag_qc1_physical" in qc:
        c = _count_step(qc["Q_flag_qc1_physical"], {"pass":0, "bad":3, "missing":9})
        qc_report.update({"Q_qc1_pass":c["pass"], "Q_qc1_bad":c["bad"], "Q_qc1_missing":c["missing"]})

    if "SSC_flag_qc1_physical" in qc:
        c = _count_step(qc["SSC_flag_qc1_physical"], {"pass":0, "bad":3, "missing":9})
        qc_report.update({"SSC_qc1_pass":c["pass"], "SSC_qc1_bad":c["bad"], "SSC_qc1_missing":c["missing"]})

    if "SSL_flag_qc1_physical" in qc:
        c = _count_step(qc["SSL_flag_qc1_physical"], {"pass":0, "bad":3, "missing":9})
        qc_report.update({"SSL_qc1_pass":c["pass"], "SSL_qc1_bad":c["bad"], "SSL_qc1_missing":c["missing"]})

    if "Q_flag_qc2_log_iqr" in qc:
        c = _count_step(qc["Q_flag_qc2_log_iqr"], {"pass":0, "suspect":2, "not_checked":8, "missing":9})
        qc_report.update({"Q_qc2_pass":c["pass"], "Q_qc2_suspect":c["suspect"], "Q_qc2_not_checked":c["not_checked"], "Q_qc2_missing":c["missing"]})

    if "SSC_flag_qc2_log_iqr" in qc:
        c = _count_step(qc["SSC_flag_qc2_log_iqr"], {"pass":0, "suspect":2, "not_checked":8, "missing":9})
        qc_report.update({"SSC_qc2_pass":c["pass"], "SSC_qc2_suspect":c["suspect"], "SSC_qc2_not_checked":c["not_checked"], "SSC_qc2_missing":c["missing"]})

    if "SSL_flag_qc2_log_iqr" in qc:
        c = _count_step(qc["SSL_flag_qc2_log_iqr"], {"pass":0, "suspect":2, "not_checked":8, "missing":9})
        qc_report.update({"SSL_qc2_pass":c["pass"], "SSL_qc2_suspect":c["suspect"], "SSL_qc2_not_checked":c["not_checked"], "SSL_qc2_missing":c["missing"]})

    if "SSC_flag_qc3_ssc_q" in qc:
        c = _count_step(qc["SSC_flag_qc3_ssc_q"], {"pass":0, "suspect":2, "not_checked":8, "missing":9})
        qc_report.update({"SSC_qc3_pass":c["pass"], "SSC_qc3_suspect":c["suspect"], "SSC_qc3_not_checked":c["not_checked"], "SSC_qc3_missing":c["missing"]})

    if "SSL_flag_qc3_from_ssc_q" in qc:
        # 你们约定：0 not_propagated, 1 propagated, 8 not_checked, 9 missing（如果你代码里是这个）
        c = _count_step(qc["SSL_flag_qc3_from_ssc_q"], {"not_propagated":0, "propagated":1, "not_checked":8, "missing":9})
        qc_report.update({"SSL_qc3_not_propagated":c["not_propagated"], "SSL_qc3_propagated":c["propagated"], "SSL_qc3_not_checked":c["not_checked"], "SSL_qc3_missing":c["missing"]})

    return qc, qc_report



def _daily_flag_reduce(x: pd.Series) -> np.int8:
    """
    Conservative daily reducer:
    - if all missing(9) -> 9
    - else take max over non-9 flags (so any suspect/bad will dominate)
    """
    arr = pd.to_numeric(x, errors="coerce").dropna().astype(int).values
    if arr.size == 0:
        return np.int8(9)
    arr = arr[arr != 9]
    if arr.size == 0:
        return np.int8(9)
    return np.int8(np.max(arr))


def process_and_convert(qc_df):
    """
    From QCed 5-min (or irregular) series -> daily mean values + daily flags.
    qc_df must have columns: Q, SSC, SSL, Q_flag, SSC_flag, SSL_flag (+ step flags optional)
    Index must be datetime.
    """
    print("  Aggregating to daily and ensuring standard units...")

    # --- daily mean values ---
    daily = qc_df[["Q", "SSC"]].resample("D").mean(numeric_only=True)
    daily["SSL"] = daily["Q"] * daily["SSC"] * 0.0864
    daily.loc[daily["SSL"] < 0, "SSL"] = np.nan

    # --- daily final flags (conservative) ---
    daily["Q_flag"] = qc_df["Q_flag"].resample("D").apply(_daily_flag_reduce)
    daily["SSC_flag"] = qc_df["SSC_flag"].resample("D").apply(_daily_flag_reduce)
    daily["SSL_flag"] = qc_df["SSL_flag"].resample("D").apply(_daily_flag_reduce)

    # --- daily step flags：只要 qc_df 里有，就一并聚合 ---
    step_cols = [c for c in qc_df.columns if ("flag_qc" in c) or c.endswith("_flag_qc1_physical") or c.endswith("_flag_qc2_log_iqr")]
    for c in step_cols:
        daily[c] = qc_df[c].resample("D").apply(_daily_flag_reduce)

    # 丢掉整天都没数据的天
    daily = daily.dropna(subset=["Q", "SSC", "SSL"], how="all")

    return daily


def create_netcdf(df, station_info, output_path, history_log):
    """Creates a CF-1.8 compliant NetCDF file."""
    print(f"  Creating NetCDF file: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
        # --- Dimensions ---
        ds.createDimension('time', None)

        # --- Global Attributes ---
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.summary = f"""This dataset provides quality-controlled, daily time series of river discharge (Q), suspended sediment concentration (SSC), and suspended sediment load (SSL) for the {station_info['station_name']} station on the {station_info['river_name']}. Data is from Robotham et al. (2022) and has been harmonized to CF-1.8 standards."""
        ds.source = "In-situ station data"
        ds.data_source_name = DATA_SOURCE_NAME
        ds.station_name = station_info['station_name']
        ds.river_name = station_info['river_name']
        ds.Source_ID = station_info['Source_ID']
        ds.geospatial_lat_min = station_info['latitude']
        ds.geospatial_lat_max = station_info['latitude']
        ds.geospatial_lon_min = station_info['longitude']
        ds.geospatial_lon_max = station_info['longitude']
        ds.geospatial_vertical_min = station_info.get('altitude', np.nan)
        ds.geospatial_vertical_max = station_info.get('altitude', np.nan)
        ds.geographic_coverage = GEOGRAPHIC_COVERAGE
        ds.time_coverage_start = df.index.min().strftime('%Y-%m-%d')
        ds.time_coverage_end = df.index.max().strftime('%Y-%m-%d')
        ds.temporal_span = f"{df.index.min().year}-{df.index.max().year}"
        ds.temporal_resolution = "daily"
        ds.variables_provided = "altitude, upstream_area, Q, SSC, SSL"
        ds.number_of_data = "1"
        ds.reference = REFERENCE
        ds.source_data_link = "https://doi.org/10.5285/9f80e349-0594-4ae1-bff3-b055638569f8"
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"
        ds.history = history_log
        ds.date_created = datetime.now().strftime('%Y-%m-%d')
        ds.date_modified = datetime.now().strftime('%Y-%m-%d')
        ds.processing_level = "Quality controlled and standardized"
        ds.comment = "Data represents daily mean values. Quality flags indicate data reliability: 0=good, 2=suspect, 3=bad, 9=missing."

        # --- Coordinate Variables ---
        time = ds.createVariable('time', 'f8', ('time',))
        time.long_name = "time"
        time.standard_name = "time"
        time.units = "days since 1970-01-01 00:00:00"
        time.calendar = "gregorian"
        reference_date = pd.Timestamp('1970-01-01')
        time[:] = (df.index - reference_date).total_seconds() / 86400.0

        lat = ds.createVariable('lat', 'f4', ())
        lat.long_name = "station latitude"
        lat.standard_name = "latitude"
        lat.units = "degrees_north"
        lat[:] = station_info['latitude']

        lon = ds.createVariable('lon', 'f4', ())
        lon.long_name = "station longitude"
        lon.standard_name = "longitude"
        lon.units = "degrees_east"
        lon[:] = station_info['longitude']

        # --- Data Variables ---
        def create_data_variable(var_name, standard_name, long_name, units, data, flag_var_name):
            var = ds.createVariable(var_name, 'f4', ('time',), fill_value=FILL_VALUE)
            var.standard_name = standard_name
            var.long_name = long_name
            var.units = units
            var.coordinates = "lat lon altitude"
            var.ancillary_variables = flag_var_name
            var[:] = data.fillna(FILL_VALUE).values

        def create_flag_variable(var_name, long_name):
            flag_var = ds.createVariable(var_name, 'b', ('time',), fill_value=np.int8(-127))
            flag_var.long_name = long_name
            flag_var.standard_name = "status_flag"
            flag_var.flag_values = np.array([0, 2, 3, 9], dtype=np.int8)
            flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
            flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source."

        # Altitude and Upstream Area
        for var, name, unit in [('altitude', 'altitude', 'm'), ('upstream_area', 'drainage_area', 'km2'),]:
            v = ds.createVariable(var, 'f4', (), fill_value=FILL_VALUE)
            v.long_name = f"station {name.replace('_', ' ')}"
            v.standard_name = name
            v.units = unit
            v[:] = station_info.get(var, np.nan)

        # Q, SSC, SSL and their flags
        create_flag_variable('Q_flag', "Quality flag for River Discharge")
        ds['Q_flag'][:] = df['Q_flag'].values.astype(np.int8)
        create_data_variable('Q', 'water_volume_transport_in_river_channel', 'River Discharge', 'm3 s-1', df['Q'], 'Q_flag')
        ds['Q'].comment = "Source: Original data from Robotham et al. (2022), converted from L/s."

        create_flag_variable('SSC_flag', "Quality flag for Suspended Sediment Concentration")
        ds['SSC_flag'][:] = df['SSC_flag'].values.astype(np.int8)
        create_data_variable('SSC', 'mass_concentration_of_suspended_matter_in_water', 'Suspended Sediment Concentration', 'mg L-1', df['SSC'], 'SSC_flag')
        ds['SSC'].comment = "Source: Original data from Robotham et al. (2022)."

        create_flag_variable('SSL_flag', "Quality flag for Suspended Sediment Load")
        ds['SSL_flag'][:] = df['SSL_flag'].values.astype(np.int8)
        create_data_variable('SSL', 'suspended_sediment_load', 'Suspended Sediment Load', 'ton day-1', df['SSL'], 'SSL_flag')
        ds['SSL'].comment = "Source: Calculated. Formula: SSL = Q * SSC * 0.0864."
        # -------------------------------------------------
        # Step-level provenance flags (QC1/QC2/QC3)
        # -------------------------------------------------
        def _add_step_flag(name, data, long_name, flag_values, flag_meanings):
            v = ds.createVariable(name, 'b', ('time',), fill_value=np.int8(-127))
            v.long_name = long_name
            v.standard_name = "status_flag"
            v.flag_values = np.array(flag_values, dtype=np.int8)
            v.flag_meanings = flag_meanings
            v[:] = np.asarray(data, dtype=np.int8)
            return v

        # QC1: physical (0 pass, 3 bad, 9 missing)
        qc1_vals = [0, 3, 9]
        qc1_mean = "pass bad missing"

        # QC2: log-IQR (0 pass, 2 suspect, 8 not_checked, 9 missing)
        qc2_vals = [0, 2, 8, 9]
        qc2_mean = "pass suspect not_checked missing"

        # QC3: SSC–Q (0 pass, 2 suspect, 8 not_checked, 9 missing)
        qc3_vals = [0, 2, 8, 9]
        qc3_mean = "pass suspect not_checked missing"

        # QC3 SSL propagation (0 not_propagated, 2 propagated, 8 not_checked, 9 missing)
        qc3_ssl_vals = [0, 2, 8, 9]
        qc3_ssl_mean = "not_propagated propagated not_checked missing"

        # 只要 df 里存在这些列，就写进 NetCDF
        for col in df.columns:
            if col == "Q_flag_qc1_physical":
                _add_step_flag(col, df[col].values, "QC1 physical check flag for discharge", qc1_vals, qc1_mean)
            elif col == "SSC_flag_qc1_physical":
                _add_step_flag(col, df[col].values, "QC1 physical check flag for SSC", qc1_vals, qc1_mean)
            elif col == "SSL_flag_qc1_physical":
                _add_step_flag(col, df[col].values, "QC1 physical check flag for SSL", qc1_vals, qc1_mean)
            elif col.endswith("flag_qc2_log_iqr"):
                _add_step_flag(col, df[col].values, f"QC2 log-IQR flag: {col}", qc2_vals, qc2_mean)
            elif col.endswith("flag_qc3_ssc_q"):
                _add_step_flag(col, df[col].values, "QC3 SSC–Q consistency flag", qc3_vals, qc3_mean)
            elif col.endswith("flag_qc3_from_ssc_q"):
                _add_step_flag(col, df[col].values, "QC3 SSL propagation from SSC–Q", qc3_ssl_vals, qc3_ssl_mean)

def generate_summary_csv(station_summaries, output_dir):
    """Generates a summary CSV for all processed stations."""
    if not station_summaries:
        print("No station data to summarize.")
        return

    summary_df = pd.DataFrame(station_summaries)
    # Add general info
    summary_df['Data Source Name'] = DATA_SOURCE_NAME
    summary_df['Type'] = "In-situ station data"
    summary_df['Temporal Resolution'] = "daily"
    summary_df['Geographic Coverage'] = GEOGRAPHIC_COVERAGE
    summary_df['Reference/DOI'] = "https://doi.org/10.5285/9f80e349-0594-4ae1-bff3-b055638569f8"

    # Reorder columns to match the desired output format
    cols = [
        'Source_ID', 'station_name', 'river_name', 'longitude', 'latitude', 'altitude', 'upstream_area',
        'Data Source Name', 'Type', 'Temporal Resolution', 'Temporal Span', 'Variables Provided',
        'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
    ]
    summary_df = summary_df[cols]

    output_path = os.path.join(output_dir, 'Robotham_station_summary.csv')
    print(f"Generating summary file: {output_path}")
    summary_df.to_csv(output_path, index=False)

def calculate_summary_stats(df, station_info):
    """Calculates summary statistics for the CSV."""
    summary = station_info.copy()
    summary['Temporal Span'] = f"{df.index.min().year}-{df.index.max().year}"
    summary['Variables Provided'] = "altitude, upstream_area, Q, SSC, SSL"
    for var in ['Q', 'SSC', 'SSL']:
        valid_data = df[df[f'{var}_flag'] == 0][var]
        if not valid_data.empty:
            summary[f'{var}_start_date'] = valid_data.index.min().strftime('%Y-%m-%d')
            summary[f'{var}_end_date'] = valid_data.index.max().strftime('%Y-%m-%d')
            total_period_days = (df.index.max() - df.index.min()).days + 1
            good_data_days = len(valid_data)
            summary[f'{var}_percent_complete'] = round((good_data_days / total_period_days) * 100, 2)
        else:
            summary[f'{var}_start_date'] = 'N/A'
            summary[f'{var}_end_date'] = 'N/A'
            summary[f'{var}_percent_complete'] = 0.0
    return summary

def print_qc_summary(station_name: str, merged: pd.DataFrame, created_path: str,
                     skipped_log_iqr: bool, skipped_ssc_q: bool):
    n_total = len(merged)

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

    # 这里按你的 df_final 列名来（如果你列名是大写 Q/SSC/SSL，就把下面改成对应的）
    qv, qf = _repr(merged['Q'].values, merged['Q_flag'].values)
    sscv, sscf = _repr(merged['SSC'].values, merged['SSC_flag'].values)
    sslv, sslf = _repr(merged['SSL'].values, merged['SSL_flag'].values)

    print(f"\nProcessing: {station_name}")
    if skipped_log_iqr:
        print(f"  ⚠ Sample size = {n_total} < 5, log-IQR statistical QC skipped.")
    if skipped_ssc_q:
        print(f"  ⚠ Sample size = {n_total} < 5, SSC–Q consistency check and diagnostic plot skipped.")
    print(f"  ✅ Created: {created_path}")
    print(f"  Q  : {qv:.2f} (flag={qf})")
    print(f"  SSC: {sscv:.2f} (flag={sscf})")
    print(f"  SSL: {sslv:.2f} (flag={sslf})")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Convert Robotham et al. (2022) data to NetCDF.")
    parser.add_argument('--source_dir', '-i', default=DEFAULT_SOURCE_DIR, 
                        help=f"Absolute path to the source data directory (default: {DEFAULT_SOURCE_DIR})")
    parser.add_argument('--output_dir', '-o', default=DEFAULT_OUTPUT_DIR, 
                        help=f"Absolute path to the target output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    print("=" * 80)
    print("Starting Robotham et al. (2022) data processing and harmonization")
    print("=" * 80)

    station_summaries = []
    history_log = (f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                   f"Converted from original Robotham et al. (2022) CSV to CF-1.8 compliant NetCDF. "
                   f"Applied daily averaging, quality control, and unit standardization. "
                   f"Script: {os.path.basename(sys.argv[0])}")

    for key, info in STATIONS.items():
        print(f"\nProcessing station: {info['station_name']} ({info['Source_ID']})")
        csv_path = os.path.join(args.source_dir, f"{key}.csv")

        if not os.path.exists(csv_path):
            print(f"  Source file not found: {csv_path}. Skipping.")
            continue

        df = read_and_prepare_data(csv_path)
        if df is None: continue

        df_truncated = truncate_time_range(df)
        if df_truncated is None or df_truncated.empty:
            print("  Skipping station due to no valid data in the specified range.")
            continue

        # 先把原始 5-min 数据做单位统一（推荐在 QC 前统一）
        Q_m3s = df_truncated["Q"].values * 0.001       # L/s -> m3/s
        SSC_mgl = df_truncated["SSC"].values           # mg/L
        SSL_tond = Q_m3s * SSC_mgl * 0.0864            # ton/day (instant-equivalent)

        qc, qc_report = apply_tool_qc(
            time=df_truncated.index.values,
            Q=Q_m3s,
            SSC=SSC_mgl,
            SSL=SSL_tond,
            station_id=info["Source_ID"],
            station_name=info["station_name"],
            plot_dir=os.path.join(args.output_dir, "diagnostic"),
        )

        if qc is None:
            print("  Skipping station due to no valid data after QC.")
            continue

        # qc dict -> DataFrame（time 做 index）
        qc_df = _qc_to_dataframe(qc)
        qc_df["time"] = pd.to_datetime(qc_df["time"])
        qc_df = qc_df.set_index("time").sort_index()

        ssc_q_bounds = qc.get("ssc_q_bounds", None)
        if ssc_q_bounds is not None:
            fig_path = os.path.join(args.output_dir, "diagnostic", f"SSC_Q_{info['Source_ID']}.png")
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plot_ssc_q_diagnostic(
                time=qc_df.index.values,
                Q=qc_df["Q"].values,
                SSC=qc_df["SSC"].values,
                Q_flag=qc_df["Q_flag"].values,
                SSC_flag=qc_df["SSC_flag"].values,
                ssc_q_bounds=ssc_q_bounds,
                station_id=info["Source_ID"],
                station_name=info["station_name"],
                out_png=fig_path,
            )


        df_final = process_and_convert(qc_df)


        if df_final[['Q', 'SSC']].isna().all().all():
            print("  Skipping station as all Q and SSC data are NaN after processing.")
            continue

        # Create NetCDF
        output_nc_path = os.path.join(args.output_dir, f"Robotham_{info['Source_ID']}.nc")
        create_netcdf(df_final, info, output_nc_path, history_log)

        # Collect summary stats
        summary = calculate_summary_stats(df_final, info)
        if qc_report is not None:
            summary.update(qc_report)
        station_summaries.append(summary)

    # Generate summary CSV
    if station_summaries:
        csv_station = os.path.join(args.output_dir, "Robotham_station_summary.csv")
        csv_qc = os.path.join(args.output_dir, "Robotham_qc_results_summary.csv")
        generate_csv_summary_tool(station_summaries, csv_station)
        generate_qc_results_csv_tool(station_summaries, csv_qc)

    print("\n" + "=" * 80)
    print(f"Processing complete. {len(station_summaries)} stations processed.")
    print("=" * 80)

if __name__ == '__main__':
    main()
