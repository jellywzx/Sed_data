#!/usr/bin/env python3
"""
Convert Aquasat and RiverSed CSV data to netCDF format
Following HYBAM example structure
Discharge is set to NaN (no in-situ discharge available)
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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

PROJECT_ROOT = Path(CURRENT_DIR).parents[1]  
SOURCE_DIR = str(PROJECT_ROOT / "Source" / "RiverSed")
OUTPUT_NC_DIR = str(PROJECT_ROOT / "Output_r" / "daily" / "RiverSed" / "nc")
OUTPUT_QC_DIR = str(PROJECT_ROOT / "Output_r" / "daily" / "RiverSed" / "qc")

def load_aquasat_data(file_path):
    """Load Aquasat TSS data"""
    print(f"Loading Aquasat data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    # Rename columns
    df = df.rename(columns={
        'value': 'tss',
        'SiteID': 'station_id'
    })

    # Select relevant columns
    cols = ['station_id', 'date', 'tss', 'lat', 'long', 'elevation']
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['station_id', 'tss'])

    print(f"  Loaded {len(df)} records from {df['station_id'].nunique()} stations")
    return df

def load_riversed_data(file_path):
    """Load RiverSed USA data"""
    print(f"Loading RiverSed data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Combine date and time
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

    # Use ID as station_id
    df['station_id'] = 'RiverSed_' + df['ID'].astype(str)

    # Select relevant columns
    cols = ['station_id', 'date', 'tss', 'elevation']
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['tss'])

    print(f"  Loaded {len(df)} records from {df['station_id'].nunique()} stations")
    return df

def find_overlap_period(tss_dates):
    """Find time period for TSS data"""
    if len(tss_dates) == 0:
        return None, None

    tss_min = tss_dates.min()
    tss_max = tss_dates.max()

    # Start from first day of the month with first data
    # End on Dec 31 of the year with last data
    start = pd.Timestamp(year=tss_min.year, month=tss_min.month, day=1)
    end = pd.Timestamp(year=tss_max.year, month=12, day=31)

    return start, end

def create_daily_timeseries(start_date, end_date):
    """Create daily time series"""
    return pd.date_range(start=start_date, end=end_date, freq='D')

def apply_satellite_ssc_qc(df, station_id, diagnostic_dir=None):
    """
    QC for satellite-only SSC (TSS) data using tool.py logic.

    Final flag convention:
      0 good, 2 suspect, 3 bad, 9 missing

    Step flags:
      - QC1 physical: 0 pass, 3 bad, 9 missing
      - QC2 log-IQR : 0 pass, 2 suspect, 8 not_checked, 9 missing
      - QC3 SSC–Q   : 0 pass, 2 suspect, 8 not_checked, 9 missing (satellite-only => mostly 8/9)
    """

    # ---- force strict 1D (避免 0D / len() 报错) ----
    ssc = np.atleast_1d(np.asarray(df["tss"], dtype=float)).reshape(-1)
    if ssc.size == 0:
        return None

    # -----------------------------
    # QC1. physical feasibility (vectorized)
    # -----------------------------
    ssc_flag_qc1 = apply_quality_flag_array(ssc, variable_name="SSC")  # 0/3/9

    # -----------------------------
    # QC2. log-IQR screening
    # -----------------------------
    ssc_flag_qc2 = np.full(ssc.shape, 8, dtype=np.int8)  # default not_checked
    # missing remains 9
    ssc_flag_qc2[ssc_flag_qc1 == 9] = 9

    lower, upper = compute_log_iqr_bounds(ssc)
    if (lower is not None) and (upper is not None) and (ssc.size >= 5):
        # for non-missing, mark pass first
        ssc_flag_qc2[(ssc_flag_qc1 != 9)] = 0
        outlier = (ssc < lower) | (ssc > upper)
        # only mark suspect where it is not missing & not already bad
        ssc_flag_qc2[outlier & (ssc_flag_qc1 != 9)] = 2

    # -----------------------------
    # QC3. SSC–Q consistency (satellite-only => not_checked)
    # -----------------------------
    ssc_flag_qc3 = np.full(ssc.shape, 8, dtype=np.int8)
    ssc_flag_qc3[ssc_flag_qc1 == 9] = 9

    # -----------------------------
    # Final flag combine (QC1 dominates bad/missing; QC2 may set suspect)
    # -----------------------------
    ssc_flag_final = ssc_flag_qc1.copy()  # start with 0/3/9
    # if QC1 pass and QC2 says suspect => final suspect
    ssc_flag_final[(ssc_flag_qc1 == 0) & (ssc_flag_qc2 == 2)] = 2

    # -----------------------------
    # Mask bad & missing values
    # -----------------------------
    ssc_clean = ssc.copy()
    ssc_clean[(ssc_flag_final == 3) | (ssc_flag_final == 9)] = np.nan

    df["tss"] = ssc_clean
    df["SSC_flag"] = ssc_flag_final.astype(np.int8)
    df["SSC_flag_qc1_physical"] = ssc_flag_qc1.astype(np.int8)
    df["SSC_flag_qc2_log_iqr"] = ssc_flag_qc2.astype(np.int8)
    df["SSC_flag_qc3_ssc_q"] = ssc_flag_qc3.astype(np.int8)

    # 至少保留一个非缺测值
    if np.all(np.isnan(df["tss"].values)):
        print(f"  -> All SSC invalid after QC for station {station_id}")
        return None

    return df

def create_netcdf_file(station_id, tss_df, output_dir):
    """Create netCDF file following HYBAM format"""

    # Check if all TSS values are NaN
    if tss_df['tss'].isna().all():
        print(f"  All TSS values are NaN for station {station_id}")
        return None

    # Find time period
    start_date, end_date = find_overlap_period(tss_df['date'])

    if start_date is None:
        print(f"  No valid dates for station {station_id}")
        return None

    # 只保留有数据的日期（按日平均）
    tss_df['date'] = pd.to_datetime(tss_df['date'], errors='coerce')
    tss_df = tss_df.dropna(subset=['date'])
    tss_df['date'] = tss_df['date'].dt.floor('D')

    tss_daily = tss_df.groupby('date', as_index=False)['tss'].mean()

    # 只保留有数据的时间点（不补全）
    daily_df = tss_daily.sort_values('date').reset_index(drop=True)

    # -----------------------------
    # Apply QC using tool.py
    # -----------------------------
    diagnostic_dir = Path(output_dir) / "diagnostic"
    daily_df = apply_satellite_ssc_qc(daily_df, station_id, diagnostic_dir=diagnostic_dir)

    if daily_df is None:
        return None
    # --- pad flags to int8 (NaN -> 9) ---
    for col in list(daily_df.columns):
        if col.endswith("_flag") or ("flag_qc" in col):
            daily_df[col] = daily_df[col].fillna(FILL_VALUE_INT).astype(np.int8)
    # -----------------------------
    # Print QC summary (station-level)
    # -----------------------------
    n_total = len(daily_df)

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

    sscv, sscf = _repr(daily_df["tss"].values, daily_df["SSC_flag"].values)

    # 这个脚本没有 Q/SSL（都写成 missing），所以这里固定输出 nan/9
    qv, qf = float("nan"), 9
    sslv, sslf = float("nan"), 9

    # 是否跳过 log-IQR（样本<5 或者 bounds=None）
    lower, upper = compute_log_iqr_bounds(daily_df["tss"].values.astype(float))
    skipped_log_iqr = (n_total < 5) or (lower is None)

    print(f"  ✓ QC summary ({station_id})")
    print(f"    Samples: {n_total}")
    print(f"    Skipped log-IQR: {skipped_log_iqr}")
    print(f"    Q  : {qv:.2f} m3/s (flag={qf})")
    print(f"    SSC: {sscv:.2f} mg/L (flag={sscf})")
    print(f"    SSL: {sslv:.2f} ton/day (flag={sslf})")



    # Get metadata (use first non-null values)
    latitude = tss_df['lat'].dropna().iloc[0] if 'lat' in tss_df.columns and not tss_df['lat'].dropna().empty else np.nan
    longitude = tss_df['long'].dropna().iloc[0] if 'long' in tss_df.columns and not tss_df['long'].dropna().empty else np.nan
    altitude = tss_df['elevation'].dropna().iloc[0] if 'elevation' in tss_df.columns and not tss_df['elevation'].dropna().empty else np.nan

    # Sanitize station_id for filename (replace invalid characters)
    safe_station_id = str(station_id).replace('/', '_').replace('\\', '_').replace(':', '_')

    # Create netCDF file
    output_file = Path(output_dir) / f"RiverSed_{safe_station_id}.nc"

    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # Create dimensions
        time_dim = ds.createDimension('time', len(daily_df))

        # Create time variable
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time of measurement'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        # Convert dates to days since 1970-01-01
        reference_date = pd.Timestamp('1970-01-01')
        time_var[:] = (daily_df['date'] - reference_date).dt.total_seconds() / 86400.0

        # Create coordinate variables
        lat_var = ds.createVariable('latitude', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = [-90.0, 90.0]
        lat_var[:] = latitude if not np.isnan(latitude) else -9999.0

        lon_var = ds.createVariable('longitude', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = [-180.0, 180.0]
        lon_var[:] = longitude if not np.isnan(longitude) else -9999.0

        alt_var = ds.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station altitude above sea level'
        alt_var.units = 'm'
        alt_var[:] = altitude if not np.isnan(altitude) else -9999.0

        area_var = ds.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Not available for satellite-derived data'
        area_var[:] = -9999.0

        # Create data variables
        Q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
        Q_var.standard_name = 'water_volume_transport_in_river_channel'
        Q_var.long_name = 'river discharge'
        Q_var.units = 'm3 s-1'
        Q_var.coordinates = 'time latitude longitude'
        Q_var.comment = 'Discharge data not available - all values set to missing'
        # Set all discharge to fill value (NaN equivalent)
        Q_var[:] = -9999.0

        Q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        Q_flag_var.long_name = 'quality flag for river discharge'
        Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='b')
        Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        Q_flag_var.comment = 'All set to 9 (missing) - discharge not available for satellite data'
        Q_flag_var[:] = np.full(len(daily_df), FILL_VALUE_INT, dtype=np.int8)

        SSC_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
        SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        SSC_var.long_name = 'suspended sediment concentration'
        SSC_var.units = 'mg L-1'
        SSC_var.coordinates = 'time latitude longitude'
        SSC_var.comment = 'SSC from satellite observations (Aquasat/RiverSed database)'
        SSC_var[:] = daily_df['tss'].fillna(-9999.0).values

        SSC_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
        SSC_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        SSC_flag_var.flag_meanings = 'good_data suspect_data bad_data missing_data'
        SSC_flag_var.comment = (
            'QC applied using tool.py: physical validity + log-IQR outlier screening. '
            'Satellite-derived SSC only; no SSC–Q consistency check.'
        )
        SSC_flag_var[:] = daily_df['SSC_flag'].values

        SSL_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
        SSL_var.long_name = 'suspended sediment load'
        SSL_var.units = 'ton day-1'
        SSL_var.coordinates = 'time latitude longitude'
        SSL_var.comment = 'Cannot be calculated without discharge data - all values set to missing'
        # Set all sediment load to fill value
        SSL_var[:] = -9999.0

        SSL_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        SSL_flag_var.long_name = 'quality flag for suspended sediment load'
        SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='b')
        SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        SSL_flag_var.comment = 'All set to 9 (missing) - cannot be calculated without discharge'
        SSL_flag_var[:] = np.full(len(daily_df), FILL_VALUE_INT, dtype=np.int8)
        # =============================
        # Step / provenance flags (SSC)
        # =============================
        def _add_step_flag(name, values, flag_values, flag_meanings, long_name):
            v = ds.createVariable(name, 'b', ('time',), fill_value=FILL_VALUE_INT)
            v.long_name = long_name
            v.standard_name = 'status_flag'
            v.flag_values = np.array(flag_values, dtype='b')
            v.flag_meanings = flag_meanings
            v.missing_value = np.int8(FILL_VALUE_INT)
            v[:] = np.asarray(values, dtype=np.int8)
            return v

        if "SSC_flag_qc1_physical" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc1_physical",
                daily_df["SSC_flag_qc1_physical"].values,
                flag_values=[0, 3, 9],
                flag_meanings="pass bad missing",
                long_name="QC1 physical flag for suspended sediment concentration",
            )

        if "SSC_flag_qc2_log_iqr" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc2_log_iqr",
                daily_df["SSC_flag_qc2_log_iqr"].values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC2 log-IQR flag for suspended sediment concentration",
            )

        if "SSC_flag_qc3_ssc_q" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc3_ssc_q",
                daily_df["SSC_flag_qc3_ssc_q"].values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC3 SSC–Q consistency flag for suspended sediment concentration (satellite-only)",
            )

        # Global attributes
        ds.Conventions = 'CF-1.8'
        ds.title = f'RiverSed Satellite-derived TSS Data for Station {station_id}'
        ds.institution = 'University of North Carolina at Chapel Hill'
        ds.source = 'Satellite-derived TSS from Aquasat/RiverSed database'
        ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf.py'
        ds.references = 'Gardner et al. (2021), The color of rivers, Geophysical Research Letters, doi:10.1029/2020GL088946'
        ds.comment = 'TSS values derived from Landsat satellite imagery. Discharge and sediment load are not available and set to missing values.'
        ds.station_id = str(station_id)
        ds.data_period_start = start_date.strftime('%Y-%m-%d')
        ds.data_period_end = end_date.strftime('%Y-%m-%d')

    # --- build station_info for CSV outputs ---
    n = len(daily_df)

    def _cnt(arr, v):
        a = np.asarray(arr, dtype=np.int8)
        return int(np.sum(a == np.int8(v)))

    # final flags
    ssc_f = daily_df["SSC_flag"].values.astype(np.int8)

    station_info = {
        # metadata summary fields
        "station_name": str(station_id).replace("_", " "),
        "Source_ID": str(station_id),
        "river_name": "",
        "longitude": float(longitude) if not np.isnan(longitude) else np.nan,
        "latitude": float(latitude) if not np.isnan(latitude) else np.nan,
        "altitude": float(altitude) if not np.isnan(altitude) else np.nan,
        "upstream_area": np.nan,
        "Data Source Name": "RiverSed / Aquasat (satellite-derived TSS)",
        "Type": "Satellite",
        "Temporal Resolution": "daily",
        "Temporal Span": f"{daily_df['date'].min().strftime('%Y-%m-%d')} to {daily_df['date'].max().strftime('%Y-%m-%d')}",
        "Variables Provided": "SSC",
        "Geographic Coverage": "",
        "Reference/DOI": "Gardner et al. (2021) doi:10.1029/2020GL088946",

        # QC summary fields (QC results CSV会自动挑存在的列)
        "QC_n_days": int(n),

        "SSC_final_good": _cnt(ssc_f, 0),
        "SSC_final_estimated": _cnt(ssc_f, 1),
        "SSC_final_suspect": _cnt(ssc_f, 2),
        "SSC_final_bad": _cnt(ssc_f, 3),
        "SSC_final_missing": _cnt(ssc_f, 9),
    }

    # step flags counts (如果列存在就加进去)
    if "SSC_flag_qc1_physical" in daily_df.columns:
        f = daily_df["SSC_flag_qc1_physical"].values.astype(np.int8)
        station_info.update({
            "SSC_qc1_pass": _cnt(f, 0),
            "SSC_qc1_bad": _cnt(f, 3),
            "SSC_qc1_missing": _cnt(f, 9),
        })

    if "SSC_flag_qc2_log_iqr" in daily_df.columns:
        f = daily_df["SSC_flag_qc2_log_iqr"].values.astype(np.int8)
        station_info.update({
            "SSC_qc2_pass": _cnt(f, 0),
            "SSC_qc2_suspect": _cnt(f, 2),
            "SSC_qc2_not_checked": _cnt(f, 8),
            "SSC_qc2_missing": _cnt(f, 9),
        })

    if "SSC_flag_qc3_ssc_q" in daily_df.columns:
        f = daily_df["SSC_flag_qc3_ssc_q"].values.astype(np.int8)
        station_info.update({
            "SSC_qc3_pass": _cnt(f, 0),
            "SSC_qc3_suspect": _cnt(f, 2),
            "SSC_qc3_not_checked": _cnt(f, 8),
            "SSC_qc3_missing": _cnt(f, 9),
        })

    print(f"  Created {output_file}")
    return station_info


def main():
    # Configuration with WSL absolute paths
    aquasat_file = os.path.join(SOURCE_DIR, 'Aquasat_TSS_v1.1.csv')
    riversed_file = os.path.join(SOURCE_DIR, 'RiverSed_USA_V1.1.txt')
    output_nc_dir = OUTPUT_NC_DIR
    output_qc_dir = OUTPUT_QC_DIR

    Path(output_nc_dir).mkdir(parents=True, exist_ok=True)
    Path(output_qc_dir).mkdir(parents=True, exist_ok=True)

    stations_info = []   # 用于两个CSV（summary + qc_results）

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    aquasat_df = load_aquasat_data(aquasat_file)
    riversed_df = load_riversed_data(riversed_file)

    # Process each dataset separately
    print("\n" + "="*80)
    print("PROCESSING AQUASAT STATIONS")
    print("="*80)

    aquasat_stations = aquasat_df['station_id'].unique()
    print(f"Processing {len(aquasat_stations)} Aquasat stations...")

    aquasat_success = 0
    aquasat_failed = 0

    print("\n" + "="*80)
    print("PROCESSING AQUASAT STATIONS")
    print("="*80)

    aquasat_stations = aquasat_df['station_id'].unique()
    print(f"Processing {len(aquasat_stations)} Aquasat stations...")

    aquasat_success = 0
    aquasat_failed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(create_netcdf_file, station_id, aquasat_df[aquasat_df['station_id'] == station_id].copy(), output_nc_dir): station_id
            for station_id in aquasat_stations
        }

        for i, future in enumerate(as_completed(futures), 1):
            station_id = futures[future]
            try:
                result = future.result()
                if isinstance(result, dict):
                    stations_info.append(result)
                    aquasat_success += 1
                else:
                    aquasat_failed += 1
            except Exception as e:
                print(f"  Station {station_id} failed with error: {e}")
                aquasat_failed += 1

            if i % 100 == 0:
                print(f"  Processed {i}/{len(aquasat_stations)} stations...")


    # For RiverSed, limit to stations with sufficient data
    riversed_station_counts = riversed_df.groupby('station_id').size()
    riversed_stations = riversed_station_counts[riversed_station_counts >= 5].index.tolist()
    print(f"Processing {len(riversed_stations)} RiverSed stations (with at least 5 observations)...")

    riversed_success = 0
    riversed_failed = 0

    print("\n" + "="*80)
    print("PROCESSING RIVERSED STATIONS")
    print("="*80)

    riversed_station_counts = riversed_df.groupby('station_id').size()
    riversed_stations = riversed_station_counts[riversed_station_counts >= 5].index.tolist()
    print(f"Processing {len(riversed_stations)} RiverSed stations (with at least 5 observations)...")

    riversed_success = 0
    riversed_failed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(create_netcdf_file, station_id, riversed_df[riversed_df['station_id'] == station_id].copy(), output_nc_dir): station_id
            for station_id in riversed_stations
        }

        for i, future in enumerate(as_completed(futures), 1):
            station_id = futures[future]
            try:
                result = future.result()
                if isinstance(result, dict):
                    stations_info.append(result)
                    aquasat_success += 1
                else:
                    aquasat_failed += 1
            except Exception as e:
                print(f"  Station {station_id} failed with error: {e}")
                riversed_failed += 1

            if i % 1000 == 0:
                print(f"  Processed {i}/{len(riversed_stations)} stations...")

    # -----------------------------
    # Generate CSV outputs (summary + QC results)
    # -----------------------------
    if stations_info:
        csv_summary = os.path.join(OUTPUT_QC_DIR, "RiverSed_station_summary.csv")
        csv_qc = os.path.join(OUTPUT_QC_DIR, "RiverSed_qc_results_summary.csv")

        generate_csv_summary_tool(stations_info, csv_summary)
        generate_qc_results_csv_tool(stations_info, csv_qc)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Aquasat:")
    print(f"  Total stations: {len(aquasat_stations)}")
    print(f"  Successfully created: {aquasat_success}")
    print(f"  Failed (all NaN or no data): {aquasat_failed}")
    print(f"\nRiverSed:")
    print(f"  Total stations (with ≥5 obs): {len(riversed_stations)}")
    print(f"  Successfully created: {riversed_success}")
    print(f"  Failed (all NaN or no data): {riversed_failed}")
    print(f"\nTotal netCDF files created: {aquasat_success + riversed_success}")
    print(f"Output directory: {output_dir}/")
    print("="*80)

if __name__ == '__main__':
    main()
