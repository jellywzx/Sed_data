#!/usr/bin/env python3
"""
Processes and standardizes the Chao Phraya River discharge and suspended
sediment dataset from PANGAEA (doi:10.1594/PANGAEA.981111) into CF-1.8
compliant NetCDF files.

This script performs the following steps:
1.  Reads the original tab-separated data file.
2.  Parses station metadata from the file header.
3.  Cleans and structures the data, handling duplicates.
4.  Converts units from source (km³/a, Mt/a) to CF standard (m³/s, ton/day).
5.  Calculates Suspended Sediment Concentration (SSC) from discharge and load.
6.  Performs quality control checks and assigns flags based on physical ranges.
7.  For each station with sediment data, creates a NetCDF file containing the
    complete annual time series.
8.  Generates a comprehensive CSV summary file for all processed stations.
"""

import os
import re
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import sys
import xarray as xr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    apply_quality_flag,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    propagate_ssc_q_inconsistency_to_ssl
)
PROJECT_ROOT = os.path.abspath(os.path.join(PARENT_DIR, '..'))

SOURCE_DATA_PATH = os.path.join(
    PROJECT_ROOT, "Source", "Chao_Phraya_River", "Chao_Phraya_River.tab"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "Output_r", "annually_climatology", "Chao_Phraya_River", "qc"
)

SUMMARY_DIR = OUTPUT_DIR

# Constants for unit conversion
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 31,557,600
KM3_TO_M3 = 1e9
MT_TO_TON = 1e6
DAYS_PER_YEAR = 365.25
# Factor for SSC = SSL / (Q * 86.4) where 86.4 = (86400 s/day * 1000 L/m³) / 1e6 mg/ton
SSC_CONVERSION_FACTOR = 86.4

# Quality control thresholds
Q_MIN_THRESHOLD = 0.0
SSC_MIN_THRESHOLD = 0.1  # As per user request (assuming mg/L)
SSC_MAX_THRESHOLD = 3000.0 # As per user request
SSL_MIN_THRESHOLD = 0.0

# Metadata
FILL_VALUE = -9999.0
REFERENCE_DATE = "1970-01-01 00:00:00"
CONVENTIONS = "CF-1.8, ACDD-1.3"
CREATOR_NAME = "Zhongwang Wei"
CREATOR_EMAIL = "weizhw6@mail.sysu.edu.cn"
CREATOR_INSTITUTION = "Sun Yat-sen University, China"
DATASET_NAME = "Chao_Phraya_River Dataset"
SOURCE_REFERENCE = "Wei, Bingbing (2025): Measured and estimated discharge and suspended sediment flux of the Chao Phraya River... PANGAEA, https://doi.org/10.1594/PANGAEA.981111"
SOURCE_LINK = "https://doi.org/10.1594/PANGAEA.981111"

# --- Helper Functions ---

def parse_station_metadata(file_path):
    """Parses station metadata from the header of the PANGAEA data file."""
    stations = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'LATITUDE:' in line and 'METHOD/DEVICE' in line:
                parts = line.strip().split('*')
                station_part = parts[0].replace('Event(s):', '').strip()
                print(f"DEBUG: Parsing station_part: '{station_part}'")
                match = re.match(r'(.+?)\s+\((.+?)\)', station_part)
                if match:
                    event_id, short_id = match.group(1).strip(), match.group(2).strip()
                    lat_match = re.search(r'LATITUDE:\s*([\d.]+)', line)
                    lon_match = re.search(r'LONGITUDE:\s*([\d.]+)', line)
                    loc_match = re.search(r'LOCATION:\s*([^*]+)', line)
                    stations[event_id] = {
                        'station_id': short_id,
                        'lat': float(lat_match.group(1)) if lat_match else None,
                        'lon': float(lon_match.group(1)) if lon_match else None,
                        'river': loc_match.group(1).strip() if loc_match else "Unknown"
                    }
            if line.startswith('Event\t'):
                break
    return stations

def read_and_clean_data(file_path):
    """Reads and cleans the tab-delimited data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('Event\t'):
                skiprows = i
                break
    df = pd.read_csv(file_path, sep='\t', skiprows=skiprows)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'Event': 'event', 'Station': 'station', 'River': 'river',
        'Comment': 'comment', 'Date/Time': 'year',
        'Q [km**3/a]': 'Q_km3_a', 'Ms [Mt/a]': 'Ms_Mt_a'
    })
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['Q_km3_a'] = pd.to_numeric(df['Q_km3_a'], errors='coerce')
    df['Ms_Mt_a'] = pd.to_numeric(df['Ms_Mt_a'], errors='coerce')
    df['data_score'] = df['Q_km3_a'].notna().astype(int) + df['Ms_Mt_a'].notna().astype(int)
    df = df.sort_values(['event', 'year', 'data_score'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['event', 'year'], keep='first')
    return df.drop(columns=['data_score'])

def convert_units_and_calculate_ssc(df):
    """Converts units and calculates SSC."""
    df['Q'] = df['Q_km3_a'] * KM3_TO_M3 / SECONDS_PER_YEAR
    df['SSL'] = df['Ms_Mt_a'] * MT_TO_TON / DAYS_PER_YEAR
    df['SSC'] = (df['SSL'] * 1e9) / (df['Q'] * 86400 * 1000)
    return df

def assign_quality_flags(df):
    df['Q_flag']   = [apply_quality_flag(v, 'Q')   for v in df['Q'].values]
    df['SSC_flag'] = [apply_quality_flag(v, 'SSC') for v in df['SSC'].values]
    df['SSL_flag'] = [apply_quality_flag(v, 'SSL') for v in df['SSL'].values]
    return df


def apply_station_level_qc(station_df):
    
    """
    Apply log-IQR and SSC–Q consistency QC using tool.py only.
    """

    Q = station_df['Q'].values.astype(float)
    SSC = station_df['SSC'].values.astype(float)
    SSL = station_df['SSL'].values.astype(float)

    Q_flag = station_df['Q_flag'].values.astype(np.int8)
    SSC_flag = station_df['SSC_flag'].values.astype(np.int8)
    SSL_flag = station_df['SSL_flag'].values.astype(np.int8)

    # ---------- log-IQR bounds ----------
    ssc_low, ssc_high = compute_log_iqr_bounds(SSC)
    ssl_low, ssl_high = compute_log_iqr_bounds(SSL)

    # ---------- SSC–Q envelope ----------
    ssc_q_bounds = build_ssc_q_envelope(
        Q_m3s=Q,
        SSC_mgL=SSC,
        k=1.5
    )

    # ---------- point-wise QC ----------
    for i in range(len(station_df)):

        # log-IQR (SSC)
        if SSC_flag[i] == 0 and ssc_low and ssc_high:
            if SSC[i] < ssc_low or SSC[i] > ssc_high:
                SSC_flag[i] = 2

        # log-IQR (SSL)
        if SSL_flag[i] == 0 and ssl_low and ssl_high:
            if SSL[i] < ssl_low or SSL[i] > ssl_high:
                SSL_flag[i] = 2

        # SSC–Q consistency
        is_inconsistent, _ = check_ssc_q_consistency(
            Q=Q[i],
            SSC=SSC[i],
            Q_flag=Q_flag[i],
            SSC_flag=SSC_flag[i],
            ssc_q_bounds=ssc_q_bounds
        )

        if is_inconsistent:
            # 1) SSC：一致性不通过 -> suspect
            if SSC_flag[i] == 0:
                SSC_flag[i] = np.int8(2)

            # 2) SSL：根据规则选择性传播
            SSL_flag[i] = propagate_ssc_q_inconsistency_to_ssl(
                inconsistent=is_inconsistent,
                Q=Q[i],
                SSC=SSC[i],
                SSL=SSL[i],
                Q_flag=Q_flag[i],
                SSC_flag=SSC_flag[i],
                SSL_flag=SSL_flag[i],
                ssl_is_derived_from_q_ssc=True,  #SSC 是由 Q 和 SSL 推出来的,所以这里设为 True。
            )


    station_df['Q_flag'] = Q_flag
    station_df['SSC_flag'] = SSC_flag
    station_df['SSL_flag'] = SSL_flag
    # --- Simple QC summary (for printing) ---
    n_valid_ssc = int(np.isfinite(SSC).sum())
    n_valid_ssl = int(np.isfinite(SSL).sum())

    log_iqr_skipped = (n_valid_ssc < 5) or (n_valid_ssl < 5)
    ssc_q_check_skipped = (n_valid_ssc < 5)

    # Flag counts
    q0 = int((Q_flag == 0).sum())
    q2 = int((Q_flag == 2).sum())
    q3 = int((Q_flag == 3).sum())
    q9 = int((Q_flag == 9).sum())

    s0 = int((SSC_flag == 0).sum())
    s2 = int((SSC_flag == 2).sum())
    s3 = int((SSC_flag == 3).sum())
    s9 = int((SSC_flag == 9).sum())

    l0 = int((SSL_flag == 0).sum())
    l2 = int((SSL_flag == 2).sum())
    l3 = int((SSL_flag == 3).sum())
    l9 = int((SSL_flag == 9).sum())

    # Attach to df attrs so main() can read without changing return type
    station_df.attrs["log_iqr_skipped"] = log_iqr_skipped
    station_df.attrs["ssc_q_check_skipped"] = ssc_q_check_skipped
    station_df.attrs["flag_counts"] = {
        "Q":   (q0, q2, q3, q9),
        "SSC": (s0, s2, s3, s9),
        "SSL": (l0, l2, l3, l9),
    }

    return station_df


def create_netcdf_file(station_id, event_id, meta, data, output_dir):
    """Creates a CF-1.8 compliant NetCDF file for a station."""
    filename = f"{DATASET_NAME.replace(' ', '_')}_{station_id.replace('.', '_')}.nc"
    filepath = os.path.join(output_dir, filename)
    
    with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
        # --- Dimensions ---
        ds.createDimension('time', None)

        # --- Global Attributes ---
        ds.Conventions = CONVENTIONS
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.source = "In-situ station data"
        ds.data_source_name = DATASET_NAME
        ds.station_name = station_id
        ds.river_name = meta['river']
        ds.Source_ID = station_id.replace('.', '_')
        ds.featureType = "timeSeries"
        ds.geospatial_lat_min = meta['lat']
        ds.geospatial_lat_max = meta['lat']
        ds.geospatial_lon_min = meta['lon']
        ds.geospatial_lon_max = meta['lon']
        ds.geographic_coverage = f"{meta['river']} Basin, Thailand"
        
        valid_data = data.dropna(subset=['Q', 'SSL'], how='all')
        if not valid_data.empty:
            start_year, end_year = int(valid_data['year'].min()), int(valid_data['year'].max())
            ds.time_coverage_start = f"{start_year}-01-01"
            ds.time_coverage_end = f"{end_year}-12-31"
            ds.temporal_span = f"{start_year}-{end_year}"

        ds.temporal_resolution = "annually"
        ds.variables_provided = "altitude, upstream_area, Q, SSC, SSL, station_name, river_name, Source_ID"
        ds.number_of_data = len(data)
        ds.reference = SOURCE_REFERENCE
        ds.source_data_link = SOURCE_LINK
        ds.creator_name = CREATOR_NAME
        ds.creator_email = CREATOR_EMAIL
        ds.creator_institution = CREATOR_INSTITUTION
        ds.date_created = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        ds.history = f"{ds.date_created}: Data processed from source. Script: process_chao_phraya.py."
        ds.summary = f"This file contains annual time series data for station {station_id} on the {meta['river']}."

        # --- Coordinate Variables ---
        time = ds.createVariable('time', 'f8', ('time',))
        time.units = f"days since {REFERENCE_DATE}"
        time.standard_name = "time"
        time.long_name = "time"
        time.calendar = "gregorian"
        time[:] = nc.date2num(pd.to_datetime(data['year'], format='%Y').to_list(), time.units, time.calendar)

        lat = ds.createVariable('lat', 'f4')
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat.long_name = "station latitude"
        lat[:] = meta['lat']

        lon = ds.createVariable('lon', 'f4')
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon.long_name = "station longitude"
        lon[:] = meta['lon']

        alt = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE)
        alt.standard_name = "altitude"
        alt.long_name = "station elevation above sea level"
        alt.units = "m"
        alt.positive = "up"
        alt.comment = "Source: Not available in original dataset."
        alt[:] = FILL_VALUE

        area = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE)
        area.long_name = "upstream drainage area"
        area.units = "km2"
        area.comment = "Source: Not available in original dataset."
        area[:] = FILL_VALUE

        # --- Data Variables ---
        def create_variable(name, std_name, long_name, units, comment, values, flag_values):
            var = ds.createVariable(name, 'f4', ('time',), fill_value=FILL_VALUE)
            var.standard_name = std_name
            var.long_name = long_name
            var.units = units
            var.comment = comment
            var.coordinates = "lat lon"
            var.ancillary_variables = f"{name}_flag"
            var[:] = values.fillna(FILL_VALUE).values

            flag_var = ds.createVariable(f"{name}_flag", 'i1', ('time',), fill_value=9)
            flag_var.long_name = f"Quality flag for {long_name}"
            flag_var.standard_name = "status_flag"
            flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
            flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
            flag_var[:] = flag_values.values
        
        create_variable('Q', 'river_discharge', 'River Discharge', 'm3 s-1',
                        f"Calculated from km³/a. Formula: Q * {KM3_TO_M3} / {SECONDS_PER_YEAR}",
                        data['Q'], data['Q_flag'])
        create_variable('SSC', 'mass_concentration_of_suspended_matter_in_water', 'Suspended Sediment Concentration', 'mg L-1',
                        f"Calculated from SSL and Q. Formula: SSC(mg/L) = (SSL(ton/day) * 1e9) / (Q(m3/s) * 86400 * 1000)",
                        data['SSC'], data['SSC_flag'])
        create_variable('SSL', 'suspended_sediment_load', 'Suspended Sediment Load', 'ton day-1',
                        f"Calculated from Mt/a. Formula: SSL * {MT_TO_TON} / {DAYS_PER_YEAR}",
                        data['SSL'], data['SSL_flag'])

    print(f"  -> Saved: {filepath}")

def generate_summary_csv(all_data, stations, output_dir):
    """Generates a summary CSV file for all stations."""
    summary_data = []
    for event, meta in stations.items():
        station_data = all_data[all_data['event'] == event]
        if station_data.empty:
            continue

        def get_stats(series_name):
            series = station_data[series_name]
            valid_series = series.dropna()
            if valid_series.empty:
                return None, None, 0.0
            start = int(station_data.loc[valid_series.index, 'year'].min())
            end = int(station_data.loc[valid_series.index, 'year'].max())
            completeness = (len(valid_series) / len(station_data)) * 100
            return start, end, completeness

        q_start, q_end, q_perc = get_stats('Q')
        ssc_start, ssc_end, ssc_perc = get_stats('SSC')
        ssl_start, ssl_end, ssl_perc = get_stats('SSL')

        summary_data.append({
            'Source_ID': meta['station_id'].replace('.', '_'),
            'station_name': meta['station_id'],
            'river_name': meta['river'],
            'longitude': meta['lon'],
            'latitude': meta['lat'],
            'altitude': FILL_VALUE,
            'upstream_area': FILL_VALUE,
            'Q_start_date': q_start or '',
            'Q_end_date': q_end or '',
            'Q_percent_complete': f"{q_perc:.1f}" if q_perc > 0 else "0.0",
            'SSC_start_date': ssc_start or '',
            'SSC_end_date': ssc_end or '',
            'SSC_percent_complete': f"{ssc_perc:.1f}" if ssc_perc > 0 else "0.0",
            'SSL_start_date': ssl_start or '',
            'SSL_end_date': ssl_end or '',
            'SSL_percent_complete': f"{ssl_perc:.1f}" if ssl_perc > 0 else "0.0",
            'Data Source Name': DATASET_NAME,
            'Type': 'In-situ station data',
            'Temporal Resolution': 'annually',
            'Temporal Span': f"{station_data['year'].min()}-{station_data['year'].max()}",
            'Variables Provided': 'Q, SSC, SSL',
            'Geographic Coverage': f"{meta['river']} Basin, Thailand",
            'Reference/DOI': SOURCE_LINK
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"{DATASET_NAME.replace(' ', '_')}_station_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  -> Saved: {summary_path}")

# --- Main Execution ---

def main():
    """Main function to run the data processing workflow."""
    print("---"" Starting Chao Phraya River Data Processing ---")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Parse Metadata
    print("[1/6] Parsing station metadata...")
    stations = parse_station_metadata(SOURCE_DATA_PATH)
    print(f"  Found metadata for {len(stations)} stations.")
    
    # 2. Read and Clean Data
    print("[2/6] Reading and cleaning data...")
    df = read_and_clean_data(SOURCE_DATA_PATH)
    print(f"  Loaded {len(df)} unique annual records.")
    
    # 3. Convert Units
    print("[3/6] Converting units and calculating SSC...")
    df = convert_units_and_calculate_ssc(df)
    
    # 4. Assign Quality Flags
    print("[4/6] Assigning quality flags...")
    df = assign_quality_flags(df)
    
    # 5. Create NetCDF files
    print("[5/6] Creating NetCDF files for stations with sediment data...")
    stations_with_sediment = df[df['SSL'].notna()]['event'].unique()
    for event, meta in stations.items():
        if event in stations_with_sediment:
            station_data = df[df['event'] == event].copy()
            station_data = apply_station_level_qc(station_data)
            # ---------- Print QC summary ----------
            sid = meta["station_id"]
            print(f"\nProcessing: {sid} +")

            if station_data.attrs.get("log_iqr_skipped", False):
                print(f"  | [{sid}] Sample size < 5, log-IQR statistical QC skipped.")
            else:
                print(f"  | [{sid}] log-IQR statistical QC applied.")

            if station_data.attrs.get("ssc_q_check_skipped", False):
                print(f"  | [{sid}] Sample size < 5, SSC–Q consistency check skipped.")
            else:
                print(f"  | [{sid}] SSC–Q consistency check and diagnostic plot.")

            # Representative values (mean of flag==0, fallback to mean of all finite)
            def _rep(x, f):
                x = x.astype(float)
                good = (f == 0) & np.isfinite(x)
                if good.any():
                    return float(np.nanmean(x[good]))
                return float(np.nanmean(x[np.isfinite(x)]))

            q_rep   = _rep(station_data["Q"].values,   station_data["Q_flag"].values)
            ssc_rep = _rep(station_data["SSC"].values, station_data["SSC_flag"].values)
            ssl_rep = _rep(station_data["SSL"].values, station_data["SSL_flag"].values)

            q0, q2, q3, q9 = station_data.attrs["flag_counts"]["Q"]
            s0, s2, s3, s9 = station_data.attrs["flag_counts"]["SSC"]
            l0, l2, l3, l9 = station_data.attrs["flag_counts"]["SSL"]

            print(f"  ✓ Flags Q   (0/2/3/9): {q0}/{q2}/{q3}/{q9}")
            print(f"  ✓ Flags SSC (0/2/3/9): {s0}/{s2}/{s3}/{s9}")
            print(f"  ✓ Flags SSL (0/2/3/9): {l0}/{l2}/{l3}/{l9}")

            print(f"  Q: {q_rep:.2f} m3/s (flag=0)")
            print(f"  SSC: {ssc_rep:.2f} mg/L (flag=0)")
            print(f"  SSL: {ssl_rep:.2f} ton/day (flag=0)")


            # =========================
            # SSC–Q diagnostic plot
            # =========================
            diag_dir = os.path.join(OUTPUT_DIR, "diagnostic")
            os.makedirs(diag_dir, exist_ok=True)

            diag_png = os.path.join(
                diag_dir,
                f"SSC_Q_{meta['station_id'].replace('.', '_')}.png"
            )

            # 重新构建 envelope（station-level）
            ssc_q_bounds = build_ssc_q_envelope(
                Q_m3s=station_data['Q'].values,
                SSC_mgL=station_data['SSC'].values,
                k=1.5
            )

            plot_ssc_q_diagnostic(
                time=pd.to_datetime(station_data['year'], format='%Y'),
                Q=station_data['Q'].values,
                SSC=station_data['SSC'].values,
                Q_flag=station_data['Q_flag'].values,
                SSC_flag=station_data['SSC_flag'].values,
                ssc_q_bounds=ssc_q_bounds,
                station_id=meta['station_id'],
                station_name=meta['river'],
                out_png=diag_png
            )

            df.loc[station_data.index, ['Q_flag','SSC_flag','SSL_flag']] = \
                station_data[['Q_flag','SSC_flag','SSL_flag']]

            # Truncate time
            valid_indices = station_data['Q'].notna() | station_data['SSL'].notna()
            if not valid_indices.any():
                print(f"  Skipping {meta['station_id']}: No valid Q or SSL data.")
                continue
            
            start_year = station_data.loc[valid_indices, 'year'].min()
            end_year = station_data.loc[valid_indices, 'year'].max()
            station_data = station_data[(station_data['year'] >= start_year) & (station_data['year'] <= end_year)]

            if station_data.empty:
                 print(f"  Skipping {meta['station_id']}: No data in range.")
                 continue

            create_netcdf_file(meta['station_id'], event, meta, station_data, OUTPUT_DIR)
        else:
            print(f"  Skipping {meta['station_id']}: No sediment data found.")

    # 6. Generate Summary CSV
    print("[6/6] Generating summary CSV...")
    generate_summary_csv(df, stations, SUMMARY_DIR)
    
    print("---"" Processing Complete ---")


if __name__ == "__main__":
    main()
