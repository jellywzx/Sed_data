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
    propagate_ssc_q_inconsistency_to_ssl,
    apply_quality_flag_array,        
    apply_hydro_qc_with_provenance, 
    generate_station_summary_csv, 
    generate_qc_results_csv,  
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
    # 1) 准备数组
    Q   = station_df["Q"].to_numpy(dtype=float)
    SSC = station_df["SSC"].to_numpy(dtype=float)
    SSL = station_df["SSL"].to_numpy(dtype=float)

    # time：这里不纠结“days since 1970”，年序列也能跑 QC（只是用于返回/画图）
    time = station_df["year"].to_numpy(dtype=float)

    # 2) 显式调用 QC1-array（你要求的第二个函数）
    Q_flag_qc1   = apply_quality_flag_array(Q,   "Q")
    SSC_flag_qc1 = apply_quality_flag_array(SSC, "SSC")
    SSL_flag_qc1 = apply_quality_flag_array(SSL, "SSL")

    # 用 QC1 的 missing(9) 来做 trim mask（和 tool 里 valid_time 逻辑一致）
    valid_time = (Q_flag_qc1 != 9) | (SSC_flag_qc1 != 9) | (SSL_flag_qc1 != 9)

    if not valid_time.any():
        return None 

    # 3) SSL, SSC (derived), Q
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSL_is_independent=True,
        SSC_is_independent=False,
        ssl_is_derived_from_q_ssc=False,
    )
    if qc is None:
        return None

    # 4) 对齐长度
    station_df = station_df.loc[valid_time].copy()


    # 额外保存 step flags 到 attrs，方便 main() 里统计 QC2/QC3
    station_df.attrs["qc_steps"] = {
        "Q_qc1":   qc["Q_flag_qc1_physical"].astype(np.int8),
        "SSC_qc1": qc["SSC_flag_qc1_physical"].astype(np.int8),
        "SSL_qc1": qc["SSL_flag_qc1_physical"].astype(np.int8),

        "Q_qc2":   qc["Q_flag_qc2_log_iqr"].astype(np.int8),
        "SSC_qc2": qc["SSC_flag_qc2_log_iqr"].astype(np.int8),
        "SSL_qc2": qc["SSL_flag_qc2_log_iqr"].astype(np.int8),

        "SSC_qc3": qc["SSC_flag_qc3_ssc_q"].astype(np.int8),
        "SSL_qc3_prop": qc["SSL_flag_qc3_from_ssc_q"].astype(np.int8),
    }


    # 5) 写回最终 flags（
    station_df["Q_flag"]   = qc["Q_flag"].astype("int8")
    station_df["SSC_flag"] = qc["SSC_flag"].astype("int8")
    station_df["SSL_flag"] = qc["SSL_flag"].astype("int8")

    Q_flag   = station_df["Q_flag"].to_numpy(np.int8)
    SSC_flag = station_df["SSC_flag"].to_numpy(np.int8)
    SSL_flag = station_df["SSL_flag"].to_numpy(np.int8)

    station_df.attrs["flag_counts"] = {
        "Q":   (int((Q_flag==0).sum()),   int((Q_flag==2).sum()),   int((Q_flag==3).sum()),   int((Q_flag==9).sum())),
        "SSC": (int((SSC_flag==0).sum()), int((SSC_flag==2).sum()), int((SSC_flag==3).sum()), int((SSC_flag==9).sum())),
        "SSL": (int((SSL_flag==0).sum()), int((SSL_flag==2).sum()), int((SSL_flag==3).sum()), int((SSL_flag==9).sum())),
    }

    station_df.attrs["log_iqr_skipped"] = (np.isfinite(station_df["SSC"]).sum() < 5) or (np.isfinite(station_df["SSL"]).sum() < 5)
    station_df.attrs["ssc_q_check_skipped"] = (np.isfinite(station_df["SSC"]).sum() < 5)

    # 确认两个函数都真的被调用
    print(
        f"[QC] QC1(good cnt): Q={(Q_flag_qc1==0).sum()}, SSC={(SSC_flag_qc1==0).sum()}, SSL={(SSL_flag_qc1==0).sum()} | "
        f"Final(good cnt): Q={(station_df['Q_flag'].to_numpy()==0).sum()}, "
        f"SSC={(station_df['SSC_flag'].to_numpy()==0).sum()}, SSL={(station_df['SSL_flag'].to_numpy()==0).sum()}"
    )

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
    stations_info = []

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
            
            # -------------------------
            # Build stations_info row for CSV summaries
            # -------------------------
            def _count_flags(arr, mapping):
                arr = np.asarray(arr, dtype=np.int8)
                return {k: int((arr == v).sum()) for k, v in mapping.items()}

            # final flags
            qf = station_data["Q_flag"].to_numpy(np.int8)
            sf = station_data["SSC_flag"].to_numpy(np.int8)
            lf = station_data["SSL_flag"].to_numpy(np.int8)

            final_map = {
                "good": 0, "estimated": 1, "suspect": 2, "bad": 3, "missing": 9
            }
            q_final = _count_flags(qf, final_map)
            s_final = _count_flags(sf, final_map)
            l_final = _count_flags(lf, final_map)

            # step flags (qc1/qc2/qc3)
            steps = station_data.attrs.get("qc_steps", {})
            qc1_map = {"pass": 0, "bad": 3, "missing": 9}
            qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}
            qc3_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}
            prop_map = {"not_propagated": 0, "propagated": 2, "not_checked": 8, "missing": 9}

            Q_qc1 = _count_flags(steps.get("Q_qc1", np.array([], dtype=np.int8)), qc1_map)
            S_qc1 = _count_flags(steps.get("SSC_qc1", np.array([], dtype=np.int8)), qc1_map)
            L_qc1 = _count_flags(steps.get("SSL_qc1", np.array([], dtype=np.int8)), qc1_map)

            Q_qc2 = _count_flags(steps.get("Q_qc2", np.array([], dtype=np.int8)), qc2_map)
            S_qc2 = _count_flags(steps.get("SSC_qc2", np.array([], dtype=np.int8)), qc2_map)
            L_qc2 = _count_flags(steps.get("SSL_qc2", np.array([], dtype=np.int8)), qc2_map)

            S_qc3 = _count_flags(steps.get("SSC_qc3", np.array([], dtype=np.int8)), qc3_map)
            L_qc3p = _count_flags(steps.get("SSL_qc3_prop", np.array([], dtype=np.int8)), prop_map)

            # time span for summary
            valid_any = station_data["Q"].notna() | station_data["SSC"].notna() | station_data["SSL"].notna()
            if valid_any.any():
                start_year = int(station_data.loc[valid_any, "year"].min())
                end_year   = int(station_data.loc[valid_any, "year"].max())
            else:
                start_year, end_year = None, None

            stations_info.append({
                "station_name": meta["station_id"],
                "Source_ID": meta["station_id"].replace(".", "_"),
                "river_name": meta["river"],
                "longitude": meta["lon"],
                "latitude": meta["lat"],

                # QC size
                "QC_n_days": int(len(station_data)),

                # final counts
                "Q_final_good": q_final["good"],
                "Q_final_estimated": q_final["estimated"],
                "Q_final_suspect": q_final["suspect"],
                "Q_final_bad": q_final["bad"],
                "Q_final_missing": q_final["missing"],

                "SSC_final_good": s_final["good"],
                "SSC_final_estimated": s_final["estimated"],
                "SSC_final_suspect": s_final["suspect"],
                "SSC_final_bad": s_final["bad"],
                "SSC_final_missing": s_final["missing"],

                "SSL_final_good": l_final["good"],
                "SSL_final_estimated": l_final["estimated"],
                "SSL_final_suspect": l_final["suspect"],
                "SSL_final_bad": l_final["bad"],
                "SSL_final_missing": l_final["missing"],

                # QC1
                "Q_qc1_pass": Q_qc1.get("pass", 0),
                "Q_qc1_bad": Q_qc1.get("bad", 0),
                "Q_qc1_missing": Q_qc1.get("missing", 0),

                "SSC_qc1_pass": S_qc1.get("pass", 0),
                "SSC_qc1_bad": S_qc1.get("bad", 0),
                "SSC_qc1_missing": S_qc1.get("missing", 0),

                "SSL_qc1_pass": L_qc1.get("pass", 0),
                "SSL_qc1_bad": L_qc1.get("bad", 0),
                "SSL_qc1_missing": L_qc1.get("missing", 0),

                # QC2
                "Q_qc2_pass": Q_qc2.get("pass", 0),
                "Q_qc2_suspect": Q_qc2.get("suspect", 0),
                "Q_qc2_not_checked": Q_qc2.get("not_checked", 0),
                "Q_qc2_missing": Q_qc2.get("missing", 0),

                "SSC_qc2_pass": S_qc2.get("pass", 0),
                "SSC_qc2_suspect": S_qc2.get("suspect", 0),
                "SSC_qc2_not_checked": S_qc2.get("not_checked", 0),
                "SSC_qc2_missing": S_qc2.get("missing", 0),

                "SSL_qc2_pass": L_qc2.get("pass", 0),
                "SSL_qc2_suspect": L_qc2.get("suspect", 0),
                "SSL_qc2_not_checked": L_qc2.get("not_checked", 0),
                "SSL_qc2_missing": L_qc2.get("missing", 0),

                # QC3
                "SSC_qc3_pass": S_qc3.get("pass", 0),
                "SSC_qc3_suspect": S_qc3.get("suspect", 0),
                "SSC_qc3_not_checked": S_qc3.get("not_checked", 0),
                "SSC_qc3_missing": S_qc3.get("missing", 0),

                "SSL_qc3_not_propagated": L_qc3p.get("not_propagated", 0),
                "SSL_qc3_propagated": L_qc3p.get("propagated", 0),
                "SSL_qc3_not_checked": L_qc3p.get("not_checked", 0),
                "SSL_qc3_missing": L_qc3p.get("missing", 0),

                # For generate_station_summary_csv (它需要这些键)
                "source_id": meta["station_id"].replace(".", "_"),
                "river_name": meta["river"],
                "longitude": meta["lon"],
                "latitude": meta["lat"],
                "altitude": np.nan,
                "upstream_area": np.nan,
                "start_year": start_year,
                "end_year": end_year,

                # 这里用“代表性 flag”给 tool.py 的 station_summary（它是按单值 0/非0 写 100/0 的）
                # 我们用“是否存在 good 数据”来定义
                "Q_flag": np.int8(0) if q_final["good"] > 0 else np.int8(9),
                "SSC_flag": np.int8(0) if s_final["good"] > 0 else np.int8(9),
                "SSL_flag": np.int8(0) if l_final["good"] > 0 else np.int8(9),
            })


        else:
            print(f"  Skipping {meta['station_id']}: No sediment data found.")

    # 6. Generate Summary CSV
    print("[6/6] Generating summary CSV...")
    # generate_summary_csv(df, stations, SUMMARY_DIR)

    # -------------------------
    # Tool-based CSV outputs
    # -------------------------
    qc_csv = os.path.join(SUMMARY_DIR, f"{DATASET_NAME.replace(' ', '_')}_qc_results.csv")
    generate_qc_results_csv(stations_info, qc_csv)

    # 注意：tool.py 的 generate_station_summary_csv 目前写死了文件名
    # 'ALi_De_Boer_station_summary.csv'，但我们仍按你的要求调用。
    # 站点汇总（通用版）
    generate_station_summary_csv(
        stations_info,
        SUMMARY_DIR,
        raw_df=df,
        stations=stations,
        dataset_name=DATASET_NAME,
        temporal_resolution="annually",
        geographic_coverage="Chao Phraya Basin, Thailand",
        reference_doi=SOURCE_LINK,
        extra_columns=["QC_n_days", "n_warnings", "warnings"],
        output_filename=f"{DATASET_NAME.replace(' ', '_')}_station_summary_combined.csv",
    )
    
    print("---"" Processing Complete ---")


if __name__ == "__main__":
    main()
