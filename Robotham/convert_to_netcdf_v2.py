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
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    # check_nc_completeness,
    # add_global_attributes
)


# --- Configuration ---

# WSL format absolute paths
DEFAULT_SOURCE_DIR = '/mnt/d/sediment_wzx_1111/Source/Robotham/data'
DEFAULT_OUTPUT_DIR = '/mnt/d/sediment_wzx_1111/Output_r/daily/Robotham'

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

def apply_tool_qc(df):
    """
    Apply unified QC using tool.py for Robotham et al. dataset.
    Includes:
    - Physical validity
    - log-IQR outlier detection
    - SSC–Q consistency check
    """

    out = df.copy()

    # =========================
    # 1. Physical QC
    # =========================
    out['Q_flag'] = np.array(
        [apply_quality_flag(v, "Q") for v in out['Q'].values],
        dtype=np.int8
    )
    out['SSC_flag'] = np.array(
        [apply_quality_flag(v, "SSC") for v in out['SSC'].values],
        dtype=np.int8
    )

    # Bad data → NaN
    out.loc[out['Q_flag'] == 3, 'Q'] = np.nan
    out.loc[out['SSC_flag'] == 3, 'SSC'] = np.nan

    # =========================
    # 2. log-IQR screening
    # =========================
    for var in ['Q', 'SSC']:
        data = out[var].values
        lower, upper = compute_log_iqr_bounds(data)
        if lower is not None:
            out.loc[(data < lower) | (data > upper), f'{var}_flag'] = 2

    # =========================
    # 3. SSC–Q consistency
    # =========================
    envelope = build_ssc_q_envelope(out['Q'].values, out['SSC'].values)

    if envelope is not None:
        for i in range(len(out)):
            inconsistent, _ = check_ssc_q_consistency(
                out['Q'].iloc[i],
                out['SSC'].iloc[i],
                out['Q_flag'].iloc[i],
                out['SSC_flag'].iloc[i],
                envelope
            )
            if inconsistent:
                out.loc[out.index[i], 'SSC_flag'] = 2

    return out, envelope



def process_and_convert(df):
    """Resamples to daily, converts units, and calculates SSL."""
    print("  Resampling to daily and converting units...")
    df_daily = df.resample('D').mean(numeric_only=True)

    df_out = pd.DataFrame(index=df_daily.index)

    # --- Unit Conversion ---
    # Q: L/s -> m³/s (multiply by 0.001)
    df_out['Q'] = df_daily['Q'] * 0.001
    # SSC: mg/L (no conversion needed)
    df_out['SSC'] = df_daily['SSC']

    # --- SSL Calculation ---
    # SSL (ton/day) = Q (m³/s) * SSC (mg/L) * 0.0864
    # (m³/s) * (g/m³) * (86400 s/day) / (1,000,000 g/ton) = ton/day
    df_out['SSL'] = df_out['Q'] * df_out['SSC'] * 0.0864
    df_out.loc[df_out['SSL'] < 0, 'SSL'] = np.nan # Mark negative SSL as bad

    # Carry over flags (use mode to get the most frequent flag of the day)
    df_out['Q_flag'] = df_daily['Q_flag'].resample('D').apply(lambda x: x.mode()[0] if not x.empty else 9)
    df_out['SSC_flag'] = df_daily['SSC_flag'].resample('D').apply(lambda x: x.mode()[0] if not x.empty else 9)
    df_out['SSL_flag'] = 0 # Initialize
    df_out.loc[df_out['SSL'].isna(), 'SSL_flag'] = 9
    df_out.loc[df_out['SSL'] < 0, 'SSL_flag'] = 3

    return df_out

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

        df_qc, envelope = apply_tool_qc(df_truncated)

        # -----------------------------
        # SSC–Q diagnostic plot
        # -----------------------------
        if envelope is not None:
            fig_path = os.path.join(
                args.output_dir,
                "diagnostic",
                f"SSC_Q_{info['Source_ID']}.png"
            )
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)

            plot_ssc_q_diagnostic(
                time=df_qc.index.values,
                Q=df_qc['Q'].values,
                SSC=df_qc['SSC'].values,
                Q_flag=df_qc['Q_flag'].values,
                SSC_flag=df_qc['SSC_flag'].values,
                ssc_q_bounds=envelope,
                station_id=info['Source_ID'],
                station_name=info['station_name'],
                out_png=fig_path
            )

        df_final = process_and_convert(df_qc)


        if df_final[['Q', 'SSC']].isna().all().all():
            print("  Skipping station as all Q and SSC data are NaN after processing.")
            continue

        # Create NetCDF
        output_nc_path = os.path.join(args.output_dir, f"Robotham_{info['Source_ID']}.nc")
        create_netcdf(df_final, info, output_nc_path, history_log)

        # errors, warnings_nc = check_nc_completeness(output_nc_path, strict=False)

        # if errors:
        #     print(f"  ✗ NetCDF completeness check failed for {info['Source_ID']}")
        #     for e in errors:
        #         print(f"    ERROR: {e}")
        #     os.remove(output_nc_path)
        #     continue


        # Collect summary stats
        station_summaries.append(calculate_summary_stats(df_final, info))

    # Generate summary CSV
    generate_summary_csv(station_summaries, args.output_dir)

    print("\n" + "=" * 80)
    print(f"Processing complete. {len(station_summaries)} stations processed.")
    print("=" * 80)

if __name__ == '__main__':
    main()
