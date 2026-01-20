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
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    # check_nc_completeness,
    # add_global_attributes
)

# --- Absolute paths (WSL format) ---
SOURCE_DIR = "C:\\Users\\fzjxw\\Desktop\\sediment_wzx_1111\\Source\\RiverSed"

OUTPUT_DIR = 'C:\\Users\\fzjxw\\Desktop\\sediment_wzx_1111\\Output_r\\daily\\RiverSed\\nc'

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
    QC for satellite-only SSC (TSS) data using tool.py

    Rules:
    - Physical validity (apply_quality_flag)
    - log-IQR outlier screening
    - NO SSC–Q consistency (no discharge)
    """

    ssc = df['tss'].values.astype(float)

    # -----------------------------
    # 1. Physical QC
    # -----------------------------
    ssc_flag = np.array(
        [apply_quality_flag(v, "SSC") for v in ssc],
        dtype=np.int8
    )

    # -----------------------------
    # 2. log-IQR screening
    # -----------------------------
    lower, upper = compute_log_iqr_bounds(ssc)

    if lower is not None:
        outlier = (ssc < lower) | (ssc > upper)
        ssc_flag[outlier] = 2  # suspect

    # -----------------------------
    # 3. Mask bad data
    # -----------------------------
    ssc_clean = ssc.copy()
    ssc_clean[ssc_flag == 3] = np.nan  # bad → NaN

    df['tss'] = ssc_clean
    df['SSC_flag'] = ssc_flag

    # 至少保留一个非缺测值
    if np.all(np.isnan(df['tss'].values)):
        print(f"  -> All SSC invalid after QC for station {station_id}")
        return None

    # Generate diagnostic plot using plot_ssc_q_diagnostic
    # Note: For satellite data without Q, ssc_q_bounds will be None and plot will be skipped
    if diagnostic_dir is not None:
        try:
            diagnostic_dir = Path(diagnostic_dir)
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            
            # For satellite data without Q, we cannot generate SSC-Q consistency plots
            # since ssc_q_bounds = None causes plot_ssc_q_diagnostic to return early
            # This is expected behavior for satellite-only data
            pass
            
        except Exception as e:
            print(f"  Warning: Failed to create diagnostic plot for {station_id}: {e}")

    return df


def create_netcdf_file(station_id, tss_df, output_dir):
    """Create netCDF file following HYBAM format"""

    # Check if all TSS values are NaN
    if tss_df['tss'].isna().all():
        print(f"  All TSS values are NaN for station {station_id}")
        return False

    # Find time period
    start_date, end_date = find_overlap_period(tss_df['date'])

    if start_date is None:
        print(f"  No valid dates for station {station_id}")
        return False

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
        return False


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

        # errors, warnings_nc = check_nc_completeness(output_file, strict=False)

        # if errors:
        #     print(f"  ✗ Completeness check failed for {station_id}")
        #     for e in errors:
        #         print(f"    ERROR: {e}")
        #     os.remove(output_file)
        #     return False

        # if warnings_nc:
        #     for w in warnings_nc:
        #         print(f"    WARNING: {w}")


    print(f"  Created {output_file}")
    return True

def main():
    # Configuration with WSL absolute paths
    aquasat_file = os.path.join(SOURCE_DIR, 'Aquasat_TSS_v1.1.csv')
    riversed_file = os.path.join(SOURCE_DIR, 'RiverSed_USA_V1.1.txt')
    output_dir = OUTPUT_DIR

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

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
            executor.submit(create_netcdf_file, station_id, aquasat_df[aquasat_df['station_id'] == station_id].copy(), output_dir): station_id
            for station_id in aquasat_stations
        }

        for i, future in enumerate(as_completed(futures), 1):
            station_id = futures[future]
            try:
                result = future.result()
                if result:
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
            executor.submit(create_netcdf_file, station_id, riversed_df[riversed_df['station_id'] == station_id].copy(), output_dir): station_id
            for station_id in riversed_stations
        }

        for i, future in enumerate(as_completed(futures), 1):
            station_id = futures[future]
            try:
                result = future.result()
                if result:
                    riversed_success += 1
                else:
                    riversed_failed += 1
            except Exception as e:
                print(f"  Station {station_id} failed with error: {e}")
                riversed_failed += 1

            if i % 1000 == 0:
                print(f"  Processed {i}/{len(riversed_stations)} stations...")


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
