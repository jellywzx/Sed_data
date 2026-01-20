#!/usr/bin/env python3
"""
Convert Vanmaercke et al. sediment data to NetCDF format
Creates one NetCDF file per station following HYBAM format
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import os
import re
from pathlib import Path

# --- Absolute paths (WSL format) ---
SOURCE_DIR = '/mnt/d/sediment_wzx_1111/Source/Vanmaercke'
OUTPUT_DIR = '/mnt/d/sediment_wzx_1111/Output_r/annually_climatology/Vanmaercke/nc'

def parse_measurement_period(mp_str):
    """
    Parse measurement period string to extract start and end years
    Examples: '1973 - 1995', '1972 - 1979', 'N.A.'
    """
    if pd.isna(mp_str) or str(mp_str).strip() in ['N.A.', 'NA', '']:
        return None, None

    # Try to extract year range
    match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', str(mp_str))
    if match:
        return int(match.group(1)), int(match.group(2))

    # Try single year
    match = re.search(r'(\d{4})', str(mp_str))
    if match:
        year = int(match.group(1))
        return year, year

    return None, None

def clean_numeric_value(val):
    """Clean and convert numeric values from Excel"""
    if pd.isna(val):
        return np.nan

    # Remove any non-numeric characters except . and -
    val_str = str(val).strip()
    val_str = re.sub(r'[^\d.\-]', '', val_str)

    try:
        return float(val_str)
    except (ValueError, AttributeError):
        return np.nan

def create_middle_time(start_year, end_year):
    """
    Create single time point at the middle of the measurement period
    Time is in days since 1970-01-01
    """
    base_date = datetime(1970, 1, 1)

    # Calculate middle date (middle of the period)
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Middle point
    middle_date = start_date + (end_date - start_date) / 2

    days_since = (middle_date - base_date).days

    return days_since, middle_date

def calculate_sediment_load(sy_value, area_km2):
    """
    Calculate sediment load from sediment yield

    SY is in t/km²/y (tons per square kilometer per year)
    We need sediment_load in ton/day

    sediment_load = SY * Area / 365 (to get daily load)
    """
    if np.isnan(sy_value) or np.isnan(area_km2):
        return np.nan

    # Convert from t/km²/y to t/day
    daily_load = (sy_value * area_km2) / 365.25

    return daily_load

def create_netcdf_file(station_data, output_path):
    """
    Create NetCDF file for a single station following HYBAM format
    """
    # Parse measurement period
    start_year, end_year = parse_measurement_period(station_data['MP'])

    if start_year is None or end_year is None:
        return False, "No valid measurement period"

    # Clean numeric values
    lat = clean_numeric_value(station_data['Lat (°)'])
    lon = clean_numeric_value(station_data['Lon (°)'])
    area = clean_numeric_value(station_data['A (km²)'])
    sy = clean_numeric_value(station_data['SY\n(t/km²/y)'])

    # Check for valid data
    if np.isnan(lat) or np.isnan(lon):
        return False, "Invalid coordinates"

    if np.isnan(sy) or np.isnan(area):
        return False, "Invalid SY or area"

    # Create single time point (middle of measurement period)
    time_value, middle_date = create_middle_time(start_year, end_year)

    # Calculate sediment load
    sediment_load = calculate_sediment_load(sy, area)

    # Check if sediment load is NaN
    if np.isnan(sediment_load):
        return False, "sediment_load is NaN"

    # Create NetCDF file
    dataset = nc.Dataset(output_path, 'w', format='NETCDF4')

    try:
        # Create dimensions
        dataset.createDimension('time', 1)

        # Create time variable
        time_var = dataset.createVariable('time', 'f8', ('time',))
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var[:] = time_value

        # Create coordinate variables (scalar - no dimensions)
        lat_var = dataset.createVariable('latitude', 'f4')
        lat_var.units = 'degrees_north'
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'
        lat_var.assignValue(lat)

        lon_var = dataset.createVariable('longitude', 'f4')
        lon_var.units = 'degrees_east'
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'
        lon_var.assignValue(lon)

        # Create altitude variable (scalar - set to NaN since not provided)
        alt_var = dataset.createVariable('altitude', 'f4')
        alt_var.units = 'm'
        alt_var.long_name = 'altitude'
        alt_var.standard_name = 'altitude'
        alt_var.comment = 'Not available in source dataset'
        alt_var.assignValue(np.nan)

        # Create upstream area variable (scalar)
        area_var = dataset.createVariable('upstream_area', 'f4')
        area_var.units = 'km2'
        area_var.long_name = 'upstream drainage area'
        area_var.comment = 'Catchment area from source dataset, used in sediment_load calculation'
        area_var.assignValue(area)

        # Create sediment load variable
        sed_var = dataset.createVariable('sediment_load', 'f4', ('time',),
                                        fill_value=-9999.0)
        sed_var.units = 'ton day-1'
        sed_var.long_name = 'suspended sediment load'
        sed_var.comment = (f'Calculated from sediment yield (SY): '
                          f'sediment_load (ton/day) = SY (t/km²/y) * upstream_area (km²) / 365.25. '
                          f'Original SY = {sy:.4f} t/km²/y, Area = {area:.2f} km²')
        sed_var[0] = sediment_load

        # Create SSC variable (NaN since no discharge data)
        ssc_var = dataset.createVariable('ssc', 'f4', ('time',),
                                        fill_value=-9999.0)
        ssc_var.units = 'mg L-1'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.comment = ('Not available in source dataset. '
                          'Would require: SSC (mg/L) = sediment_load (ton/day) / discharge (m³/s) * conversion_factor')
        ssc_var[0] = np.nan

        # Create discharge variable (NaN since not available)
        q_var = dataset.createVariable('discharge', 'f4', ('time',),
                                      fill_value=-9999.0)
        q_var.units = 'm3 s-1'
        q_var.long_name = 'river discharge'
        q_var.comment = 'Not available in source dataset (Vanmaercke et al. 2014)'
        q_var[0] = np.nan

        # Add global attributes
        dataset.Conventions = 'CF-1.8'
        dataset.title = f"Sediment data for {station_data['River/Catchment Name']}"
        dataset.institution = 'Derived from Vanmaercke et al. (2014) dataset'
        dataset.source = 'sediment.xlsx from Vanmaercke et al. (2014)'

        # Add references
        ref1 = ('Vanmaercke, M., Poesen, J., Broeckx, J., & Nyssen, J. (2014). '
                'Sediment yield in Africa. Earth-Science Reviews, 136, 350-368. '
                'https://doi.org/10.1016/j.earscirev.2014.06.004')

        ref2 = str(station_data.get('Reference', 'N.A.'))

        dataset.references = f"{ref1}; Original data reference: {ref2}"

        # Add station information
        dataset.station_name = str(station_data.get('River/Catchment Name', 'N.A.'))
        dataset.station_location = str(station_data.get('Measuring location', 'N.A.'))
        dataset.station_country = str(station_data.get('Country', 'N.A.'))
        dataset.station_id = str(station_data['ID'])
        dataset.period = str(station_data['MP'])  # Add period as global attribute
        dataset.measurement_type = str(station_data.get('Type', 'N.A.'))
        dataset.data_quality = str(station_data.get('Data Quality', 'N.A.'))
        dataset.coordinate_quality = str(station_data.get('Coord. Quality', 'N.A.'))

        # Add processing info
        dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        dataset.comment = ('Data conversion method: '
                          'sediment_load (ton/day) = SY (t/km²/y) × upstream_area (km²) ÷ 365.25. '
                          'Source data contains average sediment yield (SY) over the measurement period. '
                          'Time variable represents the middle date of the measurement period. '
                          'Discharge and SSC data are not available in the original dataset (Vanmaercke et al. 2014).')

        return True, "Success"

    finally:
        dataset.close()

def main():
    """Main conversion process"""

    print("="*60)
    print("Converting Vanmaercke sediment data to NetCDF format")
    print("="*60)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read Excel file
    source_file = Path(SOURCE_DIR) / 'sediment.xlsx'
    print(f"\nReading {source_file}...")
    df = pd.read_excel(source_file, sheet_name='Table 1', skiprows=17)

    print(f"Total stations in dataset: {len(df)}")

    # Process each station
    success_count = 0
    skip_count = 0
    error_messages = {}

    for idx, row in df.iterrows():
        station_id = str(row['ID']).strip()

        # Create output filename
        output_file = output_dir / f"Vanmaercke_{station_id}.nc"

        # Create NetCDF file
        success, message = create_netcdf_file(row, output_file)

        if success:
            success_count += 1
            if success_count % 50 == 0:
                print(f"  Processed {success_count} stations...")
        else:
            skip_count += 1
            error_messages[station_id] = message
            # Remove file if it was partially created
            if output_file.exists():
                output_file.unlink()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total stations processed: {len(df)}")
    print(f"Successfully created: {success_count}")
    print(f"Skipped: {skip_count}")

    if skip_count > 0:
        print(f"\nSkipped stations by reason:")
        reason_counts = {}
        for station_id, reason in error_messages.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("="*60)

if __name__ == '__main__':
    main()
