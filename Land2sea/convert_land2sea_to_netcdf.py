#!/usr/bin/env python3
"""
Convert Land2Sea database to netCDF format
Each station gets one file with annual average data
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from code.constants import FILL_VALUE_FLOAT

def parse_land2sea_data(filepath):
    """
    Parse the Land2Sea data file
    """
    # Read with proper encoding and parse header
    with open(filepath, 'r', encoding='latin1') as f:
        header_line = f.readline().strip().split('\t')

    # Read data
    df = pd.read_csv(filepath, sep='\t', skiprows=2, encoding='latin1', header=None)
    df.columns = header_line

    return df

def extract_station_data(row):
    """
    Extract relevant data for a single station
    """
    station = {}

    # Basic info
    station['river'] = row.iloc[0] if len(row) > 0 else None  # RIVER
    station['id'] = row.iloc[1] if len(row) > 1 else None     # ID
    station['country'] = row.iloc[2] if len(row) > 2 else None  # COUNTRY

    # Drainage area (km²) - column 3 (Ad)
    try:
        area_val = row.iloc[3] if len(row) > 3 else None
        if area_val is not None and pd.notna(area_val) and str(area_val).strip() != '':
            station['area'] = float(str(area_val).replace('"', '').replace(',', ''))
        else:
            station['area'] = None
    except:
        station['area'] = None

    # Sediment flux - column 24 (Sb)
    # Mt/yr -> ton/day: 1 Mt/yr = 1e6 ton/yr = 1e6/365.25 ton/day
    station['sediment_mt_yr'] = None
    station['sediment_load'] = None
    try:
        sediment_val = row.iloc[24] if len(row) > 24 else None
        if sediment_val is not None and pd.notna(sediment_val) and str(sediment_val).strip() != '':
            station['sediment_mt_yr'] = float(str(sediment_val).replace('"', '').replace(',', ''))
            station['sediment_load'] = station['sediment_mt_yr'] * 1e6 / 365.25
    except:
        pass

    # Water discharge - column 37 (Qb)
    # km³/yr -> m³/s: 1 km³/yr = 1e9 m³/yr = 1e9/(365.25*86400) m³/s
    station['discharge_km3_yr'] = None
    station['discharge'] = None
    try:
        discharge_val = row.iloc[37] if len(row) > 37 else None
        if discharge_val is not None and pd.notna(discharge_val) and str(discharge_val).strip() != '':
            station['discharge_km3_yr'] = float(str(discharge_val).replace('"', '').replace(',', ''))
            station['discharge'] = station['discharge_km3_yr'] * 1e9 / (365.25 * 86400)
    except:
        pass

    # Reference number (column 4 and 25 for sediment, 38 for discharge)
    try:
        ref_val = row.iloc[4] if len(row) > 4 else None
        station['ref'] = str(ref_val) if pd.notna(ref_val) else None
    except:
        station['ref'] = None

    return station

def get_reference_string(ref_num, readme_path):
    """
    Extract reference string from README based on reference number
    """
    if ref_num is None or pd.isna(ref_num):
        return None

    # Parse reference number (might be like "1, 40" or "43, 51")
    ref_str = str(ref_num).replace('"', '').replace(' ', '')
    ref_numbers = ref_str.split(',') if ',' in ref_str else [ref_str]

    # For now, return a placeholder - in production, parse the readme
    # TODO: Parse actual references from readme file
    return f"Reference {ref_str} from Land2Sea database"

def create_netcdf_file(station, output_dir, reference1, reference2=None, row_idx=None):
    """
    Create netCDF file for a station following HYBAM format
    """
    # Check if we have both sediment and discharge data
    if station['sediment_load'] is None or station['discharge'] is None:
        return None

    if np.isnan(station['sediment_load']) or np.isnan(station['discharge']):
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # File name based on unique row index (since ID is not unique - it's the drainage region)
    # Use row index as unique identifier
    if row_idx is not None:
        station_id = f"{row_idx:04d}"
    else:
        station_id = str(station['id']).replace('"', '').strip()
    filename = f"Land2Sea_{station_id}.nc"
    filepath = os.path.join(output_dir, filename)

    # Create netCDF file
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # Create dimensions
    # For annual average data, we use a single time point
    # Representing the average for the period
    time_dim = ds.createDimension('time', 1)

    # Global attributes
    ds.Conventions = 'CF-1.8'
    ds.title = f'Land2Sea Sediment and Discharge Data for Station {station_id}'
    ds.institution = 'Land2Sea Database (Peucker-Ehrenbrink, 2009)'
    ds.source = 'Annual average estimates from literature compilation'
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_land2sea_to_netcdf.py'
    ds.references = reference1
    if reference2:
        ds.references += f'; {reference2}'
    ds.comment = 'Annual average sediment flux and discharge. Sediment load calculated from annual flux estimates. Values represent long-term averages from various sources.'
    ds.station_id = station_id
    ds.station_name = station['river']
    ds.river_name = station['river']

    # Create coordinate variables
    # Latitude and longitude (placeholder - need to add actual coordinates)
    lat_var = ds.createVariable('latitude', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = np.nan  # Placeholder - actual coordinates not in Land2Sea database

    lon_var = ds.createVariable('longitude', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = np.nan  # Placeholder

    # Altitude
    alt_var = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station altitude above sea level'
    alt_var.units = 'm'
    alt_var[:] = FILL_VALUE_FLOAT  # Not available in Land2Sea

    # Upstream area
    area_var = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE_FLOAT)
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var.comment = 'Best estimate of drainage basin area from Land2Sea database'
    if station['area'] is not None:
        try:
            area_val = float(str(station['area']).replace('"', '').replace(',', ''))
            area_var[:] = area_val if np.isfinite(area_val) else FILL_VALUE_FLOAT
        except:
            area_var[:] = FILL_VALUE_FLOAT
    else:
        area_var[:] = FILL_VALUE_FLOAT

    # Time variable
    # Use a reference time and represent as days since epoch
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time of measurement'
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'
    # Use year 2000 as representative time for annual averages
    ref_time = datetime(1970, 1, 1)
    data_time = datetime(2000, 1, 1)
    days_since = (data_time - ref_time).days
    time_var[:] = days_since

    # Discharge
    q_var = ds.createVariable('discharge', 'f4', ('time',), fill_value=np.float32(-9999.0))
    q_var.standard_name = 'water_volume_transport_in_river_channel'
    q_var.long_name = 'river discharge'
    q_var.units = 'm3 s-1'
    q_var.coordinates = 'time latitude longitude'
    q_var.comment = f'Annual average discharge: {station["discharge_km3_yr"]:.2f} km3/yr'
    q_var[:] = station['discharge']

    # Sediment load
    sl_var = ds.createVariable('sediment_load', 'f4', ('time',), fill_value=np.float32(-9999.0))
    sl_var.long_name = 'suspended sediment load'
    sl_var.units = 'ton day-1'
    sl_var.coordinates = 'time latitude longitude'
    sl_var.comment = f'Annual average sediment flux: {station["sediment_mt_yr"]:.2f} Mt/yr'
    sl_var[:] = station['sediment_load']

    # SSC (suspended sediment concentration)
    # SSC (mg/L) = Sediment_load (ton/day) / Discharge (m³/s) / 0.0864
    # Derivation:
    #   Sediment: ton/day = 10^9 mg/day
    #   Discharge: m³/s = 86400 m³/day = 86,400,000 L/day
    #   SSC = (10^9 mg/day) / (86,400,000 L/day) = 10^9/86,400,000 = 1000/86.4 mg/L
    #   Therefore: SSC = Sediment_load / Discharge / 0.0864
    if station['discharge'] > 0:
        ssc = station['sediment_load'] / station['discharge'] / 0.0864
    else:
        ssc = np.nan

    ssc_var = ds.createVariable('ssc', 'f4', ('time',), fill_value=np.float32(-9999.0))
    ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.units = 'mg L-1'
    ssc_var.coordinates = 'time latitude longitude'
    ssc_var.comment = 'Calculated from annual average sediment load and discharge'
    ssc_var[:] = ssc

    # Close the file
    ds.close()

    return filepath

def main():
    """
    Main conversion function
    """
    # File paths
    data_file = 'ggge1494-sup-0002-ds01.txt'
    readme_file = 'ggge1494-sup-0001-readme.txt'
    output_dir = 'done'

    # Reference 1 (primary)
    reference1 = 'Peucker-Ehrenbrink, B. (2009), Land2Sea database of river drainage basin sizes, annual water discharges, and suspended sediment fluxes, Geochem. Geophys. Geosyst., 10, Q06014, doi:10.1029/2008GC002356.'

    print('Reading Land2Sea data...')
    df = parse_land2sea_data(data_file)
    print(f'Total stations: {len(df)}')

    # Process each station
    created_files = []
    skipped_stations = []

    for idx, row in df.iterrows():
        try:
            station = extract_station_data(row)

            # Get secondary reference
            reference2 = get_reference_string(station['ref'], readme_file)

            # Create netCDF file with row index as unique ID
            output_file = create_netcdf_file(station, output_dir, reference1, reference2, row_idx=idx)

            if output_file:
                created_files.append(output_file)
                if len(created_files) % 50 == 0:  # Print every 50 files
                    print(f'Created {len(created_files)} files...')
            else:
                skipped_stations.append(station['river'])
        except Exception as e:
            print(f'Error processing row {idx} ({station.get("river", "unknown") if "station" in locals() else "unknown"}): {str(e)}')
            continue

    print(f'\nProcessing complete!')
    print(f'Created {len(created_files)} netCDF files')
    print(f'Skipped {len(skipped_stations)} stations (missing data)')

    if len(skipped_stations) > 0 and len(skipped_stations) < 20:
        print(f'\nSkipped stations: {", ".join(skipped_stations[:20])}')

if __name__ == '__main__':
    main()
