#!/usr/bin/env python3
"""
Convert Vietnamese Mekong Delta sediment and discharge data to NetCDF format.
Uses ONLY ratings files (ADCP measurements).
Based on HYBAM data format example.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from code.constants import FILL_VALUE_FLOAT

# Station metadata
STATIONS = {
    'Cantho': {
        'name': 'Can Tho',
        'lat': 10.088109,
        'lon': 105.736458,
        'river': 'Mekong',
        'altitude': np.nan  # Not provided
    },
    'Chaudoc': {
        'name': 'Chau Doc',
        'lat': 10.708268,
        'lon': 105.134606,
        'river': 'Mekong',
        'altitude': np.nan
    },
    'Mythaun': {
        'name': 'My Thaun',
        'lat': 10.272038,
        'lon': 105.900920,
        'river': 'Mekong',
        'altitude': np.nan
    },
    'Tanchau': {
        'name': 'Tan Chau',
        'lat': 10.822642,
        'lon': 105.227879,
        'river': 'Mekong',
        'altitude': np.nan
    }
}


def read_ratings_file(filepath):
    """
    Read ratings CSV file and process to daily data.
    Returns daily averaged discharge, SSC, and sediment load.
    """
    df = pd.read_csv(filepath)

    # Fix Year column - some files have "10" instead of "2010"
    df.loc[df['Year'] == 10, 'Year'] = 2010
    df.loc[df['Year'] < 100, 'Year'] = df.loc[df['Year'] < 100, 'Year'] + 2000

    # Create datetime column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Convert Sediment Flux from kg/s to ton/day
    # kg/s × 86400 s/day / 1000 kg/ton = kg/s × 86.4 = ton/day
    df['sediment_load_ton_day'] = df['Sediment Flux (kg/s)'] * 86.4

    # Group by date and take mean of all measurements in a day
    daily = df.groupby('Date').agg({
        'Discharge (m3/s)': 'mean',
        'Section Averaged SSC (mg/l)': 'mean',
        'sediment_load_ton_day': 'mean'
    }).reset_index()

    # Rename columns
    daily = daily.rename(columns={
        'Discharge (m3/s)': 'discharge',
        'Section Averaged SSC (mg/l)': 'ssc',
        'sediment_load_ton_day': 'sediment_load'
    })

    # Add year, month, day columns
    daily['year'] = daily['Date'].dt.year
    daily['month'] = daily['Date'].dt.month
    daily['day'] = daily['Date'].dt.day

    return daily


def process_station(station_id, data_dir, output_dir):
    """Process one station and create NetCDF file."""

    print(f"\nProcessing station: {station_id}")

    # File paths
    ratings_file = os.path.join(data_dir, f'{station_id}ratings.csv')

    # Read data
    print(f"  Reading {ratings_file}")
    ratings_df = read_ratings_file(ratings_file)

    # Remove rows with NaN values in critical columns
    ratings_df = ratings_df.dropna(subset=['discharge', 'ssc', 'sediment_load'])

    if len(ratings_df) == 0:
        print(f"  WARNING: No valid data for {station_id}")
        return None

    # Check if all values are NaN
    if ratings_df['discharge'].isna().all() or ratings_df['sediment_load'].isna().all():
        print(f"  WARNING: All discharge or sediment_load values are NaN for {station_id}")
        return None

    # Find year range
    years = sorted(ratings_df['year'].unique())
    if len(years) == 0:
        print(f"  WARNING: No valid years for {station_id}")
        return None

    start_year = years[0]
    end_year = years[-1]

    print(f"  Year range: {start_year} - {end_year}")
    print(f"  Total measurement days: {len(ratings_df)}")

    # Filter data from start of first year to end of last year
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Create complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    full_df = pd.DataFrame({'Date': date_range})

    # Merge with actual data
    final_df = pd.merge(full_df, ratings_df[['Date', 'discharge', 'ssc', 'sediment_load']],
                        on='Date', how='left')

    # Create NetCDF file
    output_file = os.path.join(output_dir, f'Mekong_{station_id}.nc')
    create_netcdf(output_file, final_df, station_id, STATIONS[station_id])

    print(f"  Created: {output_file}")
    return output_file


def create_netcdf(filepath, df, station_id, station_meta):
    """Create NetCDF file following HYBAM format."""

    # Create dataset
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # Create dimensions
    time_dim = ds.createDimension('time', len(df))

    # Create coordinate variables
    times = ds.createVariable('time', 'f8', ('time',))
    times.standard_name = 'time'
    times.long_name = 'time of measurement'
    times.units = 'days since 1970-01-01 00:00:00'
    times.calendar = 'gregorian'
    times.axis = 'T'

    # Convert dates to days since 1970-01-01
    epoch = datetime(1970, 1, 1)
    times[:] = [(d - epoch).total_seconds() / 86400.0 for d in df['Date']]

    # Create scalar coordinate variables
    lat = ds.createVariable('latitude', 'f4')
    lat.standard_name = 'latitude'
    lat.long_name = 'station latitude'
    lat.units = 'degrees_north'
    lat.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat[:] = station_meta['lat']

    lon = ds.createVariable('longitude', 'f4')
    lon.standard_name = 'longitude'
    lon.long_name = 'station longitude'
    lon.units = 'degrees_east'
    lon.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon[:] = station_meta['lon']

    alt = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
    alt.standard_name = 'altitude'
    alt.long_name = 'station altitude above sea level'
    alt.units = 'm'
    if not np.isnan(station_meta['altitude']):
        alt[:] = station_meta['altitude']
    else:
        alt[:] = FILL_VALUE_FLOAT

    # Upstream area (not available)
    upstream_area = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE_FLOAT)
    upstream_area.long_name = 'upstream drainage area'
    upstream_area.units = 'km2'
    upstream_area.comment = 'Data not available'
    upstream_area[:] = FILL_VALUE_FLOAT

    # Create data variables
    discharge = ds.createVariable('discharge', 'f4', ('time',),
                                  fill_value=-9999.0, zlib=True)
    discharge.standard_name = 'water_volume_transport_in_river_channel'
    discharge.long_name = 'river discharge'
    discharge.units = 'm3 s-1'
    discharge.coordinates = 'time latitude longitude'
    discharge[:] = df['discharge'].fillna(-9999.0).values

    ssc = ds.createVariable('ssc', 'f4', ('time',),
                           fill_value=-9999.0, zlib=True)
    ssc.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    ssc.long_name = 'suspended sediment concentration'
    ssc.units = 'mg L-1'
    ssc.coordinates = 'time latitude longitude'
    ssc.comment = 'Derived from ADCP acoustic backscatter calibrated to physical samples'
    ssc[:] = df['ssc'].fillna(-9999.0).values

    sediment_load = ds.createVariable('sediment_load', 'f4', ('time',),
                                     fill_value=-9999.0, zlib=True)
    sediment_load.long_name = 'suspended sediment load'
    sediment_load.units = 'ton day-1'
    sediment_load.coordinates = 'time latitude longitude'
    sediment_load.comment = 'Converted from kg/s to ton/day (×86.4)'
    sediment_load[:] = df['sediment_load'].fillna(-9999.0).values

    # Global attributes
    ds.Conventions = 'CF-1.8'
    ds.title = f'Vietnamese Mekong Delta Sediment and Discharge Data for Station {station_meta["name"]}'
    ds.institution = 'Vietnamese Hydrological Agency'
    ds.source = 'In-situ observations from ADCP monitoring network'
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf_ratings_only.py'
    ds.references = 'Darby, S.E.; Hackney, C.R.; Parsons, D.R.; Tri, P.D.V. (2020). Water and suspended sediment discharges for the Mekong Delta, Vietnam (2005-2015) NERC Environmental Information Data Centre. https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595'
    ds.comment = 'Data derived from ADCP ratings files only. SSC values from acoustic backscatter calibrated to physical samples collected at Chau Doc (May 2017) and Can Tho (September 2017). Sediment load = Discharge × SSC × 0.0864, converted from original kg/s to ton/day (×86.4).'
    ds.station_id = station_id
    ds.station_name = station_meta['name']
    ds.river_name = station_meta['river']

    # Close dataset
    ds.close()


def main():
    """Main processing function."""

    # Directories
    data_dir = 'data'
    output_dir = 'done'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Converting Vietnamese Mekong Delta Data to NetCDF")
    print("Using ONLY ratings files (ADCP measurements)")
    print("=" * 60)

    # Process each station
    successful = []
    failed = []

    for station_id in STATIONS.keys():
        try:
            result = process_station(station_id, data_dir, output_dir)
            if result:
                successful.append(station_id)
            else:
                failed.append(station_id)
        except Exception as e:
            import traceback
            print(f"ERROR processing {station_id}: {e}")
            traceback.print_exc()
            failed.append(station_id)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {len(successful)} stations")
    for s in successful:
        print(f"  - {s}")

    if failed:
        print(f"\nFailed or skipped: {len(failed)} stations")
        for s in failed:
            print(f"  - {s}")

    print("\nDone!")


if __name__ == '__main__':
    main()
