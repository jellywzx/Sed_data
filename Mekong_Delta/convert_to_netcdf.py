#!/usr/bin/env python3
"""
Convert Vietnamese Mekong Delta sediment and discharge data to NetCDF format.
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


def read_fluxes_file(filepath):
    """
    Read fluxes CSV file (wide format) and convert to long format.
    Values are in Mt/day (megatons per day), will convert to ton/day.
    """
    df = pd.read_csv(filepath)

    # Melt from wide to long format
    df_long = df.melt(id_vars=['Date'], var_name='Year', value_name='Flux_Mt_day')

    # Parse date
    df_long['Date_str'] = df_long['Date'] + '-' + df_long['Year']
    df_long['Date_parsed'] = pd.to_datetime(df_long['Date_str'], format='%d-%b-%Y', errors='coerce')

    # Convert Mt/day to ton/day
    df_long['sediment_load'] = df_long['Flux_Mt_day'] * 1e6  # Mt to ton

    # Create year/month/day columns
    df_long['year'] = df_long['Date_parsed'].dt.year
    df_long['month'] = df_long['Date_parsed'].dt.month
    df_long['day'] = df_long['Date_parsed'].dt.day

    # Remove NaN dates and flux values
    df_long = df_long.dropna(subset=['Date_parsed', 'sediment_load'])

    return df_long[['Date_parsed', 'year', 'month', 'day', 'sediment_load']].sort_values('Date_parsed')


def read_ratings_file(filepath):
    """
    Read ratings CSV file and aggregate to daily discharge and SSC.
    """
    df = pd.read_csv(filepath)

    # Fix Year column - some files have "10" instead of "2010"
    df.loc[df['Year'] == 10, 'Year'] = 2010
    df.loc[df['Year'] < 100, 'Year'] = df.loc[df['Year'] < 100, 'Year'] + 2000

    # Create datetime column
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Group by date and take mean of discharge and SSC
    daily = df.groupby('Date').agg({
        'Discharge (m3/s)': 'mean',
        'Section Averaged SSC (mg/l)': 'mean',
        'Sediment Flux (kg/s)': 'mean'
    }).reset_index()

    daily['year'] = daily['Date'].dt.year
    daily['month'] = daily['Date'].dt.month
    daily['day'] = daily['Date'].dt.day

    # Rename columns
    daily = daily.rename(columns={
        'Discharge (m3/s)': 'discharge',
        'Section Averaged SSC (mg/l)': 'ssc_ratings',
        'Sediment Flux (kg/s)': 'flux_kg_s'
    })

    return daily


def calculate_ssc_from_load(discharge, sediment_load):
    """
    Calculate SSC from discharge and sediment load.
    Formula: Load (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864
    Derivation:
      Q (m³/s) × SSC (mg/L) = Q (m³/s) × SSC (g/m³) = Q × SSC (g/s)
      Convert to ton/day: Q × SSC (g/s) × 86400 (s/day) / (10^6 g/ton) = Q × SSC × 0.0864
    Therefore: SSC (mg/L) = Load (ton/day) / [Q (m³/s) × 0.0864]
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ssc = sediment_load / (discharge * 0.0864)
        ssc = np.where(np.isfinite(ssc), ssc, np.nan)
    return ssc


def process_station(station_id, data_dir, output_dir):
    """Process one station and create NetCDF file."""

    print(f"\nProcessing station: {station_id}")

    # File paths
    fluxes_file = os.path.join(data_dir, f'{station_id}fluxes.csv')
    ratings_file = os.path.join(data_dir, f'{station_id}ratings.csv')

    # Read data
    print(f"  Reading {fluxes_file}")
    fluxes_df = read_fluxes_file(fluxes_file)

    print(f"  Reading {ratings_file}")
    ratings_df = read_ratings_file(ratings_file)

    # Merge on date
    merged = pd.merge(fluxes_df, ratings_df,
                      left_on='Date_parsed', right_on='Date',
                      how='inner', suffixes=('_flux', '_rating'))

    if len(merged) == 0:
        print(f"  WARNING: No overlapping data between flux and ratings files for {station_id}")
        return None

    # Calculate SSC from sediment load and discharge
    merged['ssc'] = calculate_ssc_from_load(merged['discharge'], merged['sediment_load'])

    # Remove rows where either discharge or sediment_load is all NaN
    merged = merged.dropna(subset=['discharge', 'sediment_load'])

    if len(merged) == 0:
        print(f"  WARNING: All discharge or sediment_load values are NaN for {station_id}")
        return None

    # Find overlapping years (use year from ratings since it's more reliable)
    merged['year'] = merged['year_rating'] if 'year_rating' in merged.columns else merged['year_flux']
    years = sorted(merged['year'].unique())
    if len(years) == 0:
        print(f"  WARNING: No overlapping years for {station_id}")
        return None

    start_year = years[0]
    end_year = years[-1]

    print(f"  Overlapping years: {start_year} - {end_year}")
    print(f"  Total days: {len(merged)}")

    # Filter data from start of first year to end of last year
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Create complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    full_df = pd.DataFrame({'Date': date_range})

    # Merge with actual data - keep the date column from full_df
    final_df = pd.merge(full_df, merged[['Date_parsed', 'discharge', 'ssc', 'sediment_load']],
                        left_on='Date', right_on='Date_parsed',
                        how='left')

    # Drop the duplicate date column
    final_df = final_df.drop(columns=['Date_parsed'])

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
    ssc[:] = df['ssc'].fillna(-9999.0).values

    sediment_load = ds.createVariable('sediment_load', 'f4', ('time',),
                                     fill_value=-9999.0, zlib=True)
    sediment_load.long_name = 'suspended sediment load'
    sediment_load.units = 'ton day-1'
    sediment_load.coordinates = 'time latitude longitude'
    sediment_load[:] = df['sediment_load'].fillna(-9999.0).values

    # Global attributes
    ds.Conventions = 'CF-1.8'
    ds.title = f'Vietnamese Mekong Delta Sediment and Discharge Data for Station {station_meta["name"]}'
    ds.institution = 'Vietnamese Hydrological Agency'
    ds.source = 'In-situ observations from ADCP monitoring network'
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf.py'
    ds.references = 'Darby, S.E.; Hackney, C.R.; Parsons, D.R.; Tri, P.D.V. (2020). Water and suspended sediment discharges for the Mekong Delta, Vietnam (2005-2015) NERC Environmental Information Data Centre. https://doi.org/10.5285/ac5b28ca-e087-4aec-974a-5a9f84b06595'
    ds.comment = 'Sediment load calculated from fluxes data (converted from Mt/day to ton/day). SSC calculated as: SSC = Load / (Q × 0.0864) where Load is in ton/day, Q in m³/s, resulting in SSC in mg/L. Conversion factor 0.0864 = 86400 s/day / 10^6 g/ton'
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
