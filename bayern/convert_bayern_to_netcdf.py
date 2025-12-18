#!/usr/bin/env python3
"""
Convert Bayern sediment and discharge data to NetCDF format.

NOTE: This script ONLY processes DAILY average data (tmw = Tagesmittelwert).
      Other time resolutions (e.g., ezw = Einzelwert) are NOT processed.

Data source: https://www.gkd.bayern.de/en/rivers/discharge and
             https://www.gkd.bayern.de/en/rivers/suspended-sediment
"""

import os
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings
from pyproj import Transformer

warnings.filterwarnings('ignore')


def parse_bayern_csv(filepath, data_type='discharge'):
    """
    Parse Bayern CSV files with metadata headers.

    Parameters:
    -----------
    filepath : str
        Path to CSV file
    data_type : str
        Either 'discharge' or 'sediment'

    Returns:
    --------
    data : pd.DataFrame
        Time series data
    metadata : dict
        Station metadata
    """
    # Read metadata from header
    metadata = {}
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i in range(10):
            line = f.readline().strip()
            if 'Messstellen-Name' in line:
                metadata['station_name'] = line.split(';')[1].strip('"')
            elif 'Messstellen-Nr' in line:
                metadata['station_id'] = line.split(';')[1].strip('"')
            elif 'Gewässer' in line:
                metadata['river_name'] = line.split(';')[1].strip('"')
            elif 'Ostwert' in line:
                parts = line.split(';')
                metadata['easting'] = float(parts[1])
                metadata['northing'] = float(parts[3])
            elif 'Pegelnullpunktshöhe' in line:
                parts = line.split(';')[1].strip('"').split()
                try:
                    metadata['altitude'] = float(parts[0].replace(',', '.'))
                except:
                    metadata['altitude'] = np.nan

    # Read data section
    # Find the header line
    skiprows = None
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for i, line in enumerate(f):
            if ('Datum' in line and 'Mittelwert' in line) or ('Zeitpunkt' in line and 'Konzentration' in line):
                skiprows = i
                break

    if skiprows is None:
        print(f"Could not find data header in {filepath}")
        return None, metadata

    # Read the data
    try:
        df = pd.read_csv(filepath, sep=';', skiprows=skiprows, encoding='utf-8-sig')

        # Get column names
        date_col = df.columns[0]

        # For sediment data, use concentration column
        if 'Konzentration' in filepath or 'ssp' in filepath or data_type == 'sediment':
            # Find concentration column
            value_col = None
            for col in df.columns:
                if 'Konzentration' in col:
                    value_col = col
                    break
            if value_col is None:
                value_col = df.columns[1]
        else:
            # For discharge, use Mittelwert column
            value_col = df.columns[1]

        # Clean up
        df = df[[date_col, value_col]].copy()
        df.columns = ['date', 'value']

        # Convert date column to datetime
        # Try parsing as string first
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d', errors='coerce')

        # Remove rows where date parsing failed
        df = df.dropna(subset=['date'])

        # Convert to numeric, replacing commas with dots
        df['value'] = pd.to_numeric(df['value'].astype(str).str.replace(',', '.'),
                                     errors='coerce')

        # Set date as index
        df.set_index('date', inplace=True)

        return df, metadata

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None, metadata


def utm_to_latlon(easting, northing, zone=32):
    """
    Convert UTM coordinates to latitude/longitude.

    Parameters:
    -----------
    easting : float
        UTM easting coordinate
    northing : float
        UTM northing coordinate
    zone : int
        UTM zone (default: 32 for Bavaria)

    Returns:
    --------
    lat, lon : float
        Latitude and longitude in decimal degrees
    """
    # Create transformer from UTM Zone 32N to WGS84
    transformer = Transformer.from_crs(f"EPSG:326{zone}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def process_station(station_id, discharge_dir, sediment_dir, output_dir):
    """
    Process a single station: combine all files, find overlap, create NetCDF.

    Note: This function ONLY processes daily average (tmw) data files.
    Files with other time resolutions (e.g., ezw) are explicitly excluded.

    Parameters:
    -----------
    station_id : str
        Station ID
    discharge_dir : str
        Directory containing discharge files
    sediment_dir : str
        Directory containing sediment files
    output_dir : str
        Output directory for NetCDF files

    Returns:
    --------
    success : bool
        True if NetCDF file was created successfully
    """
    print(f"\nProcessing station {station_id}...")

    # IMPORTANT: Only process daily average (tmw = Tagesmittelwert) files
    # Exclude other time resolutions like ezw (Einzelwert)
    discharge_files = sorted(glob.glob(os.path.join(discharge_dir, f"{station_id}_*_tmw_*.csv")))
    sediment_files = sorted(glob.glob(os.path.join(sediment_dir, f"{station_id}_*_tmw_*.csv")))

    if not discharge_files:
        print(f"  No discharge data found for station {station_id}")
        return False

    if not sediment_files:
        print(f"  No sediment data found for station {station_id}")
        return False

    # Combine all discharge files
    discharge_dfs = []
    metadata = None

    for f in discharge_files:
        df, meta = parse_bayern_csv(f, 'discharge')
        if df is not None and not df.empty:
            discharge_dfs.append(df)
            if metadata is None:
                metadata = meta

    if not discharge_dfs:
        print(f"  Failed to read discharge data for station {station_id}")
        return False

    discharge_data = pd.concat(discharge_dfs).sort_index()
    discharge_data = discharge_data[~discharge_data.index.duplicated(keep='first')]

    # Combine all sediment files
    sediment_dfs = []

    for f in sediment_files:
        df, _ = parse_bayern_csv(f, 'sediment')
        if df is not None and not df.empty:
            sediment_dfs.append(df)

    if not sediment_dfs:
        print(f"  Failed to read sediment data for station {station_id}")
        return False

    sediment_data = pd.concat(sediment_dfs).sort_index()
    sediment_data = sediment_data[~sediment_data.index.duplicated(keep='first')]

    # Check if either dataset is all NaN
    if discharge_data['value'].isna().all():
        print(f"  Discharge data is all NaN for station {station_id}")
        return False

    if sediment_data['value'].isna().all():
        print(f"  Sediment data is all NaN for station {station_id}")
        return False

    # Find overlapping period
    discharge_start = discharge_data.index.min()
    discharge_end = discharge_data.index.max()
    sediment_start = sediment_data.index.min()
    sediment_end = sediment_data.index.max()

    # Get the intersection
    overlap_start = max(discharge_start, sediment_start)
    overlap_end = min(discharge_end, sediment_end)

    if overlap_start > overlap_end:
        print(f"  No temporal overlap for station {station_id}")
        print(f"    Discharge: {discharge_start} to {discharge_end}")
        print(f"    Sediment: {sediment_start} to {sediment_end}")
        return False


    print(f"  Overlap period: {overlap_start} to {overlap_end}")

    # Filter data to overlap period
    discharge_data = discharge_data.loc[overlap_start:overlap_end]
    sediment_data = sediment_data.loc[overlap_start:overlap_end]

    # 按真实观测日期对齐
    # merged_data = pd.DataFrame(index=discharge_data.index.union(sediment_data.index))
    # merged_data = merged_data.loc[overlap_start:overlap_end]

    # merged_data['discharge'] = discharge_data['value']
    # merged_data['ssc'] = sediment_data['value']

    #只保留Q和SSC都存在的日期【need_check】
    merged_data = discharge_data[['value']].rename(columns={'value': 'discharge'}).join(
    sediment_data[['value']].rename(columns={'value': 'ssc'}),
    how='inner')
    merged_data = merged_data.loc[overlap_start:overlap_end]

    # Calculate sediment load (ton/day)
    # Load = Q (m³/s) × SSC (g/m³) × 86400 (s/day) / 1e6 (g/ton)
    # Load = Q × SSC × 0.0864
    merged_data['sediment_load'] = merged_data['discharge'] * merged_data['ssc'] * 0.0864

    # Convert coordinates
    if 'easting' in metadata and 'northing' in metadata:
        lat, lon = utm_to_latlon(metadata['easting'], metadata['northing'])
    else:
        lat, lon = np.nan, np.nan

    # Create NetCDF file
    output_file = os.path.join(output_dir, f"Bayern_{station_id}.nc")

    try:
        create_netcdf(output_file, merged_data, metadata, lat, lon)
        print(f"  Created {output_file}")
        print(f"    {len(merged_data)} time steps")
        print(f"    Discharge: {(~merged_data['discharge'].isna()).sum()} valid values")
        print(f"    SSC: {(~merged_data['ssc'].isna()).sum()} valid values")
        return True

    except Exception as e:
        print(f"  Error creating NetCDF for station {station_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_netcdf(filename, data, metadata, lat, lon):
    """
    Create NetCDF file following HYBAM format.

    Parameters:
    -----------
    filename : str
        Output NetCDF filename
    data : pd.DataFrame
        Time series data with columns: discharge, ssc, sediment_load
    metadata : dict
        Station metadata
    lat, lon : float
        Station coordinates
    """
    # Create NetCDF file
    dataset = nc.Dataset(filename, 'w', format='NETCDF4')

    # Create dimensions
    time_dim = dataset.createDimension('time', len(data))

    # Create coordinate variables
    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time of measurement'
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'

    # Convert dates to days since 1970-01-01
    reference_date = pd.Timestamp('1970-01-01')
    time_var[:] = [(d - reference_date).total_seconds() / 86400.0 for d in data.index]

    # Create scalar coordinate variables
    lat_var = dataset.createVariable('latitude', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = lat

    lon_var = dataset.createVariable('longitude', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = lon

    alt_var = dataset.createVariable('altitude', 'f4')
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station altitude above sea level'
    alt_var.units = 'm'
    alt_var[:] = metadata.get('altitude', np.nan)

    # Note: upstream_area not available in Bayern data
    # Creating variable but setting to NaN
    area_var = dataset.createVariable('upstream_area', 'f4')
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var.comment = 'Not available in source data'
    area_var[:] = np.nan

    # Create data variables
    discharge_var = dataset.createVariable('discharge', 'f4', ('time',),
                                           fill_value=-9999.0, zlib=True)
    discharge_var.standard_name = 'water_volume_transport_in_river_channel'
    discharge_var.long_name = 'river discharge'
    discharge_var.units = 'm3 s-1'
    discharge_var.coordinates = 'time latitude longitude'
    discharge_var[:] = data['discharge'].fillna(-9999.0).values

    ssc_var = dataset.createVariable('ssc', 'f4', ('time',),
                                     fill_value=-9999.0, zlib=True)
    ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.units = 'mg L-1'
    ssc_var.coordinates = 'time latitude longitude'
    ssc_var.comment = 'Original data in g/m³, which equals mg/L'
    ssc_var[:] = data['ssc'].fillna(-9999.0).values

    load_var = dataset.createVariable('sediment_load', 'f4', ('time',),
                                      fill_value=-9999.0, zlib=True)
    load_var.long_name = 'suspended sediment load'
    load_var.units = 'ton day-1'
    load_var.coordinates = 'time latitude longitude'
    load_var.comment = 'Calculated as: Load = Q × SSC × 0.0864 (Q in m³/s, SSC in g/m³, Load in ton/day)'
    load_var[:] = data['sediment_load'].fillna(-9999.0).values

    # Global attributes
    dataset.Conventions = 'CF-1.8'
    dataset.title = f"Bayern Sediment and Discharge Data for Station {metadata.get('station_id', 'Unknown')}"
    dataset.institution = 'Bayerisches Landesamt für Umwelt'
    dataset.source = 'In-situ observations from Bayern monitoring network'
    dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by convert_bayern_to_netcdf.py"
    dataset.references = 'https://www.gkd.bayern.de/en/rivers/discharge; https://www.gkd.bayern.de/en/rivers/suspended-sediment'
    dataset.comment = 'Daily average values. Sediment load calculated as: Load = Q × SSC × 0.0864 (Q in m³/s, SSC in g/m³, Load in ton/day)'
    dataset.station_id = metadata.get('station_id', '')
    dataset.station_name = metadata.get('station_name', '')
    dataset.river_name = metadata.get('river_name', '')

    # Close the file
    dataset.close()


def main():
    """
    Main processing function.

    Note: This script ONLY processes DAILY data (tmw files).
    Other time resolutions (e.g., ezw files) are NOT processed.
    """

    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    discharge_dir = os.path.join(base_dir, 'discharge')
    sediment_dir = os.path.join(base_dir, 'ssp')
    output_dir = os.path.join(base_dir, 'done')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all unique station IDs from DAILY (tmw) discharge files only
    # tmw = Tagesmittelwert (daily average)
    # Other file types (e.g., ezw = Einzelwert) are excluded
    discharge_files = glob.glob(os.path.join(discharge_dir, '*_tmw_*.csv'))
    discharge_ids = set([os.path.basename(f).split('_')[0] for f in discharge_files])

    # Get all unique station IDs from DAILY (tmw) sediment files only
    sediment_files = glob.glob(os.path.join(sediment_dir, '*_tmw_*.csv'))
    sediment_ids = set([os.path.basename(f).split('_')[0] for f in sediment_files])

    # Process only stations that have both discharge and sediment data
    common_ids = discharge_ids.intersection(sediment_ids)

    print("="*60)
    print("NOTE: Processing DAILY data only (tmw files)")
    print("      Other time resolutions (ezw files) are excluded")
    print("="*60)
    print(f"Found {len(discharge_ids)} stations with DAILY discharge data")
    print(f"Found {len(sediment_ids)} stations with DAILY sediment data")
    print(f"Found {len(common_ids)} stations with both DAILY datasets")

    # Process each station
    success_count = 0
    failed_count = 0

    for station_id in sorted(common_ids):
        success = process_station(station_id, discharge_dir, sediment_dir, output_dir)
        if success:
            success_count += 1
        else:
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {success_count} stations")
    print(f"  Failed/skipped: {failed_count} stations")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
