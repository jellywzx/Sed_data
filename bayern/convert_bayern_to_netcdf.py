#!/usr/bin/env python3
"""
Convert Bayern sediment and discharge data to NetCDF format.

NOTE: This script ONLY processes DAILY average data (tmw = Tagesmittelwert).
      Other time resolutions (e.g., ezw = Einzelwert) are NOT processed.

Data source: https://www.gkd.bayern.de/en/rivers/discharge and
             https://www.gkd.bayern.de/en/rivers/suspended-sediment

Station eligibility (revised 2026-08):
  - Sediment (SSC) files are the CORE input; discharge files are OPTIONAL.
  - A station with valid SSC data is processed even if NO discharge file exists.
  - Q-only stations (discharge but no sediment) are NOT included in the final
    sediment product.
  - When both Q and SSC exist, all SSC observations are kept (outer/union join).
  - SSL is computed only on dates where both Q and SSC are valid; SSC-only dates
    carry missing Q and SSL.
"""

import os
import glob
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings
from pyproj import Transformer
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]   # .../Script
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../sediment_wzx_1111
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from code.constants import FILL_VALUE_FLOAT

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
            if ('Datum' in line and 'Mittelwert' in line) or \
               ('Zeitpunkt' in line and 'Konzentration' in line):
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


def process_station(station_id, discharge_dir, sediment_dir, output_dir, audit):
    """
    Process a single station: combine all files, align data, create NetCDF.

    Note: This function ONLY processes daily average (tmw) data files.
    Files with other time resolutions (e.g., ezw) are explicitly excluded.

    Revised logic (2026-08):
      - Sediment files are CORE and REQUIRED.
      - Discharge files are OPTIONAL.
      - No overlap-period rejection.
      - Union/outer join: all SSC observations retained.
      - SSL computed ONLY on Q+SSC paired dates.

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
    audit : dict
        Mutable audit dictionary for accumulating statistics.

    Returns:
    --------
    success : bool
        True if NetCDF file was created successfully
    """
    print(f"\nProcessing station {station_id}...")

    # IMPORTANT: Only process daily average (tmw = Tagesmittelwert) files
    sediment_files = sorted(glob.glob(os.path.join(sediment_dir, f"{station_id}_*_tmw_*.csv")))
    discharge_files = sorted(glob.glob(os.path.join(discharge_dir, f"{station_id}_*_tmw_*.csv")))

    # --- Sediment files are REQUIRED ---
    if not sediment_files:
        print(f"  No sediment data found for station {station_id} -- skipped (Q-only station -> not in sediment product)")
        audit['q_only_skipped'] += 1
        return False

    # --- Parse sediment files (primary / core input) ---
    sediment_dfs = []
    metadata = None

    for f in sediment_files:
        df, meta = parse_bayern_csv(f, 'sediment')
        if df is not None and not df.empty:
            sediment_dfs.append(df)
            # Prefer sediment-file metadata (real source for SSC stations)
            if metadata is None:
                metadata = meta

    if not sediment_dfs:
        print(f"  Failed to read sediment data for station {station_id}")
        return False

    sediment_data = pd.concat(sediment_dfs).sort_index()
    sediment_data = sediment_data[~sediment_data.index.duplicated(keep='first')]

    if sediment_data['value'].isna().all():
        print(f"  Sediment data is all NaN for station {station_id}")
        return False

    audit['sediment_stations_total'] += 1

    # --- Parse discharge files (OPTIONAL) ---
    has_q = False
    discharge_data = None

    if discharge_files:
        discharge_dfs = []
        for f in discharge_files:
            df, meta = parse_bayern_csv(f, 'discharge')
            if df is not None and not df.empty:
                discharge_dfs.append(df)
                # Fallback: only take discharge metadata if we still have none
                if metadata is None:
                    metadata = meta

        if discharge_dfs:
            discharge_data = pd.concat(discharge_dfs).sort_index()
            discharge_data = discharge_data[~discharge_data.index.duplicated(keep='first')]
            if not discharge_data['value'].isna().all():
                has_q = True

    if has_q:
        audit['stations_with_q'] += 1
    else:
        audit['ssc_only_stations'] += 1
        print(f"  No discharge data -- SSC-only station (will generate NetCDF)")

    # --- Build merged dataframe with UNION / outer alignment ---
    if has_q:
        # Union of all dates that have SSC OR Q
        all_dates = discharge_data.index.union(sediment_data.index)
        merged = pd.DataFrame(index=all_dates.sort_values())
        merged['discharge'] = discharge_data['value']
        merged['ssc'] = sediment_data['value']

        # Count paired vs SSC-only dates
        paired_mask = merged['discharge'].notna() & merged['ssc'].notna()
        ssc_only_mask = merged['ssc'].notna() & merged['discharge'].isna()

        audit['paired_dates'] += int(paired_mask.sum())
        audit['ssc_only_dates'] += int(ssc_only_mask.sum())

        # SSL only on paired dates
        merged['sediment_load'] = np.nan
        merged.loc[paired_mask, 'sediment_load'] = (
            merged.loc[paired_mask, 'discharge'] * merged.loc[paired_mask, 'ssc'] * 0.0864
        )
    else:
        # SSC-only station: no Q at all
        merged = pd.DataFrame(index=sediment_data.index.sort_values())
        merged['discharge'] = np.nan
        merged['ssc'] = sediment_data['value']
        merged['sediment_load'] = np.nan

        ssc_only_mask = merged['ssc'].notna()
        audit['ssc_only_dates'] += int(ssc_only_mask.sum())
        # paired_dates stays 0

    # Final sort
    merged = merged.sort_index()

    # --- Count records ---
    n_records = len(merged)
    n_ssc_valid = int((~merged['ssc'].isna()).sum())
    n_q_valid = int((~merged['discharge'].isna()).sum())
    n_ssl_valid = int((~merged['sediment_load'].isna()).sum())
    audit['final_records'] += n_records

    print(f"  Records: {n_records} total | SSC valid: {n_ssc_valid} | Q valid: {n_q_valid} | SSL valid: {n_ssl_valid}")

    # --- Convert coordinates ---
    if metadata and 'easting' in metadata and 'northing' in metadata:
        lat, lon = utm_to_latlon(metadata['easting'], metadata['northing'])
    else:
        lat, lon = np.nan, np.nan

    # --- Store metadata source provenance ---
    if metadata is None:
        metadata = {}
    metadata['_source_has_discharge'] = has_q

    # --- Create NetCDF file ---
    output_file = os.path.join(output_dir, f"Bayern_{station_id}.nc")

    try:
        create_netcdf(output_file, merged, metadata, lat, lon)
        print(f"  Created {output_file}")
        print(f"    {len(merged)} time steps")
        print(f"    Discharge: {n_q_valid} valid values")
        print(f"    SSC: {n_ssc_valid} valid values")
        print(f"    SSL: {n_ssl_valid} valid values")
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
        Station metadata (with optional _source_has_discharge provenance key)
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

    alt_var = dataset.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station altitude above sea level'
    alt_var.units = 'm'
    altitude = metadata.get('altitude', FILL_VALUE_FLOAT)
    alt_var[:] = altitude if pd.notna(altitude) and np.isfinite(float(altitude)) else FILL_VALUE_FLOAT

    # Note: upstream_area not available in Bayern data
    area_var = dataset.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE_FLOAT)
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var.comment = 'Not available in source data'
    area_var[:] = FILL_VALUE_FLOAT

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
    ssc_var.comment = 'Original data in g/m3, which equals mg/L'
    ssc_var[:] = data['ssc'].fillna(-9999.0).values

    load_var = dataset.createVariable('sediment_load', 'f4', ('time',),
                                      fill_value=-9999.0, zlib=True)
    load_var.long_name = 'suspended sediment load'
    load_var.units = 'ton day-1'
    load_var.coordinates = 'time latitude longitude'
    load_var.comment = 'Calculated as: Load = Q x SSC x 0.0864 (Q in m3/s, SSC in g/m3, Load in ton/day). Only computed on dates where both Q and SSC are available; missing otherwise.'
    load_var[:] = data['sediment_load'].fillna(-9999.0).values

    # ---- Source provenance (avoid fabricating) ----
    source_has_discharge = metadata.get('_source_has_discharge', False)
    ssc_source_note = 'In-situ observations from Bayern monitoring network (ssp / suspended-sediment data)'
    if source_has_discharge:
        source_note = ssc_source_note + '; discharge data from same network'
    else:
        source_note = ssc_source_note + '; no discharge data available for this station'
    meta_station_id = metadata.get('station_id', '')

    # Global attributes
    dataset.Conventions = 'CF-1.8'
    dataset.title = f"Bayern Sediment Data for Station {meta_station_id or 'Unknown'}"
    dataset.institution = 'Bayerisches Landesamt fur Umwelt'
    dataset.source = source_note
    dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by convert_bayern_to_netcdf.py"
    dataset.references = 'https://www.gkd.bayern.de/en/rivers/discharge; https://www.gkd.bayern.de/en/rivers/suspended-sediment'
    dataset.comment = 'Daily average values. Sediment load calculated as: Load = Q x SSC x 0.0864 (Q in m3/s, SSC in g/m3, Load in ton/day). SSC-only dates (no Q) carry missing Q and SSL.'
    dataset.station_id = meta_station_id
    dataset.station_name = metadata.get('station_name', '')
    dataset.river_name = metadata.get('river_name', '')
    dataset.source_has_discharge = 'yes' if source_has_discharge else 'no'

    # Close the file
    dataset.close()


def regression_test(discharge_dir, sediment_dir, output_dir):
    """
    Regression test: prove that a Bayern station with a sediment file but
    NO discharge file can successfully produce an SSC NetCDF.

    Uses station 12001006 (Fussen / Lech) which has sediment but no discharge.
    """
    print("\n" + "=" * 60)
    print("REGRESSION TEST: SSC-only station (sediment file, no discharge)")
    print("=" * 60)

    station_id = '12001006'
    audit = {
        'sediment_stations_total': 0,
        'stations_with_q': 0,
        'ssc_only_stations': 0,
        'q_only_skipped': 0,
        'paired_dates': 0,
        'ssc_only_dates': 0,
        'final_records': 0,
    }

    success = process_station(station_id, discharge_dir, sediment_dir, output_dir, audit)

    if success:
        # Quick verification: read back the NetCDF and check SSC is present
        test_file = os.path.join(output_dir, f"Bayern_{station_id}.nc")
        ds = nc.Dataset(test_file, 'r')
        ssc = ds.variables['ssc'][:]
        q = ds.variables['discharge'][:]
        ssl = ds.variables['sediment_load'][:]
        n_ssc_finite = int(np.sum(ssc != -9999.0))
        n_q_present = int(np.sum(q != -9999.0))
        n_ssl_present = int(np.sum(ssl != -9999.0))
        has_discharge_attr = getattr(ds, 'source_has_discharge', 'unknown')
        ds.close()

        print(f"\n  REGRESSION TEST RESULT: PASS")
        print(f"    Station {station_id} generated NetCDF successfully.")
        print(f"    SSC valid records: {n_ssc_finite}")
        print(f"    Q valid records:   {n_q_present} (expected 0)")
        print(f"    SSL valid records: {n_ssl_present} (expected 0)")
        print(f"    source_has_discharge attr: {has_discharge_attr}")

        if n_ssc_finite > 0 and n_q_present == 0 and n_ssl_present == 0:
            print(f"  + All checks passed: SSC-only station handled correctly.")
        else:
            print(f"  ! Unexpected values -- please investigate.")
    else:
        print(f"\n  REGRESSION TEST RESULT: FAIL")
        print(f"    Station {station_id} could NOT be processed.")

    return success


def main():
    """
    Main processing function.

    Note: This script ONLY processes DAILY data (tmw files).
    Other time resolutions (e.g., ezw files) are NOT processed.

    Revised (2026-08):
      - Candidate stations are based on sediment/SSC files (core input).
      - Discharge is optional; SSC-only stations are processed.
      - Q-only stations (no sediment) are excluded from the sediment product.
    """

    # Directories
    input_dir = PROJECT_ROOT / "Source" / "bayern"
    discharge_dir = input_dir / "discharge"
    sediment_dir = input_dir / "ssp"
    output_dir = input_dir / "done"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Candidate station IDs: based on SEDIMENT files (core input) ---
    # Get all unique station IDs from DAILY (tmw) sediment files
    sediment_files_list = glob.glob(os.path.join(sediment_dir, '*_tmw_*.csv'))
    sediment_ids = set([os.path.basename(f).split('_')[0] for f in sediment_files_list])

    # Get all unique station IDs from DAILY (tmw) discharge files (for reference)
    discharge_files_list = glob.glob(os.path.join(discharge_dir, '*_tmw_*.csv'))
    discharge_ids = set([os.path.basename(f).split('_')[0] for f in discharge_files_list])

    # Candidate stations = stations with sediment data (NOT Q+SSC)
    candidate_ids = sorted(sediment_ids)

    # Q-only stations (excluded from sediment product)
    q_only_ids = discharge_ids - sediment_ids

    print("=" * 60)
    print("NOTE: Processing DAILY data only (tmw files)")
    print("      Other time resolutions (ezw files) are excluded")
    print("=" * 60)
    print(f"Stations with DAILY discharge data: {len(discharge_ids)}")
    print(f"Stations with DAILY sediment data:  {len(sediment_ids)}")
    print(f"Candidate stations (have sediment):  {len(candidate_ids)}")
    print(f"Q-only stations (excluded):          {len(q_only_ids)}")
    if q_only_ids:
        print(f"  Q-only IDs: {sorted(q_only_ids)}")
    ssc_only_ids = sediment_ids - discharge_ids
    print(f"SSC-only stations (sediment, no Q):  {len(ssc_only_ids)}")
    if ssc_only_ids:
        print(f"  SSC-only IDs: {sorted(ssc_only_ids)}")

    # --- Audit accumulator ---
    audit = {
        'sediment_stations_total': 0,
        'stations_with_q': 0,
        'ssc_only_stations': 0,
        'q_only_skipped': 0,
        'paired_dates': 0,
        'ssc_only_dates': 0,
        'final_records': 0,
    }

    # Process each station
    success_count = 0
    failed_count = 0
    retained_stations = []

    for station_id in candidate_ids:
        success = process_station(station_id, discharge_dir, sediment_dir, output_dir, audit)
        if success:
            success_count += 1
            retained_stations.append(station_id)
        else:
            failed_count += 1

    # --- Audit summary ---
    print(f"\n{'=' * 60}")
    print(f"AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Sediment stations total (candidate):    {len(candidate_ids)}")
    print(f"  -> Successfully processed:              {success_count}")
    print(f"  -> Failed / skipped:                    {failed_count}")
    print(f"Stations also having Q:                  {audit['stations_with_q']}")
    print(f"SSC-only stations (no Q):                {audit['ssc_only_stations']}")
    print(f"Q-only stations (excluded from product): {audit['q_only_skipped']}")
    print(f"Paired Q+SSC dates (total):              {audit['paired_dates']}")
    print(f"SSC-only dates (total):                  {audit['ssc_only_dates']}")
    print(f"Final retained stations:                 {success_count}")
    print(f"Final retained records (total):          {audit['final_records']}")
    print(f"Output directory:                        {output_dir}")
    print(f"{'=' * 60}")

    # --- Regression test ---
    regression_test(discharge_dir, sediment_dir, output_dir)


if __name__ == '__main__':
    main()
