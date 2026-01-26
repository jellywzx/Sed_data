#!/usr/bin/env python3
"""
Process GSED monthly SSC data to CF-1.8 and ACDD-1.3 compliant netCDF format
- Implements quality control flags
- Follows CF-1.8 conventions for metadata
- Generates station summary CSV file
- Only includes stations with valid data

Author: Zhongwang Wei
Email: weizhw6@mail.sysu.edu.cn
Institution: Sun Yat-sen University, China
Date: 2025-10-26
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
import os
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
    propagate_ssc_q_inconsistency_to_ssl,
)

# Quality flag definitions
FLAG_GOOD = 0       # Good data
FLAG_ESTIMATED = 1  # Estimated data
FLAG_SUSPECT = 2    # Suspect data (e.g., extreme values)
FLAG_BAD = 3        # Bad data (e.g., negative values)
FLAG_MISSING = 9    # Missing in source


def get_geometry_info(shapefile_path, r_id):
    """
    Extract centroid coordinates from shapefile for given R_ID using ogrinfo

    Args:
        shapefile_path: Path to shapefile
        r_id: Reach ID

    Returns:
        tuple: (latitude, longitude, reach_length)
    """
    try:
        cmd = f'ogrinfo -al {shapefile_path} -where "R_ID={r_id}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            return None, None, None

        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'LINESTRING' in line:
                # Extract coordinates from LINESTRING
                coords_str = line.split('LINESTRING')[1].strip()
                coords_str = coords_str.replace('(', '').replace(')', '')
                coords = coords_str.split(',')

                # Calculate centroid from all points
                lons = []
                lats = []
                for coord in coords:
                    parts = coord.strip().split()
                    if len(parts) >= 2:
                        lons.append(float(parts[0]))
                        lats.append(float(parts[1]))

                if lons and lats:
                    lon = np.mean(lons)
                    lat = np.mean(lats)
                    length = None

                    # Try to get Length attribute
                    for line in lines:
                        if 'Length (Real)' in line:
                            try:
                                length = float(line.split('=')[1].strip())
                            except:
                                pass

                    return lat, lon, length

        return None, None, None
    except Exception as e:
        print(f"Error getting geometry for R_ID {r_id}: {e}")
        return None, None, None

def create_time_array(start_year=1985, start_month=1, n_months=432):
    """
    Create time array in days since 1970-01-01 for monthly data

    Args:
        start_year: Starting year (default: 1985)
        start_month: Starting month (default: 1)
        n_months: Number of months (default: 432, i.e., 1985-2020)

    Returns:
        numpy array: Days since 1970-01-01 for each month
    """
    base_date = datetime(1970, 1, 1)
    times = []

    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        current_date = datetime(year, month, 1)
        days_since = (current_date - base_date).days
        times.append(days_since)

    return np.array(times)

def get_year_month_from_index(start_year, start_month, index):
    """
    Convert time index to year and month

    Args:
        start_year: Starting year
        start_month: Starting month
        index: Time index

    Returns:
        tuple: (year, month)
    """
    year = start_year + (start_month - 1 + index) // 12
    month = (start_month - 1 + index) % 12 + 1
    return year, month

def apply_gsed_qc_with_tool(ssc):
    """
    Apply unified QC from tool.py for GSED SSC-only dataset.

    QC steps:
    1) physical plausibility (apply_quality_flag)
    2) log-IQR outlier detection (compute_log_iqr_bounds)

    Returns
    -------
    ssc_qc : array
        SSC values (bad values set to NaN)
    ssc_flag : array (int8)
        QC flags
    """

    n = len(ssc)

    ssc_flag = np.full(n, FILL_VALUE_INT, dtype=np.int8)
    ssc_qc = ssc.astype(float).copy()

    # -----------------------------
    # Counters
    # -----------------------------
    n_missing = 0
    n_bad = 0
    n_suspect = 0
    n_good = 0

    # --------------------------------------------------
    # 1) Physical QC
    # --------------------------------------------------
    for i in range(n):
        ssc_flag[i] = apply_quality_flag(ssc_qc[i], variable_name="SSC")

        if ssc_flag[i] == FLAG_MISSING:
            n_missing += 1
        elif ssc_flag[i] == FLAG_BAD:
            n_bad += 1
            ssc_qc[i] = np.nan
        elif ssc_flag[i] == FLAG_GOOD:
            pass
    # -----------------------------
    # 2) Log-IQR QC
    # -----------------------------
    lower, upper = compute_log_iqr_bounds(ssc_qc)

    if lower is not None:
        outlier = (
            (ssc_qc < lower) | (ssc_qc > upper)
        ) & (ssc_flag == FLAG_GOOD)

        ssc_flag[outlier] = FLAG_SUSPECT
        n_suspect = np.sum(outlier)

    # -----------------------------
    # Final GOOD count
    # -----------------------------
    n_good = np.sum(ssc_flag == FLAG_GOOD)

    qc_stats = {
        "n_total": n,
        "n_missing": int(n_missing),
        "n_bad": int(n_bad),
        "n_suspect": int(n_suspect),
        "n_good": int(n_good),
    }

    return ssc_qc, ssc_flag, qc_stats


def find_data_period(ssc_data, flags):
    """
    Find the period where SSC data exists (not all NaN or bad)

    Args:
        ssc_data: Array of SSC values
        flags: Array of quality flags

    Returns:
        tuple: (start_idx, end_idx) or (None, None) if no valid data
    """
    # Valid data means not missing and not bad
    valid_mask = (flags != FLAG_MISSING) & (flags != FLAG_BAD)

    if not np.any(valid_mask):
        return None, None

    # Find first and last valid index
    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1  # +1 to include the last month

    return start_idx, end_idx

def create_netcdf(r_id, ssc_data, time_array, lat, lon, length, output_dir):
    """
    Create CF-1.8 compliant netCDF file for a single station

    Args:
        r_id: Reach ID
        ssc_data: Array of SSC values
        time_array: Array of time values
        lat: Station latitude
        lon: Station longitude
        length: Reach length (meters)
        output_dir: Output directory path

    Returns:
        dict: Statistics for the station (for CSV summary)
    """
    # Apply QC and get flags
    ssc_qc, flags, qc_stats = apply_gsed_qc_with_tool(ssc_data)

    print(
        f"Reach {r_id} QC summary:\n"
        f"  total samples    : {qc_stats['n_total']}\n"
        f"  missing (flag=9) : {qc_stats['n_missing']}\n"
        f"  bad (flag=3)     : {qc_stats['n_bad']}\n"
        f"  suspect (flag=2) : {qc_stats['n_suspect']}\n"
        f"  good (flag=0)    : {qc_stats['n_good']}"
    )

    # Find data period
    start_idx, end_idx = find_data_period(ssc_qc, flags)

    trimmed = qc_stats['n_total'] - (end_idx - start_idx)

    print(
        f"  trimmed (no data at edges): {trimmed}\n"
        f"  retained for output       : {end_idx - start_idx}"
    )

    if start_idx is None:
        print(f"Reach {r_id}: No valid SSC data, skipping...")
        return None

    # Subset data to valid period
    ssc_subset = ssc_qc[start_idx:end_idx]
    flags_subset = flags[start_idx:end_idx]
    time_subset = time_array[start_idx:end_idx]
    n_times = len(time_subset)

    # Calculate statistics for CSV
    start_year, start_month = get_year_month_from_index(1985, 1, start_idx)
    end_year, end_month = get_year_month_from_index(1985, 1, end_idx - 1)

    # Count good data points
    good_count = np.sum(flags_subset == FLAG_GOOD)
    percent_complete = (good_count / n_times) * 100.0

    print(f"Reach {r_id}: {n_times} months ({start_year}-{start_month:02d} to {end_year}-{end_month:02d}), "
          f"{good_count} good ({percent_complete:.1f}%)")

    # Create output filename
    output_file = output_dir / f"GSED_{int(r_id)}.nc"

    # Create netCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # ===== Dimensions =====
        time_dim = ds.createDimension('time', n_times)

        # ===== Coordinate Variables =====
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'
        time_var[:] = time_subset

        # Latitude (scalar)
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
        lat_var[:] = lat if lat is not None else np.nan

        # Longitude (scalar)
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
        lon_var[:] = lon if lon is not None else np.nan

        # ===== Data Variables =====
        # Q (Discharge) - Not available in GSED
        q_var = ds.createVariable('Q', 'f4', ('time',),
                                  fill_value=-9999.0, zlib=True, complevel=4)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'lat lon'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Not available in GSED dataset (satellite-derived SSC only).'
        q_var[:] = -9999.0

        # Q Quality Flag
        q_flag_var = ds.createVariable('Q_flag', 'i1', ('time',),
                                       fill_value=np.int8(-128), zlib=True, complevel=4)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        q_flag_var[:] = FLAG_MISSING

        # SSC
        ssc_var = ds.createVariable('SSC', 'f4', ('time',),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'lat lon'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Satellite-derived monthly suspended sediment concentration from GSED dataset. Zhang et al. (2023). Scientific Data.'
        # Replace NaN with fill value
        ssc_filled = np.where(np.isnan(ssc_subset), -9999.0, ssc_subset)
        ssc_var[:] = ssc_filled

        # SSC Quality Flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'i1', ('time',),
                                         fill_value=np.int8(-128), zlib=True, complevel=4)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., extreme >3000 mg/L), 3=Bad (e.g., negative), 9=Missing in source.'
        ssc_flag_var[:] = flags_subset

        # SSL (Sediment Load) - Not available in GSED
        ssl_var = ds.createVariable('SSL', 'f4', ('time',),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'lat lon'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Not available in GSED dataset. Cannot be calculated without discharge data.'
        ssl_var[:] = -9999.0

        # SSL Quality Flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'i1', ('time',),
                                         fill_value=np.int8(-128), zlib=True, complevel=4)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssl_flag_var[:] = FLAG_MISSING

        # ===== Global Attributes (CF-1.8 & ACDD-1.3) =====
        ds.Conventions = 'CF-1.8, ACDD-1.3'
        ds.title = 'Harmonized Global River Discharge and Sediment'
        ds.summary = f'Satellite-derived monthly suspended sediment concentration for reach {int(r_id)} from the GSED dataset. Data covers {start_year}-{start_month:02d} to {end_year}-{end_month:02d}. Quality flags indicate data reliability.'

        # Data Source Information
        ds.data_source_name = 'GSED Dataset'
        ds.Source_ID = str(int(r_id))
        ds.reach_id = str(int(r_id))
        ds.source = 'Satellite station'
        ds.Type = 'Satellite'

        # Temporal Information
        ds.temporal_resolution = 'monthly'
        ds.time_coverage_start = f'{start_year}-{start_month:02d}-01'
        ds.time_coverage_end = f'{end_year}-{end_month:02d}-01'
        ds.temporal_span = f'{start_year}-{start_month:02d} to {end_year}-{end_month:02d}'

        # Spatial Information
        if lat is not None and lon is not None:
            ds.geospatial_lat_min = float(lat)
            ds.geospatial_lat_max = float(lat)
            ds.geospatial_lon_min = float(lon)
            ds.geospatial_lon_max = float(lon)
            ds.geographic_coverage = f'River reach centroid at ({lat:.4f}°N, {lon:.4f}°E)'

        if length is not None:
            ds.reach_length_m = float(length)

        # Variables
        ds.variables_provided = 'Q, SSC, SSL'
        ds.number_of_data = '1'

        # References
        ds.reference = 'Zhang, Y., Shi, H., Yu, X., Dong, J., & Wang, Z. (2023). A global dataset of monthly river suspended sediment concentration derived from satellites (1985-2020). Scientific Data, 10, 325. https://doi.org/10.1038/s41597-023-02233-0'
        ds.source_data_link = 'https://doi.org/10.1038/s41597-023-02233-0'

        # Creator Information
        ds.creator_name = 'Zhongwang Wei'
        ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ds.creator_institution = 'Sun Yat-sen University, China'

        # Processing Information
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.history = f'{current_time}: Converted from GSED CSV to CF-1.8 compliant NetCDF format. Applied quality control checks: flagged negative values as bad, values >3000 mg/L as suspect. Script: process_gsed_cf18.py'
        ds.date_created = datetime.now().strftime('%Y-%m-%d')
        ds.date_modified = datetime.now().strftime('%Y-%m-%d')
        ds.processing_level = 'Quality controlled and standardized'

        ds.comment = 'Data represents satellite-derived monthly suspended sediment concentration. Discharge and sediment load are not available in this dataset. Quality flags indicate: 0=good, 1=estimated, 2=suspect (extreme values), 3=bad (negative values), 9=missing.'

    print(f"Created: {output_file}")

    # --------------------------------------------------
    # CF-1.8 / ACDD-1.3 compliance check
    # --------------------------------------------------
    # errors, warnings = check_nc_completeness(output_file)

    # if errors:
    #     print("❌ CF/ACDD compliance FAILED:")
    #     for e in errors:
    #         print("   -", e)
    #     raise RuntimeError("NetCDF compliance check failed")

    # if warnings:
    #     print("⚠️ CF/ACDD compliance warnings:")
    #     for w in warnings:
    #         print("   -", w)


    # Return statistics for CSV
    stats = {
        'Source_ID': int(r_id),
        'reach_id': int(r_id),
        'longitude': lon if lon is not None else np.nan,
        'latitude': lat if lat is not None else np.nan,
        'reach_length_m': length if length is not None else np.nan,
        'SSC_start_date': f'{start_year}-{start_month:02d}',
        'SSC_end_date': f'{end_year}-{end_month:02d}',
        'SSC_percent_complete': percent_complete,
        'temporal_span': f'{start_year}-{start_month:02d} to {end_year}-{end_month:02d}',
        'n_months': n_times,
        'n_good': good_count
    }

    return stats

def create_summary_csv(stats_list, output_dir):
    """
    Create summary CSV file with station metadata

    Args:
        stats_list: List of station statistics dictionaries
        output_dir: Output directory path
    """
    csv_file = output_dir / 'GSED_station_summary.csv'

    # Create DataFrame
    df = pd.DataFrame(stats_list)

    # Add common metadata columns
    df['Data Source Name'] = 'GSED Dataset'
    df['Type'] = 'Satellite'
    df['Temporal Resolution'] = 'monthly'
    df['Variables Provided'] = 'Q, SSC, SSL'
    df['Geographic Coverage'] = 'Global rivers'
    df['Reference/DOI'] = 'https://doi.org/10.1038/s41597-023-02233-0'

    # Add Q and SSL columns (all missing for GSED)
    df['Q_start_date'] = ''
    df['Q_end_date'] = ''
    df['Q_percent_complete'] = 0.0
    df['SSL_start_date'] = ''
    df['SSL_end_date'] = ''
    df['SSL_percent_complete'] = 0.0

    # Reorder columns
    columns = [
        'Source_ID', 'reach_id', 'longitude', 'latitude', 'reach_length_m',
        'Data Source Name', 'Type', 'Temporal Resolution', 'temporal_span',
        'Variables Provided', 'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete',
        'n_months', 'n_good'
    ]

    df = df[columns]

    # Save to CSV
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\nCreated summary CSV: {csv_file}")
    print(f"Total stations: {len(df)}")

def main():
    """Main processing function"""
    # File paths
    script_dir = Path(__file__).resolve().parent          # .../Script/GSED
    project_root = script_dir.parents[1]                  # .../sediment_wzx_1111

    source_dir = project_root / "Source" / "GSED" / "GSED"
    csv_file = source_dir / "GSED_Reach_Monthly_SSC.csv"
    shapefile = source_dir / "GSED_Reach.shp"
    output_dir = project_root / "Output_r" / "monthly" / "GSED" / "qc"


    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GSED Monthly SSC Data Processing")
    print("CF-1.8 & ACDD-1.3 Compliant NetCDF Generation")
    print("="*70)

    print(f"\nReading GSED CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} reaches in CSV file")

    # Create time array for all months (1985-01 to 2020-12)
    time_array = create_time_array(1985, 1, 432)

    # Process each station
    stats_list = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        r_id = row['R_ID']

        # Extract SSC data (all columns except first one which is R_ID)
        ssc_data = row.iloc[1:].values.astype(float)

        try:
            # Get geometry info from shapefile
            if (idx + 1) % 100 == 0:
                print(f"\nProcessing reach {r_id} ({idx+1}/{len(df)})...")

            lat, lon, length = get_geometry_info(str(shapefile), r_id)

            if lat is None or lon is None:
                print(f"  Warning: Could not extract coordinates for R_ID {r_id}")

            # Create netCDF file
            stats = create_netcdf(r_id, ssc_data, time_array, lat, lon, length, output_dir)

            if stats is not None:
                stats_list.append(stats)
                success_count += 1
            else:
                skip_count += 1

        except Exception as e:
            print(f"Error processing reach {r_id}: {e}")
            error_count += 1
            continue

    # Create summary CSV
    if stats_list:
        create_summary_csv(stats_list, output_dir)

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Successfully created: {success_count} files")
    print(f"Skipped (no valid data): {skip_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
