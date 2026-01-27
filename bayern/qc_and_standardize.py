#!/usr/bin/env python3
"""
Bayern Dataset - Quality Control and CF-1.8 Standardization Script

This script performs comprehensive quality control, metadata standardization,
and time series trimming for daily Bayern river sediment data.

Processing Steps:
1. Quality flag creation (Q_flag, SSC_flag, SSL_flag) with physical law checks
2. Time series trimming (remove leading/trailing NaN periods)
3. Variable renaming (discharge→Q, ssc→SSC, sediment_load→SSL, latitude→lat, longitude→lon)
4. Dimension restructuring (time=UNLIMITED)
5. CF-1.8 and ACDD-1.3 metadata standardization
6. CSV summary generation

Author: Zhongwang Wei
Date: 2025-10-25
Institution: Sun Yat-sen University, China
"""


import os
import sys
# Add parent directory to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import glob
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    compute_log_iqr_bounds,
    calculate_ssc,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    propagate_ssc_q_inconsistency_to_ssl,
    apply_quality_flag_array,        
    apply_hydro_qc_with_provenance, 
)

def fmt_float(x, nd=2):
    try:
        if x is None:
            return "NA"
        if isinstance(x, float) and (x != x):  # NaN
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def log_station_qc(site_name, site_code=None, n=None,
                   skipped_log_iqr=False, skipped_ssc_q=False,
                   q=None, q_flag=None, ssc=None, ssc_flag=None, ssl=None, ssl_flag=None,
                   created_path=None):
    tag = f"{site_name}" + (f" ({site_code})" if site_code is not None else "")
    print(f"\nProcessing: {tag}")

    if n is not None:
        if skipped_log_iqr:
            print(f"  ⚠️ [{tag}] Sample size = {n} < 5, log-IQR statistical QC skipped.")
        if skipped_ssc_q:
            print(f"  ⚠️ [{tag}] Sample size = {n} < 5, SSC-Q consistency check and diagnostic plot skipped.")

    if created_path:
        print(f"  ✓ Created: {created_path}")

    # values + flag
    if q is not None or q_flag is not None:
        print(f"  Q:   {fmt_float(q)} m3/s (flag={q_flag})")
    if ssc is not None or ssc_flag is not None:
        print(f"  SSC: {fmt_float(ssc)} mg/L (flag={ssc_flag})")
    if ssl is not None or ssl_flag is not None:
        print(f"  SSL: {fmt_float(ssl)} ton/day (flag={ssl_flag})")

def log_summary(n_success, n_failed, summary_csv=None, out_dir=None):
    print("\n" + "=" * 70)
    print(f"Summary: {n_success} successful, {n_failed} failed")
    print("=" * 70)
    if summary_csv:
        print(f"Generated station summary CSV: {summary_csv}")
    if out_dir:
        print(f"All files saved to: {out_dir}")
    print("Processing complete!\n")

def find_valid_time_range(ds_in):
    """
    Find the first and last valid data points to trim time series.

    For daily data, we want to:
    - Remove leading periods with all missing data
    - Remove trailing periods with all missing data
    - Keep the continuous middle section with at least some valid data

    Parameters:
    -----------
    ds_in : netCDF4.Dataset
        Input dataset

    Returns:
    --------
    start_idx, end_idx : int
        Start and end indices for valid data range
    """

    # Read time series data 
    q_data = ds_in.variables['discharge'][:]
    ssc_data = ds_in.variables['ssc'][:]
    ssl_data = ds_in.variables['sediment_load'][:]

    # Replace fill values with NaN
    q_data = np.where(q_data == -9999.0, np.nan, q_data)
    ssc_data = np.where(ssc_data == -9999.0, np.nan, ssc_data)
    ssl_data = np.where(ssl_data == -9999.0, np.nan, ssl_data)

    # Find where we have ANY valid data
    valid_mask = (~np.isnan(q_data)) | (~np.isnan(ssc_data)) | (~np.isnan(ssl_data))

    if not np.any(valid_mask):
        # No valid data at all - should not happen but handle it
        return 0, len(q_data)

    # Find first and last valid indices
    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1  # +1 because we want inclusive end

    return start_idx, end_idx


def process_station(input_file, output_dir):
    """
    Process a single Bayern station file:
    - Apply QC checks
    - Trim time series
    - Rename variables
    - Standardize metadata

    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file
    output_dir : str
        Output directory

    Returns:
    --------
    station_info : dict or None
        Station metadata for CSV summary
    """

    try:
        # Read input file
        ds_in = nc.Dataset(input_file, 'r')

        # Read metadata
        station_id = ds_in.station_id if hasattr(ds_in, 'station_id') else os.path.basename(input_file).replace('.nc', '')
        station_name = ds_in.station_name if hasattr(ds_in, 'station_name') else ""
        river_name = ds_in.river_name if hasattr(ds_in, 'river_name') else ""

        # Read coordinates
        lat = float(ds_in.variables['latitude'][:])
        lon = float(ds_in.variables['longitude'][:])
        alt = float(ds_in.variables['altitude'][:])
        upstream_area = float(ds_in.variables['upstream_area'][:])

        # Find valid time range
        start_idx, end_idx = find_valid_time_range(ds_in)

        # Read time series data (trimmed)
        time_data = ds_in.variables['time'][start_idx:end_idx]
        q_data = ds_in.variables['discharge'][start_idx:end_idx]
        ssc_data = ds_in.variables['ssc'][start_idx:end_idx]
        ssl_data = ds_in.variables['sediment_load'][start_idx:end_idx]

        # Replace fill values with NaN for processing
        q_data = np.where(q_data == FILL_VALUE_FLOAT, np.nan, q_data)
        ssc_data = np.where(ssc_data == FILL_VALUE_FLOAT, np.nan, ssc_data)
        ssl_data = np.where(ssl_data == FILL_VALUE_FLOAT, np.nan, ssl_data)

        # Log_IQR Quality control   
        ssc_lower, ssc_upper = compute_log_iqr_bounds(ssc_data)
        ssl_lower, ssl_upper = compute_log_iqr_bounds(ssl_data)

        # ---- initialize flags ----
        q_flags = np.zeros(len(q_data), dtype=np.int8)
        ssc_flags = np.zeros(len(ssc_data), dtype=np.int8)
        ssl_flags = np.zeros(len(ssl_data), dtype=np.int8)

        # ==========================================================
        # Build station-level SSC–Q envelope (daily data)
        # ==========================================================
        ssc_q_bounds = build_ssc_q_envelope(
            Q_m3s=q_data,
            SSC_mgL=ssc_data,
            k=1.5
        )
        # QC1-array（显式调用）：
        Q_flag_qc1   = apply_quality_flag_array(q_data,  "Q")
        SSC_flag_qc1 = apply_quality_flag_array(ssc_data,"SSC")
        SSL_flag_qc1 = apply_quality_flag_array(ssl_data,"SSL")
        # ==========================================================
        qc = apply_hydro_qc_with_provenance(
            time=time_data,  
            Q=q_data,
            SSC=ssc_data,
            SSL=ssl_data,
            Q_is_independent=True, # discharge is independent variable
            SSC_is_independent=True, # ssc is independent variable
            SSL_is_independent=True, # ssl is independent variable
            ssl_is_derived_from_q_ssc=False,
            qc2_k=1.5, 
            qc2_min_samples=5,
            qc3_k=1.5, 
            qc3_min_samples=5,
        )
        print("[DBG] qc is None?", qc is None)   # <- 加在 if qc is None 之前
        if qc is None:
            return None


        # 把最终结果写回 flags 数组（后面统计/写文件都靠它）
        q_flags   = qc["Q_flag"]
        ssc_flags = qc["SSC_flag"]
        ssl_flags = qc["SSL_flag"]

        print(
            f"[QC] {station_id} QC1(good cnt): "
            f"Q={(Q_flag_qc1==0).sum()}, SSC={(SSC_flag_qc1==0).sum()}, SSL={(SSL_flag_qc1==0).sum()} | "
            f"Final(good cnt): Q={(q_flags==0).sum()}, SSC={(ssc_flags==0).sum()}, SSL={(ssl_flags==0).sum()}"
        )


        # ==========================================================
        # SSC–Q diagnostic plot (station-level, optional but recommended)
        # ==========================================================
        try:
            plot_ssc_q_diagnostic(
                Q=q_data,
                SSC_mgL=ssc_data,
                Q_flag=q_flags,
                SSC_flag=ssc_flags,
                ssc_q_bounds=ssc_q_bounds,
                station_id=station_id,
                station_name=station_name,
                out_png=output_dir
            )
        except Exception as e:
            print(f"  ⚠️ Diagnostic plot failed for {station_id}: {e}")

        # Convert time to dates for reporting
        time_units = ds_in.variables['time'].units
        time_calendar = ds_in.variables['time'].calendar
        dates = nc.num2date(time_data, units=time_units, calendar=time_calendar)

        ds_in.close()

        # # Count quality flags for reporting
        q_good = np.sum(q_flags == 0)
        q_suspect = np.sum(q_flags == 2)
        q_bad = np.sum(q_flags == 3)
        q_missing = np.sum(q_flags == 9)

        ssc_good = np.sum(ssc_flags == 0)
        ssc_suspect = np.sum(ssc_flags == 2)
        ssc_bad = np.sum(ssc_flags == 3)
        ssc_missing = np.sum(ssc_flags == 9)

        ssl_good = np.sum(ssl_flags == 0)
        ssl_suspect = np.sum(ssl_flags == 2)
        ssl_bad = np.sum(ssl_flags == 3)
        ssl_missing = np.sum(ssl_flags == 9)

        print(f"  Station: {station_name} ({river_name})")
        print(f"  Location: {lat:.3f}°, {lon:.3f}°")
        print(f"  Time series trimmed: {len(time_data)} days ({dates[0]} to {dates[-1]})")
        print(f"  Q: {q_good} good, {q_suspect} suspect, {q_bad} bad, {q_missing} missing")
        print(f"  SSC: {ssc_good} good, {ssc_suspect} suspect, {ssc_bad} bad, {ssc_missing} missing")
        print(f"  SSL: {ssl_good} good, {ssl_suspect} suspect, {ssl_bad} bad, {ssl_missing} missing")

        # Create output file
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        # === QC printout (per-station) ===
        n_samples = len(q_data)
        skipped_log_iqr = (n_samples < 5)
        skipped_ssc_q = (n_samples < 5)

        # 用最后一个有效值来展示（更像你截图那种“一行一个站点”）
        def last_valid(arr):
            arr = np.asarray(arr)
            m = np.isfinite(arr)
            return arr[m][-1] if np.any(m) else np.nan

        q_show = last_valid(q_data)
        ssc_show = last_valid(ssc_data)
        ssl_show = last_valid(ssl_data)

        # 对应的 flag 也取最后一个有效位置
        def last_valid_flag(arr, flags):
            arr = np.asarray(arr)
            flags = np.asarray(flags)
            m = np.isfinite(arr)
            return int(flags[m][-1]) if np.any(m) else 9

        qf_show = last_valid_flag(q_data, q_flags)
        sscf_show = last_valid_flag(ssc_data, ssc_flags)
        sslf_show = last_valid_flag(ssl_data, ssl_flags)

        log_station_qc(
            site_name=station_name if station_name else station_id,
            site_code=station_id,
            n=n_samples,
            skipped_log_iqr=skipped_log_iqr,
            skipped_ssc_q=skipped_ssc_q,
            q=q_show, q_flag=qf_show,
            ssc=ssc_show, ssc_flag=sscf_show,
            ssl=ssl_show, ssl_flag=sslf_show,
            created_path=output_file
        )

        with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

            # Create dimensions
            time_dim = ds.createDimension('time', None)  # UNLIMITED

            # Create coordinate variables
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.standard_name = 'time'
            time_var.long_name = 'time of measurement'
            time_var.units = 'days since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'
            time_var.axis = 'T'
            time_var[:] = time_data

            # Create scalar coordinate variables (renamed)
            lat_var = ds.createVariable('lat', 'f4')
            lat_var.standard_name = 'latitude'
            lat_var.long_name = 'station latitude'
            lat_var.units = 'degrees_north'
            lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
            lat_var[:] = lat

            lon_var = ds.createVariable('lon', 'f4')
            lon_var.standard_name = 'longitude'
            lon_var.long_name = 'station longitude'
            lon_var.units = 'degrees_east'
            lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
            lon_var[:] = lon

            alt_var = ds.createVariable('altitude', 'f4')
            alt_var.standard_name = 'altitude'
            alt_var.long_name = 'station altitude above sea level'
            alt_var.units = 'm'
            if not np.isnan(alt):
                alt_var[:] = alt
            else:
                alt_var[:] = -9999.0

            area_var = ds.createVariable('upstream_area', 'f4')
            area_var.long_name = 'upstream drainage area'
            area_var.units = 'km2'
            if not np.isnan(upstream_area):
                area_var[:] = upstream_area
            else:
                area_var[:] = -9999.0
                area_var.comment = 'Not available in source data'

            # Create data variables (renamed)
            q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0,
                                      zlib=True, complevel=4)
            q_var.standard_name = 'water_volume_transport_in_river_channel'
            q_var.long_name = 'river discharge'
            q_var.units = 'm3 s-1'
            q_var.coordinates = 'time lat lon altitude'
            q_var.ancillary_variables = 'Q_flag'
            q_var[:] = q_data

            ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0,
                                        zlib=True, complevel=4)
            ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
            ssc_var.long_name = 'suspended sediment concentration'
            ssc_var.units = 'mg L-1'
            ssc_var.coordinates = 'time lat lon altitude'
            ssc_var.ancillary_variables = 'SSC_flag'
            ssc_var.comment = 'Original data in g/m³, which equals mg/L'
            ssc_var[:] = ssc_data

            ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0,
                                        zlib=True, complevel=4)
            ssl_var.long_name = 'suspended sediment load'
            ssl_var.units = 'ton day-1'
            ssl_var.coordinates = 'time lat lon altitude'
            ssl_var.ancillary_variables = 'SSL_flag'
            ssl_var.comment = 'Calculated as: SSL = Q × SSC × 0.0864 (Q in m³/s, SSC in g/m³, SSL in ton/day)'
            ssl_var[:] = ssl_data

            # Create quality flag variables
            q_flag_var = ds.createVariable('Q_flag', 'i1', ('time',), zlib=True, complevel=4)
            q_flag_var.long_name = 'quality flag for river discharge'
            q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
            q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
            q_flag_var.comment = 'Quality flags: 0=good (passes all checks), 1=estimated, 2=suspect (Q=0 or Q>10000 m³/s), 3=bad (Q<0), 9=missing'
            q_flag_var[:] = q_flags

            ssc_flag_var = ds.createVariable('SSC_flag', 'i1', ('time',), zlib=True, complevel=4)
            ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
            ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
            ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
            ssc_flag_var.comment = 'Quality flags: 0=good (0≤SSC≤3000 mg/L), 1=estimated, 2=suspect (SSC>3000 mg/L), 3=bad (SSC<0), 9=missing'
            ssc_flag_var[:] = ssc_flags

            ssl_flag_var = ds.createVariable('SSL_flag', 'i1', ('time',), zlib=True, complevel=4)
            ssl_flag_var.long_name = 'quality flag for suspended sediment load'
            ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
            ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
            ssl_flag_var.comment = 'Quality flags: 0=good (SSL≥0), 1=estimated, 2=suspect, 3=bad (SSL<0), 9=missing'
            ssl_flag_var[:] = ssl_flags

            # Global attributes (CF-1.8 and ACDD-1.3 compliant)
            ds.Conventions = 'CF-1.8, ACDD-1.3'
            ds.title = 'Harmonized Global River Discharge and Sediment'
            ds.summary = f'Daily time series of river discharge, suspended sediment concentration, and sediment load for {river_name} at {station_name}, Bavaria, Germany. Data from Bayerisches Landesamt für Umwelt (Bavarian Environment Agency) monitoring network.'
            ds.source = 'In-situ station data'
            ds.data_source_name = 'Bayern State Environmental Agency (LfU) River Monitoring Network'
            ds.Type = 'In-situ'
            ds.Temporal_Resolution = 'daily'
            ds.Temporal_Span = f'{dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}'

            # Determine which variables are provided
            vars_provided = []
            if q_good > 0:
                vars_provided.append('Q')
            if ssc_good > 0:
                vars_provided.append('SSC')
            if ssl_good > 0:
                vars_provided.append('SSL')
            vars_provided_str = ', '.join(vars_provided) if vars_provided else 'none'

            ds.Variables_Provided = vars_provided_str
            ds.Geographic_Coverage = 'Bavaria, Germany'
            ds.Reference = 'Data from Bayerisches Landesamt für Umwelt (LfU). Available at: https://www.gkd.bayern.de/en/rivers/discharge and https://www.gkd.bayern.de/en/rivers/suspended-sediment'
            ds.source_data_link = 'https://www.gkd.bayern.de/en/'

            # Creator information
            ds.creator_name = 'Zhongwang Wei'
            ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
            ds.creator_institution = 'Sun Yat-sen University, China'

            # Processing history
            original_history = f"Created on 2025-10-24 by convert_bayern_to_netcdf.py"
            qc_history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Quality controlled and standardized to CF-1.8/ACDD-1.3 compliance. Added quality flags (Q_flag, SSC_flag, SSL_flag), renamed variables (discharge→Q, ssc→SSC, sediment_load→SSL, latitude→lat, longitude→lon), trimmed time series from {len(time_data)} to {len(time_data)} days (removed leading/trailing NaN periods), changed time dimension to UNLIMITED. Script: qc_and_standardize.py"
            ds.history = f"{original_history}; {qc_history}"

            # Station-specific attributes
            ds.location_id = station_id
            ds.station_name = station_name
            ds.river_name = river_name
            ds.country = 'Germany'
            ds.continent_region = 'Europe, Central Europe'

        # Prepare station info for CSV
        station_info = {
            'station_name': station_name,
            'Source_ID': station_id,
            'river_name': river_name,
            'longitude': lon,
            'latitude': lat,
            'altitude': alt if not np.isnan(alt) else 'N/A',
            'upstream_area': upstream_area if not np.isnan(upstream_area) else 'N/A',
            'Data Source Name': 'Bayern State Environmental Agency (LfU) River Monitoring Network',
            'Type': 'In-situ',
            'Temporal Resolution': 'daily',
            'Temporal Span': f'{dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}',
            'Variables Provided': vars_provided_str,
            'Geographic Coverage': 'Bavaria, Germany',
            'Reference/DOI': 'https://www.gkd.bayern.de/en/',
            'Q_start_date': dates[0].strftime('%Y-%m-%d') if q_good > 0 else 'N/A',
            'Q_end_date': dates[-1].strftime('%Y-%m-%d') if q_good > 0 else 'N/A',
            'Q_percent_complete': f"{100.0 * q_good / len(q_data):.1f}" if q_good > 0 else 'N/A',
            'SSC_start_date': dates[0].strftime('%Y-%m-%d') if ssc_good > 0 else 'N/A',
            'SSC_end_date': dates[-1].strftime('%Y-%m-%d') if ssc_good > 0 else 'N/A',
            'SSC_percent_complete': f"{100.0 * ssc_good / len(ssc_data):.1f}" if ssc_good > 0 else 'N/A',
            'SSL_start_date': dates[0].strftime('%Y-%m-%d') if ssl_good > 0 else 'N/A',
            'SSL_end_date': dates[-1].strftime('%Y-%m-%d') if ssl_good > 0 else 'N/A',
            'SSL_percent_complete': f"{100.0 * ssl_good / len(ssl_data):.1f}" if ssl_good > 0 else 'N/A'
        }

        return station_info

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main processing function."""

    print("="*80)
    print("Bayern Dataset - Quality Control and CF-1.8 Standardization")
    print("="*80)
    print()

    # Paths
    project_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

    input_dir = os.path.join(project_root, "Source", "bayern", "done")
    output_dir = os.path.join(project_root, "Output_r", "daily", "Bayern", "qc")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all input files
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Bayern_*.nc')))

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    # Process each station
    station_info_list = []
    processed_count = 0
    skipped_count = 0

    for i, input_file in enumerate(input_files):
        print(f"Processing: {os.path.basename(input_file)}")

        station_info = process_station(input_file, output_dir)

        if station_info:
            station_info_list.append(station_info)
            processed_count += 1
        else:
            skipped_count += 1

        print()

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"--- Progress: {i+1}/{len(input_files)} files processed ---")
            print()

    # Generate CSV summary
    print("="*80)
    print("Generating Station Summary CSV")
    print("="*80)
    print()

    if len(station_info_list) > 0:
        df = pd.DataFrame(station_info_list)

        # Reorder columns
        column_order = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'altitude', 'upstream_area', 'Data Source Name', 'Type',
            'Temporal Resolution', 'Temporal Span', 'Variables Provided',
            'Geographic Coverage', 'Reference/DOI',
            'Q_start_date', 'Q_end_date', 'Q_percent_complete',
            'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
            'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
        ]

        df = df[column_order]

        csv_file = os.path.join(output_dir, 'Bayern_station_summary.csv')
        df.to_csv(csv_file, index=False)

        print(f"Station summary saved to: {csv_file}")
        print(f"Total stations: {len(df)}")
    else:
        print("WARNING: No successful stations processed, CSV not created")

    print()
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no valid data): {skipped_count}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print()

    # Quality control summary
    # print("="*80)
    # print("Quality Control Summary")
    # print("="*80)
    # print("Quality Checks Applied:")
    # print("  Q (Discharge):")
    # print("    - Q < 0: Flagged as BAD (flag=3)")
    # print("    - Q == 0: Flagged as SUSPECT (flag=2)")
    # print("    - Q > 10000 m³/s: Flagged as SUSPECT (flag=2)")
    # print("    - Valid Q: Flagged as GOOD (flag=0)")
    # print("  SSC (Concentration):")
    # print("    - SSC < 0: Flagged as BAD (flag=3)")
    # print("    - SSC > 3000 mg/L: Flagged as SUSPECT (flag=2)")
    # print("    - Valid SSC: Flagged as GOOD (flag=0)")
    # print("  SSL (Load):")
    # print("    - SSL < 0: Flagged as BAD (flag=3)")
    # print("    - Valid SSL: Flagged as GOOD (flag=0)")
    # print()
    # print("Time Series Trimming:")
    # print("  - Removed leading/trailing periods with all missing data")
    # print("  - Kept continuous middle section with valid measurements")
    # print("="*80)
    # print()


if __name__ == '__main__':
    main()
