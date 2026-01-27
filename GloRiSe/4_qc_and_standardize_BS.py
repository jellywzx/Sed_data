#!/usr/bin/env python3
"""
Comprehensive Quality Control and CF-1.8 Standardization for GloRiSe Dataset

This script:
1. Corrects unit conversion formula (SSL = Q × SSC × 0.0864, not 86.4)
2. Implements physical quality checks with flags
3. Standardizes metadata to CF-1.8 and ACDD-1.3 compliance
4. Trims time ranges to data availability periods
5. Removes invalid stations
6. Generates station summary CSV

Author: Zhongwang Wei
Date: 2025-10-26
"""

import netCDF4 as nc4
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')
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
    propagate_ssc_q_inconsistency_to_ssl,
)


# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]   

INPUT_DIR = PROJECT_ROOT / "Source" / "GloRiSe" / "netcdf_output_BS"
OUTPUT_DIR = PROJECT_ROOT / "Output_r" / "daily" / "GloRiSe" / "BS" / "qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original data source information
DATA_SOURCE = {
    'name': 'GloRiSe Dataset',
    'full_name': 'Global River Sediment Database v1.1',
    'type': 'In-situ',
    'temporal_resolution': 'varies by station',
    'reference': 'Müller, G., Middelburg, J. J., and Sluijs, A.: Introducing GloRiSe – a global database on river sediment composition, Earth Syst. Sci. Data, 13, 3565–3575, https://doi.org/10.5194/essd-13-3565-2021, 2021.',
    'data_link': 'https://doi.org/10.5281/zenodo.4485795',
    'creator_name': 'Zhongwang Wei',
    'creator_email': 'weizhw6@mail.sysu.edu.cn',
    'creator_institution': 'Sun Yat-sen University, China'
}

# Quality flag definitions
QC_FLAGS = {
    0: 'good_data',
    1: 'estimated_data',
    2: 'suspect_data',
    3: 'bad_data',
    9: 'missing_data'
}

# # Physical thresholds for QC
# THRESHOLDS = {
#     'Q_min': 0,          # m³/s - negative values are bad
#     'Q_max': 300000,     # m³/s - extreme high (only largest rivers exceed this)
#     'SSC_min': 0,        # mg/L - negative values are bad
#     'SSC_max': 3000,     # mg/L - extreme high suspended sediment concentration
#     'SSL_min': 0         # ton/day - negative values are bad
# }

def apply_tool_qc(discharge, ssc, ssl, return_envelope=True):
    """
    Unified QC using tool.py:
    - physical plausibility
    - log-IQR on SSL
    - SSC–Q consistency envelope
    """

    n = len(discharge)
    qc_report = {
        "n_total": n,
        "Q_physical_bad": 0,
        "SSC_physical_bad": 0,
        "SSL_physical_bad": 0,
        "SSL_logIQR_suspect": 0,
        "SSC_Q_inconsistent": 0,
        "SSL_inherited_suspect": 0,
        "SSL_propagated_from_ssc_q": 0,

    }
    

   # -------------------------
    # 1. basic physical QC
    # -------------------------
    q_flag   = np.array([apply_quality_flag(v, "Q")   for v in discharge], dtype=np.int8)
    ssc_flag = np.array([apply_quality_flag(v, "SSC") for v in ssc],       dtype=np.int8)
    ssl_flag = np.array([apply_quality_flag(v, "SSL") for v in ssl],       dtype=np.int8)

    qc_report["Q_physical_bad"]   = int(np.sum(q_flag   != 0))
    qc_report["SSC_physical_bad"] = int(np.sum(ssc_flag != 0))
    qc_report["SSL_physical_bad"] = int(np.sum(ssl_flag != 0))


    # -------------------------
    # 2. log-IQR screening (SSL)
    # -------------------------
    lower, upper = compute_log_iqr_bounds(ssl, k=1.5)
    if lower is not None:
        outlier = (
            (ssl_flag == 0) &
            ((ssl < lower) | (ssl > upper))
        )
        ssl_flag[outlier] = 2   # suspect
        qc_report["SSL_logIQR_suspect"] = int(np.sum(outlier))

    # -------------------------
    # 3. SSC–Q envelope consistency
    # -------------------------
    ssc_q_bounds = build_ssc_q_envelope(
        Q_m3s=discharge,
        SSC_mgL=ssc,
        k=1.5,
        min_samples=5
    )

    if ssc_q_bounds is not None:
        bad_cnt = 0
        for i in range(n):
            bad, _ = check_ssc_q_consistency(
                Q=discharge[i],
                SSC=ssc[i],
                Q_flag=q_flag[i],
                SSC_flag=ssc_flag[i],
                ssc_q_bounds=ssc_q_bounds
            )
            if bad:
                bad_cnt += 1
                ssc_q_inconsistent = True

                # 先把 SSC_flag 从 0 -> 2（suspect）
                if ssc_flag[i] == 0:
                    ssc_flag[i] = np.int8(2)

                # 传播到 SSL_flag（SSL 是由 Q 和 SSC 推导的）
                old_ssl_flag = int(ssl_flag[i])
                ssl_flag[i] = np.int8(
                    propagate_ssc_q_inconsistency_to_ssl(
                        inconsistent=ssc_q_inconsistent,
                        Q=float(discharge[i]),
                        SSC=float(ssc[i]),
                        SSL=float(ssl[i]),
                        Q_flag=int(q_flag[i]),
                        SSC_flag=int(ssc_flag[i]),
                        SSL_flag=int(ssl_flag[i]),
                        ssl_is_derived_from_q_ssc=True,
                    )
                )
                if old_ssl_flag == 0 and int(ssl_flag[i]) != 0:
                    qc_report["SSL_propagated_from_ssc_q"] += 1

        qc_report["SSC_Q_inconsistent"] = bad_cnt
    else:
        print("    ℹ️ SSC–Q diagnostic skipped (insufficient samples)")

    # -------------------------
    # 4. propagate to SSL
    # -------------------------
    inherited = (ssl_flag == 0) & ((q_flag != 0) | (ssc_flag != 0))
    ssl_flag[inherited] = 2
    qc_report["SSL_inherited_suspect"] = int(np.sum(inherited))

    if return_envelope:
        return q_flag, ssc_flag, ssl_flag, ssc_q_bounds, qc_report
    else:
        return q_flag, ssc_flag, ssl_flag, qc_report


def get_valid_time_range(discharge, ssc, ssl, time_values):
    """
    Get the time range where at least one of discharge or sediment data is valid.

    Returns:
    --------
    start_idx, end_idx : indices of valid time range
    None if no valid data
    """
    n_time = len(time_values)

    # Find valid indices (not missing and not NaN)
    valid_q = (discharge != -9999.0) & (~np.isnan(discharge))
    valid_ssc = (ssc != -9999.0) & (~np.isnan(ssc))
    valid_ssl = (ssl != -9999.0) & (~np.isnan(ssl))

    # At least one variable should have valid data
    valid_any = valid_q | valid_ssc | valid_ssl

    if not np.any(valid_any):
        return None, None

    # Find first and last valid indices
    valid_indices = np.where(valid_any)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1  # +1 for Python slicing

    return start_idx, end_idx


def convert_time_to_datetime(time_values, time_units):
    """Convert time values to datetime objects."""
    from netCDF4 import num2date
    return num2date(time_values, units=time_units, calendar='gregorian')


def standardize_station_file(input_file):
    """
    Process a single GloRiSe station file with QC and standardization.

    Returns:
    --------
    station_info : dict with station metadata for CSV summary, or None if invalid
    """
    station_id = input_file.stem.replace('GloRiSe_', '')
    print(f"\nProcessing {station_id}...")

    # Read input file
    ds_in = nc4.Dataset(input_file, 'r')

    try:
        # Read data
        time_in = ds_in.variables['time'][:]
        time_units = ds_in.variables['time'].units
        time_calendar = getattr(ds_in.variables['time'], 'calendar', 'gregorian')

        discharge_in = ds_in.variables['Discharge_m3_s'][:]
        ssc_in = ds_in.variables['TSS_mg_L'][:]

        lat = float(ds_in.variables['latitude'][:])
        lon = float(ds_in.variables['longitude'][:])
        alt = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else np.nan
        upstream_area = float(ds_in.variables['upstream_area'][:]) if 'upstream_area' in ds_in.variables else np.nan

        # Get metadata
        country = ds_in.getncattr('country') if hasattr(ds_in, 'country') else 'Unknown'
        # Always use the current DATA_SOURCE reference for consistency
        references = DATA_SOURCE['reference']

        # CRITICAL FIX: Recalculate SSL with correct formula
        # SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864 (NOT 86.4!)
        ssl_in = np.where(
            (discharge_in == -9999.0) | (ssc_in == -9999.0) |
            np.isnan(discharge_in) | np.isnan(ssc_in),
            -9999.0,
            discharge_in * ssc_in * 0.0864  # CORRECTED from 86.4
        )

        # Trim to valid time range
        start_idx, end_idx = get_valid_time_range(discharge_in, ssc_in, ssl_in, time_in)

        if start_idx is None:
            print(f"  ✗ Skipped: No valid data")
            ds_in.close()
            return None

        # Trim arrays
        time = time_in[start_idx:end_idx]
        discharge = discharge_in[start_idx:end_idx]
        ssc = ssc_in[start_idx:end_idx]
        ssl = ssl_in[start_idx:end_idx]
        
        # Apply QC checks
        q_flag, ssc_flag, ssl_flag, ssc_q_bounds,qc_report = apply_tool_qc(
            discharge,
            ssc,
            ssl,
            return_envelope=True
        )
        print("    QC summary:")
        print(f"      total records           : {qc_report['n_total']}")
        print(f"      Q physical flagged       : {qc_report['Q_physical_bad']}")
        print(f"      SSC physical flagged     : {qc_report['SSC_physical_bad']}")
        print(f"      SSL physical flagged     : {qc_report['SSL_physical_bad']}")
        print(f"      SSL log-IQR suspect      : {qc_report['SSL_logIQR_suspect']}")
        print(f"      SSC–Q inconsistent       : {qc_report['SSC_Q_inconsistent']}")
        print(f"      SSL propagated from SSC–Q : {qc_report['SSL_propagated_from_ssc_q']}")
        print(f"      SSL inherited suspect    : {qc_report['SSL_inherited_suspect']}")
        # ---- Representative values for quick look (median of good, fallback to all valid) ----
        def _repr(v, f):
            v = np.asarray(v, dtype=float)
            f = np.asarray(f, dtype=np.int8)
            ok = np.isfinite(v) & (v != -9999.0)
            ok_good = ok & (f == 0)
            if np.any(ok_good):
                return float(np.nanmedian(v[ok_good])), 0
            if np.any(ok):
                return float(np.nanmedian(v[ok])), int(np.min(f[ok]))
            return float("nan"), 9

        qv, qf = _repr(discharge, q_flag)
        sscv, sscf = _repr(ssc, ssc_flag)
        sslv, sslf = _repr(ssl, ssl_flag)

        print(f"    Q  : {qv:.2f} m3/s (flag={qf})")
        print(f"    SSC: {sscv:.2f} mg/L (flag={sscf})")
        print(f"    SSL: {sslv:.2f} ton/day (flag={sslf})")



        # Convert time to datetime for summary
        time_dates = convert_time_to_datetime(time, time_units)

        # ----------------------------------
        # SSC–Q diagnostic plot
        # ----------------------------------
        diag_dir = OUTPUT_DIR / "ssc_q_diagnostic"
        diag_dir.mkdir(exist_ok=True)

        if ssc_q_bounds is not None:
            diag_png = diag_dir / f"GloRiSe_{station_id}_ssc_q_diagnostic.png"

            plot_ssc_q_diagnostic(
                time=time_dates,
                Q=discharge,
                SSC=ssc,
                Q_flag=q_flag,
                SSC_flag=ssc_flag,
                ssc_q_bounds=ssc_q_bounds,
                station_id=station_id,
                station_name=station_id,
                out_png=str(diag_png)
            )

        # Calculate statistics for each variable
        def calc_stats(data, flags):
            valid_mask = (data != -9999.0) & (~np.isnan(data))
            good_mask = valid_mask & (flags == 0)

            if not np.any(valid_mask):
                return None, None, 0.0

            start_date = time_dates[np.where(valid_mask)[0][0]].strftime('%Y-%m-%d')
            end_date = time_dates[np.where(valid_mask)[0][-1]].strftime('%Y-%m-%d')
            percent_complete = 100.0 * np.sum(good_mask) / len(data)

            return start_date, end_date, percent_complete

        q_start, q_end, q_pct = calc_stats(discharge, q_flag)
        ssc_start, ssc_end, ssc_pct = calc_stats(ssc, ssc_flag)
        ssl_start, ssl_end, ssl_pct = calc_stats(ssl, ssl_flag)

        # Determine overall temporal span
        all_dates = [d for d in [q_start, ssc_start, ssl_start] if d is not None]
        if not all_dates:
            print(f"  ✗ Skipped: No valid dates")
            ds_in.close()
            return None

        temporal_start = min(all_dates)
        temporal_end = max([d for d in [q_end, ssc_end, ssl_end] if d is not None])

        # Create output file
        output_file = OUTPUT_DIR / f"GloRiSe_{station_id}.nc"
        ds_out = nc4.Dataset(output_file, 'w', format='NETCDF4')

        # Create dimensions
        time_dim = ds_out.createDimension('time', len(time))

        # Create coordinate variables
        time_var = ds_out.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = time_units
        time_var.calendar = time_calendar
        time_var.axis = 'T'
        time_var[:] = time

        lat_var = ds_out.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
        lat_var[:] = lat

        lon_var = ds_out.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
        lon_var[:] = lon

        alt_var = ds_out.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var.comment = 'Source: Original data provided by GloRiSe database.'
        alt_var[:] = alt

        area_var = ds_out.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Source: Original data provided by GloRiSe database. May not be available for all stations.'
        area_var[:] = upstream_area

        # Create data variables
        q_var = ds_out.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'time lat lon altitude'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Source: Original data provided by GloRiSe database.'
        q_var[:] = discharge

        ssc_var = ds_out.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time lat lon altitude'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Original data provided by GloRiSe database.'
        ssc_var[:] = ssc

        ssl_var = ds_out.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon altitude'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864, where 0.0864 = 86400 s/day × 10⁻⁶ ton/mg.'
        ssl_var[:] = ssl

        # Create quality flag variables
        q_flag_var = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=9, zlib=True, complevel=4)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        q_flag_var[:] = q_flag

        ssc_flag_var = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=9, zlib=True, complevel=4)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssc_flag_var[:] = ssc_flag

        ssl_flag_var = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=9, zlib=True, complevel=4)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssl_flag_var[:] = ssl_flag

        # Add global attributes (CF-1.8 and ACDD-1.3 compliant)
        ds_out.Conventions = 'CF-1.8, ACDD-1.3'
        ds_out.title = 'Harmonized Global River Discharge and Sediment'
        ds_out.summary = f'River discharge and suspended sediment data for station {station_id}. This dataset contains quality-controlled time series data including discharge, suspended sediment concentration, and sediment load with associated quality flags.'

        ds_out.data_source_name = DATA_SOURCE['name']
        ds_out.source_data_type = DATA_SOURCE['type']
        ds_out.source = f'{DATA_SOURCE["full_name"]} - quality controlled and standardized'
        ds_out.station_name = station_id
        ds_out.Source_ID = station_id

        ds_out.temporal_resolution = DATA_SOURCE['temporal_resolution']
        ds_out.temporal_span = f'{temporal_start} to {temporal_end}'
        ds_out.time_coverage_start = temporal_start
        ds_out.time_coverage_end = temporal_end

        ds_out.geospatial_lat_min = lat
        ds_out.geospatial_lat_max = lat
        ds_out.geospatial_lon_min = lon
        ds_out.geospatial_lon_max = lon
        ds_out.geographic_coverage = f'{country}'

        ds_out.variables_provided = 'altitude, upstream_area, Q, SSC, SSL'

        ds_out.reference = references
        ds_out.source_data_link = DATA_SOURCE['data_link']

        ds_out.creator_name = DATA_SOURCE['creator_name']
        ds_out.creator_email = DATA_SOURCE['creator_email']
        ds_out.creator_institution = DATA_SOURCE['creator_institution']

        # History (provenance)
        history_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: " \
                       f"Quality controlled and standardized to CF-1.8/ACDD-1.3 format. " \
                       f"Corrected SSL calculation (factor 0.0864 instead of 86.4). " \
                       f"Applied physical quality checks. " \
                       f"Trimmed to valid data period. " \
                       f"Script: qc_and_standardize_glorise.py"

        if hasattr(ds_in, 'history'):
            ds_out.history = ds_in.history + '; ' + history_entry
        else:
            ds_out.history = history_entry

        ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
        ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
        ds_out.processing_level = 'Quality controlled and standardized'

        # Close files
        ds_out.close()

        # errors, warnings = check_nc_completeness(str(output_file))

        # if errors:
        #     print("  ❌ CF/ACDD compliance FAILED:")
        #     for e in errors:
        #         print("     -", e)
        #     return None

        # if warnings:
        #     print("  ⚠️ CF/ACDD compliance warnings:")
        #     for w in warnings:
        #         print("     -", w)

        # ds_in.close()

        print(f"  ✓ Processed: {len(time)} records, {temporal_start} to {temporal_end}")
        print(f"    Q: {q_pct:.1f}% complete, SSC: {ssc_pct:.1f}% complete, SSL: {ssl_pct:.1f}% complete")

        # Return station info for CSV
        station_info = {
            'station_name': station_id,
            'Source_ID': station_id,
            'river_name': '',  # Not available in GloRiSe
            'longitude': lon,
            'latitude': lat,
            'altitude': alt if not np.isnan(alt) else '',
            'upstream_area': upstream_area if not np.isnan(upstream_area) else '',
            'Data Source Name': DATA_SOURCE['name'],
            'Type': DATA_SOURCE['type'],
            'Temporal Resolution': DATA_SOURCE['temporal_resolution'],
            'Temporal Span': f'{temporal_start} to {temporal_end}',
            'Variables Provided': 'Q, SSC, SSL',
            'Geographic Coverage': country,
            'Reference/DOI': DATA_SOURCE['data_link'],
            'Q_start_date': q_start if q_start else '',
            'Q_end_date': q_end if q_end else '',
            'Q_percent_complete': f'{q_pct:.1f}' if q_pct else '',
            'SSC_start_date': ssc_start if ssc_start else '',
            'SSC_end_date': ssc_end if ssc_end else '',
            'SSC_percent_complete': f'{ssc_pct:.1f}' if ssc_pct else '',
            'SSL_start_date': ssl_start if ssl_start else '',
            'SSL_end_date': ssl_end if ssl_end else '',
            'SSL_percent_complete': f'{ssl_pct:.1f}' if ssl_pct else ''
        }

        return station_info, qc_report

    except Exception as e:
        print(f"  ✗ Error: {e}")
        ds_in.close()
        return None


def main():

    global_qc = {
    "stations": 0,
    "records": 0,
    "Q_physical_bad": 0,
    "SSC_physical_bad": 0,
    "SSL_physical_bad": 0,
    "SSL_logIQR_suspect": 0,
    "SSC_Q_inconsistent": 0,
    "SSL_inherited_suspect": 0,
    }

    """Main processing function."""
    print("="*80)
    print("GloRiSe Dataset: Comprehensive QC and CF-1.8 Standardization")
    print("="*80)
    print(f"\nInput directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Get all GloRiSe NetCDF files
    input_files = sorted(INPUT_DIR.glob('GloRiSe_*.nc'))

    if not input_files:
        print("\n✗ No GloRiSe NetCDF files found!")
        return

    print(f"\nFound {len(input_files)} station files to process.\n")

    # Process each station
    station_list = []
    processed_count = 0
    skipped_count = 0

    for input_file in input_files:
        station_info, qc_report = standardize_station_file(input_file)

        if station_info is not None:
            station_list.append(station_info)
            processed_count += 1
        else:
            skipped_count += 1
        
        if qc_report is not None:
            global_qc["stations"] += 1
            global_qc["records"] += qc_report["n_total"]

            for k in global_qc:
                if k in qc_report:
                    global_qc[k] += qc_report[k]

    # Generate CSV summary
    if station_list:
        csv_file = OUTPUT_DIR / 'GloRiSe_station_summary.csv'
        df = pd.DataFrame(station_list)

        # Reorder columns to match reference format
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
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Generated CSV summary: {csv_file}")
        print(f"  {len(station_list)} stations included")

    # Final summary
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"Successfully processed: {processed_count} stations")
    print(f"Skipped (no valid data): {skipped_count} stations")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nIMPORTANT CORRECTIONS APPLIED:")
    print("  • Fixed SSL calculation: Q × SSC × 0.0864 (was incorrectly 86.4)")
    print("  • Added quality flags for all variables")
    print("  • Trimmed time ranges to data availability")
    print("  • Standardized metadata to CF-1.8 and ACDD-1.3")
    print("="*80)
    
    print("\nQC GLOBAL SUMMARY")
    print("-" * 80)
    for k, v in global_qc.items():
        print(f"{k:30s}: {v}")



if __name__ == '__main__':
    main()
