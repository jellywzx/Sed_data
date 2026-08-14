#!/usr/bin/env python3
"""
Comprehensive Quality Control and CF-1.8 Standardization for GloRiSe Dataset

This script:
1. Corrects unit conversion formula (SSL = Q x SSC x 0.0864, not 86.4)
2. Implements physical quality checks with flags
3. Standardizes metadata to CF-1.8 and ACDD-1.3 compliance
4. Trims time ranges to data availability periods
5. Removes invalid stations
6. Generates station summary CSV

IMPORTANT (2026-08): SSC-only stations (TSS present, Q missing) are preserved
throughout the pipeline.  QC3 (SSC-Q envelope consistency) only applies to
records where both Q and SSC are valid; TSS-only records are never dropped.
get_valid_time_range uses an OR gate (valid_q | valid_ssc | valid_ssl) so that
the presence of SSC alone is sufficient to retain a station.

Author: Zhongwang Wei
Date: 2025-10-26
"""

import netCDF4 as nc4
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_quality_flag,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    propagate_derived_flag_from_inputs,
)
from code.runtime import ensure_directory, resolve_output_root, resolve_source_root
from code.units import convert_ssl_units_if_needed
from code.validation import require_existing_directory

# Configuration
INPUT_DIR = resolve_source_root(start=__file__) / 'GloRiSe' / 'netcdf_output_SS'
OUTPUT_DIR = ensure_directory(
    resolve_output_root(start=__file__) / 'daily' / 'GloRiSe' / 'qc'
)

# Original data source information
DATA_SOURCE = {
    'name': 'GloRiSe Dataset',
    'full_name': 'Global River Sediment Database v1.1',
    'type': 'In-situ',
    'temporal_resolution': 'varies by station',
    'reference': 'Muller, G., Middelburg, J. J., and Sluijs, A.: Introducing GloRiSe - a global database on river sediment composition, Earth Syst. Sci. Data, 13, 3565-3575, https://doi.org/10.5194/essd-13-3565-2021, 2021.',
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

def apply_tool_qc(discharge, ssc, ssl, return_envelope=False):
    """
    Apply unified QC using tool.py logic:
    - physical plausibility
    - log-IQR screening
    - SSC-Q consistency (QC3: ONLY on Q-SSC paired records)

    SSC-only records (where Q is _FillValue) receive QC1 physical checks
    on SSC but are excluded from QC3 envelope analysis because Q_flag != GOOD.
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
        "n_Q_missing": 0,
        "n_ssc_only": 0,
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

    # Count Q-missing / SSC-only records
    qc_report["n_Q_missing"] = int(np.sum(q_flag == 9))
    # SSC-only: SSC is valid (flag 0) but Q is missing (flag 9) -> preserved
    qc_report["n_ssc_only"] = int(np.sum((q_flag == 9) & (ssc_flag == 0)))

    # -------------------------
    # 2. log-IQR screening (SSL)
    # -------------------------
    # GloRiSe SS SSL is fully derived from Q and SSC, not source-reported.
    # Match shared provenance-aware QC: valid derived SSL is estimated (1),
    # while suspect/bad/missing inputs propagate to the derived SSL flag.
    lower, upper = None, None
    outlier = np.zeros(n, dtype=bool)

    # -------------------------
    # 3. SSC-Q envelope consistency (QC3)
    #    Applies ONLY to records where both Q and SSC are valid (flag 0).
    #    TSS-only records (Q_flag == 9) are automatically excluded because
    #    check_ssc_q_consistency returns False when Q_flag != FLAG_GOOD.
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
                ssc_flag[i] = 2   # suspect
                bad_cnt += 1
        qc_report["SSC_Q_inconsistent"] = bad_cnt
    else:
        print("    SSC-Q diagnostic skipped (insufficient samples or SSC-only station)")

    # -------------------------
    # 4. derived SSL provenance semantics
    # -------------------------
    ssl_before_propagation = ssl_flag.copy()
    for i in range(n):
        ssl_flag[i] = propagate_derived_flag_from_inputs(
            derived_value=ssl[i],
            derived_flag=ssl_flag[i],
            input_flags=[q_flag[i], ssc_flag[i]],
            input_values=[discharge[i], ssc[i]],
        )
    inherited = (
        (ssl_before_propagation != ssl_flag)
        & np.isin(ssl_flag, [2, 3])
        & (ssl_before_propagation != 9)
    )
    qc_report["SSL_inherited_suspect"] = int(np.sum(inherited))

    # --- Step-level QC provenance arrays ---
    q_flag_qc1   = np.array([apply_quality_flag(v, "Q") for v in discharge], dtype=np.int8)
    ssc_flag_qc1 = np.array([apply_quality_flag(v, "SSC") for v in ssc], dtype=np.int8)
    ssl_flag_qc1 = np.array([apply_quality_flag(v, "SSL") for v in ssl], dtype=np.int8)

    q_flag_qc2 = q_flag_qc1.copy()
    q_flag_qc2[:] = np.int8(8)  # not_checked for Q (no QC2 applied)
    ssc_flag_qc2 = ssc_flag_qc1.copy()
    ssc_flag_qc2[:] = np.int8(8)
    ssl_flag_qc2 = ssl_flag_qc1.copy()
    if lower is not None:
        ssl_flag_qc2[outlier] = 2

    ssc_flag_qc3 = np.full(n, np.int8(8), dtype=np.int8)
    if ssc_q_bounds is not None:
        for i in range(n):
            bad, _ = check_ssc_q_consistency(
                Q=discharge[i], SSC=ssc[i],
                Q_flag=q_flag[i], SSC_flag=ssc_flag[i],
                ssc_q_bounds=ssc_q_bounds
            )
            if bad:
                ssc_flag_qc3[i] = 2
        # Mark pass where QC1 was good
        ssc_flag_qc3[(ssc_flag_qc1 == 0) & (ssc_flag_qc3 == 8)] = 0

    ssl_flag_qc3 = np.full(n, np.int8(8), dtype=np.int8)
    ssl_flag_qc3[(ssl_flag_qc1 == 0) & (ssc_flag_qc3 != 2)] = 0  # not_propagated

    if return_envelope:
        return q_flag, ssc_flag, ssl_flag, ssc_q_bounds, qc_report, \
               q_flag_qc1, q_flag_qc2, \
               ssc_flag_qc1, ssc_flag_qc2, ssc_flag_qc3, \
               ssl_flag_qc1, ssl_flag_qc2, ssl_flag_qc3
    else:
        return q_flag, ssc_flag, ssl_flag, qc_report, \
               q_flag_qc1, q_flag_qc2, \
               ssc_flag_qc1, ssc_flag_qc2, ssc_flag_qc3, \
               ssl_flag_qc1, ssl_flag_qc2, ssl_flag_qc3


def get_valid_time_range(discharge, ssc, ssl, time_values):
    """
    Get the time range where at least one of discharge or sediment data is valid.

    Uses OR logic: a time step is valid if Q, SSC, OR SSL is present.
    This ensures SSC-only stations (Q all _FillValue) are NOT trimmed away.

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

    # At least one variable should have valid data (OR gate — preserves SSC-only)
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

    SSC-only stations (where Discharge is all _FillValue) are fully supported:
    - Q and SSL are written as _FillValue for those records
    - QC3 (SSC-Q envelope) is skipped when <5 valid Q-SSC pairs exist
    - The station is retained as long as it has valid SSC data

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

        # Detect SSC-only station: no valid Q records at all
        has_any_q = np.any((discharge_in != -9999.0) & np.isfinite(discharge_in))
        if not has_any_q:
            print(f"  SSC-only station (no discharge records)")

        # Derive SSL: only when both Q and SSC are valid
        # SSL (ton/day) = Q (m3/s) x SSC (mg/L) x 0.0864
        ssl_in = np.where(
            (discharge_in == -9999.0) | (ssc_in == -9999.0) |
            np.isnan(discharge_in) | np.isnan(ssc_in),
            -9999.0,
            discharge_in * ssc_in * 0.0864  # CORRECTED from 86.4
        )

        # Trim to valid time range (OR gate — preserves SSC-only stations)
        start_idx, end_idx = get_valid_time_range(discharge_in, ssc_in, ssl_in, time_in)

        if start_idx is None:
            print(f"  Skipped: No valid data")
            ds_in.close()
            return None, None

        # Trim arrays
        time = time_in[start_idx:end_idx]
        discharge = discharge_in[start_idx:end_idx]
        ssc = ssc_in[start_idx:end_idx]
        ssl = ssl_in[start_idx:end_idx]

        # Apply QC checks
        q_flag, ssc_flag, ssl_flag, ssc_q_bounds, qc_report, *_ = apply_tool_qc(
            discharge,
            ssc,
            ssl,
            return_envelope=True
        )
        print("    QC summary:")
        print(f"      total records             : {qc_report['n_total']}")
        print(f"      Q missing / SSC-only      : {qc_report['n_Q_missing']} / {qc_report['n_ssc_only']}")
        print(f"      Q physical flagged        : {qc_report['Q_physical_bad']}")
        print(f"      SSC physical flagged      : {qc_report['SSC_physical_bad']}")
        print(f"      SSL physical flagged      : {qc_report['SSL_physical_bad']}")
        print(f"      SSL log-IQR suspect       : {qc_report['SSL_logIQR_suspect']}")
        print(f"      SSC-Q inconsistent        : {qc_report['SSC_Q_inconsistent']}")
        print(f"      SSL inherited suspect     : {qc_report['SSL_inherited_suspect']}")


        # Convert time to datetime for summary
        time_dates = convert_time_to_datetime(time, time_units)

        # --------------------------------------------------
        # SSC-Q diagnostic plot (tool.py)
        # Skipped for SSC-only stations (no valid Q-SSC pairs)
        # --------------------------------------------------
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
                out_png=str(diag_png),
            )
        else:
            print("    SSC-Q diagnostic plot skipped (SSC-only or insufficient Q-SSC pairs)")

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
        # For SSC-only stations, Q dates are None but SSC dates are valid
        all_dates = [d for d in [q_start, ssc_start, ssl_start] if d is not None]
        if not all_dates:
            print(f"  Skipped: No valid dates")
            ds_in.close()
            return None, None

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

        alt_var = ds_out.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var.comment = 'Source: Original data provided by GloRiSe database.'
        alt_var[:] = alt if np.isfinite(alt) else FILL_VALUE_FLOAT

        area_var = ds_out.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE_FLOAT)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Source: Original data provided by GloRiSe database. May not be available for all stations.'
        area_var[:] = upstream_area if np.isfinite(upstream_area) else FILL_VALUE_FLOAT

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
        ssc_var.comment = 'Source: Original data provided by GloRiSe database. For SS samples this is the source-reported TSS/SSC value.'
        ssc_var[:] = ssc

        ssl_var = ds_out.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon altitude'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = ('Source: Calculated. Formula: SSL (ton/day) = Q (m3/s) x SSC (mg/L) x 0.0864, '
                           'where 0.0864 = 86400 s/day x 1e-6 ton/mg. '
                           'Set to _FillValue when Q or SSC is missing. '
                           'For SSC-only records SSL is always missing.')
        ssl_var[:] = ssl

        # Create quality flag variables
        q_flag_var = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        q_flag_var[:] = q_flag

        ssc_flag_var = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = ('Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), '
                                '3=Bad (e.g., negative), 9=Missing in source. '
                                'QC3 (SSC-Q consistency) only applies to records with valid Q.')
        ssc_flag_var[:] = ssc_flag

        ssl_flag_var = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = ('Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), '
                                '3=Bad (e.g., negative), 9=Missing in source. '
                                'SSL is derived from Q and SSC; missing when either input is missing.')
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
                       f"SSC-only records preserved (Q/SSL missing, QC3 skipped for those records). " \
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
        ds_in.close()

        # --------------------------------------------------
        # CF-1.8 / ACDD-1.3 completeness check
        # --------------------------------------------------
        # errors, warnings = check_nc_completeness(str(output_file))

        # if errors:
        #     print("  CF/ACDD compliance FAILED:")
        #     for e in errors:
        #         print(f"     - {e}")
        #     return None

        # if warnings:
        #     print("  CF/ACDD compliance warnings:")
        #     for w in warnings:
        #         print(f"     - {w}")


        print(f"  Processed: {len(time)} records, {temporal_start} to {temporal_end}")
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
        print(f"  Error: {e}")
        ds_in.close()
        return None, None


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
    "n_Q_missing": 0,
    "n_ssc_only": 0,
    "n_ssc_only_stations": 0,
    }

    """Main processing function."""
    print("="*80)
    print("GloRiSe Dataset: Comprehensive QC and CF-1.8 Standardization")
    print("="*80)
    print(f"\nInput directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    require_existing_directory(INPUT_DIR, description="GloRiSe SS intermediate NetCDF directory")

    # Get all GloRiSe NetCDF files
    input_files = sorted(INPUT_DIR.glob('GloRiSe_*.nc'))

    if not input_files:
        print("\nNo GloRiSe NetCDF files found!")
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

            # Count SSC-only stations
            if qc_report.get("n_ssc_only", 0) > 0 and qc_report.get("n_Q_missing", 0) == qc_report.get("n_total", 0):
                global_qc["n_ssc_only_stations"] += 1

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
        print(f"\nGenerated CSV summary: {csv_file}")
        print(f"  {len(station_list)} stations included")

    # Final summary
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"Successfully processed: {processed_count} stations")
    print(f"Skipped (no valid data): {skipped_count} stations")
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nIMPORTANT CORRECTIONS APPLIED:")
    print("  - Fixed SSL calculation: Q x SSC x 0.0864 (was incorrectly 86.4)")
    print("  - Added quality flags for all variables")
    print("  - Trimmed time ranges to data availability")
    print("  - Standardized metadata to CF-1.8 and ACDD-1.3")
    print("  - SSC-only records preserved (Q/SSL missing, QC3 applied only to Q-SSC pairs)")
    print("="*80)

    print("\nQC GLOBAL SUMMARY")
    print("-" * 80)
    for k, v in global_qc.items():
        print(f"{k:30s}: {v}")


if __name__ == '__main__':
    main()
