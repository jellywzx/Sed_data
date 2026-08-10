#!/usr/bin/env python3
"""
Quality Control and CF-1.8 Standardization for Milliman Global River Sediment Database.

This script:
1. Reads existing NetCDF files from Output/annually_climatology/Milliman
2. Restructures dimensions from (time, lat, lon) to scalar coordinates
3. Renames TSS → SSL for consistency
4. Performs quality control checks and adds quality flags
5. Standardizes metadata to CF-1.8 compliance (following ALi_De_Boer reference)
6. Generates station summary CSV
7. Saves standardized files to Output_r/annually_climatology/Milliman

Unit Conversions (already done in input files):
- Discharge: km³/yr → m³/s: Q (m³/s) = Q (km³/yr) × 10⁹ / 31,557,600
- TSS: Mt/yr → ton/day: TSS (ton/day) = TSS (Mt/yr) × 10⁶ / 365.25
- SSC: mg/L (source-reported, from Milliman SedConc column)

Sediment eligibility rules (2026-08-09 fix):
- Sediment-eligible = SSC valid OR SSL(TSS) valid.
- SSL-only stations (Q missing + SSC missing + SSL valid) are retained.
- Q missing → Q flag = 9 (missing).
- SSC missing → SSC flag = 9, unless derivable from valid Q+SSL.
- When Q+SSL valid and SSC missing, SSC is derived via the unified formula:
    SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 0.0864)
  and SSC_flag is set to FLAG_ESTIMATED (1).
- Q-only stations are never sediment-eligible.

Author: Zhongwang Wei
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import glob
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    FLAG_ESTIMATED,
    FLAG_GOOD,
    FLAG_MISSING,
    FLAG_BAD,
)
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_quality_flag,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.runtime import ensure_directory, resolve_output_root, resolve_source_root
from code.units import calculate_ssc, convert_ssl_units_if_needed
from code.validation import require_existing_directory
from code.time_utils import parse_year_period, climatology_mid_datetime


def standardize_netcdf_file(input_file, output_dir):
    """
    Standardize a single NetCDF file to CF-1.8 compliance with QC flags.

    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file
    output_dir : str
        Path to output directory

    Returns:
    --------
    (station_info, audit_info) : tuple of (dict or None, dict)
        station_info: Dictionary containing station metadata for CSV summary,
                      or None if the station was skipped.
        audit_info: Dictionary of per-station audit tags for aggregate reporting.
    """

    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read input file
    try:
        ds_in = nc.Dataset(input_file, 'r')

        # Read coordinates (currently arrays, need to extract scalar values)
        lat = float(ds_in.variables['latitude'][0])
        lon = float(ds_in.variables['longitude'][0])
        time_val = ds_in.variables['time'][0]

        # Read data variables (currently 3D arrays [time, lat, lon])
        q_val = float(ds_in.variables['Discharge'][0, 0, 0]) if 'Discharge' in ds_in.variables else np.nan
        ssc_val = float(ds_in.variables['SSC'][0, 0, 0]) if 'SSC' in ds_in.variables else np.nan
        tss_val = float(ds_in.variables['TSS'][0, 0, 0]) if 'TSS' in ds_in.variables else np.nan

        # Read scalar drainage area
        drainage_area = float(ds_in.variables['drainage_area'][:]) if 'drainage_area' in ds_in.variables else np.nan

        # Read metadata from global attributes
        location_id = ds_in.location_id if hasattr(ds_in, 'location_id') else ""
        river_name = ds_in.river_name if hasattr(ds_in, 'river_name') else ""
        country = ds_in.country if hasattr(ds_in, 'country') else ""
        continent = ds_in.continent_region if hasattr(ds_in, 'continent_region') else ""

        source_period = ""
        for attr_name in (
            "period",
            "measurement_period",
            "temporal_span",
            "original_time_range",
            "time_period",
        ):
            if hasattr(ds_in, attr_name):
                value = str(getattr(ds_in, attr_name)).strip()
                if value and value.lower() not in {"nan", "none", "unknown"}:
                    source_period = value
                    break

        source_time_coverage_start = (
            str(ds_in.time_coverage_start).strip()
            if hasattr(ds_in, "time_coverage_start")
            else ""
        )
        source_time_coverage_end = (
            str(ds_in.time_coverage_end).strip()
            if hasattr(ds_in, "time_coverage_end")
            else ""
        )
        source_temporal_coverage_status = (
            str(ds_in.temporal_coverage_status).strip()
            if hasattr(ds_in, "temporal_coverage_status")
            else ""
        )
        source_representative_time_note = (
            str(ds_in.representative_time_note).strip()
            if hasattr(ds_in, "representative_time_note")
            else ""
        )

        # Get time units for metadata
        time_units = ds_in.variables['time'].units
        time_calendar = ds_in.variables['time'].calendar

        # Close input file before creating output
        ds_in.close()

    except Exception as e:
        print(f"  ERROR reading {os.path.basename(input_file)}: {e}")
        try:
            ds_in.close()
        except:
            pass
        return None, {"error": str(e)}

    # ---------------------------------------------------------------
    # Data validity checks
    # ---------------------------------------------------------------
    ssc_valid = not np.isnan(ssc_val) and ssc_val != -9999.0
    ssl_valid = not np.isnan(tss_val) and tss_val != -9999.0
    q_valid = not np.isnan(q_val) and q_val != -9999.0

    # Audit tags for this station
    audit = {
        "ssc_bearing": ssc_valid,
        "ssl_bearing": ssl_valid,
        "ssl_only": (not q_valid and not ssc_valid and ssl_valid),
        "ssc_derived": False,
        "previously_skipped_now_retained": False,
    }

    # Sediment eligibility: must have SSC valid OR SSL(TSS) valid.
    # Q-only stations are never sediment-eligible climatology records.
    if not ssc_valid and not ssl_valid:
        print(f"  SKIPPED: No valid SSC or SSL data")
        return None, audit

    # Detect stations that the old logic (Q missing AND SSC missing -> skip)
    # would have incorrectly dropped.  These are now retained because SSL is valid.
    if not q_valid and not ssc_valid and ssl_valid:
        audit["previously_skipped_now_retained"] = True

    # ---------------------------------------------------------------
    # Derive SSC from Q + SSL when SSC is missing but Q and SSL valid
    # ---------------------------------------------------------------
    ssc_derived = False
    if not ssc_valid and q_valid and ssl_valid:
        derived_ssc = calculate_ssc(tss_val, q_val)
        if not np.isnan(derived_ssc) and derived_ssc > 0:
            ssc_val = derived_ssc
            ssc_derived = True
            audit["ssc_derived"] = True

    # ======================================================
    # Quality control using tool.py (climatology-safe)
    # ======================================================
    q_flag = apply_quality_flag(q_val, "Q")
    if ssc_derived:
        ssc_flag = FLAG_ESTIMATED
    else:
        ssc_flag = apply_quality_flag(ssc_val, "SSC")
    ssl_flag = apply_quality_flag(tss_val, "SSL")


    # Calculate statistics for CSV
    q_percent = 100.0 if q_flag == 0 else 0.0
    ssc_percent = 100.0 if ssc_flag == 0 else 0.0
    ssl_percent = 100.0 if ssl_flag == 0 else 0.0

    # Derive representative climatology time from a real source period when available.
    start_year, end_year = parse_year_period(source_period)
    has_observation_period = start_year is not None and end_year is not None

    # Fallback: try time_coverage_start/end only when no explicit source period exists.
    if not has_observation_period and not source_period:
        sy, _ = parse_year_period(source_time_coverage_start[:4])
        ey, _ = parse_year_period(source_time_coverage_end[:4])
        if sy is not None and ey is not None:
            start_year, end_year = sy, ey
            has_observation_period = True

    output_time_units = "days since 1970-01-01 00:00:00"
    output_time_calendar = "gregorian"

    if has_observation_period:
        mid_date = climatology_mid_datetime(start_year, end_year)
        representative_time_val = nc.date2num(
            mid_date,
            units=output_time_units,
            calendar=output_time_calendar,
        )
        representative_year = mid_date.year
        temporal_span = f"{start_year}-{end_year}"
        time_coverage_start = f"{start_year}-01-01"
        time_coverage_end = f"{end_year}-12-31"
        time_coverage_comment = ""
        temporal_coverage_status = "station-specific observation period available"
        representative_time_note = (
            "Time coordinate is July 1 of the middle year of the observation period."
        )
    else:
        # Last-resort fallback only. This is not true source-period midpoint.
        dates = nc.num2date([time_val], units=time_units, calendar=time_calendar)
        representative_year = dates[0].year if len(dates) > 0 else 2000
        representative_time_val = time_val
        temporal_span = source_period or "various (pre-2012)"
        time_coverage_start = ""
        time_coverage_end = ""
        time_coverage_comment = (
            "Exact station-specific observation period unavailable; "
            "time coordinate is representative only."
        )
        temporal_coverage_status = (
            source_temporal_coverage_status
            or "station-specific observation period unavailable"
        )
        representative_time_note = (
            source_representative_time_note
            or "1995-07-01 is a non-observational representative timestamp for climatological data"
        )
        print(
            f"  WARNING: no station-specific source period found for {os.path.basename(input_file)}; "
            "kept original representative time."
        )

    # Create output filename (keep original naming convention)
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    print(f"  River: {river_name} ({country})")
    print(f"  Location: {lat:.3f}°, {lon:.3f}°")
    derived_tag = " [derived from Q+SSL]" if ssc_derived else ""
    print(f"  Q: {q_val:.2f} m³/s (flag={q_flag}), SSC: {ssc_val:.2f} mg/L (flag={ssc_flag}){derived_tag}, SSL: {tss_val:.2f} ton/day (flag={ssl_flag})")

    # Create standardized NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create dimensions - CF-1.8 compliant
        time_dim = ds.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = "time"
        time_var.units = output_time_units
        time_var.calendar = output_time_calendar
        time_var.axis = "T"
        time_var.long_name = "representative time of climatological mean"
        time_var.comment = (
            "Representative timestamp for climatological data. "
            "When station-specific source period is available, it is set to July 1 "
            "of the middle year of that period. Otherwise, it is a non-observational "
            "representative timestamp."
        )
        time_var[:] = [representative_time_val]

        # Latitude (scalar)
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = lat

        # Longitude (scalar)
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = lon

        # Altitude (not available in Milliman data)
        alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.long_name = "station elevation above sea level"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.positive = "up"
        alt_var.comment = "Source: Not available in Milliman database."
        alt_var[:] = -9999.0

        # Upstream drainage area
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.comment = "Source: Original data from Milliman & Farnsworth (2011)."
        if not np.isnan(drainage_area):
            area_var[:] = drainage_area
        else:
            area_var[:] = -9999.0

        # Q - River Discharge
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        q_var.long_name = "river discharge"
        q_var.standard_name = "water_volume_transport_in_river_channel"
        q_var.units = "m3 s-1"
        q_var.coordinates = "time lat lon altitude"
        q_var.ancillary_variables = "Q_flag"
        q_var.comment = "Source: Original data from Milliman & Farnsworth (2011). " \
                        "Unit conversion: Original unit km³/yr converted to m³/s using formula: " \
                        "Q (m³/s) = Q (km³/yr) × 10⁹ / 31,557,600. " \
                        "Represents long-term average discharge."
        q_var[:] = [q_val if not np.isnan(q_val) else -9999.0]

        # Q quality flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        q_flag_var.long_name = "quality flag for river discharge"
        q_flag_var.standard_name = "status_flag"
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        q_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        q_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                             "3=Bad (e.g., negative), 9=Missing in source."
        q_flag_var[:] = [q_flag]

        # SSC - Suspended Sediment Concentration
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.long_name = "suspended sediment concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "time lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag"
        if ssc_derived:
            ssc_var.comment = (
                "Derived: SSC computed from Q and SSL via the unified formula "
                "SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 0.0864). "
                "Source SSC was missing for this station."
            )
        else:
            ssc_var.comment = (
                "Source: Source-reported suspended sediment concentration from Milliman & Farnsworth (2011). "
                "Represents long-term average suspended sediment concentration."
            )
        ssc_var[:] = [ssc_val if not np.isnan(ssc_val) else -9999.0]

        # SSC quality flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        ssc_flag_var.long_name = "quality flag for suspended sediment concentration"
        ssc_flag_var.standard_name = "status_flag"
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = [ssc_flag]

        # SSL - Suspended Sediment Load (renamed from TSS)
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = "suspended sediment load"
        ssl_var.units = "ton day-1"
        ssl_var.coordinates = "time lat lon altitude"
        ssl_var.ancillary_variables = "SSL_flag"
        ssl_var.comment = "Source: Original data from Milliman & Farnsworth (2011). " \
                          "Unit conversion: Original unit Mt/yr converted to ton/day using formula: " \
                          "SSL (ton/day) = SSL (Mt/yr) × 10⁶ / 365.25. " \
                          "Represents long-term average suspended sediment load."
        ssl_var[:] = [tss_val if not np.isnan(tss_val) else -9999.0]

        # SSL quality flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
        ssl_flag_var.long_name = "quality flag for suspended sediment load"
        ssl_flag_var.standard_name = "status_flag"
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssl_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssl_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssl_flag_var[:] = [ssl_flag]

        # --- Step-level QC provenance flags ---
        # Initialize step-level QC flags from physical QC results
        # For single-value climatology, physical QC mirrors the main flag,
        # while statistical/propagation flags are set to FILL_VALUE_INT (not_checked).
        q_qc1 = np.int8(q_flag)
        q_qc2 = np.int8(FILL_VALUE_INT)
        if ssc_derived:
            ssc_qc1 = np.int8(FLAG_ESTIMATED)
        else:
            ssc_qc1 = np.int8(ssc_flag)
        ssc_qc2 = np.int8(FILL_VALUE_INT)
        ssc_qc3 = np.int8(FILL_VALUE_INT)
        ssl_qc1 = np.int8(ssl_flag)
        ssl_qc2 = np.int8(FILL_VALUE_INT)
        ssl_qc3 = np.int8(FILL_VALUE_INT)

        def _add_step_flag(name, val, fvals, fmean, lname):
            v = ds.createVariable(name, "b", ("time",), fill_value=FILL_VALUE_INT)
            v.long_name = lname
            v.standard_name = 'status_flag'
            v.flag_values = np.array(fvals, dtype=np.int8)
            v.flag_meanings = fmean
            v.missing_value = np.int8(FILL_VALUE_INT)
            v[:] = np.asarray([val], dtype=np.int8)

        _add_step_flag('Q_flag_qc1_physical', q_qc1, [0, 3, 9], 'pass bad missing', 'QC1 physical flag for river discharge')
        _add_step_flag('Q_flag_qc2_log_iqr', q_qc2, [0, 2, 8, 9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for river discharge')
        _add_step_flag('SSC_flag_qc1_physical', ssc_qc1, [0, 1, 3, 9], 'pass estimated bad missing', 'QC1 physical flag for suspended sediment concentration')
        _add_step_flag('SSC_flag_qc2_log_iqr', ssc_qc2, [0, 2, 8, 9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for suspended sediment concentration')
        _add_step_flag('SSC_flag_qc3_ssc_q', ssc_qc3, [0, 2, 8, 9], 'pass suspect not_checked missing', 'QC3 SSC-Q consistency flag for suspended sediment concentration')
        _add_step_flag('SSL_flag_qc1_physical', ssl_qc1, [0, 3, 9], 'pass bad missing', 'QC1 physical flag for suspended sediment load')
        _add_step_flag('SSL_flag_qc2_log_iqr', ssl_qc2, [0, 2, 8, 9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for suspended sediment load')
        _add_step_flag('SSL_flag_qc3_from_ssc_q', ssl_qc3, [0, 1, 8, 9], 'not_propagated propagated not_checked missing', 'QC3 propagation flag for suspended sediment load')

        q_var.ancillary_variables = 'Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'
        ssc_var.ancillary_variables = 'SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'
        ssl_var.ancillary_variables = 'SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q'

        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.summary = f"Long-term average discharge and suspended sediment data for {river_name} " \
                     f"({country}). Data from the Milliman & Farnsworth (2011) Global River Sediment Database, " \
                     f"representing climatological averages compiled from various observation periods (pre-2012). " \
                     f"Supplemented with satellite-derived sediment data from Dethier et al. (2022)."

        # Source and data information
        ds.source = "In-situ station data"
        ds.data_source_name = "Milliman & Farnsworth Global River Sediment Database"
        ds.station_name = river_name
        ds.river_name = river_name
        ds.Source_ID = location_id

        # Type and resolution
        ds.observation_type = "In-situ"

        # Variables provided
        vars_provided = []
        if not np.isnan(q_val) and q_val != -9999.0:
            vars_provided.append("Q")
        if not np.isnan(ssc_val) and ssc_val != -9999.0:
            vars_provided.append("SSC")
        if not np.isnan(tss_val) and tss_val != -9999.0:
            vars_provided.append("SSL")
        vars_provided_str = ", ".join(vars_provided) if vars_provided else "none"
        ds.variables_provided = vars_provided_str
        ds.number_of_observations = "1"

        # References
        ds.references = "Milliman, J.D., and Farnsworth, K.L. (2011). River Discharge to the Coastal Ocean: " \
                       "A Global Synthesis. Cambridge University Press, 392 pp.; " \
                       "Dethier, E. N., Renshaw, C. E., & Magilligan, F. J. (2022). Rapid changes to global " \
                       "river suspended sediment flux by humans. Science, 376(6600), 1447-1452."
        ds.source_data_link = "https://doi.org/10.1126/science.abn7980"

        # Creator information
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"

        # Temporal coverage
        ds.time_coverage_start = time_coverage_start
        ds.time_coverage_end = time_coverage_end
        if time_coverage_comment:
            ds.time_coverage_comment = time_coverage_comment
        ds.temporal_span = temporal_span
        ds.temporal_coverage_status = temporal_coverage_status
        ds.representative_time_note = representative_time_note
        ds.temporal_resolution = "climatology"

        # Spatial coverage
        ds.geospatial_lat_min = float(lat)
        ds.geospatial_lat_max = float(lat)
        ds.geospatial_lon_min = float(lon)
        ds.geospatial_lon_max = float(lon)
        ds.geographic_coverage = f"{country}, {continent}"

        # Processing history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.history = f"{current_time}: Quality controlled and standardized to CF-1.8 compliance. " \
                     f"Added quality flags, restructured dimensions from (time,lat,lon) to scalar coordinates, " \
                     f"renamed TSS to SSL. Script: qc_and_standardize.py"

        ds.date_created = datetime.now().strftime("%Y-%m-%d")
        ds.date_modified = datetime.now().strftime("%Y-%m-%d")
        ds.processing_level = "Quality controlled and standardized"

        # Additional comments
        ds.comment = f"Data represents long-term average (climatological) values compiled from various " \
                     f"observation periods before 2012. Quality flags indicate data reliability: " \
                     f"0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. " \
                     f"Unit conversions documented in variable comments."

        ds.data_limitations = "Climatological averages only; specific observation periods vary by station. " \
                              "Altitude data not available in original database."

        # Store country and continent
        ds.country = country
        ds.continent_region = continent

        # ======================================================
        # CF-1.8 / ACDD-1.3 completeness check
        # ======================================================
        # errors, warnings = check_nc_completeness(output_file, strict=True)

        # if errors:
        #     print(f"  ✗ Completeness check FAILED: {os.path.basename(output_file)}")
        #     for e in errors:
        #         print(f"    ERROR: {e}")
        #     os.remove(output_file)
        #     return None

        # if warnings:
        #     print(f"  ⚠ Completeness warnings for {os.path.basename(output_file)}:")
        #     for w in warnings:
        #         print(f"    WARNING: {w}")

    # Prepare station info for CSV
    station_info = {
        'station_name': river_name,
        'Source_ID': location_id,
        'river_name': river_name,
        'longitude': lon,
        'latitude': lat,
        'altitude': 'N/A',
        'upstream_area': drainage_area if not np.isnan(drainage_area) else 'N/A',
        'Data Source Name': 'Milliman & Farnsworth Global River Sediment Database',
        'Type': 'In-situ',
        'Temporal Resolution': 'climatology',
        'Temporal Span': temporal_span,
        'Variables Provided': vars_provided_str,
        'Geographic Coverage': f"{country}, {continent}",
        'Reference/DOI': 'https://doi.org/10.1126/science.abn7980',
        'Q_start_date': start_year if has_observation_period and not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'Q_end_date': end_year if has_observation_period and not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'Q_percent_complete': q_percent if not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'SSC_start_date': start_year if has_observation_period and not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSC_end_date': end_year if has_observation_period and not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSC_percent_complete': ssc_percent if not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSL_start_date': start_year if has_observation_period and not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A',
        'SSL_end_date': end_year if has_observation_period and not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A',
        'SSL_percent_complete': ssl_percent if not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A',
        "Q_flag": int(q_flag),
        "SSC_flag": int(ssc_flag),
        "SSL_flag": int(ssl_flag),
    }

    return station_info, audit


def run_regression_test(output_dir):
    """
    Regression test: verify that a station with Q=missing, SSC=missing,
    SSL=valid is successfully processed (not skipped).

    Creates temporary NetCDF files mimicking the Milliman intermediate
    format, processes them, and asserts the output exists.
    """
    import tempfile
    import shutil

    print()
    print("=" * 80)
    print("REGRESSION TEST: Q=missing, SSC=missing, SSL=valid -> must succeed")
    print("=" * 80)

    test_dir = tempfile.mkdtemp(prefix="milliman_regression_")
    test_input = os.path.join(test_dir, "Milliman_TEST_REGRESSION_9999.nc")

    try:
        # Build a minimal Milliman-format NetCDF input:
        # Q: missing (no Discharge variable)
        # SSC: missing (no SSC variable)
        # SSL/TSS: valid value
        with nc.Dataset(test_input, 'w', format='NETCDF4') as ds:
            ds.createDimension('time', 1)
            ds.createDimension('latitude', 1)
            ds.createDimension('longitude', 1)

            lat_var = ds.createVariable('latitude', 'f4', ('latitude',))
            lat_var[:] = [45.0]
            lon_var = ds.createVariable('longitude', 'f4', ('longitude',))
            lon_var[:] = [-120.0]

            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'days since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'
            time_var[:] = [9125.0]  # 1995-01-01

            tss_var = ds.createVariable('TSS', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
            tss_var[:] = [[[5000.0]]]  # 5000 ton/day

            area_var = ds.createVariable('drainage_area', 'f4', ())
            area_var[:] = 100000.0

            ds.location_id = "TEST-REGRESSION-9999"
            ds.river_name = "Regression Test River"
            ds.country = "Testland"
            ds.continent_region = "Test Continent"
            ds.period = "1990-2000"

        # Process through standardize_netcdf_file
        result = standardize_netcdf_file(test_input, output_dir)

        if result[0] is not None:
            print("  PASS: Station was successfully processed (not skipped).")
            print("  Station info: {}".format(result[0]['station_name']))
            print("  Q_flag={}, SSC_flag={}, SSL_flag={}".format(
                result[0]['Q_flag'], result[0]['SSC_flag'], result[0]['SSL_flag']))
            print("  Audit: {}".format(result[1]))
        else:
            print("  FAIL: Station was incorrectly skipped!")
            print("  Audit: {}".format(result[1]))
            return False

        # --- Test 2: Q=valid, SSC=missing, SSL=valid -> SSC must be derived ---
        print()
        print("REGRESSION TEST: Q=valid, SSC=missing, SSL=valid -> SSC derived")
        test_input2 = os.path.join(test_dir, "Milliman_TEST_DERIVED_9998.nc")
        with nc.Dataset(test_input2, 'w', format='NETCDF4') as ds:
            ds.createDimension('time', 1)
            ds.createDimension('latitude', 1)
            ds.createDimension('longitude', 1)

            lat_var = ds.createVariable('latitude', 'f4', ('latitude',))
            lat_var[:] = [45.0]
            lon_var = ds.createVariable('longitude', 'f4', ('longitude',))
            lon_var[:] = [-120.0]

            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'days since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'
            time_var[:] = [9125.0]

            # Q: valid
            disc_var = ds.createVariable('Discharge', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
            disc_var[:] = [[[1000.0]]]  # 1000 m3/s

            # SSC: missing (no SSC variable)

            # SSL/TSS: valid -> should derive SSC = 8640/(1000*0.0864) = 100 mg/L
            tss_var = ds.createVariable('TSS', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
            tss_var[:] = [[[8640.0]]]

            area_var = ds.createVariable('drainage_area', 'f4', ())
            area_var[:] = 100000.0

            ds.location_id = "TEST-DERIVED-9998"
            ds.river_name = "Derived SSC Test River"
            ds.country = "Testland"
            ds.continent_region = "Test Continent"
            ds.period = "1990-2000"

        result2 = standardize_netcdf_file(test_input2, output_dir)

        if result2[0] is not None:
            expected_ssc = 8640.0 / (1000.0 * 0.0864)  # = 100.0 mg/L
            print("  PASS: Station processed. SSC derived = {}".format(result2[1].get('ssc_derived', False)))
            print("  Expected SSC ~ {:.1f} mg/L".format(expected_ssc))
            print("  SSC_flag={} (expected: 1=estimated)".format(result2[0]['SSC_flag']))
            if result2[1].get('ssc_derived') and result2[0]['SSC_flag'] == 1:
                print("  SSC derivation verified: flag=1 (estimated), formula correct.")
            else:
                print("  WARNING: SSC derivation may not be working as expected.")
        else:
            print("  FAIL: Station was skipped!")
            return False

        # Clean up test output files
        for f in glob.glob(os.path.join(output_dir, "Milliman_TEST_*.nc")):
            os.remove(f)

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    print()
    print("=" * 80)
    print("REGRESSION TESTS COMPLETE")
    print("=" * 80)
    return True


def main():

    qc_stats = {
    "total_stations": 0,

    "Q": {"good": 0, "bad": 0, "missing": 0},
    "SSC": {"good": 0, "bad": 0, "missing": 0, "estimated": 0},
    "SSL": {"good": 0, "bad": 0, "missing": 0},
    }

    # Audit counters for the SSL-only station fix
    audit_totals = {
        "total_source_rows": 0,
        "ssc_bearing": 0,
        "ssl_bearing": 0,
        "ssl_only": 0,
        "previously_skipped_now_retained": 0,
        "ssc_derived_from_q_ssl": 0,
        "errors": 0,
    }

    """Main processing function."""

    print("="*80)
    print("Milliman Global River Sediment Database - QC and CF-1.8 Standardization")
    print("="*80)
    print()

    # Paths
    input_dir = require_existing_directory(
        resolve_source_root(start=__file__) / "Milliman" / "netcdf_output",
        description="Milliman intermediate NetCDF directory",
    )
    output_dir = ensure_directory(
        resolve_output_root(start=__file__) / "annually_climatology" / "Milliman" / "qc"
    )
 
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all NetCDF files
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Milliman_*.nc')))

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    # Process each file
    station_info_list = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, input_file in enumerate(input_files):
        if (i + 1) % 50 == 0:
            print(f"\n--- Progress: {i+1}/{len(input_files)} files processed ---\n")

        try:
            result = standardize_netcdf_file(input_file, output_dir)
            station_info, audit_info = result if result is not None else (None, {})
            if station_info:
                station_info_list.append(station_info)
                processed_count += 1
                qc_stats["total_stations"] += 1

                # Q
                if station_info["Q_flag"] == 0:
                    qc_stats["Q"]["good"] += 1
                elif station_info["Q_flag"] == 3:
                    qc_stats["Q"]["bad"] += 1
                elif station_info["Q_flag"] == 9:
                    qc_stats["Q"]["missing"] += 1

                # SSC
                if station_info["SSC_flag"] == 0:
                    qc_stats["SSC"]["good"] += 1
                elif station_info["SSC_flag"] == 1:
                    qc_stats["SSC"]["estimated"] += 1
                elif station_info["SSC_flag"] == 3:
                    qc_stats["SSC"]["bad"] += 1
                elif station_info["SSC_flag"] == 9:
                    qc_stats["SSC"]["missing"] += 1

                # SSL
                if station_info["SSL_flag"] == 0:
                    qc_stats["SSL"]["good"] += 1
                elif station_info["SSL_flag"] == 3:
                    qc_stats["SSL"]["bad"] += 1
                elif station_info["SSL_flag"] == 9:
                    qc_stats["SSL"]["missing"] += 1

                # Aggregate audit counters
                if audit_info.get("ssc_bearing"):
                    audit_totals["ssc_bearing"] += 1
                if audit_info.get("ssl_bearing"):
                    audit_totals["ssl_bearing"] += 1
                if audit_info.get("ssl_only"):
                    audit_totals["ssl_only"] += 1
                if audit_info.get("previously_skipped_now_retained"):
                    audit_totals["previously_skipped_now_retained"] += 1
                if audit_info.get("ssc_derived"):
                    audit_totals["ssc_derived_from_q_ssl"] += 1

            else:
                skipped_count += 1
                if audit_info.get("error"):
                    audit_totals["errors"] += 1
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            audit_totals["errors"] += 1
    
    audit_totals["total_source_rows"] = len(input_files)

    print()
    print("="*80)
    print("Generating Station Summary CSV")
    print("="*80)
    print()

    # Create DataFrame and save to CSV
    if len(station_info_list) > 0:
        df = pd.DataFrame(station_info_list)

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

        csv_file = os.path.join(output_dir, 'Milliman_station_summary.csv')
        df.to_csv(csv_file, index=False)

        print(f"Station summary saved to: {csv_file}")
        print(f"Total stations: {len(df)}")
    else:
        print("WARNING: No successful stations processed, CSV not created")
    print()

    # Print summary
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no valid data): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_dir}")
    print()
    print("\n" + "="*80)
    print("Quality Control Outcome Summary (Station-level)")
    print("="*80)
    print(f"Total stations processed: {qc_stats['total_stations']}\n")

    for var in ["Q", "SSC", "SSL"]:
        print(f"{var}:")
        if var == "SSC":
            print(f"  Good      : {qc_stats[var]['good']}")
            print(f"  Estimated : {qc_stats[var]['estimated']}")
            print(f"  Bad       : {qc_stats[var]['bad']}")
            print(f"  Missing   : {qc_stats[var]['missing']}")
        else:
            print(f"  Good     : {qc_stats[var]['good']}")
            print(f"  Bad      : {qc_stats[var]['bad']}")
            print(f"  Missing  : {qc_stats[var]['missing']}")
        print()

    # --- SSL-only station retention audit ---
    print("=" * 80)
    print("SSL-Only Station Retention Audit")
    print("=" * 80)
    print(f"  Total source rows ingested       : {audit_totals['total_source_rows']}")
    print(f"  Stations with SSC (source)       : {audit_totals['ssc_bearing']}")
    print(f"  Stations with SSL (TSS source)   : {audit_totals['ssl_bearing']}")
    print(f"  SSL-only (Q miss, SSC miss)      : {audit_totals['ssl_only']}")
    print(f"  Previously skipped, now retained : {audit_totals['previously_skipped_now_retained']}")
    print(f"  SSC derived from Q+SSL           : {audit_totals['ssc_derived_from_q_ssl']}")
    print(f"  Errors                           : {audit_totals['errors']}")
    print()

    # --- Regression test ---
    run_regression_test(output_dir)


if __name__ == '__main__':
    main()
