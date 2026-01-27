#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert HMA catchments CSV data to CF-1.8 compliant NetCDF format with QC
Following CF-1.8 conventions with comprehensive quality control and metadata

Author: Zhongwang Wei
Institution: Sun Yat-sen University, China
Email: weizhw6@mail.sysu.edu.cn
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import os
import re
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    calculate_discharge,
    apply_quality_flag,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    # convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl,
)

def parse_period(period_str):
    """
    Extract start and end year from period string like '1957-2017'

    Args:
        period_str: String containing year range

    Returns:
        tuple: (start_year, end_year) or (None, None) if invalid
    """
    if pd.isna(period_str) or period_str.strip() == '':
        return None, None
    period_str = period_str.strip()
    match = re.search(r'(\d{4})-(\d{4})', period_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def convert_Q_to_discharge(Q_km3_yr):
    """
    Convert discharge from km³/yr to m³/s

    Formula: Q(m³/s) = Q(km³/yr) × 10⁹ / 31,557,600

    Where:
    - 1 km³ = 10⁹ m³
    - 1 year = 365.25 days × 86,400 s/day = 31,557,600 s

    Args:
        Q_km3_yr: Discharge in km³/yr

    Returns:
        float: Discharge in m³/s or -9999.0 if missing
    """
    if pd.isna(Q_km3_yr):
        return -9999.0
    return Q_km3_yr * 1e9 / 31557600.0


def convert_Qs_to_SSL(Qs_Mt_yr):
    """
    Convert sediment load from Mt/yr to ton/day

    Formula: SSL(ton/day) = Qs(Mt/yr) × 10⁶ / 365.25

    Where:
    - 1 Mt = 10⁶ ton
    - 1 year = 365.25 days

    Args:
        Qs_Mt_yr: Sediment load in Mt/yr

    Returns:
        float: Sediment load in ton/day or -9999.0 if missing
    """
    if pd.isna(Qs_Mt_yr):
        return -9999.0
    return Qs_Mt_yr * 1e6 / 365.25


def calculate_SSC(SSL_ton_day, Q_m3_s):
    """
    Calculate SSC from sediment load and discharge

    Formula: SSC(mg/L) = SSL(ton/day) / (Q(m³/s) × 86.4)

    Derivation:
    - SSL = Q × SSC × conversion_factor
    - Q (m³/s) × SSC (mg/L) × 86400 s/day × 1000 L/m³ × 10⁻⁶ ton/mg = SSL (ton/day)
    - Therefore: conversion_factor = 86.4

    Args:
        SSL_ton_day: Sediment load in ton/day
        Q_m3_s: Discharge in m³/s

    Returns:
        float: SSC in mg/L or -9999.0 if missing/invalid
    """
    if SSL_ton_day == -9999.0 or Q_m3_s == -9999.0 or Q_m3_s == 0:
        return -9999.0
    return SSL_ton_day / (Q_m3_s * 0.0864)


def calculate_SSL_from_yield(sediment_yield_t_km2_yr, basin_area_km2):
    """
    Calculate sediment load from sediment yield and basin area

    Formula: SSL(ton/day) = sediment_yield(t/km²/yr) × basin_area(km²) / 365.25

    Args:
        sediment_yield_t_km2_yr: Sediment yield in t/km²/yr
        basin_area_km2: Basin area in km²

    Returns:
        float: Sediment load in ton/day or -9999.0 if missing
    """
    if pd.isna(sediment_yield_t_km2_yr) or pd.isna(basin_area_km2):
        return -9999.0
    annual_load_ton = sediment_yield_t_km2_yr * basin_area_km2
    daily_load_ton = annual_load_ton / 365.25
    return daily_load_ton


def parse_value_with_uncertainty(value_str):
    """
    Parse values like '1.02 ± 0.29' and return the mean value

    Args:
        value_str: String containing value with optional uncertainty

    Returns:
        float: Parsed value or np.nan if invalid
    """
    if pd.isna(value_str):
        return np.nan
    value_str = str(value_str).strip()
    if value_str == '' or value_str == 'nan':
        return np.nan
    # Extract the first number before ±
    match = re.match(r'([\d.]+)', value_str)
    if match:
        return float(match.group(1))
    return np.nan


def apply_hma_climatology_qc(
    Q,
    SSC,
    SSL,
    time,
    station_id,
    station_name,
    output_dir,
    min_samples=5
):
    
    """
    HMA climatological QC using tool.py logic, with explicit
    statistical diagnostics and conditional SSC–Q plotting.

    Parameters
    ----------
    Q, SSC, SSL : float
        Climatological values
    time : array-like of datetime
        Time coordinate (length = number of samples)
    station_id : str
    station_name : str
    output_dir : str
        Base output directory
    min_samples : int
        Minimum samples required for statistical QC and plotting

    Returns
    -------
    Q_flag, SSC_flag, SSL_flag : int
    """

    # ============================================================
    # 1. Physical QC (always applied)
    # ============================================================
    Q_flag = apply_quality_flag(Q, "Q")
    SSC_flag = apply_quality_flag(SSC, "SSC")
    SSL_flag = apply_quality_flag(SSL, "SSL")

    # ============================================================
    # 2. Prepare arrays
    # ============================================================
    Q_arr = np.atleast_1d(Q).astype(float)
    SSC_arr = np.atleast_1d(SSC).astype(float)
    time_arr = np.atleast_1d(time)

    n = len(Q_arr)

    # ============================================================
    # 3. Log-IQR statistical QC (explicit decision)
    # ============================================================
    iqr_lower, iqr_upper = compute_log_iqr_bounds(SSC_arr)

    if iqr_lower is None:
        print(
            f"  [✓] [{station_name} +] "
            f"Sample size = {n} < {min_samples}, log-IQR statistical QC skipped."
        )
    # ============================================================
    # 4. SSC–Q envelope & consistency
    # ============================================================
    ssc_q_bounds = build_ssc_q_envelope(
        Q_arr, SSC_arr, min_samples=min_samples
    )

    if ssc_q_bounds is None:
        print(
            f"  [✓] [{station_name} +] "
            f"Sample size = {n} < {min_samples}, SSC–Q consistency check and diagnostic plot skipped."
        )

    else:
        # Consistency check (theoretically unreachable for climatology,
        # but keeps logic unified with time-series workflow)
        is_inconsistent, resid = check_ssc_q_consistency(
            Q, SSC, Q_flag, SSC_flag, ssc_q_bounds
        )

        ssc_q_inconsistent, resid = check_ssc_q_consistency(
            Q, SSC, Q_flag, SSC_flag, ssc_q_bounds
        )

        if ssc_q_inconsistent and SSC_flag == 0:
            SSC_flag = np.int8(2)  # suspect
            SSL_flag = propagate_ssc_q_inconsistency_to_ssl(
                inconsistent=ssc_q_inconsistent,
                Q=Q,
                SSC=SSC,
                SSL=SSL,
                Q_flag=Q_flag,
                SSC_flag=SSC_flag,
                SSL_flag=SSL_flag,
                ssl_is_derived_from_q_ssc=False,
            )

            print(
                f"  ⚠️  [{station_name}] "
                "SSC–Q inconsistency detected (SSC_flag->2, SSL_flag propagated)."
            )


        # ========================================================
        # 5. SSC–Q diagnostic plot (only when statistics valid)
        # ========================================================
        diag_dir = os.path.join(output_dir, "qc_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        out_png = os.path.join(
            diag_dir,
            f"{station_id}_ssc_q_diagnostic.png"
        )

        plot_ssc_q_diagnostic(
            time=time_arr,
            Q=Q_arr,
            SSC=SSC_arr,
            Q_flag=np.atleast_1d(Q_flag),
            SSC_flag=np.atleast_1d(SSC_flag),
            ssc_q_bounds=ssc_q_bounds,
            station_id=station_id,
            station_name=station_name,
            out_png=out_png,
        )

        print(
            f"  📈 [{station_name}] "
            f"SSC–Q diagnostic plot saved: {out_png}"
        )

    return Q_flag, SSC_flag, SSL_flag


def extract_source_id(station_name):
    """
    Extract Source_ID from station name like 'Changmapu (S1)'

    Args:
        station_name: Station name string

    Returns:
        str: Source ID (e.g., 'S1') or None
    """
    match = re.search(r'\(S(\d+)\)', station_name)
    if match:
        return f"S{match.group(1)}"
    return None


def create_netcdf_for_station(station_data, output_dir, data_source_csv):
    """
    Create a CF-1.8 compliant NetCDF file for a single station with QC

    Args:
        station_data: pandas Series containing station data
        output_dir: Output directory path
        data_source_csv: Path to original CSV file

    Returns:
        dict: Processing summary or None if skipped
    """
    station_name = station_data['Stations']

    # Extract Source_ID
    source_id = extract_source_id(station_name)
    if source_id is None:
        print(f"Warning: Could not extract Source_ID for {station_name}, skipping...")
        return None

    # Clean station name for filename
    safe_station_name = re.sub(r'[^\w\s-]', '', station_name)
    safe_station_name = re.sub(r'\s+', '_', safe_station_name.strip())
    safe_station_name = re.sub(r'_\(S\d+\)', '', safe_station_name)
    safe_station_name = re.sub(r'\+', '', safe_station_name)

    filename = f"HMA_{safe_station_name}.nc"
    filepath = os.path.join(output_dir, filename)

    # Parse values from CSV
    longitude = parse_value_with_uncertainty(station_data['Longitude'])
    latitude = parse_value_with_uncertainty(station_data['Latitude'])
    basin_area = parse_value_with_uncertainty(station_data['Basin area (km2)'])
    glacier_cover = parse_value_with_uncertainty(station_data.get('Glacier cover (%)'))
    permafrost_cover = parse_value_with_uncertainty(station_data.get('Permafrost cover (%)'))
    Q_km3_yr = parse_value_with_uncertainty(station_data['Q (km3/yr)'])
    Qs_Mt_yr = parse_value_with_uncertainty(station_data['Qs (Mt/yr)'])
    sediment_yield = parse_value_with_uncertainty(station_data['sediment yield\xa0（(t/km2/y)）'])

    # Get periods
    period_Q = station_data['Period for Q']
    period_Qs = station_data['Period for Qs']
    start_year_Q, end_year_Q = parse_period(period_Q)
    start_year_Qs, end_year_Qs = parse_period(period_Qs)

    # Determine primary period
    if start_year_Q is not None and end_year_Q is not None:
        start_year = start_year_Q
        end_year = end_year_Q
        period_str = f"{start_year_Q}-{end_year_Q}"
    elif start_year_Qs is not None and end_year_Qs is not None:
        start_year = start_year_Qs
        end_year = end_year_Qs
        period_str = f"{start_year_Qs}-{end_year_Qs}"
    else:
        print(f"Warning: No valid period for {station_name}, skipping...")
        return None

    # Convert units
    Q = convert_Q_to_discharge(Q_km3_yr)

    # Calculate SSL: priority - from Qs, then from sediment yield
    SSL_source = ""
    if not np.isnan(Qs_Mt_yr):
        SSL = convert_Qs_to_SSL(Qs_Mt_yr)
        SSL_source = f"Calculated. Formula: SSL (ton/day) = Qs (Mt/yr) × 10⁶ / 365.25. Original Qs: {Qs_Mt_yr} Mt/yr. Represents mean annual value over period {period_str}."
    elif not np.isnan(sediment_yield):
        SSL = calculate_SSL_from_yield(sediment_yield, basin_area)
        SSL_source = f"Calculated. Formula: SSL (ton/day) = sediment_yield (t/km²/yr) × basin_area (km²) / 365.25. Original sediment yield: {sediment_yield} t/km²/yr. Represents mean annual value over period {period_str}."
    else:
        SSL = -9999.0
        SSL_source = "Missing in source data."

    # Calculate SSC
    SSC = calculate_SSC(SSL, Q)

    # Skip if all data are missing
    if Q == -9999.0 and SSL == -9999.0:
        print(f"Warning: No valid Q or SSL data for {station_name}, skipping...")
        return None

    # Calculate time
    reference_date = datetime(1970, 1, 1)
    # Use mid-point of period for climatology
    mid_year = (start_year + end_year) // 2
    mid_date = datetime(mid_year, 7, 1)  # July 1st of mid-year
    days_since_1970 = (mid_date - reference_date).days

    # Apply quality checks
    Q_flag, SSC_flag, SSL_flag = apply_hma_climatology_qc(
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        time=[mid_date],          # 👈 必须给
        station_id=source_id,     # 👈 必须给
        station_name=station_name,
        output_dir=output_dir
    )

    # Create NetCDF file
    nc = Dataset(filepath, 'w', format='NETCDF4')

    # Create dimensions
    time_dim = nc.createDimension('time', None)  # UNLIMITED dimension

    # ===== Coordinate Variables =====

    # Time
    time_var = nc.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'
    time_var[:] = [days_since_1970]

    # Latitude
    lat_var = nc.createVariable('lat', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = latitude if not np.isnan(latitude) else -9999.0

    # Longitude
    lon_var = nc.createVariable('lon', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = longitude if not np.isnan(longitude) else -9999.0

    # ===== Station Metadata Variables =====

    # Altitude (not in CSV, set to missing)
    alt_var = nc.createVariable('altitude', 'f4', fill_value=-9999.0)
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station elevation above sea level'
    alt_var.units = 'm'
    alt_var.positive = 'up'
    alt_var.comment = 'Source: Not available in original dataset.'
    alt_var[:] = -9999.0

    # Upstream area
    area_var = nc.createVariable('upstream_area', 'f4', fill_value=-9999.0)
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var.comment = 'Source: Original data provided by Li et al. (2021).'
    area_var[:] = basin_area if not np.isnan(basin_area) else -9999.0

    # ===== Data Variables =====

    # Q (Discharge)
    Q_var = nc.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
    Q_var.standard_name = 'water_volume_transport_in_river_channel'
    Q_var.long_name = 'river discharge'
    Q_var.units = 'm3 s-1'
    Q_var.coordinates = 'time lat lon'
    Q_var.ancillary_variables = 'Q_flag'
    if Q != -9999.0:
        Q_comment = f"Source: Calculated. Formula: Q (m³/s) = Q (km³/yr) × 10⁹ / 31,557,600. Original Q: {Q_km3_yr} km³/yr. Represents mean annual value over period {period_str}."
    else:
        Q_comment = "Source: Missing in source data."
    Q_var.comment = Q_comment
    Q_var[:] = [Q]

    # Q_flag
    Q_flag_var = nc.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
    Q_flag_var.long_name = 'quality flag for river discharge'
    Q_flag_var.standard_name = 'status_flag'
    Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    Q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    Q_flag_var[:] = [Q_flag]

    # SSC
    SSC_var = nc.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    SSC_var.long_name = 'suspended sediment concentration'
    SSC_var.units = 'mg L-1'
    SSC_var.coordinates = 'time lat lon'
    SSC_var.ancillary_variables = 'SSC_flag'
    if SSC != -9999.0:
        SSC_comment = f"Source: Calculated. Formula: SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 86.4), where 86.4 = 86400 s/day × 1000 L/m³ × 10⁻⁶ ton/mg. Represents mean annual value over period {period_str}."
    else:
        SSC_comment = "Source: Could not be calculated due to missing Q or SSL data."
    SSC_var.comment = SSC_comment
    SSC_var[:] = [SSC]

    # SSC_flag
    SSC_flag_var = nc.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
    SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
    SSC_flag_var.standard_name = 'status_flag'
    SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    SSC_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSC_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    SSC_flag_var[:] = [SSC_flag]

    # SSL
    SSL_var = nc.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    SSL_var.long_name = 'suspended sediment load'
    SSL_var.units = 'ton day-1'
    SSL_var.coordinates = 'time lat lon'
    SSL_var.ancillary_variables = 'SSL_flag'
    SSL_var.comment = SSL_source
    SSL_var[:] = [SSL]

    # SSL_flag
    SSL_flag_var = nc.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
    SSL_flag_var.long_name = 'quality flag for suspended sediment load'
    SSL_flag_var.standard_name = 'status_flag'
    SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSL_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    SSL_flag_var[:] = [SSL_flag]

    # Sediment yield
    if not np.isnan(sediment_yield):
        sy_var = nc.createVariable('sediment_yield', 'f4', ('time',), fill_value=-9999.0)
        sy_var.long_name = 'sediment yield per unit drainage area'
        sy_var.units = 't km-2 yr-1'
        sy_var.coordinates = 'time lat lon'
        sy_var.comment = 'Source: Original data provided by Li et al. (2021).'
        sy_var[:] = [sediment_yield]

    # ===== Global Attributes =====

    nc.Conventions = 'CF-1.8, ACDD-1.3'
    nc.title = 'Harmonized Global River Discharge and Sediment'

    # Summary
    basin_name = str(station_data.get('Basin', '')).strip() if not pd.isna(station_data.get('Basin')) else 'High Mountain Asia'
    river_name = str(station_data.get('Headwaters', '')).strip() if not pd.isna(station_data.get('Headwaters')) else 'Unknown'
    nc.summary = f"River discharge and suspended sediment data for {station_name.split('(')[0].strip()} station on the {river_name} in {basin_name}. This dataset contains mean annual values including discharge, suspended sediment concentration, and sediment load. Data represents climatological average over the period of record."

    nc.source = 'In-situ station data'
    nc.data_source_name = 'HMA Dataset (Li et al. 2021)'
    nc.station_name = station_name
    nc.river_name = river_name
    nc.Source_ID = source_id

    # Geographic coverage
    if not np.isnan(latitude) and not np.isnan(longitude):
        nc.geospatial_lat_min = float(latitude)
        nc.geospatial_lat_max = float(latitude)
        nc.geospatial_lon_min = float(longitude)
        nc.geospatial_lon_max = float(longitude)
    nc.geospatial_vertical_min = -9999.0
    nc.geospatial_vertical_max = -9999.0
    nc.geographic_coverage = f"{basin_name}, High Mountain Asia"

    # Time coverage
    nc.time_coverage_start = f"{start_year}-01-01"
    nc.time_coverage_end = f"{end_year}-12-31"
    nc.temporal_span = period_str
    nc.temporal_resolution = 'climatology'

    # Variables provided
    vars_provided = []
    if not np.isnan(basin_area):
        vars_provided.append('upstream_area')
    if Q != -9999.0:
        vars_provided.append('Q')
    if SSC != -9999.0:
        vars_provided.append('SSC')
    if SSL != -9999.0:
        vars_provided.append('SSL')
    if not np.isnan(sediment_yield):
        vars_provided.append('sediment_yield')
    nc.variables_provided = ', '.join(vars_provided)
    nc.number_of_data = '1'

    # Reference
    nc.reference = 'Li, D., Lu, X., Overeem, I., Walling, D. E., Syvitski, J., Kettner, A. J., ... & Zhang, T. (2021). Exceptional increases in fluvial sediment fluxes in a warmer and wetter High Mountain Asia. Science, 374(6567), 599-603. https://doi.org/10.1126/science.abi9649'
    nc.source_data_link = 'https://doi.org/10.1126/science.abi9649'

    # Creator
    nc.creator_name = 'Zhongwang Wei'
    nc.creator_email = 'weizhw6@mail.sysu.edu.cn'
    nc.creator_institution = 'Sun Yat-sen University, China'

    # History
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nc.history = f"{now}: Converted from {os.path.basename(data_source_csv)} to CF-1.8 compliant NetCDF format. Applied quality control checks and standardized units. Script: convert_to_netcdf_cf18_qc.py"
    nc.date_created = datetime.now().strftime('%Y-%m-%d')
    nc.date_modified = datetime.now().strftime('%Y-%m-%d')
    nc.processing_level = 'Quality controlled and standardized'

    # Comments
    notes_str = str(station_data.get('Notes', '')).strip() if not pd.isna(station_data.get('Notes')) else ''
    nc.comment = f"Data represents mean annual values calculated from observations over the period {period_str}. Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. {notes_str}"

    nc.close()

    # ============================================================
    # CF-1.8 / ACDD-1.3 compliance check
    # ============================================================
    # errors, warnings = check_nc_completeness(filepath)

    # if errors:
    #     print("❌ CF/ACDD compliance FAILED:")
    #     for e in errors:
    #         print("   -", e)
    #     raise RuntimeError("NetCDF compliance check failed")

    # if warnings:
    #     print("⚠️ CF/ACDD compliance warnings:")
    #     for w in warnings:
    #         print("   -", w)


    # Return summary
    return {
        'filename': filename,
        'station_name': station_name,
        'source_id': source_id,
        'river_name': river_name,
        'basin_name': basin_name,
        'longitude': longitude if not np.isnan(longitude) else None,
        'latitude': latitude if not np.isnan(latitude) else None,
        'altitude': None,
        'upstream_area': basin_area if not np.isnan(basin_area) else None,
        'Q': Q if Q != -9999.0 else None,
        'Q_flag': Q_flag,
        'SSC': SSC if SSC != -9999.0 else None,
        'SSC_flag': SSC_flag,
        'SSL': SSL if SSL != -9999.0 else None,
        'SSL_flag': SSL_flag,
        'sediment_yield': sediment_yield if not np.isnan(sediment_yield) else None,
        'start_year': start_year,
        'end_year': end_year,
        'period_str': period_str
    }


def generate_station_summary_csv(summaries, output_dir):
    """
    Generate a CSV file summarizing all stations

    Args:
        summaries: List of station summary dicts
        output_dir: Output directory path
    """
    csv_path = os.path.join(output_dir, 'HMA_station_summary.csv')

    rows = []
    for s in summaries:
        if s is None:
            continue

        # Calculate percent complete (for climatology, it's 100% if data exists)
        Q_percent = 100.0 if s['Q'] is not None else 0.0
        SSC_percent = 100.0 if s['SSC'] is not None else 0.0
        SSL_percent = 100.0 if s['SSL'] is not None else 0.0

        row = {
            'station_name': s['station_name'],
            'Source_ID': s['source_id'],
            'river_name': s['river_name'],
            'longitude': s['longitude'] if s['longitude'] is not None else 'N/A',
            'latitude': s['latitude'] if s['latitude'] is not None else 'N/A',
            'altitude': s['altitude'] if s['altitude'] is not None else 'N/A',
            'upstream_area': s['upstream_area'] if s['upstream_area'] is not None else 'N/A',
            'Data Source Name': 'HMA Dataset (Li et al. 2021)',
            'Type': 'In-situ',
            'Temporal Resolution': 'climatology',
            'Temporal Span': s['period_str'],
            'Variables Provided': ', '.join([v for v in ['Q', 'SSC', 'SSL'] if s[v] is not None]),
            'Geographic Coverage': f"{s['basin_name']}, High Mountain Asia",
            'Reference/DOI': 'https://doi.org/10.1126/science.abi9649',
            'Q_start_date': s['start_year'],
            'Q_end_date': s['end_year'],
            'Q_percent_complete': Q_percent,
            'SSC_start_date': s['start_year'] if s['SSC'] is not None else 'N/A',
            'SSC_end_date': s['end_year'] if s['SSC'] is not None else 'N/A',
            'SSC_percent_complete': SSC_percent,
            'SSL_start_date': s['start_year'] if s['SSL'] is not None else 'N/A',
            'SSL_end_date': s['end_year'] if s['SSL'] is not None else 'N/A',
            'SSL_percent_complete': SSL_percent
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nGenerated station summary CSV: {csv_path}")
    return csv_path


# Main processing
if __name__ == '__main__':
    # Paths
    # Paths (relative-style: derive from project root)
    def _find_project_root(start_dir, marker="sediment_wzx_1111"):
        d = os.path.abspath(start_dir)
        while True:
            if os.path.basename(d) == marker:
                return d
            parent = os.path.dirname(d)
            if parent == d:
                return None
            d = parent

    PROJECT_ROOT = _find_project_root(CURRENT_DIR) or os.path.abspath(os.path.join(CURRENT_DIR, ".."))

    source_csv = os.path.join(PROJECT_ROOT, "Source", "HMA", "HMA_catchments.csv")
    output_dir = os.path.join(PROJECT_ROOT, "Output_r", "annually_climatology", "HMA", "qc")


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    print(f"Reading CSV: {source_csv}")
    df = pd.read_csv(source_csv, encoding='utf-8-sig')

    # Filter to only process stations with Source_ID (S1-S28)
    df_filtered = df[df['Stations'].str.contains(r'\(S\d+\)', regex=True, na=False)]

    print(f"\nProcessing {len(df_filtered)} stations with Source_ID...")
    print("=" * 80)

    successful = 0
    failed = 0
    summaries = []

    for idx, row in df_filtered.iterrows():
        station_name = row['Stations']
        print(f"\nProcessing: {station_name} +")

        try:
            summary = create_netcdf_for_station(row, output_dir, source_csv)
            if summary:
                print(f"  ✓ Created: {summary['filename']}")
                if summary['Q'] is not None:
                    print(f"    Q: {summary['Q']:.2f} m³/s (flag={summary['Q_flag']})")
                if summary['SSC'] is not None:
                    print(f"    SSC: {summary['SSC']:.2f} mg/L (flag={summary['SSC_flag']})")
                if summary['SSL'] is not None:
                    print(f"    SSL: {summary['SSL']:.2f} ton/day (flag={summary['SSL_flag']})")
                summaries.append(summary)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"Summary: {successful} successful, {failed} failed")
    print("=" * 80)

    # Generate summary CSV
    if summaries:
        generate_station_summary_csv(summaries, output_dir)

    print(f"\nAll files saved to: {output_dir}")
    print("Processing complete!")
