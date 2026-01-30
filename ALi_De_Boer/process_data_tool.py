#!/usr/bin/env python3
"""
Convert Ali & De Boer (2007) station data to CF-1.8 compliant NetCDF format

This script processes sediment and discharge data from the upper Indus River basin
and creates individual NetCDF files for each station following CF-1.8 and ACDD-1.3 conventions.

Data source: Ali, K. F., & De Boer, D. H. (2007). Spatial patterns and variation
of suspended sediment yield in the upper Indus River basin, northern Pakistan.
Journal of Hydrology, 334(3-4), 368-387. https://doi.org/10.1016/j.jhydrol.2006.10.013

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Created: 2024-10-21
Modified: 2024-10-24
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset, stringtochar
from datetime import datetime
import os
import re
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    parse_dms_to_decimal,
    parse_period,
    calculate_discharge,
    calculate_ssl_from_mt_yr,
    calculate_ssc,
    compute_log_iqr_bounds,
    generate_station_summary_csv,
    build_ssc_q_envelope,
    apply_quality_flag_array,        
    apply_hydro_qc_with_provenance, 
    summarize_warning_types as summarize_warning_types_tool,
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
    generate_warning_summary_csv as generate_warning_summary_csv_tool,
)


FILL_VALUE_FLOAT = np.float32(-9999.0)
FILL_VALUE_INT = np.int8(9)


def create_station_netcdf(row, idx, output_dir, input_file,ssl_iqr_bounds, ssc_q_bounds):
    """Create a CF-1.8 compliant NetCDF file for one station."""

    station_name = row['Station']
    river_name = row['River']
    source_id = f"ALI{idx+1:03d}"

    # Use Source_ID for filename
    filename = f"ALi_De_Boer_{source_id}.nc"
    filepath = os.path.join(output_dir, filename)

    print(f"Processing station {idx+1}: {station_name} ({source_id})")

    # Parse coordinates
    latitude = parse_dms_to_decimal(row['Latitude'])
    longitude = parse_dms_to_decimal(row['Longitude'])

    # Parse period of record
    start_year, end_year = parse_period(row['Period of record'])

    # Get other data
    drainage_area = row['Drainage area (km2)']
    elevation = row['Elevation (masl)']
    runoff = row['Runoff (mm)']
    sediment_mt_yr = row['Sediment（(Mt yr−1)）']

    # ==========================================================
    # Log-IQR outlier screening for SSL (source data)
    # ==========================================================
    ssl_log_iqr_outlier = False

    if ssl_iqr_bounds is not None and ssl_iqr_bounds[0] is not None:
        lower, upper = ssl_iqr_bounds
        if (
            not pd.isna(sediment_mt_yr)
            and sediment_mt_yr > 0
            and (sediment_mt_yr < lower or sediment_mt_yr > upper)
        ):
            ssl_log_iqr_outlier = True
    


    sediment_yield = row['Sediment（t km−2 yr−1）']

    # Calculate discharge, SSL, SSC
    Q = calculate_discharge(runoff, drainage_area)
    SSL = calculate_ssl_from_mt_yr(sediment_mt_yr)
    SSC = calculate_ssc(SSL, Q)

    # Apply quality flags
    Q_flag_qc1   = int(apply_quality_flag_array([Q],   "Q")[0])
    SSC_flag_qc1 = int(apply_quality_flag_array([SSC], "SSC")[0])
    SSL_flag_qc1 = int(apply_quality_flag_array([SSL], "SSL")[0])

    # 2) 构造 time（你后面本来就要算 days_since_1970，这里先算出来用于 hydro qc）
    if start_year and end_year:
        mid_year = (start_year + end_year) // 2
        mid_date = datetime(mid_year, 7, 1)
        base_date = datetime(1970, 1, 1)
        days_since_1970 = (mid_date - base_date).days
    else:
        days_since_1970 = np.nan

    # 3) 调用 apply_hydro_qc_with_provenance（QC2/QC3 + provenance）
    qc = apply_hydro_qc_with_provenance(
        time=np.array([days_since_1970], dtype=float),
        Q=np.array([Q], dtype=float),
        SSC=np.array([SSC], dtype=float),
        SSL=np.array([SSL], dtype=float),
        Q_is_independent=True,#calculate_discharge
        SSC_is_independent=False,#calculate_ssc
        SSL_is_independent=True,#calculate_ssl，only change unit
        ssl_is_derived_from_q_ssc=False,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )

    # 如果 time 无效，tool 可能返回 None
    if qc is None:
        # 退回到 QC1 结果
        Q_flag = np.int8(Q_flag_qc1)
        SSC_flag = np.int8(SSC_flag_qc1)
        SSL_flag = np.int8(SSL_flag_qc1)
    else:
        # 用 tool 的最终 flag（同样取 [0]）
        Q_flag = np.int8(qc["Q_flag"][0])
        SSC_flag = np.int8(qc["SSC_flag"][0])
        SSL_flag = np.int8(qc["SSL_flag"][0])

    # 4) 保留你原来的 log-IQR outlier 降级逻辑（仍然有效）
    if ssl_log_iqr_outlier and SSL_flag == 0:
        SSL_flag = np.int8(2)  # suspect

    print(
        f"[QC] {source_id} QC1-array: Q={Q_flag_qc1}, SSC={SSC_flag_qc1}, SSL={SSL_flag_qc1} | "
        f"Final: Q={int(Q_flag)}, SSC={int(SSC_flag)}, SSL={int(SSL_flag)}"
    )
    
    # --------------------------
    # Build warnings for summary tools
    # --------------------------
    warnings = []

    # 1) QC fallback
    if qc is None:
        warnings.append("QC: invalid time -> fallback to QC1 flags")

    # 2) SSL log-IQR outlier screening
    if ssl_log_iqr_outlier:
        warnings.append("SSL: log-IQR outlier (source Mt/yr)")

    # 3) Flag-based warnings
    def _flag_warn(var, flag):
        # 0 good, 1 estimated, 2 suspect, 3 bad, 9 missing
        if flag == 1:
            return f"{var}: estimated (flag=1)"
        if flag == 2:
            return f"{var}: suspect (flag=2)"
        if flag == 3:
            return f"{var}: bad (flag=3)"
        if flag == 9:
            return f"{var}: missing (flag=9)"
        return None

    for var, flag in [("Q", int(Q_flag)), ("SSC", int(SSC_flag)), ("SSL", int(SSL_flag))]:
        w = _flag_warn(var, flag)
        if w:
            warnings.append(w)

    # 给工具函数用的两个字段
    n_warnings = len(warnings)
    warnings_str = "; ".join(warnings[:20])  # 也可以不截断

    # Downgrade SSL flag if statistically anomalous (log-IQR)
    if ssl_log_iqr_outlier and SSL_flag == 0:
        SSL_flag = np.int8(2)  # suspect

    # Calculate time (middle of period)
    if start_year and end_year:
        mid_year = (start_year + end_year) // 2
        mid_date = datetime(mid_year, 7, 1)
        base_date = datetime(1970, 1, 1)
        days_since_1970 = (mid_date - base_date).days
    else:
        days_since_1970 = np.nan

    # Create NetCDF file
    nc = Dataset(filepath, 'w', format='NETCDF4')

    # ========== DIMENSIONS ==========
    time_dim = nc.createDimension('time', None)  # UNLIMITED dimension

    # ========== COORDINATE VARIABLES ==========

    # Time coordinate
    time_var = nc.createVariable('time', 'f8', ('time',),fill_value=FILL_VALUE_FLOAT)
    time_var.standard_name = 'time'
    time_var.long_name = 'representative time of climatological mean'
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'
    time_var.comment = (
        'This time value represents the midpoint of the period of record and '
        'is used as a representative timestamp for climatological (multi-year mean) data. '
        'The values in this file are not instantaneous observations but averages over the '
        'entire period of record.'
    )

    if not pd.isna(days_since_1970):
        time_var[0] = days_since_1970

    # Latitude (scalar)
    lat_var = nc.createVariable('lat', 'f4',fill_value=FILL_VALUE_FLOAT)
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    if not pd.isna(latitude):
        lat_var[:] = latitude

    # Longitude (scalar)
    lon_var = nc.createVariable('lon', 'f4',fill_value=FILL_VALUE_FLOAT)
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    if not pd.isna(longitude):
        lon_var[:] = longitude

    # ========== DATA VARIABLES ==========

    # Altitude (scalar data variable, not coordinate)
    alt_var = nc.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station elevation above sea level'
    alt_var.units = 'm'
    alt_var.positive = 'up'
    alt_var.comment = 'Source: Original data provided by Ali & De Boer (2007).'
    if not pd.isna(elevation):
        alt_var[:] = elevation

    # Upstream drainage area (scalar data variable)
    area_var = nc.createVariable('upstream_area', 'f4', fill_value=-9999.0)
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var.comment = 'Source: Original data provided by Ali & De Boer (2007).'
    area_var[:] = drainage_area if not pd.isna(drainage_area) else -9999.0

    # Discharge (Q)
    Q_var = nc.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
    Q_var.standard_name = 'water_volume_transport_in_river_channel'
    Q_var.long_name = 'river discharge'
    Q_var.units = 'm3 s-1'
    Q_var.coordinates = 'time lat lon'
    Q_var.ancillary_variables = 'Q_flag'
    Q_var.comment = ('Source: Calculated. Formula: Q (m³/s) = runoff (mm/yr) × '
                     'drainage_area (km²) / 31557.6, where the divisor accounts for '
                     'conversion from mm·km²/yr to m³/s. Original runoff: '
                     f'{runoff} mm/yr. Represents mean annual value over period of record.')
    if not pd.isna(Q):
        Q_var[0] = Q
    else:
        Q_var[0] = -9999.0

    # Q quality flag
    Q_flag_var = nc.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
    Q_flag_var.long_name = 'quality flag for river discharge'
    Q_flag_var.standard_name = 'status_flag'
    Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    Q_flag_var.comment = ('Flag definitions: 0=Good, 1=Estimated, '
                          '2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), '
                          '9=Missing in source.')
    Q_flag_var[0] = Q_flag

    # SSC (Suspended Sediment Concentration)
    SSC_var = nc.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    SSC_var.long_name = 'suspended sediment concentration'
    SSC_var.units = 'mg L-1'
    SSC_var.coordinates = 'time lat lon'
    SSC_var.ancillary_variables = 'SSC_flag'
    SSC_var.comment = ('Source: Calculated. Formula: SSC (mg/L) = SSL (ton/day) / '
                       '(Q (m³/s) × 86.4), where 86.4 = 86400 s/day × 1000 L/m³ × 10⁻⁶ ton/mg. '
                       'Represents mean annual value over period of record.')
    if not pd.isna(SSC):
        SSC_var[0] = SSC
    else:
        SSC_var[0] = -9999.0

    # SSC quality flag
    SSC_flag_var = nc.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
    SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
    SSC_flag_var.standard_name = 'status_flag'
    SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    SSC_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSC_flag_var.comment = ('Flag definitions: 0=Good, 1=Estimated, '
                            '2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), '
                            '9=Missing in source.')
    SSC_flag_var[0] = SSC_flag

    # SSL (Suspended Sediment Load)
    SSL_var = nc.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    SSL_var.long_name = 'suspended sediment load'
    SSL_var.units = 'ton day-1'
    SSL_var.coordinates = 'time lat lon'
    SSL_var.ancillary_variables = 'SSL_flag'
    SSL_var.comment = ('Source: Calculated. Formula: SSL (ton/day) = '
                       'sediment_load (Mt/yr) × 10⁶ / 365, where 1 Mt = 10⁶ ton. '
                       f'Original sediment load: {sediment_mt_yr} Mt/yr. '
                       'Represents mean annual value over period of record.')
    if not pd.isna(SSL):
        SSL_var[0] = SSL
    else:
        SSL_var[0] = -9999.0

    # SSL quality flag
    SSL_flag_var = nc.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
    SSL_flag_var.long_name = 'quality flag for suspended sediment load'
    SSL_flag_var.standard_name = 'status_flag'
    SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
    SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSL_flag_var.comment = ('Flag definitions: 0=Good, 1=Estimated, '
                            '2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), '
                            '9=Missing in source.')
    SSL_flag_var[0] = SSL_flag

    # Sediment yield
    yield_var = nc.createVariable('sediment_yield', 'f4', ('time',), fill_value=-9999.0)
    yield_var.long_name = 'sediment yield per unit drainage area'
    yield_var.units = 't km-2 yr-1'
    yield_var.coordinates = 'time lat lon'
    yield_var.comment = 'Source: Original data provided by Ali & De Boer (2007).'
    if not pd.isna(sediment_yield):
        yield_var[0] = sediment_yield
    else:
        yield_var[0] = -9999.0

    # ========== GLOBAL ATTRIBUTES (CF-1.8 and ACDD-1.3) ==========

    # Conventions
    nc.Conventions = 'CF-1.8, ACDD-1.3'

    # Dataset identification
    nc.title = 'Harmonized Global River Discharge and Sediment'
    nc.summary = (f'River discharge and suspended sediment data for {station_name} station '
                  f'on the {river_name} River in the upper Indus River basin, northern Pakistan. '
                  f'This dataset contains mean annual values including discharge, suspended '
                  f'sediment concentration, sediment load, and sediment yield. Data represents '
                  f'climatological average over the period of record.')

    # Data source information
    nc.source = 'In-situ station data'
    nc.data_source_name = 'ALi_De_Boer Dataset'

    # Station metadata (as global attributes)
    nc.station_name = station_name
    nc.river_name = river_name
    nc.Source_ID = source_id

    # Geographic coverage
    if not pd.isna(latitude) and not pd.isna(longitude):
        nc.geospatial_lat_min = float(latitude)
        nc.geospatial_lat_max = float(latitude)
        nc.geospatial_lon_min = float(longitude)
        nc.geospatial_lon_max = float(longitude)
    if not pd.isna(elevation):
        nc.geospatial_vertical_min = float(elevation)
        nc.geospatial_vertical_max = float(elevation)
    nc.geographic_coverage = 'Upper Indus River Basin, Northern Pakistan and Western Himalayas'

    # Temporal coverage
    if start_year and end_year:
        nc.time_coverage_start = f"{start_year}-01-01"
        nc.time_coverage_end = f"{end_year}-12-31"
        nc.temporal_span = f"{start_year}-{end_year}"
    nc.temporal_resolution = 'climatological'
    nc.time_coverage_resolution = 'climatological'
    nc.climatology = (
        'This dataset contains climatological (multi-year mean) values derived '
        'from observations over the stated period of record.'
        )

    

    # Variables provided
    nc.variables_provided = 'altitude, upstream_area, Q, SSC, SSL, sediment_yield'
    nc.number_of_data = '1'

    # References
    nc.reference = ('Ali, K. F., & De Boer, D. H. (2007). Spatial patterns and variation '
                    'of suspended sediment yield in the upper Indus River basin, northern '
                    'Pakistan. Journal of Hydrology, 334(3-4), 368-387. '
                    'https://doi.org/10.1016/j.jhydrol.2006.10.013')
    nc.source_data_link = 'https://doi.org/10.1016/j.jhydrol.2006.10.013'

    # Creator information
    nc.creator_name = 'Zhongwang Wei'
    nc.creator_email = 'weizhw6@mail.sysu.edu.cn'
    nc.creator_institution = 'Sun Yat-sen University, China'

    # History (provenance tracking)
    history_entry = (f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: '
                     f'Converted from {os.path.basename(input_file)} to CF-1.8 compliant '
                     f'NetCDF format. Applied quality control checks and standardized units. '
                     f'Script: convert_to_netcdf.py')
    nc.history = history_entry

    # Date information
    nc.date_created = datetime.now().strftime("%Y-%m-%d")
    nc.date_modified = datetime.now().strftime("%Y-%m-%d")

    # Processing information
    nc.processing_level = 'Quality controlled and standardized'
    nc.comment = ('Data represents mean annual values calculated from observations over '
                  'the period of record. Discharge calculated from runoff and drainage area. '
                  'SSC calculated from sediment load and discharge. Quality flags indicate '
                  'data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.')
    nc.comment += (
    ' A statistical outlier pre-screening was applied only to suspended sediment load (SSL) '
    'using a log-transformed interquartile range (IQR) method based on source-reported '
    'annual sediment load (Mt yr−1). Values identified as statistical outliers were not '
    'removed but flagged as suspect (flag=2).')
    nc.comment += (' A statistical outlier pre-screening was applied only to suspended sediment load (SSL) ')
    nc.comment += (
    ' In addition, a hydrological consistency check based on the SSC–Q relationship '
    'was applied at the dataset level. SSC values that deviated substantially from the '
    'log–log SSC–Q envelope were flagged as suspect (flag=2) but not removed.'
    )


    # Close file
    nc.close()

    return {
        'station_name': station_name,
        'source_id': source_id,
        'river_name': river_name,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': elevation,
        'upstream_area': drainage_area,
        'start_year': start_year,
        'end_year': end_year,
        'Q': Q,
        'Q_flag': Q_flag,
        'SSC': SSC,
        'SSC_flag': SSC_flag,
        'SSL': SSL,
        'SSL_flag': SSL_flag,
        'filepath': filepath,
        "warnings": warnings_str,
        "n_warnings": n_warnings
    }


def generate_station_summary_csv(station_data, output_dir):
    """Generate a CSV summary file of station metadata and data completeness."""

    csv_file = os.path.join(output_dir, 'ALi_De_Boer_station_summary.csv')

    summary_data = []
    for data in station_data:
        # For climatology data, completeness is either 100% (good data) or 0% (not good)
        Q_complete = 100.0 if data['Q_flag'] == 0 else 0.0
        SSC_complete = 100.0 if data['SSC_flag'] == 0 else 0.0
        SSL_complete = 100.0 if data['SSL_flag'] == 0 else 0.0

        # Date formatting
        Q_start = str(data['start_year']) if data['start_year'] else "N/A"
        Q_end = str(data['end_year']) if data['end_year'] else "N/A"

        # Temporal span
        temporal_span = f"{Q_start}-{Q_end}" if Q_start != "N/A" and Q_end != "N/A" else "N/A"

        # Variables provided (based on data availability)
        vars_provided = []
        if Q_complete > 0:
            vars_provided.append('Q')
        if SSC_complete > 0:
            vars_provided.append('SSC')
        if SSL_complete > 0:
            vars_provided.append('SSL')
        vars_str = ', '.join(vars_provided) if vars_provided else "N/A"

        summary_data.append({
            'station_name': data['station_name'],
            'Source_ID': data['source_id'],
            'river_name': data['river_name'],
            'longitude': f"{data['longitude']:.6f}" if not pd.isna(data['longitude']) else "N/A",
            'latitude': f"{data['latitude']:.6f}" if not pd.isna(data['latitude']) else "N/A",
            'altitude': f"{data['altitude']:.1f}" if not pd.isna(data['altitude']) else "N/A",
            'upstream_area': f"{data['upstream_area']:.1f}" if not pd.isna(data['upstream_area']) else "N/A",
            'Data Source Name': 'ALi_De_Boer Dataset',
            'Type': 'In-situ',
            'Temporal Resolution': 'climatology',
            'Temporal Span': temporal_span,
            'Variables Provided': vars_str,
            'Geographic Coverage': 'Upper Indus River Basin, Northern Pakistan',
            'Reference/DOI': 'https://doi.org/10.1016/j.jhydrol.2006.10.013',
            'Q_start_date': Q_start,
            'Q_end_date': Q_end,
            'Q_percent_complete': f"{Q_complete:.1f}",
            'SSC_start_date': Q_start,
            'SSC_end_date': Q_end,
            'SSC_percent_complete': f"{SSC_complete:.1f}",
            'SSL_start_date': Q_start,
            'SSL_end_date': Q_end,
            'SSL_percent_complete': f"{SSL_complete:.1f}",
        })

    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_file, index=False, encoding='utf-8')

    print(f"\nCreated station summary CSV: {csv_file}")

    return csv_file


def main():
    """Main conversion function."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    source_dir = os.path.join(project_root, "Source", "ALi_De_Boer")
    output_dir = os.path.join(project_root, "Output_r", "annually_climatology", "ALi_De_Boer", "qc")

    input_file = os.path.join(source_dir, "ALi_De_Boer.xlsx")

    os.makedirs(output_dir, exist_ok=True)

    print("Input file:", input_file)
    print("Output directory:", output_dir)

    print("=" * 80)
    print("ALi_De_Boer Dataset Conversion to CF-1.8 NetCDF")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read Excel file
    print("Reading Excel file...")
    df = pd.read_excel(input_file, sheet_name='Sheet2', header=0)
    print(f"Found {len(df)} stations")
    print()
    # ==========================================================
    # Log-IQR bounds for SSL (Mt/yr, source data)
    # ==========================================================
    ssl_iqr_bounds = compute_log_iqr_bounds(
        df['Sediment（(Mt yr−1)）']
    )
    print("Log-IQR bounds for SSL (source sediment load):")
    print(f"  SSL bounds (Mt/yr): {ssl_iqr_bounds}")
    print()

    # 从源字段推导标准物理量（仍然在主脚本）
    valid = (
        (df['Runoff (mm)'] > 0) &
        (df['Drainage area (km2)'] > 0) &
        (df['Sediment（(Mt yr−1)）'] > 0)
    )

    Q_ref = (
        df.loc[valid, 'Runoff (mm)']
        * df.loc[valid, 'Drainage area (km2)']
        / 31557.6
    )

    SSL_ref = df.loc[valid, 'Sediment（(Mt yr−1)）'] * 1e6 / 365.25
    SSC_ref = SSL_ref / (Q_ref * 0.0864)
    
    ssc_q_bounds = build_ssc_q_envelope(
        Q_m3s=Q_ref.values,
        SSC_mgL=SSC_ref.values
    )

    print("SSC–Q consistency envelope:")
    print(ssc_q_bounds)

    # Process each station
    print("Creating CF-1.8 compliant NetCDF files...")
    station_data = []
    for idx, row in df.iterrows():
        data = create_station_netcdf(row, idx, output_dir, input_file,ssl_iqr_bounds,ssc_q_bounds)
        station_data.append(data)
        print(f"  Created: {os.path.basename(data['filepath'])}")
        print(f"    Q={data['Q']:.2f} m³/s (flag={data['Q_flag']}), "
              f"SSC={data['SSC']:.2f} mg/L (flag={data['SSC_flag']}), "
              f"SSL={data['SSL']:.2f} ton/day (flag={data['SSL_flag']})")

    print()

    # Generate station summary CSV
    print("Generating station summary CSV...")
    csv_file = generate_station_summary_csv(station_data, output_dir)

    # Summary statistics
    good_Q = sum(1 for d in station_data if d['Q_flag'] == 0)
    good_SSC = sum(1 for d in station_data if d['SSC_flag'] == 0)
    good_SSL = sum(1 for d in station_data if d['SSL_flag'] == 0)

    # ------------------------------------------------------------
    # Extra standardized QC outputs (same as HYDAT example tools)
    # ------------------------------------------------------------
    stations_info = station_data  

    # 1) print warning types summary (returns whatever tool defines)
    warning_summary = summarize_warning_types_tool(stations_info)
    print("\n[WARN] summary by type:")
    print(warning_summary)

    # 2) CSVs
    csv_summary_path = os.path.join(output_dir, "csv_summary_tool.csv")
    qc_results_path = os.path.join(output_dir, "qc_results_tool.csv")
    warning_summary_path = os.path.join(output_dir, "warning_summary_tool.csv")

    generate_csv_summary_tool(stations_info, csv_summary_path)
    generate_qc_results_csv_tool(stations_info, qc_results_path)
    generate_warning_summary_csv_tool(stations_info, warning_summary_path)

    print("\n[CSV] Tool summaries written:")
    print("  -", csv_summary_path)
    print("  -", qc_results_path)
    print("  -", warning_summary_path)

    print()
    print("=" * 80)
    print("Conversion complete!")
    print(f"  Created {len(df)} NetCDF files in: {output_dir}")
    print(f"  CSV summary: {csv_file}")
    print(f"  Data quality:")
    print(f"    - Good Q data: {good_Q}/{len(df)}")
    print(f"    - Good SSC data: {good_SSC}/{len(df)}")
    print(f"    - Good SSL data: {good_SSL}/{len(df)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
