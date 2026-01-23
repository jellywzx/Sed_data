#!/usr/bin/env python3
"""
Quality Control and CF-1.8 Standardization for Yellow River Sediment Data.

This script:
1. Reads existing NetCDF files from Output/annually_climatology/Huanghe
2. Performs quality control checks and adds quality flags
3. Standardizes metadata to CF-1.8 compliance (following ALi_De_Boer reference)
4. Generates station summary CSV
5. Saves standardized files to Output_r/annually_climatology/Huanghe

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
    station_info : dict
        Dictionary containing station metadata for CSV summary
    """

    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read input file
    with nc.Dataset(input_file, 'r') as ds_in:
        # Read all data
        lon = float(ds_in.variables['longitude'][:])
        lat = float(ds_in.variables['latitude'][:])
        alt = float(ds_in.variables['altitude'][:])
        upstream_area = float(ds_in.variables['upstream_area'][:])
        time_vals = ds_in.variables['time'][:]
        ssc_vals = ds_in.variables['ssc'][:]

        # Read metadata
        station_id = ds_in.station_id
        station_name = ds_in.station_name
        station_name_chinese = ds_in.station_name_chinese
        river_name = ds_in.river_name
        river_name_chinese = ds_in.river_name_chinese
        original_time_range = ds_in.original_time_range if hasattr(ds_in, 'original_time_range') else "2015-2019"
        representative_year = ds_in.representative_year if hasattr(ds_in, 'representative_year') else "2017"

        # Get time units for conversion
        time_units = ds_in.variables['time'].units
        time_calendar = ds_in.variables['time'].calendar

    # Perform QC checks
    # ======================================================
    # L0: basic physical QC (missing / negative)
    # ======================================================
    SSC_flag = np.array(
        [apply_quality_flag(v, "SSC") for v in ssc_vals],
        dtype=np.int8
    )
    # ======================================================
    # L1: log-IQR screening for SSC
    # ======================================================
    valid_ssc = ssc_vals[(SSC_flag == 0) & np.isfinite(ssc_vals) & (ssc_vals > 0)]

    if len(valid_ssc) < 5:
        print(
            f"  ℹ️  Station {station_id}: "
            f"valid SSC samples = {len(valid_ssc)} < 5, "
            "log-IQR QC skipped."
        )
    else:
        lower, upper = compute_log_iqr_bounds(valid_ssc)

        if lower is not None:
            for i, v in enumerate(ssc_vals):
                if (
                    SSC_flag[i] == 0
                    and np.isfinite(v)
                    and v > 0
                    and (v < lower or v > upper)
                ):
                    SSC_flag[i] = np.int8(2)  # suspect

    # ======================================================
    # L2: SSC–Q consistency check (only if Q exists)
    # ======================================================
    # if "Q" in ds_in.variables:
    #     Q_vals = ds_in.variables["Q"][:]
    #     Q_flag = np.array(
    #         [apply_quality_flag(v, "Q") for v in Q_vals],
    #         dtype=np.int8
    #     )

    #     ssc_q_bounds = build_ssc_q_envelope(Q_vals, ssc_vals)

    #     if ssc_q_bounds is not None:
    #         for i in range(len(ssc_vals)):
    #             inconsistent, resid = check_ssc_q_consistency(
    #                 Q_vals[i],
    #                 ssc_vals[i],
    #                 Q_flag[i],
    #                 SSC_flag[i],
    #                 ssc_q_bounds
    #             )
    #             if inconsistent:
    #                 SSC_flag[i] = np.int8(2)  # suspect

    # ======================================================
    # SSC–Q diagnostic plot
    # ======================================================
    # if ssc_q_bounds is not None and "Q" in ds_in.variables:
    #     out_png = os.path.join(
    #         output_dir,
    #         f"{station_id}_ssc_q_diagnostic.png"
    #     )

    #     plot_ssc_q_diagnostic(
    #         time=dates,
    #         Q=Q_vals,
    #         SSC=ssc_vals,
    #         Q_flag=Q_flag,
    #         SSC_flag=SSC_flag,
    #         ssc_q_bounds=ssc_q_bounds,
    #         station_id=station_id,
    #         station_name=station_name,
    #         out_png=out_png,
    #     )



    # Calculate statistics for CSV
    n_total = len(SSC_flag)
    n_good = np.sum(SSC_flag == 0)
    ssc_percent_complete = 100.0 * n_good / n_total if n_total > 0 else 0.0

    # ======================================================
    # QC final statistics (per station)
    # ======================================================
    qc_counts = {
        "total": int(n_total),
        "good": int(np.sum(SSC_flag == 0)),
        "suspect": int(np.sum(SSC_flag == 2)),
        "bad": int(np.sum(SSC_flag == 3)),
        "missing": int(np.sum(SSC_flag == 9)),
    }

    print(
        f"  QC summary:\n"
        f"    total samples : {qc_counts['total']}\n"
        f"    good (flag=0) : {qc_counts['good']}\n"
        f"    suspect (2)   : {qc_counts['suspect']}\n"
        f"    bad (3)       : {qc_counts['bad']}\n"
        f"    missing (9)   : {qc_counts['missing']}"
    )

    # Determine temporal span
    dates = nc.num2date(time_vals, units=time_units, calendar=time_calendar)
    if len(dates) > 0:
        ssc_start_date = dates[0].year
        ssc_end_date = dates[-1].year
    else:
        ssc_start_date = representative_year
        ssc_end_date = representative_year

    # Create output filename
    output_file = os.path.join(output_dir, f'Huanghe_{station_id}.nc')

    print(f"  Station: {station_name} ({station_name_chinese})")
    print(f"  River: {river_name} ({river_name_chinese})")
    print(f"  Location: {lat:.4f}°N, {lon:.4f}°E")
    valid = SSC_flag == 0
    if np.any(valid):
        ssc_repr = np.mean(ssc_vals[valid])
        flag_repr = 0
    else:
        ssc_repr = ssc_vals[0]
        flag_repr = SSC_flag[0]

    print(f"  SSC (representative): {ssc_repr:.2f} mg/L (flag={flag_repr})")

    print(f"  Output: {os.path.basename(output_file)}")

    # Create standardized NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create dimensions
        time_dim = ds.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var.axis = "T"
        time_var[:] = time_vals

        # Latitude
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = lat

        # Longitude
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = lon

        # Altitude
        alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.long_name = "station elevation above sea level"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.positive = "up"
        alt_var.comment = "Source: Not available in original dataset."
        if np.isnan(alt):
            alt_var[:] = -9999.0
        else:
            alt_var[:] = alt

        # Upstream area
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.comment = "Source: Original data provided by Yellow River Sediment Bulletin (2015-2019)."
        area_var[:] = upstream_area

        # SSC data variable
        ssc_var = ds.createVariable('SSC', 'f4', ('time',),
                                     fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.long_name = "suspended sediment concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "time lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag"
        ssc_var.comment = "Source: Original data provided by Yellow River Sediment Bulletin (2015-2019). " \
                          "Unit conversion: Original unit kg/m³ × 1000 = mg/L. " \
                          "Represents climatological annual average from year 2017 (middle year of 2015-2019 period)."
        ssc_var[:] = ssc_vals

        # SSC quality flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',),
                                          fill_value=9, zlib=True, complevel=4)
        ssc_flag_var.long_name = "quality flag for suspended sediment concentration"
        ssc_flag_var.standard_name = "status_flag"
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = SSC_flag

        # Global attributes - following CF-1.8 and ACDD-1.3
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.summary = f"Suspended sediment concentration data for {station_name} station on the {river_name} " \
                     f"in the Yellow River Basin, China. This dataset contains climatological annual average " \
                     f"value representing the 2015-2019 period. Data represents the middle year (2017) of " \
                     f"the observation period."

        # Source and data information
        ds.source = "In-situ station data"
        ds.data_source_name = "Yellow River Sediment Bulletin Dataset"
        ds.station_name = station_name
        ds.river_name = river_name
        ds.Source_ID = station_id

        # Type and resolution
        ds.Type = "In-situ"
        ds.Temporal_Resolution = "climatology"
        ds.Temporal_Span = f"{original_time_range}"
        ds.Geographic_Coverage = "Yellow River Basin, China"

        # Variables provided
        ds.Variables_Provided = "SSC"
        ds.Number_of_data = "1"

        # References and links
        ds.Reference = "Zhang Yaonan, Kang jianfang, Liu chun. (2021). Data on Sediment Observation in the " \
                       "Yellow River Basin from 2015 to 2019. National Cryosphere Desert Data Center. " \
                       "https://doi.org/10.12072/ncdc.YRiver.db0054.2021"
        ds.source_data_link = "https://doi.org/10.12072/ncdc.YRiver.db0054.2021"

        # Creator information
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"

        # Temporal coverage
        ds.time_coverage_start = f"{representative_year}-01-01"
        ds.time_coverage_end = f"{representative_year}-12-31"
        ds.temporal_span = original_time_range
        ds.temporal_resolution = "climatology"

        # Spatial coverage
        ds.geospatial_lat_min = float(lat)
        ds.geospatial_lat_max = float(lat)
        ds.geospatial_lon_min = float(lon)
        ds.geospatial_lon_max = float(lon)
        if not np.isnan(alt) and alt != -9999.0:
            ds.geospatial_vertical_min = float(alt)
            ds.geospatial_vertical_max = float(alt)

        ds.geographic_coverage = "Yellow River Basin, China"

        # Processing history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.history = f"{current_time}: Quality controlled and standardized to CF-1.8 compliance. " \
                     f"Added quality flags based on physical law checks. Script: qc_and_standardize.py"

        ds.date_created = datetime.now().strftime("%Y-%m-%d")
        ds.date_modified = datetime.now().strftime("%Y-%m-%d")
        ds.processing_level = "Quality controlled and standardized"

        # Additional comments
        ds.comment = f"Data represents climatological annual average sediment concentration from year {representative_year} " \
                     f"(middle year of {original_time_range} dataset). Quality flags indicate data reliability: " \
                     f"0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. " \
                     f"Note: Discharge and sediment load data are NOT available in the original dataset."

        ds.data_limitations = "Only annual average SSC available; no discharge (Q) or sediment load (SSL) data in original dataset. " \
                              "Climatological value represents single year (2017) from 2015-2019 period."

        # Store Chinese names as additional attributes
        ds.station_name_chinese = station_name_chinese
        ds.river_name_chinese = river_name_chinese

    # Prepare station info for CSV
    station_info = {
        'station_name': station_name,
        'Source_ID': station_id,
        'river_name': river_name,
        'longitude': lon,
        'latitude': lat,
        'altitude': alt if not np.isnan(alt) else 'N/A',
        'upstream_area': upstream_area,
        'Data Source Name': 'Yellow River Sediment Bulletin Dataset',
        'Type': 'In-situ',
        'Temporal Resolution': 'climatology',
        'Temporal Span': original_time_range,
        'Variables Provided': 'SSC',
        'Geographic Coverage': 'Yellow River Basin, China',
        'Reference/DOI': 'https://doi.org/10.12072/ncdc.YRiver.db0054.2021',
        'Q_start_date': 'N/A',
        'Q_end_date': 'N/A',
        'Q_percent_complete': 'N/A',
        'SSC_start_date': ssc_start_date,
        'SSC_end_date': ssc_end_date,
        'SSC_percent_complete': ssc_percent_complete,
        'SSL_start_date': 'N/A',
        'SSL_end_date': 'N/A',
        'SSL_percent_complete': 'N/A'
    }
    station_info.update({
    'SSC_n_total': qc_counts['total'],
    'SSC_n_good': qc_counts['good'],
    'SSC_n_suspect': qc_counts['suspect'],
    'SSC_n_bad': qc_counts['bad'],
    'SSC_n_missing': qc_counts['missing'],
    })

    # ======================================================
    # CF / ACDD completeness check
    # ======================================================
    # errors, warnings = check_nc_completeness(output_file, strict=True)

    # if errors:
    #     raise RuntimeError(
    #         f"CF/ACDD completeness check failed for {output_file}:\n"
    #         + "\n".join(errors)
    #     )

    # if warnings:
    #     print(f"Warnings for {output_file}:")
    #     for w in warnings:
    #         print("  -", w)

    return station_info


def main():
    """Main processing function."""

    print("="*80)
    print("Yellow River Sediment Data - QC and CF-1.8 Standardization")
    print("="*80)
    print()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    input_dir = os.path.join(project_root, "Source", "HuangHe", "netcdf")
    output_dir = os.path.join(project_root, "Output_r", "annually_climatology", "Huanghe")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all NetCDF files
    input_files = sorted(glob.glob(os.path.join(input_dir, 'HuangHe_*.nc')))

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    # Process each file
    station_info_list = []
    processed_count = 0
    error_count = 0

    for input_file in input_files:
        try:
            station_info = standardize_netcdf_file(input_file, output_dir)
            station_info_list.append(station_info)
            processed_count += 1
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print()
    print("="*80)
    print("Generating Station Summary CSV")
    print("="*80)
    print()

    # Create DataFrame and save to CSV
    df = pd.DataFrame(station_info_list)

    # Reorder columns to match reference format
    column_order = [
        'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
        'altitude', 'upstream_area', 'Data Source Name', 'Type',
        'Temporal Resolution', 'Temporal Span', 'Variables Provided',
        'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete',
        'SSC_n_total',
        'SSC_n_good',
        'SSC_n_suspect',
        'SSC_n_bad',
        'SSC_n_missing',

    ]

    df = df[column_order]

    csv_file = os.path.join(output_dir, 'Huanghe_station_summary.csv')
    df.to_csv(csv_file, index=False)

    print(f"Station summary saved to: {csv_file}")
    print(f"Total stations: {len(df)}")
    print("\nGlobal QC summary (SSC):")
    print(df[['SSC_n_good', 'SSC_n_suspect', 'SSC_n_bad', 'SSC_n_missing']].sum())

    print()

    # Print summary
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_dir}")
    print()
    print("="*80)
    print("Quality Control Summary")
    print("="*80)
    print("NOTE: Discharge (Q) and sediment load (SSL) data are NOT available")
    print("      in the original Yellow River dataset (2015-2019).")
    print("="*80)
    print()
    


if __name__ == '__main__':
    main()
