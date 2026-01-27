#!/usr/bin/env python3
"""
Quality Control and CF-1.8 Standardization for Vanmaercke et al. African Sediment Dataset.

This script:
1. Reads existing NetCDF files from Output/annually_climatology/Vanmaercke
2. Renames variables (sediment_load → SSL, ssc → SSC, discharge → Q, latitude → lat, longitude → lon)
3. Performs quality control checks and adds quality flags
4. Standardizes metadata to CF-1.8 compliance (following ALi_De_Boer reference)
5. Generates station summary CSV
6. Saves standardized files to Output_r/annually_climatology/Vanmaercke

Unit Conversion (already done in input files - VERIFIED):
- SSL (ton/day) = SY (t/km²/y) × upstream_area (km²) / 365.25
  - Where SY = sediment yield from source data
  - This formula correctly converts annual yield to daily load

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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
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
    # check_nc_completeness,
    # add_global_attributes
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
    station_info : dict or None
        Dictionary containing station metadata for CSV summary
    """

    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read input file
    try:
        ds_in = nc.Dataset(input_file, 'r')

        # Read coordinates (scalar variables)
        lat = float(ds_in.variables['latitude'][:])
        lon = float(ds_in.variables['longitude'][:])
        alt = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else np.nan
        time_val = ds_in.variables['time'][0]

        # Read data variables
        ssl_val = float(ds_in.variables['sediment_load'][0]) if 'sediment_load' in ds_in.variables else np.nan
        ssc_val = float(ds_in.variables['ssc'][0]) if 'ssc' in ds_in.variables else np.nan
        q_val = float(ds_in.variables['discharge'][0]) if 'discharge' in ds_in.variables else np.nan

        # Read upstream area
        upstream_area = float(ds_in.variables['upstream_area'][:]) if 'upstream_area' in ds_in.variables else np.nan

        # Read metadata from global attributes
        station_id = ds_in.station_id if hasattr(ds_in, 'station_id') else ""
        station_name = ds_in.station_name if hasattr(ds_in, 'station_name') else ""
        station_location = ds_in.station_location if hasattr(ds_in, 'station_location') else ""
        country = ds_in.station_country if hasattr(ds_in, 'station_country') else ""
        period = ds_in.period if hasattr(ds_in, 'period') else ""

        # Get time units for metadata
        time_units = ds_in.variables['time'].units
        time_calendar = ds_in.variables['time'].calendar

        # Close input file
        ds_in.close()

    except Exception as e:
        print(f"  ERROR reading {os.path.basename(input_file)}: {e}")
        try:
            ds_in.close()
        except:
            pass
        return None

    # Skip if no valid SSL data
    if np.isnan(ssl_val) or ssl_val == -9999.0:
        print(f"  SKIPPED: No valid SSL data")
        return None

# --------------------------------------------------
    # Quality control using tool.py
    # --------------------------------------------------

    # SSL is observed (derived from SY × area) → QC allowed
    ssl_flag = apply_quality_flag(ssl_val, "SSL")

    # Q / SSC are NOT observed in Vanmaercke → force MISSING
    q_flag = FILL_VALUE_INT   # = 9
    ssc_flag = FILL_VALUE_INT # = 9

    # Calculate statistics for CSV
    ssl_percent = 100.0 if ssl_flag == 0 else 0.0
    ssc_percent = 100.0 if ssc_flag == 0 else 0.0
    q_percent = 100.0 if q_flag == 0 else 0.0

    import re
    match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', str(period))
    if match:
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        temporal_span = f"{start_year}-{end_year}"
    else:
        start_year = 2000
        end_year = 2000
        temporal_span = period

    # Create output filename
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    print(f"  Station: {station_name} ({country})")
    print(f"  Location: {lat:.3f}°, {lon:.3f}°")
    print(f"  SSL: {ssl_val:.2f} ton/day (flag={ssl_flag}), Period: {period}")
    print(f"  Q: missing in source (flag={q_flag})")
    print(f"  SSC: missing in source (flag={ssc_flag})")


    # Create standardized NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create dimensions - CF-1.8 compliant
        time_dim = ds.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var.axis = "T"
        time_var[:] = [time_val]

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

        # Altitude
        alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.long_name = "station elevation above sea level"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.positive = "up"
        alt_var.comment = "Source: Not available in Vanmaercke et al. (2014) dataset."
        if not np.isnan(alt):
            alt_var[:] = alt
        else:
            alt_var[:] = -9999.0

        # Upstream drainage area
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.comment = "Source: Original data from Vanmaercke et al. (2014). " \
                           "Used to calculate SSL from sediment yield."
        if not np.isnan(upstream_area):
            area_var[:] = upstream_area
        else:
            area_var[:] = -9999.0

        # Q - River Discharge (NOT AVAILABLE)
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        q_var.long_name = "river discharge"
        q_var.standard_name = "water_volume_transport_in_river_channel"
        q_var.units = "m3 s-1"
        q_var.coordinates = "time lat lon altitude"
        q_var.ancillary_variables = "Q_flag"
        q_var.comment = "Source: Not available in Vanmaercke et al. (2014) dataset. " \
                        "Original study focused on sediment yield, not discharge."
        q_var[:] = [-9999.0]

        # Q quality flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        q_flag_var.long_name = "quality flag for river discharge"
        q_flag_var.standard_name = "status_flag"
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        q_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        q_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                             "3=Bad (e.g., negative), 9=Missing in source."
        q_flag_var[:] = [q_flag]

        # SSC - Suspended Sediment Concentration (NOT AVAILABLE)
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.long_name = "suspended sediment concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "time lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag"
        ssc_var.comment = "Source: Not available in Vanmaercke et al. (2014) dataset. " \
                          "Would require: SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 86.4)."
        ssc_var[:] = [-9999.0]

        # SSC quality flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        ssc_flag_var.long_name = "quality flag for suspended sediment concentration"
        ssc_flag_var.standard_name = "status_flag"
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = [ssc_flag]

        # SSL - Suspended Sediment Load
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = "suspended sediment load"
        ssl_var.units = "ton day-1"
        ssl_var.coordinates = "time lat lon altitude"
        ssl_var.ancillary_variables = "SSL_flag"
        ssl_var.comment = "Source: Calculated from sediment yield (SY) and drainage area. " \
                          "Formula: SSL (ton/day) = SY (t/km²/y) × upstream_area (km²) / 365.25. " \
                          "Represents climatological average over measurement period."
        ssl_var[:] = [ssl_val if not np.isnan(ssl_val) else -9999.0]

        # SSL quality flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        ssl_flag_var.long_name = "quality flag for suspended sediment load"
        ssl_flag_var.standard_name = "status_flag"
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssl_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssl_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssl_flag_var[:] = [ssl_flag]

        # Global attributes - CF-1.8 and ACDD-1.3 compliant
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.summary = f"Sediment load data for {station_name} ({station_location}) in {country}. " \
                     f"Data from the Vanmaercke et al. (2014) comprehensive sediment yield database for Africa, " \
                     f"representing climatological average over measurement period ({period}). " \
                     f"Only sediment load (SSL) available; discharge and SSC not measured in original study."

        # Source and data information
        ds.source = "In-situ station data"
        ds.data_source_name = "Vanmaercke et al. (2014) African Sediment Yield Database"
        ds.station_name = station_name
        ds.river_name = station_name  # River/catchment name
        ds.Source_ID = station_id

        # Type and resolution
        ds.Type = "In-situ"
        ds.Temporal_Resolution = "climatology"
        ds.Temporal_Span = temporal_span
        ds.Geographic_Coverage = f"Africa, {country}"

        # Variables provided
        vars_provided = "SSL"  # Only SSL available
        ds.Variables_Provided = vars_provided
        ds.Number_of_data = "1"

        # References
        ds.Reference = "Vanmaercke, M., Poesen, J., Broeckx, J., & Nyssen, J. (2014). " \
                       "Sediment yield in Africa. Earth-Science Reviews, 136, 350-368. " \
                       "https://doi.org/10.1016/j.earscirev.2014.06.004"
        ds.source_data_link = "https://doi.org/10.1016/j.earscirev.2014.06.004"

        # Creator information
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"

        # Temporal coverage
        ds.time_coverage_start = f"{start_year}-01-01"
        ds.time_coverage_end = f"{end_year}-12-31"
        ds.temporal_span = temporal_span
        ds.temporal_resolution = "climatology"

        # Spatial coverage
        ds.geospatial_lat_min = float(lat)
        ds.geospatial_lat_max = float(lat)
        ds.geospatial_lon_min = float(lon)
        ds.geospatial_lon_max = float(lon)
        ds.geographic_coverage = f"Africa, {country}"

        # Processing history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.history = f"{current_time}: Quality controlled and standardized to CF-1.8 compliance. " \
                     f"Added quality flags, renamed variables (sediment_load→SSL, ssc→SSC, discharge→Q, " \
                     f"latitude→lat, longitude→lon), changed time dimension to UNLIMITED. " \
                     f"Script: qc_and_standardize.py"

        ds.date_created = datetime.now().strftime("%Y-%m-%d")
        ds.date_modified = datetime.now().strftime("%Y-%m-%d")
        ds.processing_level = "Quality controlled and standardized"

        # Additional comments
        ds.comment = f"Data represents climatological average sediment load over measurement period ({period}). " \
                     f"Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. " \
                     f"Note: Only sediment load (SSL) is available from the original dataset. Discharge (Q) and " \
                     f"sediment concentration (SSC) were not measured in the Vanmaercke et al. (2014) study."

        ds.data_limitations = "Only sediment load (SSL) available; discharge (Q) and SSC not measured. " \
                              "Climatological averages only; specific observation periods vary by station. " \
                              "Altitude data not available."

        # Store country and location
        ds.country = country
        ds.station_location = station_location
        ds.measurement_period = period

        # --------------------------------------------------
        # NetCDF completeness check
        # --------------------------------------------------
        # errors, warnings = check_nc_completeness(output_file, strict=False)

        # if errors:
        #     print(f"  ✗ NetCDF failed CF/ACDD completeness check: {station_id}")
        #     for err in errors:
        #         print(f"    ERROR: {err}")
        #     os.remove(output_file)
        #     return None

    # --------------------------------------------------
    # Generate diagnostic plot using plot_ssc_q_diagnostic
    # --------------------------------------------------
    try:
        diagnostic_dir = os.path.join(output_dir, "diagnostic")
        os.makedirs(diagnostic_dir, exist_ok=True)
        
        # For Vanmaercke data without Q and SSC, create placeholder arrays
        time_array = np.array([time_val])
        Q = np.full_like(time_array, np.nan, dtype=float)  # All NaN (missing discharge)
        SSC = np.full_like(time_array, np.nan, dtype=float)  # All NaN (missing SSC)
        Q_flag_array = np.array([q_flag], dtype=np.int8)  # All 9 (missing)
        SSC_flag_array = np.array([ssc_flag], dtype=np.int8)  # All 9 (missing)
        SSL_array = np.array([ssl_val if not np.isnan(ssl_val) else -9999.0], dtype=float)
        
        # No SSC-Q consistency bounds available
        ssc_q_bounds = None
        
        # Generate diagnostic plot
        out_png = os.path.join(diagnostic_dir, f"SSC_Q_Vanmaercke_{station_id}.png")
        plot_ssc_q_diagnostic(
            time=time_array,
            Q=Q,
            SSC=SSC,
            Q_flag=Q_flag_array,
            SSC_flag=SSC_flag_array,
            ssc_q_bounds=ssc_q_bounds,
            station_id=station_id,
            station_name=f"Vanmaercke_{station_id}",
            out_png=out_png
        )
    except Exception as e:
        print(f"  Warning: Failed to create diagnostic plot for {station_id}: {e}")

    # Prepare station info for CSV
    station_info = {
        'station_name': station_name,
        'Source_ID': station_id,
        'river_name': station_name,
        'longitude': lon,
        'latitude': lat,
        'altitude': 'N/A',
        'upstream_area': upstream_area if not np.isnan(upstream_area) else 'N/A',
        'Data Source Name': 'Vanmaercke et al. (2014) African Sediment Yield Database',
        'Type': 'In-situ',
        'Temporal Resolution': 'climatology',
        'Temporal Span': temporal_span,
        'Variables Provided': vars_provided,
        'Geographic Coverage': f"Africa, {country}",
        'Reference/DOI': 'https://doi.org/10.1016/j.earscirev.2014.06.004',
        'Q_start_date': 'N/A',
        'Q_end_date': 'N/A',
        'Q_percent_complete': 'N/A',
        'SSC_start_date': 'N/A',
        'SSC_end_date': 'N/A',
        'SSC_percent_complete': 'N/A',
        'SSL_start_date': start_year,
        'SSL_end_date': end_year,
        'SSL_percent_complete': ssl_percent
    }

    return station_info


def main():
    """Main processing function."""

    print("="*80)
    print("Vanmaercke et al. African Sediment Dataset - QC and CF-1.8 Standardization")
    print("="*80)
    print()

    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    input_dir = os.path.join(PROJECT_ROOT, "Output_r", "annually_climatology", "Vanmaercke", "nc")
    output_dir = os.path.join(PROJECT_ROOT, "Output_r", "annually_climatology", "Vanmaercke", "qc")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all NetCDF files
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Vanmaercke_*.nc')))

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
            station_info = standardize_netcdf_file(input_file, output_dir)
            if station_info:
                station_info_list.append(station_info)
                processed_count += 1
            else:
                skipped_count += 1
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

        csv_file = os.path.join(output_dir, 'Vanmaercke_station_summary.csv')
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
    print("="*80)
    print("Quality Control Summary")
    print("="*80)
    print("Quality Checks Applied:")
    print("  SSL (Suspended Sediment Load):")
    print("    - SSL < 0: Flagged as BAD (flag=3)")
    print("    - Valid SSL: Flagged as GOOD (flag=0)")
    print("  Q (Discharge): All marked as MISSING (flag=9) - not available in source")
    print("  SSC (Concentration): All marked as MISSING (flag=9) - not available in source")
    print()
    print("NOTE: The Vanmaercke et al. (2014) dataset contains only sediment yield data,")
    print("      which has been converted to sediment load (SSL). Discharge and SSC are")
    print("      not available in the original study.")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
