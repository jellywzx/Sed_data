#!/usr/bin/env python3
"""
Convert Milliman sediment database and specific river time series to CF-compliant NetCDF files.

This script creates individual NetCDF files for each station, following the CF-1.8 conventions
and matching the GloRiSe format.

Author: Claude Code
Date: 2025-10-19
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = CURRENT_DIR.parent
CODE_DIR = SCRIPT_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from runtime import ensure_directory, resolve_source_root
from validation import require_existing_directory


def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


def days_since_1970(year, month=1, day=1):
    """Convert year/month/day to days since 1970-01-01."""
    reference_date = datetime(1970, 1, 1)
    target_date = datetime(year, month, day)
    delta = target_date - reference_date
    return delta.days


def create_netcdf_file(output_path, location_id, river_name, latitude, longitude,
                       country, continent, times, tss_values, discharge_values,
                       references, additional_info=None):
    """
    Create a CF-compliant NetCDF file for a single station.

    Parameters:
    -----------
    output_path : str
        Full path to output NetCDF file
    location_id : str
        Unique identifier for the station
    river_name : str
        Name of the river
    latitude : float
        Station latitude
    longitude : float
        Station longitude
    country : str
        Country name
    continent : str
        Continent/region name
    times : array-like
        Array of days since 1970-01-01
    tss_values : array-like
        Total suspended sediment concentration (mg/L) or flux (Mt/yr)
    discharge_values : array-like
        River discharge (m3/s) or runoff (km3/yr)
    references : str
        Citation/reference for the data
    additional_info : dict, optional
        Additional metadata to include
    """
    # Create NetCDF file
    dataset = nc.Dataset(output_path, 'w', format='NETCDF4')

    # Create dimensions
    n_times = len(times)
    dataset.createDimension('time', n_times)
    dataset.createDimension('latitude', 1)
    dataset.createDimension('longitude', 1)

    # Create coordinate variables
    lat_var = dataset.createVariable('latitude', 'f4', ('latitude',))
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lat_var.axis = 'Y'
    lat_var[:] = latitude

    lon_var = dataset.createVariable('longitude', 'f4', ('longitude',))
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'
    lon_var.axis = 'X'
    lon_var[:] = longitude

    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.axis = 'T'
    time_var[:] = times

    # Create data variables
    # TSS - can be concentration (mg/L) or flux (Mt/yr)
    tss_var = dataset.createVariable('TSS', 'f4', ('time', 'latitude', 'longitude'),
                                     fill_value=-9999.0)

    # Determine if we have concentration or flux
    if np.nanmax(tss_values) < 10000:  # Likely mg/L if max < 10000
        tss_var.long_name = 'Total Suspended Sediment concentration'
        tss_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        tss_var.units = 'mg L-1'
    else:  # Likely Mt/yr if larger values
        tss_var.long_name = 'Total Suspended Sediment flux'
        tss_var.standard_name = 'sediment_flux'
        tss_var.units = 'Mt yr-1'

    tss_var.coordinates = 'time latitude longitude'

    # Create 3D array and fill with data
    tss_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    for i, val in enumerate(tss_values):
        if not np.isnan(val):
            tss_data[i, 0, 0] = val
    tss_var[:] = tss_data

    # Discharge - can be m3/s or km3/yr
    if discharge_values is not None and len(discharge_values) > 0:
        discharge_var = dataset.createVariable('Discharge', 'f4',
                                              ('time', 'latitude', 'longitude'),
                                              fill_value=-9999.0)

        # Determine units based on magnitude
        if np.nanmax(discharge_values) < 100000:  # Likely m3/s
            discharge_var.standard_name = 'water_volume_transport_in_river_channel'
            discharge_var.long_name = 'River discharge'
            discharge_var.units = 'm3 s-1'
        else:  # Likely km3/yr
            discharge_var.standard_name = 'runoff_flux'
            discharge_var.long_name = 'River runoff'
            discharge_var.units = 'km3 yr-1'

        discharge_var.coordinates = 'time latitude longitude'

        discharge_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
        for i, val in enumerate(discharge_values):
            if not np.isnan(val):
                discharge_data[i, 0, 0] = val
        discharge_var[:] = discharge_data

    # Global attributes
    dataset.title = f"River sediment data for {river_name} (station {location_id})"
    dataset.institution = "Milliman & Farnsworth Global River Sediment Database"
    dataset.source = "Milliman and Farnsworth (2011); individual river studies"
    dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    dataset.references = references
    dataset.Conventions = 'CF-1.8'
    dataset.location_id = location_id
    dataset.river_name = river_name
    dataset.latitude = float(latitude)
    dataset.longitude = float(longitude)
    dataset.country = country
    dataset.continent_region = continent
    dataset.observations = str(n_times)

    # Add additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            setattr(dataset, key, str(value))

    # Close the file
    dataset.close()
    print(f"Created: {output_path}")


def convert_milliman_database(input_csv, output_dir):
    """
    Convert Milliman sediment database to individual NetCDF files per station.

    Parameters:
    -----------
    input_csv : str
        Path to milliman-2012-sediment-database.csv
    output_dir : str
        Directory to save NetCDF files
    """
    print(f"\nProcessing Milliman database: {input_csv}")

    # Read the CSV file
    df = pd.read_csv(input_csv, encoding='utf-8-sig')

    # Clean column names (remove BOM and spaces)
    df.columns = df.columns.str.strip()

    created_count = 0
    skipped_count = 0

    # Process each river station
    for idx, row in df.iterrows():
        # Skip if no TSS data
        if pd.isna(row['TSS']):
            skipped_count += 1
            continue

        # Extract information
        river_id = str(row['ID']).strip()
        river_name = str(row['RiverName']).strip()
        latitude = float(row['LATITUDE'])
        longitude = float(row['LONGITUDE'])
        country = str(row['Country']).strip() if pd.notna(row['Country']) else 'Unknown'
        continent = str(row['Continent_Region']).strip() if pd.notna(row['Continent_Region']) else 'Unknown'

        # TSS data (annual average, so use mid-year as time point)
        tss_value = float(row['TSS'])  # Mt/yr
        discharge_value = float(row['Q']) if pd.notna(row['Q']) else np.nan  # km3/yr

        # Use a representative year (mid-period of data collection, typically ~1980-2000)
        # Since Milliman 2012 database represents long-term averages
        year = 1995  # Representative mid-period
        time_days = [days_since_1970(year, 7, 1)]  # Mid-year

        # Create location ID
        location_id = f"MILLIMAN-{river_id}"

        # Create filename
        safe_name = river_name.replace(' ', '_').replace('/', '-')
        filename = f"Milliman_{safe_name}_{location_id}.nc"
        output_path = os.path.join(output_dir, filename)

        # References
        references = ("Milliman, J.D., and Farnsworth, K.L. (2011). River Discharge to the "
                     "Coastal Ocean: A Global Synthesis. Cambridge University Press, 392 pp.")

        # Additional metadata
        additional_info = {
            'drainage_area_km2': row['Area'] if pd.notna(row['Area']) else 'N/A',
            'river_length_km': row['Length'] if pd.notna(row['Length']) else 'N/A',
            'climate': f"{row['Climate_T']}-{row['Climate_R']}-{row['Climate_S']}" if pd.notna(row['Climate_T']) else 'N/A',
            'geology': row['PrimGeo'] if pd.notna(row['PrimGeo']) else 'N/A',
            'ocean': row['Ocean'] if pd.notna(row['Ocean']) else 'N/A',
            'data_type': 'long-term average',
            'time_period': 'various (pre-2012)',
            'sediment_concentration_mg_L': row['SedConc'] if pd.notna(row['SedConc']) else 'N/A',
        }

        # Create NetCDF file
        try:
            create_netcdf_file(
                output_path=output_path,
                location_id=location_id,
                river_name=river_name,
                latitude=latitude,
                longitude=longitude,
                country=country,
                continent=continent,
                times=time_days,
                tss_values=[tss_value],
                discharge_values=[discharge_value] if not np.isnan(discharge_value) else [],
                references=references,
                additional_info=additional_info
            )
            created_count += 1
        except Exception as e:
            print(f"Error creating file for {river_name}: {e}")
            skipped_count += 1

    print(f"\nMilliman database conversion complete:")
    print(f"  Created: {created_count} files")
    print(f"  Skipped: {skipped_count} stations (no TSS data)")


def convert_timeseries_river(input_csv, output_dir, river_name, location_id,
                             latitude, longitude, country, references,
                             discharge_km3_yr=None):
    """
    Convert a specific river time series to NetCDF.

    Parameters:
    -----------
    input_csv : str
        Path to CSV file with time series data
    output_dir : str
        Directory to save NetCDF file
    river_name : str
        Name of the river
    location_id : str
        Station identifier
    latitude : float
        Station latitude
    longitude : float
        Station longitude
    country : str
        Country name
    references : str
        Citation for the data
    discharge_km3_yr : float, optional
        Annual discharge if constant (km3/yr)
    """
    print(f"\nProcessing {river_name} time series: {input_csv}")

    # Read the CSV file
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()

    # Extract data
    years = df['year'].values

    # Convert years to days since 1970 (use mid-year)
    times = [days_since_1970(int(year), 7, 1) for year in years]

    # TSS flux (Mt/yr)
    tss_values = df['Mt_yr'].values

    # SSC if available
    ssc_values = df['SSC_mgL'].values if 'SSC_mgL' in df.columns else None

    # Discharge (use constant value if provided, otherwise leave empty)
    if discharge_km3_yr is not None:
        discharge_values = [discharge_km3_yr] * len(years)
    else:
        discharge_values = []

    # Create filename
    safe_name = river_name.replace(' ', '_')
    filename = f"TimeSeries_{safe_name}_{location_id}.nc"
    output_path = os.path.join(output_dir, filename)

    # Additional metadata
    additional_info = {
        'data_type': 'annual time series',
        'time_period': f"{int(years[0])}-{int(years[-1])}",
        'number_of_years': str(len(years)),
    }

    # Create NetCDF file with TSS flux
    create_netcdf_file(
        output_path=output_path,
        location_id=location_id,
        river_name=river_name,
        latitude=latitude,
        longitude=longitude,
        country=country,
        continent='Asia' if country == 'China' else 'North America',
        times=times,
        tss_values=tss_values,
        discharge_values=discharge_values,
        references=references,
        additional_info=additional_info
    )

    # If SSC data is available, create a separate file for concentration
    if ssc_values is not None:
        filename_ssc = f"TimeSeries_{safe_name}_{location_id}_SSC.nc"
        output_path_ssc = os.path.join(output_dir, filename_ssc)

        additional_info_ssc = additional_info.copy()
        additional_info_ssc['variable'] = 'suspended_sediment_concentration'

        # For SSC file, use SSC values as "TSS" variable (will be detected as mg/L)
        create_netcdf_file(
            output_path=output_path_ssc,
            location_id=location_id + '_SSC',
            river_name=river_name,
            latitude=latitude,
            longitude=longitude,
            country=country,
            continent='Asia',
            times=times,
            tss_values=ssc_values,
            discharge_values=[],
            references=references,
            additional_info=additional_info_ssc
        )
        print(f"  Also created SSC concentration file: {filename_ssc}")


def main():
    """Main conversion function."""
    print("="*70)
    print("Converting Milliman and River Time Series Data to CF-NetCDF Format")
    print("="*70)

    # Set up paths
    source_root = resolve_source_root(start=__file__)
    base_dir = require_existing_directory(
        source_root / "Milliman" / "evandethier_2022_global_sediment_flux_required_sediment_files",
        description="Milliman raw source directory",
    )
    output_dir = ensure_directory(source_root / "Milliman" / "netcdf_output")

    # Create output directory
    create_output_directory(output_dir)

    # 1. Convert Milliman database
    milliman_csv = os.path.join(base_dir, "milliman-2012-sediment-database.csv")
    if os.path.exists(milliman_csv):
        convert_milliman_database(milliman_csv, output_dir)
    else:
        print(f"Warning: Milliman database not found at {milliman_csv}")

    # 2. Convert Huaihe River
    huaihe_csv = os.path.join(base_dir, "huaihe-1954-2016-sediment-Li-2018.csv")
    if os.path.exists(huaihe_csv):
        convert_timeseries_river(
            input_csv=huaihe_csv,
            output_dir=output_dir,
            river_name="Huaihe River",
            location_id="CHN-HUAIHE-BENGBU",
            latitude=32.95,  # Bengbu station
            longitude=117.38,
            country="China",
            references="Li, D., Lu, X. X., Yang, X., Chen, L., & Lin, L. (2018). "
                      "Sediment load responses to climate variation and cascade reservoirs in "
                      "the Yangtze River: A case study of the Jinsha River. "
                      "Geomorphology, 322, 41-52.",
            discharge_km3_yr=62.0  # Approximate annual discharge
        )

    # 3. Convert Yangtze (Changjiang) River
    yangtze_csv = os.path.join(base_dir, "yangzte-1956-2015-sediment.csv")
    if os.path.exists(yangtze_csv):
        convert_timeseries_river(
            input_csv=yangtze_csv,
            output_dir=output_dir,
            river_name="Yangtze River",
            location_id="CHN-YANGTZE-DATONG",
            latitude=30.77,  # Datong station
            longitude=117.62,
            country="China",
            references="Yang, S. L., Xu, K. H., Milliman, J. D., Yang, H. F., & Wu, C. S. (2015). "
                      "Decline of Yangtze River water and sediment discharge: Impact from natural "
                      "and anthropogenic changes. Scientific Reports, 5, 12581.",
            discharge_km3_yr=900.0  # Approximate annual discharge
        )

    # 4. Convert Zhujiang (Pearl River)
    zhujiang_csv = os.path.join(base_dir, "zhujiang-1954-2016-sediment.csv")
    if os.path.exists(zhujiang_csv):
        convert_timeseries_river(
            input_csv=zhujiang_csv,
            output_dir=output_dir,
            river_name="Pearl River",
            location_id="CHN-PEARL-GAOYAO",
            latitude=23.05,  # Gaoyao station
            longitude=112.45,
            country="China",
            references="Zhang, W., Mu, S., Zhang, Y., & Chen, K. (2016). "
                      "Temporal variation of suspended sediment load in the Pearl River "
                      "due to human activities. International Journal of Sediment Research.",
            discharge_km3_yr=326.0  # Approximate annual discharge
        )

    # 5. Convert Mississippi River
    mississippi_csv = os.path.join(base_dir, "mississippi_tarbert_historical_flux.csv")
    if os.path.exists(mississippi_csv):
        convert_timeseries_river(
            input_csv=mississippi_csv,
            output_dir=output_dir,
            river_name="Mississippi River",
            location_id="USA-MISS-TARBERT",
            latitude=31.01,  # Tarbert Landing
            longitude=-91.62,
            country="United States",
            references="Meade, R. H., & Moody, J. A. (2010). "
                      "Causes for the decline of suspended-sediment discharge in the "
                      "Mississippi River system, 1940-2007. Hydrological Processes, 24(1), 35-49.",
            discharge_km3_yr=580.0  # Approximate annual discharge
        )

    print("\n" + "="*70)
    print("Conversion complete!")
    print(f"All NetCDF files saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
