#!/usr/bin/env python3
"""
Script to convert NERC Hampshire Avon dataset to CF-1.8 compliant NetCDF format.

This script processes daily discharge and water chemistry data from four tributaries
of the Hampshire Avon (Sem, Nadder, West Avon, Ebble), applies quality control,
standardizes metadata to CF-1.8, and generates NetCDF files and a summary CSV.

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import warnings
import sys

warnings.filterwarnings('ignore')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_INT

# Station metadata based on NERC documentation
# Reference: Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage
# and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble)
STATION_METADATA = {
    'AS': {
        'station_name': 'Ashley',
        'river_name': 'Sem',
        'Source_ID': 'NERC_AS',
        'latitude': 51.0,
        'longitude': -1.8,
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'CE': {
        'station_name': 'Cerne Abbas',
        'river_name': 'Nadder',
        'Source_ID': 'NERC_CE',
        'latitude': 50.9,
        'longitude': -1.9,
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GA': {
        'station_name': 'Gale',
        'river_name': 'West Avon',
        'Source_ID': 'NERC_GA',
        'latitude': 51.1,
        'longitude': -1.7,
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GN': {
        'station_name': 'Green Lane',
        'river_name': 'Ebble',
        'Source_ID': 'NERC_GN',
        'latitude': 51.0,
        'longitude': -2.0,
        'altitude': -9999.0,
        'upstream_area': -9999.0
    }
}

def apply_quality_checks(df):
    """
    Apply physical validity checks and set quality flags.

    Flag definitions:
    0 = Good data
    2 = Suspect data (zero/extreme values)
    3 = Bad data (negative/invalid values)
    9 = Missing data
    """
    df['Q_flag'] = 9
    df['SSC_flag'] = 9
    df['SSL_flag'] = 9

    # Q (discharge) flags
    if 'Q' in df.columns:
        df.loc[df['Q'].notna(), 'Q_flag'] = 0
        df.loc[df['Q'] == 0, 'Q_flag'] = 2
        df.loc[df['Q'] < 0, 'Q_flag'] = 3
        df.loc[df['Q'] > 1000, 'Q_flag'] = 2 # Extreme value threshold

    # SSC (sediment concentration) flags
    if 'SSC' in df.columns:
        df.loc[df['SSC'].notna(), 'SSC_flag'] = 0
        df.loc[df['SSC'] < 0, 'SSC_flag'] = 3
        df.loc[df['SSC'] > 3000, 'SSC_flag'] = 2 # Extreme value threshold

    # SSL (sediment load) flags
    if 'SSL' in df.columns:
        df.loc[df['SSL'].notna(), 'SSL_flag'] = 0
        df.loc[df['SSL'] < 0, 'SSL_flag'] = 3

    return df

def calculate_SSL(Q, SSC):
    """
    Calculate Suspended Sediment Load (SSL) in ton/day.
    Formula: SSL (ton/day) = Q (m³/s) * SSC (mg/L) * 0.0864
    """
    if pd.isna(Q) or pd.isna(SSC):
        return np.nan
    return Q * SSC * 0.0864

def process_station(station_code, data_dir, output_dir):
    """
    Process data for a single station and create CF-1.8 compliant NetCDF file.
    """
    metadata = STATION_METADATA[station_code]
    
    discharge_file = os.path.join(data_dir, f"{station_code}_Discharge_data.csv")
    chemistry_file = os.path.join(data_dir, f"{station_code}_surfacewater_chemistry.csv")

    if not os.path.exists(discharge_file) or not os.path.exists(chemistry_file):
        print(f"Data files for station {station_code} not found. Skipping.")
        return None, None, None, None

    df_q = pd.read_csv(discharge_file, parse_dates=['Date'], dayfirst=True)
    df_q.rename(columns={'Date': 'date', df_q.columns[1]: 'Q'}, inplace=True)
    df_q['Q'] = pd.to_numeric(df_q['Q'], errors='coerce')

    df_chem = pd.read_csv(chemistry_file, parse_dates=['Date'], dayfirst=True)
    df_chem.rename(columns={'Date': 'date', 'SSC (mg L-1)': 'SSC'}, inplace=True)
    df_chem['SSC'] = pd.to_numeric(df_chem['SSC'], errors='coerce')

    df = pd.merge(df_q[['date', 'Q']], df_chem[['date', 'SSC']], on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)
    df.dropna(subset=['date'], inplace=True)

    df['SSL'] = df.apply(lambda row: calculate_SSL(row['Q'], row['SSC']), axis=1)
    
    df = apply_quality_checks(df)

    # Time slicing
    valid_data = df[(df['Q_flag'] == 0) | (df['SSC_flag'] == 0)]
    if valid_data.empty:
        print(f"No valid data for station {station_code}. Skipping.")
        return None, None, None, None
        
    start_year = valid_data['date'].min().year
    end_year = valid_data['date'].max().year
    
    start_date_slice = pd.Timestamp(f'{start_year}-01-01')
    end_date_slice = pd.Timestamp(f'{end_year}-12-31')
    
    df = df[(df['date'] >= start_date_slice) & (df['date'] <= end_date_slice)]
    
    if df.empty:
        print(f"No data within the desired time slice for station {station_code}. Skipping.")
        return None, None, None, None

    reference_date = pd.Timestamp('1970-01-01')
    df['time'] = (df['date'] - reference_date).dt.total_seconds() / 86400.0

    start_date_str = df['date'].min().strftime('%Y-%m-%d')
    end_date_str = df['date'].max().strftime('%Y-%m-%d')
    
    # Create NetCDF file
    output_file = os.path.join(output_dir, f"NERC_{station_code}.nc")
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Dimensions
        ncfile.createDimension('time', None)

        # Coordinates
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        time_var.long_name = 'time'
        time_var.standard_name = 'time'
        time_var.units = f'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'

        lat_var = ncfile.createVariable('lat', 'f4')
        lat_var.long_name = 'station latitude'
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'

        lon_var = ncfile.createVariable('lon', 'f4')
        lon_var.long_name = 'station longitude'
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'

        # Data variables
        q_var = ncfile.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
        q_var.long_name = 'River Discharge'
        q_var.standard_name = 'river_discharge'
        q_var.units = 'm3 s-1'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Source: Original data provided by Heppell & Binley (2016).'

        ssc_var = ncfile.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
        ssc_var.long_name = 'Suspended Sediment Concentration'
        ssc_var.standard_name = 'suspended_sediment_concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Original data provided by Heppell & Binley (2016).'

        ssl_var = ncfile.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
        ssl_var.long_name = 'Suspended Sediment Load'
        ssl_var.standard_name = 'suspended_sediment_load'
        ssl_var.units = 'ton day-1'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Source: Calculated. Formula: SSL = Q * SSC * 0.0864'

        alt_var = ncfile.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'

        area_var = ncfile.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'

        # Flag variables
        q_flag_var = ncfile.createVariable('Q_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        q_flag_var.long_name = 'Quality flag for River Discharge'
        q_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        q_flag_var.flag_meanings = 'good_data suspect_data bad_data missing_data'
        
        ssc_flag_var = ncfile.createVariable('SSC_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        ssc_flag_var.long_name = 'Quality flag for Suspended Sediment Concentration'
        ssc_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        ssc_flag_var.flag_meanings = 'good_data suspect_data bad_data missing_data'

        ssl_flag_var = ncfile.createVariable('SSL_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        ssl_flag_var.long_name = 'Quality flag for Suspended Sediment Load'
        ssl_flag_var.flag_values = np.array([0, 3, 9], dtype='b')
        ssl_flag_var.flag_meanings = 'good_data bad_data missing_data'

        # Write data
        time_var[:] = df['time'].values
        lat_var[:] = metadata['latitude']
        lon_var[:] = metadata['longitude']
        q_var[:] = df['Q'].fillna(-9999.0).values
        ssc_var[:] = df['SSC'].fillna(-9999.0).values
        ssl_var[:] = df['SSL'].fillna(-9999.0).values
        alt_var[:] = metadata['altitude']
        area_var[:] = metadata['upstream_area']
        q_flag_var[:] = df['Q_flag'].values
        ssc_flag_var[:] = df['SSC_flag'].values
        ssl_flag_var[:] = df['SSL_flag'].values

        # Global attributes
        ncfile.title = 'Harmonized Global River Discharge and Sediment'
        ncfile.source = 'NERC Dataset'
        ncfile.station_name = metadata['station_name']
        ncfile.river_name = metadata['river_name']
        ncfile.Source_ID = metadata['Source_ID']
        ncfile.featureType = 'timeSeries'
        ncfile.cdm_data_type = 'Station'
        ncfile.summary = 'This dataset contains daily river discharge and suspended sediment data.'
        ncfile.creator_name = 'Zhongwang Wei'
        ncfile.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ncfile.creator_institution = 'Sun Yat-sen University, China'
        ncfile.Conventions = 'CF-1.8, ACDD-1.3'
        ncfile.history = f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}: Data processed from source files."
        ncfile.source_data_link = 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5'
        ncfile.reference = 'Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble). NERC Environmental Information Data Centre.'
        ncfile.temporal_resolution = 'daily'
        ncfile.time_coverage_start = start_date_str
        ncfile.time_coverage_end = end_date_str
        ncfile.geographic_coverage = 'Hampshire Avon Basin, Southern England, UK'

    return df, metadata, start_date_str, end_date_str

def generate_summary_csv(station_summaries, output_dir):
    """
    Generate station summary CSV file.
    """
    if not station_summaries:
        return

    summary_df = pd.DataFrame(station_summaries)
    
    # Ensure all required columns are present
    required_cols = [
        'Source_ID', 'station_name', 'river_name', 'longitude', 'latitude', 'altitude', 'upstream_area',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete',
        'Data Source Name', 'Type', 'Temporal Resolution', 'Temporal Span',
        'Variables Provided', 'Geographic Coverage', 'Reference/DOI'
    ]
    for col in required_cols:
        if col not in summary_df.columns:
            summary_df[col] = 'N/A'
            
    summary_df = summary_df[required_cols]

    output_file = os.path.join(output_dir, "NERC_station_summary.csv")
    summary_df.to_csv(output_file, index=False)

def calculate_percent_complete(df, var_name, flag_name):
    """Calculate percentage of good data for a variable."""
    if var_name not in df.columns or df.empty:
        return 0.0
    good_data = (df[flag_name] == 0).sum()
    return (good_data / len(df)) * 100.0

def main():
    """Main processing function."""
    source_data_dir = '/Users/zhongwangwei/Downloads/Sediment/Source/NERC/data'
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/NERC'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    station_codes = ['AS', 'CE', 'GA', 'GN']
    station_summaries = []

    for station_code in station_codes:
        df, metadata, start_date, end_date = process_station(station_code, source_data_dir, output_dir)

        if df is not None:
            q_dates = df[df['Q_flag'] == 0]['date']
            ssc_dates = df[df['SSC_flag'] == 0]['date']
            ssl_dates = df[df['SSL_flag'] == 0]['date']

            summary = {
                'Source_ID': metadata['Source_ID'],
                'station_name': metadata['station_name'],
                'river_name': metadata['river_name'],
                'longitude': metadata['longitude'],
                'latitude': metadata['latitude'],
                'altitude': metadata['altitude'],
                'upstream_area': metadata['upstream_area'],
                'Q_start_date': q_dates.min().year if not q_dates.empty else 'N/A',
                'Q_end_date': q_dates.max().year if not q_dates.empty else 'N/A',
                'Q_percent_complete': round(calculate_percent_complete(df, 'Q', 'Q_flag'), 2),
                'SSC_start_date': ssc_dates.min().year if not ssc_dates.empty else 'N/A',
                'SSC_end_date': ssc_dates.max().year if not ssc_dates.empty else 'N/A',
                'SSC_percent_complete': round(calculate_percent_complete(df, 'SSC', 'SSC_flag'), 2),
                'SSL_start_date': ssl_dates.min().year if not ssl_dates.empty else 'N/A',
                'SSL_end_date': ssl_dates.max().year if not ssl_dates.empty else 'N/A',
                'SSL_percent_complete': round(calculate_percent_complete(df, 'SSL', 'SSL_flag'), 2),
                'Data Source Name': 'NERC Dataset',
                'Type': 'In-situ station data',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{start_date.split('-')[0]}-{end_date.split('-')[0]}",
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Hampshire Avon Basin, Southern England, UK',
                'Reference/DOI': 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5'
            }
            station_summaries.append(summary)

    generate_summary_csv(station_summaries, output_dir)

if __name__ == '__main__':
    main()
