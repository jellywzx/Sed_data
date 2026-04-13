#!/usr/bin/env python3
"""
Convert Yellow River sediment observation data to NetCDF format.

This script converts annual sediment concentration data from the Yellow River Basin
(2015-2019) from Excel format to CF-compliant NetCDF files, one file per station.

Note: The input Excel file only contains annual average sediment concentration (SSC) data.
      Monthly data and discharge data are not available in the current dataset.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import xlrd
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = CURRENT_DIR.parent
CODE_DIR = SCRIPT_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from runtime import ensure_directory, resolve_source_root
from validation import require_existing_file

def load_station_coordinates(coord_file):
    """Load station coordinates from the national hydrological stations file."""
    print("Loading station coordinates...")

    workbook = xlrd.open_workbook(coord_file, formatting_info=False)
    sheet = workbook.sheet_by_index(0)

    # Create dictionary of station coordinates
    station_coords = {}

    # Yellow River main stream and major tributaries
    yellow_river_systems = [
        '黄河',  # Main Yellow River
        '洮河',  # Tao River
        '窟野河',  # Kuye River
        '无定河',  # Wuding River
        '泾河',  # Jing River
        '渭河',  # Wei River
        '汾河',  # Fen River
        '伊洛河',  # Yiluo River
        '沁河',  # Qin River
        '北洛河'  # Beiluo River
    ]

    # Manual mapping for stations with different names in coordinates file
    station_name_mapping = {
        '狱头': '状头',  # Yutou is written as Zhuangtou in coordinates file
    }

    # Header is in row 0: ['站号', '站名', '河名', '水系', '流域', '东经', '北纬', ...]
    for row_idx in range(1, sheet.nrows):
        station_name = str(sheet.cell_value(row_idx, 1)).strip()  # 站名
        river_name = str(sheet.cell_value(row_idx, 2))  # 河名
        water_system = str(sheet.cell_value(row_idx, 3))  # 水系
        basin_name = str(sheet.cell_value(row_idx, 4)).strip()  # 流域（如：黄河、长江、海河等）

        # Check if station is in Yellow River basin (main stream or tributaries)
        # is_yellow_river = any(system in water_system or system in river_name
                            #  for system in yellow_river_systems)

        is_yellow_river = (
            basin_name == '黄河' and
            any(system in water_system or system in river_name
                for system in yellow_river_systems)
        )

        if is_yellow_river:
            try:
                lon = float(sheet.cell_value(row_idx, 5))  # 东经
                lat = float(sheet.cell_value(row_idx, 6))  # 北纬
                station_id = str(sheet.cell_value(row_idx, 0)).strip()  # 站号
                basin_area_cell = sheet.cell_value(row_idx, 8) if sheet.ncols > 8 else ''

                station_coords[station_name] = {
                    'station_id': station_id,
                    'longitude': lon,
                    'latitude': lat,
                    'river_name': river_name,
                    'water_system': water_system
                }
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse coordinates for station {station_name}: {e}")
                continue

    # Add mappings for alternative station names
    for alt_name, real_name in station_name_mapping.items():
        if real_name in station_coords and alt_name not in station_coords:
            station_coords[alt_name] = station_coords[real_name].copy()

    print(f"Loaded coordinates for {len(station_coords)} Yellow River basin stations")
    return station_coords

def translate_river_name(chinese_name):
    """Translate Chinese river names to English."""
    translations = {
        '黄河': 'Yellow River',
        '洮河': 'Tao River',
        '皇甫川': 'Huangfu River',
        '窟野河': 'Kuye River',
        '无定河': 'Wuding River',
        '廷河': 'Yan River',
        '延河': 'Yan River',
        '泾河': 'Jing River',
        '渭河': 'Wei River',
        '北洛河': 'Beiluo River',
        '汾河': 'Fen River',
        '伊洛河': 'Yiluo River',
        '沁河': 'Qin River',
        '洮河': 'Tao River'
    }
    return translations.get(chinese_name, chinese_name)

def translate_station_name(chinese_name):
    """Translate Chinese station names to Pinyin."""
    translations = {
        '唐乃亥': 'Tangnaihai',
        '兰州': 'Lanzhou',
        '石嘴山': 'Shizuishan',
        '头道拐': 'Toudaoguai',
        '龙门': 'Longmen',
        '潼关': 'Tongguan',
        '三门峡': 'Sanmenxia',
        '小浪底': 'Xiaolangdi',
        '花园口': 'Huayuankou',
        '高村': 'Gaocun',
        '艾山': 'Aishan',
        '利津': 'Lijin',
        '红旗': 'Hongqi',
        '皇甫': 'Huangfu',
        '温家川': 'Wenjiachuan',
        '白家川': 'Baijiachuan',
        '甘谷驿': 'Ganguyi',
        '张家山': 'Zhangjiashan',
        '咸阳': 'Xianyang',
        '狱头': 'Yutou',
        '状头': 'Zhuangtou',  # Alternative name for Yutou
        '华县': 'Huaxian',
        '河津': 'Hejin',
        '黑石关': 'Heishiguan',
        '武陟': 'Wuzhi'
    }
    return translations.get(chinese_name, chinese_name)

def parse_sediment_data(excel_file):
    """Parse sediment data from the Excel file."""
    print("Reading sediment data from Excel...")

    # Read main stream stations
    df_main = pd.read_excel(excel_file, sheet_name='干流控制水文站', header=None)

    # Read tributary stations
    df_trib = pd.read_excel(excel_file, sheet_name='支流重要控制水文站', header=None)

    stations_data = {}

    # Process main stream stations
    # Row 2: station names, Row 3: basin area, Rows 6-10: 2015-2019 data
    station_names = df_main.iloc[2, 2:].values
    basin_areas = df_main.iloc[3, 2:].values

    for idx, station_name in enumerate(station_names):
        if pd.isna(station_name):
            continue

        col_idx = idx + 2

        # Extract annual SSC data for 2015-2019
        ssc_data = {}
        for year_idx, year in enumerate([2015, 2016, 2017, 2018, 2019]):
            row_idx = 6 + year_idx
            value = df_main.iloc[row_idx, col_idx]
            try:
                ssc_data[year] = float(value) if not pd.isna(value) and value != '' else np.nan
            except (ValueError, TypeError):
                ssc_data[year] = np.nan

        stations_data[station_name] = {
            'river_name': '黄河',  # Main stream
            'basin_area': basin_areas[idx] * 10000 if not pd.isna(basin_areas[idx]) else np.nan,  # Convert to km²
            'ssc_annual': ssc_data,
            'source_sheet': 'main_stream'
        }

    # Process tributary stations
    # Row 2: river names, Row 3: station names, Row 4: basin area, Rows 7-11: 2015-2019 data
    river_names = df_trib.iloc[2, 2:].values
    station_names_trib = df_trib.iloc[3, 2:].values
    basin_areas_trib = df_trib.iloc[4, 2:].values

    for idx, station_name in enumerate(station_names_trib):
        if pd.isna(station_name):
            continue

        col_idx = idx + 2

        # Extract annual SSC data for 2015-2019
        ssc_data = {}
        for year_idx, year in enumerate([2015, 2016, 2017, 2018, 2019]):
            row_idx = 7 + year_idx
            value = df_trib.iloc[row_idx, col_idx]
            try:
                ssc_data[year] = float(value) if not pd.isna(value) and value != '' else np.nan
            except (ValueError, TypeError):
                ssc_data[year] = np.nan

        stations_data[station_name] = {
            'river_name': river_names[idx] if not pd.isna(river_names[idx]) else 'Unknown',
            'basin_area': basin_areas_trib[idx] * 10000 if not pd.isna(basin_areas_trib[idx]) else np.nan,  # Convert to km²
            'ssc_annual': ssc_data,
            'source_sheet': 'tributary'
        }

    print(f"Parsed data for {len(stations_data)} stations")
    return stations_data


def create_netcdf_file(station_name, station_data, station_coords, output_dir):
    """Create a NetCDF file for a single station following CF conventions."""

    # Get station information
    station_name_en = translate_station_name(station_name)
    station_id = station_coords.get('station_id', station_name)
    lon = station_coords.get('longitude', np.nan)
    lat = station_coords.get('latitude', np.nan)
    river_name = station_data.get('river_name', 'Unknown')
    river_name_en = translate_river_name(river_name)
    basin_area = station_data.get('basin_area', np.nan)

    # Check if we have valid data
    ssc_annual = station_data['ssc_annual']
    valid_years = [year for year, value in ssc_annual.items() if not np.isnan(value)]

    if len(valid_years) == 0:
        print(f"  Skipping {station_name}: all SSC values are NaN")
        return None

    # Only keep years with valid values
    years = sorted([y for y, v in ssc_annual.items() if not np.isnan(v)])
    ssc_values = [ssc_annual[y] for y in years]

    # Convert each year to a single representative time stamp (e.g., mid-year)
    times = [datetime(year, 7, 1) for year in years]  # 推荐使用中年 7 月 1 日

    filtered_times = times
    filtered_ssc = ssc_values

    # Create output filename
    output_file = os.path.join(output_dir, f'HuangHe_{station_id}.nc')

    print(f"  Creating {output_file} for {station_name_en} ({station_name})...")
    print(f"    Time period: {filtered_times[0].strftime('%Y-%m')} to {filtered_times[-1].strftime('%Y-%m')}")
    print(f"    Number of time steps: {len(filtered_times)}")


    # Create NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create dimensions
        time_dim = ds.createDimension('time', len(filtered_times))

        # Create coordinate variables
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'representative date for annual mean observation'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        # Convert times to days since 1970-01-01
        reference_date = datetime(1970, 1, 1)
        time_values = [(t - reference_date).total_seconds() / 86400.0 for t in filtered_times]
        time_var[:] = time_values

        # Create scalar coordinate variables
        lon_var = ds.createVariable('longitude', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
        lon_var[:] = lon

        lat_var = ds.createVariable('latitude', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
        lat_var[:] = lat

        # Altitude (not available, set to NaN)
        alt_var = ds.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station altitude above sea level'
        alt_var.units = 'm'
        alt_var[:] = np.nan

        # Upstream drainage area
        area_var = ds.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Drainage area from Yellow River Basin data'
        area_var[:] = basin_area

        # Create data variables
        # SSC - convert from kg/m³ to mg/L (1 kg/m³ = 1000 mg/L)
        ssc_var = ds.createVariable('ssc', 'f4', ('time',),
                                     fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time latitude longitude'
        ssc_var.comment = 'Converted from annual average values in kg/m³. Each data point represents an annual mean.'
        ssc_values_converted = np.array(filtered_ssc) * 1000.0  # kg/m³ to mg/L
        ssc_var[:] = ssc_values_converted

        # Note: Discharge data is not available in the source Excel file
        # Sediment load cannot be calculated without discharge data

        # Global attributes
        ds.Conventions = 'CF-1.8'
        ds.title = f'Yellow River Sediment Data for Station {station_name_en}'
        ds.institution = 'National Cryosphere Desert Data Center'
        ds.source = 'Annual sediment concentration observations from Yellow River Basin monitoring network (2015-2019)'
        ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf.py'
        ds.references = 'Zhang Yaonan, Kang jianfang, Liu chun. Data on Sediment Observation in the Yellow River Basin from 2015 to 2019. National Cryosphere Desert Data Center(http://www.ncdc.ac.cn), 2021. https://www.doi.org/10.12072/ncdc.YRiver.db0054.2021'
        ds.comment = 'Original data contains only annual average sediment concentration. Monthly values are replicated from annual averages. Discharge data not available in source dataset.'
        ds.station_id = str(station_id)
        ds.station_name = station_name_en
        ds.station_name_chinese = station_name
        ds.river_name = river_name_en
        ds.river_name_chinese = river_name
        ds.data_source = 'Yellow River Sediment Bulletin (2015-2019)'
        ds.data_limitation = 'Only annual average SSC available; no discharge or monthly data in original dataset'

    print(f"    Created successfully")
    return output_file

def main():
    """Main conversion function."""

    # File paths
    source_dir = resolve_source_root(start=__file__) / 'HuangHe'
    excel_file = require_existing_file(
        source_dir / '黄河流域泥沙观测数据.xlsx',
        description='Huanghe sediment workbook',
    )
    coord_file = require_existing_file(
        source_dir / '全国河流水文站坐标.xls',
        description='Huanghe station coordinate workbook',
    )
    output_dir = ensure_directory(source_dir / 'netcdf')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Yellow River Sediment Data Conversion to NetCDF")
    print("="*70)
    print()

    # Load station coordinates
    station_coords_dict = load_station_coordinates(coord_file)

    # Parse sediment data
    stations_data = parse_sediment_data(excel_file)

    print()
    print("="*70)
    print("Creating NetCDF files...")
    print("="*70)
    print()

    created_files = []
    skipped_stations = []

    for station_name, station_data in stations_data.items():
        # Get coordinates for this station
        coords = station_coords_dict.get(station_name)

        if coords is None:
            print(f"Warning: No coordinates found for station '{station_name}', skipping...")
            skipped_stations.append((station_name, "no coordinates"))
            continue

        try:
            output_file = create_netcdf_file(station_name, station_data, coords, output_dir)
            if output_file:
                created_files.append(output_file)
            else:
                skipped_stations.append((station_name, "no valid data"))
        except Exception as e:
            print(f"  Error creating file for {station_name}: {e}")
            skipped_stations.append((station_name, str(e)))
            import traceback
            traceback.print_exc()

    print()
    print("="*70)
    print("Conversion Summary")
    print("="*70)
    print(f"Total stations processed: {len(stations_data)}")
    print(f"NetCDF files created: {len(created_files)}")
    print(f"Stations skipped: {len(skipped_stations)}")

    if skipped_stations:
        print("\nSkipped stations:")
        for station, reason in skipped_stations:
            print(f"  - {station}: {reason}")

    print()
    print("IMPORTANT NOTES:")
    print("- Original data contains only ANNUAL AVERAGE sediment concentration (SSC)")
    print("- NO DISCHARGE DATA available in the source Excel file")
    print("- Monthly values are replicated from annual averages")
    print("- Sediment load cannot be calculated without discharge data")
    print("- Only 2015-2019 data available")
    print()
    print("Files saved to:", os.path.abspath(output_dir))
    print()

if __name__ == '__main__':
    main()
