#!/usr/bin/env python3
"""
Update sediment NetCDF files to match the required specifications.
This script:
1. Merges sediment and discharge data
2. Standardizes variable names and attributes
3. Converts lat/lon from dimensions to scalar variables
4. Unifies time dimensions
5. Calculates missing variables if possible
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import sys
import os
from netCDF4 import num2date, date2num
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.runtime import resolve_output_root




def update_sediment_file(sediment_file, discharge_file, output_file=None):
    """
    Update a sediment NetCDF file to match specifications.

    Parameters:
    -----------
    sediment_file : str or Path
        Path to the input sediment NetCDF file
    discharge_file : str or Path
        Path to the corresponding discharge NetCDF file
    output_file : str or Path, optional
        Path to the output file. If None, overwrites the input file.
    """

    if output_file is None:
        output_file = sediment_file

    print(f"Processing: {Path(sediment_file).name}")

    # Read input files
    with nc.Dataset(sediment_file, 'r') as ds_sed, nc.Dataset(discharge_file, 'r') as ds_dis:

        # Extract scalar coordinates
        latitude = float(ds_sed['lat'][0])
        longitude = float(ds_sed['lon'][0])

        # Extract drainage area (upstream_area)
        upstream_area = float(ds_dis['drainage_area'][:]) if 'drainage_area' in ds_dis.variables else np.nan

        # Extract altitude if available (set to NaN if not found)
        altitude = np.nan  # Not available in current data

        # Get sediment data
        time_sed_load = ds_sed['time_sed_load'][:] if 'time_sed_load' in ds_sed.variables else None
        if 'sediment_load' in ds_sed.variables:
            sed_var = ds_sed['sediment_load']
            if len(sed_var.shape) == 3:
                sediment_load_raw = sed_var[:, 0, 0]
            else:
                sediment_load_raw = sed_var[:]
        else:
            sediment_load_raw = None

        time_sed_suscon = ds_sed['time_sed_suscon'][:] if 'time_sed_suscon' in ds_sed.variables else None
        if 'suspended_sediment_concentration' in ds_sed.variables:
            ssc_var = ds_sed['suspended_sediment_concentration']
            if len(ssc_var.shape) == 3:
                ssc_raw = ssc_var[:, 0, 0]
            else:
                ssc_raw = ssc_var[:]
        else:
            ssc_raw = None

        # Get discharge data
        time_flow = ds_dis['time_flow'][:]
        dis_var = ds_dis['discharge']
        if len(dis_var.shape) == 3:
            discharge_raw = dis_var[:, 0, 0]
        else:
            discharge_raw = dis_var[:]

        # Determine unified time axis (use the union of all time points)
        all_times = []
        if time_sed_load is not None:
            all_times.append(time_sed_load)
        if time_sed_suscon is not None:
            all_times.append(time_sed_suscon)
        all_times.append(time_flow)

        # Use the time dimension with most coverage or merge all unique times
        time_combined = np.unique(np.concatenate(all_times))
        time_combined.sort()

        # Initialize arrays with fill values
        fill_value = -9999.0
        n_time = len(time_combined)

        sediment_load = np.full(n_time, fill_value, dtype=np.float32)
        ssc = np.full(n_time, fill_value, dtype=np.float32)
        discharge = np.full(n_time, fill_value, dtype=np.float32)
        SSC_derived = np.zeros(n_time, dtype=bool)
        SSL_derived = np.zeros(n_time, dtype=bool)

        # Map sediment_load to unified time
        if time_sed_load is not None and sediment_load_raw is not None:
            for i, t in enumerate(time_sed_load):
                idx = np.where(time_combined == t)[0]
                if len(idx) > 0:
                    val = sediment_load_raw[i]
                    if val != -999.0:  # Original fill value
                        # Convert from "tonnes" to "ton day-1" (assuming original is already per day)
                        sediment_load[idx[0]] = val

        # Map ssc to unified time
        if time_sed_suscon is not None and ssc_raw is not None:
            for i, t in enumerate(time_sed_suscon):
                idx = np.where(time_combined == t)[0]
                if len(idx) > 0:
                    val = ssc_raw[i]
                    if val != -999.0:  # Original fill value
                        ssc[idx[0]] = val

        # Map discharge to unified time
        for i, t in enumerate(time_flow):
            idx = np.where(time_combined == t)[0]
            if len(idx) > 0:
                val = discharge_raw[i]
                if val != -999.0:  # Original fill value
                    discharge[idx[0]] = val

        # Calculate missing values if possible
        # If sediment_load and discharge exist but ssc is missing, calculate ssc.
        # Formula: sediment_load (ton day-1) = discharge (m3 s-1) * ssc (mg L-1) * 0.0864.
        # Therefore: ssc = sediment_load / (discharge * 0.0864).
        for i in range(n_time):
            if (ssc[i] == fill_value and
                sediment_load[i] != fill_value and
                discharge[i] != fill_value and
                discharge[i] > 0):
                # sediment_load is in ton day-1 and discharge is in m3 s-1.
                # This expression is equivalent to sediment_load / (discharge * 0.0864).
                ssc[i] = (sediment_load[i] * 1000.0) / (discharge[i] * 86.4)
                SSC_derived[i] = True

            # If ssc and discharge exist but sediment_load is missing, calculate it.
            elif (sediment_load[i] == fill_value and
                  ssc[i] != fill_value and
                  discharge[i] != fill_value):
                # sediment_load (ton day-1) = discharge (m3 s-1) * ssc (mg L-1) * 0.0864.
                sediment_load[i] = discharge[i] * ssc[i] * 86.4 / 1000.0
                SSL_derived[i] = True

        # Get global attributes from original file
        station_id = ds_sed.station_id if hasattr(ds_sed, 'station_id') else ''
        station_name = ds_sed.station_name if hasattr(ds_sed, 'station_name') else ''
        province = ds_sed.province_territory if hasattr(ds_sed, 'province_territory') else ''

    # Create output file
        # BEFORE closing the discharge file, record time units:
    with nc.Dataset(sediment_file, 'r') as ds_sed, nc.Dataset(discharge_file, 'r') as ds_dis:

        ...
        # 把时间单位提取放在 WITH 内
        if 'units' in ds_dis['time_flow'].__dict__:
            original_time_units = ds_dis['time_flow'].units
        else:
            original_time_units = 'days since 1850-01-01 00:00:00'

    # Create output file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:

        ds_out.createDimension('time', n_time)


        # 转换到 datetime
        time_as_datetime = num2date(time_combined, units=original_time_units, calendar='gregorian')

        # 目标统一单位
        target_units = 'days since 1970-01-01 00:00:00'
        time_output = date2num(time_as_datetime, units=target_units, calendar='gregorian')

        # 写入变量
        var_time = ds_out.createVariable('time', 'f8', ('time',))
        var_time.standard_name = 'time'
        var_time.long_name = 'time of measurement'
        var_time.units = target_units
        var_time.calendar = 'gregorian'
        var_time.axis = 'T'
        var_time[:] = time_output

        # Latitude (scalar)
        var_lat = ds_out.createVariable('latitude', 'f4')
        var_lat.standard_name = 'latitude'
        var_lat.long_name = 'station latitude'
        var_lat.units = 'degrees_north'
        var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
        var_lat[:] = latitude

        # Longitude (scalar)
        var_lon = ds_out.createVariable('longitude', 'f4')
        var_lon.standard_name = 'longitude'
        var_lon.long_name = 'station longitude'
        var_lon.units = 'degrees_east'
        var_lon.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
        var_lon[:] = longitude

        # Altitude (scalar)
        var_alt = ds_out.createVariable('altitude', 'f4', fill_value=fill_value)
        var_alt.standard_name = 'altitude'
        var_alt.long_name = 'station altitude above sea level'
        var_alt.units = 'm'
        var_alt[:] = altitude if not np.isnan(altitude) else fill_value

        # Upstream area (scalar)
        var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=fill_value)
        var_area.long_name = 'upstream drainage area'
        var_area.units = 'km2'
        var_area[:] = upstream_area if not np.isnan(upstream_area) else fill_value

        # Discharge
        var_dis = ds_out.createVariable('discharge', 'f4', ('time',),
                                        fill_value=fill_value, chunksizes=[n_time])
        var_dis.standard_name = 'water_volume_transport_in_river_channel'
        var_dis.long_name = 'river discharge'
        var_dis.units = 'm3 s-1'
        var_dis.coordinates = 'time latitude longitude'
        var_dis[:] = discharge

        # SSC
        var_ssc = ds_out.createVariable('ssc', 'f4', ('time',),
                                        fill_value=fill_value, chunksizes=[n_time])
        var_ssc.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        var_ssc.long_name = 'suspended sediment concentration'
        var_ssc.units = 'mg L-1'
        var_ssc.coordinates = 'time latitude longitude'
        var_ssc.ancillary_variables = 'SSC_derived'
        var_ssc[:] = ssc

        var_ssc_derived = ds_out.createVariable('SSC_derived', 'i1', ('time',),
                                                chunksizes=[n_time])
        var_ssc_derived.long_name = 'record-level provenance flag for suspended sediment concentration'
        var_ssc_derived.flag_values = np.array([0, 1], dtype=np.int8)
        var_ssc_derived.flag_meanings = 'source_or_not_derived derived'
        var_ssc_derived.comment = (
            '0 indicates source/non-derived SSC or missing SSC; 1 indicates SSC derived '
            'from sediment_load / (discharge * 0.0864). Missingness is determined from '
            'the SSC value and downstream QC flags.'
        )
        var_ssc_derived[:] = SSC_derived.astype(np.int8)

        # Sediment load
        var_load = ds_out.createVariable('sediment_load', 'f4', ('time',),
                                         fill_value=fill_value, chunksizes=[n_time])
        var_load.long_name = 'suspended sediment load'
        var_load.units = 'ton day-1'
        var_load.coordinates = 'time latitude longitude'
        var_load.ancillary_variables = 'SSL_derived'
        var_load.comment = (
            'Suspended sediment load may be source-reported from HYDAT SED_DLY_LOADS '
            'or derived where source sediment_load is missing using sediment_load = '
            'discharge * ssc * 0.0864. See SSL_derived for record-level provenance.'
        )
        var_load[:] = sediment_load

        var_ssl_derived = ds_out.createVariable('SSL_derived', 'i1', ('time',),
                                                chunksizes=[n_time])
        var_ssl_derived.long_name = 'record-level provenance flag for suspended sediment load'
        var_ssl_derived.flag_values = np.array([0, 1], dtype=np.int8)
        var_ssl_derived.flag_meanings = 'source_or_not_derived derived'
        var_ssl_derived.comment = (
            '0 indicates source/non-derived sediment_load or missing sediment_load; '
            '1 indicates sediment_load derived from discharge * ssc * 0.0864. '
            'Missingness is determined from the sediment_load value and downstream QC flags.'
        )
        var_ssl_derived[:] = SSL_derived.astype(np.int8)

        # Global attributes
        ds_out.Conventions = 'CF-1.8'
        ds_out.title = f'HYDAT Station {station_id} - Sediment and Discharge Data'
        ds_out.institution = 'Water Survey of Canada / Environment and Climate Change Canada'
        ds_out.source = 'HYDAT - Canadian Hydrometric Database'
        ds_out.history = f'Updated to standardized format on {np.datetime64("today")}'
        ds_out.references = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
        ds_out.station_id = station_id
        ds_out.station_name = station_name
        ds_out.province_territory = province
        ds_out.geospatial_lat_min = float(latitude)
        ds_out.geospatial_lat_max = float(latitude)
        ds_out.geospatial_lon_min = float(longitude)
        ds_out.geospatial_lon_max = float(longitude)

    print(f"  ✓ Successfully updated: {Path(output_file).name}")
    return True



def main():
    output_root = resolve_output_root(start=__file__, create=True)
    hydat_dir = output_root / "daily" / "HYDAT"
    sediment_dir = hydat_dir / "sediment"
    discharge_dir = hydat_dir / "discharge_waterlevel"
    output_dir = hydat_dir / "sediment_update"
    output_dir.mkdir(exist_ok=True)

    sediment_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    tasks = []
    for sed_file in sediment_files:
        station_id = sed_file.stem.replace('HYDAT_', '').replace('_SEDIMENT', '')
        dis_file = discharge_dir / f'HYDAT_{station_id}.nc'
        out_file = output_dir / f'HYDAT_{station_id}_SEDIMENT.nc'
        if dis_file.exists():
            tasks.append((sed_file, dis_file, out_file))
        else:
            print(f"Warning: discharge file not found for {station_id}")

    print(f"Found {len(tasks)} matched sediment-discharge file pairs")

    # 并行处理所有站点
    max_workers = min(16, len(tasks))  # 视你的机器/磁盘性能调整，16比较安全
    success_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(update_sediment_file, sed_file, dis_file, out_file): (sed_file, dis_file, out_file)
            for sed_file, dis_file, out_file in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing stations"):
            sed_file, dis_file, out_file = future_to_task[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as e:
                print(f"✗ Error processing {sed_file.name}: {e}")

    print(f"\nUpdated {success_count}/{len(tasks)} files successfully")


if __name__ == '__main__':
    main()
