#!/usr/bin/env python3
"""
Update sediment NetCDF files to match the required specifications.
This script:
1. Merges sediment and optional discharge data
2. Standardizes variable names and attributes
3. Converts lat/lon from dimensions to scalar variables
4. Unifies time dimensions
5. Calculates missing variables if possible (Q-dependent derivations)

Stations with SSC or SSL data are retained regardless of discharge availability.
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


FILL_VALUE = -9999.0


def update_sediment_file(sediment_file, discharge_file=None, output_file=None):
    """
    Update a sediment NetCDF file to match specifications.

    Parameters:
    -----------
    sediment_file : str or Path
        Path to the input sediment NetCDF file.
    discharge_file : str or Path or None, optional
        Path to the corresponding discharge NetCDF file.  If None, Q will be
        written as all-missing and derivation of SSC/SSL from Q is skipped.
    output_file : str or Path, optional
        Path to the output file. If None, overwrites the input file.
    """

    if output_file is None:
        output_file = sediment_file

    has_discharge = discharge_file is not None and Path(discharge_file).exists()
    print(f"Processing: {Path(sediment_file).name}"
          f"{' (with discharge)' if has_discharge else ' (NO discharge)'}")

    # --- Read sediment data -------------------------------------------------
    with nc.Dataset(sediment_file, 'r') as ds_sed:

        # Extract scalar coordinates
        latitude = float(ds_sed['lat'][0])
        longitude = float(ds_sed['lon'][0])

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

        # Get global attributes from sediment file
        station_id = ds_sed.station_id if hasattr(ds_sed, 'station_id') else ''
        station_name = ds_sed.station_name if hasattr(ds_sed, 'station_name') else ''
        province = ds_sed.province_territory if hasattr(ds_sed, 'province_territory') else ''

    # --- Read discharge data (optional) -------------------------------------
    if has_discharge:
        with nc.Dataset(discharge_file, 'r') as ds_dis:
            upstream_area = float(ds_dis['drainage_area'][:]) if 'drainage_area' in ds_dis.variables else np.nan

            time_flow = ds_dis['time_flow'][:]
            dis_var = ds_dis['discharge']
            if len(dis_var.shape) == 3:
                discharge_raw = dis_var[:, 0, 0]
            else:
                discharge_raw = dis_var[:]

            if 'units' in ds_dis['time_flow'].__dict__:
                original_time_units = ds_dis['time_flow'].units
            else:
                original_time_units = 'days since 1850-01-01 00:00:00'
    else:
        upstream_area = np.nan
        time_flow = None
        discharge_raw = None
        # When no discharge file is available, guess time units from sediment
        # time variables (HYDAT convention).
        original_time_units = 'days since 1850-01-01 00:00:00'

    altitude = np.nan  # Not available in current data

    # --- Build unified time axis (union of all available time arrays) -------
    all_times = []
    if time_sed_load is not None:
        all_times.append(time_sed_load)
    if time_sed_suscon is not None:
        all_times.append(time_sed_suscon)
    if time_flow is not None:
        all_times.append(time_flow)

    time_combined = np.unique(np.concatenate(all_times))
    time_combined.sort()
    n_time = len(time_combined)

    # --- Allocate arrays - all missing by default ---------------------------
    sediment_load = np.full(n_time, FILL_VALUE, dtype=np.float32)
    ssc = np.full(n_time, FILL_VALUE, dtype=np.float32)
    discharge = np.full(n_time, FILL_VALUE, dtype=np.float32)

    # Track which values are source-reported (to protect from derivation).
    ssc_source = np.zeros(n_time, dtype=bool)
    ssl_source = np.zeros(n_time, dtype=bool)

    # --- Map sediment_load (SSL) to unified time ----------------------------
    if time_sed_load is not None and sediment_load_raw is not None:
        for i, t in enumerate(time_sed_load):
            idx = np.where(time_combined == t)[0]
            if len(idx) > 0:
                val = sediment_load_raw[i]
                if val != -999.0:  # Original fill value
                    sediment_load[idx[0]] = val
                    ssl_source[idx[0]] = True

    # --- Map SSC to unified time --------------------------------------------
    if time_sed_suscon is not None and ssc_raw is not None:
        for i, t in enumerate(time_sed_suscon):
            idx = np.where(time_combined == t)[0]
            if len(idx) > 0:
                val = ssc_raw[i]
                if val != -999.0:  # Original fill value
                    ssc[idx[0]] = val
                    ssc_source[idx[0]] = True

    # --- Map discharge to unified time (if available) -----------------------
    if time_flow is not None and discharge_raw is not None:
        for i, t in enumerate(time_flow):
            idx = np.where(time_combined == t)[0]
            if len(idx) > 0:
                val = discharge_raw[i]
                if val != -999.0:
                    discharge[idx[0]] = val

    # --- Derive SSC / SSL where possible (only when Q is present) -----------
    # Source-reported values are NEVER overwritten by derived values.
    if has_discharge:
        for i in range(n_time):
            q_present = (discharge[i] != FILL_VALUE and discharge[i] > 0)

            # Derive SSC: Q + SSL present, SSC not source-reported
            if (not ssc_source[i]
                    and ssc[i] == FILL_VALUE
                    and sediment_load[i] != FILL_VALUE
                    and q_present):
                # sediment_load (ton day-1) = Q (m3 s-1) * SSC (mg L-1) * 0.0864
                ssc[i] = (sediment_load[i] * 1000.0) / (discharge[i] * 86.4)

            # Derive SSL: Q + SSC present, SSL not source-reported
            if (not ssl_source[i]
                    and sediment_load[i] == FILL_VALUE
                    and ssc[i] != FILL_VALUE
                    and q_present):
                sediment_load[i] = discharge[i] * ssc[i] * 86.4 / 1000.0

    # --- Write output NetCDF -------------------------------------------------
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:

        ds_out.createDimension('time', n_time)

        # Convert to datetime then to target units
        time_as_datetime = num2date(time_combined, units=original_time_units, calendar='gregorian')
        target_units = 'days since 1970-01-01 00:00:00'
        time_output = date2num(time_as_datetime, units=target_units, calendar='gregorian')

        # Time variable
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

        # Altitude (scalar) - missing when unknown
        var_alt = ds_out.createVariable('altitude', 'f4', fill_value=FILL_VALUE)
        var_alt.standard_name = 'altitude'
        var_alt.long_name = 'station altitude above sea level'
        var_alt.units = 'm'
        var_alt[:] = altitude if not np.isnan(altitude) else FILL_VALUE

        # Upstream area (scalar) - missing when no discharge file
        var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE)
        var_area.long_name = 'upstream drainage area'
        var_area.units = 'km2'
        var_area[:] = upstream_area if not np.isnan(upstream_area) else FILL_VALUE

        # Discharge
        var_dis = ds_out.createVariable('discharge', 'f4', ('time',),
                                        fill_value=FILL_VALUE, chunksizes=[n_time])
        var_dis.standard_name = 'water_volume_transport_in_river_channel'
        var_dis.long_name = 'river discharge'
        var_dis.units = 'm3 s-1'
        var_dis.coordinates = 'time latitude longitude'
        var_dis[:] = discharge

        # SSC
        var_ssc = ds_out.createVariable('ssc', 'f4', ('time',),
                                        fill_value=FILL_VALUE, chunksizes=[n_time])
        var_ssc.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        var_ssc.long_name = 'suspended sediment concentration'
        var_ssc.units = 'mg L-1'
        var_ssc.coordinates = 'time latitude longitude'
        var_ssc[:] = ssc

        # Sediment load
        var_load = ds_out.createVariable('sediment_load', 'f4', ('time',),
                                         fill_value=FILL_VALUE, chunksizes=[n_time])
        var_load.long_name = 'suspended sediment load'
        var_load.units = 'ton day-1'
        var_load.coordinates = 'time latitude longitude'
        var_load.comment = 'Calculated as: sediment_load (ton day-1) = discharge (m3 s-1) * ssc (mg L-1) * 0.0864.'
        var_load[:] = sediment_load

        # Global attributes
        ds_out.Conventions = 'CF-1.8'
        ds_out.title = f'HYDAT Station {station_id} - Sediment Data'
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

        # Provenance: record whether discharge was available
        ds_out.has_discharge = 'true' if has_discharge else 'false'

    # --- Collect provenance info for audit ----------------------------------
    has_any_ssc = bool(np.any(ssc != FILL_VALUE))
    has_any_ssl = bool(np.any(sediment_load != FILL_VALUE))

    prov = {
        'station_id': station_id,
        'has_discharge': has_discharge,
        'has_ssc': has_any_ssc,
        'has_ssl': has_any_ssl,
        'ssc_only': has_any_ssc and not has_any_ssl,
        'ssl_only': has_any_ssl and not has_any_ssc,
    }

    print(f"  + Successfully updated: {Path(output_file).name}"
          f"  [discharge={'Y' if has_discharge else 'N'},"
          f" SSC={'Y' if has_any_ssc else 'N'},"
          f" SSL={'Y' if has_any_ssl else 'N'}]")
    return True, prov


def main():
    output_root = resolve_output_root(start=__file__, create=True)
    hydat_dir = output_root / "daily" / "HYDAT"
    sediment_dir = hydat_dir / "sediment"
    discharge_dir = hydat_dir / "discharge_waterlevel"
    output_dir = hydat_dir / "sediment_update"
    output_dir.mkdir(exist_ok=True)

    sediment_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    # Audit counters
    audit = {
        'sediment_total': 0,
        'with_discharge': 0,
        'without_discharge': 0,
        'success': 0,
        'ssc_only': 0,
        'ssl_only': 0,
        'both_ssc_ssl': 0,
    }

    tasks = []
    for sed_file in sediment_files:
        station_id = sed_file.stem.replace('HYDAT_', '').replace('_SEDIMENT', '')
        dis_file = discharge_dir / f'HYDAT_{station_id}.nc'
        out_file = output_dir / f'HYDAT_{station_id}_SEDIMENT.nc'

        has_dis = dis_file.exists()
        tasks.append((sed_file, dis_file if has_dis else None, out_file,
                      has_dis, station_id))

    audit['sediment_total'] = len(tasks)
    audit['with_discharge'] = sum(1 for t in tasks if t[3])
    audit['without_discharge'] = sum(1 for t in tasks if not t[3])

    print(f"\n{'='*70}")
    print(f"Sediment files total:       {audit['sediment_total']}")
    print(f"  With discharge:           {audit['with_discharge']}")
    print(f"  Without discharge:        {audit['without_discharge']}")
    print(f"{'='*70}\n")

    # Process all stations (with or without discharge)
    max_workers = min(16, max(1, len(tasks)))
    success_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(update_sediment_file, sed_file, dis_file, out_file): info
            for sed_file, dis_file, out_file, has_dis, stid in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task),
                           desc="Processing stations"):
            sed_file, dis_file, out_file, has_dis, stid = future_to_task[future]
            try:
                result = future.result()
                if result:
                    success, prov = result
                    if success:
                        success_count += 1
                        audit['success'] += 1
                        if prov['ssc_only']:
                            audit['ssc_only'] += 1
                        elif prov['ssl_only']:
                            audit['ssl_only'] += 1
                        elif prov['has_ssc'] and prov['has_ssl']:
                            audit['both_ssc_ssl'] += 1
            except Exception as e:
                print(f"+ Error processing {sed_file.name}: {e}")

    print(f"\n{'='*70}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"  Sediment stations total:   {audit['sediment_total']}")
    print(f"    With discharge:          {audit['with_discharge']}")
    print(f"    Without discharge:       {audit['without_discharge']}")
    print(f"  Successfully written:      {audit['success']}")
    print(f"    SSC-only stations:       {audit['ssc_only']}")
    print(f"    SSL-only stations:       {audit['ssl_only']}")
    print(f"    Both SSC+SSL stations:   {audit['both_ssc_ssl']}")
    print(f"{'='*70}\n")

    print(f"Updated {success_count}/{len(tasks)} files successfully")


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------
def _regression_test():
    """
    Minimal regression test: create a synthetic sediment NetCDF WITHOUT a
    matching discharge file, run update_sediment_file, and verify that the
    output contains the SSC/SSL data and that Q is all-missing.
    """
    import tempfile

    print("\n--- Regression test: sediment-only (no discharge) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sed_path = tmp / 'HYDAT_TEST01_SEDIMENT.nc'
        out_path = tmp / 'HYDAT_TEST01_SEDIMENT_out.nc'

        # --- Build a tiny sediment NetCDF ----------------------------------
        with nc.Dataset(sed_path, 'w', format='NETCDF4') as ds:
            ds.createDimension('station', 1)
            ds.createDimension('nv', 2)
            # Separate time dims for SSC and SSL (mimics real HYDAT structure)
            ds.createDimension('time_ssc', 3)
            ds.createDimension('time_ssl', 2)

            vlat = ds.createVariable('lat', 'f4', ('station',))
            vlat.units = 'degrees_north'
            vlat[:] = [49.0]

            vlon = ds.createVariable('lon', 'f4', ('station',))
            vlon.units = 'degrees_east'
            vlon[:] = [-120.0]

            # SSC: 3 time steps
            tssc = ds.createVariable('time_sed_suscon', 'f8', ('time_ssc',))
            tssc.units = 'days since 1850-01-01 00:00:00'
            tssc[:] = [60000.0, 60001.0, 60002.0]

            vssc = ds.createVariable('suspended_sediment_concentration', 'f4',
                                     ('time_ssc', 'station', 'nv'))
            vssc[:] = np.array([[[10.0, 0]], [[20.0, 0]], [[15.0, 0]]], dtype=np.float32)

            # SSL: 2 time steps (different from SSC to test union)
            tssl = ds.createVariable('time_sed_load', 'f8', ('time_ssl',))
            tssl.units = 'days since 1850-01-01 00:00:00'
            tssl[:] = [60000.0, 60003.0]  # overlaps at t=60000, extra at 60003

            vssl = ds.createVariable('sediment_load', 'f4',
                                     ('time_ssl', 'station', 'nv'))
            vssl[:] = np.array([[[5.0, 0]], [[8.0, 0]]], dtype=np.float32)

            ds.station_id = 'TEST01'
            ds.station_name = 'TEST RIVER'
            ds.province_territory = 'BC'

        # --- Run update WITHOUT discharge -----------------------------------
        success, prov = update_sediment_file(str(sed_path), discharge_file=None,
                                             output_file=str(out_path))
        assert success, "update_sediment_file should return True"
        assert not prov['has_discharge'], "should be marked as no-discharge"
        assert prov['has_ssc'], "SSC should be present"
        assert prov['has_ssl'], "SSL should be present"

        # Verify output NetCDF contents
        # Open with auto_mask=False to get raw numpy arrays (not masked)
        with nc.Dataset(out_path, 'r') as ds:
            ntime = len(ds.dimensions['time'])
            # 4 unique times: 60000, 60001, 60002 (SSC), 60003 (SSL)
            assert ntime == 4, f"expected 4 time steps, got {ntime}"

            ssc_out = ds['ssc'][:].data   # raw numpy (not masked)
            ssl_out = ds['sediment_load'][:].data
            q_out = ds['discharge'][:].data

            # SSC: 10, 20, 15 at t=0,1,2; missing at t=3
            assert ssc_out[0] == 10.0, f"ssc[0]={ssc_out[0]}"
            assert ssc_out[1] == 20.0, f"ssc[1]={ssc_out[1]}"
            assert ssc_out[2] == 15.0, f"ssc[2]={ssc_out[2]}"
            assert ssc_out[3] == FILL_VALUE, f"ssc[3] should be missing, got {ssc_out[3]}"

            # SSL: 5, missing, missing, 8
            assert ssl_out[0] == 5.0, f"ssl[0]={ssl_out[0]}"
            assert ssl_out[1] == FILL_VALUE, f"ssl[1] should be missing"
            assert ssl_out[2] == FILL_VALUE, f"ssl[2] should be missing"
            assert ssl_out[3] == 8.0, f"ssl[3]={ssl_out[3]}"

            # Q: all missing
            assert np.all(q_out == FILL_VALUE), "Q should be all missing"

            # Metadata - scalars with fill_value may be masked
            assert np.ma.is_masked(ds['upstream_area'][:]), "upstream_area should be masked/missing"
            assert np.ma.is_masked(ds['altitude'][:]), "altitude should be masked/missing"
            assert ds.has_discharge == 'false'

            # CF attributes
            assert ds['ssc'].units == 'mg L-1'
            assert ds['sediment_load'].units == 'ton day-1'
            assert ds['discharge'].units == 'm3 s-1'

        # --- Run update WITH discharge (synthetic) --------------------------
        dis_path = tmp / 'HYDAT_TEST01.nc'
        with nc.Dataset(dis_path, 'w', format='NETCDF4') as ds:
            ds.createDimension('station', 1)
            ds.createDimension('time', 3)
            ds.createDimension('nv', 2)

            tq = ds.createVariable('time_flow', 'f8', ('time',))
            tq.units = 'days since 1850-01-01 00:00:00'
            tq[:] = [60000.0, 60001.0, 60002.0]

            vq = ds.createVariable('discharge', 'f4', ('time', 'station', 'nv'))
            vq[:] = np.array([[[100.0, 0]], [[200.0, 0]], [[150.0, 0]]], dtype=np.float32)

            vda = ds.createVariable('drainage_area', 'f4', ('station',))
            vda[:] = [5000.0]

        out2_path = tmp / 'HYDAT_TEST01_SEDIMENT_out2.nc'
        success2, prov2 = update_sediment_file(str(sed_path), discharge_file=str(dis_path),
                                               output_file=str(out2_path))
        assert success2
        assert prov2['has_discharge']

        with nc.Dataset(out2_path, 'r') as ds:
            assert float(ds['upstream_area'][:]) == 5000.0
            assert ds.has_discharge == 'true'
            q_out2 = ds['discharge'][:].data
            # Q should NOT be all missing
            assert not np.all(q_out2 == FILL_VALUE), "Q should have data"
            assert q_out2[0] == 100.0, f"Q[0]={q_out2[0]}"
            assert q_out2[1] == 200.0, f"Q[1]={q_out2[1]}"
            assert q_out2[2] == 150.0, f"Q[2]={q_out2[2]}"

        print("  + All regression tests passed.\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Run regression test only')
    args = parser.parse_args()

    if args.test:
        _regression_test()
    else:
        main()
