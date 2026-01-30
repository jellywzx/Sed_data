#!/usr/bin/env python3
"""
HYBAM Comprehensive Data Processing Pipeline
=========================================
Implements complete QC workflow per CF-1.8 and ACDD-1.3 standards.

PROCESSING WORKFLOW:
1. Data Merging: Combine discharge and SSC data on common time axis
2. Unit Verification: Confirm units are m³/s and mg/L respectively
3. QC Checks: Physical consistency checks with quality flagging
4. Derived Variables: Calculate SSL = Q × SSC × 86.4 (ton/day)
5. CF-1.8 Standardization: Create compliant NetCDF files with:
   - Unlimited time dimension
   - Coordinate variables (time, lat, lon)
   - Quality flag variables with flag_values and flag_meanings
   - Comprehensive global attributes (ACDD-1.3)
6. CSV Output: Generate station metadata summary with coverage stats

QUALITY FLAGS:
  0 - good_data: Passes all QC checks
  1 - estimated_data: Interpolated or derived
  2 - suspect_data: Extreme or questionable values
  3 - bad_data: Negative, physically impossible values
  9 - missing_data: No measurement available

UNIT CONVERSIONS:
  SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4
  where 86.4 = 86400 s/day × 1000 L/m³ × 10⁻⁶ ton/mg

REFERENCE:
  ORE-HYBAM: Hydrologie et Géochimie du Bassin Amazonien
  URL: http://www.ore-hybam.org
"""

import os
import re
import csv
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    apply_quality_flag_array,                
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    apply_hydro_qc_with_provenance,           
    generate_csv_summary as generate_csv_summary_tool,          
    generate_qc_results_csv as generate_qc_results_csv_tool,
) #add 4 functions from tool.py

STATION_INFO = {
    "4071002205": {"lon": -63.40258, "lat": -18.90892, "alt": 430},
    "15900000":   {"lon": -59.59945, "lat": -4.389167, "alt": None},
    "10064000":   {"lon": -77.54837, "lat": -4.47023,  "alt": 200},
    "50800000":   {"lon": 15.31667,  "lat": -4.26667,  "alt": 270},
    "14710000":   {"lon": -61.12361, "lat": 1.821389,  "alt": None},
    "40800000":   {"lon": -63.6,     "lat": 8.14,      "alt": 8},
    "15860000":   {"lon": -60.02528, "lat": -4.897222, "alt": None},
    "10080900":   {"lon": -76.98917, "lat": -0.4411111,"alt": 330},
    "17730000":   {"lon": -57.58333, "lat": -4.283333, "alt": None},
    "10073500":   {"lon": -73.87119, "lat": -10.60762, "alt": 195},
    "2604100121": {"lon": -54.43333, "lat": 4.983333,  "alt": None},
    "14100000":   {"lon": -60.60944, "lat": -3.308333, "alt": 20},
    "17050001":   {"lon": -55.51111, "lat": -1.947222, "alt": None},
    "15400000":   {"lon": -63.92028, "lat": -8.736667, "alt": None},
    "15275100":   {"lon": -67.53496, "lat": -14.44091, "alt": 216},
    "2604500124": {"lon": -51.88334, "lat": 3.816667,  "alt": None},
    "14420000":   {"lon": -64.82889, "lat": -0.4819444,"alt": None}
}


class HYBAMProcessor:
    """Process HYBAM discharge and sediment data with full QC and CF-1.8 compliance."""

    # Quality flag definitions
    FLAG_GOOD = 0
    FLAG_ESTIMATED = 1
    FLAG_SUSPECT = 2
    FLAG_BAD = 3
    FLAG_MISSING = 9

    FLAG_MEANINGS = "good_data estimated_data suspect_data bad_data missing_data"

    def __init__(self, source_dir, output_dir, output_r_dir):
        """Initialize processor with directory paths."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_r_dir = Path(output_r_dir)
        self.station_metadata = {}

    def find_station_dirs(self):
        """Find all station directories in source."""
        return sorted([d for d in self.source_dir.iterdir() if d.is_dir() and '-' in d.name])

    def extract_station_info(self, station_dir):
        """Extract station ID, name, and river from directory name.

        Expected format: Name_-_RiverName_-_StationID
        Example: Borja_-_Marañon_-_10064000
        """
        parts = station_dir.name.split('_-_')
        if len(parts) >= 3:
            station_name = parts[0].replace('_', ' ')
            river_name = parts[1].replace('_', ' ')
            station_id = parts[2]
            return station_id, station_name, river_name
        return None, None, None

    def read_hybrid_metadata(self):
        """Read station metadata from output nc files."""
        # For HYBAM, we'll extract metadata from existing NC files or source data
        for nc_file in self.output_dir.glob('HYBAM_*.nc'):
            try:
                with nc.Dataset(nc_file, 'r') as ds:
                    station_id = ds.getncattr('station_id')
                    station_name = ds.getncattr('station_name')
                    river_name = ds.getncattr('river_name')

                    self.station_metadata[station_id] = {
                        'station_name': station_name,
                        'river_name': river_name,
                        'nc_file': str(nc_file)
                    }
            except:
                pass

    def find_data_files(self, station_dir, station_id):
        """Find discharge and SSC files for a station.

        Returns:
            tuple: (discharge_file, ssc_file, metadata_dict)
        """
        discharge_file = None
        ssc_file = None
        metadata = {
            'station_id': station_id,
            'has_discharge': False,
            'has_ssc': False,
        }

        # Look for discharge file (*_D_J1_*m3*s*.nc)
        discharge_files = list(station_dir.glob('*_D_J1_*m3*s*.nc'))
        if discharge_files:
            discharge_file = discharge_files[0]
            metadata['has_discharge'] = True

        # Look for SSC file (*_Q_*mg*l*.nc or *_IMsO_*mg*l*.nc)
        ssc_files = list(station_dir.glob('*_Q_*mg*l*.nc')) + \
                    list(station_dir.glob('*_IMsO_*mg*l*.nc'))
        if ssc_files:
            ssc_file = ssc_files[0]
            metadata['has_ssc'] = True

        return discharge_file, ssc_file, metadata

    def read_nc_data(self, nc_file):
        """Read time and data from NC file.

        Returns:
            tuple: (time_seconds, data_values, data_varname, fill_value, origine, qualite)
        """
        with nc.Dataset(nc_file, 'r') as ds:
            # Read time (unix seconds)
            time_var = ds.variables['Date']
            time_seconds = time_var[:]

            # Find data variable (skip metadata variables)
            data_varname = None
            for var in ds.variables:
                if var not in ['Date', '_Origine', '_Qualité']:
                    data_varname = var
                    break

            if not data_varname:
                return None, None, None, None, None, None

            # Read data values
            data_values = ds.variables[data_varname][:]
            fill_value = getattr(ds.variables[data_varname], '_FillValue', FILL_VALUE_FLOAT)

            # Read quality information
            origine = ds.variables.get('_Origine', [None] * len(time_seconds))[:]
            qualite = ds.variables.get('_Qualité', [None] * len(time_seconds))[:]

            return time_seconds, data_values, data_varname, fill_value, origine, qualite

    def merge_discharge_ssc(self, discharge_file, ssc_file):
        """Merge discharge and SSC data on common time axis.

        Returns:
            dict: Merged data with time, discharge, ssc, and related metadata
        """
        result = {
            'time': None,
            'discharge': None,
            'ssc': None,
            'discharge_origin': None,
            'discharge_quality': None,
            'ssc_origin': None,
            'ssc_quality': None,
            'time_coverage_start': None,
            'time_coverage_end': None,
            'q_start': None,
            'q_end': None,
            'ssc_start': None,
            'ssc_end': None,
        }

        discharge_time = None
        ssc_time = None

        if discharge_file:
            discharge_time, discharge_data, _, discharge_fill, discharge_origin, discharge_quality = \
                self.read_nc_data(discharge_file)
            result['discharge_raw'] = discharge_data
            result['discharge_fill'] = discharge_fill

        if ssc_file:
            ssc_time, ssc_data, _, ssc_fill, ssc_origin, ssc_quality = \
                self.read_nc_data(ssc_file)
            result['ssc_raw'] = ssc_data
            result['ssc_fill'] = ssc_fill

        # Determine time axis: use intersection or single source
        if discharge_time is not None and ssc_time is not None:
            # Find overlapping time range
            q_start_idx = np.where(discharge_time >= np.min(ssc_time))[0]
            q_end_idx = np.where(discharge_time <= np.max(ssc_time))[0]

            if len(q_start_idx) > 0 and len(q_end_idx) > 0:
                q_start = q_start_idx[0]
                q_end = q_end_idx[-1]

                # Use discharge time as base
                result['time'] = discharge_time[q_start:q_end+1]
                result['discharge'] = discharge_data[q_start:q_end+1]
                result['discharge_origin'] = discharge_origin[q_start:q_end+1] if discharge_origin is not None else None
                result['discharge_quality'] = discharge_quality[q_start:q_end+1] if discharge_quality is not None else None

                # Map SSC to discharge time axis
                ssc_mapped = np.full(len(result['time']), FILL_VALUE_FLOAT, dtype='f4')
                for i, q_time in enumerate(result['time']):
                    # Find closest SSC time
                    idx = np.argmin(np.abs(ssc_time - q_time))
                    if np.abs(ssc_time[idx] - q_time) < 86400:  # Within 1 day
                        ssc_mapped[i] = ssc_data[idx]

                result['ssc'] = ssc_mapped
                result['q_start'] = q_start
                result['q_end'] = q_end
        elif discharge_time is not None:
            result['time'] = discharge_time
            result['discharge'] = discharge_data
            result['discharge_origin'] = discharge_origin if discharge_origin is not None else None
            result['discharge_quality'] = discharge_quality if discharge_quality is not None else None
        elif ssc_time is not None:
            result['time'] = ssc_time
            result['ssc'] = ssc_data

        # Set time coverage
        if result['time'] is not None and len(result['time']) > 0:
            result['time_coverage_start'] = datetime.utcfromtimestamp(result['time'][0]).strftime('%Y-%m-%d')
            result['time_coverage_end'] = datetime.utcfromtimestamp(result['time'][-1]).strftime('%Y-%m-%d')

        return result

    def apply_qc_checks(self, data_dict):
        """
        Apply QC via tool.apply_hydro_qc_with_provenance (QC1/QC2/QC3 + provenance flags)
        and make sure all arrays are 1D and length-aligned.
        new:change SSC/time/Q to iD arrays and aligned lengths
        """

        # ---------- helper: force 1D ----------
        def _as_1d(x):
            if x is None:
                return None
            arr = np.asarray(x)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            else:
                arr = np.squeeze(arr)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
            return arr

        # ---------- helper: align to length n ----------
        def _align_len(arr, n, fill=np.nan):
            if arr is None:
                return None
            a = _as_1d(arr)
            if a.size == n:
                return a
            if a.size > n:
                return a[:n]
            # pad
            out = np.full(n, fill, dtype=float)
            out[:a.size] = a.astype(float)
            return out

        # ---- force 1D time ----
        time_sec = _as_1d(data_dict.get("time"))
        if time_sec is None or time_sec.size == 0:
            return None

        n = time_sec.size

        Q = _align_len(data_dict.get("discharge"), n, fill=np.nan)
        SSC = _align_len(data_dict.get("ssc"), n, fill=np.nan)

        # ---- derive SSL (ton/day) if Q&SSC exist ----
        # 注意：如果你最终 nc 里 SSL 的单位是 "ton day-1"，系数应是 0.0864
        # (kg/s -> ton/day) = 86400/1000 = 86.4 ; 但 mg/L -> kg/m3 还有 1e-3，合起来就是 0.0864
        SSL = np.full(n, np.nan, dtype=float)
        if Q is not None and SSC is not None:
            m = np.isfinite(Q) & np.isfinite(SSC)
            SSL[m] = Q[m] * SSC[m] * 0.0864

        qc = apply_hydro_qc_with_provenance(
            time=time_sec,
            Q=np.full(n, np.nan) if Q is None else Q,
            SSC=np.full(n, np.nan) if SSC is None else SSC,
            SSL=SSL,
            Q_is_independent=True,
            SSC_is_independent=True,
            SSL_is_independent=False,
            ssl_is_derived_from_q_ssc=True,
        )

        if qc is None:
            return None

        # ---- write back (统一字段名) ----
        data_dict["time"] = qc["time"]
        data_dict["discharge"] = qc["Q"]
        data_dict["ssc"] = qc["SSC"]
        data_dict["SSL"] = qc["SSL"]

        # final flags
        data_dict["Q_flag"] = qc["Q_flag"]
        data_dict["SSC_flag"] = qc["SSC_flag"]
        data_dict["SSL_flag"] = qc["SSL_flag"]

        # step/provenance flags（分步骤写入的关键）
        data_dict["Q_flag_qc1_physical"] = qc["Q_flag_qc1_physical"]
        data_dict["SSC_flag_qc1_physical"] = qc["SSC_flag_qc1_physical"]
        data_dict["SSL_flag_qc1_physical"] = qc["SSL_flag_qc1_physical"]

        data_dict["Q_flag_qc2_log_iqr"] = qc["Q_flag_qc2_log_iqr"]
        data_dict["SSC_flag_qc2_log_iqr"] = qc["SSC_flag_qc2_log_iqr"]
        data_dict["SSL_flag_qc2_log_iqr"] = qc["SSL_flag_qc2_log_iqr"]

        data_dict["SSC_flag_qc3_ssc_q"] = qc["SSC_flag_qc3_ssc_q"]
        data_dict["SSL_flag_qc3_from_ssc_q"] = qc["SSL_flag_qc3_from_ssc_q"]

        # qc3 envelope for plotting (optional)
        data_dict["ssc_q_bounds"] = qc.get("ssc_q_bounds", None)

        return data_dict


    def get_reference_info(self):
        """Get reference information for HYBAM dataset."""
        return {
            'reference': 'ORE-HYBAM: Observatoire de Recherche sur l\'Environnement en Amazonie - Hydrologie et Géochimie du Bassin Amazonien. http://www.ore-hybam.org',
            'source_data_link': 'http://www.ore-hybam.org',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
        }

    def write_cf18_netcdf(self, station_id, station_name, river_name, latitude, longitude,
                         altitude, upstream_area, data_dict, output_file):
        """Write CF-1.8 compliant NetCDF file with quality flags.

        CF-1.8 Requirements:
        - Dimensions: time (UNLIMITED)
        - Coordinates: time, lat, lon (as scalars)
        - Coordinate variables with standard_name, long_name, units
        - Data variables with ancillary_variables pointing to flags
        - Quality flag variables with flag_values and flag_meanings
        """

        # Prepare data
        time_days = data_dict['time'] / 86400.0
        n_times = len(time_days)

        # Prepare fill value (from tool.py)
        fill_value = FILL_VALUE_FLOAT

        with nc.Dataset(output_file, 'w', format='NETCDF4', diskless=False) as ds:
            # =============
            # Dimensions
            # =============
            time_dim = ds.createDimension('time', None)  # UNLIMITED

            # =============
            # Coordinate Variables
            # =============
            # Time
            time_var = ds.createVariable('time', 'f8', ('time',), zlib=True)
            time_var.standard_name = 'time'
            time_var.long_name = 'time'
            time_var.units = f'days since 1970-01-01 00:00:00'
            time_var.calendar = 'gregorian'
            time_var.axis = 'T'
            time_var[:] = time_days

            # Latitude (scalar)
            lat_var = ds.createVariable('lat', 'f4', zlib=True)
            lat_var.standard_name = 'latitude'
            lat_var.long_name = 'station latitude'
            lat_var.units = 'degrees_north'
            lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
            lat_var[:] = latitude

            # Longitude (scalar)
            lon_var = ds.createVariable('lon', 'f4', zlib=True)
            lon_var.standard_name = 'longitude'
            lon_var.long_name = 'station longitude'
            lon_var.units = 'degrees_east'
            lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
            lon_var[:] = longitude

            # =============
            # Static Variables
            # =============
            if altitude is not None:
                alt_var = ds.createVariable('altitude', 'f4', zlib=True)
                alt_var.standard_name = 'altitude'
                alt_var.long_name = 'station elevation above sea level'
                alt_var.units = 'm'
                alt_var.positive = 'up'
                alt_var[:] = altitude

            if upstream_area is not None:
                area_var = ds.createVariable('upstream_area', 'f4', zlib=True)
                area_var.long_name = 'upstream drainage area'
                area_var.units = 'km2'
                area_var[:] = upstream_area

            # =============
            # Data Variables with Quality Flags
            # =============
            # Discharge (Q)
            if data_dict['discharge'] is not None:
                Q_var = ds.createVariable('Q', 'f4', ('time',), zlib=True, complevel=4, fill_value=fill_value)
                Q_var.standard_name = 'water_volume_transport_in_river_channel'
                Q_var.long_name = 'river discharge'
                Q_var.units = 'm3 s-1'
                Q_var.coordinates = 'time lat lon'
                Q_var.ancillary_variables = 'Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'
                Q_var.comment = 'Source: Original data from ORE-HYBAM monitoring network.'
                Q_var.missing_value = fill_value
                Q_var[:] = data_dict['discharge']

                # Q Quality Flag
                Q_flag_var = ds.createVariable('Q_flag', 'i1', ('time',), zlib=True, complevel=4, fill_value=FILL_VALUE_INT)
                Q_flag_var.long_name = 'quality flag for river discharge'
                Q_flag_var.standard_name = 'status_flag'
                Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
                Q_flag_var.flag_meanings = self.FLAG_MEANINGS
                Q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                Q_flag_var.missing_value = FILL_VALUE_INT
                Q_flag_var[:] = data_dict['Q_flag']
                def _add_step_flag(ds, name, values, *, flag_values, flag_meanings, long_name):
                    v = ds.createVariable(name, 'i1', ('time',), zlib=True, complevel=4, fill_value=FILL_VALUE_INT)
                    v.long_name = long_name
                    v.standard_name = 'status_flag'
                    v.c = np.array(flag_values, dtype=np.int8)
                    v.flag_meanings = flag_meanings
                    v.missing_value = np.int8(FILL_VALUE_INT)
                    v[:] = np.asarray(values, dtype=np.int8)
                    return v

                # ---- QC1 physical: 0 pass, 3 bad, 9 missing ----
                _add_step_flag(ds,'Q_flag_qc1_physical', data_dict['Q_flag_qc1_physical'],
                            flag_values=[0, 3, 9], 
                            flag_meanings='pass bad missing',
                            long_name='QC1 physical flag for river discharge')

                # ---- QC2 log-IQR: 0 pass, 2 suspect, 8 not_checked, 9 missing ----
                _add_step_flag(ds,'Q_flag_qc2_log_iqr', data_dict['Q_flag_qc2_log_iqr'],
                            flag_values=[0, 2, 8, 9], 
                            flag_meanings='pass suspect not_checked missing',
                            long_name='QC2 log-IQR flag for river discharge')


            # Suspended Sediment Concentration (SSC)
            if data_dict['ssc'] is not None:
                SSC_var = ds.createVariable('SSC', 'f4', ('time',), zlib=True, complevel=4, fill_value=fill_value)
                SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                SSC_var.long_name = 'suspended sediment concentration'
                SSC_var.units = 'mg L-1'
                SSC_var.coordinates = 'time lat lon'
                SSC_var.ancillary_variables = 'SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'
                SSC_var.comment = 'Source: Original data from ORE-HYBAM monitoring network.'
                SSC_var.missing_value = fill_value
                SSC_var[:] = data_dict['ssc']

                # SSC Quality Flag
                SSC_flag_var = ds.createVariable('SSC_flag', 'i1', ('time',), zlib=True, complevel=4, fill_value=FILL_VALUE_INT)
                SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
                SSC_flag_var.standard_name = 'status_flag'
                SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
                SSC_flag_var.flag_meanings = self.FLAG_MEANINGS
                SSC_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                SSC_flag_var.missing_value = FILL_VALUE_INT
                SSC_flag_var[:] = data_dict['SSC_flag']
                # -------------------------------
                # Step flags for SSC (QC1/QC2/QC3)
                # -------------------------------

                # QC1: physical (0 pass, 3 bad, 9 missing)
                _add_step_flag(
                    ds, "SSC_flag_qc1_physical", data_dict["SSC_flag_qc1_physical"],
                    flag_values=[0, 3, 9],
                    flag_meanings="pass bad missing",
                    long_name="QC1 physical check flag for suspended sediment concentration"
                )

                # QC2: log-IQR (0 pass, 2 suspect, 8 not_checked, 9 missing)
                _add_step_flag(
                    ds, "SSC_flag_qc2_log_iqr", data_dict["SSC_flag_qc2_log_iqr"],
                    flag_values=[0, 2, 8, 9],
                    flag_meanings="pass suspect not_checked missing",
                    long_name="QC2 log-IQR flag for suspended sediment concentration"
                )

                # QC3: SSC-Q consistency (0 pass, 2 suspect, 8 not_checked, 9 missing)
                _add_step_flag(
                    ds, "SSC_flag_qc3_ssc_q", data_dict["SSC_flag_qc3_ssc_q"],
                    flag_values=[0, 2, 8, 9],
                    flag_meanings="pass suspect not_checked missing",
                    long_name="QC3 SSC–Q consistency flag for suspended sediment concentration"
                )

                SSC_var.ancillary_variables = (
                    "SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q"
                )


            # Suspended Sediment Load (SSL)
            if 'SSL' in data_dict:
                SSL_var = ds.createVariable('SSL', 'f4', ('time',), zlib=True, complevel=4, fill_value=fill_value)
                SSL_var.standard_name = 'suspended_sediment_transport_in_river'
                SSL_var.long_name = 'suspended sediment load'
                SSL_var.units = 'ton day-1'
                SSL_var.coordinates = 'time lat lon'
                SSL_var.ancillary_variables = 'SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q'
                SSL_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4'
                SSL_var.missing_value = fill_value
                SSL_var[:] = data_dict['SSL']

                # SSL Quality Flag
                SSL_flag_var = ds.createVariable('SSL_flag', 'i1', ('time',), zlib=True, complevel=4, fill_value=FILL_VALUE_INT)
                SSL_flag_var.long_name = 'quality flag for suspended sediment load'
                SSL_flag_var.standard_name = 'status_flag'
                SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
                SSL_flag_var.flag_meanings = self.FLAG_MEANINGS
                SSL_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                SSL_flag_var.missing_value = FILL_VALUE_INT
                SSL_flag_var[:] = data_dict['SSL_flag']
                # -------------------------------
                # Step flags for SSL (QC1/QC2/QC3)
                # -------------------------------
                # QC1: physical (0 pass, 3 bad, 9 missing)
                _add_step_flag(
                    ds, "SSL_flag_qc1_physical", data_dict["SSL_flag_qc1_physical"],
                    flag_values=[0, 3, 9],
                    flag_meanings="pass bad missing",
                    long_name="QC1 physical check flag for suspended sediment load"
                )

                # QC2: log-IQR (0 pass, 2 suspect, 8 not_checked, 9 missing)
                _add_step_flag(
                    ds, "SSL_flag_qc2_log_iqr", data_dict["SSL_flag_qc2_log_iqr"],
                    flag_values=[0, 2, 8, 9],
                    flag_meanings="pass suspect not_checked missing",
                    long_name="QC2 log-IQR flag for suspended sediment load"
                )

                # QC3: propagated-from-SSC-Q inconsistency
                # 0 not_propagated, 2 propagated, 8 not_checked, 9 missing
                _add_step_flag(
                    ds, "SSL_flag_qc3_from_ssc_q", data_dict["SSL_flag_qc3_from_ssc_q"],
                    flag_values=[0, 2, 8, 9],
                    flag_meanings="not_propagated propagated not_checked missing",
                    long_name="QC3 flag: SSL marked/propagated from SSC–Q inconsistency"
                )

                SSL_var.ancillary_variables = (
                    "SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q"
                )


            # =============
            # Global Attributes (CF-1.8 & ACDD-1.3)
            # =============
            ds.Conventions = 'CF-1.8, ACDD-1.3'
            ds.title = 'Harmonized Global River Discharge and Sediment'
            ds.summary = f'River discharge and suspended sediment data for {station_name} station on the {river_name} River, part of the Amazon Basin. This dataset contains daily measurements including discharge, suspended sediment concentration, and calculated sediment load.'
            ds.source = 'In-situ station data'
            ds.data_source_name = 'HYBAM Dataset'

            # Station information
            ds.station_name = station_name
            ds.river_name = river_name
            ds.Source_ID = station_id

            # Geospatial attributes
            ds.geospatial_lat_min = float(latitude)
            ds.geospatial_lat_max = float(latitude)
            ds.geospatial_lon_min = float(longitude)
            ds.geospatial_lon_max = float(longitude)
            if altitude is not None:
                ds.geospatial_vertical_min = float(altitude)
                ds.geospatial_vertical_max = float(altitude)

            # Geographic coverage
            ds.geographic_coverage = 'Amazon Basin'

            # Temporal attributes
            if data_dict['time_coverage_start']:
                ds.time_coverage_start = data_dict['time_coverage_start']
                ds.time_coverage_end = data_dict['time_coverage_end']
                ds.temporal_span = f"{data_dict['time_coverage_start'][:4]}-{data_dict['time_coverage_end'][:4]}"

            ds.temporal_resolution = 'daily'

            # Variables provided
            vars_provided = []
            if data_dict['discharge'] is not None:
                vars_provided.append('Q')
            if data_dict['ssc'] is not None:
                vars_provided.append('SSC')
            if 'SSL' in data_dict:
                vars_provided.append('SSL')
            if altitude is not None:
                vars_provided.append('altitude')
            if upstream_area is not None:
                vars_provided.append('upstream_area')

            ds.variables_provided = ', '.join(vars_provided)

            # Reference and creator
            ref_info = self.get_reference_info()
            ds.reference = ref_info['reference']
            ds.source_data_link = ref_info['source_data_link']
            ds.creator_name = ref_info['creator_name']
            ds.creator_email = ref_info['creator_email']
            ds.creator_institution = ref_info['creator_institution']

            # Processing history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Processed with hybam_comprehensive_processor.py - Merged discharge and sediment data, applied QC checks, generated quality flags, standardized to CF-1.8 format."
            ds.history = history

            # Dates
            ds.date_created = datetime.now().strftime('%Y-%m-%d')
            ds.date_modified = datetime.now().strftime('%Y-%m-%d')

            # Processing level
            ds.processing_level = 'Quality controlled and standardized'

            ds.number_of_data = str(len([d for d in data_dict['discharge'] if d != fill_value]) if data_dict['discharge'] is not None else 0)
            # Add legacy/alternate global attribute names expected by completeness checker
            ds.Type = 'In-situ'
            ds.Temporal_Resolution = ds.temporal_resolution
            ds.Temporal_Span = ds.temporal_span if hasattr(ds, 'temporal_span') else ''
            ds.Variables_Provided = ds.variables_provided
            ds.Geographic_Coverage = ds.geographic_coverage
            ds.Reference = ds.reference
            # Location and administrative metadata (best-effort)
            ds.location_id = station_id
            ds.country = ''
            ds.continent_region = ''

    def process_station(self, station_dir):
        """Process a single station through the complete pipeline."""
        # Extract metadata
        station_id, station_name, river_name = self.extract_station_info(station_dir)

        if not station_id:
            print(f"  ✗ Could not extract station info from {station_dir.name}")
            return False

        print(f"\n  Processing: {station_id} ({station_name} / {river_name})")

        # Find data files
        discharge_file, ssc_file, file_metadata = self.find_data_files(station_dir, station_id)

        if not discharge_file and not ssc_file:
            print(f"    ✗ No discharge or SSC data found")
            return False

        print(f"    ✓ Discharge: {discharge_file.name if discharge_file else 'N/A'}")
        print(f"    ✓ SSC: {ssc_file.name if ssc_file else 'N/A'}")

        # Merge data
        data = self.merge_discharge_ssc(discharge_file, ssc_file)

        if data['time'] is None or len(data['time']) == 0:
            print(f"    ✗ No time data extracted")
            return False

        print(f"    ✓ Time range: {data['time_coverage_start']} to {data['time_coverage_end']} ({len(data['time'])} days)")

        # Apply QC checks
        # Ensure identifying metadata present for plotting and outputs
        data['station_id'] = station_id
        data['station_name'] = station_name
        data['river_name'] = river_name

        data = self.apply_qc_checks(data)
        if data is None or data.get("time") is None or len(data["time"]) == 0:
            print("    ✗ No valid data after QC (all missing)")
            return False

        # Get metadata from existing HYBAM file
        latitude = FILL_VALUE_FLOAT
        longitude = FILL_VALUE_FLOAT
        altitude = FILL_VALUE_FLOAT
        upstream_area = FILL_VALUE_FLOAT

        info = STATION_INFO.get(station_id, None)

        if info:
            latitude = info["lat"]
            longitude = info["lon"]
            altitude = info["alt"]
        else:
            latitude = FILL_VALUE_FLOAT
            longitude = FILL_VALUE_FLOAT
            altitude = FILL_VALUE_FLOAT

        # Write CF-1.8 compliant NetCDF
        output_file = self.output_r_dir / f'HYBAM_{station_id}.nc'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.write_cf18_netcdf(station_id, station_name, river_name, latitude, longitude,
                              altitude, upstream_area, data, output_file)

        return {
            'station_id': station_id,
            'station_name': station_name,
            'river_name': river_name,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'upstream_area': upstream_area,
            'time_coverage_start': data['time_coverage_start'],
            'time_coverage_end': data['time_coverage_end'],
            'data': data,
        }

    def generate_csv_summary(self, stations_data, output_file):
        """Generate CSV summary of station metadata and data coverage."""
        fieldnames = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude', 'altitude',
            'upstream_area', 'Data Source Name', 'Type', 'Temporal Resolution', 'Temporal Span',
            'Variables Provided', 'Geographic Coverage', 'Reference/DOI',
            'Q_start_date', 'Q_end_date', 'Q_percent_complete',
            'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
            'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
        ]

        rows = []
        for station in stations_data:
            data = station['data']

            # Calculate percent complete for Q
            if data['discharge'] is not None:
                q_good = np.sum(data['Q_flag'] == self.FLAG_GOOD)
                q_total = np.sum(data['Q_flag'] != self.FLAG_MISSING)
                q_pct = (q_good / q_total * 100) if q_total > 0 else 0
            else:
                q_pct = 0

            # Calculate percent complete for SSC
            if data['ssc'] is not None:
                ssc_good = np.sum(data['SSC_flag'] == self.FLAG_GOOD)
                ssc_total = np.sum(data['SSC_flag'] != self.FLAG_MISSING)
                ssc_pct = (ssc_good / ssc_total * 100) if ssc_total > 0 else 0
            else:
                ssc_pct = 0

            # Calculate percent complete for SSL
            if 'SSL' in data:
                ssl_good = np.sum(data['SSL_flag'] == self.FLAG_GOOD)
                ssl_total = np.sum(data['SSL_flag'] != self.FLAG_MISSING)
                ssl_pct = (ssl_good / ssl_total * 100) if ssl_total > 0 else 0
            else:
                ssl_pct = 0

            # Determine variables provided
            vars_prov = []
            if data['discharge'] is not None:
                vars_prov.append('Q')
            if data['ssc'] is not None:
                vars_prov.append('SSC')
            if 'SSL' in data:
                vars_prov.append('SSL')

            row = {
                'station_name': station['station_name'],
                'Source_ID': station['station_id'],
                'river_name': station['river_name'],
                'longitude': station['longitude'],
                'latitude': station['latitude'],
                'altitude': station['altitude'],
                'upstream_area': station['upstream_area'],
                'Data Source Name': 'HYBAM Dataset',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{data['time_coverage_start'][:4]}-{data['time_coverage_end'][:4]}" if data['time_coverage_start'] else 'N/A',
                'Variables Provided': ', '.join(vars_prov),
                'Geographic Coverage': 'Amazon Basin',
                'Reference/DOI': 'http://www.ore-hybam.org',
                'Q_start_date': data['time_coverage_start'] if data['discharge'] is not None else '',
                'Q_end_date': data['time_coverage_end'] if data['discharge'] is not None else '',
                'Q_percent_complete': f"{q_pct:.1f}" if data['discharge'] is not None else '',
                'SSC_start_date': data['time_coverage_start'] if data['ssc'] is not None else '',
                'SSC_end_date': data['time_coverage_end'] if data['ssc'] is not None else '',
                'SSC_percent_complete': f"{ssc_pct:.1f}" if data['ssc'] is not None else '',
                'SSL_start_date': data['time_coverage_start'] if 'SSL' in data else '',
                'SSL_end_date': data['time_coverage_end'] if 'SSL' in data else '',
                'SSL_percent_complete': f"{ssl_pct:.1f}" if 'SSL' in data else '',
            }
            rows.append(row)

        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n  ✓ CSV summary written: {output_file.name}")

    def run(self):
        """Run the complete processing pipeline."""
        print("="*70)
        print("HYBAM Comprehensive Processing Pipeline")
        print("="*70)

        # Find stations
        station_dirs = self.find_station_dirs()
        print(f"\nFound {len(station_dirs)} station directories")

        # Process each station
        successful_stations = []
        failed_stations = []

        for i, station_dir in enumerate(station_dirs, 1):
            print(f"\n[{i}/{len(station_dirs)}]", end="")
            try:
                result = self.process_station(station_dir)
                if result:
                    successful_stations.append(result)
                else:
                    failed_stations.append(station_dir.name)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                failed_stations.append(station_dir.name)

        # Generate CSV summary
        if successful_stations:
            csv_output = self.output_r_dir / 'HYBAM_station_summary.csv'
            generate_csv_summary_tool(successful_stations, csv_output)
            # 2) 生成 QC 结果汇总 CSV
            qc_rows = []
            for s in successful_stations:
                data = s.get("data", {})
                row = {
                    "station_name": s.get("station_name", ""),
                    "Source_ID": s.get("station_id", ""),
                    "river_name": s.get("river_name", ""),
                    "longitude": s.get("longitude", ""),
                    "latitude": s.get("latitude", ""),
                    "QC_n_days": len(data.get("time", [])),
                }

                def _cnt(arr, v):
                    a = np.asarray(arr) if arr is not None else np.asarray([])
                    return int(np.sum(a == np.int8(v)))

                # final counts
                for var in ["Q", "SSC", "SSL"]:
                    f = data.get(f"{var}_flag", None)
                    row[f"{var}_final_good"] = _cnt(f, 0)
                    row[f"{var}_final_estimated"] = _cnt(f, 1)
                    row[f"{var}_final_suspect"] = _cnt(f, 2)
                    row[f"{var}_final_bad"] = _cnt(f, 3)
                    row[f"{var}_final_missing"] = _cnt(f, 9)

                # step counts（按工具函数约定）
                # qc1: 0 pass, 3 bad, 9 missing
                for var in ["Q", "SSC", "SSL"]:
                    f = data.get(f"{var}_flag_qc1_physical", None)
                    row[f"{var}_qc1_pass"] = _cnt(f, 0)
                    row[f"{var}_qc1_bad"] = _cnt(f, 3)
                    row[f"{var}_qc1_missing"] = _cnt(f, 9)

                # qc2: 0 pass, 2 suspect, 8 not_checked, 9 missing
                for var in ["Q", "SSC", "SSL"]:
                    f = data.get(f"{var}_flag_qc2_log_iqr", None)
                    row[f"{var}_qc2_pass"] = _cnt(f, 0)
                    row[f"{var}_qc2_suspect"] = _cnt(f, 2)
                    row[f"{var}_qc2_not_checked"] = _cnt(f, 8)
                    row[f"{var}_qc2_missing"] = _cnt(f, 9)

                # qc3（SSC/SSL有）
                f3 = data.get("SSC_flag_qc3_ssc_q", None)
                row["SSC_qc3_pass"] = _cnt(f3, 0)
                row["SSC_qc3_suspect"] = _cnt(f3, 2)
                row["SSC_qc3_not_checked"] = _cnt(f3, 8)
                row["SSC_qc3_missing"] = _cnt(f3, 9)

                fssl3 = data.get("SSL_flag_qc3_from_ssc_q", None)
                row["SSL_qc3_not_propagated"] = _cnt(fssl3, 0)
                row["SSL_qc3_propagated"] = _cnt(fssl3, 2)
                row["SSL_qc3_not_checked"] = _cnt(fssl3, 8)
                row["SSL_qc3_missing"] = _cnt(fssl3, 9)

                qc_rows.append(row)

            qc_csv = self.output_r_dir / "HYBAM_qc_results.csv"
            generate_qc_results_csv_tool(qc_rows, qc_csv)


        # Print summary
        print("\n" + "="*70)
        print("Processing Complete!")
        print(f"Successfully processed: {len(successful_stations)} stations")
        print(f"Failed: {len(failed_stations)} stations")
        if failed_stations:
            print(f"Failed stations: {', '.join(failed_stations[:5])}")
            if len(failed_stations) > 5:
                print(f"  ... and {len(failed_stations) - 5} more")

        print(f"\nOutput directory: {self.output_r_dir}")
        print("="*70)


def main():
    """Main entry point."""
    # Get base directory (project root) - go up two levels from Script/HYBAM/
    base_dir = Path(__file__).parent.parent.parent
    source_dir = base_dir / 'Source' / 'HYBAM' / 'source'
    output_dir = base_dir / 'Output_r' / 'daily' / 'HYBAM' / 'Output'
    output_r_dir = base_dir / 'Output_r' / 'daily' / 'HYBAM' / 'qc'

    processor = HYBAMProcessor(source_dir, output_dir, output_r_dir)
    processor.run()


if __name__ == '__main__':
    main()
