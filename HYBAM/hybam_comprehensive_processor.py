#!/usr/bin/env python3
"""
HYBAM Comprehensive Data Processing Pipeline
=========================================
Implements complete QC workflow per CF-1.8 and ACDD-1.3 standards.

Raw discharge and SSC series are first consolidated to one UTC-day record per
variable before alignment and QC. Daily SSL is then calculated from the daily
Q and SSC values as SSL = Q x SSC x 0.0864 ton/day.
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
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.daily_aggregation import aggregate_unix_series_to_daily, align_daily_series
from code.output import (
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)
from code.qc import (
    apply_quality_flag,
    apply_hydro_qc_with_provenance,
    apply_quality_flag_array,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
)
from code.runtime import resolve_output_root, resolve_source_root
from code.units import convert_ssl_units_if_needed

STATION_INFO = {
    "4071002205": {"lon": -63.40258, "lat": -18.90892, "alt": 430, "country": "Bolivia", "continent_region": "South America", "iso_a3": "BOL"},
    "15900000": {"lon": -59.59945, "lat": -4.389167, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "10064000": {"lon": -77.54837, "lat": -4.47023, "alt": 200, "country": "Peru", "continent_region": "South America", "iso_a3": "PER"},
    "50800000": {"lon": 15.31667, "lat": -4.26667, "alt": 270, "country": "Republic of the Congo", "continent_region": "Africa", "iso_a3": "COG"},
    "14710000": {"lon": -61.12361, "lat": 1.821389, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "40800000": {"lon": -63.6, "lat": 8.14, "alt": 8, "country": "Venezuela", "continent_region": "South America", "iso_a3": "VEN"},
    "15860000": {"lon": -60.02528, "lat": -4.897222, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "10080900": {"lon": -76.98917, "lat": -0.4411111, "alt": 330, "country": "Ecuador", "continent_region": "South America", "iso_a3": "ECU"},
    "17730000": {"lon": -57.58333, "lat": -4.283333, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "10073500": {"lon": -73.87119, "lat": -10.60762, "alt": 195, "country": "Peru", "continent_region": "South America", "iso_a3": "PER"},
    "2604100121": {"lon": -54.43333, "lat": 4.983333, "alt": None, "country": "Suriname", "continent_region": "South America", "iso_a3": "SUR"},
    "14100000": {"lon": -60.60944, "lat": -3.308333, "alt": 20, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "17050001": {"lon": -55.51111, "lat": -1.947222, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "15400000": {"lon": -63.92028, "lat": -8.736667, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
    "15275100": {"lon": -67.53496, "lat": -14.44091, "alt": 216, "country": "Bolivia", "continent_region": "South America", "iso_a3": "BOL"},
    "2604500124": {"lon": -51.88334, "lat": 3.816667, "alt": None, "country": "French Guiana", "continent_region": "South America", "iso_a3": "GUF"},
    "14420000": {"lon": -64.82889, "lat": -0.4819444, "alt": None, "country": "Brazil", "continent_region": "South America", "iso_a3": "BRA"},
}


class HYBAMProcessor:
    """Process HYBAM discharge and sediment data with QC and CF-1.8 output."""

    FLAG_GOOD = 0
    FLAG_ESTIMATED = 1
    FLAG_SUSPECT = 2
    FLAG_BAD = 3
    FLAG_MISSING = 9
    FLAG_MEANINGS = "good_data estimated_data suspect_data bad_data missing_data"

    def __init__(self, source_dir, output_dir, output_r_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_r_dir = Path(output_r_dir)
        self.station_metadata = {}

    @staticmethod
    def _to_float_array(values, fill_value=None):
        if np.ma.isMaskedArray(values):
            values = np.ma.filled(values, np.nan)
        arr = np.asarray(values, dtype=float)
        fill_candidates = [FILL_VALUE_FLOAT]
        if fill_value is not None:
            fill_candidates.append(fill_value)
        for candidate in fill_candidates:
            try:
                fill = float(candidate)
            except (TypeError, ValueError):
                continue
            arr[np.isclose(arr, fill, rtol=1e-5, atol=1e-5)] = np.nan
        return arr

    @staticmethod
    def _to_netcdf_values(values, fill_value):
        arr = np.asarray(values, dtype=float)
        return np.where(np.isfinite(arr), arr, fill_value)

    def find_station_dirs(self):
        return sorted([d for d in self.source_dir.iterdir() if d.is_dir() and '-' in d.name])

    def extract_station_info(self, station_dir):
        parts = station_dir.name.split('_-_')
        if len(parts) >= 3:
            return parts[2], parts[0].replace('_', ' '), parts[1].replace('_', ' ')
        return None, None, None

    def read_hybrid_metadata(self):
        for nc_file in self.output_dir.glob('HYBAM_*.nc'):
            try:
                with nc.Dataset(nc_file, 'r') as ds:
                    station_id = ds.getncattr('station_id')
                    self.station_metadata[station_id] = {
                        'station_name': ds.getncattr('station_name'),
                        'river_name': ds.getncattr('river_name'),
                        'nc_file': str(nc_file),
                    }
            except Exception:
                pass

    def find_data_files(self, station_dir, station_id):
        discharge_file = None
        ssc_file = None
        metadata = {'station_id': station_id, 'has_discharge': False, 'has_ssc': False}
        discharge_files = list(station_dir.glob('*_D_J1_*m3*s*.nc'))
        if discharge_files:
            discharge_file = discharge_files[0]
            metadata['has_discharge'] = True
        ssc_files = list(station_dir.glob('*_Q_*mg*l*.nc')) + list(station_dir.glob('*_IMsO_*mg*l*.nc'))
        if ssc_files:
            ssc_file = ssc_files[0]
            metadata['has_ssc'] = True
        return discharge_file, ssc_file, metadata

    def read_nc_data(self, nc_file):
        with nc.Dataset(nc_file, 'r') as ds:
            time_seconds = self._to_float_array(ds.variables['Date'][:])
            data_varname = None
            for var in ds.variables:
                if var not in ['Date', '_Origine', '_Qualité']:
                    data_varname = var
                    break
            if not data_varname:
                return None, None, None, None, None, None
            fill_value = getattr(ds.variables[data_varname], '_FillValue', FILL_VALUE_FLOAT)
            data_values = self._to_float_array(ds.variables[data_varname][:], fill_value)
            origine = ds.variables.get('_Origine', [None] * len(time_seconds))[:]
            qualite = ds.variables.get('_Qualité', [None] * len(time_seconds))[:]
            return time_seconds, data_values, data_varname, fill_value, origine, qualite

    def merge_discharge_ssc(self, discharge_file, ssc_file):
        """Aggregate raw Q/SSC independently to UTC days, then align daily dates.

        This replaces the prior nearest-SSC-to-each-Q-timestamp mapping, which
        could repeat one SSC observation across multiple sub-daily Q timestamps.
        """
        result = {
            'time': None, 'discharge': None, 'ssc': None,
            'discharge_origin': None, 'discharge_quality': None,
            'ssc_origin': None, 'ssc_quality': None,
            'time_coverage_start': None, 'time_coverage_end': None,
            'q_start': None, 'q_end': None, 'ssc_start': None, 'ssc_end': None,
        }
        q_raw_t = q_raw_v = None
        s_raw_t = s_raw_v = None
        q_daily_t = q_daily_v = np.asarray([], dtype=float)
        s_daily_t = s_daily_v = np.asarray([], dtype=float)

        if discharge_file:
            q_raw_t, q_raw_v, _, q_fill, _, _ = self.read_nc_data(discharge_file)
            result['discharge_raw'] = q_raw_v
            result['discharge_fill'] = q_fill
            q_daily_t, q_daily_v = aggregate_unix_series_to_daily(
                q_raw_t, q_raw_v, fill_values=(q_fill, FILL_VALUE_FLOAT))

        if ssc_file:
            s_raw_t, s_raw_v, _, s_fill, _, _ = self.read_nc_data(ssc_file)
            result['ssc_raw'] = s_raw_v
            result['ssc_fill'] = s_fill
            s_daily_t, s_daily_v = aggregate_unix_series_to_daily(
                s_raw_t, s_raw_v, fill_values=(s_fill, FILL_VALUE_FLOAT))

        if q_raw_t is not None and s_raw_t is not None:
            if q_daily_t.size and s_daily_t.size:
                overlap_start = max(float(np.min(q_daily_t)), float(np.min(s_daily_t)))
                overlap_end = min(float(np.max(q_daily_t)), float(np.max(s_daily_t)))
                if overlap_start <= overlap_end:
                    time_axis, aligned = align_daily_series(
                        {'discharge': (q_daily_t, q_daily_v), 'ssc': (s_daily_t, s_daily_v)},
                        start=overlap_start, end=overlap_end)
                    result['time'] = time_axis
                    result['discharge'] = aligned['discharge']
                    result['ssc'] = aligned['ssc']
                else:
                    result['time'] = np.asarray([], dtype=float)
                    result['discharge'] = np.asarray([], dtype=float)
                    result['ssc'] = np.asarray([], dtype=float)
            else:
                result['time'] = np.asarray([], dtype=float)
                result['discharge'] = np.asarray([], dtype=float)
                result['ssc'] = np.asarray([], dtype=float)
        elif q_raw_t is not None:
            result['time'] = q_daily_t
            result['discharge'] = q_daily_v
            result['ssc'] = np.full(q_daily_t.size, np.nan, dtype=float)
        elif s_raw_t is not None:
            result['time'] = s_daily_t
            result['ssc'] = s_daily_v
            result['discharge'] = np.full(s_daily_t.size, np.nan, dtype=float)

        if result['time'] is not None and len(result['time']) > 0:
            result['time_coverage_start'] = datetime.utcfromtimestamp(float(result['time'][0])).strftime('%Y-%m-%d')
            result['time_coverage_end'] = datetime.utcfromtimestamp(float(result['time'][-1])).strftime('%Y-%m-%d')

        before_q = 0 if q_raw_t is None else int(np.size(q_raw_t))
        before_s = 0 if s_raw_t is None else int(np.size(s_raw_t))
        if before_q > q_daily_t.size or before_s > s_daily_t.size:
            print('    Daily aggregation: Q {0}->{1} rows; SSC {2}->{3} rows.'.format(
                before_q, int(q_daily_t.size), before_s, int(s_daily_t.size)))
        return result

    def apply_qc_checks(self, data_dict):
        def _as_1d(x):
            if x is None:
                return None
            if np.ma.isMaskedArray(x):
                x = np.ma.filled(x, np.nan)
            arr = np.asarray(x)
            if arr.ndim == 0:
                return arr.reshape(1)
            arr = np.squeeze(arr)
            return arr.reshape(1) if arr.ndim == 0 else arr

        def _align_len(arr, n, fill=np.nan):
            if arr is None:
                return None
            a = _as_1d(arr)
            if a.size == n:
                return a.astype(float, copy=False)
            if a.size > n:
                return a[:n].astype(float, copy=False)
            out = np.full(n, fill, dtype=float)
            out[:a.size] = a.astype(float)
            return out

        time_sec = _as_1d(data_dict.get('time'))
        if time_sec is None or time_sec.size == 0:
            return None
        n = time_sec.size
        Q = _align_len(data_dict.get('discharge'), n, fill=np.nan)
        SSC = _align_len(data_dict.get('ssc'), n, fill=np.nan)
        SSL = np.full(n, np.nan, dtype=float)
        if Q is not None and SSC is not None:
            q_ok = np.isfinite(Q) & ~np.isclose(Q, FILL_VALUE_FLOAT, rtol=1e-5, atol=1e-5) & (Q >= 0)
            ssc_ok = np.isfinite(SSC) & ~np.isclose(SSC, FILL_VALUE_FLOAT, rtol=1e-5, atol=1e-5) & (SSC >= 0)
            m = q_ok & ssc_ok
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
        data_dict['time'] = qc['time']
        data_dict['discharge'] = qc['Q']
        data_dict['ssc'] = qc['SSC']
        data_dict['SSL'] = qc['SSL']
        for key in [
            'Q_flag', 'SSC_flag', 'SSL_flag',
            'Q_flag_qc1_physical', 'SSC_flag_qc1_physical', 'SSL_flag_qc1_physical',
            'Q_flag_qc2_log_iqr', 'SSC_flag_qc2_log_iqr', 'SSL_flag_qc2_log_iqr',
            'SSC_flag_qc3_ssc_q', 'SSL_flag_qc3_from_ssc_q',
        ]:
            data_dict[key] = qc[key]
        data_dict['ssc_q_bounds'] = qc.get('ssc_q_bounds')
        return data_dict

    def get_reference_info(self):
        return {
            'reference': "ORE-HYBAM: Observatoire de Recherche sur l'Environnement en Amazonie - Hydrologie et Géochimie du Bassin Amazonien. http://www.ore-hybam.org",
            'source_data_link': 'http://www.ore-hybam.org',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
        }

    def write_cf18_netcdf(self, station_id, station_name, river_name, latitude, longitude,
                          altitude, upstream_area, data_dict, output_file,
                          country='', continent_region='', iso_a3=''):
        time_days = data_dict['time'] / 86400.0
        fill_value = FILL_VALUE_FLOAT
        with nc.Dataset(output_file, 'w', format='NETCDF4', diskless=False) as ds:
            ds.createDimension('time', None)
            time_var = ds.createVariable('time', 'f8', ('time',), zlib=True)
            time_var.standard_name = 'time'; time_var.long_name = 'time'; time_var.units = 'days since 1970-01-01 00:00:00'; time_var.calendar = 'gregorian'; time_var.axis = 'T'; time_var[:] = time_days
            lat_var = ds.createVariable('lat', 'f4', zlib=True); lat_var.standard_name='latitude'; lat_var.long_name='station latitude'; lat_var.units='degrees_north'; lat_var.valid_range=np.array([-90.0,90.0],dtype='f4'); lat_var[:] = latitude
            lon_var = ds.createVariable('lon', 'f4', zlib=True); lon_var.standard_name='longitude'; lon_var.long_name='station longitude'; lon_var.units='degrees_east'; lon_var.valid_range=np.array([-180.0,180.0],dtype='f4'); lon_var[:] = longitude
            if altitude is not None:
                alt_var=ds.createVariable('altitude','f4',zlib=True,fill_value=FILL_VALUE_FLOAT); alt_var.standard_name='altitude'; alt_var.long_name='station elevation above sea level'; alt_var.units='m'; alt_var.positive='up'; alt_var[:] = altitude if np.isfinite(altitude) else FILL_VALUE_FLOAT
            if upstream_area is not None:
                area_var=ds.createVariable('upstream_area','f4',zlib=True,fill_value=FILL_VALUE_FLOAT); area_var.long_name='upstream drainage area'; area_var.units='km2'; area_var[:] = upstream_area if np.isfinite(upstream_area) else FILL_VALUE_FLOAT

            def _add_step_flag(name, values, flag_values, flag_meanings, long_name):
                v=ds.createVariable(name,'i1',('time',),zlib=True,complevel=4,fill_value=FILL_VALUE_INT); v.long_name=long_name; v.standard_name='status_flag'; v.flag_values=np.array(flag_values,dtype=np.int8); v.flag_meanings=flag_meanings; v.missing_value=np.int8(FILL_VALUE_INT); v[:] = np.asarray(values,dtype=np.int8)

            if data_dict['discharge'] is not None:
                q=ds.createVariable('Q','f4',('time',),zlib=True,complevel=4,fill_value=fill_value); q.standard_name='water_volume_transport_in_river_channel'; q.long_name='river discharge'; q.units='m3 s-1'; q.coordinates='time lat lon'; q.ancillary_variables='Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'; q.comment='Source: ORE-HYBAM; sub-daily values are consolidated to arithmetic UTC-day means before QC.'; q[:] = self._to_netcdf_values(data_dict['discharge'],fill_value)
                f=ds.createVariable('Q_flag','i1',('time',),zlib=True,complevel=4,fill_value=FILL_VALUE_INT); f.long_name='quality flag for river discharge'; f.standard_name='status_flag'; f.flag_values=np.array([0,1,2,3,9],dtype='i1'); f.flag_meanings=self.FLAG_MEANINGS; f[:] = data_dict['Q_flag']
                _add_step_flag('Q_flag_qc1_physical',data_dict['Q_flag_qc1_physical'],[0,3,9],'pass bad missing','QC1 physical flag for river discharge')
                _add_step_flag('Q_flag_qc2_log_iqr',data_dict['Q_flag_qc2_log_iqr'],[0,2,8,9],'pass suspect not_checked missing','QC2 log-IQR flag for river discharge')
            if data_dict['ssc'] is not None:
                s=ds.createVariable('SSC','f4',('time',),zlib=True,complevel=4,fill_value=fill_value); s.standard_name='mass_concentration_of_suspended_matter_in_water'; s.long_name='suspended sediment concentration'; s.units='mg L-1'; s.coordinates='time lat lon'; s.ancillary_variables='SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'; s.comment='Source: ORE-HYBAM; sub-daily values are consolidated to arithmetic UTC-day means before QC.'; s[:] = self._to_netcdf_values(data_dict['ssc'],fill_value)
                f=ds.createVariable('SSC_flag','i1',('time',),zlib=True,complevel=4,fill_value=FILL_VALUE_INT); f.long_name='quality flag for suspended sediment concentration'; f.standard_name='status_flag'; f.flag_values=np.array([0,1,2,3,9],dtype='i1'); f.flag_meanings=self.FLAG_MEANINGS; f[:] = data_dict['SSC_flag']
                _add_step_flag('SSC_flag_qc1_physical',data_dict['SSC_flag_qc1_physical'],[0,3,9],'pass bad missing','QC1 physical check flag for suspended sediment concentration')
                _add_step_flag('SSC_flag_qc2_log_iqr',data_dict['SSC_flag_qc2_log_iqr'],[0,2,8,9],'pass suspect not_checked missing','QC2 log-IQR flag for suspended sediment concentration')
                _add_step_flag('SSC_flag_qc3_ssc_q',data_dict['SSC_flag_qc3_ssc_q'],[0,2,8,9],'pass suspect not_checked missing','QC3 SSC-Q consistency flag for suspended sediment concentration')
            if 'SSL' in data_dict:
                l=ds.createVariable('SSL','f4',('time',),zlib=True,complevel=4,fill_value=fill_value); l.standard_name='suspended_sediment_transport_in_river'; l.long_name='suspended sediment load'; l.units='ton day-1'; l.coordinates='time lat lon'; l.ancillary_variables='SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q'; l.comment='Calculated after daily aggregation: SSL = Q_daily x SSC_daily x 0.0864.'; l[:] = self._to_netcdf_values(data_dict['SSL'],fill_value)
                f=ds.createVariable('SSL_flag','i1',('time',),zlib=True,complevel=4,fill_value=FILL_VALUE_INT); f.long_name='quality flag for suspended sediment load'; f.standard_name='status_flag'; f.flag_values=np.array([0,1,2,3,9],dtype='i1'); f.flag_meanings=self.FLAG_MEANINGS; f[:] = data_dict['SSL_flag']
                _add_step_flag('SSL_flag_qc1_physical',data_dict['SSL_flag_qc1_physical'],[0,3,9],'pass bad missing','QC1 physical check flag for suspended sediment load')
                _add_step_flag('SSL_flag_qc2_log_iqr',data_dict['SSL_flag_qc2_log_iqr'],[0,2,8,9],'pass suspect not_checked missing','QC2 log-IQR flag for suspended sediment load')
                _add_step_flag('SSL_flag_qc3_from_ssc_q',data_dict['SSL_flag_qc3_from_ssc_q'],[0,2,8,9],'not_propagated propagated not_checked missing','QC3 flag propagated from SSC-Q inconsistency')

            ds.Conventions='CF-1.8, ACDD-1.3'; ds.title='Harmonized Global River Discharge and Sediment'; ds.summary=f'River discharge and suspended sediment data for {station_name} station on the {river_name} River from ORE-HYBAM. Raw sub-daily observations are consolidated to daily values before quality control and SSL calculation.'; ds.source='In-situ station data'; ds.data_source_name='HYBAM Dataset'; ds.station_name=station_name; ds.river_name=river_name; ds.Source_ID=station_id
            ds.geospatial_lat_min=float(latitude); ds.geospatial_lat_max=float(latitude); ds.geospatial_lon_min=float(longitude); ds.geospatial_lon_max=float(longitude)
            if altitude is not None: ds.geospatial_vertical_min=float(altitude) if np.isfinite(altitude) else FILL_VALUE_FLOAT; ds.geospatial_vertical_max=float(altitude) if np.isfinite(altitude) else FILL_VALUE_FLOAT
            ds.geographic_coverage='Amazon Basin'
            if data_dict['time_coverage_start']:
                ds.time_coverage_start=data_dict['time_coverage_start']; ds.time_coverage_end=data_dict['time_coverage_end']; ds.temporal_span=f"{data_dict['time_coverage_start'][:4]}-{data_dict['time_coverage_end'][:4]}"
            ds.temporal_resolution='daily'
            vars_provided=[]
            if data_dict['discharge'] is not None: vars_provided.append('Q')
            if data_dict['ssc'] is not None: vars_provided.append('SSC')
            if 'SSL' in data_dict: vars_provided.append('SSL')
            if altitude is not None: vars_provided.append('altitude')
            if upstream_area is not None: vars_provided.append('upstream_area')
            ds.variables_provided=', '.join(vars_provided)
            ref=self.get_reference_info(); ds.reference=ref['reference']; ds.source_data_link=ref['source_data_link']; ds.creator_name=ref['creator_name']; ds.creator_email=ref['creator_email']; ds.creator_institution=ref['creator_institution']; ds.history=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Consolidated raw Q and SSC independently to UTC-day means, aligned daily dates, applied QC, calculated daily SSL, and standardized to CF-1.8 format with hybam_comprehensive_processor.py."; ds.date_created=datetime.now().strftime('%Y-%m-%d'); ds.date_modified=datetime.now().strftime('%Y-%m-%d'); ds.processing_level='Quality controlled and standardized'; ds.number_of_data=str(int(np.sum(np.isfinite(data_dict['discharge']))) if data_dict['discharge'] is not None else 0); ds.location_id=station_id; ds.country=country; ds.continent_region=continent_region; ds.iso_a3=iso_a3

    def process_station(self, station_dir):
        station_id, station_name, river_name = self.extract_station_info(station_dir)
        if not station_id:
            print(f"  ✗ Could not extract station info from {station_dir.name}")
            return False
        print(f"\n  Processing: {station_id} ({station_name} / {river_name})")
        discharge_file, ssc_file, _ = self.find_data_files(station_dir, station_id)
        if not discharge_file and not ssc_file:
            print("    ✗ No discharge or SSC data found")
            return False
        print(f"    ✓ Discharge: {discharge_file.name if discharge_file else 'N/A'}")
        print(f"    ✓ SSC: {ssc_file.name if ssc_file else 'N/A'}")
        data = self.merge_discharge_ssc(discharge_file, ssc_file)
        if data['time'] is None or len(data['time']) == 0:
            print("    ✗ No time data extracted")
            return False
        print(f"    ✓ Time range: {data['time_coverage_start']} to {data['time_coverage_end']} ({len(data['time'])} days)")
        data['station_id']=station_id; data['station_name']=station_name; data['river_name']=river_name
        data=self.apply_qc_checks(data)
        if data is None or data.get('time') is None or len(data['time']) == 0:
            print("    ✗ No valid data after QC (all missing)")
            return False
        info=STATION_INFO.get(station_id)
        if info:
            latitude=info['lat']; longitude=info['lon']; altitude=info['alt']; country=info.get('country',''); continent_region=info.get('continent_region',''); iso_a3=info.get('iso_a3','')
        else:
            latitude=longitude=altitude=FILL_VALUE_FLOAT; country=continent_region=iso_a3=''
        upstream_area=FILL_VALUE_FLOAT
        output_file=self.output_r_dir / f'HYBAM_{station_id}.nc'; output_file.parent.mkdir(parents=True, exist_ok=True)
        self.write_cf18_netcdf(station_id,station_name,river_name,latitude,longitude,altitude,upstream_area,data,output_file,country=country,continent_region=continent_region,iso_a3=iso_a3)
        return {'station_id':station_id,'station_name':station_name,'river_name':river_name,'latitude':latitude,'longitude':longitude,'altitude':altitude,'upstream_area':upstream_area,'time_coverage_start':data['time_coverage_start'],'time_coverage_end':data['time_coverage_end'],'data':data}

    def generate_csv_summary(self, stations_data, output_file):
        fieldnames=['station_name','Source_ID','river_name','longitude','latitude','altitude','upstream_area','Data Source Name','Type','Temporal Resolution','Temporal Span','Variables Provided','Geographic Coverage','Reference/DOI','Q_start_date','Q_end_date','Q_percent_complete','SSC_start_date','SSC_end_date','SSC_percent_complete','SSL_start_date','SSL_end_date','SSL_percent_complete']
        rows=[]
        for station in stations_data:
            data=station['data']
            def pct(flag):
                if flag is None: return 0
                good=np.sum(flag==self.FLAG_GOOD); total=np.sum(flag!=self.FLAG_MISSING)
                return good/total*100 if total>0 else 0
            q_pct=pct(data.get('Q_flag')); ssc_pct=pct(data.get('SSC_flag')); ssl_pct=pct(data.get('SSL_flag'))
            vars_prov=[]
            if data['discharge'] is not None: vars_prov.append('Q')
            if data['ssc'] is not None: vars_prov.append('SSC')
            if 'SSL' in data: vars_prov.append('SSL')
            rows.append({'station_name':station['station_name'],'Source_ID':station['station_id'],'river_name':station['river_name'],'longitude':station['longitude'],'latitude':station['latitude'],'altitude':station['altitude'],'upstream_area':station['upstream_area'],'Data Source Name':'HYBAM Dataset','Type':'In-situ','Temporal Resolution':'daily','Temporal Span':f"{data['time_coverage_start'][:4]}-{data['time_coverage_end'][:4]}" if data['time_coverage_start'] else 'N/A','Variables Provided':', '.join(vars_prov),'Geographic Coverage':'Amazon Basin','Reference/DOI':'http://www.ore-hybam.org','Q_start_date':data['time_coverage_start'] if data['discharge'] is not None else '','Q_end_date':data['time_coverage_end'] if data['discharge'] is not None else '','Q_percent_complete':f'{q_pct:.1f}' if data['discharge'] is not None else '','SSC_start_date':data['time_coverage_start'] if data['ssc'] is not None else '','SSC_end_date':data['time_coverage_end'] if data['ssc'] is not None else '','SSC_percent_complete':f'{ssc_pct:.1f}' if data['ssc'] is not None else '','SSL_start_date':data['time_coverage_start'] if 'SSL' in data else '','SSL_end_date':data['time_coverage_end'] if 'SSL' in data else '','SSL_percent_complete':f'{ssl_pct:.1f}' if 'SSL' in data else ''})
        with open(output_file,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
        print(f"\n  ✓ CSV summary written: {output_file.name}")

    def run(self):
        print('='*70); print('HYBAM Comprehensive Processing Pipeline'); print('='*70)
        station_dirs=self.find_station_dirs(); print(f"\nFound {len(station_dirs)} station directories")
        successful_stations=[]; failed_stations=[]
        for i, station_dir in enumerate(station_dirs,1):
            print(f"\n[{i}/{len(station_dirs)}]",end='')
            try:
                result=self.process_station(station_dir)
                if result: successful_stations.append(result)
                else: failed_stations.append(station_dir.name)
            except Exception as e:
                print(f"    ✗ Error: {e}"); failed_stations.append(station_dir.name)
        if successful_stations:
            generate_csv_summary_tool(successful_stations,self.output_r_dir/'HYBAM_station_summary.csv')
            qc_rows=[]
            for s in successful_stations:
                data=s.get('data',{}); row={'station_name':s.get('station_name',''),'Source_ID':s.get('station_id',''),'river_name':s.get('river_name',''),'longitude':s.get('longitude',''),'latitude':s.get('latitude',''),'QC_n_days':len(data.get('time',[]))}
                def cnt(arr,v):
                    a=np.asarray(arr) if arr is not None else np.asarray([]); return int(np.sum(a==np.int8(v)))
                for var in ['Q','SSC','SSL']:
                    f=data.get(f'{var}_flag'); row[f'{var}_final_good']=cnt(f,0); row[f'{var}_final_estimated']=cnt(f,1); row[f'{var}_final_suspect']=cnt(f,2); row[f'{var}_final_bad']=cnt(f,3); row[f'{var}_final_missing']=cnt(f,9)
                    f=data.get(f'{var}_flag_qc1_physical'); row[f'{var}_qc1_pass']=cnt(f,0); row[f'{var}_qc1_bad']=cnt(f,3); row[f'{var}_qc1_missing']=cnt(f,9)
                    f=data.get(f'{var}_flag_qc2_log_iqr'); row[f'{var}_qc2_pass']=cnt(f,0); row[f'{var}_qc2_suspect']=cnt(f,2); row[f'{var}_qc2_not_checked']=cnt(f,8); row[f'{var}_qc2_missing']=cnt(f,9)
                f3=data.get('SSC_flag_qc3_ssc_q'); row['SSC_qc3_pass']=cnt(f3,0); row['SSC_qc3_suspect']=cnt(f3,2); row['SSC_qc3_not_checked']=cnt(f3,8); row['SSC_qc3_missing']=cnt(f3,9)
                f3=data.get('SSL_flag_qc3_from_ssc_q'); row['SSL_qc3_not_propagated']=cnt(f3,0); row['SSL_qc3_propagated']=cnt(f3,2); row['SSL_qc3_not_checked']=cnt(f3,8); row['SSL_qc3_missing']=cnt(f3,9); qc_rows.append(row)
            generate_qc_results_csv_tool(qc_rows,self.output_r_dir/'HYBAM_qc_results.csv')
        print('\n'+'='*70); print('Processing Complete!'); print(f'Successfully processed: {len(successful_stations)} stations'); print(f'Failed: {len(failed_stations)} stations'); print(f'\nOutput directory: {self.output_r_dir}'); print('='*70)


def main():
    source_root=resolve_source_root(__file__)
    output_root=resolve_output_root(__file__,create=True)
    source_dir=source_root/'HYBAM'/'source'
    output_dir=output_root/'daily'/'HYBAM'/'Output'
    output_r_dir=output_root/'daily'/'HYBAM'/'qc'
    HYBAMProcessor(source_dir,output_dir,output_r_dir).run()


if __name__ == '__main__':
    main()
