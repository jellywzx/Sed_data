#!/usr/bin/env python3
"""
HYDAT dataset comprehensive quality control and CF-1.8 standardization script.

Features:
1. Data validation & flagging (QC1 physical, QC2 log-IQR, QC3 SSC-Q consistency)
2. CF-1.8 compliant metadata
3. Physical plausibility checks
4. Time truncation and invalid-station removal
5. Data provenance tracking

Stations with SSC or SSL data are retained; discharge/Q is NOT required.

Author: Zhongwang Wei
Date: 2025-10-26
Modified: 2026-08-09 - Q is now optional; SSC/SSL-only stations supported
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.metadata import check_variable_metadata_tiered
from code.output import (
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
    generate_warning_summary_csv as generate_warning_summary_csv_tool,
    summarize_warning_types as summarize_warning_types_tool,
)
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_hydro_qc_with_provenance,
    apply_quality_flag,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.runtime import resolve_output_root
from code.units import convert_ssl_units_if_needed
from code.validation import check_nc_completeness


def _has_real_data(values, flag, fill_float=FILL_VALUE_FLOAT, fill_int=FILL_VALUE_INT):
    """Return True if any element in *values* passes the present check."""
    v = np.asarray(values, dtype=float)
    f = np.asarray(flag, dtype=np.int8)
    return bool(np.any(
        (f != fill_int)
        & np.isfinite(v)
        & (~np.isclose(v, float(fill_float), rtol=1e-5, atol=1e-5))
    ))


def apply_tool_qc(
    time,
    Q,
    SSC,
    SSL,
    station_id,
    station_name,
    plot_dir=None,
    q_present=True,
):
    """
    Apply QC using the end-to-end pipeline WITH step-level provenance flags.
    Also fixes valid-time logic using value-based missing detection.

    Q, SSC, SSL may be all-missing arrays when the source file lacks that
    variable.  The QC still runs on whatever is present.

    q_present : bool
        Whether Q has real data.  When False, SSL is treated as independent
        (source-reported) rather than derived from Q*SSC.
    """

    # When Q is absent, SSL cannot be derived from Q*SSC.
    ssl_is_independent = not q_present
    ssl_derived_from_q_ssc = q_present

    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=ssl_is_independent,
        ssl_is_derived_from_q_ssc=ssl_derived_from_q_ssc,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )

    if qc is None:
        return None

    # Fix valid-time logic: filter out time steps where ALL variables are missing
    def _present(v, f):
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int8)
        return (
            (f != FILL_VALUE_INT)
            & np.isfinite(v)
            & (~np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5))
        )

    present_Q   = _present(qc["Q"], qc["Q_flag"])
    present_SSC = _present(qc["SSC"], qc["SSC_flag"])
    present_SSL = _present(qc["SSL"], qc["SSL_flag"])

    valid_time = present_Q | present_SSC | present_SSL
    if not np.any(valid_time):
        return None

    # Trim to valid time
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and len(qc[k]) == len(valid_time):
            qc[k] = qc[k][valid_time]

    # Diagnostic plot (optional)
    if plot_dir is not None and qc.get("ssc_q_bounds") is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_ssc_q_diagnostic(
            time=pd.to_datetime(qc["time"], unit="D", origin="1970-01-01"),
            Q=qc["Q"],
            SSC=qc["SSC"],
            Q_flag=qc["Q_flag"],
            SSC_flag=qc["SSC_flag"],
            ssc_q_bounds=qc["ssc_q_bounds"],
            station_id=station_id,
            station_name=station_name,
            out_png=plot_dir / f"{station_id}_ssc_q.png",
        )

    return qc


class HYDATQualityControl:
    """HYDAT batch quality control and CF-1.8 standardization."""

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_stations': 0,
            'processed_stations': 0,
            'removed_stations': 0,
            'stations_info': [],
            # New audit fields
            'with_discharge': 0,
            'without_discharge': 0,
            'ssc_only': 0,
            'ssl_only': 0,
            'both_ssc_ssl': 0,
            'q_only_skipped': 0,
            'no_sediment_skipped': 0,
        }

    def _count_flags(self, f):
        f = np.asarray(f, dtype=np.int8)
        return {
            "good": int(np.sum(f == 0)),
            "estimated": int(np.sum(f == 1)),
            "suspect": int(np.sum(f == 2)),
            "bad": int(np.sum(f == 3)),
            "missing": int(np.sum(f == FILL_VALUE_INT)),
        }

    def calculate_completeness(self, data_array, flag_array, start_date, end_date):
        """Calculate data completeness (percentage of good data)."""
        total_days = (end_date - start_date).days + 1
        good_data_count = np.sum(flag_array == 0)
        if total_days > 0:
            return (good_data_count / total_days) * 100.0
        return 0.0

    def _count_final_flags(self, f):
        f = np.asarray(f, dtype=np.int8)
        return {
            "good": int(np.sum(f == 0)),
            "estimated": int(np.sum(f == 1)),
            "suspect": int(np.sum(f == 2)),
            "bad": int(np.sum(f == 3)),
            "missing": int(np.sum(f == FILL_VALUE_INT)),
        }

    def _count_step_flags(self, f, mapping: dict):
        f = np.asarray(f, dtype=np.int8)
        out = {}
        for name, val in mapping.items():
            out[name] = int(np.sum(f == np.int8(val)))
        return out

    def process_station(self, input_file):
        """Process a single station file.

        Station eligibility: must have SSC OR SSL data (Q is optional).
        """
        print(f"Processing station: {input_file.name}")

        try:
            with nc.Dataset(input_file, 'r') as ds_in:
                # --- Basic info ---
                station_id = ds_in.station_id if hasattr(ds_in, 'station_id') else ''
                station_name = ds_in.station_name if hasattr(ds_in, 'station_name') else ''
                province = ds_in.province_territory if hasattr(ds_in, 'province_territory') else ''

                # Check for discharge availability (from provenance attribute)
                has_discharge_src = getattr(ds_in, 'has_discharge', 'true') != 'false'

                # --- Coordinates ---
                if 'latitude' in ds_in.variables:
                    lat = float(ds_in.variables['latitude'][:])
                elif 'lat' in ds_in.variables:
                    lat = float(ds_in.variables['lat'][:])
                else:
                    raise ValueError("Cannot find latitude variable")

                if 'longitude' in ds_in.variables:
                    lon = float(ds_in.variables['longitude'][:])
                elif 'lon' in ds_in.variables:
                    lon = float(ds_in.variables['lon'][:])
                else:
                    raise ValueError("Cannot find longitude variable")

                # --- Scalars ---
                altitude = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else -9999.0
                upstream_area = float(ds_in.variables['upstream_area'][:]) if 'upstream_area' in ds_in.variables else -9999.0

                # --- Time ---
                time = ds_in.variables['time'][:]
                n_time = len(time)

                # --- Q (discharge) -- OPTIONAL ---
                Q_present = False
                if 'discharge' in ds_in.variables:
                    Q = ds_in.variables['discharge'][:]
                    Q_present = _has_real_data(Q, np.zeros(n_time, dtype=np.int8))
                elif 'Q' in ds_in.variables:
                    Q = ds_in.variables['Q'][:]
                    Q_present = _has_real_data(Q, np.zeros(n_time, dtype=np.int8))
                else:
                    Q = np.full(n_time, FILL_VALUE_FLOAT, dtype=np.float32)

                # --- SSC -- OPTIONAL (but need SSC or SSL) ---
                SSC_present = False
                if 'ssc' in ds_in.variables:
                    SSC = ds_in.variables['ssc'][:]
                    SSC_present = _has_real_data(SSC, np.zeros(n_time, dtype=np.int8))
                elif 'SSC' in ds_in.variables:
                    SSC = ds_in.variables['SSC'][:]
                    SSC_present = _has_real_data(SSC, np.zeros(n_time, dtype=np.int8))
                else:
                    SSC = np.full(n_time, FILL_VALUE_FLOAT, dtype=np.float32)

                # --- SSL -- OPTIONAL (but need SSC or SSL) ---
                SSL_present = False
                if 'sediment_load' in ds_in.variables:
                    SSL = ds_in.variables['sediment_load'][:]
                    SSL_present = _has_real_data(SSL, np.zeros(n_time, dtype=np.int8))
                elif 'SSL' in ds_in.variables:
                    SSL = ds_in.variables['SSL'][:]
                    SSL_present = _has_real_data(SSL, np.zeros(n_time, dtype=np.int8))
                else:
                    SSL = np.full(n_time, FILL_VALUE_FLOAT, dtype=np.float32)

                # --- Station eligibility: SSC OR SSL required ---
                if not SSC_present and not SSL_present:
                    print(f"  + No SSC or SSL data, skip station {station_id}")
                    return False, None

                # --- Run QC pipeline ---
                qc = apply_tool_qc(
                    time=time,
                    Q=Q,
                    SSC=SSC,
                    SSL=SSL,
                    station_id=station_id,
                    station_name=station_name,
                    plot_dir=self.output_dir / "diagnostic_plots",
                    q_present=Q_present,
                )

                if qc is None:
                    print(f"  + No valid data after QC, skip station {station_id}")
                    return False, None

                time = qc["time"]
                Q = qc["Q"]
                SSC = qc["SSC"]
                SSL = qc["SSL"]
                Q_flag = qc["Q_flag"]
                SSC_flag = qc["SSC_flag"]
                SSL_flag = qc["SSL_flag"]

                # --- Post-QC sediment check: must still have SSC or SSL ---
                has_ssc_after = _has_real_data(SSC, SSC_flag)
                has_ssl_after = _has_real_data(SSL, SSL_flag)
                if not has_ssc_after and not has_ssl_after:
                    print(f"  + SSC/SSL all removed by QC, skip station {station_id}")
                    return False, None

                # Re-determine what's actually present after QC
                Q_present = _has_real_data(Q, Q_flag)

                # --- Time range ---
                reference_date = pd.Timestamp('1970-01-01')
                start_date = reference_date + pd.Timedelta(days=float(time[0]))
                end_date = reference_date + pd.Timedelta(days=float(time[-1]))

                # --- Completeness ---
                Q_completeness = self.calculate_completeness(Q, Q_flag, start_date, end_date) if Q_present else 0.0
                SSC_completeness = self.calculate_completeness(SSC, SSC_flag, start_date, end_date) if has_ssc_after else 0.0
                SSL_completeness = self.calculate_completeness(SSL, SSL_flag, start_date, end_date) if has_ssl_after else 0.0

                # --- Write output ---
                output_file = self.output_dir / f"HYDAT_{station_id}.nc"

                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    ds_out.createDimension('time', len(time))

                    # Time
                    var_time = ds_out.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.units = 'days since 1970-01-01 00:00:00'
                    var_time.calendar = 'gregorian'
                    var_time.axis = 'T'
                    var_time[:] = time

                    # Lat
                    var_lat = ds_out.createVariable('lat', 'f4')
                    var_lat.standard_name = 'latitude'
                    var_lat.long_name = 'station latitude'
                    var_lat.units = 'degrees_north'
                    var_lat.axis = 'Y'
                    var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
                    var_lat[:] = lat

                    # Lon
                    var_lon = ds_out.createVariable('lon', 'f4')
                    var_lon.standard_name = 'longitude'
                    var_lon.long_name = 'station longitude'
                    var_lon.units = 'degrees_east'
                    var_lon.axis = 'X'
                    var_lon.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
                    var_lon[:] = lon

                    # Altitude
                    var_alt = ds_out.createVariable('altitude', 'f4', fill_value=-9999.0)
                    var_alt.standard_name = 'altitude'
                    var_alt.long_name = 'station elevation above sea level'
                    var_alt.units = 'm'
                    var_alt.positive = 'up'
                    var_alt.comment = 'Source: HYDAT database.'
                    var_alt[:] = altitude

                    # Upstream area
                    var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=-9999.0)
                    var_area.long_name = 'upstream drainage area'
                    var_area.units = 'km2'
                    var_area.comment = 'Source: HYDAT database.'
                    var_area[:] = upstream_area

                    # --- Q variable ---
                    var_Q = ds_out.createVariable('Q', 'f4', ('time',),
                                                   fill_value=-9999.0, zlib=True, complevel=4)
                    var_Q.standard_name = 'water_volume_transport_in_river_channel'
                    var_Q.long_name = 'river discharge'
                    var_Q.units = 'm3 s-1'
                    var_Q.coordinates = 'time lat lon'
                    var_Q.ancillary_variables = 'Q_flag'
                    var_Q.comment = 'Source: Original data from HYDAT database.'
                    var_Q[:] = Q

                    var_Q_flag = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
                    var_Q_flag.long_name = 'quality flag for river discharge'
                    var_Q_flag.standard_name = 'status_flag'
                    var_Q_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_Q_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_Q_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_Q_flag[:] = Q_flag

                    # --- SSC variable ---
                    var_SSC = ds_out.createVariable('SSC', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSC.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                    var_SSC.long_name = 'suspended sediment concentration'
                    var_SSC.units = 'mg L-1'
                    var_SSC.coordinates = 'time lat lon'
                    var_SSC.ancillary_variables = 'SSC_flag'
                    var_SSC.comment = 'Source: Original data from HYDAT database.'
                    var_SSC[:] = SSC

                    var_SSC_flag = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
                    var_SSC_flag.long_name = 'quality flag for suspended sediment concentration'
                    var_SSC_flag.standard_name = 'status_flag'
                    var_SSC_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSC_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSC_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSC_flag[:] = SSC_flag

                    # --- SSL variable ---
                    var_SSL = ds_out.createVariable('SSL', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSL.long_name = 'suspended sediment load'
                    var_SSL.units = 'ton day-1'
                    var_SSL.coordinates = 'time lat lon'
                    var_SSL.ancillary_variables = 'SSL_flag'
                    var_SSL.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m+/s) x SSC (mg/L) x 86.4, where 86.4 = 86400 s/day x 10^-6 ton/mg x 1000 L/m3.'
                    var_SSL[:] = SSL

                    var_SSL_flag = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
                    var_SSL_flag.long_name = 'quality flag for suspended sediment load'
                    var_SSL_flag.standard_name = 'status_flag'
                    var_SSL_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSL_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSL_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSL_flag[:] = SSL_flag

                    # --- Step-level QC provenance flags ---
                    def _add_step_flag(name, arr, *, flag_values, flag_meanings, long_name):
                        if arr is None:
                            return
                        v = ds_out.createVariable(name, 'i1', ('time',), fill_value=FILL_VALUE_INT, zlib=True, complevel=4)
                        v.long_name = long_name
                        v.standard_name = 'status_flag'
                        v.flag_values = np.array(flag_values, dtype=np.int8)
                        v.flag_meanings = flag_meanings
                        v.missing_value = np.int8(FILL_VALUE_INT)
                        v[:] = np.asarray(arr, dtype=np.int8)

                    # Q steps
                    _add_step_flag('Q_flag_qc1_physical', qc.get('Q_flag_qc1_physical'),
                        flag_values=[0, 3, 9], flag_meanings='pass bad missing',
                        long_name='QC1 physical flag for river discharge')
                    _add_step_flag('Q_flag_qc2_log_iqr', qc.get('Q_flag_qc2_log_iqr'),
                        flag_values=[0, 2, 8, 9], flag_meanings='pass suspect not_checked missing',
                        long_name='QC2 log-IQR flag for river discharge')

                    # SSC steps
                    _add_step_flag('SSC_flag_qc1_physical', qc.get('SSC_flag_qc1_physical'),
                        flag_values=[0, 3, 9], flag_meanings='pass bad missing',
                        long_name='QC1 physical flag for suspended sediment concentration')
                    _add_step_flag('SSC_flag_qc2_log_iqr', qc.get('SSC_flag_qc2_log_iqr'),
                        flag_values=[0, 2, 8, 9], flag_meanings='pass suspect not_checked missing',
                        long_name='QC2 log-IQR flag for suspended sediment concentration')
                    _add_step_flag('SSC_flag_qc3_ssc_q', qc.get('SSC_flag_qc3_ssc_q'),
                        flag_values=[0, 2, 8, 9], flag_meanings='pass suspect not_checked missing',
                        long_name='QC3 SSC-Q consistency flag for suspended sediment concentration')

                    # SSL steps
                    _add_step_flag('SSL_flag_qc1_physical', qc.get('SSL_flag_qc1_physical'),
                        flag_values=[0, 3, 9], flag_meanings='pass bad missing',
                        long_name='QC1 physical flag for suspended sediment load')
                    _add_step_flag('SSL_flag_qc2_log_iqr', qc.get('SSL_flag_qc2_log_iqr'),
                        flag_values=[0, 2, 8, 9], flag_meanings='pass suspect not_checked missing',
                        long_name='QC2 log-IQR flag for suspended sediment load')
                    _add_step_flag('SSL_flag_qc3_from_ssc_q', qc.get('SSL_flag_qc3_from_ssc_q'),
                        flag_values=[0, 2, 8, 9], flag_meanings='not_propagated propagated not_checked missing',
                        long_name='QC3 propagation flag for suspended sediment load')

                    # Update ancillary_variables
                    var_Q.ancillary_variables = 'Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'
                    var_SSC.ancillary_variables = 'SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'
                    var_SSL.ancillary_variables = 'SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q'

                    # --- Dynamic variables_provided ---
                    vars_list = ['altitude', 'upstream_area']
                    if Q_present:
                        vars_list.append('Q')
                    if has_ssc_after:
                        vars_list.append('SSC')
                    if has_ssl_after:
                        vars_list.append('SSL')
                    variables_provided_str = ', '.join(vars_list)

                    # --- Global attributes ---
                    ds_out.Conventions = 'CF-1.8, ACDD-1.3'
                    ds_out.title = 'Harmonized Global River Discharge and Sediment'
                    ds_out.summary = (
                        f'River discharge and suspended sediment data for station {station_name} '
                        f'(ID: {station_id}) from the HYDAT database (Water Survey of Canada). '
                        f'This dataset contains daily observations with quality control flags.'
                    )
                    ds_out.source = 'In-situ station data'
                    ds_out.data_source_name = 'HYDAT Dataset'
                    ds_out.station_name = station_name
                    river_name = station_name.split(' AT ')[0] if ' AT ' in station_name else station_name.split(' NEAR ')[0] if ' NEAR ' in station_name else ''
                    ds_out.river_name = river_name
                    ds_out.location_id = station_id
                    ds_out.type = 'In-situ station data'
                    ds_out.temporal_resolution = 'daily'
                    ds_out.temporal_span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    ds_out.variables_provided = variables_provided_str
                    ds_out.geographic_coverage = f"{province}, Canada"
                    ds_out.country = 'Canada'
                    ds_out.continent_region = 'North America'
                    ds_out.time_coverage_start = start_date.strftime('%Y-%m-%d')
                    ds_out.time_coverage_end = end_date.strftime('%Y-%m-%d')
                    ds_out.number_of_data = '1'
                    ds_out.reference = 'HYDAT - Canadian Hydrometric Database, Water Survey of Canada'
                    ds_out.source_data_link = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
                    ds_out.creator_name = 'Zhongwang Wei'
                    ds_out.creator_email = 'weizhw6@mail.sysu.edu.cn'
                    ds_out.creator_institution = 'Sun Yat-sen University, China'
                    ds_out.geospatial_lat_min = lat
                    ds_out.geospatial_lat_max = lat
                    ds_out.geospatial_lon_min = lon
                    ds_out.geospatial_lon_max = lon

                    history_entry = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"Converted from HYDAT database to CF-1.8 compliant NetCDF format. "
                        f"Applied quality control checks including physical constraint validation "
                        f"(Q range check, SSC range check, SSL negative check). "
                        f"Trimmed data to valid time range from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}. "
                        f"Script: process_hydat_cf18.py"
                    )
                    ds_out.history = history_entry
                    ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
                    ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
                    ds_out.processing_level = 'Quality controlled and standardized'
                    ds_out.comment = (
                        "Quality flags: 0=good, 1=estimated (derived), 2=suspect, 3=bad, 9=missing. "
                        "QC1: physical feasibility; QC2: log-IQR screening (independent variables only); "
                        "QC3: SSC-Q consistency and propagation to derived SSL."
                    )

                # --- NetCDF completeness check ---
                errors, warnings = check_nc_completeness(output_file, strict=False)
                var_errs, var_warns = check_variable_metadata_tiered(output_file, tier="recommended")
                errors.extend(var_errs)
                warnings.extend(var_warns)

                if errors:
                    print(f"  + NetCDF completeness check FAILED for {station_id}")
                    for e in errors:
                        print(f"    ERROR: {e}")
                    try:
                        output_file.unlink()
                        print(f"    -> Invalid NetCDF removed: {output_file.name}")
                    except Exception:
                        pass
                    return False, None

                if warnings:
                    print(f"  + NetCDF completeness warnings for {station_id}: {len(warnings)}")
                    for w in warnings:
                        print(f"    WARNING: {w}")
                    with nc.Dataset(output_file, "a") as ds_out:
                        ds_out.history = (
                            ds_out.history
                            + f"; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            + f"Completeness check warnings ({len(warnings)}): "
                            + "; ".join(warnings[:3])
                        )

                station_warnings = warnings.copy() if warnings else []

                # --- Station info for CSV ---
                station_info = {
                    'station_name': station_name,
                    'Source_ID': station_id,
                    'river_name': river_name,
                    'longitude': lon,
                    'latitude': lat,
                    'altitude': altitude if altitude != -9999.0 else np.nan,
                    'upstream_area': upstream_area if upstream_area != -9999.0 else np.nan,
                    'Data Source Name': 'HYDAT Dataset',
                    'Type': 'In-situ',
                    'Temporal Resolution': 'daily',
                    'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'Variables Provided': variables_provided_str,
                    'Geographic Coverage': f"{province}, Canada",
                    'Reference/DOI': 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html',
                    'Q_start_date': start_date.year if Q_present else np.nan,
                    'Q_end_date': end_date.year if Q_present else np.nan,
                    'Q_percent_complete': round(Q_completeness, 2),
                    'SSC_start_date': start_date.year if has_ssc_after else np.nan,
                    'SSC_end_date': end_date.year if has_ssc_after else np.nan,
                    'SSC_percent_complete': round(SSC_completeness, 2),
                    'SSL_start_date': start_date.year if has_ssl_after else np.nan,
                    'SSL_end_date': end_date.year if has_ssl_after else np.nan,
                    'SSL_percent_complete': round(SSL_completeness, 2),
                    # New provenance fields
                    'has_discharge': has_discharge_src,
                    'ssc_only': has_ssc_after and not has_ssl_after,
                    'ssl_only': has_ssl_after and not has_ssc_after,
                }

                # --- QC stats ---
                station_info["QC_n_days"] = int(len(time))

                q_cnt   = self._count_final_flags(Q_flag)
                ssc_cnt = self._count_final_flags(SSC_flag)
                ssl_cnt = self._count_final_flags(SSL_flag)

                station_info.update({
                    "Q_final_good": q_cnt["good"],
                    "Q_final_estimated": q_cnt["estimated"],
                    "Q_final_suspect": q_cnt["suspect"],
                    "Q_final_bad": q_cnt["bad"],
                    "Q_final_missing": q_cnt["missing"],
                    "SSC_final_good": ssc_cnt["good"],
                    "SSC_final_estimated": ssc_cnt["estimated"],
                    "SSC_final_suspect": ssc_cnt["suspect"],
                    "SSC_final_bad": ssc_cnt["bad"],
                    "SSC_final_missing": ssc_cnt["missing"],
                    "SSL_final_good": ssl_cnt["good"],
                    "SSL_final_estimated": ssl_cnt["estimated"],
                    "SSL_final_suspect": ssl_cnt["suspect"],
                    "SSL_final_bad": ssl_cnt["bad"],
                    "SSL_final_missing": ssl_cnt["missing"],
                })

                # Step flags
                qc1_map = {"pass": 0, "bad": 3, "missing": 9}
                qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}
                qc3_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}
                qc3_ssl_map = {"not_propagated": 0, "propagated": 2, "not_checked": 8, "missing": 9}

                for key, qc_key, mapping in [
                    ("Q_flag_qc1_physical", "Q_flag_qc1_physical", qc1_map),
                    ("SSC_flag_qc1_physical", "SSC_flag_qc1_physical", qc1_map),
                    ("SSL_flag_qc1_physical", "SSL_flag_qc1_physical", qc1_map),
                    ("Q_flag_qc2_log_iqr", "Q_flag_qc2_log_iqr", qc2_map),
                    ("SSC_flag_qc2_log_iqr", "SSC_flag_qc2_log_iqr", qc2_map),
                    ("SSL_flag_qc2_log_iqr", "SSL_flag_qc2_log_iqr", qc2_map),
                    ("SSC_flag_qc3_ssc_q", "SSC_flag_qc3_ssc_q", qc3_map),
                    ("SSL_flag_qc3_from_ssc_q", "SSL_flag_qc3_from_ssc_q", qc3_ssl_map),
                ]:
                    if qc_key in qc:
                        c = self._count_step_flags(qc[qc_key], mapping)
                        prefix = qc_key.replace("_flag_", "_").replace("flag_", "")
                        station_info.update({f"{prefix}_{k}": v for k, v in c.items()})

                station_info.update({
                    "n_warnings": len(station_warnings),
                    "warnings": " | ".join(station_warnings[:5])
                })

                self.stats['processed_stations'] += 1
                print(f"  + Successfully processed")
                print(f"    Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                print(f"    Completeness: Q={Q_completeness:.1f}%, SSC={SSC_completeness:.1f}%, SSL={SSL_completeness:.1f}%")

                return True, station_info

        except Exception as e:
            print(f"  + Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None

    def process_all_stations(self):
        """Process all stations in parallel."""

        print(f"\n{'='*80}")
        print(f"HYDAT Dataset Quality Control and CF-1.8 Standardization (parallel)")
        print(f"{'='*80}\n")

        input_files = sorted(self.input_dir.glob('HYDAT_*_SEDIMENT.nc'))
        self.stats['total_stations'] = len(input_files)

        print(f"Found {len(input_files)} station files")
        print(f"Using CPU cores: {os.cpu_count()} parallel processing\n")
        print(f"Input directory:  {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")

        results = []

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_station = {executor.submit(self.process_station, f): f for f in input_files}
            for future in as_completed(future_to_station):
                success, station_info = future.result()
                if success and station_info:
                    results.append(station_info)

        # --- Update statistics ---
        self.stats['processed_stations'] = len(results)
        self.stats['removed_stations'] = self.stats['total_stations'] - len(results)
        self.stats['stations_info'] = results

        # Count subtypes
        self.stats['with_discharge'] = sum(1 for r in results if r.get('has_discharge', True))
        self.stats['without_discharge'] = sum(1 for r in results if not r.get('has_discharge', True))
        self.stats['ssc_only'] = sum(1 for r in results if r.get('ssc_only', False))
        self.stats['ssl_only'] = sum(1 for r in results if r.get('ssl_only', False))
        self.stats['both_ssc_ssl'] = sum(
            1 for r in results
            if not r.get('ssc_only', False) and not r.get('ssl_only', False)
        )

        print(f"\n{'='*80}")
        print(f"Processing complete! (parallel)")
        print(f"{'='*80}")
        print(f"  Total stations:            {self.stats['total_stations']}")
        print(f"  Successfully processed:    {self.stats['processed_stations']}")
        print(f"  Removed:                   {self.stats['removed_stations']}")
        print(f"  ---")
        print(f"  With discharge:            {self.stats['with_discharge']}")
        print(f"  Without discharge:         {self.stats['without_discharge']}")
        print(f"  SSC-only stations:         {self.stats['ssc_only']}")
        print(f"  SSL-only stations:         {self.stats['ssl_only']}")
        print(f"  Both SSC+SSL stations:     {self.stats['both_ssc_ssl']}")
        print(f"{'='*80}\n")

        return self.stats

    def summarize_warning_types(self):
        return summarize_warning_types_tool(self.stats['stations_info'])

    def generate_csv_summary(self, output_csv):
        generate_csv_summary_tool(self.stats['stations_info'], output_csv)

    def generate_qc_results_csv(self, output_csv):
        generate_qc_results_csv_tool(self.stats['stations_info'], output_csv)


def main():
    """Main entry point."""
    output_root = resolve_output_root(start=__file__, create=True)
    input_dir = output_root / "daily" / "HYDAT" / "sediment_update"
    output_dir = output_root / "daily" / "HYDAT" / "qc"

    csv_file = output_dir / 'HYDAT_station_summary.csv'

    qc = HYDATQualityControl(input_dir, output_dir)
    qc_csv = output_dir / "HYDAT_qc_results_summary.csv"

    stats = qc.process_all_stations()

    qc.generate_csv_summary(csv_file)
    qc.generate_qc_results_csv(qc_csv)

    print(f"\n+ All done!")
    print(f"  Output directory: {output_dir}")
    print(f"  CSV summary:      {csv_file}")


if __name__ == '__main__':
    main()
