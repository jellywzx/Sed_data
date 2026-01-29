#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs quality control, CF-1.8 standardization, and metadata
enhancement for the Myanmar (Irrawaddy and Salween Rivers) sediment and
discharge dataset.

Original data source:
Baronas, J.J.; Tipper, E.T.; Bickle, M.J.; Stevenson, E.I.; Hilton, R.G. (2020).
Flow velocity, discharge, and suspended sediment compositions of the Irrawaddy
and Salween Rivers, 2017-2019. NERC Environmental Information Data Centre.
https://doi.org/10.5285/86f17d61-141f-4500-9aa5-26a82aef0b33

The script performs the following steps:
1.  Reads the source CSV files for discharge and sediment samples.
2.  Merges discharge (Q) and sediment concentration (SSC) data by date for
    each station (cross-section).
3.  Calculates suspended sediment load (SSL).
4.  Performs Quality Control (QC) on Q, SSC, and SSL, generating CF-compliant
    quality flags.
5.  Writes CF-1.8 compliant NetCDF files for each station, containing only
    time points with valid data.
6.  Generates a summary CSV file with metadata and statistics for all stations.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import warnings
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
)

# --- CONFIGURATION ---

# Input and Output directories from user request
BASE_DIR = os.path.abspath(os.path.join(PARENT_DIR, '..')) 
SOURCE_DATA_DIR = os.path.join(BASE_DIR, "Source/Myanmar")
TARGET_NC_DIR = os.path.join(BASE_DIR, "Output_r/daily/Myanmar/qc")
TARGET_CSV_PATH = os.path.join(BASE_DIR, "Output_r/daily/Myanmar/qc")

# --- HELPER FUNCTIONS ---

def calculate_ssl(q, ssc):
    """
    Calculate Suspended Sediment Load (SSL) from discharge (Q) and concentration (SSC).

    Formula: SSL (ton/day) = Q (m³/s) * SSC (mg/L) * 0.0864
    Derivation of the coefficient 0.0864:
    - SSC (mg/L) is equivalent to SSC (g/m³)
    - Q (m³/s) * SSC (g/m³) = Load (g/s)
    - To convert g/s to ton/day:
      (Load g/s) * (86400 s/day) / (1,000,000 g/ton) = Load (ton/day)
      Coefficient = 86400 / 1,000,000 = 0.0864
    """
    if pd.isna(q) or pd.isna(ssc):
        return np.nan
    return q * ssc * 0.0864

def apply_tool_qc(time, Q, SSC, SSL, station_id, station_name, plot_dir=None):
    """
    Apply QC using tool.py end-to-end pipeline WITH step-level provenance flags.
    Also robust to scalar / mismatched shapes.
    """

    # --- force 1D & align length (avoid len() of scalar / 0d array) ---
    time = np.atleast_1d(np.asarray(time)).reshape(-1)
    Q    = np.atleast_1d(np.asarray(Q, dtype=float)).reshape(-1)
    SSC  = np.atleast_1d(np.asarray(SSC, dtype=float)).reshape(-1)
    SSL  = np.atleast_1d(np.asarray(SSL, dtype=float)).reshape(-1)

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None
    time, Q, SSC, SSL = time[:n], Q[:n], SSC[:n], SSL[:n]

    # --- tool.py pipeline: QC1/QC2/QC3 + provenance ---
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )
    if qc is None:
        return None

    # --- valid-time logic: value-based "present" ---
    def _present(v, f):
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int8)
        return (
            (f != FILL_VALUE_INT)  # final flag 不是 missing
            & np.isfinite(v)
            & (~np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5))
        )

    present_Q   = _present(qc["Q"],   qc["Q_flag"])
    present_SSC = _present(qc["SSC"], qc["SSC_flag"])
    present_SSL = _present(qc["SSL"], qc["SSL_flag"])

    valid_time = np.atleast_1d(present_Q | present_SSC | present_SSL)
    if not np.any(valid_time):
        return None

    # trim ALL arrays incl. step flags
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and qc[k].shape[0] == valid_time.shape[0]:
            qc[k] = qc[k][valid_time]

    # optional plot (if tool returned bounds; 若 qc 里有 ssc_q_bounds 你也可用)
    # 这里先保持你原逻辑：只要 plot_dir 不 None，就画 final 的诊断图（可选）
    # 如果你想严格复用 tool 的 envelope/bounds，可以再加一层判断。

    return qc



def get_summary_stats(df, var_name):
    """Calculate summary statistics for a variable."""
    flag_name = f"{var_name}_flag"
    valid_data = df[(df[flag_name] == 0) & (df[var_name].notna())]
    
    if valid_data.empty:
        return np.nan, np.nan, 0.0

    start_date = valid_data['time'].min()
    end_date = valid_data['time'].max()
    
    # For non-padded data, total_days is the span from first to last measurement
    total_days = (df['time'].max() - df['time'].min()).days + 1
    good_data_count = len(valid_data)
    percent_complete = (good_data_count / total_days) * 100 if total_days > 0 else 0
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), round(percent_complete, 2)

def create_netcdf_file(filepath, df, station_meta):
    """Create a CF-1.8 compliant NetCDF file."""
    
    start_date_obj = df['time'].iloc[0]
    time_units = f"days since {start_date_obj.strftime('%Y-%m-%d')} 00:00:00"

    with nc.Dataset(filepath, 'w', format='NETCDF4') as ds:
        # === DIMENSIONS ===
        ds.createDimension('time', None)

        # === COORDINATE VARIABLES ===
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = time_units
        time_var.calendar = "gregorian"
        # ---- time handling (SAFE) ----
        # Convert series to list of python datetimes to avoid Series.to_pydatetime()/dt.to_pydatetime issues
        time_series = pd.to_datetime(df['time'])
        time_py = [t.to_pydatetime() for t in time_series.tolist()]

        time_var[:] = nc.date2num(time_py, units=time_units, calendar="gregorian")


        lat_var = ds.createVariable('lat', 'f4')
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = station_meta['latitude']

        lon_var = ds.createVariable('lon', 'f4')
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = station_meta['longitude']

        # === GLOBAL ATTRIBUTES ===
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.dataset_name = "Myanmar (Irrawaddy and Salween Rivers)"
        ds.station_name = station_meta['station_name']
        ds.river_name = station_meta['river_name']
        ds.Source_ID = station_meta['Source_ID']
        ds.source_url = "https://doi.org/10.5285/86f17d61-141f-4500-9aa5-26a82aef0b33"
        ds.reference = "Baronas, J.J.; Tipper, E.T.; Bickle, M.J.; Stevenson, E.I.; Hilton, R.G. (2020). Flow velocity, discharge, and suspended sediment compositions of the Irrawaddy and Salween Rivers, 2017-2019. NERC Environmental Information Data Centre. (Dataset)."
        ds.summary = "This dataset provides in-situ daily time series of river discharge and sediment transport for the Irrawaddy and Salween Rivers in Myanmar, harmonized and quality-controlled."
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"
        ds.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using convert_to_netcdf.py. Data is not padded; only timestamps with measurements are included."
        ds.Type = "In-situ station data"
        ds.Temporal_Resolution = "daily"
        ds.Temporal_Span = f"{df['time'].min().strftime('%Y-%m-%d')} to {df['time'].max().strftime('%Y-%m-%d')}"
        ds.Geographic_Coverage = "Irrawaddy and Salween Rivers, Myanmar"
        ds.Variables_Provided = "altitude, upstream_area, Q, SSC, SSL, station_name, river_name, Source_ID"
        ds.Number_of_data = 1
        
        # === DATA VARIABLES ===
        fill_value = -9999.0
        flag_fill_value = np.int8(-127)

        # altitude and upstream_area (not available)
        # altitude
        alt_var = ds.createVariable('altitude', 'f4', fill_value=fill_value)
        alt_var.long_name = 'station altitude'
        alt_var.units = 'm'
        alt_var.missing_value = fill_value
        alt_var[:] = fill_value

        # upstream_area
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=fill_value)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.missing_value = fill_value
        area_var[:] = fill_value

        # Q
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=fill_value)
        q_var.long_name = "River Discharge"
        q_var.standard_name = "river_discharge"
        q_var.units = "m3 s-1"
        q_var.ancillary_variables = "Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr"
        q_var.ancillary_variables = "Q_flag"
        q_var.comment = "Source: Original data from Baronas et al. (2020)."
        q_var[:] = df['Q'].fillna(fill_value).values

        # SSC
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=fill_value)
        ssc_var.long_name = "Suspended Sediment Concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water_body"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q"
        ssc_var.comment = "Source: Original data from Baronas et al. (2020)."
        ssc_var[:] = df['SSC'].fillna(fill_value).values

        # SSL
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=fill_value)
        ssl_var.long_name = "Suspended Sediment Load"
        ssl_var.units = "ton day-1"
        ssl_var.coordinates = "lat lon altitude"
        ssl_var.ancillary_variables = "SSL_flag SSL_flag_qc1_physical SSL_flag_qc3_from_ssc_q"
        ssl_var.comment = "Source: Calculated. Formula: SSL = Q * SSC * 0.0864."
        ssl_var[:] = df['SSL'].fillna(fill_value).values
        # === STEP/PROVENANCE FLAG VARIABLES ===
        def _add_step_flag(ds, name, values, *, flag_values, flag_meanings, long_name):
            v = ds.createVariable(name, 'b', ('time',), fill_value=flag_fill_value)
            v.long_name = long_name
            v.standard_name = 'status_flag'
            v.flag_values = np.array(flag_values, dtype='b')
            v.flag_meanings = flag_meanings
            v.missing_value = np.int8(9)
            v[:] = np.asarray(values, dtype=np.int8)
            return v

        # QC1 physical: 0 pass, 3 bad, 9 missing
        if "Q_flag_qc1_physical" in df.columns:
            _add_step_flag(ds, "Q_flag_qc1_physical", df["Q_flag_qc1_physical"].fillna(9).values,
                          flag_values=[0, 3, 9],
                          flag_meanings="pass bad missing",
                          long_name="QC1 physical flag for river discharge")

        if "SSC_flag_qc1_physical" in df.columns:
            _add_step_flag(ds, "SSC_flag_qc1_physical", df["SSC_flag_qc1_physical"].fillna(9).values,
                          flag_values=[0, 3, 9],
                          flag_meanings="pass bad missing",
                          long_name="QC1 physical flag for suspended sediment concentration")

        if "SSL_flag_qc1_physical" in df.columns:
            _add_step_flag(ds, "SSL_flag_qc1_physical", df["SSL_flag_qc1_physical"].fillna(9).values,
                          flag_values=[0, 3, 9],
                          flag_meanings="pass bad missing",
                          long_name="QC1 physical flag for suspended sediment load")

        # QC2 log-IQR: 0 pass, 2 suspect, 8 not_checked, 9 missing
        if "Q_flag_qc2_log_iqr" in df.columns:
            _add_step_flag(ds, "Q_flag_qc2_log_iqr", df["Q_flag_qc2_log_iqr"].fillna(9).values,
                          flag_values=[0, 2, 8, 9],
                          flag_meanings="pass suspect not_checked missing",
                          long_name="QC2 log-IQR flag for river discharge")

        if "SSC_flag_qc2_log_iqr" in df.columns:
            _add_step_flag(ds, "SSC_flag_qc2_log_iqr", df["SSC_flag_qc2_log_iqr"].fillna(9).values,
                          flag_values=[0, 2, 8, 9],
                          flag_meanings="pass suspect not_checked missing",
                          long_name="QC2 log-IQR flag for suspended sediment concentration")

        # QC3 SSC–Q (for SSC): 0 pass, 2 suspect, 8 not_checked, 9 missing
        if "SSC_flag_qc3_ssc_q" in df.columns:
            _add_step_flag(ds, "SSC_flag_qc3_ssc_q", df["SSC_flag_qc3_ssc_q"].fillna(9).values,
                          flag_values=[0, 2, 8, 9],
                          flag_meanings="pass suspect not_checked missing",
                          long_name="QC3 SSC–Q consistency flag for SSC")

        # QC3 propagation to SSL: 0 not_propagated, 2 propagated, 8 not_checked, 9 missing
        if "SSL_flag_qc3_from_ssc_q" in df.columns:
            _add_step_flag(ds, "SSL_flag_qc3_from_ssc_q", df["SSL_flag_qc3_from_ssc_q"].fillna(9).values,
                          flag_values=[0, 2, 8, 9],
                          flag_meanings="not_propagated propagated not_checked missing",
                          long_name="QC3 propagated flag to SSL from SSC–Q inconsistency")


        # === FLAG VARIABLES ===
        # Q_flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=flag_fill_value)
        q_flag_var.long_name = "Quality flag for River Discharge"
        q_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        q_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
        q_flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source."
        q_flag_var[:] = df['Q_flag'].fillna(flag_fill_value).values

        # SSC_flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=flag_fill_value)
        ssc_flag_var.long_name = "Quality flag for Suspended Sediment Concentration"
        ssc_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        ssc_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = df['SSC_flag'].fillna(flag_fill_value).values

        # SSL_flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=flag_fill_value)
        ssl_flag_var.long_name = "Quality flag for Suspended Sediment Load"
        ssl_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        ssl_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
        ssl_flag_var.comment = "Flag definitions: 0=Good, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source."
        ssl_flag_var[:] = df['SSL_flag'].fillna(flag_fill_value).values

def main():
    """Main processing function"""
    print("Starting Myanmar dataset processing (unpadded)...")
    os.makedirs(TARGET_NC_DIR, exist_ok=True)

    # Read source files
    try:
        q_df = pd.read_csv(os.path.join(SOURCE_DATA_DIR, 'Q_crosssections.csv'))
        samples_df = pd.read_csv(os.path.join(SOURCE_DATA_DIR, 'samples_data.csv'))
    except FileNotFoundError as e:
        warnings.warn(f"Source file not found: {e}. Aborting.")
        return

    # Prepare dataframes
    q_df['Date'] = pd.to_datetime(q_df['Date'], format='%d/%m/%Y')
    samples_df['Date'] = pd.to_datetime(samples_df['Sample_time'], format='%d/%m/%Y %H:%M:%S', errors='coerce').dt.date
    samples_df['Date'] = pd.to_datetime(samples_df['Date'])
    
    # Group samples by date and site to get daily average SSC
    daily_ssc = samples_df.groupby(['Site_season', 'Date'])['SSC_mgL'].mean().reset_index()

    # Use Crosssection_ID as the station identifier
    stations = q_df['Crosssection_ID'].unique()
    station_summaries = []

    for station_id in stations:
        print(f"Processing station: {station_id}...")
        
        station_q_data = q_df[q_df['Crosssection_ID'] == station_id].copy()
        station_ssc_data = daily_ssc[daily_ssc['Site_season'] == station_id].copy()

        # Merge Q and SSC data on the date
        merged_df = pd.merge(station_q_data[['Date', 'Discharge_m3_per_s', 'Latitude', 'Longitude']],
                             station_ssc_data[['Date', 'SSC_mgL']],
                             on='Date', how='outer')
        
        merged_df = merged_df.rename(columns={'Discharge_m3_per_s': 'Q', 'SSC_mgL': 'SSC', 'Date': 'time'})

        if merged_df.empty or (merged_df['Q'].isna().all() and merged_df['SSC'].isna().all()):
            warnings.warn(f"No data for station {station_id}. Skipping.")
            continue

        # Calculate SSL
        merged_df['SSL'] = merged_df.apply(lambda row: calculate_ssl(row['Q'], row['SSC']), axis=1)

        # Apply QC
        qc = apply_tool_qc(
            time=merged_df['time'].values,
            Q=merged_df['Q'].values,
            SSC=merged_df['SSC'].values,
            SSL=merged_df['SSL'].values,
            station_id=station_id,
            station_name=station_id.replace('_', ' '),
            plot_dir=os.path.join(TARGET_NC_DIR, "diagnostic_plots"),
        )

        if qc is None:
            warnings.warn(f"No valid data for station {station_id} after QC. Skipping.")
            continue

        qc_df = pd.DataFrame(qc)


        # --- MODIFIED LOGIC: NO PADDING ---
        # Filter to only rows that have at least one valid data point
        final_df = qc_df.dropna(subset=['Q', 'SSC', 'SSL'], how='all').copy()
        final_df.sort_values(by='time', inplace=True)
        # also pad step/provenance flags (NaN -> 9)
        for col in final_df.columns:
            if col.endswith("_flag") or ("flag_qc" in col):
                final_df[col] = final_df[col].fillna(9).astype(np.int8)

        if final_df.empty:
            warnings.warn(f"No valid data for station {station_id} after QC. Skipping.")
            continue
        # --- 打印每个站点经过质量控制后的 flag 情况（仅打印，不写入文件） ---
        print(f"  QC flags summary for station: {station_id}")
        for flag_col in ['Q_flag', 'SSC_flag', 'SSL_flag']:
            if flag_col in final_df.columns:
                vals, counts = np.unique(final_df[flag_col].values, return_counts=True)
                counts_str = ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(vals, counts)])
                print(f"    {flag_col}: {counts_str}")
            else:
                print(f"    {flag_col}: (missing)")
        # 打印前 5 行示例（time 与 flags）
        sample_cols = [c for c in ['time', 'Q_flag', 'SSC_flag', 'SSL_flag'] if c in final_df.columns]
        if sample_cols:
            print(final_df[sample_cols].head(5).to_string(index=False))
        print("")
        # --- END OF MODIFICATION ---

        # Get metadata for this station
        lat = station_q_data['Latitude'].mean()
        lon = station_q_data['Longitude'].mean()
        river_name = 'Irrawaddy' if 'IRR' in station_id else 'Salween' if 'SAL' in station_id else 'Unknown'
        
        station_meta = {
            'Source_ID': station_id,
            'station_name': station_id.replace('_', ' '),
            'river_name': river_name,
            'latitude': lat,
            'longitude': lon,
        }

        # Create NetCDF
        output_filename = f"Myanmar_{station_id}.nc"
        output_filepath = os.path.join(TARGET_NC_DIR, output_filename)
        create_netcdf_file(output_filepath, final_df, station_meta)
        print(f"  Successfully created {output_filepath}")

        # Collect summary stats
        q_start, q_end, q_perc = get_summary_stats(final_df, 'Q')
        ssc_start, ssc_end, ssc_perc = get_summary_stats(final_df, 'SSC')
        ssl_start, ssl_end, ssl_perc = get_summary_stats(final_df, 'SSL')

        summary = {
            'Source_ID': station_id,
            'station_name': station_meta['station_name'],
            'river_name': river_name,
            'longitude': lon,
            'latitude': lat,
            'altitude': np.nan,
            'upstream_area': np.nan,
            'Q_start_date': q_start, 'Q_end_date': q_end, 'Q_percent_complete': q_perc,
            'SSC_start_date': ssc_start, 'SSC_end_date': ssc_end, 'SSC_percent_complete': ssc_perc,
            'SSL_start_date': ssl_start, 'SSL_end_date': ssl_end, 'SSL_percent_complete': ssl_perc,
            'Data Source Name': 'Myanmar (Irrawaddy and Salween Rivers)',
            'Type': 'In-situ',
            'Temporal Resolution': 'daily',
            'Temporal Span': f"{final_df['time'].min().strftime('%Y-%m-%d')} to {final_df['time'].max().strftime('%Y-%m-%d')}" if not final_df.empty else 'N/A',
            'Variables Provided': 'Q, SSC, SSL',
            'Geographic Coverage': 'Irrawaddy and Salween Rivers, Myanmar',
            'Reference/DOI': 'https://doi.org/10.5285/86f17d61-141f-4500-9aa5-26a82aef0b33'
        }
        station_summaries.append(summary)

    # Generate summary CSV
    if station_summaries:
        csv_station = os.path.join(TARGET_CSV_PATH, "Myanmar_station_summary.csv")
        csv_qc = os.path.join(TARGET_CSV_PATH, "Myanmar_qc_results_summary.csv")

        generate_csv_summary_tool(station_summaries, csv_station)
        generate_qc_results_csv_tool(station_summaries, csv_qc)

        print(f"\n✓ Generated CSV summary: {csv_station}")
        print(f"✓ Generated QC results : {csv_qc}")


if __name__ == "__main__":
    main()
