#!/usr/bin/env python3
"""
Enhanced processing script for Fukushima Niida River dataset (DOI 10.34355/CRiED.U.Tsukuba.00147)
Includes:
1. Quality Control (QC) checking and flagging
2. CF-1.8 and ACDD-1.3 metadata compliance
3. Data provenance tracking
4. Unit conversion verification
"""

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings
import sys
# warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    propagate_ssc_q_inconsistency_to_ssl
)

warnings.filterwarnings('ignore')


def read_doi00147_data(filepath):
    """
    Read DOI00147 Excel data file and extract all sheets.
    
    Parameters:
    -----------
    filepath : str
        Path to Excel file
    
    Returns:
    --------
    all_data : dict
        Dictionary with station names as keys and dataframes as values
    """
    # Read all sheets
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    
    # Dictionary to store data by station
    station_data = {}
    
    for sheet_name, df in all_sheets.items():
        print(f"Processing sheet: {sheet_name}")
        
        # Skip first 2 rows (header info) and read data
        # Column mapping based on format description:
        # 0: DOI, 1: DID, 2: station, 3-7: yyyy,mm,dd,hh,min, 8: xyear
        # 9: LatDir, 10: Nsflag, 11: xlat, 12: LonDir, 13: Ewflag, 14: xlong
        # 15: altdepflag, 16: sampdep, 17: sample type
        # 18: Water discharge (m3/s), 19: SSC (g/L), 20: Uncertainty SSC (g/L)
        
        data = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=2, header=None)
        
        # Skip the first row which contains units
        data = data.iloc[1:].reset_index(drop=True)
        
        # Extract station name
        station_name = data.iloc[0, 2]  # Column 2 is station
        
        # Create datetime from components
        data['datetime'] = pd.to_datetime({
            'year': data[3].astype(int),
            'month': data[4].astype(int),
            'day': data[5].astype(int),
            'hour': data[6].astype(int),
            'minute': data[7].astype(int)
        })
        
        # Extract relevant columns
        df_clean = pd.DataFrame({
            'datetime': data['datetime'],
            'latitude': data[11].astype(float),
            'longitude': data[14].astype(float),
            'depth': data[16].astype(float),
            'discharge': data[18].astype(float),  # m3/s
            'ssc': data[19].astype(float),  # g/L
            'ssc_uncertainty': data[20].astype(float)  # g/L
        })
        
        # Convert SSC from g/L to mg/L (multiply by 1000)
        df_clean['ssc_mg_L'] = df_clean['ssc'] * 1000
        df_clean['ssc_uncertainty_mg_L'] = df_clean['ssc_uncertainty'] * 1000
        
        # Calculate sediment load (ton/day)
        # Formula derivation:
        # Load = Q (m³/s) × SSC (g/L) × 86.4
        # = Q (m³/s) × SSC (g/L) × 86400 (s/day) / 1000 (g/kg) / 1000 (kg/ton)
        # Unit verification:
        # m³/s × g/L × 86400 s/day = m³ × g × 86400 / (s × L × s) = m³ × g × 86400 / L
        # = 1000 L × g × 86400 / L = 86,400,000 g/day
        # = 86.4 tons/day (since 1 ton = 1,000,000 g)
        df_clean['sediment_load'] = df_clean['discharge'] * df_clean['ssc'] * 86.4
        
        # Add to station data
        if station_name not in station_data:
            station_data[station_name] = []
        station_data[station_name].append(df_clean)
    
    # Combine all data for each station
    combined_data = {}
    for station, data_list in station_data.items():
        combined = pd.concat(data_list, ignore_index=True)
        combined = combined.sort_values('datetime').reset_index(drop=True)
        combined_data[station] = combined
    
    return combined_data


def aggregate_to_daily(df):
    """
    Aggregate high-frequency data to daily averages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index and measurements
    
    Returns:
    --------
    daily_df : pd.DataFrame
        Daily averaged data
    """
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Resample to daily, taking mean
    daily = df.resample('D').mean()
    
    # Recalculate sediment load from daily averages
    daily['sediment_load'] = daily['discharge'] * daily['ssc'] * 86.4
    
    return daily.reset_index()

def log_station_qc(station_name, source_id, n_samples,
                   q_value, ssc_value, ssl_value,
                   q_flag, ssc_flag, ssl_flag,
                   skipped_log_iqr, skipped_ssc_q,
                   created_path):
    print(f"\nProcessing: {station_name} ({source_id})")
    if skipped_log_iqr:
        print(f"  [{station_name} ({source_id})] Sample size = {n_samples} < 5, log-IQR statistical QC skipped.")
    if skipped_ssc_q:
        print(f"  [{station_name} ({source_id})] Sample size = {n_samples} < 5, SSC-Q consistency check and diagnostic plot skipped.")

    print(f"✓ Created: {created_path}")
    print(f"  Q: {q_value:.2f} m3/s (flag={int(q_flag)})")
    print(f"  SSC: {ssc_value:.2f} mg/L (flag={int(ssc_flag)})")
    print(f"  SSL: {ssl_value:.2f} ton/day (flag={int(ssl_flag)})")


def perform_qc_checks(daily_df):
    """
    Perform quality control checks and create flag variables.
    
    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily data with measurements
    
    Returns:
    --------
    qc_df : pd.DataFrame
        Data with added flag columns
    """
    qc_df = daily_df.copy()
    
    # -------------------------
    # 1. Basic physical QC
    # -------------------------
    qc_df["Q_flag"] = qc_df["discharge"].apply(
        lambda x: apply_quality_flag(x, "Q")
    )
    qc_df["SSC_flag"] = qc_df["ssc_mg_L"].apply(
        lambda x: apply_quality_flag(x, "SSC")
    )
    qc_df["SSL_flag"] = qc_df["sediment_load"].apply(
        lambda x: apply_quality_flag(x, "SSL")
    )

    # -------------------------
    # 2. SSL log-IQR screening
    # -------------------------
    lower, upper = compute_log_iqr_bounds(qc_df["sediment_load"].values)

    if lower is not None:
        is_outlier = (
            (qc_df["sediment_load"] < lower) |
            (qc_df["sediment_load"] > upper)
        ) & (qc_df["SSL_flag"] == 0)

        qc_df.loc[is_outlier, "SSL_flag"] = 2  # suspect

    # -------------------------
    # 3. SSC–Q consistency check
    # -------------------------
    ssc_q_bounds = build_ssc_q_envelope(
        Q_m3s=qc_df["discharge"].values,
        SSC_mgL=qc_df["ssc_mg_L"].values,
        k=1.5,
        min_samples=5
    )

    for i, row in qc_df.iterrows():
        is_bad, _ = check_ssc_q_consistency(
            Q=row["discharge"],
            SSC=row["ssc_mg_L"],
            Q_flag=row["Q_flag"],
            SSC_flag=row["SSC_flag"],
            ssc_q_bounds=ssc_q_bounds
        )
        if is_bad:
            ssc_q_inconsistent = True

            # 先把 SSC_flag 改为 suspect（仅当原来是 good）
            if qc_df.at[i, "SSC_flag"] == 0:
                qc_df.at[i, "SSC_flag"] = np.int8(2)  # suspect

            # 再把不一致性传播到 SSL_flag
            qc_df.at[i, "SSL_flag"] = propagate_ssc_q_inconsistency_to_ssl(
                inconsistent=ssc_q_inconsistent,
                Q=row["discharge"],
                SSC=row["ssc_mg_L"],
                SSL=row["sediment_load"],
                Q_flag=qc_df.at[i, "Q_flag"],
                SSC_flag=qc_df.at[i, "SSC_flag"],
                SSL_flag=qc_df.at[i, "SSL_flag"],
                ssl_is_derived_from_q_ssc=True,
            )

    return qc_df, ssc_q_bounds


def create_netcdf_cf18(filepath, data, station_name, river_name, source_id=None):
    """
    Create CF-1.8 and ACDD-1.3 compliant NetCDF file with quality flags.
    
    Parameters:
    -----------
    filepath : str
        Output NetCDF filename
    data : pd.DataFrame
        Time series data with QC flags
    station_name : str
        Station name
    river_name : str
        River name
    source_id : str
        Source identifier (e.g., 'DOI00147_Haramachi')
    """
    # Get coordinates from first row (they're constant)
    lat = data['latitude'].iloc[0]
    lon = data['longitude'].iloc[0]
    depth = data['depth'].iloc[0]
    
    # Create NetCDF file
    dataset = nc.Dataset(filepath, 'w', format='NETCDF4')
    
    # Create unlimited time dimension
    time_dim = dataset.createDimension('time', None)  # UNLIMITED dimension
    
    # Create coordinate variables
    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.units = f'days since {data["datetime"].min().strftime("%Y-%m-%d")} 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'
    
    # Convert dates to days since epoch
    ref = pd.to_datetime(data["datetime"].min().date())

    time_var.units = f"days since {ref.strftime('%Y-%m-%d')} 00:00:00"
    time_var.calendar = "gregorian"

    time_var[:] = (data["datetime"].dt.floor("D") - ref).dt.days.values


    
    # Create scalar coordinate variables
    lat_var = dataset.createVariable('lat', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = lat
    
    lon_var = dataset.createVariable('lon', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = lon
    
    # alt_var = dataset.createVariable('altitude', 'f4')
    # alt_var.standard_name = 'altitude'
    # alt_var.long_name = 'station elevation above sea level'
    # alt_var.units = 'm'
    # alt_var.positive = 'up'
    # alt_var._FillValue = -9999.0
    # alt_var.comment = f'Sampling depth: {depth} m below surface. Negative values indicate below water surface.'
    # alt_var[:] = -depth  # Negative for below surface
    
    # area_var = dataset.createVariable('upstream_area', 'f4')
    # area_var.long_name = 'upstream drainage area'
    # area_var.standard_name = 'upstream_area'
    # area_var.units = 'km2'
    # # area_var._FillValue = -9999.0
    # area_var.comment = 'Not available in source data'
    # area_var[:] = -9999.0
    
    # Create data variables with quality flags
    # Q (discharge)
    Q_var = dataset.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    Q_var.standard_name = 'water_volume_transport_in_river_channel'
    Q_var.long_name = 'river discharge'
    Q_var.units = 'm3 s-1'
    Q_var.coordinates = 'lat lon'
    Q_var.ancillary_variables = 'Q_flag'
    Q_var.comment = f'Source: Original data provided by Feng et al. (2022, DOI:10.34355/CRiED.U.Tsukuba.00147). Unit: m³/s (cubic meters per second).'
    Q_var[:] = data['discharge'].fillna(-9999.0).values
    
    # Q_flag
    Q_flag_var = dataset.createVariable('Q_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
    Q_flag_var.long_name = 'quality flag for river discharge'
    Q_flag_var.standard_name = 'status_flag'
    Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    Q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    Q_flag_var[:] = data['Q_flag'].values
    
    # SSC (suspended sediment concentration)
    SSC_var = dataset.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    SSC_var.long_name = 'suspended sediment concentration'
    SSC_var.units = 'mg L-1'
    SSC_var.coordinates = 'lat lon'
    SSC_var.ancillary_variables = 'SSC_flag'
    SSC_var.comment = 'Source: Original data provided by Feng et al. (2022, DOI:10.34355/CRiED.U.Tsukuba.00147). Unit conversion: multiplied by 1000 to convert from g/L to mg/L.'
    SSC_var[:] = data['ssc_mg_L'].fillna(-9999.0).values
    
    # SSC_flag
    SSC_flag_var = dataset.createVariable('SSC_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
    SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
    SSC_flag_var.standard_name = 'status_flag'
    SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSC_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSC_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    SSC_flag_var[:] = data['SSC_flag'].values
    
    # SSL (sediment load)
    SSL_var = dataset.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSL_var.long_name = 'suspended sediment load'
    SSL_var.units = 'ton day-1'
    SSL_var.coordinates = 'lat lon'
    SSL_var.ancillary_variables = 'SSL_flag'
    SSL_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (g/L) × 86.4, where 86.4 = 86400 s/day / 1000 L/m³ / 1000 kg/ton. Represents daily average.'
    SSL_var[:] = data['sediment_load'].fillna(-9999.0).values
    
    # SSL_flag
    SSL_flag_var = dataset.createVariable('SSL_flag', 'i1', ('time',), fill_value=FILL_VALUE_INT)
    SSL_flag_var.long_name = 'quality flag for suspended sediment load'
    SSL_flag_var.standard_name = 'status_flag'
    SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSL_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source. Inherits flags from Q and SSC.'
    SSL_flag_var[:] = data['SSL_flag'].values
    
    # Global attributes - CF-1.8 and ACDD-1.3 compliant
    dataset.Conventions = 'CF-1.8, ACDD-1.3'
    dataset.title = 'Harmonized Global River Discharge and Sediment'
    
    time_start = data['datetime'].min().strftime('%Y-%m-%d')
    time_end = data['datetime'].max().strftime('%Y-%m-%d')
    
    dataset.summary = f'River discharge and suspended sediment data for {station_name} station on the {river_name} in Fukushima, Japan. This dataset contains daily averages of water discharge, suspended sediment concentration, and calculated sediment load over the period {time_start} to {time_end}. Data has been quality checked and flagged.'
    
    dataset.source = 'In-situ station data'
    dataset.data_source_name = 'Fukushima Niida River Dataset'

    dataset.comment_auxiliary_variables = (
    "Station altitude and upstream drainage area are not included in the current "
    "version due to lack of reliable source information. These variables will be "
    "added in future releases when available."
    )
    
    # Station information
    dataset.station_name = station_name
    dataset.river_name = river_name
    if source_id:
        dataset.Source_ID = source_id
    
    # Geospatial attributes
    dataset.geospatial_lat_min = lat
    dataset.geospatial_lat_max = lat
    dataset.geospatial_lon_min = lon
    dataset.geospatial_lon_max = lon
    dataset.geospatial_vertical_min = -depth
    dataset.geospatial_vertical_max = -depth
    dataset.geospatial_bounds_crs = 'EPSG:4326'
    
    dataset.geographic_coverage = 'Niida River Basin, Fukushima Prefecture, Japan'
    
    # Temporal attributes
    dataset.time_coverage_start = time_start
    dataset.time_coverage_end = time_end
    dataset.Temporal_Resolution = 'daily'
    dataset.Variables_Provided = 'Q, SSC, SSL'
    
    # References and provenance
    dataset.references = 'DOI: 10.34355/CRiED.U.Tsukuba.00147'
    dataset.reference1 = 'Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., & Zhang, Y. (2022). Dataset of water discharge and suspended sediment at Niida river basin downstream (Haramachi) during 2013 to 2018 and upstream (Notegami) during 2015 to 2018. CRiED, University of Tsukuba. https://doi.org/10.34355/CRiED.U.Tsukuba.00147'
    dataset.reference2 = 'Published in Nature Sustainability (June 2022): "Persistent impact of Fukushima decontamination on soil erosion and suspended sediment"'
    
    dataset.creator_name = 'Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., Zhang, Y.'
    dataset.creator_institution = 'University of Tsukuba, Center for Research in Isotopes and Environmental Dynamics'
    dataset.contributor_name = 'Zhongwang Wei'
    dataset.contributor_email = 'weizhw6@mail.sysu.edu.cn'
    dataset.contributor_institution = 'Sun Yat-sen University, China'
    dataset.contributor_role = 'Data processor and QC'
    
    # Data processing history
    history_msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} - Enhanced with QC flags, CF-1.8 metadata, and standardized formatting by fukushima_qc_and_cf_enhancement.py; "
    history_msg += f"Aggregated from high-frequency measurements to daily averages; "
    history_msg += f"Applied physical constraint QC checks; "
    history_msg += f"Added quality flag variables (Q_flag, SSC_flag, SSL_flag)"
    dataset.history = history_msg
    
    dataset.processing_level = '3 - Derived data'
    dataset.project = 'Global River Harmonized Sediment and Discharge Dataset'
    
    # Close the file
    dataset.close()
    
    return filepath


def process_fukushima_data():
    """Main processing function for Fukushima Niida River data."""
    
    # File paths
    # File paths (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # .../sediment_wzx_1111

    base_dir = os.path.join(project_root, "Source", "Fukushima")
    data_file = os.path.join(base_dir, "DOI00147_data.xls")
    output_dir = os.path.join(project_root, "Output_r", "daily", "Fukushima", "qc")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 90)
    print("FUKUSHIMA NIIDA RIVER DATA - QC & CF-1.8 ENHANCEMENT")
    print("DOI: 10.34355/CRiED.U.Tsukuba.00147")
    print("=" * 90)
    print()
    
    # Read all data
    print("Reading Excel data file...")
    station_data = read_doi00147_data(data_file)
    
    print(f"\nFound {len(station_data)} stations:")
    for station in station_data.keys():
        print(f"  - {station}")
    
    # Process each station
    print("\nProcessing stations with QC checks...")
    print("-" * 90)
    success_count = 0
    summary_data = []
    
    for station_name, data in station_data.items():
        print(f"\n{station_name}:")
        print(f"  Total records: {len(data)}")
        print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"  Valid discharge: {(~data['discharge'].isna()).sum()}")
        print(f"  Valid SSC: {(~data['ssc'].isna()).sum()}")
        
        # Aggregate to daily
        daily_data = aggregate_to_daily(data)
        print(f"  Daily records: {len(daily_data)}")
        
        # Perform QC checks
        qc_data, ssc_q_bounds = perform_qc_checks(daily_data)
        
        # Count good data
        Q_good = (qc_data['Q_flag'] == 0).sum()
        SSC_good = (qc_data['SSC_flag'] == 0).sum()
        SSL_good = (qc_data['SSL_flag'] == 0).sum()
        
        Q_pct = Q_good / len(qc_data) * 100
        SSC_pct = SSC_good / len(qc_data) * 100
        SSL_pct = SSL_good / len(qc_data) * 100
        
        print(f"  Quality flags (% good):")
        print(f"    Q: {Q_good}/{len(qc_data)} ({Q_pct:.1f}%)")
        print(f"    SSC: {SSC_good}/{len(qc_data)} ({SSC_pct:.1f}%)")
        print(f"    SSL: {SSL_good}/{len(qc_data)} ({SSL_pct:.1f}%)")


        # Create source ID
        safe_name = station_name.replace(' ', '_').replace('/', '_')
        source_id = f"DOI00147_{safe_name}"
        
        # --------------------------------------------------
        # SSC–Q diagnostic plot (station-level)
        # --------------------------------------------------
        if ssc_q_bounds is not None:
            diag_dir = os.path.join(output_dir, "ssc_q_diagnostic")
            os.makedirs(diag_dir, exist_ok=True)

            diag_png = os.path.join(
                diag_dir,
                f"Fukushima_{safe_name}_ssc_q_diagnostic.png"
            )

            plot_ssc_q_diagnostic(
                time=qc_data["datetime"].values,
                Q=qc_data["discharge"].values,
                SSC=qc_data["ssc_mg_L"].values,
                Q_flag=qc_data["Q_flag"].values,
                SSC_flag=qc_data["SSC_flag"].values,
                ssc_q_bounds=ssc_q_bounds,
                station_id=source_id,
                station_name=station_name,
                out_png=diag_png,
            )


        # Create NetCDF file
        output_file = os.path.join(output_dir, f"Fukushima_{safe_name}.nc")
        
        try:
            create_netcdf_cf18(output_file, qc_data, station_name, 'Niida River', source_id)

        # -----------------------------------------------
        # CF-1.8 / ACDD-1.3 completeness check
        # -----------------------------------------------
        # errors, warnings = check_nc_completeness(output_file)

        # if errors:
        #     print("  ❌ CF/ACDD compliance FAILED:")
        #     for e in errors:
        #         print(f"     - {e}")
        #     raise RuntimeError("NetCDF compliance check failed")

        # if warnings:
        #     print("  ⚠️ CF/ACDD compliance warnings:")
        #     for w in warnings:
        #         print(f"     - {w}")


            print(f"  Created: {output_file}")
            success_count += 1
                        # ---- QC printout (like your screenshot) ----
            n_samples = len(qc_data)
            skipped_log_iqr = (n_samples < 5)
            skipped_ssc_q = (n_samples < 5)

            # 代表值：优先取 flag=0 的均值；如果没有 good，就取总体均值
            def _mean_or_all(series, flag_series):
                good = series[flag_series == 0]
                if len(good) > 0:
                    return float(np.nanmean(good))
                return float(np.nanmean(series))

            q_value = _mean_or_all(qc_data["discharge"].values, qc_data["Q_flag"].values)
            ssc_value = _mean_or_all(qc_data["ssc_mg_L"].values, qc_data["SSC_flag"].values)
            ssl_value = _mean_or_all(qc_data["sediment_load"].values, qc_data["SSL_flag"].values)

            # 用“最后一天”的 flag 作为展示（也可改成众数/最大）
            q_flag_show = qc_data["Q_flag"].values[-1]
            ssc_flag_show = qc_data["SSC_flag"].values[-1]
            ssl_flag_show = qc_data["SSL_flag"].values[-1]

            log_station_qc(
                station_name=station_name,
                source_id=source_id,
                n_samples=n_samples,
                q_value=q_value, ssc_value=ssc_value, ssl_value=ssl_value,
                q_flag=q_flag_show, ssc_flag=ssc_flag_show, ssl_flag=ssl_flag_show,
                skipped_log_iqr=skipped_log_iqr,
                skipped_ssc_q=skipped_ssc_q,
                created_path=output_file,
            )
            
            # Store summary info
            summary_data.append({
                'station_name': station_name,
                'Source_ID': source_id,
                'river_name': 'Niida River',
                'longitude': data['longitude'].iloc[0],
                'latitude': data['latitude'].iloc[0],
                'altitude': -data['depth'].iloc[0],  # Negative for below surface
                'upstream_area': np.nan,
                'Data Source Name': 'Fukushima Niida River Dataset',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{data['datetime'].min().strftime('%Y-%m-%d')} to {data['datetime'].max().strftime('%Y-%m-%d')}",
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Niida River Basin, Fukushima, Japan',
                'Reference/DOI': 'https://doi.org/10.34355/CRiED.U.Tsukuba.00147',
                'Q_start_date': data[~data['discharge'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['discharge'].isna()).any() else 'N/A',
                'Q_end_date': data[~data['discharge'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['discharge'].isna()).any() else 'N/A',
                'Q_percent_complete': Q_pct,
                'SSC_start_date': data[~data['ssc'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['ssc'].isna()).any() else 'N/A',
                'SSC_end_date': data[~data['ssc'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['ssc'].isna()).any() else 'N/A',
                'SSC_percent_complete': SSC_pct,
                'SSL_start_date': data[~data['sediment_load'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['sediment_load'].isna()).any() else 'N/A',
                'SSL_end_date': data[~data['sediment_load'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['sediment_load'].isna()).any() else 'N/A',
                'SSL_percent_complete': SSL_pct,
            })
            
        except Exception as e:
            print(f"  Error creating NetCDF: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, 'Fukushima_station_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nGenerated summary CSV: {summary_csv}")
    
    print(f"\n{'=' * 90}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {success_count} stations")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    process_fukushima_data()
