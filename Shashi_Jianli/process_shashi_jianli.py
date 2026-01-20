
import pandas as pd
import numpy as np
import xarray as xr
import json
import os
from pathlib import Path
from datetime import datetime
import sys
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
    # check_nc_completeness,
    # add_global_attributes
)

def apply_tool_qc_shashi(df, station_id, diagnostic_dir=None):
    """
    Unified QC for Shashi–Jianli using tool.py

    Includes:
    - Physical validity (apply_quality_flag)
    - log-IQR outlier detection
    - SSC–Q consistency
    - Optional SSC–Q diagnostic plot
    """

    out = df.copy()

    # ======================================================
    # 1. Physical QC
    # ======================================================
    out['Q_flag'] = np.array(
        [apply_quality_flag(v, "Q") for v in out['Q'].values],
        dtype=np.int8
    )
    out['SSC_flag'] = np.array(
        [apply_quality_flag(v, "SSC") for v in out['SSC'].values],
        dtype=np.int8
    )

    # Bad data → NaN
    out.loc[out['Q_flag'] == 3, 'Q'] = np.nan
    out.loc[out['SSC_flag'] == 3, 'SSC'] = np.nan

    # ======================================================
    # 2. log-IQR screening
    # ======================================================
    for var in ['Q', 'SSC']:
        values = out[var].values
        lower, upper = compute_log_iqr_bounds(values)
        if lower is not None:
            out.loc[(values < lower) | (values > upper), f'{var}_flag'] = 2

    # ======================================================
    # 3. SSC–Q consistency
    # ======================================================
    envelope = build_ssc_q_envelope(out['Q'].values, out['SSC'].values)

    if envelope is not None:
        for i in range(len(out)):
            inconsistent, _ = check_ssc_q_consistency(
                out['Q'].iloc[i],
                out['SSC'].iloc[i],
                out['Q_flag'].iloc[i],
                out['SSC_flag'].iloc[i],
                envelope
            )
            if inconsistent:
                out.loc[out.index[i], 'SSC_flag'] = 2

        # --------------------------------------------------
        # 4. Diagnostic plot
        # --------------------------------------------------
        if diagnostic_dir is not None:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            fig_path = diagnostic_dir / f"SSC_Q_{station_id}.png"

            plot_ssc_q_diagnostic(
                time=out.index.values,
                Q=out['Q'].values,
                SSC=out['SSC'].values,
                Q_flag=out['Q_flag'].values,
                SSC_flag=out['SSC_flag'].values,
                ssc_q_bounds=envelope,
                station_id=station_id,
                station_name=station_id,
                out_png=str(fig_path)
            )

    # ======================================================
    # 5. SSL & SSL_flag
    # ======================================================
    out['SSL'] = out['Q'] * out['SSC'] * 0.0864
    out['SSL_flag'] = np.array(
        [apply_quality_flag(v, "SSL") for v in out['SSL'].values],
        dtype=np.int8
    )
    out.loc[out['SSL_flag'] == 3, 'SSL'] = np.nan

    return out


def process_shashi_jianli():
    # Define paths
    source_dir = Path("/mnt/d/sediment_wzx_1111/Source/Shashi_Jianli")
    output_dir = Path("/mnt/d/sediment_wzx_1111/Output_r/daily/Shashi_Jianli/qc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(source_dir / 'station_data.csv')
    with open(source_dir / 'station_coords.json', 'r') as f:
        coords = json.load(f)

    # Station information
    stations = {
        'SS': {'name': 'Shashi', 'id': 'SS', 'river_name': 'Yangtze River'},
        'JL': {'name': 'Jianli', 'id': 'JL', 'river_name': 'Yangtze River'}
    }

    summary_data = []

    for station_id, station_info in stations.items():
        print(f"\n{'='*80}")
        print(f"Processing station: {station_info['name']} ({station_id})")
        print(f"{'='*80}")
        
        df_station = df[['Date', f'{station_id}_discharge', f'{station_id}_SSC']].copy()
        df_station.columns = ['Date', 'Q', 'SSC_kg_m3']
        df_station['Date'] = pd.to_datetime(df_station['Date'])

        # Convert to numeric, coercing errors
        df_station['Q'] = pd.to_numeric(df_station['Q'], errors='coerce')
        df_station['SSC_kg_m3'] = pd.to_numeric(df_station['SSC_kg_m3'], errors='coerce')
        
        # Drop rows where both Q and SSC are NaN
        df_station.dropna(subset=['Q', 'SSC_kg_m3'], how='all', inplace=True)

        if df_station.empty:
            print("  ✗ No data available for this station.")
            continue

        print(f"  Original data points: {len(df_station)}")

        # Unit conversions
        df_station['SSC'] = df_station['SSC_kg_m3'] * 1000  # kg/m3 → mg/L

        # Apply QC to observed variables
        df_station = apply_tool_qc_shashi(
            df_station,
            station_id=station_id,
            diagnostic_dir=output_dir / "diagnostic"
        )

        # --------------------------------------------------
        # SSL is a derived variable → calculated AFTER QC
        # --------------------------------------------------
        df_station['SSL'] = df_station['Q'] * df_station['SSC'] * 0.0864

        # SSL flag: derived from inputs
        df_station['SSL_flag'] = np.where(
            df_station['SSL'].isna(),
            FILL_VALUE_INT,
            0
        ).astype(np.int8)

        # Print QC results
        print("\n  Quality Control Results:")
        for var in ['Q', 'SSC', 'SSL']:
            flag_col = f'{var}_flag'
            good = (df_station[flag_col] == 0).sum()
            suspect = (df_station[flag_col] == 2).sum()
            bad = (df_station[flag_col] == 3).sum()
            missing = (df_station[flag_col] == FILL_VALUE_INT).sum()
            total = len(df_station)
            
            good_pct = 100.0 * good / total if total > 0 else 0.0
            suspect_pct = 100.0 * suspect / total if total > 0 else 0.0
            bad_pct = 100.0 * bad / total if total > 0 else 0.0
            missing_pct = 100.0 * missing / total if total > 0 else 0.0
            
            print(f"    {var}:")
            print(f"      Total:     {total:6d} ({100.0:5.1f}%)")
            print(f"      Good:      {good:6d} ({good_pct:5.1f}%)")
            print(f"      Suspect:   {suspect:6d} ({suspect_pct:5.1f}%)")
            print(f"      Bad:       {bad:6d} ({bad_pct:5.1f}%)")
            print(f"      Missing:   {missing:6d} ({missing_pct:5.1f}%)")

        # Time cropping
        valid_data = df_station.dropna(subset=['Q', 'SSC_kg_m3'], how='all')
        if valid_data.empty:
            continue
        
        start_date = valid_data['Date'].min()
        end_date = valid_data['Date'].max()
        
        date_index = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-12-31", freq='D')
        df_station.set_index('Date', inplace=True)
        df_station = df_station.reindex(date_index)
        df_station.index.name = 'time'


        # Create xarray Dataset
        ds = xr.Dataset()
        ds['time'] = ('time', df_station.index)
        
        # Add variables to dataset
        variables = {
            'Q': {'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var, attrs in variables.items():
            ds[var] = ('time', df_station[var].values.astype(np.float32))
            ds[var].attrs = {
                'long_name': attrs['long_name'],
                'standard_name': attrs['standard_name'],
                'units': attrs['units'],
                '_FillValue': -9999.0,
                'ancillary_variables': f'{var}_flag',
                'comment': f"Source: Original data from reference. Calculated if applicable."
            }
            
            # Add flag variables
            ds[f'{var}_flag'] = ('time', df_station[f'{var}_flag'].values.astype(np.byte))
            ds[f'{var}_flag'].attrs = {
                'long_name': f'Quality flag for {attrs["long_name"]}',
                '_FillValue': -127,
                'flag_values': np.array([0, 1, 2, 3], dtype=np.byte),
                'flag_meanings': 'good_data suspect_data bad_data missing_data',
                'comment': "Flag definitions: 0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data"
            }

        # Add coordinate variables
        ds['lat'] = ((), coords[station_id]['lat'], {'long_name': 'station latitude', 'standard_name': 'latitude', 'units': 'degrees_north'})
        ds['lon'] = ((), coords[station_id]['lon'], {'long_name': 'station longitude', 'standard_name': 'longitude', 'units': 'degrees_east'})
        ds['altitude'] = ((), np.nan, {'long_name': 'station altitude', 'standard_name': 'altitude', 'units': 'm', 'comment': 'Not available in source data'})
        ds['upstream_area'] = ((), np.nan, {'long_name': 'upstream drainage area', 'units': 'km2', 'comment': 'Not available in source data'})

        # Global attributes
        ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'Shashi_Jianli Dataset',
            'station_name': station_info['name'],
            'river_name': station_info['river_name'],
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'daily',
            'Temporal_Span': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'Geographic_Coverage': 'Yangtze River Basin, China',
            'Variables_Provided': 'Q, SSC, SSL',
            'Reference': 'Nones, M., Guo, C. (2025). Remote sensing as a support tool to map suspended sediment concentration over extended river reaches. Acta Geophysica, 73:4655-4668. https://doi.org/10.1007/s11600-025-01638-x',
            'summary': 'This dataset contains daily river discharge and suspended sediment data for the Shashi and Jianli stations on the Yangtze River.',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }

        # Save to NetCDF
        output_file = output_dir / f'Shashi_Jianli_{station_id}.nc'
        ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': f'days since {start_date.year}-01-01'}})
        print(f"\n  ✓ NetCDF file created: {output_file}")

        # Summary data for CSV
        for var in ['Q', 'SSC', 'SSL']:
            good_data = df_station[df_station[f'{var}_flag'] == 0]
            if not good_data.empty:
                summary_data.append({
                    'Source_ID': station_id,
                    'station_name': station_info['name'],
                    'river_name': station_info['river_name'],
                    'longitude': coords[station_id]['lon'],
                    'latitude': coords[station_id]['lat'],
                    'altitude': np.nan,
                    'upstream_area': np.nan,
                    'Variable': var,
                    'Start_Date': good_data.index.min().strftime('%Y-%m-%d'),
                    'End_Date': good_data.index.max().strftime('%Y-%m-%d'),
                    'Percent_Complete': 100 * len(good_data) / len(df_station.loc[good_data.index.min():good_data.index.max()]),
                    'Mean': good_data[var].mean(),
                    'Median': good_data[var].median(),
                    'Range': f"{good_data[var].min()} - {good_data[var].max()}"
                })

    # Create and save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'Shashi_Jianli_station_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("Processing complete.")
    print(f"Summary CSV saved: {output_dir / 'Shashi_Jianli_station_summary.csv'}")
    print("="*80)

if __name__ == '__main__':
    process_shashi_jianli()
