
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
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
    propagate_ssc_q_inconsistency_to_ssl,
    # check_nc_completeness,
    # add_global_attributes
)


def apply_tool_qc_yajiang(df, station_id, diagnostic_dir=None):
    """
    Unified QC for Yajiang daily Q and SSC using tool.py
    """

    out = df.copy()

    # --------------------------------------------------
    # 1. Physical validity (tool.py)
    # --------------------------------------------------
    if 'Q' in out.columns:
        out['Q_flag'] = np.array(
            [apply_quality_flag(v, "Q") for v in out['Q'].values],
            dtype=np.int8
        )
        # Bad data → NaN
        out.loc[out['Q_flag'] == 3, 'Q'] = np.nan
    
    if 'SSC' in out.columns:
        out['SSC_flag'] = np.array(
            [apply_quality_flag(v, "SSC") for v in out['SSC'].values],
            dtype=np.int8
        )
        # Bad data → NaN
        out.loc[out['SSC_flag'] == 3, 'SSC'] = np.nan

    # --------------------------------------------------
    # 2. log-IQR screening (observed variables only)
    # --------------------------------------------------
    for var in ['Q', 'SSC']:
        if var in out.columns:
            vals = out[var].values
            lower, upper = compute_log_iqr_bounds(vals)
            if (lower is not None) and (upper is not None):
                out.loc[(vals < lower) | (vals > upper), f'{var}_flag'] = 2


    # --------------------------------------------------
    # 3. SSC–Q consistency check (only if both exist)
    # --------------------------------------------------
    if 'Q' in out.columns and 'SSC' in out.columns:
        envelope = build_ssc_q_envelope(out['Q'].values, out['SSC'].values)
        out['ssc_q_inconsistent'] = False
        if envelope is not None:
            inconsistent = []
            for i in range(len(out)):
                is_bad, _ = check_ssc_q_consistency(
                    Q=out['Q'].values[i],
                    SSC=out['SSC'].values[i],
                    Q_flag=out['Q_flag'].values[i],
                    SSC_flag=out['SSC_flag'].values[i],
                    ssc_q_bounds=envelope
                )
                inconsistent.append(bool(is_bad))

            inconsistent = np.array(inconsistent, dtype=bool)
            out.loc[inconsistent, 'ssc_q_inconsistent'] = True
            out.loc[inconsistent, 'SSC_flag'] = np.int8(2)

            if diagnostic_dir is not None:
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                plot_ssc_q_diagnostic(
                    Q=out['Q'].values,
                    SSC=out['SSC'].values,
                    SSC_flag=out['SSC_flag'].values,
                    envelope=envelope,
                    station_name=f"Yajiang_{station_id}",
                    save_path=diagnostic_dir / f"SSC_Q_Yajiang_{station_id}.png"
                )

    return out



def process_yajiang():
    PROJECT_ROOT = Path(CURRENT_DIR).resolve().parent.parent

    input_dir = PROJECT_ROOT / "Output_r" / "daily" / "Yajiang" / "nc"
    output_dir = PROJECT_ROOT / "Output_r" / "daily" / "Yajiang" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_station_summary_data = []
    nc_files = sorted(list(input_dir.glob('Yajiang_a*.nc')))

    for nc_file in nc_files:
        station_id = nc_file.stem.split('_')[1]
        print(f"Processing station {station_id} from {nc_file.name}...")

        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:
            print(f"  Skipping station {station_id}: could not open NetCDF file. Error: {e}")
            continue

        df = ds.to_dataframe()

        # Handle case where index is MultiIndex (time, lat, lon) -> extract time and reset index
        if isinstance(df.index, pd.MultiIndex):
            time_index = df.index.get_level_values('time')
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(time_index):
                try:
                    time_index = pd.to_datetime(time_index)
                except:
                    pass
            df = df.reset_index(drop=False)
            df.index = time_index
        
        # Unit conversion and calculation
        if 'Q' in df:
            pass  # Q already in m³/s
        elif 'discharge' in df:
            df['Q'] = df['discharge']
        
        if 'SSC' in df:
            pass  # SSC already in g/L
        elif 'ssc' in df:
            df['SSC'] = df['ssc']
        
        if 'Q' in df and 'SSC' in df:
            df['SSL'] = df['Q'] * df['SSC'] * 0.0864

        # Time cropping - allow processing with either Q or SSC
        subset_cols = [col for col in ['Q', 'SSC'] if col in df.columns]
        if not subset_cols:
            print(f"  Skipping station {station_id}: No Q or SSC data.")
            continue
        
        valid_data = df.dropna(subset=subset_cols, how='all')
        if valid_data.empty:
            print(f"  Skipping station {station_id}: No valid data.")
            continue
        start_date = valid_data.index.min()
        end_date = valid_data.index.max()
        
        # Convert timestamp to datetime if needed
        if isinstance(start_date, (tuple, np.ndarray)):
            start_date = pd.Timestamp(start_date).to_pydatetime().date()
        if isinstance(end_date, (tuple, np.ndarray)):
            end_date = pd.Timestamp(end_date).to_pydatetime().date()

        # --------------------------------------------------
        # Apply unified QC (tool.py)
        # --------------------------------------------------
        diagnostic_dir = output_dir / "diagnostic"

        df = apply_tool_qc_yajiang(
            df,
            station_id=station_id,
            diagnostic_dir=diagnostic_dir
        )

        # --------------------------------------------------
        # Calculate SSL AFTER QC (derived variable)
        # --------------------------------------------------
        if 'Q' in df.columns and 'SSC' in df.columns:
            df['SSL'] = df['Q'] * df['SSC'] * 0.0864
            df['SSL_flag'] = FILL_VALUE_INT
            # Only mark SSL as good if both Q and SSC are good
            valid_ssl = (
                (df['Q_flag'] == 0) &
                (df['SSC_flag'] == 0) &
                df['SSL'].notna()
            )
            df.loc[valid_ssl, 'SSL_flag'] = 0
            # propagate SSC-Q inconsistency to SSL_flag
            if 'ssc_q_inconsistent' in df.columns:
                bad = (df['ssc_q_inconsistent'] == True) & (df['SSC_flag'] == 0)
                df.loc[bad, 'SSC_flag'] = np.int8(2)  # suspect
                for i in df.index[bad]:
                    df.at[i, 'SSL_flag'] = propagate_ssc_q_inconsistency_to_ssl(
                        inconsistent=True,
                        Q=df.at[i, 'Q'],
                        SSC=df.at[i, 'SSC'],
                        SSL=df.at[i, 'SSL'],
                        Q_flag=np.int8(df.at[i, 'Q_flag']),
                        SSC_flag=np.int8(df.at[i, 'SSC_flag']),
                        SSL_flag=np.int8(df.at[i, 'SSL_flag']),
                        ssl_is_derived_from_q_ssc=True,
                    )

        elif 'Q' in df.columns:
            # Only Q available - mark Q_flag as FILL if missing
            if 'Q_flag' not in df.columns:
                df['Q_flag'] = FILL_VALUE_INT
        elif 'SSC' in df.columns:
            # Only SSC available - mark SSC_flag as FILL if missing
            if 'SSC_flag' not in df.columns:
                df['SSC_flag'] = FILL_VALUE_INT


        # Create new xarray Dataset
        new_ds = xr.Dataset()
        new_ds['time'] = ('time', df.index)

        variables = {
            'Q': {'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var_key, attrs in variables.items():
            if var_key in df.columns:
                new_ds[var_key] = ('time', df[var_key].astype(np.float32).values)
                new_ds[var_key].attrs = {
                    'long_name': attrs['long_name'], 'standard_name': attrs['standard_name'], 'units': attrs['units'],
                    '_FillValue': -9999.0, 'ancillary_variables': f'{var_key}_flag',
                    'comment': "Source: Original data. Calculated if applicable."
                }
                if f'{var_key}_flag' in df.columns:
                    new_ds[f'{var_key}_flag'] = ('time', df[f'{var_key}_flag'].astype(np.byte).values)
                    new_ds[f'{var_key}_flag'].attrs = {
                        'long_name': f'Quality flag for {attrs["long_name"]}',
                        '_FillValue': FILL_VALUE_INT,
                        'flag_values': np.array([0, 1, 2, 3, 9], dtype=np.int8),
                        'flag_meanings': 'good_data estimated_data suspect_data bad_data missing_data',
                    }


        # Coordinates and other metadata
        # Get lat/lon from global attributes or variables
        lat = np.nan
        lon = np.nan
        
        # Try to get from variables first
        if 'latitude' in ds.data_vars:
            try:
                lat = float(ds.latitude.item())
            except:
                pass
        if 'longitude' in ds.data_vars:
            try:
                lon = float(ds.longitude.item())
            except:
                pass
        
        # If not found in variables, get from global attributes
        if np.isnan(lat) and 'lat' in ds.attrs:
            try:
                lat = float(ds.attrs['lat'])
            except:
                pass
        if np.isnan(lon) and 'lon' in ds.attrs:
            try:
                lon = float(ds.attrs['lon'])
            except:
                pass
        
        # Add as variables to output dataset
        new_ds['lat'] = ((), lat)
        new_ds['lon'] = ((), lon)
        new_ds['lat'].attrs = {'long_name': 'station latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}
        new_ds['lon'].attrs = {'long_name': 'station longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}
        new_ds['altitude'] = ((), ds.altitude.item() if 'altitude' in ds else np.nan)
        new_ds['upstream_area'] = ((), np.nan) # Not available

        # Global attributes
        new_ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'Yajiang Dataset',
            'station_name': ds.attrs.get('station_name', 'N/A'),
            'river_name': 'Yarlung Tsangpo River',
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'daily',
            'Temporal_Span': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'Geographic_Coverage': 'Yarlung Tsangpo River Basin, China',
            'Variables_Provided': ', '.join([var for var in ['Q', 'SSC', 'SSL'] if var in new_ds.variables]),
            'Reference': 'doi:10.11888/Hydro.tpdc.270293',
            'summary': 'This dataset contains daily river discharge and suspended sediment data for the Yarlung Tsangpo River.',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'{ds.attrs.get("history", "")}; Processed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }

        # Save to NetCDF
        # -------- QC summary print --------
        n_total = len(df)
        def _repr(v, f):
            v = np.asarray(v, dtype=float)
            f = np.asarray(f, dtype=np.int8)
            ok = np.isfinite(v) & (v > 0)
            ok_good = ok & (f == 0)
            if np.any(ok_good):
                return float(np.nanmedian(v[ok_good])), 0
            if np.any(ok):
                return float(np.nanmedian(v[ok])), int(np.min(f[ok]))
            return float("nan"), 9

        qv, qf = (np.nan, 9)
        sscv, sscf = (np.nan, 9)
        sslv, sslf = (np.nan, 9)

        if 'Q' in df.columns and 'Q_flag' in df.columns:
            qv, qf = _repr(df['Q'].values, df['Q_flag'].values)
        if 'SSC' in df.columns and 'SSC_flag' in df.columns:
            sscv, sscf = _repr(df['SSC'].values, df['SSC_flag'].values)
        if 'SSL' in df.columns and 'SSL_flag' in df.columns:
            sslv, sslf = _repr(df['SSL'].values, df['SSL_flag'].values)

        print(f"\n✅ QC summary (Yajiang_{station_id})")
        print(f"   Samples: {n_total}")
        if 'Q' in df.columns:   print(f"   Q  : {qv:.2f} m3/s (flag={qf})")
        if 'SSC' in df.columns: print(f"   SSC: {sscv:.2f} mg/L (flag={sscf})")
        if 'SSL' in df.columns: print(f"   SSL: {sslv:.2f} ton/day (flag={sslf})")
# -------------------------------

        output_file = output_dir / f'Yajiang_{station_id}.nc'
        new_ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': f'days since {start_date.year}-01-01'}})
        


        # Summary for CSV
        for var_key in ['Q', 'SSC', 'SSL']:
            if f'{var_key}_flag' in df.columns:
                good_data = df[df[f'{var_key}_flag'] == 0]
                if not good_data.empty:
                    all_station_summary_data.append({
                        'Source_ID': station_id,
                        'station_name': new_ds.attrs['station_name'],
                        'river_name': new_ds.attrs['river_name'],
                        'longitude': new_ds.lon.item(),
                        'latitude': new_ds.lat.item(),
                        'altitude': new_ds.altitude.item(),
                        'upstream_area': new_ds.upstream_area.item(),
                        'Variable': var_key,
                        'Start_Date': good_data.index.min().strftime('%Y-%m-%d'),
                        'End_Date': good_data.index.max().strftime('%Y-%m-%d'),
                        'Percent_Complete': 100 * len(good_data) / len(df.loc[good_data.index.min():good_data.index.max()]),
                        'Mean': good_data[var_key].mean(),
                        'Median': good_data[var_key].median(),
                        'Range': f"{good_data[var_key].min()} - {good_data[var_key].max()}"
                    })
        ds.close()

    # Create and save summary CSV
    summary_df = pd.DataFrame(all_station_summary_data)
    summary_df.to_csv(output_dir / 'Yajiang_station_summary.csv', index=False)

    print("Processing complete.")

if __name__ == '__main__':
    process_yajiang()
