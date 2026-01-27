import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout
import sys
import os
from concurrent.futures import ProcessPoolExecutor
import logging
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
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

# --------------------------
# Unit conversion constants
# --------------------------
CFS_TO_CMS = 0.028316846592  # cubic feet per second → cubic meters per second
FEET_TO_METERS = 0.3048
MILES_TO_KM = 1.60934

def apply_tool_qc_usgs(df, station_id, diagnostic_dir=None):
    """
    Unified QC for USGS Q and SSC using tool.py

    NOTE:
    - QC ONLY for observed variables (Q, SSC)
    - SSL is calculated OUTSIDE this function
    """

    out = df.copy()

    # ======================================================
    # 1. Physical validity check
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
    # 2. log-IQR screening (observed variables only)
    # ======================================================
    for var in ['Q', 'SSC']:
        values = out[var].values
        lower, upper = compute_log_iqr_bounds(values)
        if lower is not None:
            out.loc[(values < lower) | (values > upper), f'{var}_flag'] = 2
    # ======================================================
    # 2.5 SSL & SSL_flag (derived, for propagation only)
    # ======================================================
    out["SSL"] = out["Q"] * out["SSC"] * 0.0864
    out["SSL_flag"] = np.array(
        [apply_quality_flag(v, "SSL") for v in out["SSL"].values],
        dtype=np.int8
    )
    out.loc[out["SSL_flag"] == 3, "SSL"] = np.nan
    # ======================================================
    # 3. SSC–Q consistency check
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
                # 先把 SSC 标为 suspect（仅当原来是 good）
                sscf = int(out["SSC_flag"].iloc[i])
                if sscf == 0:
                    out.loc[out.index[i], "SSC_flag"] = np.int8(2)
                    sscf = 2

                # 再把不一致传播到 SSL_flag
                out.loc[out.index[i], "SSL_flag"] = propagate_ssc_q_inconsistency_to_ssl(
                    inconsistent=inconsistent,
                    Q=float(out["Q"].iloc[i]),
                    SSC=float(out["SSC"].iloc[i]),
                    SSL=float(out["SSL"].iloc[i]) if np.isfinite(out["SSL"].iloc[i]) else np.nan,
                    Q_flag=int(out["Q_flag"].iloc[i]),
                    SSC_flag=int(sscf),
                    SSL_flag=int(out["SSL_flag"].iloc[i]),
                    ssl_is_derived_from_q_ssc=True,   # USGS 这里 SSL = Q*SSC 派生的
                )

        # Diagnostic plot
        if diagnostic_dir is not None:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            plot_ssc_q_diagnostic(
                time=out["datetime"].values if "datetime" in out.columns else (out.index.values if hasattr(out.index, "values") else np.arange(len(out))),
                Q=out['Q'].values,
                SSC=out['SSC'].values,
                Q_flag=out['Q_flag'].values,
                SSC_flag=out['SSC_flag'].values,
                ssc_q_bounds=envelope,
                station_id=station_id,
                station_name=f"USGS_{station_id}",
                out_png=str(diagnostic_dir / f"SSC_Q_USGS_{station_id}.png")
            )

    return out

def print_qc_summary(station_id, df):
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

    qv, qf = _repr(df["Q"].values, df["Q_flag"].values)
    sscv, sscf = _repr(df["SSC"].values, df["SSC_flag"].values)
    sslv, sslf = _repr(df["SSL"].values, df["SSL_flag"].values)

    print(f"\nProcessing station {station_id}")
    print(f"✅ QC summary (USGS_{station_id})")
    print(f"   Samples: {n_total}")
    print(f"   Q  : {qv:.2f} m3/s (flag={qf})")
    print(f"   SSC: {sscv:.2f} mg/L (flag={sscf})")
    print(f"   SSL: {sslv:.2f} ton/day (flag={sslf})")


def process_single_station(args):
    """
    Process a single USGS station independently (for multiprocessing).
    Returns a summary dict or None if processing failed.
    """
    station_dir, sites_info_df, output_dir = args
    station_id = station_dir.name.split('_')[1]
    
    try:
        # --------------------------
        # Read discharge and sediment data
        # --------------------------
        discharge_file = station_dir / f"{station_id}_discharge.csv"
        sediment_file = station_dir / f"{station_id}_sediment.csv"

        if not (discharge_file.exists() and sediment_file.exists()):
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'missing discharge or sediment file'}

        discharge_df = pd.read_csv(discharge_file, comment="#", low_memory=False)
        sediment_df = pd.read_csv(sediment_file, comment='#')

        # --------------------------
        # Metadata
        # --------------------------
        station_info = sites_info_df[sites_info_df['site_no'] == station_id]
        if station_info.empty:
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'metadata not found'}
        station_info = station_info.iloc[0]

        # --------------------------
        # Extract and align Q / SSC
        # --------------------------
        discharge_df['datetime'] = pd.to_datetime(discharge_df['datetime'])
        sediment_df['datetime'] = pd.to_datetime(sediment_df['datetime'])

        q_col = next((c for c in discharge_df.columns if '00060' in c), None)
        ssc_col = next((c for c in sediment_df.columns if '80154' in c), None)
        if q_col is None or ssc_col is None:
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'missing 00060 or 80154 column'}

        discharge_df = discharge_df[['datetime', q_col]].rename(columns={q_col: 'Q'})
        sediment_df = sediment_df[['datetime', ssc_col]].rename(columns={ssc_col: 'SSC'})

        # Inner join → only retain times where both Q and SSC exist
        df = pd.merge(discharge_df, sediment_df, on='datetime', how='inner')
        if df.empty:
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'no overlapping Q and SSC records'}

        # --------------------------
        # Unit conversion and QC
        # --------------------------
        df['Q'] = pd.to_numeric(df['Q'], errors='coerce') * CFS_TO_CMS
        df['SSC'] = pd.to_numeric(df['SSC'], errors='coerce')
        df.dropna(subset=['Q', 'SSC'], inplace=True)

        if df.empty:
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'all overlapping values invalid'}

        # Compute SSL (ton/day)
        df['SSL'] = df['Q'] * df['SSC'] * 0.0864

        # --------------------------------------------------
        # Apply unified QC (tool.py)
        # --------------------------------------------------
        diagnostic_dir = output_dir / "diagnostic"

        df = apply_tool_qc_usgs(
            df,
            station_id=station_id,
            diagnostic_dir=diagnostic_dir
        )
        print_qc_summary(station_id, df)

        start_date, end_date = df['datetime'].min(), df['datetime'].max()

        # --------------------------
        # Build xarray Dataset
        # --------------------------
        ds = xr.Dataset()
        ds['time'] = ('time', df['datetime'])

        variables = {
            'Q': {'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var, attrs in variables.items():
            ds[var] = ('time', df[var].astype(np.float32))
            ds[var].attrs = {
                **attrs,
                '_FillValue': -9999.0,
                'ancillary_variables': f'{var}_flag',
            }
            flag_var = f"{var}_flag"
            ds[flag_var] = ('time', df[flag_var].astype(np.int8))
            ds[flag_var].attrs = {
                'flag_values': np.array([0, 1, 2, 3], dtype=np.int8),
                'flag_meanings': 'good_data suspect_data bad_data missing_data',
            }

        # Coordinates
        ds['lat'] = ((), station_info['dec_lat_va'])
        ds['lon'] = ((), station_info['dec_long_va'])
        ds['altitude'] = ((), station_info['alt_va'] * FEET_TO_METERS if pd.notna(station_info['alt_va']) else np.nan)
        ds['upstream_area'] = ((), station_info['drain_area_va'] * MILES_TO_KM**2 if pd.notna(station_info['drain_area_va']) else np.nan)

        # Global attributes
        ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'USGS NWIS',
            'station_name': station_info['station_nm'],
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'irregular_daily_overlap',
            'Temporal_Span': f'{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}',
            'Reference': 'https://waterdata.usgs.gov/nwis',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'Created on {datetime.now():%Y-%m-%d %H:%M:%S}',
        }

        # --------------------------
        # Save NetCDF
        # --------------------------
        output_file = output_dir / f"USGS_{station_id}.nc"
        ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': 'days since 1970-01-01'}})

        # --------------------------
        # Summary record
        # --------------------------
        good_df = df[(df['Q_flag'] == 0) & (df['SSC_flag'] == 0)]
        if not good_df.empty:
            return {
                'status': 'success',
                'station_id': station_id,
                'record_count': len(df),
                'good_count': len(good_df),
                'Source_ID': station_id,
                'station_name': station_info['station_nm'],
                'longitude': station_info['dec_long_va'],
                'latitude': station_info['dec_lat_va'],
                'Start_Date': good_df['datetime'].min().strftime('%Y-%m-%d'),
                'End_Date': good_df['datetime'].max().strftime('%Y-%m-%d'),
                'Count': len(good_df),
                'Mean_Q': good_df['Q'].mean(),
                'Mean_SSC': good_df['SSC'].mean(),
                'Mean_SSL': good_df['SSL'].mean(),
            }
        else:
            return {'status': 'skipped', 'station_id': station_id, 'reason': 'no good data after QC', 'record_count': len(df)}
            
    except Exception as e:
        return {'status': 'error', 'station_id': station_id, 'error': str(e)}


def process_usgs(num_workers=None):
    """
    Main USGS data processing function with multiprocessing support.
    
    Args:
        num_workers (int): Number of worker processes. If None, uses CPU count.
    """
    # --------------------------
    # Paths
    source_dir = Path(REPO_ROOT) / "Source" / "USGS" / "usgs_data_by_station"
    output_dir = Path(REPO_ROOT) / "Output_r" / "daily" / "USGS" / "qc"
    metadata_file = Path(REPO_ROOT) / "Source" / "USGS" / "common_sites_info.xlsx"
    log_file = output_dir / "processing_log.txt"
    output_dir.mkdir(parents=True, exist_ok=True)


    with open(log_file, "w", encoding="utf-8") as log:
        with redirect_stdout(log):
            print("-------------------------------------------------------------")
            print("USGS Daily Data Processing Script (Multiprocess Mode)")
            print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"Source directory: {source_dir}")
            print(f"Output directory: {output_dir}")
            print("-------------------------------------------------------------")

            station_dirs = sorted(list(source_dir.glob('station_*')))
            print(f"Found {len(station_dirs)} stations to process\n")

            # Load metadata once (will be passed to workers)
            sites_info_df = pd.read_excel(metadata_file, dtype={'site_no': str}, engine='openpyxl')
            sites_info_df['site_no'] = sites_info_df['site_no'].astype(str)

            # Prepare arguments for each station
            args_list = [(sd, sites_info_df, output_dir) for sd in station_dirs]

            # Process stations in parallel
            results = []
            processed = 0
            skipped = 0
            errors = 0

            if num_workers is None:
                num_workers = os.cpu_count() or 4

            print(f"Using {num_workers} worker processes...\n")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for result in executor.map(process_single_station, args_list):
                    status = result.get('status', 'unknown')
                    station_id = result.get('station_id', 'unknown')

                    if status == 'success':
                        print(f"✓ Station {station_id}: {result.get('record_count')} records, {result.get('good_count')} good")
                        results.append(result)
                        processed += 1
                    elif status == 'skipped':
                        print(f"⊘ Station {station_id}: {result.get('reason')}")
                        skipped += 1
                    elif status == 'error':
                        print(f"✗ Station {station_id}: {result.get('error')}")
                        errors += 1

            # --------------------------
            # Save summary CSV
            # --------------------------
            if results:
                summary_data = [
                    {
                        'Source_ID': r['Source_ID'],
                        'station_name': r['station_name'],
                        'longitude': r['longitude'],
                        'latitude': r['latitude'],
                        'Start_Date': r['Start_Date'],
                        'End_Date': r['End_Date'],
                        'Count': r['Count'],
                        'Mean_Q': r['Mean_Q'],
                        'Mean_SSC': r['Mean_SSC'],
                        'Mean_SSL': r['Mean_SSL'],
                    }
                    for r in results
                ]
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(output_dir / "USGS_station_summary.csv", index=False)
                print(f"\nSummary CSV saved: USGS_station_summary.csv ({len(summary_data)} stations)")
            else:
                print("\nNo valid stations processed.")

            print("\n" + "="*60)
            print(f"Summary: {processed} processed, {skipped} skipped, {errors} errors")
            print("="*60)
            print(f"Finished at {datetime.now():%Y-%m-%d %H:%M:%S}")
            print(f"Log file saved to: {log_file}")
            print("-------------------------------------------------------------")


if __name__ == "__main__":
    import sys
    
    # Get number of workers from command line argument
    num_workers = None
    if len(sys.argv) > 1:
        try:
            num_workers = int(sys.argv[1])
            print(f"Using {num_workers} worker processes")
        except ValueError:
            print(f"Invalid worker count: {sys.argv[1]}, using default (CPU count)")
    
    process_usgs(num_workers=num_workers)
