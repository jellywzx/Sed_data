import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout
import sys
import os
import inspect
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
    propagate_ssc_q_inconsistency_to_ssl,
    apply_quality_flag_array,
    apply_hydro_qc_with_provenance,
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)

# --------------------------
# Unit conversion constants
# --------------------------
CFS_TO_CMS = 0.028316846592  # cubic feet per second → cubic meters per second
FEET_TO_METERS = 0.3048
MILES_TO_KM = 1.60934

def apply_tool_qc_usgs(df, station_id, diagnostic_dir=None, station_name=None):
    """
    Unified QC for USGS Q / SSC / SSL using tool pipeline (apply_hydro_qc_with_provenance).

    Outputs:
      - final flags: Q_flag / SSC_flag / SSL_flag
      - step/provenance flags (if tool provides): e.g., Q_qc1_*, Q_qc2_*, SSC_qc3_*, SSL_qc3_*
    """
    out = df.copy()

    # ---- strict 1D(time)
    time = np.atleast_1d(out["datetime"].values).reshape(-1)
    Q = np.atleast_1d(out["Q"].values).reshape(-1)
    SSC = np.atleast_1d(out["SSC"].values).reshape(-1)

    # ---- SSL derived from Q * SSC (ton/day)
    SSL = Q * SSC * 0.0864

    # ---- Build kwargs and filter by tool signature (avoid unexpected keyword errors)
    qc_kwargs = dict(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,

        # 你的规则：单位换算得到的量也 independent=True；派生量 False
        Q_is_independent=True,         # CFS->CMS 仅单位换算
        SSC_is_independent=True,       # 原始 mg/L（或仅类型转换）
        SSL_is_independent=False,      # SSL=Q*SSC 推导
        ssl_is_derived_from_q_ssc=True,

        diagnostic_dir=str(diagnostic_dir) if diagnostic_dir is not None else None,
        station_id=station_id,
        station_name=station_name if station_name is not None else station_id,
    )
    sig = inspect.signature(apply_hydro_qc_with_provenance)
    qc_kwargs = {k: v for k, v in qc_kwargs.items() if k in sig.parameters}

    res = apply_hydro_qc_with_provenance(**qc_kwargs)

    # ---- compat: res = (qc, prov) or qc-only
    if isinstance(res, tuple) and len(res) == 2:
        qc, prov = res
    else:
        qc, prov = res, None

    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ---- write back values
    out["Q"] = _get(qc, "Q", Q)
    out["SSC"] = _get(qc, "SSC", SSC)
    out["SSL"] = _get(qc, "SSL", SSL)

    # ---- defaults: if qc doesn't return flags, compute with apply_quality_flag_array (fallback)
    try:
        _Q_default = np.asarray(apply_quality_flag_array(Q, "Q"), dtype=np.int8)
    except Exception:
        _Q_default = np.asarray([apply_quality_flag(v, "Q") for v in Q], dtype=np.int8)

    try:
        _SSC_default = np.asarray(apply_quality_flag_array(SSC, "SSC"), dtype=np.int8)
    except Exception:
        _SSC_default = np.asarray([apply_quality_flag(v, "SSC") for v in SSC], dtype=np.int8)

    try:
        _SSL_default = np.asarray(apply_quality_flag_array(SSL, "SSL"), dtype=np.int8)
    except Exception:
        _SSL_default = np.asarray([apply_quality_flag(v, "SSL") for v in SSL], dtype=np.int8)

    out["Q_flag"] = _get(qc, "Q_flag", _Q_default)
    out["SSC_flag"] = _get(qc, "SSC_flag", _SSC_default)
    out["SSL_flag"] = _get(qc, "SSL_flag", _SSL_default)

    # ---- step/provenance flags -> add as columns (if present)
    if isinstance(prov, dict):
        for k, v in prov.items():
            vv = np.atleast_1d(v).reshape(-1)
            if vv.shape[0] == out.shape[0]:
                out[k] = vv

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

def _count_final_flags(flag_arr, fill_value):
    a = np.asarray(flag_arr, dtype=np.int16)
    return {
        "good": int(np.sum(a == 0)),
        "estimated": int(np.sum(a == 1)),
        "suspect": int(np.sum(a == 2)),
        "bad": int(np.sum(a == 3)),
        "missing": int(np.sum(a == fill_value)),
    }

def _count_step_flags(step_arr, fill_value):
    # 约定：0=pass, 1=not_checked, 2=suspect, 3=bad, fill=missing
    a = np.asarray(step_arr, dtype=np.int16)
    return {
        "pass": int(np.sum(a == 0)),
        "not_checked": int(np.sum(a == 1)),
        "suspect": int(np.sum(a == 2)),
        "bad": int(np.sum(a == 3)),
        "missing": int(np.sum(a == fill_value)),
    }

def _pick_step_col(df, preferred_names, prefix):
    for n in preferred_names:
        if n in df.columns:
            return n
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return cols[0] if cols else None

def build_qc_results_summary_row(df_station, station_info, station_id, fill_value):
    station_name = station_info["station_nm"]
    river_name = station_info.get("river_name", station_name) if hasattr(station_info, "get") else station_name

    lon = float(station_info["dec_long_va"])
    lat = float(station_info["dec_lat_va"])
    n_days = int(len(df_station))

    qf = _count_final_flags(df_station["Q_flag"].values, fill_value)
    sf = _count_final_flags(df_station["SSC_flag"].values, fill_value)
    lf = _count_final_flags(df_station["SSL_flag"].values, fill_value)

    row = {
        "station_name": station_name,
        "Source_ID": station_id,
        "river_name": river_name,
        "longitude": lon,
        "latitude": lat,
        "QC_n_days": n_days,

        "Q_final_good": qf["good"],
        "Q_final_estimated": qf["estimated"],
        "Q_final_suspect": qf["suspect"],
        "Q_final_bad": qf["bad"],
        "Q_final_missing": qf["missing"],

        "SSC_final_good": sf["good"],
        "SSC_final_estimated": sf["estimated"],
        "SSC_final_suspect": sf["suspect"],
        "SSC_final_bad": sf["bad"],
        "SSC_final_missing": sf["missing"],

        "SSL_final_good": lf["good"],
        "SSL_final_estimated": lf["estimated"],
        "SSL_final_suspect": lf["suspect"],
        "SSL_final_bad": lf["bad"],
        "SSL_final_missing": lf["missing"],
    }

    # qc1
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station, [f"{v}_qc1_physical"], f"{v}_qc1")
        if col is None:
            row[f"{v}_qc1_pass"] = row[f"{v}_qc1_bad"] = row[f"{v}_qc1_missing"] = 0
        else:
            c = _count_step_flags(df_station[col].values, fill_value)
            row[f"{v}_qc1_pass"] = c["pass"]
            row[f"{v}_qc1_bad"] = c["bad"]
            row[f"{v}_qc1_missing"] = c["missing"]

    # qc2
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station, [f"{v}_qc2_log_iqr"], f"{v}_qc2")
        if col is None:
            row[f"{v}_qc2_pass"] = row[f"{v}_qc2_suspect"] = row[f"{v}_qc2_not_checked"] = row[f"{v}_qc2_missing"] = 0
        else:
            c = _count_step_flags(df_station[col].values, fill_value)
            row[f"{v}_qc2_pass"] = c["pass"]
            row[f"{v}_qc2_suspect"] = c["suspect"]
            row[f"{v}_qc2_not_checked"] = c["not_checked"]
            row[f"{v}_qc2_missing"] = c["missing"]

    # qc3 SSC
    col = _pick_step_col(df_station, [], "SSC_qc3")
    if col is None:
        row["SSC_qc3_pass"] = row["SSC_qc3_suspect"] = row["SSC_qc3_not_checked"] = row["SSC_qc3_missing"] = 0
    else:
        c = _count_step_flags(df_station[col].values, fill_value)
        row["SSC_qc3_pass"] = c["pass"]
        row["SSC_qc3_suspect"] = c["suspect"]
        row["SSC_qc3_not_checked"] = c["not_checked"]
        row["SSC_qc3_missing"] = c["missing"]

    # qc3 SSL（传播）
    col = _pick_step_col(df_station, [], "SSL_qc3")
    if col is None:
        row["SSL_qc3_not_propagated"] = row["SSL_qc3_propagated"] = row["SSL_qc3_not_checked"] = row["SSL_qc3_missing"] = 0
    else:
        c = _count_step_flags(df_station[col].values, fill_value)
        row["SSL_qc3_not_propagated"] = c["pass"]
        row["SSL_qc3_propagated"] = c["suspect"]
        row["SSL_qc3_not_checked"] = c["not_checked"]
        row["SSL_qc3_missing"] = c["missing"]

    return row

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
            diagnostic_dir=diagnostic_dir,
            station_name=station_info["station_nm"],
        )

        print_qc_summary(station_id, df)
        # QC results summary row (final + step flags)
        qc_row = build_qc_results_summary_row(
            df_station=df,
            station_info=station_info,
            station_id=station_id,
            fill_value=FILL_VALUE_INT,
        )

        # Per-station QC results CSV (safe for multiprocessing)
        qc_csv_dir = output_dir / "qc_results_csv"
        qc_csv_dir.mkdir(parents=True, exist_ok=True)
        qc_csv = qc_csv_dir / f"USGS_{station_id}_qc_results.csv"

        try:
            qc_csv_kwargs = dict(
                station_name=station_info["station_nm"],
                Source_ID=station_id,
                time=df["datetime"].values,
                Q=df["Q"].values,
                SSC=df["SSC"].values,
                SSL=df["SSL"].values,
                Q_flag=df["Q_flag"].values,
                SSC_flag=df["SSC_flag"].values,
                SSL_flag=df["SSL_flag"].values,
                out_csv=str(qc_csv),
            )
            sig2 = inspect.signature(generate_qc_results_csv_tool)
            qc_csv_kwargs = {k: v for k, v in qc_csv_kwargs.items() if k in sig2.parameters}
            generate_qc_results_csv_tool(**qc_csv_kwargs)
        except Exception:
            pass

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
                'qc_row': qc_row,
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
            qc_rows = []
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
                    if 'qc_row' in result and isinstance(result['qc_row'], dict):
                        qc_rows.append(result['qc_row'])
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
            # Save qc_results_summary.csv
            # --------------------------
            if qc_rows:
                qc_summary_df = pd.DataFrame(qc_rows)

                desired_cols = [
                    "station_name","Source_ID","river_name","longitude","latitude","QC_n_days",
                    "Q_final_good","Q_final_estimated","Q_final_suspect","Q_final_bad","Q_final_missing",
                    "SSC_final_good","SSC_final_estimated","SSC_final_suspect","SSC_final_bad","SSC_final_missing",
                    "SSL_final_good","SSL_final_estimated","SSL_final_suspect","SSL_final_bad","SSL_final_missing",
                    "Q_qc1_pass","Q_qc1_bad","Q_qc1_missing",
                    "SSC_qc1_pass","SSC_qc1_bad","SSC_qc1_missing",
                    "SSL_qc1_pass","SSL_qc1_bad","SSL_qc1_missing",
                    "Q_qc2_pass","Q_qc2_suspect","Q_qc2_not_checked","Q_qc2_missing",
                    "SSC_qc2_pass","SSC_qc2_suspect","SSC_qc2_not_checked","SSC_qc2_missing",
                    "SSL_qc2_pass","SSL_qc2_suspect","SSL_qc2_not_checked","SSL_qc2_missing",
                    "SSC_qc3_pass","SSC_qc3_suspect","SSC_qc3_not_checked","SSC_qc3_missing",
                    "SSL_qc3_not_propagated","SSL_qc3_propagated","SSL_qc3_not_checked","SSL_qc3_missing",
                ]
                for c in desired_cols:
                    if c not in qc_summary_df.columns:
                        qc_summary_df[c] = 0
                qc_summary_df = qc_summary_df[desired_cols]

                out_csv = output_dir / "qc_results_summary.csv"
                try:
                    # 用工具函数写（若签名不匹配则自动 fallback）
                    sum_kwargs = dict(summary_data=qc_summary_df.to_dict(orient="records"), out_csv=str(out_csv))
                    sig3 = inspect.signature(generate_csv_summary_tool)
                    sum_kwargs = {k: v for k, v in sum_kwargs.items() if k in sig3.parameters}
                    generate_csv_summary_tool(**sum_kwargs)
                except Exception:
                    qc_summary_df.to_csv(out_csv, index=False)

                print(f"\nQC results summary saved: qc_results_summary.csv ({len(qc_rows)} stations)")
            else:
                print("\nNo QC results summary rows produced.")

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
