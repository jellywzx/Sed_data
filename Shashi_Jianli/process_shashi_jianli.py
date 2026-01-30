
import pandas as pd
import numpy as np
import xarray as xr
import json
import os
import inspect
from pathlib import Path
from datetime import datetime
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
def find_repo_root(start: Path, max_up: int = 6) -> Path:
    p = start.resolve()
    for _ in range(max_up):
        if (p / "Source").exists() and (p / "Output_r").exists():
            return p
        p = p.parent
    # 兜底：找不到就用“脚本所在目录的上两级”，你也可以按需要改
    return start.resolve().parents[2]
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
    apply_quality_flag_array,
    apply_hydro_qc_with_provenance,
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,

)

def apply_tool_qc_shashi(df, station_id, diagnostic_dir=None):
    """
    Unified QC using apply_hydro_qc_with_provenance (tool pipeline).
    """
    out = df.copy()

    # ---- time / Q / SSC: 强制 1D(time)
    # 这里优先用 Date 列作为 time；如果你后面改成 index=time，也能兼容
    if "Date" in out.columns:
        time = np.atleast_1d(pd.to_datetime(out["Date"]).values).reshape(-1)
    else:
        time = np.atleast_1d(out.index.values).reshape(-1)

    Q = np.atleast_1d(out["Q"].values).reshape(-1)
    SSC = np.atleast_1d(out["SSC"].values).reshape(-1)

    #SL：派生量（由 Q 和 SSC 计算得到）
    SSL = Q * SSC * 0.0864

    #进 QC 管线
    qc_kwargs = dict(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        diagnostic_dir=str(diagnostic_dir) if diagnostic_dir is not None else None,

        # 这些参数 tool.py 可能不收；先放进来，后面会自动过滤掉
        station_id=station_id,
        station_name=station_id,
    )

    sig = inspect.signature(apply_hydro_qc_with_provenance)
    qc_kwargs = {k: v for k, v in qc_kwargs.items() if k in sig.parameters}

    res = apply_hydro_qc_with_provenance(**qc_kwargs)


    # ---- 兼容两种返回：res = (qc, prov) 或 res = qc
    if isinstance(res, tuple) and len(res) == 2:
        qc, prov = res
    else:
        qc, prov = res, None

    # ---- 从 qc 中取回最终值与最终 flag（按 dict/object 两种都兼容）
    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    out["Q"] = _get(qc, "Q", out["Q"].values)
    out["SSC"] = _get(qc, "SSC", out["SSC"].values)
    out["SSL"] = _get(qc, "SSL", SSL)

    out["Q_flag"] = _get(qc, "Q_flag", out.get("Q_flag", np.full_like(Q, FILL_VALUE_INT, dtype=np.int8)))
    out["SSC_flag"] = _get(qc, "SSC_flag", out.get("SSC_flag", np.full_like(SSC, FILL_VALUE_INT, dtype=np.int8)))
    out["SSL_flag"] = _get(qc, "SSL_flag", out.get("SSL_flag", np.full_like(SSL, FILL_VALUE_INT, dtype=np.int8)))

    # ---- provenance（分步 flags）如果有，就全部塞回 dataframe
    if isinstance(prov, dict):
        for k, v in prov.items():
            out[k] = np.atleast_1d(v).reshape(-1)

    return out

def _count_flags(flag_arr, fill_value=-127):
    """Return counts for (good, estimated, suspect, bad, missing) using 0/1/2/3/fill_value."""
    a = np.asarray(flag_arr)
    return {
        "good": int(np.sum(a == 0)),
        "estimated": int(np.sum(a == 1)),
        "suspect": int(np.sum(a == 2)),
        "bad": int(np.sum(a == 3)),
        "missing": int(np.sum(a == fill_value)),
    }

def _count_step(step_arr, fill_value=-127):
    """
    Generic step flag counts (pass/bad/missing) or (pass/suspect/not_checked/missing)
    We'll count by values: 0=pass, 1=not_checked, 2=suspect, 3=bad, fill=missing.
    """
    a = np.asarray(step_arr)
    return {
        "pass": int(np.sum(a == 0)),
        "not_checked": int(np.sum(a == 1)),
        "suspect": int(np.sum(a == 2)),
        "bad": int(np.sum(a == 3)),
        "missing": int(np.sum(a == fill_value)),
    }

def _pick_step_col(df_station, prefix):
    """Pick the first provenance column that starts with prefix, e.g., 'Q_qc1'."""
    cols = [c for c in df_station.columns if str(c).startswith(prefix)]
    return cols[0] if len(cols) > 0 else None

def build_qc_results_summary_row(df_station, station_info, station_id, lon, lat, fill_value=-127):
    """
    Build ONE row for qc_results_summary.csv with the columns you specified.
    Assumptions:
      - final flags are Q_flag / SSC_flag / SSL_flag with 0/1/2/3/fill
      - step flags exist as columns starting with:
          Q_qc1..., SSC_qc1..., SSL_qc1...
          Q_qc2..., SSC_qc2..., SSL_qc2...
          SSC_qc3..., SSL_qc3...
        If not found, those counts will be 0.
    """
    n_days = int(len(df_station))

    # ---- final flags
    qf = _count_flags(df_station["Q_flag"].values, fill_value)
    sf = _count_flags(df_station["SSC_flag"].values, fill_value)
    lf = _count_flags(df_station["SSL_flag"].values, fill_value)

    row = {
        "station_name": station_info["name"],
        "Source_ID": station_id,
        "river_name": station_info["river_name"],
        "longitude": float(lon),
        "latitude": float(lat),
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

    # ---- qc1
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station, f"{v}_qc1")
        if col is None:
            row[f"{v}_qc1_pass"] = 0
            row[f"{v}_qc1_bad"] = 0
            row[f"{v}_qc1_missing"] = 0
        else:
            c = _count_step(df_station[col].values, fill_value)
            row[f"{v}_qc1_pass"] = c["pass"]
            row[f"{v}_qc1_bad"] = c["bad"]
            row[f"{v}_qc1_missing"] = c["missing"]

    # ---- qc2
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station, f"{v}_qc2")
        if col is None:
            row[f"{v}_qc2_pass"] = 0
            row[f"{v}_qc2_suspect"] = 0
            row[f"{v}_qc2_not_checked"] = 0
            row[f"{v}_qc2_missing"] = 0
        else:
            c = _count_step(df_station[col].values, fill_value)
            row[f"{v}_qc2_pass"] = c["pass"]
            row[f"{v}_qc2_suspect"] = c["suspect"]
            row[f"{v}_qc2_not_checked"] = c["not_checked"]
            row[f"{v}_qc2_missing"] = c["missing"]

    # ---- qc3 (SSC)
    col = _pick_step_col(df_station, "SSC_qc3")
    if col is None:
        row["SSC_qc3_pass"] = 0
        row["SSC_qc3_suspect"] = 0
        row["SSC_qc3_not_checked"] = 0
        row["SSC_qc3_missing"] = 0
    else:
        c = _count_step(df_station[col].values, fill_value)
        row["SSC_qc3_pass"] = c["pass"]
        row["SSC_qc3_suspect"] = c["suspect"]
        row["SSC_qc3_not_checked"] = c["not_checked"]
        row["SSC_qc3_missing"] = c["missing"]

    # ---- qc3 (SSL propagation style)
    col = _pick_step_col(df_station, "SSL_qc3")
    if col is None:
        row["SSL_qc3_not_propagated"] = 0
        row["SSL_qc3_propagated"] = 0
        row["SSL_qc3_not_checked"] = 0
        row["SSL_qc3_missing"] = 0
    else:
        c = _count_step(df_station[col].values, fill_value)
        # 约定：0=not_propagated/pass, 2=propagated/suspect, 1=not_checked
        row["SSL_qc3_not_propagated"] = c["pass"]
        row["SSL_qc3_propagated"] = c["suspect"]
        row["SSL_qc3_not_checked"] = c["not_checked"]
        row["SSL_qc3_missing"] = c["missing"]

    return row


def process_shashi_jianli():
    # Define paths
    repo_root = find_repo_root(Path(CURRENT_DIR))
    source_dir = repo_root / "Source" / "Shashi_Jianli"
    output_dir = repo_root / "Output_r" / "daily" / "Shashi_Jianli" / "qc"

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
    qc_results_summary_rows = []

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

        # Print QC results
        n_total = len(df_station)
        n_valid_q = int(np.isfinite(df_station["Q"].values).sum())
        n_valid_ssc = int(np.isfinite(df_station["SSC"].values).sum())
        n_valid_ssl = int(np.isfinite(df_station["SSL"].values).sum())

        print(f"  [QC] n_total={n_total}, n_valid(Q)={n_valid_q}, n_valid(SSC)={n_valid_ssc}, n_valid(SSL)={n_valid_ssl}")
        if min(n_valid_q, n_valid_ssc) < 5:
            print(f"  [QC] Sample size < 5 → SSC–Q consistency check & diagnostic plot skipped")

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
        # 打印一个“代表值”（比如第一个 good 的值；没有就显示 nan）
        def first_good_value(vname: str):
            flag = df_station[f"{vname}_flag"]
            good_idx = df_station.index[flag == 0]
            if len(good_idx) == 0:
                return np.nan, 9
            t0 = good_idx[0]
            return float(df_station.loc[t0, vname]), int(df_station.loc[t0, f"{vname}_flag"])

        q0, qf0 = first_good_value("Q")
        s0, sf0 = first_good_value("SSC")
        l0, lf0 = first_good_value("SSL")
        print(f"  [QC] Q:   {q0:.4g} (flag={qf0})")
        print(f"  [QC] SSC: {s0:.4g} (flag={sf0})")
        print(f"  [QC] SSL: {l0:.4g} (flag={lf0})")

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

        # ---- QC results summary row (per station)
        row = build_qc_results_summary_row(
            df_station=df_station,
            station_info=station_info,
            station_id=station_id,
            lon=coords[station_id]["lon"],
            lat=coords[station_id]["lat"],
            fill_value=FILL_VALUE_INT,
        )
        qc_results_summary_rows.append(row)

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
        # QC results CSV (append per station)
        qc_csv = output_dir / "Shashi_Jianli_qc_results.csv"

        try:
            generate_qc_results_csv_tool(
                station_id=station_id,
                time=df_station.index.values if "time" in df_station.index.name else df_station["Date"].values,
                Q=df_station["Q"].values,
                SSC=df_station["SSC"].values,
                SSL=df_station["SSL"].values,
                Q_flag=df_station["Q_flag"].values,
                SSC_flag=df_station["SSC_flag"].values,
                SSL_flag=df_station["SSL_flag"].values,
                out_csv=str(qc_csv),
                mode="append",
            )
        except Exception:
            # 兜底：不影响主流程（你也可以在这里 print 一句提示）
            pass


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
    summary_csv = output_dir / 'Shashi_Jianli_station_summary.csv'
    qc_sum_csv = output_dir / "qc_results_summary.csv"
    pd.DataFrame(qc_results_summary_rows).to_csv(qc_sum_csv, index=False)
    print(f"\n✓ QC results summary saved: {qc_sum_csv}")

    try:
        generate_csv_summary_tool(summary_data, out_csv=str(summary_csv))
    except Exception:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)
    
    print("\n" + "="*80)
    print("Processing complete.")
    print(f"Summary CSV saved: {output_dir / 'Shashi_Jianli_station_summary.csv'}")
    print("="*80)

if __name__ == '__main__':
    process_shashi_jianli()
