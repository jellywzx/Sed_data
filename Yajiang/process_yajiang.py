
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os
import sys
import inspect
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    apply_quality_flag_array,
    apply_hydro_qc_with_provenance,
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl,
    # check_nc_completeness,
    # add_global_attributes
)


def apply_tool_qc_yajiang(df, station_id, diagnostic_dir=None, station_name=None):
    """
    Unified QC for Yajiang daily data using tool pipeline (apply_hydro_qc_with_provenance).

    Outputs on the returned DataFrame:
      - final flags: Q_flag / SSC_flag / SSL_flag
      - step/provenance flags (if tool provides): e.g., Q_qc1_*, Q_qc2_*, SSC_qc3_*, SSL_qc3_*
    """

    out = df.copy()

    # ---- strict 1D(time)
    time = np.atleast_1d(out.index.values).reshape(-1)

    # ---- observed variables (may be missing)
    Q = np.atleast_1d(out["Q"].values).reshape(-1) if "Q" in out.columns else None
    SSC = np.atleast_1d(out["SSC"].values).reshape(-1) if "SSC" in out.columns else None

    # ---- derived SSL (ton/day) only if both exist
    SSL = None
    if Q is not None and SSC is not None:
        SSL = Q * SSC * 0.0864

    # ---- Build kwargs and filter by tool signature (avoid unexpected keyword errors)
    qc_kwargs = dict(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,

        # Your rule:
        # - unit conversion derived (or directly provided variables) => independent=True
        # - SSL derived from Q*SSC => independent=False
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
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
    if Q is not None:
        out["Q"] = _get(qc, "Q", Q)
    if SSC is not None:
        out["SSC"] = _get(qc, "SSC", SSC)
    if SSL is not None:
        out["SSL"] = _get(qc, "SSL", SSL)

    # ---- defaults: if qc doesn't return flags, compute with apply_quality_flag_array (fallback to apply_quality_flag)
    def _default_flags(arr, varname):
        if arr is None:
            return None
        try:
            return np.asarray(apply_quality_flag_array(arr, varname), dtype=np.int8)
        except Exception:
            return np.asarray([apply_quality_flag(v, varname) for v in np.asarray(arr).reshape(-1)], dtype=np.int8)

    Q_def = _default_flags(Q, "Q")
    SSC_def = _default_flags(SSC, "SSC")
    SSL_def = _default_flags(SSL, "SSL") if SSL is not None else None

    if Q is not None:
        out["Q_flag"] = _get(qc, "Q_flag", Q_def if Q_def is not None else np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8))
    else:
        out["Q_flag"] = np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8)

    if SSC is not None:
        out["SSC_flag"] = _get(qc, "SSC_flag", SSC_def if SSC_def is not None else np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8))
    else:
        out["SSC_flag"] = np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8)

    if SSL is not None:
        out["SSL_flag"] = _get(qc, "SSL_flag", SSL_def if SSL_def is not None else np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8))
    else:
        # if no SSL computed, still keep column for downstream summary
        out["SSL_flag"] = np.full(out.shape[0], FILL_VALUE_INT, dtype=np.int8)

    # ---- step/provenance flags -> add as columns (if present)
    if isinstance(prov, dict):
        for k, v in prov.items():
            vv = np.atleast_1d(v).reshape(-1)
            if vv.shape[0] == out.shape[0]:
                out[k] = vv

    return out


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
    # Convention: 0=pass, 1=not_checked, 2=suspect, 3=bad, fill=missing
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
    station_name = station_info.get("station_name", station_id)
    river_name = station_info.get("river_name", station_name)
    lon = float(station_info.get("longitude", np.nan))
    lat = float(station_info.get("latitude", np.nan))

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

    # ---- qc1 (physical): pass / bad / missing
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station,
                             preferred_names=[f"{v}_qc1_physical"],
                             prefix=f"{v}_qc1")
        if col is None:
            row[f"{v}_qc1_pass"] = 0
            row[f"{v}_qc1_bad"] = 0
            row[f"{v}_qc1_missing"] = 0
        else:
            c = _count_step_flags(df_station[col].values, fill_value)
            row[f"{v}_qc1_pass"] = c["pass"]
            row[f"{v}_qc1_bad"] = c["bad"]
            row[f"{v}_qc1_missing"] = c["missing"]

    # ---- qc2 (log iqr): pass / suspect / not_checked / missing
    for v in ["Q", "SSC", "SSL"]:
        col = _pick_step_col(df_station,
                             preferred_names=[f"{v}_qc2_log_iqr"],
                             prefix=f"{v}_qc2")
        if col is None:
            row[f"{v}_qc2_pass"] = 0
            row[f"{v}_qc2_suspect"] = 0
            row[f"{v}_qc2_not_checked"] = 0
            row[f"{v}_qc2_missing"] = 0
        else:
            c = _count_step_flags(df_station[col].values, fill_value)
            row[f"{v}_qc2_pass"] = c["pass"]
            row[f"{v}_qc2_suspect"] = c["suspect"]
            row[f"{v}_qc2_not_checked"] = c["not_checked"]
            row[f"{v}_qc2_missing"] = c["missing"]

    # ---- qc3 (SSC: envelope/consistency): pass / suspect / not_checked / missing
    col = _pick_step_col(df_station, preferred_names=[], prefix="SSC_qc3")
    if col is None:
        row["SSC_qc3_pass"] = 0
        row["SSC_qc3_suspect"] = 0
        row["SSC_qc3_not_checked"] = 0
        row["SSC_qc3_missing"] = 0
    else:
        c = _count_step_flags(df_station[col].values, fill_value)
        row["SSC_qc3_pass"] = c["pass"]
        row["SSC_qc3_suspect"] = c["suspect"]
        row["SSC_qc3_not_checked"] = c["not_checked"]
        row["SSC_qc3_missing"] = c["missing"]

    # ---- qc3 (SSL propagation): not_propagated / propagated / not_checked / missing
    col = _pick_step_col(df_station, preferred_names=[], prefix="SSL_qc3")
    if col is None:
        row["SSL_qc3_not_propagated"] = 0
        row["SSL_qc3_propagated"] = 0
        row["SSL_qc3_not_checked"] = 0
        row["SSL_qc3_missing"] = 0
    else:
        c = _count_step_flags(df_station[col].values, fill_value)
        # Convention used in our pipeline: 0=not_propagated, 2=propagated, 1=not_checked
        row["SSL_qc3_not_propagated"] = c["pass"]
        row["SSL_qc3_propagated"] = c["suspect"]
        row["SSL_qc3_not_checked"] = c["not_checked"]
        row["SSL_qc3_missing"] = c["missing"]

    return row
def process_yajiang():
    PROJECT_ROOT = Path(CURRENT_DIR).resolve().parent.parent

    input_dir = PROJECT_ROOT / "Output_r" / "daily" / "Yajiang" / "nc"
    output_dir = PROJECT_ROOT / "Output_r" / "daily" / "Yajiang" / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_station_summary_data = []
    qc_rows = []
    nc_files = sorted(list(input_dir.glob('Yajiang_a*.nc')))

    for nc_file in nc_files:
        station_id = nc_file.stem.split('_')[1]
        print(f"Processing station {station_id} from {nc_file.name}...")

        try:
            ds = xr.open_dataset(nc_file)
        except Exception as e:
            print(f"  Skipping station {station_id}: could not open NetCDF file. Error: {e}")
            continue


        # --------------------------
        # Station metadata (for qc_results_summary.csv)
        # --------------------------
        station_name = ds.attrs.get('station_name', station_id)
        river_name = ds.attrs.get('river_name', 'Yarlung Tsangpo River')

        lat = np.nan
        lon = np.nan
        # Try to get from variables first
        if 'latitude' in ds.data_vars:
            try:
                lat = float(ds.latitude.item())
            except Exception:
                pass
        if 'longitude' in ds.data_vars:
            try:
                lon = float(ds.longitude.item())
            except Exception:
                pass
        # If not found in variables, get from global attributes
        if np.isnan(lat) and 'lat' in ds.attrs:
            try:
                lat = float(ds.attrs['lat'])
            except Exception:
                pass
        if np.isnan(lon) and 'lon' in ds.attrs:
            try:
                lon = float(ds.attrs['lon'])
            except Exception:
                pass

        station_info = {
            "station_name": station_name,
            "river_name": river_name,
            "longitude": lon,
            "latitude": lat,
        }
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
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

        df = apply_tool_qc_yajiang(
            df,
            station_id=station_id,
            diagnostic_dir=diagnostic_dir,
            station_name=station_name,
        )

        # --------------------------------------------------

        # --------------------------------------------------
        # SSL derived (ton/day) - keep tool output if present
        # --------------------------------------------------
        if ('Q' in df.columns) and ('SSC' in df.columns) and ('SSL' not in df.columns):
            df['SSL'] = df['Q'] * df['SSC'] * 0.0864
        if 'SSL' in df.columns and 'SSL_flag' not in df.columns:
            # fallback: if tool didn't output SSL_flag, mark good only if both Q & SSC are good
            df['SSL_flag'] = FILL_VALUE_INT
            if ('Q_flag' in df.columns) and ('SSC_flag' in df.columns):
                valid_ssl = (df['Q_flag'] == 0) & (df['SSC_flag'] == 0) & df['SSL'].notna()
                df.loc[valid_ssl, 'SSL_flag'] = 0

        # --------------------------
        # QC results summary row (final + step flags)
        # --------------------------
        try:
            qc_row = build_qc_results_summary_row(
                df_station=df,
                station_info=station_info,
                station_id=station_id,
                fill_value=FILL_VALUE_INT,
            )
            qc_rows.append(qc_row)
        except Exception:
            pass

        # --------------------------
        # Per-station QC results CSV (time series)
        # --------------------------
        qc_csv_dir = output_dir / "qc_results_csv"
        qc_csv_dir.mkdir(parents=True, exist_ok=True)
        qc_csv = qc_csv_dir / f"Yajiang_{station_id}_qc_results.csv"

        try:
            qc_csv_kwargs = dict(
                station_name=station_name,
                Source_ID=station_id,
                time=df.index.values,
                Q=df["Q"].values if "Q" in df.columns else None,
                SSC=df["SSC"].values if "SSC" in df.columns else None,
                SSL=df["SSL"].values if "SSL" in df.columns else None,
                Q_flag=df["Q_flag"].values if "Q_flag" in df.columns else None,
                SSC_flag=df["SSC_flag"].values if "SSC_flag" in df.columns else None,
                SSL_flag=df["SSL_flag"].values if "SSL_flag" in df.columns else None,
                out_csv=str(qc_csv),
            )
            sig2 = inspect.signature(generate_qc_results_csv_tool)
            qc_csv_kwargs = {k: v for k, v in qc_csv_kwargs.items() if (k in sig2.parameters) and (v is not None)}
            generate_qc_results_csv_tool(**qc_csv_kwargs)
        except Exception:
            # Tool signature may differ; do not break main processing
            pass
        if 'Q' in df.columns:
            # Only Q available - mark Q_flag as FILL if missing
            if 'Q_flag' not in df.columns:
                df['Q_flag'] = FILL_VALUE_INT
        if 'SSC' in df.columns:
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

    # --------------------------
    # Save qc_results_summary.csv (final + step flags)
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

        qc_out = output_dir / "qc_results_summary.csv"
        try:
            sum_kwargs = dict(summary_data=qc_summary_df.to_dict(orient="records"), out_csv=str(qc_out))
            sig3 = inspect.signature(generate_csv_summary_tool)
            sum_kwargs = {k: v for k, v in sum_kwargs.items() if k in sig3.parameters}
            generate_csv_summary_tool(**sum_kwargs)
        except Exception:
            qc_summary_df.to_csv(qc_out, index=False)

        print(f"QC results summary saved: {qc_out} ({len(qc_rows)} stations)")
    else:
        print("No QC results summary rows produced.")

    print("Processing complete.")

if __name__ == '__main__':
    process_yajiang()
