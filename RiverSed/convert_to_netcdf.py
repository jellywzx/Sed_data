#!/usr/bin/env python3
"""
Convert Aquasat and RiverSed CSV data to netCDF format
Following HYBAM example structure
Discharge is set to NaN (no in-situ discharge available)

Workflow overview
-----------------
1. Read Aquasat source rows and standardize them to a common station/date/tss schema.
2. Read RiverSed source rows and load the modified NHDPlus DBF lookup table.
3. Normalize RiverSed IDs in both tables, then attach reach/basin metadata by ID.
4. Build per-station row indices so each station can be processed independently.
5. In parallel, convert each station's raw observations into daily SSC time series.
6. Apply SSC QC, then write one netCDF file per station/reach.
7. Collect station-level metadata/QC summaries and export CSV summary tables.

Important RiverSed note
-----------------------
The RiverSed "matching" step in this script is a table join on ID, not a new
runtime GIS spatial join. The modified DBF is assumed to already encode the
reach/basin assignment prepared upstream.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import multiprocessing as mp
import struct
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.output import (
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)
from code.qc import (
    compute_log_iqr_bounds,
)
from code.runtime import resolve_output_root, resolve_source_root

# Resolve the data roots once so the script can be launched from any cwd.
SOURCE_DIR = os.fspath(resolve_source_root(start=__file__) / "RiverSed")

OUTPUT_QC_DIR = os.fspath(
    resolve_output_root(start=__file__) / "daily" / "RiverSed" / "qc"
)

OUTPUT_NC_DIR = OUTPUT_QC_DIR

# The DBF acts as the RiverSed reach/basin lookup table. It maps RiverSed IDs
# to NHDPlus-derived metadata such as COMID, reach code, VPU/RPU, and area.
RIVERSED_METADATA_DBF = os.path.join(SOURCE_DIR, "nhdplusv2_modified_v1.0.dbf")
RIVERSED_METADATA_FIELD_MAP = {
    "ID": "ID",
    "COMID": "comid",
    "GNIS_NA": "river_name",
    "REACHCO": "reach_code",
    "VPUID": "vpu_id",
    "RPUID": "rpu_id",
    "TtDASKM": "upstream_area",
}

# On fork-based systems the worker processes reuse these in-memory tables
# instead of receiving a full station DataFrame through IPC for every task.
_WORKER_DATASETS = {}
_WORKER_GROUP_INDICES = {}
MAX_WORKERS = 16
VERBOSE_STATION_LOGS = False
PROGRESS_BAR_WIDTH = 30
PROGRESS_UPDATE_INTERVAL_SECONDS = 0.25


def _resolve_num_workers(max_workers=MAX_WORKERS):
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, max_workers))


def _compute_chunksize(total_tasks, num_workers):
    if total_tasks <= 0:
        return 1
    return max(1, min(64, (total_tasks + (num_workers * 4) - 1) // (num_workers * 4)))


def _format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _render_progress(stage_label, completed, total, start_time, success, failed):
    # Render a single-line progress bar for the current processing stage.
    # This stays in the main process so worker output does not garble the UI.
    total = max(total, 1)
    progress_ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(PROGRESS_BAR_WIDTH * progress_ratio)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    elapsed = max(0.0, time.perf_counter() - start_time)
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = ((total - completed) / rate) if rate > 0 else 0.0
    line = (
        f"\r  {stage_label:<8} |{bar}| "
        f"{completed:>6}/{total:<6} "
        f"{progress_ratio * 100:5.1f}% "
        f"ok={success:<6} fail={failed:<6} "
        f"elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(eta)}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def _build_station_group_indices(df, station_ids=None):
    # Store row positions once so we do not rescan the full table per station.
    grouped_indices = df.groupby("station_id", sort=False).indices
    if station_ids is None:
        return grouped_indices

    valid_station_ids = set(station_ids)
    return {
        station_id: positions
        for station_id, positions in grouped_indices.items()
        if station_id in valid_station_ids
    }


def _first_valid_numeric(df, column_name):
    if column_name not in df.columns:
        return np.nan

    numeric_values = pd.to_numeric(df[column_name], errors="coerce").dropna()
    if numeric_values.empty:
        return np.nan
    return float(numeric_values.iloc[0])


def _first_nonempty_text(df, column_name):
    if column_name not in df.columns:
        return ""
    values = df[column_name].dropna()
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def _run_station_task(station_id, station_df, output_dir, verbose):
    # Wrap station processing so the pool always returns a uniform
    # (station_id, result, error) tuple instead of propagating raw exceptions.
    try:
        result = create_netcdf_file(
            station_id,
            station_df,
            output_dir,
            verbose=verbose,
        )
        return station_id, result, None
    except Exception as exc:
        return station_id, None, str(exc)


def _process_station_from_shared_state(task):
    # Fast path for Linux/fork: recover station rows from the global shared
    # DataFrame snapshot inherited by child processes.
    dataset_name, station_id, output_dir, verbose = task
    station_df = _WORKER_DATASETS[dataset_name].take(
        _WORKER_GROUP_INDICES[dataset_name][station_id]
    ).copy()
    return _run_station_task(station_id, station_df, output_dir, verbose)


def _process_station_from_payload(task):
    # Fallback path for platforms that do not share the parent memory image.
    station_id, station_df, output_dir, verbose = task
    return _run_station_task(station_id, station_df, output_dir, verbose)


def _process_station_collection(
    dataset_name,
    source_df,
    group_indices,
    station_ids,
    output_dir,
    num_workers,
    stations_info,
    *,
    stage_label,
    verbose_station_logs=False,
):
    # Each worker only needs a station key. The actual rows are recovered from
    # shared parent-process memory on Linux/fork, with a payload fallback when
    # fork sharing is unavailable.
    total_stations = len(station_ids)
    if total_stations == 0:
        return 0, 0, 0.0

    chunksize = _compute_chunksize(total_stations, num_workers)
    use_shared_state = mp.get_start_method() == "fork"

    if use_shared_state:
        worker_fn = _process_station_from_shared_state
        task_iter = (
            (dataset_name, station_id, output_dir, verbose_station_logs)
            for station_id in station_ids
        )
    else:
        worker_fn = _process_station_from_payload
        task_iter = (
            (
                station_id,
                source_df.take(group_indices[station_id]).copy(),
                output_dir,
                verbose_station_logs,
            )
            for station_id in station_ids
        )

    success = 0
    failed = 0
    started_at = time.perf_counter()
    last_render_at = 0.0

    _render_progress(stage_label, 0, total_stations, started_at, success, failed)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, (station_id, result, error) in enumerate(
            executor.map(worker_fn, task_iter, chunksize=chunksize),
            1,
        ):
            if error is not None:
                sys.stdout.write("\n")
                sys.stdout.flush()
                print(f"  Station {station_id} failed with error: {error}")
                failed += 1
            elif isinstance(result, dict):
                stations_info.append(result)
                success += 1
            else:
                failed += 1

            now = time.perf_counter()
            if (now - last_render_at >= PROGRESS_UPDATE_INTERVAL_SECONDS) or (i == total_stations):
                _render_progress(stage_label, i, total_stations, started_at, success, failed)
                last_render_at = now

    sys.stdout.write("\n")
    sys.stdout.flush()
    return success, failed, time.perf_counter() - started_at


def _normalize_riversed_id(value):
    """Normalize RiverSed reach IDs to stable string keys."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text == "":
        return ""

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return text


def _clean_dbf_value(text):
    """Convert DBF field text to a clean scalar value."""
    value = text.strip()
    if not value:
        return None
    if value.upper() == "NA":
        return None
    if set(value) == {"*"}:
        return None
    return value


def load_riversed_station_metadata(dbf_path):
    """Load RiverSed reach metadata from the modified NHDPlusV2 DBF."""
    print(f"Loading RiverSed metadata from {dbf_path}...")

    with open(dbf_path, "rb") as handle:
        # Read the DBF directly to avoid an extra shapefile/DBF dependency for
        # a one-time lookup table that only needs a handful of fields.
        header = handle.read(32)
        if len(header) < 32:
            raise ValueError("Invalid DBF header in RiverSed metadata file.")

        _, _, _, _, record_count, header_length, record_length = struct.unpack(
            "<BBBBIHH20x", header
        )

        fields = []
        field_count = (header_length - 33) // 32
        for _ in range(field_count):
            descriptor = handle.read(32)
            name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", "ignore")
            fields.append((name, descriptor[16]))

        handle.read(1)  # field descriptor terminator

        field_index = {name: idx for idx, (name, _) in enumerate(fields)}
        missing_fields = sorted(
            set(RIVERSED_METADATA_FIELD_MAP.keys()) - set(field_index.keys())
        )
        if missing_fields:
            raise ValueError(
                "RiverSed metadata DBF is missing fields: {0}".format(
                    ", ".join(missing_fields)
                )
            )

        rows = []
        for _ in range(record_count):
            record = handle.read(record_length)
            if not record:
                break
            if record[0:1] == b"*":
                continue

            pos = 1
            values = {}
            for name, length in fields:
                raw = record[pos:pos + length]
                pos += length
                if name not in RIVERSED_METADATA_FIELD_MAP:
                    continue
                clean_value = _clean_dbf_value(raw.decode("latin1", "ignore"))
                values[RIVERSED_METADATA_FIELD_MAP[name]] = clean_value

            rows.append(values)

    metadata_df = pd.DataFrame(rows)
    if metadata_df.empty:
        raise ValueError("No RiverSed metadata records were loaded from DBF.")

    # Normalize to the same key format used by the RiverSed observation table
    # before checking uniqueness or attempting the metadata join.
    metadata_df["ID"] = metadata_df["ID"].map(_normalize_riversed_id)
    metadata_df = metadata_df[metadata_df["ID"] != ""].copy()

    duplicated_ids = metadata_df["ID"].duplicated()
    if duplicated_ids.any():
        dup_sample = metadata_df.loc[duplicated_ids, "ID"].head(10).tolist()
        raise ValueError(
            "RiverSed metadata DBF contains duplicate IDs: {0}".format(dup_sample)
        )

    metadata_df["upstream_area"] = pd.to_numeric(
        metadata_df["upstream_area"], errors="coerce"
    )

    print(
        "  Loaded {0} metadata rows covering {1} RiverSed IDs".format(
            len(metadata_df), metadata_df["ID"].nunique()
        )
    )
    return metadata_df

def load_aquasat_data(file_path):
    """Load Aquasat TSS data"""
    print(f"Loading Aquasat data from {file_path}...")
    # Only read the columns used later. This reduces startup time and memory
    # footprint for a very large source file.
    required_columns = {"SiteID", "date", "value", "lat", "long", "elevation"}
    df = pd.read_csv(
        file_path,
        low_memory=False,
        usecols=lambda column_name: column_name in required_columns,
    )

    # Convert the raw schema into the common schema used by the rest of the
    # script: station_id / date / tss plus optional metadata columns.
    # Parse date
    df['date'] = pd.to_datetime(df['date'])

    # Rename columns
    df = df.rename(columns={
        'value': 'tss',
        'SiteID': 'station_id'
    })

    # Select relevant columns
    cols = ['station_id', 'date', 'tss', 'lat', 'long', 'elevation']
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['station_id', 'tss'])

    print(f"  Loaded {len(df)} records from {df['station_id'].nunique()} stations")
    return df

def load_riversed_data(file_path, metadata_dbf_path=RIVERSED_METADATA_DBF):
    """Load RiverSed USA data"""
    print(f"Loading RiverSed data from {file_path}...")
    # RiverSed rows themselves only contain the measurement-side information.
    # The reach/basin identity is added later from the modified NHDPlus DBF.
    required_columns = {"ID", "date", "time", "tss", "elevation"}
    df = pd.read_csv(
        file_path,
        low_memory=False,
        usecols=lambda column_name: column_name in required_columns,
    )
    metadata_df = load_riversed_station_metadata(metadata_dbf_path)

    df["ID"] = df["ID"].map(_normalize_riversed_id)

    # This is the core RiverSed workflow:
    #   RiverSed observation ID -> modified NHDPlus reach metadata row
    # Every observation ID must resolve before we allow the export to proceed.
    source_ids = set(df["ID"].dropna()) - {""}
    metadata_ids = set(metadata_df["ID"].dropna()) - {""}
    missing_ids = sorted(source_ids - metadata_ids)
    if missing_ids:
        raise ValueError(
            "RiverSed metadata missing for {0} IDs. Sample: {1}".format(
                len(missing_ids), missing_ids[:10]
            )
        )

    # RiverSed stores date and time separately, so unify them before station
    # grouping and daily aggregation.
    # Combine date and time
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')

    # The merge is many-observations-to-one-reach. It is not a spatial join;
    # the spatial linkage has already been encoded in the modified DBF table.
    df = df.merge(metadata_df, on="ID", how="left", validate="m:1")

    # In the RiverSed branch a "station" in the output is really a reach-level
    # identity derived from the normalized RiverSed ID.
    # Use ID as station_id
    df['station_id'] = 'RiverSed_' + df['ID'].astype(str)

    # Select relevant columns
    cols = [
        'ID',
        'station_id',
        'date',
        'tss',
        'elevation',
        'river_name',
        'comid',
        'reach_code',
        'vpu_id',
        'rpu_id',
        'upstream_area',
    ]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=['tss'])

    print(f"  Loaded {len(df)} records from {df['station_id'].nunique()} stations")
    return df


def apply_satellite_ssc_qc(df, station_id, diagnostic_dir=None):
    """
    QC for satellite-only SSC (TSS) data using tool.py logic.

    Final flag convention:
      0 good, 2 suspect, 3 bad, 9 missing

    Step flags:
      - QC1 physical: 0 pass, 3 bad, 9 missing
      - QC2 log-IQR : 0 pass, 2 suspect, 8 not_checked, 9 missing
      - QC3 SSC–Q   : 0 pass, 2 suspect, 8 not_checked, 9 missing (satellite-only => mostly 8/9)
    """

    # Workflow inside this function:
    # 1. Convert the station's SSC series to a clean 1D NumPy array.
    # 2. Apply physical QC (missing / negative checks).
    # 3. Apply log-IQR outlier screening if there are enough valid samples.
    # 4. Combine step flags into one final SSC flag.
    # 5. Mask bad/missing SSC values before export.
    #
    # ---- force strict 1D (避免 0D / len() 报错) ----
    ssc = np.atleast_1d(np.asarray(df["tss"], dtype=float)).reshape(-1)
    if ssc.size == 0:
        return None

    # -----------------------------
    # QC1. physical feasibility (vectorized)
    # -----------------------------
    fill_value_float = float(FILL_VALUE_FLOAT)
    missing_mask = ~np.isfinite(ssc) | np.isclose(
        ssc,
        fill_value_float,
        rtol=1e-5,
        atol=1e-5,
    )
    ssc_flag_qc1 = np.full(ssc.shape, 0, dtype=np.int8)
    ssc_flag_qc1[missing_mask] = 9
    ssc_flag_qc1[(~missing_mask) & (ssc < 0)] = 3

    # -----------------------------
    # QC2. log-IQR screening
    # -----------------------------
    ssc_flag_qc2 = np.full(ssc.shape, 8, dtype=np.int8)  # default not_checked
    # QC2 only evaluates values that survived QC1 and have enough positive
    # samples to estimate a robust log-space IQR envelope.
    # missing remains 9
    ssc_flag_qc2[ssc_flag_qc1 == 9] = 9

    lower, upper = compute_log_iqr_bounds(ssc)
    if (lower is not None) and (upper is not None) and (ssc.size >= 5):
        # for non-missing, mark pass first
        ssc_flag_qc2[(ssc_flag_qc1 != 9)] = 0
        outlier = (ssc < lower) | (ssc > upper)
        # only mark suspect where it is not missing & not already bad
        ssc_flag_qc2[outlier & (ssc_flag_qc1 != 9)] = 2

    # -----------------------------
    # QC3. SSC–Q consistency (satellite-only => not_checked)
    # -----------------------------
    ssc_flag_qc3 = np.full(ssc.shape, 8, dtype=np.int8)
    ssc_flag_qc3[ssc_flag_qc1 == 9] = 9

    # -----------------------------
    # Final flag combine (QC1 dominates bad/missing; QC2 may set suspect)
    # -----------------------------
    ssc_flag_final = ssc_flag_qc1.copy()  # start with 0/3/9
    # if QC1 pass and QC2 says suspect => final suspect
    ssc_flag_final[(ssc_flag_qc1 == 0) & (ssc_flag_qc2 == 2)] = 2

    # -----------------------------
    # Mask bad & missing values
    # -----------------------------
    ssc_clean = ssc.copy()
    ssc_clean[(ssc_flag_final == 3) | (ssc_flag_final == 9)] = np.nan

    df["tss"] = ssc_clean
    df["SSC_flag"] = ssc_flag_final.astype(np.int8)
    df["SSC_flag_qc1_physical"] = ssc_flag_qc1.astype(np.int8)
    df["SSC_flag_qc2_log_iqr"] = ssc_flag_qc2.astype(np.int8)
    df["SSC_flag_qc3_ssc_q"] = ssc_flag_qc3.astype(np.int8)

    # 至少保留一个非缺测值
    if np.all(np.isnan(df["tss"].values)):
        if diagnostic_dir:
            print(f"  -> All SSC invalid after QC for station {station_id}")
        return None

    return df

def create_netcdf_file(station_id, tss_df, output_dir, *, verbose=False):
    """Create netCDF file following HYBAM format"""

    # Station-level workflow:
    # 1. Validate raw station rows.
    # 2. Aggregate sub-daily observations to daily SSC.
    # 3. Apply SSC QC and compute summary metadata.
    # 4. Write one HYBAM-style netCDF file.
    # 5. Return one station_info dictionary for the summary CSV outputs.

    # Check if all TSS values are NaN
    if tss_df['tss'].isna().all():
        if verbose:
            print(f"  All TSS values are NaN for station {station_id}")
        return None

    # Find time period
    if not pd.api.types.is_datetime64_any_dtype(tss_df['date']):
        tss_df['date'] = pd.to_datetime(tss_df['date'], errors='coerce')
    tss_df = tss_df.dropna(subset=['date'])

    if tss_df.empty:
        if verbose:
            print(f"  No valid dates for station {station_id}")
        return None

    # Collapse multiple same-day satellite observations to one daily SSC value.
    tss_df['date'] = tss_df['date'].dt.floor('D')

    # If multiple scenes or observations fall on the same day, collapse them
    # to a single daily mean because the output product is daily resolution.
    tss_daily = tss_df.groupby('date', as_index=False)['tss'].mean()

    # 只保留有数据的时间点（不补全）
    daily_df = tss_daily.sort_values('date').reset_index(drop=True)

    # After daily aggregation we no longer need the original sub-daily rows for
    # QC. All downstream flags refer to the daily series stored in daily_df.
    # -----------------------------
    # Apply QC using tool.py
    # -----------------------------
    daily_df = apply_satellite_ssc_qc(daily_df, station_id)

    if daily_df is None:
        return None

    actual_start = daily_df['date'].min()
    actual_end = daily_df['date'].max()

    if pd.isna(actual_start) or pd.isna(actual_end):
        temporal_span = ""
    elif actual_start == actual_end:
        temporal_span = actual_start.strftime('%Y-%m-%d')
    else:
        temporal_span = f"{actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}"

    # Normalize all flag columns to compact int8 arrays so they can be written
    # directly to netCDF byte variables without implicit dtype promotion.
    # --- pad flags to int8 (NaN -> 9) ---
    for col in list(daily_df.columns):
        if col.endswith("_flag") or ("flag_qc" in col):
            daily_df[col] = daily_df[col].fillna(FILL_VALUE_INT).astype(np.int8)
    # -----------------------------
    # Print QC summary (station-level)
    # -----------------------------
    n_total = len(daily_df)

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

    sscv, sscf = _repr(daily_df["tss"].values, daily_df["SSC_flag"].values)

    # 这个脚本没有 Q/SSL（都写成 missing），所以这里固定输出 nan/9
    qv, qf = float("nan"), 9
    sslv, sslf = float("nan"), 9

    # 是否跳过 log-IQR（样本<5 或者 bounds=None）
    skipped_log_iqr = (
        "SSC_flag_qc2_log_iqr" not in daily_df.columns
        or not np.any(
            np.isin(
                daily_df["SSC_flag_qc2_log_iqr"].values.astype(np.int8),
                np.array([0, 2], dtype=np.int8),
            )
        )
    )

    if verbose:
        print(f"  ✓ QC summary ({station_id})")
        print(f"    Samples: {n_total}")
        print(f"    Skipped log-IQR: {skipped_log_iqr}")
        print(f"    Q  : {qv:.2f} m3/s (flag={qf})")
        print(f"    SSC: {sscv:.2f} mg/L (flag={sscf})")
        print(f"    SSL: {sslv:.2f} ton/day (flag={sslf})")

    # Metadata columns are station-level after the RiverSed ID join, so taking
    # the first non-null value is enough for file-level attributes/variables.
    # Aquasat only contributes the columns it actually has.
    # Get metadata (use first non-null values)
    latitude = _first_valid_numeric(tss_df, "lat")
    longitude = _first_valid_numeric(tss_df, "long")
    altitude = _first_valid_numeric(tss_df, "elevation")
    upstream_area = _first_valid_numeric(tss_df, "upstream_area")

    river_name = _first_nonempty_text(tss_df, "river_name")
    comid = _first_nonempty_text(tss_df, "comid")
    reach_code = _first_nonempty_text(tss_df, "reach_code")
    vpu_id = _first_nonempty_text(tss_df, "vpu_id")
    rpu_id = _first_nonempty_text(tss_df, "rpu_id")
    has_riversed_metadata = any(
        column in tss_df.columns
        for column in ["river_name", "comid", "reach_code", "vpu_id", "rpu_id", "upstream_area"]
    )

    geographic_coverage_parts = []
    if vpu_id:
        geographic_coverage_parts.append(f"VPUID={vpu_id}")
    if rpu_id:
        geographic_coverage_parts.append(f"RPUID={rpu_id}")
    geographic_coverage = "; ".join(geographic_coverage_parts)

    # Sanitize station_id for filename (replace invalid characters)
    safe_station_id = str(station_id).replace('/', '_').replace('\\', '_').replace(':', '_')

    # Create netCDF file
    output_file = Path(output_dir) / f"RiverSed_{safe_station_id}.nc"

    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # The file layout mirrors the common project convention:
        #   coordinates/time
        #   Q + Q_flag
        #   SSC + SSC_flag
        #   SSL + SSL_flag
        #   optional intermediate SSC QC step flags
        # Create dimensions
        time_dim = ds.createDimension('time', len(daily_df))

        # Create time variable
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time of measurement'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        # Convert dates to days since 1970-01-01
        reference_date = pd.Timestamp('1970-01-01')
        time_var[:] = (daily_df['date'] - reference_date).dt.total_seconds() / 86400.0

        # Create coordinate variables
        lat_var = ds.createVariable('latitude', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = [-90.0, 90.0]
        lat_var[:] = latitude if not np.isnan(latitude) else -9999.0

        lon_var = ds.createVariable('longitude', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = [-180.0, 180.0]
        lon_var[:] = longitude if not np.isnan(longitude) else -9999.0

        alt_var = ds.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station altitude above sea level'
        alt_var.units = 'm'
        alt_var[:] = altitude if not np.isnan(altitude) else -9999.0

        area_var = ds.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        if np.isfinite(upstream_area):
            area_var.comment = 'Upstream drainage area from modified NHDPlusV2 metadata joined by RiverSed reach ID'
            area_var[:] = upstream_area
        else:
            area_var.comment = 'Not available for satellite-derived data'
            area_var[:] = -9999.0

        # Create data variables. Only SSC contains observations here; Q and SSL
        # are written as missing placeholders to preserve the common schema.
        # Create data variables
        Q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
        Q_var.standard_name = 'water_volume_transport_in_river_channel'
        Q_var.long_name = 'river discharge'
        Q_var.units = 'm3 s-1'
        Q_var.coordinates = 'time latitude longitude'
        Q_var.comment = 'Discharge data not available - all values set to missing'
        # Set all discharge to fill value (NaN equivalent)
        Q_var[:] = -9999.0

        Q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        Q_flag_var.long_name = 'quality flag for river discharge'
        Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='b')
        Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        Q_flag_var.comment = 'All set to 9 (missing) - discharge not available for satellite data'
        Q_flag_var[:] = np.full(len(daily_df), FILL_VALUE_INT, dtype=np.int8)

        SSC_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
        SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        SSC_var.long_name = 'suspended sediment concentration'
        SSC_var.units = 'mg L-1'
        SSC_var.coordinates = 'time latitude longitude'
        SSC_var.comment = 'SSC from satellite observations (Aquasat/RiverSed database)'
        SSC_var[:] = daily_df['tss'].fillna(-9999.0).values

        SSC_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
        SSC_flag_var.flag_values = np.array([0, 2, 3, 9], dtype='b')
        SSC_flag_var.flag_meanings = 'good_data suspect_data bad_data missing_data'
        SSC_flag_var.comment = (
            'QC applied using tool.py: physical validity + log-IQR outlier screening. '
            'Satellite-derived SSC only; no SSC–Q consistency check.'
        )
        SSC_flag_var[:] = daily_df['SSC_flag'].values

        SSL_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
        SSL_var.long_name = 'suspended sediment load'
        SSL_var.units = 'ton day-1'
        SSL_var.coordinates = 'time latitude longitude'
        SSL_var.comment = 'Cannot be calculated without discharge data - all values set to missing'
        # Set all sediment load to fill value
        SSL_var[:] = -9999.0

        SSL_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=FILL_VALUE_INT)
        SSL_flag_var.long_name = 'quality flag for suspended sediment load'
        SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='b')
        SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        SSL_flag_var.comment = 'All set to 9 (missing) - cannot be calculated without discharge'
        SSL_flag_var[:] = np.full(len(daily_df), FILL_VALUE_INT, dtype=np.int8)
        # =============================
        # Step / provenance flags (SSC)
        # =============================
        def _add_step_flag(name, values, flag_values, flag_meanings, long_name):
            v = ds.createVariable(name, 'b', ('time',), fill_value=FILL_VALUE_INT)
            v.long_name = long_name
            v.standard_name = 'status_flag'
            v.flag_values = np.array(flag_values, dtype='b')
            v.flag_meanings = flag_meanings
            v.missing_value = np.int8(FILL_VALUE_INT)
            v[:] = np.asarray(values, dtype=np.int8)
            return v

        if "SSC_flag_qc1_physical" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc1_physical",
                daily_df["SSC_flag_qc1_physical"].values,
                flag_values=[0, 3, 9],
                flag_meanings="pass bad missing",
                long_name="QC1 physical flag for suspended sediment concentration",
            )

        if "SSC_flag_qc2_log_iqr" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc2_log_iqr",
                daily_df["SSC_flag_qc2_log_iqr"].values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC2 log-IQR flag for suspended sediment concentration",
            )

        if "SSC_flag_qc3_ssc_q" in daily_df.columns:
            _add_step_flag(
                "SSC_flag_qc3_ssc_q",
                daily_df["SSC_flag_qc3_ssc_q"].values,
                flag_values=[0, 2, 8, 9],
                flag_meanings="pass suspect not_checked missing",
                long_name="QC3 SSC–Q consistency flag for suspended sediment concentration (satellite-only)",
            )

        # Global attributes
        ds.Conventions = 'CF-1.8'
        ds.title = f'RiverSed Satellite-derived TSS Data for Station {station_id}'
        ds.institution = 'University of North Carolina at Chapel Hill'
        ds.source = 'Satellite-derived TSS from Aquasat/RiverSed database'
        ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf.py'
        ds.references = 'Gardner et al. (2021), The color of rivers, Geophysical Research Letters, doi:10.1029/2020GL088946'
        ds.comment = 'TSS values derived from Landsat satellite imagery. Discharge and sediment load are not available and set to missing values.'
        ds.station_id = str(station_id)
        ds.data_period_start = actual_start.strftime('%Y-%m-%d')
        ds.data_period_end = actual_end.strftime('%Y-%m-%d')
        if has_riversed_metadata:
            ds.river_name = river_name
            ds.comid = comid
            ds.reach_code = reach_code
            ds.vpu_id = vpu_id
            ds.rpu_id = rpu_id

    # Build a compact station-level summary row from the same in-memory data
    # used for the netCDF. This avoids re-reading the source data later.
    # --- build station_info for CSV outputs ---
    n = len(daily_df)

    def _cnt(arr, v):
        a = np.asarray(arr, dtype=np.int8)
        return int(np.sum(a == np.int8(v)))

    # final flags
    ssc_f = daily_df["SSC_flag"].values.astype(np.int8)

    station_info = {
        # metadata summary fields
        "station_name": str(station_id).replace("_", " "),
        "Source_ID": str(station_id),
        "river_name": river_name,
        "comid": comid,
        "reach_code": reach_code,
        "vpu_id": vpu_id,
        "rpu_id": rpu_id,
        "longitude": float(longitude) if not np.isnan(longitude) else np.nan,
        "latitude": float(latitude) if not np.isnan(latitude) else np.nan,
        "altitude": float(altitude) if not np.isnan(altitude) else np.nan,
        "upstream_area": float(upstream_area) if np.isfinite(upstream_area) else np.nan,
        "Data Source Name": "RiverSed / Aquasat (satellite-derived TSS)",
        "Type": "Satellite",
        "Temporal Resolution": "daily",
        "Temporal Span": temporal_span,
        "Variables Provided": "SSC",
        "Geographic Coverage": geographic_coverage,
        "Reference/DOI": "Gardner et al. (2021) doi:10.1029/2020GL088946",

        # QC summary fields (QC results CSV会自动挑存在的列)
        "QC_n_days": int(n),

        "SSC_final_good": _cnt(ssc_f, 0),
        "SSC_final_estimated": _cnt(ssc_f, 1),
        "SSC_final_suspect": _cnt(ssc_f, 2),
        "SSC_final_bad": _cnt(ssc_f, 3),
        "SSC_final_missing": _cnt(ssc_f, 9),
    }

    # step flags counts (如果列存在就加进去)
    if "SSC_flag_qc1_physical" in daily_df.columns:
        f = daily_df["SSC_flag_qc1_physical"].values.astype(np.int8)
        station_info.update({
            "SSC_qc1_pass": _cnt(f, 0),
            "SSC_qc1_bad": _cnt(f, 3),
            "SSC_qc1_missing": _cnt(f, 9),
        })

    if "SSC_flag_qc2_log_iqr" in daily_df.columns:
        f = daily_df["SSC_flag_qc2_log_iqr"].values.astype(np.int8)
        station_info.update({
            "SSC_qc2_pass": _cnt(f, 0),
            "SSC_qc2_suspect": _cnt(f, 2),
            "SSC_qc2_not_checked": _cnt(f, 8),
            "SSC_qc2_missing": _cnt(f, 9),
        })

    if "SSC_flag_qc3_ssc_q" in daily_df.columns:
        f = daily_df["SSC_flag_qc3_ssc_q"].values.astype(np.int8)
        station_info.update({
            "SSC_qc3_pass": _cnt(f, 0),
            "SSC_qc3_suspect": _cnt(f, 2),
            "SSC_qc3_not_checked": _cnt(f, 8),
            "SSC_qc3_missing": _cnt(f, 9),
        })

    if verbose:
        print(f"  Created {output_file}")
    return station_info


def main():
    total_started_at = time.perf_counter()

    # End-to-end workflow in main():
    # 1. Resolve input/output paths and runtime settings.
    # 2. Load Aquasat and RiverSed tables.
    # 3. Build station group indices for parallel processing.
    # 4. Process Aquasat stations in parallel.
    # 5. Process RiverSed reaches/stations in parallel.
    # 6. Write station summary CSV products.
    # 7. Print overall counts and timing diagnostics.

    # Configuration with WSL absolute paths
    aquasat_file = os.path.join(SOURCE_DIR, 'Aquasat_TSS_v1.1.csv')
    riversed_file = os.path.join(SOURCE_DIR, 'RiverSed_USA_V1.1.txt')
    riversed_metadata_dbf = RIVERSED_METADATA_DBF
    output_nc_dir = OUTPUT_NC_DIR
    output_qc_dir = OUTPUT_QC_DIR

    num_workers = _resolve_num_workers()
    verbose_station_logs = VERBOSE_STATION_LOGS

    Path(output_nc_dir).mkdir(parents=True, exist_ok=True)
    Path(output_qc_dir).mkdir(parents=True, exist_ok=True)

    stations_info = []   # 用于两个CSV（summary + qc_results）
    timing_stats = {}

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    # Stage 1: load the two source tables and attach RiverSed metadata.
    load_started_at = time.perf_counter()
    aquasat_df = load_aquasat_data(aquasat_file)
    riversed_df = load_riversed_data(riversed_file, metadata_dbf_path=riversed_metadata_dbf)
    timing_stats["load_data"] = time.perf_counter() - load_started_at

    # Stage 2: precompute station membership once. These indices are the bridge
    # between the monolithic source tables and the station-level worker tasks.
    group_started_at = time.perf_counter()
    # Build station-to-row lookups once. These indices drive both the parallel
    # scheduler and the RiverSed minimum-observation filter.
    aquasat_group_indices = _build_station_group_indices(aquasat_df)
    aquasat_stations = list(aquasat_group_indices.keys())

    riversed_all_group_indices = _build_station_group_indices(riversed_df)
    # RiverSed exports only reaches with at least 5 original observations so
    # the daily series and QC have a minimally useful sample size.
    riversed_stations = [
        station_id
        for station_id, positions in riversed_all_group_indices.items()
        if len(positions) >= 5
    ]
    riversed_group_indices = {
        station_id: riversed_all_group_indices[station_id]
        for station_id in riversed_stations
    }

    global _WORKER_DATASETS, _WORKER_GROUP_INDICES
    # Child workers recover station slices from shared in-memory tables instead
    # of receiving a pickled DataFrame for each submitted station.
    _WORKER_DATASETS = {
        "aquasat": aquasat_df,
        "riversed": riversed_df,
    }
    _WORKER_GROUP_INDICES = {
        "aquasat": aquasat_group_indices,
        "riversed": riversed_group_indices,
    }
    timing_stats["build_station_groups"] = time.perf_counter() - group_started_at

    # Stage 3: process the two datasets separately but with the same station-
    # level export logic so the downstream netCDF/CSV products stay consistent.
    # Process each dataset separately
    print("\n" + "="*80)
    print("PROCESSING AQUASAT STATIONS")
    print("="*80)
    print(f"Processing {len(aquasat_stations)} Aquasat stations...")
    print(f"Using {num_workers} parallel workers...")
    aquasat_success, aquasat_failed, timing_stats["process_aquasat"] = _process_station_collection(
        "aquasat",
        aquasat_df,
        aquasat_group_indices,
        aquasat_stations,
        output_nc_dir,
        num_workers,
        stations_info,
        stage_label="Aquasat",
        verbose_station_logs=verbose_station_logs,
    )

    print("\n" + "="*80)
    print("PROCESSING RIVERSED STATIONS")
    print("="*80)
    print(f"Processing {len(riversed_stations)} RiverSed stations (with at least 5 observations)...")
    print(f"Using {num_workers} parallel workers...")
    riversed_success, riversed_failed, timing_stats["process_riversed"] = _process_station_collection(
        "riversed",
        riversed_df,
        riversed_group_indices,
        riversed_stations,
        output_nc_dir,
        num_workers,
        stations_info,
        stage_label="RiverSed",
        verbose_station_logs=verbose_station_logs,
    )

    # Stage 4: write aggregated station-level summary products after both
    # datasets have finished. stations_info already contains one dict per file.
    # -----------------------------
    # Generate CSV outputs (summary + QC results)
    # -----------------------------
    csv_started_at = time.perf_counter()
    if stations_info:
        csv_summary = os.path.join(OUTPUT_QC_DIR, "RiverSed_station_summary.csv")
        csv_qc = os.path.join(OUTPUT_QC_DIR, "RiverSed_qc_results_summary.csv")

        generate_csv_summary_tool(stations_info, csv_summary)
        generate_qc_results_csv_tool(stations_info, csv_qc)
    timing_stats["write_csv"] = time.perf_counter() - csv_started_at
    timing_stats["total"] = time.perf_counter() - total_started_at

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Aquasat:")
    print(f"  Total stations: {len(aquasat_stations)}")
    print(f"  Successfully created: {aquasat_success}")
    print(f"  Failed (all NaN or no data): {aquasat_failed}")
    print(f"\nRiverSed:")
    print(f"  Total stations (with ≥5 obs): {len(riversed_stations)}")
    print(f"  Successfully created: {riversed_success}")
    print(f"  Failed (all NaN or no data): {riversed_failed}")
    print(f"\nTotal netCDF files created: {aquasat_success + riversed_success}")
    print(f"\nTiming:")
    print(f"  Load data: {_format_duration(timing_stats['load_data'])}")
    print(f"  Build groups: {_format_duration(timing_stats['build_station_groups'])}")
    print(f"  Aquasat processing: {_format_duration(timing_stats['process_aquasat'])}")
    print(f"  RiverSed processing: {_format_duration(timing_stats['process_riversed'])}")
    print(f"  Write CSV: {_format_duration(timing_stats['write_csv'])}")
    print(f"  Total runtime: {_format_duration(timing_stats['total'])}")
    print(f"Output directory: {output_nc_dir}/")
    print("="*80)

if __name__ == '__main__':
    main()
