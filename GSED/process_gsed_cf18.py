#!/usr/bin/env python3
"""
Process GSED monthly SSC data to CF-1.8 and ACDD-1.3 compliant netCDF format
- Implements quality control flags
- Follows CF-1.8 conventions for metadata
- Generates station summary CSV file
- Only includes stations with valid data

Author: Zhongwang Wei
Email: weizhw6@mail.sysu.edu.cn
Institution: Sun Yat-sen University, China
Date: 2025-10-26
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import struct
import warnings
warnings.filterwarnings('ignore')
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
)
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_quality_flag,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.runtime import resolve_output_root, resolve_source_root
from code.units import convert_ssl_units_if_needed

# Quality flag definitions
FLAG_GOOD = 0       # Good data
FLAG_ESTIMATED = 1  # Estimated data
FLAG_SUSPECT = 2    # Suspect data (e.g., extreme values)
FLAG_BAD = 3        # Bad data (e.g., negative values)
FLAG_MISSING = 9    # Missing in source

GSED_AREA_LOOKUP_FILENAMES = (
    'GSED_Reach_upstream_area.csv',
    'GSED_Reach_upstream_area.tsv',
    'GSED_Reach_upstream_area.txt',
    'GSED_Reach_upstream_area.xlsx',
    'GSED_Reach_upstream_area.xls',
)
GSED_AREA_ID_COLUMNS = (
    'R_ID',
    'r_id',
    'RID',
    'reach_id',
    'reach_code',
    'station_id',
    'Source_ID',
)
GSED_AREA_VALUE_COLUMNS = (
    'upstream_area_km2',
    'upstream_area',
    'drainage_area_km2',
    'drainage_area',
    'basin_area',
    'catchment_area',
    'uparea_km2',
    'uparea',
)
GSED_AREA_ACCEPT_COLUMNS = (
    'merit_lookup_accept',
    'lookup_accept',
    'accept',
)

def process_one_reach(task):
    idx, total, r_id, ssc_data, r_id_str, reach_meta, time_array, output_dir = task

    try:
        if (idx + 1) % 100 == 0:
            print(f"\nProcessing reach {r_id} ({idx+1}/{total})...")

        reach_meta = dict(reach_meta) if reach_meta is not None else {}
        reach_meta.setdefault('r_level', None)
        reach_meta.setdefault('reach_length_m', None)
        reach_meta.setdefault('latitude', None)
        reach_meta.setdefault('longitude', None)

        if pd.isna(reach_meta.get('latitude')) or pd.isna(reach_meta.get('longitude')):
            print(f"  Warning: Could not extract coordinates for R_ID {r_id}")

        stats = create_netcdf(r_id, ssc_data, time_array, reach_meta, output_dir)
        if stats is not None:
            return ("success", stats)
        return ("skip", None)

    except Exception as e:
        return ("error", f"R_ID {r_id}: {e}")

def _normalize_gsed_rid(value):
    """
    Normalize GSED reach IDs to a stable string representation.

    Args:
        value: Raw R_ID value from CSV/DBF

    Returns:
        str: Reach ID without scientific notation
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _derive_basin_info_from_rid(r_id):
    """
    Recover hierarchical basin prefixes from the public GSED R_ID code.

    The public GSED release only exposes R_ID/R_level in the shapefile.
    The original intermediary catchment fields were removed before export,
    so we preserve the recoverable basin hierarchy through R_ID prefixes.
    """
    r_id_str = _normalize_gsed_rid(r_id)
    if r_id_str is None:
        return {
            'r_id_str': None,
            'basin_code_l1': None,
            'basin_code_l2': None,
            'basin_code_l3': None,
            'basin_code_l4': None,
        }

    return {
        'r_id_str': r_id_str,
        'basin_code_l1': r_id_str[:2] if len(r_id_str) >= 2 else r_id_str,
        'basin_code_l2': r_id_str[:5] if len(r_id_str) >= 5 else r_id_str,
        'basin_code_l3': r_id_str[:8] if len(r_id_str) >= 8 else r_id_str,
        'basin_code_l4': r_id_str,
    }


def _read_gsed_dbf_records(dbf_path):
    """
    Read the minimal attribute table needed from GSED_Reach.dbf.

    Returns records in shapefile order so they can be paired with .shp
    geometries without any external GIS dependency.
    """
    records = []
    with open(dbf_path, 'rb') as handle:
        header = handle.read(32)
        if len(header) < 32:
            raise ValueError(f"Invalid DBF header: {dbf_path}")

        record_count = struct.unpack('<I', header[4:8])[0]
        record_length = struct.unpack('<H', header[10:12])[0]

        fields = []
        while True:
            first = handle.read(1)
            if not first:
                break
            if first == b'\r':
                break
            descriptor = first + handle.read(31)
            name = descriptor[:11].split(b'\x00', 1)[0].decode('ascii', 'ignore')
            fields.append((name, descriptor[11:12].decode('ascii', 'ignore'), descriptor[16], descriptor[17]))

        for _ in range(record_count):
            record = handle.read(record_length)
            if not record:
                break

            deleted = record[:1] == b'*'
            row = {'_deleted': deleted}
            offset = 1

            for name, field_type, length, decimals in fields:
                raw = record[offset:offset + length]
                offset += length
                text = raw.decode('latin1', 'ignore').strip()

                if name not in {'R_ID', 'R_level', 'Length'}:
                    continue

                if not text:
                    row[name] = None
                    continue

                if field_type in {'N', 'F'}:
                    try:
                        value = float(text)
                        if decimals == 0:
                            value = int(value)
                        row[name] = value
                    except ValueError:
                        row[name] = text
                else:
                    row[name] = text

            records.append(row)

    return records


def _extract_polyline_parts(record_content):
    """Parse a PolyLine shapefile record into ordered point parts."""
    if len(record_content) < 44:
        return []

    shape_type = struct.unpack('<i', record_content[:4])[0]
    if shape_type == 0:
        return []

    if shape_type != 3:
        raise ValueError(f"Unsupported shapefile geometry type: {shape_type}")

    num_parts = struct.unpack('<i', record_content[36:40])[0]
    num_points = struct.unpack('<i', record_content[40:44])[0]
    points_offset = 44 + 4 * num_parts
    points_end = points_offset + num_points * 16

    if points_end > len(record_content):
        raise ValueError("Corrupted polyline record in shapefile.")

    if num_points == 0:
        return []

    part_starts_offset = 44
    part_starts = [
        struct.unpack('<i', record_content[part_starts_offset + i * 4: part_starts_offset + (i + 1) * 4])[0]
        for i in range(num_parts)
    ]
    if not part_starts:
        part_starts = [0]
    part_starts.append(num_points)

    points = []
    for i in range(num_points):
        x, y = struct.unpack('<2d', record_content[points_offset + i * 16: points_offset + (i + 1) * 16])
        points.append((float(x), float(y)))

    parts = []
    for part_start, part_end in zip(part_starts[:-1], part_starts[1:]):
        if part_end <= part_start:
            continue
        parts.append(points[part_start:part_end])

    return parts


def _extract_polyline_midpoint(record_content):
    """Extract the 50%-along-line midpoint from a polyline shapefile record."""
    parts = _extract_polyline_parts(record_content)
    if not parts:
        return None, None

    fallback_point = None
    total_length = 0.0
    for part_points in parts:
        if not part_points:
            continue
        if fallback_point is None:
            fallback_point = part_points[0]
        for (lon0, lat0), (lon1, lat1) in zip(part_points[:-1], part_points[1:]):
            total_length += float(np.hypot(lon1 - lon0, lat1 - lat0))

    if fallback_point is None:
        return None, None

    if total_length <= 0.0:
        fallback_lon, fallback_lat = fallback_point
        return float(fallback_lat), float(fallback_lon)

    midpoint_distance = total_length / 2.0
    traversed = 0.0
    last_point = fallback_point

    for part_points in parts:
        if not part_points:
            continue
        last_point = part_points[-1]
        for (lon0, lat0), (lon1, lat1) in zip(part_points[:-1], part_points[1:]):
            segment_length = float(np.hypot(lon1 - lon0, lat1 - lat0))
            if segment_length <= 0.0:
                continue

            next_traversed = traversed + segment_length
            if next_traversed >= midpoint_distance:
                ratio = (midpoint_distance - traversed) / segment_length
                midpoint_lon = lon0 + ratio * (lon1 - lon0)
                midpoint_lat = lat0 + ratio * (lat1 - lat0)
                return float(midpoint_lat), float(midpoint_lon)
            traversed = next_traversed

    last_lon, last_lat = last_point
    return float(last_lat), float(last_lon)


def load_gsed_reach_metadata(shapefile_path, target_rids=None):
    """
    Load GSED reach geometry and basin metadata once.

    Args:
        shapefile_path: Path to GSED_Reach.shp
        target_rids: Optional set of reach IDs to keep

    Returns:
        dict: {R_ID string: metadata dict}
    """
    shapefile_path = Path(shapefile_path)
    dbf_records = _read_gsed_dbf_records(shapefile_path.with_suffix('.dbf'))
    target_rids = {_normalize_gsed_rid(r_id) for r_id in target_rids} if target_rids else None
    metadata = {}

    try:
        with open(shapefile_path, 'rb') as handle:
            handle.read(100)  # shapefile header

            for record_index, dbf_row in enumerate(dbf_records):
                record_header = handle.read(8)
                if not record_header:
                    break

                content_length_words = struct.unpack('>i', record_header[4:8])[0]
                content = handle.read(content_length_words * 2)

                if dbf_row.get('_deleted'):
                    continue

                r_id_str = _normalize_gsed_rid(dbf_row.get('R_ID'))
                if r_id_str is None:
                    continue

                if target_rids and r_id_str not in target_rids:
                    continue

                lat, lon = _extract_polyline_midpoint(content)
                basin_info = _derive_basin_info_from_rid(r_id_str)
                metadata[r_id_str] = {
                    'r_id_str': r_id_str,
                    'r_level': int(dbf_row['R_level']) if dbf_row.get('R_level') is not None else None,
                    'reach_length_m': float(dbf_row['Length']) if dbf_row.get('Length') is not None else None,
                    'latitude': lat,
                    'longitude': lon,
                    'midpoint_latitude': lat,
                    'midpoint_longitude': lon,
                    **basin_info,
                }

        return metadata
    except Exception as e:
        print(f"Error loading GSED reach metadata from {shapefile_path}: {e}")
        return metadata


def find_gsed_area_lookup_file(source_dir):
    """
    Locate an optional external lookup table that maps R_ID to upstream area.

    Search order:
    1. Environment variable GSED_UPSTREAM_AREA_FILE
    2. Standard filenames in the nested GSED source folder
    3. Standard filenames in the parent GSED source folder
    """
    env_path = os.environ.get('GSED_UPSTREAM_AREA_FILE')
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate
        print(f"Warning: GSED_UPSTREAM_AREA_FILE not found: {candidate}")

    search_roots = [Path(source_dir), Path(source_dir).parent]
    for root in search_roots:
        for filename in GSED_AREA_LOOKUP_FILENAMES:
            candidate = root / filename
            if candidate.exists():
                return candidate

    return None


def _read_gsed_area_table(file_path):
    """Read a CSV/TSV/TXT/Excel lookup table with pandas."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(file_path)
    if suffix == '.tsv':
        return pd.read_csv(file_path, sep='\t')
    if suffix == '.txt':
        return pd.read_csv(file_path, sep=None, engine='python')
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported lookup table format: {file_path}")


def load_gsed_area_lookup(file_path, target_rids=None):
    """
    Load an external R_ID -> upstream_area_km2 table.

    The table is expected to provide at least one ID column and one area
    column. Area values are interpreted as km2 because downstream basin
    scripts expect reported_area in square kilometres.
    """
    df = _read_gsed_area_table(file_path)
    if df.empty:
        raise ValueError(f"Area lookup table is empty: {file_path}")

    id_col = next((col for col in GSED_AREA_ID_COLUMNS if col in df.columns), None)
    if id_col is None:
        raise ValueError(
            f"Area lookup table {file_path} is missing an R_ID column. "
            f"Supported names: {', '.join(GSED_AREA_ID_COLUMNS)}"
        )

    area_col = next((col for col in GSED_AREA_VALUE_COLUMNS if col in df.columns), None)
    if area_col is None:
        raise ValueError(
            f"Area lookup table {file_path} is missing an upstream area column. "
            f"Supported names: {', '.join(GSED_AREA_VALUE_COLUMNS)}"
        )

    area_df = df[[id_col, area_col]].copy()
    accept_col = next((col for col in GSED_AREA_ACCEPT_COLUMNS if col in df.columns), None)
    if accept_col is not None:
        accept_mask = (
            df[accept_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({'1', 'true', 'yes', 'y', 't'})
        )
        area_df = area_df.loc[accept_mask].copy()
    area_df['r_id_str'] = area_df[id_col].map(_normalize_gsed_rid)
    area_df['upstream_area_km2'] = pd.to_numeric(area_df[area_col], errors='coerce')
    area_df = area_df.dropna(subset=['r_id_str', 'upstream_area_km2'])

    if target_rids is not None:
        area_df = area_df[area_df['r_id_str'].isin(target_rids)]

    duplicate_count = int(area_df['r_id_str'].duplicated(keep='first').sum())
    if duplicate_count:
        print(
            f"Warning: {duplicate_count} duplicate R_ID rows found in "
            f"{file_path.name}; keeping the first non-missing value."
        )
        area_df = area_df.drop_duplicates(subset=['r_id_str'], keep='first')

    return {
        row['r_id_str']: {
            'upstream_area_km2': float(row['upstream_area_km2']),
            'upstream_area_source': str(Path(file_path).name),
        }
        for _, row in area_df.iterrows()
    }


def merge_gsed_area_lookup(reach_metadata, area_lookup, target_rids=None):
    """
    Merge an external upstream-area lookup into the reach metadata mapping.
    """
    target_rids = set(target_rids) if target_rids is not None else None
    attached_count = 0

    for r_id_str, area_meta in area_lookup.items():
        if target_rids is not None and r_id_str not in target_rids:
            continue

        reach_meta = reach_metadata.setdefault(r_id_str, _derive_basin_info_from_rid(r_id_str))
        reach_meta.setdefault('r_level', None)
        reach_meta.setdefault('reach_length_m', None)
        reach_meta.setdefault('latitude', None)
        reach_meta.setdefault('longitude', None)
        reach_meta.update(area_meta)
        attached_count += 1

    return attached_count

def create_time_array(start_year=1985, start_month=1, n_months=432):
    """
    Create time array in days since 1970-01-01 for monthly data

    Args:
        start_year: Starting year (default: 1985)
        start_month: Starting month (default: 1)
        n_months: Number of months (default: 432, i.e., 1985-2020)

    Returns:
        numpy array: Days since 1970-01-01 for each month
    """
    base_date = datetime(1970, 1, 1)
    times = []

    for i in range(n_months):
        year = start_year + (start_month - 1 + i) // 12
        month = (start_month - 1 + i) % 12 + 1
        current_date = datetime(year, month, 1)
        days_since = (current_date - base_date).days
        times.append(days_since)

    return np.array(times)

def get_year_month_from_index(start_year, start_month, index):
    """
    Convert time index to year and month

    Args:
        start_year: Starting year
        start_month: Starting month
        index: Time index

    Returns:
        tuple: (year, month)
    """
    year = start_year + (start_month - 1 + index) // 12
    month = (start_month - 1 + index) % 12 + 1
    return year, month

def apply_gsed_qc_with_tool(ssc):
    """
    Apply unified QC from tool.py for GSED SSC-only dataset.

    QC steps:
    1) physical plausibility (apply_quality_flag)
    2) log-IQR outlier detection (compute_log_iqr_bounds)

    Returns
    -------
    ssc_qc : array
        SSC values (bad values set to NaN)
    ssc_flag : array (int8)
        QC flags
    """

    n = len(ssc)

    ssc_flag = np.full(n, FILL_VALUE_INT, dtype=np.int8)
    ssc_qc = ssc.astype(float).copy()

    # -----------------------------
    # Counters
    # -----------------------------
    n_missing = 0
    n_bad = 0
    n_suspect = 0
    n_good = 0

    # --------------------------------------------------
    # 1) Physical QC
    # --------------------------------------------------
    for i in range(n):
        ssc_flag[i] = apply_quality_flag(ssc_qc[i], variable_name="SSC")

        if ssc_flag[i] == FLAG_MISSING:
            n_missing += 1
        elif ssc_flag[i] == FLAG_BAD:
            n_bad += 1
            ssc_qc[i] = np.nan
        elif ssc_flag[i] == FLAG_GOOD:
            pass
    # -----------------------------
    # 2) Log-IQR QC
    # -----------------------------
    lower, upper = compute_log_iqr_bounds(ssc_qc)

    if lower is not None:
        outlier = (
            (ssc_qc < lower) | (ssc_qc > upper)
        ) & (ssc_flag == FLAG_GOOD)

        ssc_flag[outlier] = FLAG_SUSPECT
        n_suspect = np.sum(outlier)

    # -----------------------------
    # Final GOOD count
    # -----------------------------
    n_good = np.sum(ssc_flag == FLAG_GOOD)

    qc_stats = {
        "n_total": n,
        "n_missing": int(n_missing),
        "n_bad": int(n_bad),
        "n_suspect": int(n_suspect),
        "n_good": int(n_good),
    }

    return ssc_qc, ssc_flag, qc_stats


def find_data_period(ssc_data, flags):
    """
    Find the period where SSC data exists (not all NaN or bad)

    Args:
        ssc_data: Array of SSC values
        flags: Array of quality flags

    Returns:
        tuple: (start_idx, end_idx) or (None, None) if no valid data
    """
    # Valid data means not missing and not bad
    valid_mask = (flags != FLAG_MISSING) & (flags != FLAG_BAD)

    if not np.any(valid_mask):
        return None, None

    # Find first and last valid index
    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1  # +1 to include the last month

    return start_idx, end_idx

def create_netcdf(r_id, ssc_data, time_array, reach_meta, output_dir):
    """
    Create CF-1.8 compliant netCDF file for a single station

    Args:
        r_id: Reach ID
        ssc_data: Array of SSC values
        time_array: Array of time values
        reach_meta: Reach metadata dictionary
        output_dir: Output directory path

    Returns:
        dict: Statistics for the station (for CSV summary)
    """
    lat = reach_meta.get('latitude')
    lon = reach_meta.get('longitude')
    length = reach_meta.get('reach_length_m')
    reach_level = reach_meta.get('r_level')
    basin_code_l1 = reach_meta.get('basin_code_l1')
    basin_code_l2 = reach_meta.get('basin_code_l2')
    basin_code_l3 = reach_meta.get('basin_code_l3')
    basin_code_l4 = reach_meta.get('basin_code_l4')
    upstream_area_km2 = reach_meta.get('upstream_area_km2')
    upstream_area_source = reach_meta.get('upstream_area_source')

    # Apply QC and get flags
    ssc_qc, flags, qc_stats = apply_gsed_qc_with_tool(ssc_data)

    print(
        f"Reach {r_id} QC summary:\n"
        f"  total samples    : {qc_stats['n_total']}\n"
        f"  missing (flag=9) : {qc_stats['n_missing']}\n"
        f"  bad (flag=3)     : {qc_stats['n_bad']}\n"
        f"  suspect (flag=2) : {qc_stats['n_suspect']}\n"
        f"  good (flag=0)    : {qc_stats['n_good']}"
    )

    # Find data period
    start_idx, end_idx = find_data_period(ssc_qc, flags)

    if start_idx is None:
        print(f"Reach {r_id}: No valid SSC data, skipping...")
        return None

    trimmed = qc_stats['n_total'] - (end_idx - start_idx)

    print(
        f"  trimmed (no data at edges): {trimmed}\n"
        f"  retained for output       : {end_idx - start_idx}"
    )

    # Subset data to valid period
    ssc_subset = ssc_qc[start_idx:end_idx]
    flags_subset = flags[start_idx:end_idx]
    time_subset = time_array[start_idx:end_idx]
    n_times = len(time_subset)

    # Calculate statistics for CSV
    start_year, start_month = get_year_month_from_index(1985, 1, start_idx)
    end_year, end_month = get_year_month_from_index(1985, 1, end_idx - 1)

    # Count good data points
    good_count = np.sum(flags_subset == FLAG_GOOD)
    percent_complete = (good_count / n_times) * 100.0

    print(f"Reach {r_id}: {n_times} months ({start_year}-{start_month:02d} to {end_year}-{end_month:02d}), "
          f"{good_count} good ({percent_complete:.1f}%)")

    # Create output filename
    output_file = output_dir / f"GSED_{int(r_id)}.nc"

    # Create netCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # ===== Dimensions =====
        time_dim = ds.createDimension('time', n_times)

        # ===== Coordinate Variables =====
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'
        time_var[:] = time_subset

        # Latitude (scalar)
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
        lat_var[:] = lat if lat is not None else np.nan

        # Longitude (scalar)
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
        lon_var[:] = lon if lon is not None else np.nan

        # Optional drainage area metadata. scripts_basin_test reads this field
        # as reported_area when it exists.
        if upstream_area_km2 is not None and pd.notna(upstream_area_km2):
            area_var = ds.createVariable('upstream_area', 'f4')
            area_var.long_name = 'upstream drainage area'
            area_var.units = 'km2'
            area_var.comment = 'Upstream drainage area used as reported_area by scripts_basin_test.'
            area_var[:] = float(upstream_area_km2)

        # ===== Data Variables =====
        # Q (Discharge) - Not available in GSED
        q_var = ds.createVariable('Q', 'f4', ('time',),
                                  fill_value=-9999.0, zlib=True, complevel=4)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'lat lon'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Not available in GSED dataset (satellite-derived SSC only).'
        q_var[:] = -9999.0

        # Q Quality Flag
        q_flag_var = ds.createVariable('Q_flag', 'i1', ('time',),
                                       fill_value=np.int8(-128), zlib=True, complevel=4)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        q_flag_var[:] = FLAG_MISSING

        # SSC
        ssc_var = ds.createVariable('SSC', 'f4', ('time',),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'lat lon'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Satellite-derived monthly suspended sediment concentration from GSED dataset. Zhang et al. (2023). Scientific Data.'
        # Replace NaN with fill value
        ssc_filled = np.where(np.isnan(ssc_subset), -9999.0, ssc_subset)
        ssc_var[:] = ssc_filled

        # SSC Quality Flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'i1', ('time',),
                                         fill_value=np.int8(-128), zlib=True, complevel=4)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., extreme >3000 mg/L), 3=Bad (e.g., negative), 9=Missing in source.'
        ssc_flag_var[:] = flags_subset

        # SSL (Sediment Load) - Not available in GSED
        ssl_var = ds.createVariable('SSL', 'f4', ('time',),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'lat lon'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Not available in GSED dataset. Cannot be calculated without discharge data.'
        ssl_var[:] = -9999.0

        # SSL Quality Flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'i1', ('time',),
                                         fill_value=np.int8(-128), zlib=True, complevel=4)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([FLAG_GOOD, FLAG_ESTIMATED, FLAG_SUSPECT, FLAG_BAD, FLAG_MISSING], dtype=np.int8)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssl_flag_var[:] = FLAG_MISSING

        # ===== Global Attributes (CF-1.8 & ACDD-1.3) =====
        ds.Conventions = 'CF-1.8, ACDD-1.3'
        ds.title = 'Harmonized Global River Discharge and Sediment'
        ds.summary = f'Satellite-derived monthly suspended sediment concentration for reach {int(r_id)} from the GSED dataset. Data covers {start_year}-{start_month:02d} to {end_year}-{end_month:02d}. Quality flags indicate data reliability.'

        # Data Source Information
        ds.data_source_name = 'GSED Dataset'
        ds.Source_ID = str(int(r_id))
        ds.station_id = str(int(r_id))
        ds.station_name = str(int(r_id))
        ds.reach_id = str(int(r_id))
        ds.source = 'Satellite station'
        ds.Type = 'Satellite'
        if pd.notna(reach_level):
            ds.reach_level = int(reach_level)
        if basin_code_l1:
            ds.basin_code_l1 = basin_code_l1
        if basin_code_l2:
            ds.basin_code_l2 = basin_code_l2
        if basin_code_l3:
            ds.basin_code_l3 = basin_code_l3
        if basin_code_l4:
            ds.basin_code_l4 = basin_code_l4
            ds.reach_code = basin_code_l4
        if basin_code_l1:
            ds.vpu_id = basin_code_l1
        if basin_code_l3:
            ds.rpu_id = basin_code_l3
        ds.basin_info_note = (
            'Public GSED source data exposes basin hierarchy through R_ID and '
            'R_level. Explicit basin-name/catchment-name fields are not included '
            'in the released shapefile.'
        )
        ds.source_reach_hierarchy_note = (
            'reach_code uses the original public GSED R_ID; vpu_id and rpu_id '
            'store coarse and intermediate basin prefixes derived from that R_ID '
            'so downstream basin scripts can retain GSED basin hierarchy without '
            'custom code changes.'
        )

        # Temporal Information
        ds.temporal_resolution = 'monthly'
        ds.time_coverage_start = f'{start_year}-{start_month:02d}-01'
        ds.time_coverage_end = f'{end_year}-{end_month:02d}-01'
        ds.temporal_span = f'{start_year}-{start_month:02d} to {end_year}-{end_month:02d}'

        # Spatial Information
        if lat is not None and lon is not None:
            ds.geospatial_lat_min = float(lat)
            ds.geospatial_lat_max = float(lat)
            ds.geospatial_lon_min = float(lon)
            ds.geospatial_lon_max = float(lon)
            ds.geographic_coverage = f'River reach midpoint at ({lat:.4f}°N, {lon:.4f}°E)'

        if length is not None:
            ds.reach_length_m = float(length)
        if upstream_area_km2 is not None and pd.notna(upstream_area_km2):
            ds.upstream_area = float(upstream_area_km2)
            if upstream_area_source:
                ds.upstream_area_source = upstream_area_source

        # Variables
        ds.variables_provided = 'Q, SSC, SSL'
        ds.number_of_data = '1'

        # References
        ds.reference = 'Zhang, Y., Shi, H., Yu, X., Dong, J., & Wang, Z. (2023). A global dataset of monthly river suspended sediment concentration derived from satellites (1985-2020). Scientific Data, 10, 325. https://doi.org/10.1038/s41597-023-02233-0'
        ds.source_data_link = 'https://doi.org/10.1038/s41597-023-02233-0'

        # Creator Information
        ds.creator_name = 'Zhongwang Wei'
        ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ds.creator_institution = 'Sun Yat-sen University, China'

        # Processing Information
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds.history = f'{current_time}: Converted from GSED CSV to CF-1.8 compliant NetCDF format. Applied quality control checks: flagged negative values as bad, values >3000 mg/L as suspect. Script: process_gsed_cf18.py'
        ds.date_created = datetime.now().strftime('%Y-%m-%d')
        ds.date_modified = datetime.now().strftime('%Y-%m-%d')
        ds.processing_level = 'Quality controlled and standardized'

        ds.comment = 'Data represents satellite-derived monthly suspended sediment concentration. Discharge and sediment load are not available in this dataset. Quality flags indicate: 0=good, 1=estimated, 2=suspect (extreme values), 3=bad (negative values), 9=missing.'

    print(f"Created: {output_file}")

    # --------------------------------------------------
    # CF-1.8 / ACDD-1.3 compliance check
    # --------------------------------------------------
    # errors, warnings = check_nc_completeness(output_file)

    # if errors:
    #     print("❌ CF/ACDD compliance FAILED:")
    #     for e in errors:
    #         print("   -", e)
    #     raise RuntimeError("NetCDF compliance check failed")

    # if warnings:
    #     print("⚠️ CF/ACDD compliance warnings:")
    #     for w in warnings:
    #         print("   -", w)


    # Return statistics for CSV
    stats = {
        'Source_ID': int(r_id),
        'reach_id': int(r_id),
        'reach_level': reach_level if reach_level is not None else np.nan,
        'basin_code_l1': basin_code_l1 if basin_code_l1 is not None else '',
        'basin_code_l2': basin_code_l2 if basin_code_l2 is not None else '',
        'basin_code_l3': basin_code_l3 if basin_code_l3 is not None else '',
        'basin_code_l4': basin_code_l4 if basin_code_l4 is not None else '',
        'longitude': lon if lon is not None else np.nan,
        'latitude': lat if lat is not None else np.nan,
        'reach_length_m': length if length is not None else np.nan,
        'upstream_area_km2': upstream_area_km2 if upstream_area_km2 is not None else np.nan,
        'upstream_area_source': upstream_area_source if upstream_area_source is not None else '',
        'SSC_start_date': f'{start_year}-{start_month:02d}',
        'SSC_end_date': f'{end_year}-{end_month:02d}',
        'SSC_percent_complete': percent_complete,
        'temporal_span': f'{start_year}-{start_month:02d} to {end_year}-{end_month:02d}',
        'n_months': n_times,
        'n_good': good_count
    }

    return stats

def create_summary_csv(stats_list, output_dir):
    """
    Create summary CSV file with station metadata

    Args:
        stats_list: List of station statistics dictionaries
        output_dir: Output directory path
    """
    csv_file = output_dir / 'GSED_station_summary.csv'

    # Create DataFrame
    df = pd.DataFrame(stats_list)

    # Add common metadata columns
    df['Data Source Name'] = 'GSED Dataset'
    df['Type'] = 'Satellite'
    df['Temporal Resolution'] = 'monthly'
    df['Variables Provided'] = 'Q, SSC, SSL'
    df['Geographic Coverage'] = 'Global rivers'
    df['Reference/DOI'] = 'https://doi.org/10.1038/s41597-023-02233-0'

    # Add Q and SSL columns (all missing for GSED)
    df['Q_start_date'] = ''
    df['Q_end_date'] = ''
    df['Q_percent_complete'] = 0.0
    df['SSL_start_date'] = ''
    df['SSL_end_date'] = ''
    df['SSL_percent_complete'] = 0.0

    # Reorder columns
    columns = [
        'Source_ID', 'reach_id', 'reach_level',
        'basin_code_l1', 'basin_code_l2', 'basin_code_l3', 'basin_code_l4',
        'longitude', 'latitude', 'reach_length_m', 'upstream_area_km2',
        'upstream_area_source',
        'Data Source Name', 'Type', 'Temporal Resolution', 'temporal_span',
        'Variables Provided', 'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete',
        'n_months', 'n_good'
    ]

    df = df[columns]

    # Save to CSV
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\nCreated summary CSV: {csv_file}")
    print(f"Total stations: {len(df)}")

def main():
    """Main processing function"""
    source_root = resolve_source_root(__file__)
    output_root = resolve_output_root(__file__, create=True)
    source_dir = source_root / "GSED" / "GSED"
    csv_file = source_dir / "GSED_Reach_Monthly_SSC.csv"
    shapefile = source_dir / "GSED_Reach.shp"
    output_dir = output_root / "monthly" / "GSED" / "qc"


    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GSED Monthly SSC Data Processing")
    print("CF-1.8 & ACDD-1.3 Compliant NetCDF Generation")
    print("="*70)

    print(f"\nReading GSED CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} reaches in CSV file")

    target_rids = {_normalize_gsed_rid(r_id) for r_id in df['R_ID'].tolist()}
    print(f"Loading reach metadata from shapefile: {shapefile}")
    reach_metadata = load_gsed_reach_metadata(shapefile, target_rids=target_rids)
    print(f"Loaded metadata for {len(reach_metadata)} reaches used by GSED")

    area_lookup_file = find_gsed_area_lookup_file(source_dir)
    if area_lookup_file is not None:
        print(f"Loading external upstream area lookup: {area_lookup_file}")
        area_lookup = load_gsed_area_lookup(area_lookup_file, target_rids=target_rids)
        attached_count = merge_gsed_area_lookup(reach_metadata, area_lookup, target_rids=target_rids)
        print(
            f"Attached upstream area metadata to {attached_count} reaches from "
            f"{area_lookup_file.name}"
        )
    else:
        print(
            "No external upstream-area lookup found. Continuing without "
            "reported drainage area for GSED."
        )

    # Create time array for all months (1985-01 to 2020-12)
    time_array = create_time_array(1985, 1, 432)

    # Process each station
    stats_list = []
    success_count = 0
    skip_count = 0
    error_count = 0
    tasks = []
    total = len(df)

    for idx, row in df.iterrows():
        r_id = row['R_ID']
        r_id_str = _normalize_gsed_rid(r_id)
        ssc_data = row.iloc[1:].values.astype(float)

        reach_meta = reach_metadata.get(r_id_str, _derive_basin_info_from_rid(r_id_str))

        tasks.append((
            idx,
            total,
            r_id,
            ssc_data,
            r_id_str,
            reach_meta,
            time_array,
            output_dir,
        ))

    stats_list = []
    success_count = 0
    skip_count = 0
    error_count = 0

    max_workers = max(1, min(os.cpu_count() or 1, 16))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_reach, task) for task in tasks]

        for future in as_completed(futures):
            status, payload = future.result()
            if status == "success":
                stats_list.append(payload)
                success_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                print(f"Error processing reach: {payload}")
                error_count += 1

    # Create summary CSV
    if stats_list:
        create_summary_csv(stats_list, output_dir)

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Successfully created: {success_count} files")
    print(f"Skipped (no valid data): {skip_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()
