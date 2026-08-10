#!/usr/bin/env python3
"""
Process EUSEDcollab Dataset to CF-1.8 Compliant NetCDF Format

This script processes the EUSEDcollab (European Sediment Collaboration) dataset
into CF-1.8 compliant NetCDF files with quality control flags and comprehensive metadata.

Data Processing Steps:
1. Read original CSV data (Q_SSL and METADATA files)
2. Convert units:
   - Q: m³/day, m³/month, m³/event, m³/timestep → m³/s
   - SSC: kg/m³ → mg/L (× 1,000); g/m³ → mg/L (× 1)
   - SSL: kg/day, kg/month, kg/event, kg/timestep → ton/day
3. Collapse sub-daily observations to one daily record where applicable
4. Apply quality control checks and create quality flags
5. Trim time series to valid data range
6. Write CF-1.8 compliant NetCDF files
7. Generate station summary CSV

Author: Zhongwang Wei
Institution: Sun Yat-sen University, China
Email: weizhw6@mail.sysu.edu.cn
Date: 2025-10-25
"""

import os
import sys
import inspect
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
warnings.filterwarnings('ignore')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.plot import plot_ssc_q_diagnostic
from code.qc import (
    apply_hydro_qc_with_provenance as shared_apply_hydro_qc_with_provenance,
    apply_quality_flag,
    apply_quality_flag_array,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    compute_log_iqr_bounds,
    propagate_ssc_q_inconsistency_to_ssl,
)
from code.runtime import resolve_output_root, resolve_source_root
from code.units import convert_ssl_units_if_needed
from code.daily_aggregation import aggregate_eused_to_daily


# =============================================================================
# Configuration
# =============================================================================
SOURCE_DIR = os.fspath(resolve_source_root(start=__file__) / "EUSEDcollab")
OUTPUT_DIR = os.fspath(
    resolve_output_root(start=__file__) / "monthly" / "EUSEDcollab" / "qc"
)
METADATA_FILE = os.path.join(SOURCE_DIR, "ALL_METADATA.csv")

EUSED_COUNTRY_MAP = {
    "BE": {"country": "Belgium", "continent_region": "Europe", "iso_a3": "BEL"},
    "CZ": {"country": "Czech Republic", "continent_region": "Europe", "iso_a3": "CZE"},
    "DK": {"country": "Denmark", "continent_region": "Europe", "iso_a3": "DNK"},
    "ES": {"country": "Spain", "continent_region": "Europe", "iso_a3": "ESP"},
    "FR": {"country": "France", "continent_region": "Europe", "iso_a3": "FRA"},
    "GR": {"country": "Greece", "continent_region": "Europe", "iso_a3": "GRC"},
    "IT": {"country": "Italy", "continent_region": "Europe", "iso_a3": "ITA"},
    "PL": {"country": "Poland", "continent_region": "Europe", "iso_a3": "POL"},
    "PT": {"country": "Portugal", "continent_region": "Europe", "iso_a3": "PRT"},
    "SI": {"country": "Slovenia", "continent_region": "Europe", "iso_a3": "SVN"},
}


def _default_worker_count():
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _env_bool(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        print(f"Warning: ignoring invalid {name}={raw_value!r}; using {default}")
        return default


RUN_IN_PARALLEL = _env_bool("EUSED_RUN_IN_PARALLEL", True)
N_WORKERS = _env_int("EUSED_N_WORKERS", _default_worker_count())
QC_IQR_K = 1.5
QC_MIN_SAMPLES_ENVELOPE = 5
WRITE_DIAGNOSTIC_PLOTS = True
DIAGNOSTIC_DIR = os.path.join(OUTPUT_DIR, "diagnostic")
DIAGNOSTIC_PLOT_DIR = os.path.join(OUTPUT_DIR, "diagnostic_plots")

FILL_VALUE = -9999.0
SSC_Q_SSL_RATIO_TARGETS = (
    1.0 / 86400.0,
    86400.0,
    1.0 / 1000.0,
    1000.0,
)
SSC_Q_SSL_RATIO_TOLERANCE_LOG10 = 0.08

FLAG_GOOD = np.int8(0)
FLAG_ESTIMATED = np.int8(1)
FLAG_SUSPECT = np.int8(2)
FLAG_BAD = np.int8(3)
FLAG_MISSING = np.int8(9)


def detect_and_convert_columns(df):
    """Detect and convert Q, SSC and SSL to m3/s, mg/L and ton/day."""
    df = df.copy()
    days_in_month = df['date'].dt.days_in_month if 'date' in df.columns else None
    duration_days = _event_duration_days(df)
    timestep_seconds = _timestep_seconds(df)

    q_col = _select_q_column(df)
    if q_col:
        col = q_col.lower()
        values = pd.to_numeric(df[q_col], errors='coerce')
        if 'event' in col:
            df['Q'] = values / (duration_days * 86400.0)
        elif 'm-1' in col and days_in_month is not None:
            df['Q'] = values / (days_in_month * 86400.0)
        elif 'd-1' in col:
            df['Q'] = values / 86400.0
        elif 'ts-1' in col:
            df['Q'] = values / timestep_seconds
        elif 's-1' in col or '/s' in col:
            df['Q'] = values
        else:
            df['Q'] = values
    else:
        df['Q'] = np.nan

    ssc_col = _select_ssc_column(df)
    if ssc_col:
        col = _normalize_unit_text(ssc_col)
        values = pd.to_numeric(df[ssc_col], errors='coerce')
        if _column_has_unit(col, "kg", "m-3"):
            df['SSC'] = values * 1e3
        elif _column_has_unit(col, "g", "m-3"):
            df['SSC'] = values
        else:
            df['SSC'] = np.nan
    else:
        df['SSC'] = np.nan

    ssl_col = _select_ssl_column(df)
    if ssl_col:
        col = ssl_col.lower()
        values = pd.to_numeric(df[ssl_col], errors='coerce')
        if 'kg' in col and 'm-1' in col and days_in_month is not None:
            df['SSL'] = values / days_in_month / 1000.0
        elif 'kg' in col and 'd-1' in col:
            df['SSL'] = values / 1000.0
        elif 'kg' in col and 'event' in col:
            df['SSL'] = values / duration_days / 1000.0
        elif ('event' in col and ('t ' in col or '(t' in col or 'ton' in col)):
            df['SSL'] = values / duration_days
        elif 'kg' in col and 'ts-1' in col:
            df['SSL'] = values * 86400.0 / timestep_seconds / 1000.0
        elif ('ts-1' in col and ('t ' in col or '(t' in col or 'ton' in col)):
            df['SSL'] = values * 86400.0 / timestep_seconds
        else:
            df['SSL'] = np.nan
    else:
        df['SSL'] = np.nan
    return df


def _select_q_column(df):
    q_cols = [c for c in df.columns if c.lower().startswith('q') and '(' in c.lower()]
    return _first_by_priority(q_cols, [
        lambda c: 'm3 s-1' in c.lower() or 'm3/s' in c.lower() or '/s' in c.lower(),
        lambda c: 'm3 ts-1' in c.lower(),
        lambda c: 'm3 event-1' in c.lower(),
        lambda c: 'm3 d-1' in c.lower(),
        lambda c: 'm3 m-1' in c.lower(),
    ])


def _select_ssc_column(df):
    ssc_cols = [c for c in df.columns if 'ssc' in c.lower()]
    return _first_by_priority(ssc_cols, [
        lambda c: _column_has_unit(c, "g", "m-3"),
        lambda c: _column_has_unit(c, "kg", "m-3"),
    ])


def _normalize_unit_text(text):
    text = str(text).lower()
    replacements = {
        "³": "3", "⁻": "-", "−": "-", "/": " ",
        "(": " ", ")": " ", "_": " ", "^": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _column_has_unit(column_name, mass_unit, volume_unit):
    text = _normalize_unit_text(column_name)
    tokens = set(text.replace("-", " -").split())
    return mass_unit in tokens and volume_unit in text


def _select_ssl_column(df):
    ssl_cols = [c for c in df.columns if 'ssl' in c.lower()]
    return _first_by_priority(ssl_cols, [
        lambda c: 'kg' in c.lower() and 'ts-1' in c.lower(),
        lambda c: 'ts-1' in c.lower() and ('t ' in c.lower() or '(t' in c.lower() or 'ton' in c.lower()),
        lambda c: 'kg' in c.lower() and 'event' in c.lower(),
        lambda c: 'event' in c.lower() and ('t ' in c.lower() or '(t' in c.lower() or 'ton' in c.lower()),
        lambda c: 'kg' in c.lower() and 'd-1' in c.lower(),
        lambda c: 'kg' in c.lower() and 'm-1' in c.lower(),
    ])


def _first_by_priority(columns, predicates):
    for predicate in predicates:
        for col in columns:
            if predicate(col):
                return col
    return columns[0] if columns else None


def _event_duration_days(df):
    if 'Start date (DD/MM/YYYY)' in df.columns and 'End date (DD/MM/YYYY)' in df.columns:
        start = pd.to_datetime(df['Start date (DD/MM/YYYY)'], errors='coerce', dayfirst=True)
        end = pd.to_datetime(df['End date (DD/MM/YYYY)'], errors='coerce', dayfirst=True)
    elif 'Start date' in df.columns and 'End date' in df.columns:
        start = pd.to_datetime(df['Start date'], errors='coerce')
        end = pd.to_datetime(df['End date'], errors='coerce')
    else:
        return pd.Series(1.0, index=df.index, dtype=float)
    duration = (end - start).dt.total_seconds() / 86400.0
    return duration.where(duration > 0).fillna(1.0)


def _timestep_seconds(df):
    interval_col = next((c for c in df.columns if 'time_interval' in c.lower()), None)
    if interval_col:
        seconds = pd.to_timedelta(df[interval_col], errors='coerce').dt.total_seconds()
        return seconds.where(seconds > 0)
    duration_col = next((c for c in df.columns if 'sampling duration' in c.lower() and '(d)' in c.lower()), None)
    if duration_col:
        seconds = pd.to_numeric(df[duration_col], errors='coerce') * 86400.0
        return seconds.where(seconds > 0)
    if 'date' not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    times = pd.to_datetime(df['date'], errors='coerce')
    group_col = next((c for c in df.columns if c.lower() == 'event_index'), None)
    if group_col:
        grouped = times.groupby(df[group_col])
        prev_delta = grouped.diff().dt.total_seconds()
        next_delta = (grouped.shift(-1) - times).dt.total_seconds()
        seconds = prev_delta.where(prev_delta > 0).fillna(next_delta.where(next_delta > 0))
        group_median = seconds.groupby(df[group_col]).transform(lambda s: s[s > 0].median())
        seconds = seconds.fillna(group_median)
    else:
        prev_delta = times.diff().dt.total_seconds()
        next_delta = (times.shift(-1) - times).dt.total_seconds()
        seconds = prev_delta.where(prev_delta > 0).fillna(next_delta.where(next_delta > 0))
    valid = seconds[seconds > 0]
    if len(valid) > 0:
        seconds = seconds.fillna(float(valid.median()))
    return seconds.where(seconds > 0)


def _finite_positive(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.notna() & (values > 0)


def _station_median_q_ssc_ssl_ratio(df):
    valid = _finite_positive(df["Q"]) & _finite_positive(df["SSC"]) & _finite_positive(df["SSL"])
    if int(valid.sum()) < 5:
        return np.nan
    expected = 0.0864 * df.loc[valid, "Q"].astype(float) * df.loc[valid, "SSC"].astype(float)
    ratio = df.loc[valid, "SSL"].astype(float) / expected
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) < 5:
        return np.nan
    return float(np.nanmedian(ratio))


def _is_close_on_log10(value, target, tolerance=SSC_Q_SSL_RATIO_TOLERANCE_LOG10):
    if not np.isfinite(value) or value <= 0 or target <= 0:
        return False
    return abs(np.log10(float(value)) - np.log10(float(target))) <= float(tolerance)


def _detect_ssc_seconds_mismatch(df):
    ratio = _station_median_q_ssc_ssl_ratio(df)
    return _is_close_on_log10(ratio, 1.0 / 86400.0), ratio


def _fixed_ratio_mismatch_mask(Q, SSC, SSL):
    Q = np.asarray(Q, dtype=float)
    SSC = np.asarray(SSC, dtype=float)
    SSL = np.asarray(SSL, dtype=float)
    expected = 0.0864 * Q * SSC
    ratio = np.full(len(Q), np.nan, dtype=float)
    valid = (
        np.isfinite(Q) & np.isfinite(SSC) & np.isfinite(SSL)
        & (Q > 0) & (SSC > 0) & (SSL > 0)
        & np.isfinite(expected) & (expected > 0)
    )
    ratio[valid] = SSL[valid] / expected[valid]
    mask = np.zeros(len(Q), dtype=bool)
    for target in SSC_Q_SSL_RATIO_TARGETS:
        log_delta = np.full(len(Q), np.nan, dtype=float)
        positive = valid & (ratio > 0)
        log_delta[positive] = np.abs(np.log10(ratio[positive]) - np.log10(target))
        mask |= positive & (log_delta <= SSC_Q_SSL_RATIO_TOLERANCE_LOG10)
    return mask


def parse_date_flexible(date_str):
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    for fmt in ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%Y %H:%M:%S']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(date_str)
    except Exception:
        print(f"Warning: Could not parse date: {date_str}")
        return None


def trim_to_valid_data(df, date_col='date', _audit=False):
    """Trim to the period containing valid sediment data (SSC OR SSL)."""
    valid_ssc = df['SSC'].notna() & (df['SSC'] != FILL_VALUE)
    valid_ssl = df['SSL'].notna() & (df['SSL'] != FILL_VALUE)
    valid_sediment = valid_ssc | valid_ssl
    if not valid_sediment.any():
        if _audit:
            return None, _classify_records(df)
        return None
    valid_indices = valid_sediment[valid_sediment].index
    df_trimmed = df.loc[valid_indices[0]:valid_indices[-1]].copy()
    if _audit:
        return df_trimmed, _classify_records(df), _classify_records(df_trimmed)
    return df_trimmed


def _classify_records(df):
    valid_q = df['Q'].notna() & (df['Q'] != FILL_VALUE)
    valid_ssc = df['SSC'].notna() & (df['SSC'] != FILL_VALUE)
    valid_ssl = df['SSL'].notna() & (df['SSL'] != FILL_VALUE)
    return {
        'n_total': len(df),
        'n_ssc_only': int((valid_ssc & ~valid_q & ~valid_ssl).sum()),
        'n_ssl_only': int((valid_ssl & ~valid_q & ~valid_ssc).sum()),
        'n_q_only': int((valid_q & ~valid_ssc & ~valid_ssl).sum()),
        'n_paired': int((valid_q & (valid_ssc | valid_ssl)).sum()),
        'n_any_sediment': int((valid_ssc | valid_ssl).sum()),
        'n_any_q': int(valid_q.sum()),
    }


def _print_audit(station_id, before, after):
    labels = [
        ('total', 'n_total'), ('SSC-only', 'n_ssc_only'),
        ('SSL-only', 'n_ssl_only'), ('Q-only', 'n_q_only'),
        ('paired', 'n_paired'), ('any_sediment', 'n_any_sediment'),
        ('any_Q', 'n_any_q'),
    ]
    print(f"  AUDIT station {station_id}:")
    for label, key in labels:
        b = before[key]
        a = after[key]
        delta = a - b
        delta_str = f"{delta:+d}" if delta != 0 else "  ±"
        print(f"    {label:>14s}: {b:>6d} -> {a:>6d}  ({delta_str})")


def _to_float_array(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)


def _mask_valid_positive(arr):
    arr = np.asarray(arr, dtype=float)
    return np.isfinite(arr) & (arr > 0)


def qc_with_toolpy(df, station_id, station_name, diagnostic_dir=None,
                   iqr_k=1.5, min_samples_envelope=5, flag_estimated_mask=None):
    out = df.copy()
    for v in ["Q", "SSC", "SSL"]:
        if v in out.columns:
            out[v] = pd.to_numeric(out[v], errors="coerce")
            out.loc[out[v] == float(FILL_VALUE_FLOAT), v] = np.nan
            out.loc[out[v] == -9999.0, v] = np.nan
    Q = _to_float_array(out["Q"])
    SSC = _to_float_array(out["SSC"])
    SSL = _to_float_array(out["SSL"])
    Q_flag = np.array([apply_quality_flag(v, "Q") for v in Q], dtype=np.int8)
    SSC_flag = np.array([apply_quality_flag(v, "SSC") for v in SSC], dtype=np.int8)
    SSL_flag = np.array([apply_quality_flag(v, "SSL") for v in SSL], dtype=np.int8)
    if flag_estimated_mask:
        for var, mask in flag_estimated_mask.items():
            mask = np.asarray(mask, dtype=bool)
            if var == "SSC":
                SSC_flag = np.where(mask & (SSC_flag == 0), np.int8(1), SSC_flag)
            if var == "Q":
                Q_flag = np.where(mask & (Q_flag == 0), np.int8(1), Q_flag)
            if var == "SSL":
                SSL_flag = np.where(mask & (SSL_flag == 0), np.int8(1), SSL_flag)
    fixed_ratio_mismatch = _fixed_ratio_mismatch_mask(Q, SSC, SSL)
    SSC_flag = np.where(fixed_ratio_mismatch & np.isin(SSC_flag, [FLAG_GOOD, FLAG_ESTIMATED]), FLAG_SUSPECT, SSC_flag)
    SSL_flag = np.where(fixed_ratio_mismatch & np.isin(SSL_flag, [FLAG_GOOD, FLAG_ESTIMATED]), FLAG_SUSPECT, SSL_flag)
    ssc_lb, ssc_ub = compute_log_iqr_bounds(SSC, k=iqr_k)
    if ssc_lb is not None:
        bad = np.isfinite(SSC) & (SSC > 0) & ((SSC < ssc_lb) | (SSC > ssc_ub))
        SSC_flag = np.where(bad & (SSC_flag == 0), np.int8(2), SSC_flag)
    ssl_lb, ssl_ub = compute_log_iqr_bounds(SSL, k=iqr_k)
    if ssl_lb is not None:
        bad = np.isfinite(SSL) & (SSL > 0) & ((SSL < ssl_lb) | (SSL > ssl_ub))
        SSL_flag = np.where(bad & (SSL_flag == 0), np.int8(2), SSL_flag)
    out["Q_flag"] = Q_flag.astype(np.int8)
    out["SSC_flag"] = SSC_flag.astype(np.int8)
    out["SSL_flag"] = SSL_flag.astype(np.int8)
    out["Q_flag_qc1_physical"] = np.array([apply_quality_flag(v, "Q") for v in Q], dtype=np.int8)
    out["SSC_flag_qc1_physical"] = np.array([apply_quality_flag(v, "SSC") for v in SSC], dtype=np.int8)
    out["SSL_flag_qc1_physical"] = np.array([apply_quality_flag(v, "SSL") for v in SSL], dtype=np.int8)
    ssc_qc2 = out["SSC_flag_qc1_physical"].copy()
    ssl_qc2 = out["SSL_flag_qc1_physical"].copy()
    if ssc_lb is not None:
        bad_ssc_qc2 = np.isfinite(SSC) & (SSC > 0) & ((SSC < ssc_lb) | (SSC > ssc_ub))
        ssc_qc2 = np.where(bad_ssc_qc2 & (ssc_qc2 == 0), np.int8(2), ssc_qc2)
    out["SSC_flag_qc2_log_iqr"] = ssc_qc2
    if ssl_lb is not None:
        bad_ssl_qc2 = np.isfinite(SSL) & (SSL > 0) & ((SSL < ssl_lb) | (SSL > ssl_ub))
        ssl_qc2 = np.where(bad_ssl_qc2 & (ssl_qc2 == 0), np.int8(2), ssl_qc2)
    out["SSL_flag_qc2_log_iqr"] = ssl_qc2
    out["Q_flag_qc2_log_iqr"] = np.full(len(out), np.int8(8), dtype=np.int8)
    out["SSC_flag_qc3_ssc_q"] = np.full(len(out), np.int8(8), dtype=np.int8)
    out["SSL_flag_qc3_from_ssc_q"] = np.full(len(out), np.int8(8), dtype=np.int8)
    ssc_q_bounds = build_ssc_q_envelope(Q_m3s=out["Q"].to_numpy(dtype=float), SSC_mgL=out["SSC"].to_numpy(dtype=float), k=iqr_k, min_samples=min_samples_envelope)
    resid_arr = np.full(len(out), np.nan, dtype=float)
    if ssc_q_bounds is not None:
        Q_arr = out["Q"].to_numpy(dtype=float)
        SSC_arr = out["SSC"].to_numpy(dtype=float)
        Qf_arr = out["Q_flag"].to_numpy(dtype=np.int8)
        SSCf_arr = out["SSC_flag"].to_numpy(dtype=np.int8)
        for i in range(len(out)):
            inconsistent, resid = check_ssc_q_consistency(Q_arr[i], SSC_arr[i], int(Qf_arr[i]), int(SSCf_arr[i]), ssc_q_bounds)
            resid_arr[i] = np.nan if resid is None else float(resid)
            if inconsistent:
                out.at[out.index[i], "SSC_flag_qc3_ssc_q"] = np.int8(2)
        out["SSC_flag_qc3_ssc_q"] = np.where(
            (out["SSC_flag_qc1_physical"].values == 0) & (out["SSC_flag_qc3_ssc_q"].values == 8),
            np.int8(0), out["SSC_flag_qc3_ssc_q"].values)
    out["ssc_q_resid"] = resid_arr
    out["SSL_flag_qc3_from_ssc_q"] = np.where(out["SSL_flag_qc1_physical"].values == 0, np.int8(0), out["SSL_flag_qc3_from_ssc_q"].values)
    if diagnostic_dir is not None:
        os.makedirs(diagnostic_dir, exist_ok=True)
        out_png = os.path.join(diagnostic_dir, f"EUSEDcollab_{station_id}_{station_name}_ssc_q.png")
        try:
            plot_ssc_q_diagnostic(time=out["date"].to_numpy(), Q=Q, SSC=SSC, Q_flag=Q_flag, SSC_flag=SSC_flag,
                                  ssc_q_bounds=ssc_q_bounds, station_id=str(station_id), station_name=str(station_name), out_png=out_png)
        except Exception as e:
            print(f"  Warning: diagnostic plot failed: {e}")
    return out, ssc_q_bounds


def apply_quality_flag_with_provenance(value, var_name):
    flag = apply_quality_flag(value, var_name)
    if flag == FLAG_MISSING:
        reason = 9
    elif flag == FLAG_BAD:
        reason = 3
    elif flag == FLAG_SUSPECT:
        reason = 2
    elif flag == FLAG_ESTIMATED:
        reason = 1
    else:
        reason = 0
    return np.int8(flag), np.int8(reason)


def apply_hydro_qc_with_provenance(df, station_id, station_name, output_dir=None,
                                   diagnostic_dir=None, iqr_k=1.5,
                                   min_samples_envelope=5, flag_estimated_mask=None):
    del flag_estimated_mask
    expected_params = {
        "time", "Q", "SSC", "SSL", "Q_is_independent", "SSC_is_independent",
        "SSL_is_independent", "Q_derived_mask", "SSC_derived_mask", "SSL_derived_mask",
        "ssl_is_derived_from_q_ssc", "qc2_k", "qc2_min_samples", "qc3_k", "qc3_min_samples",
    }
    shared_params = inspect.signature(shared_apply_hydro_qc_with_provenance).parameters
    missing_params = sorted(expected_params.difference(shared_params))
    if missing_params:
        raise TypeError("shared_apply_hydro_qc_with_provenance missing expected parameters: " + ", ".join(missing_params))
    df_qc = df.copy()

    def _qc_array(column):
        values = pd.to_numeric(df_qc[column], errors="coerce").to_numpy(dtype=float)
        missing = np.isclose(values, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5) | np.isclose(values, -9999.0, rtol=1e-5, atol=1e-5)
        values[missing] = np.nan
        df_qc[column] = values
        return values

    Q = _qc_array("Q")
    SSC = _qc_array("SSC")
    SSL = _qc_array("SSL")
    valid_qc_mask = np.isfinite(Q) | np.isfinite(SSC) | np.isfinite(SSL)
    qc_result = shared_apply_hydro_qc_with_provenance(
        time=df["date"].to_numpy(), Q=Q, SSC=SSC, SSL=SSL,
        Q_derived_mask=df["Q_derived"].to_numpy(dtype=bool),
        SSC_derived_mask=df["SSC_derived"].to_numpy(dtype=bool),
        SSL_derived_mask=df["SSL_derived"].to_numpy(dtype=bool),
        Q_is_independent=True, SSC_is_independent=True, SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=True, qc2_k=iqr_k,
        qc2_min_samples=min_samples_envelope, qc3_k=iqr_k,
        qc3_min_samples=min_samples_envelope)
    flag_columns = [
        "Q_flag", "SSC_flag", "SSL_flag", "Q_flag_qc1_physical",
        "SSC_flag_qc1_physical", "SSL_flag_qc1_physical", "Q_flag_qc2_log_iqr",
        "SSC_flag_qc2_log_iqr", "SSL_flag_qc2_log_iqr", "SSC_flag_qc3_ssc_q",
        "SSL_flag_qc3_from_ssc_q",
    ]
    for column in flag_columns:
        df_qc[column] = np.full(len(df_qc), FILL_VALUE_INT, dtype=np.int8)
    ssc_q_bounds = None
    if qc_result is not None:
        expected_len = int(np.sum(valid_qc_mask))
        actual_len = len(qc_result["Q_flag"])
        if actual_len != expected_len:
            raise ValueError(f"shared QC result length mismatch: expected {expected_len}, got {actual_len}")
        for column in flag_columns:
            df_qc.loc[valid_qc_mask, column] = np.asarray(qc_result[column], dtype=np.int8)
        ssc_q_bounds = qc_result.get("ssc_q_bounds")
    df_qc["ssc_q_resid"] = np.full(len(df_qc), np.nan, dtype=float)
    if diagnostic_dir is not None:
        os.makedirs(diagnostic_dir, exist_ok=True)
        out_png = os.path.join(diagnostic_dir, f"EUSEDcollab_{station_id}_{station_name}_ssc_q.png")
        try:
            plot_ssc_q_diagnostic(time=df_qc["date"].to_numpy(), Q=Q, SSC=SSC,
                                  Q_flag=df_qc["Q_flag"].to_numpy(dtype=np.int8),
                                  SSC_flag=df_qc["SSC_flag"].to_numpy(dtype=np.int8),
                                  ssc_q_bounds=ssc_q_bounds, station_id=str(station_id),
                                  station_name=str(station_name), out_png=out_png)
        except Exception as e:
            print(f"  Warning: diagnostic plot failed: {e}")

    def _flag_stats(flag_arr):
        flag_arr = np.asarray(flag_arr, dtype=np.int8)
        total = int(flag_arr.size)
        result = {
            "total": total, "good": int(np.sum(flag_arr == 0)),
            "estimated": int(np.sum(flag_arr == 1)), "suspect": int(np.sum(flag_arr == 2)),
            "bad": int(np.sum(flag_arr == 3)), "missing": int(np.sum(flag_arr == 9)),
        }
        for k in ["good", "estimated", "suspect", "bad", "missing"]:
            result[k + "_pct"] = float(result[k] / total * 100.0) if total > 0 else 0.0
        return result

    prov = {"station_id": str(station_id), "station_name": str(station_name),
            "Q": _flag_stats(df_qc["Q_flag"].to_numpy()),
            "SSC": _flag_stats(df_qc["SSC_flag"].to_numpy()),
            "SSL": _flag_stats(df_qc["SSL_flag"].to_numpy())}
    if output_dir is not None:
        prov_dir = os.path.join(output_dir, "qc_provenance")
        os.makedirs(prov_dir, exist_ok=True)
        prov_path = os.path.join(prov_dir, f"EUSEDcollab_{station_id}_{station_name}_qc_provenance.json")
        try:
            with open(prov_path, "w", encoding="utf-8") as f:
                json.dump(prov, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  Warning: failed to write provenance JSON: {e}")
    return df_qc, ssc_q_bounds, prov


def calculate_data_completeness_from_flag(flag_arr):
    flag_arr = np.asarray(flag_arr, dtype=np.int8)
    if flag_arr.size == 0:
        return 0.0
    return float(np.sum(flag_arr == 0) / flag_arr.size * 100.0)


def read_station_metadata(station_id):
    meta_df = pd.read_csv(METADATA_FILE, encoding='utf-8-sig')
    station_row = meta_df[meta_df['Catchment ID'] == station_id]
    if len(station_row) == 0:
        print(f"Warning: Station ID {station_id} not found in metadata")
        return None
    station_row = station_row.iloc[0]
    return {
        'catchment_id': station_id,
        'station_name': station_row['Catchment name'],
        'latitude': station_row['Latitude (4 decimal places)'],
        'longitude': station_row['Longitude (4 decimal places)'],
        'country': station_row['Country'],
        'drainage_area': station_row['Drainage area (ha)'] / 100.0,
        'stream_type': station_row['Stream type'],
        'data_type': station_row['Data type'],
        'land_use_agriculture': station_row.get('Land use: % agriculture', np.nan),
        'land_use_forest': station_row.get('Land use: % forest', np.nan),
        'start_date': parse_date_flexible(station_row['Measurement start date (DD/MM/YYYY)']),
        'end_date': parse_date_flexible(station_row['Measurement end date (DD/MM/YYYY)']),
        'references': station_row['Relevant references with full details'],
        'contact_name': station_row['Contact name'],
        'contact_email': station_row['Contact email'],
    }


def read_station_data(station_id, country):
    """Read, standardize, derive and consolidate a station to daily values."""
    q_ssl_file = os.path.join(SOURCE_DIR, 'Q_SSL', f'ID_{station_id}_Q_SSL_{country}.csv')
    if not os.path.exists(q_ssl_file):
        print(f"Warning: Data file not found: {q_ssl_file}")
        return None
    df = pd.read_csv(q_ssl_file)
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if len(date_cols) == 0:
        print(f"  Warning: No date column found in: {q_ssl_file}")
        return None
    df['date'] = df[date_cols[0]].apply(parse_date_flexible)
    df = df[df['date'].notna()].copy()
    if len(df) == 0:
        print("  Warning: No valid dates found")
        return None
    df = df.sort_values('date').reset_index(drop=True)
    df = detect_and_convert_columns(df)
    for col in ['Q', 'SSC', 'SSL']:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

    factor = 0.0864
    q_derived = np.zeros(len(df), dtype=bool)
    ssc_derived = np.zeros(len(df), dtype=bool)
    ssl_derived = np.zeros(len(df), dtype=bool)
    seconds_mismatch, station_ratio = _detect_ssc_seconds_mismatch(df)
    if seconds_mismatch:
        ssc_fix_mask = _finite_positive(df["SSC"])
        df.loc[ssc_fix_mask, "SSC"] = df.loc[ssc_fix_mask, "SSC"] / 86400.0
        ssc_derived |= ssc_fix_mask.to_numpy()
        print("  Corrected station-level SSC/Q/SSL seconds mismatch "
              "(median SSL/(0.0864*Q*SSC)={:.6g}); divided SSC by 86400".format(station_ratio))

    valid_q = df['Q'].notna() & (df['Q'] > 0)
    valid_ssc = df['SSC'].notna() & (df['SSC'] > 0)
    valid_ssl = df['SSL'].notna() & (df['SSL'] > 0)
    mask = df['SSL'].isna() & valid_q & valid_ssc
    df.loc[mask, 'SSL'] = df.loc[mask, 'Q'] * df.loc[mask, 'SSC'] * factor
    ssl_derived |= mask.to_numpy()
    valid_ssl = df['SSL'].notna() & (df['SSL'] > 0)
    mask = df['SSC'].isna() & valid_q & valid_ssl
    df.loc[mask, 'SSC'] = df.loc[mask, 'SSL'] / (df.loc[mask, 'Q'] * factor)
    ssc_derived |= mask.to_numpy()
    valid_ssc = df['SSC'].notna() & (df['SSC'] > 0)
    mask = df['Q'].isna() & valid_ssc & valid_ssl
    df.loc[mask, 'Q'] = df.loc[mask, 'SSL'] / (df.loc[mask, 'SSC'] * factor)
    q_derived |= mask.to_numpy()
    for col in ['Q', 'SSC', 'SSL']:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(FILL_VALUE)
    df['Q_derived'] = q_derived
    df['SSC_derived'] = ssc_derived
    df['SSL_derived'] = ssl_derived

    # Source records can contain several observations in the same calendar day.
    # Aggregate before QC so all downstream products contain at most one record
    # per day. Exact duplicate timestamps are collapsed first by the shared helper.
    day_counts = pd.to_datetime(df['date'], errors='coerce').dt.floor('D').value_counts()
    n_problem_days = int((day_counts > 1).sum())
    n_rows_before_daily = len(df)
    df = aggregate_eused_to_daily(df, fill_value=FILL_VALUE, factor=factor)
    if n_problem_days:
        print(f"  Daily aggregation: collapsed {n_rows_before_daily} rows to {len(df)} daily rows "
              f"across {n_problem_days} day(s) with multiple records.")
    return df[['date', 'Q', 'SSC', 'SSL', 'Q_derived', 'SSC_derived', 'SSL_derived']]


def process_station(station_id, country):
    print(f"\nProcessing station ID_{station_id}_{country}...")
    metadata = read_station_metadata(station_id)
    if metadata is None:
        return None
    df = read_station_data(station_id, country)
    if df is None or len(df) == 0:
        print("  Skipping: No data available")
        return None
    audit_before = _classify_records(df)
    df = trim_to_valid_data(df)
    if df is None or len(df) == 0:
        print("  Skipping: No valid sediment data after trimming")
        _print_audit(station_id, audit_before, {'n_total':0,'n_ssc_only':0,'n_ssl_only':0,'n_q_only':0,'n_paired':0,'n_any_sediment':0,'n_any_q':0})
        return None
    audit_after = _classify_records(df)
    _print_audit(station_id, audit_before, audit_after)
    estimated_mask = {
        "Q": df.get("Q_derived", pd.Series(False, index=df.index)).values,
        "SSC": df.get("SSC_derived", pd.Series(False, index=df.index)).values,
        "SSL": df.get("SSL_derived", pd.Series(False, index=df.index)).values,
    }
    df_qc, ssc_q_bounds, prov = apply_hydro_qc_with_provenance(
        df=df, station_id=station_id, station_name=metadata["station_name"],
        output_dir=OUTPUT_DIR, diagnostic_dir=DIAGNOSTIC_DIR if WRITE_DIAGNOSTIC_PLOTS else None,
        iqr_k=QC_IQR_K, min_samples_envelope=QC_MIN_SAMPLES_ENVELOPE,
        flag_estimated_mask=estimated_mask)
    print("  QC provenance summary:")
    print(f"    Q   good={prov['Q']['good']} ({prov['Q']['good_pct']:.1f}%)  missing={prov['Q']['missing']} ({prov['Q']['missing_pct']:.1f}%)")
    print(f"    SSC good={prov['SSC']['good']} ({prov['SSC']['good_pct']:.1f}%) missing={prov['SSC']['missing']} ({prov['SSC']['missing_pct']:.1f}%)")
    print(f"    SSL good={prov['SSL']['good']} ({prov['SSL']['good_pct']:.1f}%) missing={prov['SSL']['missing']} ({prov['SSL']['missing_pct']:.1f}%)")
    q_flag = df_qc["Q_flag"].to_numpy(dtype=np.int8)
    ssc_flag = df_qc["SSC_flag"].to_numpy(dtype=np.int8)
    ssl_flag = df_qc["SSL_flag"].to_numpy(dtype=np.int8)
    df = df_qc
    q_completeness = calculate_data_completeness_from_flag(q_flag)
    ssc_completeness = calculate_data_completeness_from_flag(ssc_flag)
    ssl_completeness = calculate_data_completeness_from_flag(ssl_flag)
    start_date = df['date'].min()
    end_date = df['date'].max()
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Data points: {len(df)}")
    print(f"  Q completeness: {q_completeness:.1f}%")
    print(f"  SSC completeness: {ssc_completeness:.1f}%")
    print(f"  SSL completeness: {ssl_completeness:.1f}%")
    if WRITE_DIAGNOSTIC_PLOTS:
        os.makedirs(DIAGNOSTIC_PLOT_DIR, exist_ok=True)
        plot_file = os.path.join(DIAGNOSTIC_PLOT_DIR, f"EUSEDcollab_{country}-{metadata['station_name']}-ID{station_id}_ssc_q.png")
        plot_ssc_q_diagnostic(time=df['date'].values, Q=df['Q'].values, SSC=df['SSC'].values,
                              Q_flag=q_flag, SSC_flag=ssc_flag, ssc_q_bounds=ssc_q_bounds,
                              station_id=str(station_id), station_name=metadata['station_name'], out_png=plot_file)
    output_file = os.path.join(OUTPUT_DIR, f'EUSEDcollab_{country}-{metadata["station_name"]}-ID{station_id}.nc')
    write_netcdf(df, metadata, q_flag, ssc_flag, ssl_flag, output_file, step_flags=df)
    print(f"  ✓ Created: {output_file}")
    return {
        'station_name': metadata['station_name'], 'Source_ID': f'EUSED_{station_id}',
        'river_name': '', 'longitude': metadata['longitude'], 'latitude': metadata['latitude'],
        'altitude': np.nan, 'upstream_area': metadata['drainage_area'],
        'Data Source Name': 'EUSEDcollab Dataset', 'Type': 'In-situ',
        'Temporal Resolution': 'monthly',
        'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'Variables Provided': 'Q, SSC, SSL', 'Geographic Coverage': f"{metadata['country']}",
        'Reference/DOI': metadata['references'], 'Q_start_date': start_date.strftime('%Y'),
        'Q_end_date': end_date.strftime('%Y'), 'Q_percent_complete': q_completeness,
        'SSC_start_date': start_date.strftime('%Y'), 'SSC_end_date': end_date.strftime('%Y'),
        'SSC_percent_complete': ssc_completeness, 'SSL_start_date': start_date.strftime('%Y'),
        'SSL_end_date': end_date.strftime('%Y'), 'SSL_percent_complete': ssl_completeness,
    }


def write_netcdf(df, metadata, q_flag, ssc_flag, ssl_flag, output_file, step_flags=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        ds.createDimension('time', None)
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'; time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'; time_var.calendar = 'gregorian'; time_var.axis = 'T'
        epoch = datetime(1970, 1, 1)
        time_var[:] = np.array([(d - epoch).total_seconds() / 86400.0 for d in df['date']])
        lat_var = ds.createVariable('lat', 'f4'); lat_var.standard_name = 'latitude'; lat_var.long_name = 'station latitude'; lat_var.units = 'degrees_north'; lat_var.valid_range = np.array([-90.0, 90.0], dtype=np.float32); lat_var[:] = metadata['latitude']
        lon_var = ds.createVariable('lon', 'f4'); lon_var.standard_name = 'longitude'; lon_var.long_name = 'station longitude'; lon_var.units = 'degrees_east'; lon_var.valid_range = np.array([-180.0, 180.0], dtype=np.float32); lon_var[:] = metadata['longitude']
        alt_var = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT); alt_var.standard_name = 'altitude'; alt_var.long_name = 'station elevation above sea level'; alt_var.units = 'm'; alt_var.positive = 'up'; alt_var.comment = 'Source: Not available in EUSEDcollab metadata.'; alt_var[:] = FILL_VALUE
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE_FLOAT); area_var.long_name = 'upstream drainage area'; area_var.units = 'km2'; area_var.comment = 'Source: Original data provided by EUSEDcollab. Converted from hectares.'; area_var[:] = metadata['drainage_area'] if pd.notna(metadata['drainage_area']) else FILL_VALUE
        q_data = pd.to_numeric(df['Q'], errors='coerce').to_numpy(dtype=np.float32)
        ssc_data = pd.to_numeric(df['SSC'], errors='coerce').to_numpy(dtype=np.float32)
        ssl_data = pd.to_numeric(df['SSL'], errors='coerce').to_numpy(dtype=np.float32)
        q_data[~np.isfinite(q_data)] = FILL_VALUE_FLOAT; ssc_data[~np.isfinite(ssc_data)] = FILL_VALUE_FLOAT; ssl_data[~np.isfinite(ssl_data)] = FILL_VALUE_FLOAT

        def _derived_mask(column):
            if column not in df.columns:
                return np.zeros(len(df), dtype=bool)
            return df[column].fillna(False).to_numpy(dtype=bool)

        def _valid_data_mask(values):
            return np.isfinite(values) & ~np.isclose(values, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5)

        ssc_derived_mask = _derived_mask("SSC_derived"); ssl_derived_mask = _derived_mask("SSL_derived")
        ssc_valid_mask = _valid_data_mask(ssc_data); ssl_valid_mask = _valid_data_mask(ssl_data)

        def _provenance_kind(valid_mask, derived_mask):
            source_present = bool(np.any(valid_mask & (~derived_mask))); derived_present = bool(np.any(valid_mask & derived_mask))
            if source_present and derived_present: return "mixed"
            if derived_present: return "derived"
            if source_present: return "source"
            return "missing"

        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=FILL_VALUE_FLOAT); q_var.standard_name = 'water_volume_transport_in_river_channel'; q_var.long_name = 'river discharge'; q_var.units = 'm3 s-1'; q_var.coordinates = 'time lat lon'; q_var.ancillary_variables = 'Q_flag'; q_var.comment = 'Source: Original data provided by EUSEDcollab; values standardized to m3 s-1 and same-day observations aggregated to daily means where applicable.'; q_var[:] = q_data
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=FILL_VALUE_INT); q_flag_var.long_name = 'quality flag for river discharge'; q_flag_var.standard_name = 'status_flag'; q_flag_var.flag_values = np.array([0,1,2,3,9], dtype=np.int8); q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'; q_flag_var[:] = q_flag
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=FILL_VALUE_FLOAT); ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'; ssc_var.long_name = 'suspended sediment concentration'; ssc_var.units = 'mg L-1'; ssc_var.coordinates = 'time lat lon'; ssc_var.ancillary_variables = 'SSC_flag SSC_derived_mask'; ssc_var.source = {"mixed":"Mixed source-reported and derived EUSEDcollab data.","derived":"Derived from EUSEDcollab Q and SSL records.","source":"Source-reported EUSEDcollab data.","missing":"No valid SSC records."}[_provenance_kind(ssc_valid_mask, ssc_derived_mask)]; ssc_var[:] = ssc_data
        ssc_flag_var = ds.createVariable('SSC_flag','b',('time',),fill_value=FILL_VALUE_INT); ssc_flag_var.long_name='quality flag for suspended sediment concentration'; ssc_flag_var.standard_name='status_flag'; ssc_flag_var.flag_values=np.array([0,1,2,3,9],dtype=np.int8); ssc_flag_var.flag_meanings='good_data estimated_data suspect_data bad_data missing_data'; ssc_flag_var[:] = ssc_flag
        ssc_derived_var = ds.createVariable('SSC_derived_mask','b',('time',)); ssc_derived_var.long_name='record-level derived-value mask for suspended sediment concentration'; ssc_derived_var.flag_values=np.array([0,1],dtype=np.int8); ssc_derived_var.flag_meanings='source_reported derived'; ssc_derived_var[:] = ssc_derived_mask.astype(np.int8)
        ssl_var = ds.createVariable('SSL','f4',('time',),fill_value=FILL_VALUE_FLOAT); ssl_var.long_name='suspended sediment load'; ssl_var.units='ton day-1'; ssl_var.coordinates='time lat lon'; ssl_var.ancillary_variables='SSL_flag SSL_derived_mask'; ssl_var.source={"mixed":"Mixed source-reported and derived EUSEDcollab data.","derived":"Derived from EUSEDcollab Q and SSC records.","source":"Source-reported EUSEDcollab data.","missing":"No valid SSL records."}[_provenance_kind(ssl_valid_mask, ssl_derived_mask)]; ssl_var.comment='Same-day Q and SSC are aggregated by arithmetic mean; daily SSL is recalculated from daily Q and SSC when both are available. SSL-only days retain the mean available SSL.'; ssl_var[:] = ssl_data
        ssl_flag_var = ds.createVariable('SSL_flag','b',('time',),fill_value=FILL_VALUE_INT); ssl_flag_var.long_name='quality flag for suspended sediment load'; ssl_flag_var.standard_name='status_flag'; ssl_flag_var.flag_values=np.array([0,1,2,3,9],dtype=np.int8); ssl_flag_var.flag_meanings='good_data estimated_data suspect_data bad_data missing_data'; ssl_flag_var[:] = ssl_flag
        ssl_derived_var = ds.createVariable('SSL_derived_mask','b',('time',)); ssl_derived_var.long_name='record-level derived-value mask for suspended sediment load'; ssl_derived_var.flag_values=np.array([0,1],dtype=np.int8); ssl_derived_var.flag_meanings='source_reported derived'; ssl_derived_var[:] = ssl_derived_mask.astype(np.int8)

        def _add_step_flag(name, values, flag_values, flag_meanings, long_name):
            if values is None: return
            v = ds.createVariable(name,'b',('time',),fill_value=FILL_VALUE_INT); v.long_name=long_name; v.standard_name='status_flag'; v.flag_values=np.array(flag_values,dtype=np.int8); v.flag_meanings=flag_meanings; v.missing_value=np.int8(FILL_VALUE_INT); v[:] = np.asarray(values,dtype=np.int8)
        if step_flags is not None:
            _add_step_flag('Q_flag_qc1_physical', step_flags.get('Q_flag_qc1_physical'), [0,3,9], 'pass bad missing', 'QC1 physical flag for river discharge')
            _add_step_flag('Q_flag_qc2_log_iqr', step_flags.get('Q_flag_qc2_log_iqr'), [0,2,8,9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for river discharge')
            _add_step_flag('SSC_flag_qc1_physical', step_flags.get('SSC_flag_qc1_physical'), [0,3,9], 'pass bad missing', 'QC1 physical flag for suspended sediment concentration')
            _add_step_flag('SSC_flag_qc2_log_iqr', step_flags.get('SSC_flag_qc2_log_iqr'), [0,2,8,9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for suspended sediment concentration')
            _add_step_flag('SSC_flag_qc3_ssc_q', step_flags.get('SSC_flag_qc3_ssc_q'), [0,2,8,9], 'pass suspect not_checked missing', 'QC3 SSC-Q consistency flag for suspended sediment concentration')
            _add_step_flag('SSL_flag_qc1_physical', step_flags.get('SSL_flag_qc1_physical'), [0,3,9], 'pass bad missing', 'QC1 physical flag for suspended sediment load')
            _add_step_flag('SSL_flag_qc2_log_iqr', step_flags.get('SSL_flag_qc2_log_iqr'), [0,2,8,9], 'pass suspect not_checked missing', 'QC2 log-IQR flag for suspended sediment load')
            _add_step_flag('SSL_flag_qc3_from_ssc_q', step_flags.get('SSL_flag_qc3_from_ssc_q'), [0,2,8,9], 'not_propagated propagated not_checked missing', 'QC3 propagation flag for suspended sediment load')
            q_var.ancillary_variables='Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'; ssc_var.ancillary_variables='SSC_flag SSC_derived_mask SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'; ssl_var.ancillary_variables='SSL_flag SSL_derived_mask SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q'

        ds.Conventions='CF-1.8, ACDD-1.3'; ds.title='Harmonized Global River Discharge and Sediment'; ds.summary=f'River discharge and suspended sediment data for {metadata["station_name"]} station from the EUSEDcollab database, standardized with source-specific temporal handling and quality-control flags.'; ds.source='In-situ station data'; ds.data_source_name='EUSEDcollab Dataset'; ds.station_name=metadata['station_name']; ds.river_name=''; ds.Source_ID=f'EUSED_{metadata["catchment_id"]}'; ds.station_id=f'EUSED_{metadata["catchment_id"]}'
        start_date=df['date'].min(); end_date=df['date'].max(); ds.temporal_resolution='monthly'; ds.temporal_span=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"; ds.time_coverage_start=start_date.strftime('%Y-%m-%d'); ds.time_coverage_end=end_date.strftime('%Y-%m-%d')
        ds.geospatial_lat_min=float(metadata['latitude']); ds.geospatial_lat_max=float(metadata['latitude']); ds.geospatial_lon_min=float(metadata['longitude']); ds.geospatial_lon_max=float(metadata['longitude']); ds.geographic_coverage=f"{metadata['country']}, {metadata['stream_type']} stream"
        country_info=EUSED_COUNTRY_MAP.get(metadata.get('country',''),{}); ds.country=country_info.get('country',''); ds.continent_region=country_info.get('continent_region',''); ds.iso_a3=country_info.get('iso_a3',''); ds.variables_provided='altitude, upstream_area, Q, SSC, SSL'; ds.number_of_data='1'; ds.reference=metadata['references']; ds.source_data_link='https://esdac.jrc.ec.europa.eu/content/european-sediment-collaboration-eusedcollab-database'; ds.creator_name='Zhongwang Wei'; ds.creator_email='weizhw6@mail.sysu.edu.cn'; ds.creator_institution='Sun Yat-sen University, China'
        if pd.notna(metadata.get('contact_name')): ds.contributor_name=metadata['contact_name']
        if pd.notna(metadata.get('contact_email')): ds.contributor_email=metadata['contact_email']
        now=datetime.now(); ds.date_created=now.strftime('%Y-%m-%d'); ds.date_modified=now.strftime('%Y-%m-%d'); ds.processing_level='Quality controlled and standardized'; ds.history=f"{now.strftime('%Y-%m-%d %H:%M:%S')}: Converted EUSEDcollab CSV data to CF-1.8 NetCDF, standardized units, consolidated same-day observations where present, recalculated daily SSL from daily Q and SSC where possible, and applied QC. Script: process_eusedcollab_to_cf18_wzx.py"; ds.comment=f'Data type: {metadata["data_type"]}. Stream type: {metadata["stream_type"]}. Quality flags: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.'
    print(f"  Written: {output_file}")


def generate_summary_csv(station_list, output_dir):
    if len(station_list) == 0:
        print("\nNo stations processed, skipping summary CSV generation")
        return
    summary_df = pd.DataFrame(station_list)
    column_order = ['station_name','Source_ID','river_name','longitude','latitude','altitude','upstream_area','Data Source Name','Type','Temporal Resolution','Temporal Span','Variables Provided','Geographic Coverage','Reference/DOI','Q_start_date','Q_end_date','Q_percent_complete','SSC_start_date','SSC_end_date','SSC_percent_complete','SSL_start_date','SSL_end_date','SSL_percent_complete']
    summary_df[column_order].to_csv(os.path.join(output_dir, 'EUSEDcollab_station_summary.csv'), index=False)
    print(f"\nStation summary CSV written: {os.path.join(output_dir, 'EUSEDcollab_station_summary.csv')}")
    print(f"Total stations processed: {len(station_list)}")


def _build_station_tasks(meta_df):
    return [(int(idx), row['Catchment ID'], row['Country']) for idx, row in meta_df.iterrows()]


def _process_station_task(task):
    idx, station_id, country = task
    log_stream = StringIO()
    with redirect_stdout(log_stream), redirect_stderr(log_stream):
        try:
            station_info = process_station(station_id, country); ok=True; error=None
        except Exception as exc:
            station_info=None; ok=False; error=str(exc); traceback.print_exc()
    return {"idx":idx,"station_id":station_id,"country":country,"ok":ok,"error":error,"station_info":station_info,"log":log_stream.getvalue()}


def _run_station_tasks_parallel(tasks):
    if not tasks: return []
    max_workers=min(N_WORKERS,len(tasks)); print(f"\nParallel processing enabled: {max_workers} worker(s)")
    results=[]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task={executor.submit(_process_station_task, task):task for task in tasks}
        for completed, future in enumerate(as_completed(future_to_task), start=1):
            _,station_id,country=future_to_task[future]
            try: result=future.result()
            except Exception as exc:
                print(f"\n  ERROR processing station ID_{station_id}_{country}: {exc}"); traceback.print_exc(); continue
            log_text=result.get('log','').rstrip()
            if log_text: print(log_text)
            print(f"  [{completed}/{len(tasks)}] {'Finished' if result['ok'] else 'ERROR processing'} station ID_{station_id}_{country}" + ('' if result['ok'] else f": {result['error']}"))
            results.append(result)
    results.sort(key=lambda item:item['idx'])
    return [item['station_info'] for item in results if item['ok'] and item['station_info'] is not None]


def _run_station_tasks_sequential(tasks):
    station_list=[]
    for completed, task in enumerate(tasks,start=1):
        _,station_id,country=task
        try:
            station_info=process_station(station_id,country)
            if station_info is not None: station_list.append(station_info)
            print(f"  [{completed}/{len(tasks)}] Finished station ID_{station_id}_{country}")
        except Exception as e:
            print(f"  ERROR processing station ID_{station_id}_{country}: {str(e)}"); traceback.print_exc()
    return station_list


def main():
    print("="*80); print("EUSEDcollab Dataset Processing to CF-1.8 Format"); print("="*80)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if WRITE_DIAGNOSTIC_PLOTS:
        os.makedirs(DIAGNOSTIC_DIR, exist_ok=True); os.makedirs(DIAGNOSTIC_PLOT_DIR, exist_ok=True)
    meta_df=pd.read_csv(METADATA_FILE,encoding='utf-8-sig')
    print(f"\nFound {len(meta_df)} stations in metadata"); print(f"Output directory: {OUTPUT_DIR}"); print(f"Parallel mode: {RUN_IN_PARALLEL}"); print(f"Configured workers: {N_WORKERS}")
    tasks=_build_station_tasks(meta_df)
    station_list=_run_station_tasks_parallel(tasks) if RUN_IN_PARALLEL and N_WORKERS > 1 else _run_station_tasks_sequential(tasks)
    generate_summary_csv(station_list,OUTPUT_DIR)
    print("\n"+"="*80); print("Processing complete!"); print("="*80)


if __name__ == '__main__':
    main()
