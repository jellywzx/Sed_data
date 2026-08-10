"""
Daily aggregation helpers for harmonized sediment dataset processing.

Provides functions to:
1. Collapse duplicate timestamps (same exact time -> single observation)
2. Aggregate sub-daily Q and SSC to daily means
3. Re-derive daily SSL from daily Q and SSC

These functions are designed to be called AFTER QC and BEFORE NetCDF output.
They operate on numeric arrays (Q, SSC, SSL) with associated flag arrays,
aggregating by calendar day computed from UTC epoch seconds.

Flag propagation rules for aggregation:
- Q_daily flag: worst-case propagation from sub-daily Q flags
  (bad > suspect > estimated > good; missing only if all missing)
- SSC_daily flag: same worst-case propagation
- SSL_daily flag: if recalculated from daily Q and SSC, uses estimated (1)
  with propagation of suspect/bad from inputs; if direct source SSL, uses
  worst-case from sub-daily SSL flags.

Flag definitions used (consistent with code/constants.py):
    0 = good
    1 = estimated
    2 = suspect
    3 = bad
    9 = missing
"""

import numpy as np
from datetime import datetime, timezone


def _worst_flag(flags):
    """Return the worst (highest severity) flag from an array of flags."""
    flags = np.asarray(flags, dtype=np.int8)
    if len(flags) == 0:
        return np.int8(9)
    for severity in [9, 3, 2, 1, 0]:
        if np.any(flags == severity):
            return np.int8(severity)
    return np.int8(9)


def collapse_duplicate_timestamps(timestamps, values, flags, fill_value=-9999.0):
    """
    Collapse identical timestamps by taking the mean of valid values.

    When the same timestamp appears multiple times:
    - If values are identical (or only one is valid), the result is that value.
    - If multiple different valid values exist, the result is their arithmetic mean.
    - Flags are propagated as worst-case.

    Parameters
    ----------
    timestamps : ndarray
        Unix epoch seconds (float64).
    values : ndarray
        Variable values (Q, SSC, or SSL).
    flags : ndarray
        Quality flags (int8).
    fill_value : float
        Sentinel value to treat as missing.

    Returns
    -------
    unique_ts : ndarray
        Unique timestamps, sorted.
    collapsed_vals : ndarray
        Mean values per unique timestamp.
    collapsed_flags : ndarray
        Worst-case flags per unique timestamp.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=float).copy()
    fl = np.asarray(flags, dtype=np.int8)

    # Treat fill_value as NaN
    vals[np.isclose(vals, float(fill_value), rtol=1e-5, atol=1e-5)] = np.nan

    # Identify unique timestamps and their inverse indices
    unique_ts, inverse = np.unique(ts, return_inverse=True)
    n_unique = len(unique_ts)

    collapsed_vals = np.full(n_unique, np.nan, dtype=float)
    collapsed_flags = np.full(n_unique, np.int8(9), dtype=np.int8)

    for i in range(n_unique):
        mask = inverse == i
        group_vals = vals[mask]
        group_flags = fl[mask]

        valid = np.isfinite(group_vals) & (group_vals >= 0) & (group_flags != 9)
        if valid.any():
            collapsed_vals[i] = float(np.nanmean(group_vals[valid]))
            collapsed_flags[i] = _worst_flag(group_flags[valid])
        else:
            collapsed_vals[i] = np.nan
            collapsed_flags[i] = _worst_flag(group_flags)

    return unique_ts, collapsed_vals, collapsed_flags


def aggregate_daily(
    time_seconds,
    Q, SSC, SSL,
    Q_flag, SSC_flag, SSL_flag,
    Q_derived_mask=None,
    SSC_derived_mask=None,
    SSL_derived_mask=None,
    fill_value=-9999.0,
    ssl_factor=0.0864,
):
    """
    Aggregate sub-daily observations to daily resolution.

    Processing steps:
    1. For each variable (Q, SSC, SSL), collapse duplicate timestamps.
    2. Group collapsed timestamps by calendar day (UTC).
    3. Q_daily = arithmetic mean of valid Q in the day.
    4. SSC_daily = arithmetic mean of valid SSC in the day.
    5. SSL_daily = Q_daily * SSC_daily * ssl_factor (only when both exist).
       Direct source SSL is retained for days without Q+SSC.

    Parameters
    ----------
    time_seconds : ndarray
        Unix epoch seconds (float64), shape (n,).
    Q, SSC, SSL : ndarray
        Variable values, shape (n,). May contain NaN or fill_value.
    Q_flag, SSC_flag, SSL_flag : ndarray
        Quality flags (int8), shape (n,).
    Q_derived_mask, SSC_derived_mask, SSL_derived_mask : ndarray or None
        Boolean masks marking derived records.
    fill_value : float
        Sentinel value.
    ssl_factor : float
        Conversion factor: SSL (t/d) = Q (m3/s) * SSC (mg/L) * ssl_factor.

    Returns
    -------
    dict with keys:
        time, Q, SSC, SSL, Q_flag, SSC_flag, SSL_flag,
        Q_derived_mask, SSC_derived_mask, SSL_derived_mask
    Each is a 1-D ndarray of length n_days.
    """
    n = len(time_seconds)

    def _prepare(arr, fill):
        a = np.asarray(arr, dtype=float).copy()
        a[np.isclose(a, float(fill), rtol=1e-5, atol=1e-5)] = np.nan
        return a

    def _prepare_bool(arr, default, name=''):
        if arr is not None:
            return np.asarray(arr, dtype=bool).copy()
        return np.full(n, bool(default), dtype=bool)

    Q_raw = _prepare(Q, fill_value)
    SSC_raw = _prepare(SSC, fill_value)
    SSL_raw = _prepare(SSL, fill_value)

    Qf = np.asarray(Q_flag, dtype=np.int8).copy()
    SSCf = np.asarray(SSC_flag, dtype=np.int8).copy()
    SSLf = np.asarray(SSL_flag, dtype=np.int8).copy()

    Q_der = _prepare_bool(Q_derived_mask, False)
    SSC_der = _prepare_bool(SSC_derived_mask, False)
    SSL_der = _prepare_bool(SSL_derived_mask, False)

    ts = np.asarray(time_seconds, dtype=np.float64)

    # Step 1: Collapse duplicate timestamps per variable
    ts_q, Q_c, Qf_c = collapse_duplicate_timestamps(ts, Q_raw, Qf, fill_value=fill_value)
    ts_ssc, SSC_c, SSCf_c = collapse_duplicate_timestamps(ts, SSC_raw, SSCf, fill_value=fill_value)
    ts_ssl, SSL_c, SSLf_c = collapse_duplicate_timestamps(ts, SSL_raw, SSLf, fill_value=fill_value)

    # Step 2: Determine calendar days for each collapsed series
    def _calendar_dates(ts_arr):
        """Convert epoch seconds to UTC date strings (YYYY-MM-DD)."""
        result = np.empty(len(ts_arr), dtype='<U10')
        for i, t in enumerate(ts_arr):
            result[i] = datetime.fromtimestamp(max(0, float(t)), tz=timezone.utc).strftime('%Y-%m-%d')
        return result

    days_q = _calendar_dates(ts_q)
    days_ssc = _calendar_dates(ts_ssc)
    days_ssl = _calendar_dates(ts_ssl)

    # Step 3: Aggregate Q by day
    unique_q_days, q_day_idx = np.unique(days_q, return_inverse=True)
    n_q_days = len(unique_q_days)
    Q_daily = np.full(n_q_days, np.nan, dtype=float)
    Q_flag_daily = np.full(n_q_days, np.int8(9), dtype=np.int8)

    for i in range(n_q_days):
        mask = q_day_idx == i
        vals = Q_c[mask]
        flags = Qf_c[mask]
        valid = np.isfinite(vals) & (vals >= 0) & (flags != 9)
        if valid.any():
            Q_daily[i] = float(np.nanmean(vals[valid]))
            Q_flag_daily[i] = _worst_flag(flags[valid])
        else:
            Q_daily[i] = np.nan
            Q_flag_daily[i] = _worst_flag(flags)

    q_daily_by_day = dict(zip(unique_q_days, zip(Q_daily, Q_flag_daily)))

    # Step 4: Aggregate SSC by day
    unique_ssc_days, ssc_day_idx = np.unique(days_ssc, return_inverse=True)
    n_ssc_days = len(unique_ssc_days)
    SSC_daily = np.full(n_ssc_days, np.nan, dtype=float)
    SSC_flag_daily = np.full(n_ssc_days, np.int8(9), dtype=np.int8)

    for i in range(n_ssc_days):
        mask = ssc_day_idx == i
        vals = SSC_c[mask]
        flags = SSCf_c[mask]
        valid = np.isfinite(vals) & (vals >= 0) & (flags != 9)
        if valid.any():
            SSC_daily[i] = float(np.nanmean(vals[valid]))
            SSC_flag_daily[i] = _worst_flag(flags[valid])
        else:
            SSC_daily[i] = np.nan
            SSC_flag_daily[i] = _worst_flag(flags)

    ssc_daily_by_day = dict(zip(unique_ssc_days, zip(SSC_daily, SSC_flag_daily)))

    # Step 5: Aggregate direct source SSL by day (preserved for SSC-only days)
    unique_ssl_days, ssl_day_idx = np.unique(days_ssl, return_inverse=True)
    n_ssl_days = len(unique_ssl_days)
    SSL_source_daily = np.full(n_ssl_days, np.nan, dtype=float)
    SSL_source_flag_daily = np.full(n_ssl_days, np.int8(9), dtype=np.int8)

    for i in range(n_ssl_days):
        mask = ssl_day_idx == i
        vals = SSL_c[mask]
        flags = SSLf_c[mask]
        valid = np.isfinite(vals) & (vals >= 0) & (flags != 9)
        if valid.any():
            SSL_source_daily[i] = float(np.nanmean(vals[valid]))
            SSL_source_flag_daily[i] = _worst_flag(flags[valid])

    ssl_daily_by_day = dict(zip(unique_ssl_days, zip(SSL_source_daily, SSL_source_flag_daily)))

    # Step 6: Build the unified daily timeline
    all_days = sorted(set(unique_q_days) | set(unique_ssc_days) | set(unique_ssl_days))
    n_out = len(all_days)
    if n_out == 0:
        n_out = 1
        all_days = ['1970-01-01']

    out_time = np.zeros(n_out, dtype=np.float64)
    out_Q = np.full(n_out, np.nan, dtype=float)
    out_SSC = np.full(n_out, np.nan, dtype=float)
    out_SSL = np.full(n_out, np.nan, dtype=float)
    out_Q_flag = np.full(n_out, np.int8(9), dtype=np.int8)
    out_SSC_flag = np.full(n_out, np.int8(9), dtype=np.int8)
    out_SSL_flag = np.full(n_out, np.int8(9), dtype=np.int8)
    out_Q_der = np.zeros(n_out, dtype=bool)
    out_SSC_der = np.zeros(n_out, dtype=bool)
    out_SSL_der = np.zeros(n_out, dtype=bool)

    for idx, day in enumerate(all_days):
        # Use noon UTC as representative timestamp
        try:
            dt = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out_time[idx] = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() + 43200.0

        q_val, q_fl = q_daily_by_day.get(day, (np.nan, np.int8(9)))
        ssc_val, ssc_fl = ssc_daily_by_day.get(day, (np.nan, np.int8(9)))
        ssl_val, ssl_fl = ssl_daily_by_day.get(day, (np.nan, np.int8(9)))

        # Q
        out_Q[idx] = q_val
        out_Q_flag[idx] = q_fl
        out_Q_der[idx] = False

        # SSC
        out_SSC[idx] = ssc_val
        out_SSC_flag[idx] = ssc_fl
        out_SSC_der[idx] = False

        # SSL: prefer recalculated from daily Q and SSC
        q_valid = np.isfinite(q_val) and q_val >= 0 and q_fl != 9
        ssc_valid = np.isfinite(ssc_val) and ssc_val >= 0 and ssc_fl != 9

        if q_valid and ssc_valid:
            out_SSL[idx] = float(q_val * ssc_val * float(ssl_factor))
            # Derived SSL: flag = estimated (1), propagate suspect/bad from inputs
            if q_fl == 3 or ssc_fl == 3:
                out_SSL_flag[idx] = np.int8(3)
            elif q_fl == 2 or ssc_fl == 2:
                out_SSL_flag[idx] = np.int8(2)
            else:
                out_SSL_flag[idx] = np.int8(1)
            out_SSL_der[idx] = True
        elif np.isfinite(ssl_val) and ssl_val >= 0 and ssl_fl != 9:
            # Keep direct source SSL for sediment-only days
            out_SSL[idx] = ssl_val
            out_SSL_flag[idx] = ssl_fl
            out_SSL_der[idx] = False
        else:
            out_SSL[idx] = np.nan
            out_SSL_flag[idx] = np.int8(9)
            out_SSL_der[idx] = False

    # Keep all days that have at least one valid variable
    keep = np.isfinite(out_Q) | np.isfinite(out_SSC) | np.isfinite(out_SSL)
    if not keep.any():
        keep[:] = True

    return {
        "time": out_time[keep],
        "Q": out_Q[keep],
        "SSC": out_SSC[keep],
        "SSL": out_SSL[keep],
        "Q_flag": out_Q_flag[keep],
        "SSC_flag": out_SSC_flag[keep],
        "SSL_flag": out_SSL_flag[keep],
        "Q_derived_mask": out_Q_der[keep],
        "SSC_derived_mask": out_SSC_der[keep],
        "SSL_derived_mask": out_SSL_der[keep],
    }
