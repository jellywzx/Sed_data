"""
Daily aggregation helpers for harmonized sediment dataset processing.

Provides functions to:
1. Collapse duplicate timestamps (same exact time -> single observation)
2. Aggregate sub-daily Q and SSC to daily means
3. Re-derive daily SSL from daily Q and SSC only when source SSL is absent

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
    3. Q_daily = arithmetic mean of valid source Q in the day, or derived Q
       only when no source Q is available.
    4. SSC_daily = arithmetic mean of valid source SSC in the day, or derived
       SSC only when no source SSC is available.
    5. SSL_daily = valid source SSL when available. If no source SSL exists,
       derive from daily Q and SSC when possible, otherwise aggregate derived
       SSL records.

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

    def _calendar_dates(ts_arr):
        """Convert epoch seconds to UTC date strings (YYYY-MM-DD)."""
        result = np.empty(len(ts_arr), dtype='<U10')
        for i, t in enumerate(ts_arr):
            result[i] = datetime.fromtimestamp(float(t), tz=timezone.utc).strftime('%Y-%m-%d')
        return result

    days = _calendar_dates(ts)
    all_days = sorted(set(days))
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

    def _has_valid(vals, flags):
        return np.isfinite(vals) & (vals >= 0) & (flags != 9)

    def _aggregate_selected(sel, vals, flags):
        if not np.any(sel):
            return np.nan, np.int8(9), False

        ts_c, vals_c, flags_c = collapse_duplicate_timestamps(
            ts[sel],
            vals[sel],
            flags[sel],
            fill_value=fill_value,
        )
        del ts_c
        valid = _has_valid(vals_c, flags_c)
        if not np.any(valid):
            return np.nan, np.int8(9), False

        return (
            float(np.nanmean(vals_c[valid])),
            _worst_flag(flags_c[valid]),
            True,
        )

    def _aggregate_with_source_priority(day_mask, vals, flags, derived_mask):
        valid = _has_valid(vals, flags)

        source_sel = day_mask & (~derived_mask) & valid
        value, flag, present = _aggregate_selected(source_sel, vals, flags)
        if present:
            return value, flag, False, True

        derived_sel = day_mask & derived_mask & valid
        value, flag, present = _aggregate_selected(derived_sel, vals, flags)
        if present:
            return value, flag, True, True

        return np.nan, np.int8(9), False, False

    for idx, day in enumerate(all_days):
        # Use noon UTC as representative timestamp
        try:
            dt = datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out_time[idx] = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() + 43200.0

        day_mask = days == day
        q_val, q_fl, q_is_derived, q_present = _aggregate_with_source_priority(
            day_mask, Q_raw, Qf, Q_der
        )
        ssc_val, ssc_fl, ssc_is_derived, ssc_present = _aggregate_with_source_priority(
            day_mask, SSC_raw, SSCf, SSC_der
        )
        ssl_val, ssl_fl, ssl_is_derived, ssl_present = _aggregate_with_source_priority(
            day_mask, SSL_raw, SSLf, SSL_der
        )

        out_Q[idx] = q_val
        out_Q_flag[idx] = q_fl
        out_Q_der[idx] = q_is_derived and q_present

        out_SSC[idx] = ssc_val
        out_SSC_flag[idx] = ssc_fl
        out_SSC_der[idx] = ssc_is_derived and ssc_present

        # SSL: source-reported records outrank formula-derived daily SSL.
        q_valid = np.isfinite(q_val) and q_val >= 0 and q_fl != 9
        ssc_valid = np.isfinite(ssc_val) and ssc_val >= 0 and ssc_fl != 9

        if ssl_present and not ssl_is_derived:
            out_SSL[idx] = ssl_val
            out_SSL_flag[idx] = ssl_fl
            out_SSL_der[idx] = False
        elif q_valid and ssc_valid:
            out_SSL[idx] = float(q_val * ssc_val * float(ssl_factor))
            # Derived SSL: flag = estimated (1), propagate suspect/bad from inputs
            if q_fl == 3 or ssc_fl == 3:
                out_SSL_flag[idx] = np.int8(3)
            elif q_fl == 2 or ssc_fl == 2:
                out_SSL_flag[idx] = np.int8(2)
            else:
                out_SSL_flag[idx] = np.int8(1)
            out_SSL_der[idx] = True
        elif ssl_present:
            out_SSL[idx] = ssl_val
            out_SSL_flag[idx] = ssl_fl
            out_SSL_der[idx] = True
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
