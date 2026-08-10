"""Utilities for collapsing sub-daily observations to one record per UTC/calendar day.

These helpers are intentionally small and source-agnostic. They are used by
EUSEDcollab and HYBAM so the source-level products entering the daily workflow
cannot contain multiple timestamps for the same day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(values, fill_values=()):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    for fill in fill_values:
        try:
            fill = float(fill)
        except (TypeError, ValueError):
            continue
        arr[np.isclose(arr, fill, rtol=1e-5, atol=1e-5)] = np.nan
    return arr


def aggregate_eused_to_daily(df, *, fill_value=-9999.0, factor=0.0864):
    """Collapse EUSEDcollab rows to one record per calendar day.

    Q and SSC are arithmetic daily means of all available within-day values.
    When both daily Q and daily SSC are available, daily SSL is recalculated as
    ``Q * SSC * factor`` rather than averaging sub-daily SSL. If a day lacks Q
    or SSC but contains source/previously-derived SSL, its daily SSL falls back
    to the arithmetic mean of those available SSL values so sediment-only days
    are not discarded.

    Derived masks are propagated conservatively with ``any``; recalculated SSL
    is always marked derived.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out = out[out["date"].notna()].copy()
    if out.empty:
        return out

    for var in ("Q", "SSC", "SSL"):
        if var not in out.columns:
            out[var] = np.nan
        out[var] = _as_float(out[var], fill_values=(fill_value,))

    for mask_col in ("Q_derived", "SSC_derived", "SSL_derived"):
        if mask_col not in out.columns:
            out[mask_col] = False
        out[mask_col] = out[mask_col].fillna(False).astype(bool)

    grouped = out.groupby("date", sort=True, as_index=False)
    daily = grouped[["Q", "SSC", "SSL"]].mean(numeric_only=True)

    mask_daily = grouped[["Q_derived", "SSC_derived", "SSL_derived"]].any()
    daily = daily.merge(mask_daily, on="date", how="left", validate="1:1")

    paired = np.isfinite(daily["Q"].to_numpy(dtype=float)) & np.isfinite(
        daily["SSC"].to_numpy(dtype=float)
    )
    daily.loc[paired, "SSL"] = (
        daily.loc[paired, "Q"] * daily.loc[paired, "SSC"] * float(factor)
    )
    daily.loc[paired, "SSL_derived"] = True

    for var in ("Q", "SSC", "SSL"):
        daily[var] = pd.to_numeric(daily[var], errors="coerce").fillna(float(fill_value))
    for mask_col in ("Q_derived", "SSC_derived", "SSL_derived"):
        daily[mask_col] = daily[mask_col].fillna(False).astype(bool)

    return daily.sort_values("date").reset_index(drop=True)


def aggregate_unix_series_to_daily(time_seconds, values, *, fill_values=()):
    """Return unique UTC-day timestamps and arithmetic daily means.

    Parameters
    ----------
    time_seconds : array-like
        Unix timestamps in seconds.
    values : array-like
        Observation values aligned with ``time_seconds``.
    fill_values : iterable
        Numeric fill values to treat as missing.
    """
    if time_seconds is None or values is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = _as_float(time_seconds)
    v = _as_float(values, fill_values=fill_values)
    n = min(t.size, v.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = t[:n]
    v = v[:n]
    valid_time = np.isfinite(t)
    if not np.any(valid_time):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    frame = pd.DataFrame({"time": t[valid_time], "value": v[valid_time]})
    frame["day"] = np.floor(frame["time"] / 86400.0) * 86400.0
    daily = frame.groupby("day", sort=True, as_index=False)["value"].mean()
    return daily["day"].to_numpy(dtype=float), daily["value"].to_numpy(dtype=float)


def align_daily_series(series_by_name, *, start=None, end=None):
    """Align named ``(day, value)`` series on the union of unique daily dates."""
    days = []
    normalized = {}
    for name, (time_values, data_values) in series_by_name.items():
        t = np.asarray(time_values, dtype=float)
        v = np.asarray(data_values, dtype=float)
        n = min(t.size, v.size)
        t, v = t[:n], v[:n]
        if start is not None:
            keep = t >= float(start)
            t, v = t[keep], v[keep]
        if end is not None:
            keep = t <= float(end)
            t, v = t[keep], v[keep]
        normalized[name] = (t, v)
        if t.size:
            days.append(t)

    if not days:
        return np.asarray([], dtype=float), {
            name: np.asarray([], dtype=float) for name in series_by_name
        }

    union = np.unique(np.concatenate(days))
    union.sort()
    aligned = {}
    for name, (t, v) in normalized.items():
        values_out = np.full(union.size, np.nan, dtype=float)
        if t.size:
            pos = np.searchsorted(union, t)
            values_out[pos] = v
        aligned[name] = values_out
    return union, aligned
