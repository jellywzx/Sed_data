"""Utilities for collapsing sub-daily observations to one record per UTC/calendar day.

The two-stage aggregation used here first collapses exact duplicate timestamps
and only then computes a daily mean. This prevents duplicated rows at one
instant from receiving more weight than other genuine within-day observations.
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

    Exact duplicate timestamps are averaged first. The resulting unique
    timestamps are then averaged by calendar day for Q and SSC. When both daily
    Q and SSC are available, daily SSL is recalculated as ``Q * SSC * factor``.
    If a day lacks Q or SSC but contains source/previously-derived SSL, daily
    SSL falls back to the arithmetic mean of available SSL values so
    sediment-only days are retained.

    Derived masks are propagated conservatively with ``any``; recalculated SSL
    is always marked derived.
    """
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
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

    # Stage 1: exact-timestamp duplicates should count as one sampling instant.
    exact_values = (
        out.groupby("date", sort=True, as_index=False)[["Q", "SSC", "SSL"]]
        .mean(numeric_only=True)
    )
    exact_masks = (
        out.groupby("date", sort=True, as_index=False)[
            ["Q_derived", "SSC_derived", "SSL_derived"]
        ]
        .any()
    )
    exact = exact_values.merge(exact_masks, on="date", how="left", validate="1:1")
    exact["day"] = exact["date"].dt.floor("D")

    # Stage 2: arithmetic daily means over unique sampling instants.
    daily_values = (
        exact.groupby("day", sort=True, as_index=False)[["Q", "SSC", "SSL"]]
        .mean(numeric_only=True)
        .rename(columns={"day": "date"})
    )
    daily_masks = (
        exact.groupby("day", sort=True, as_index=False)[
            ["Q_derived", "SSC_derived", "SSL_derived"]
        ]
        .any()
        .rename(columns={"day": "date"})
    )
    daily = daily_values.merge(daily_masks, on="date", how="left", validate="1:1")

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

    Exact duplicate Unix timestamps are averaged first, then unique sampling
    instants are averaged within UTC calendar days.
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
    exact = frame.groupby("time", sort=True, as_index=False)["value"].mean()
    exact["day"] = np.floor(exact["time"] / 86400.0) * 86400.0
    daily = exact.groupby("day", sort=True, as_index=False)["value"].mean()
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
