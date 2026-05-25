#!/usr/bin/env python3
"""Fixed EUSEDcollab processing entrypoint.

This wrapper keeps the original EUSEDcollab processing workflow but replaces the
SSC unit conversion logic before running the pipeline.

Why this exists
---------------
The original ``process_eusedcollab_to_cf18_wzx.py`` converts:

    kg m-3 -> mg L-1 using x1,000,000
    g  m-3 -> mg L-1 using x1,000

Both are too large by a factor of 1000 for concentration units:

    1 kg m-3 = 1000 mg L-1
    1 g  m-3 = 1 mg L-1

Using the old conversion makes SSC 1000x too high and therefore makes existing
SSL look 1000x too low when checked against:

    SSL ton/day = Q m3/s * SSC mg/L * 0.0864

Run this file instead of the old EUSEDcollab processor when regenerating the
EUSEDcollab monthly NetCDF files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import process_eusedcollab_to_cf18_wzx as original


def detect_and_convert_columns_fixed(df):
    """Detect and convert Q, SSC, SSL to m3/s, mg/L, and ton/day.

    This is a drop-in replacement for
    ``process_eusedcollab_to_cf18_wzx.detect_and_convert_columns`` with corrected
    concentration conversions:

    - kg m-3 -> mg L-1: multiply by 1e3
    - g  m-3 -> mg L-1: multiply by 1
    """
    df = df.copy()

    if "date" in df.columns:
        days_in_month = df["date"].dt.days_in_month
    else:
        days_in_month = None

    # ---- Q ----
    q_col = next((c for c in df.columns if c.lower().startswith("q") and "(" in c.lower()), None)
    if q_col:
        col = q_col.lower()
        if "event" in col:
            df["Q_event"] = pd.to_numeric(df[q_col], errors="coerce")
        elif "m-1" in col and days_in_month is not None:
            df["Q"] = pd.to_numeric(df[q_col], errors="coerce") / (days_in_month * 86400.0)
        elif "d-1" in col:
            df["Q"] = pd.to_numeric(df[q_col], errors="coerce") / 86400.0
        elif "s-1" in col or "ts-1" in col or "/s" in col:
            df["Q"] = pd.to_numeric(df[q_col], errors="coerce")
        else:
            df["Q"] = pd.to_numeric(df[q_col], errors="coerce")
    else:
        df["Q"] = np.nan

    # ---- SSC ----
    ssc_col = next((c for c in df.columns if "ssc" in c.lower() or "turbidity" in c.lower()), None)
    if ssc_col:
        col = ssc_col.lower()
        if "kg" in col and "m-3" in col:
            # Correct: 1 kg/m3 = 1000 mg/L.
            df["SSC"] = pd.to_numeric(df[ssc_col], errors="coerce") * 1e3
        elif "g" in col and "m-3" in col:
            # Correct: 1 g/m3 = 1 mg/L.
            df["SSC"] = pd.to_numeric(df[ssc_col], errors="coerce")
        elif "turbidity" in col:
            df["SSC"] = pd.to_numeric(df[ssc_col], errors="coerce")
            df["SSC_flag"] = original.FLAG_ESTIMATED
        else:
            df["SSC"] = np.nan
    else:
        df["SSC"] = np.nan

    # ---- SSL ----
    ssl_col = next((c for c in df.columns if "ssl" in c.lower()), None)
    if ssl_col:
        col = ssl_col.lower()
        if "kg" in col and "m-1" in col and days_in_month is not None:
            df["SSL"] = pd.to_numeric(df[ssl_col], errors="coerce") / days_in_month / 1000.0
        elif "kg" in col and "d-1" in col:
            df["SSL"] = pd.to_numeric(df[ssl_col], errors="coerce") / 1000.0
        elif "kg" in col and "event" in col:
            df["SSL_event"] = pd.to_numeric(df[ssl_col], errors="coerce")
        elif ("t" in col or "ton" in col) and "event" in col:
            df["SSL_event"] = pd.to_numeric(df[ssl_col], errors="coerce")
        else:
            df["SSL"] = np.nan
    else:
        df["SSL"] = np.nan

    return df


# Monkey-patch the original module before running the original main workflow.
original.detect_and_convert_columns = detect_and_convert_columns_fixed
original.RUN_IN_PARALLEL = False  # ensure the patched function is used reliably across platforms


if __name__ == "__main__":
    original.main()
