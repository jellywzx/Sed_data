#!/usr/bin/env python3
"""EUSEDcollab processor with explicit sub-daily to daily consolidation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.daily_aggregation import aggregate_eused_to_daily

_IMPL_PATH = Path(__file__).with_name("process_eusedcollab_to_cf18_wzx_impl.py")
_SPEC = importlib.util.spec_from_file_location("eusedcollab_processor_impl", _IMPL_PATH)
_impl = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_impl)

# Preserve the public surface of the existing script so imports/tests keep working.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

_read_station_data_impl = _impl.read_station_data


def read_station_data(station_id, country):
    """Read/standardize a station and collapse all same-day rows to daily values.

    The original parser still performs source-unit conversion and conservative
    same-record Q/SSC/SSL derivation first. This wrapper then applies the daily
    rule used by the release: arithmetic daily means for Q and SSC, followed by
    recalculation of SSL from daily Q and SSC when both are available. SSL-only
    days are retained using the mean of the available source/derived SSL values.
    """
    df = _read_station_data_impl(station_id, country)
    if df is None or len(df) == 0:
        return df

    dates = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    counts = dates.value_counts(dropna=True)
    n_problem_days = int((counts > 1).sum())
    before = len(df)

    daily = aggregate_eused_to_daily(
        df,
        fill_value=getattr(_impl, "FILL_VALUE", -9999.0),
        factor=0.0864,
    )

    if n_problem_days:
        print(
            f"  Daily aggregation: collapsed {before} rows to {len(daily)} daily rows "
            f"across {n_problem_days} day(s) with multiple records."
        )

    return daily


# The implementation module resolves globals inside process_station at runtime,
# so patch its reader as well as this wrapper's exported name.
_impl.read_station_data = read_station_data
globals()["read_station_data"] = read_station_data


if __name__ == "__main__":
    _impl.main()
