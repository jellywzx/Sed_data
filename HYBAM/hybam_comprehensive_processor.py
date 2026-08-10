#!/usr/bin/env python3
"""HYBAM processor with explicit raw-series daily aggregation before QC."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FILL_VALUE_FLOAT
from code.daily_aggregation import aggregate_unix_series_to_daily, align_daily_series

_IMPL_PATH = Path(__file__).with_name("hybam_comprehensive_processor_impl.py")
_SPEC = importlib.util.spec_from_file_location("hybam_processor_impl", _IMPL_PATH)
_impl = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_impl)

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def _daily_merge_discharge_ssc(self, discharge_file, ssc_file):
    """Aggregate raw Q and SSC independently to UTC-day means, then align them.

    This replaces the former nearest-SSC-to-every-Q-timestamp mapping, which
    could duplicate one SSC observation across several sub-daily discharge
    timestamps. When both source series exist, the legacy overlap-period rule
    is preserved, but the aligned daily axis is the union of Q and SSC days in
    that overlap. Existing downstream QC then recalculates daily SSL from the
    daily Q and SSC values.
    """
    result = {
        "time": None,
        "discharge": None,
        "ssc": None,
        "discharge_origin": None,
        "discharge_quality": None,
        "ssc_origin": None,
        "ssc_quality": None,
        "time_coverage_start": None,
        "time_coverage_end": None,
        "q_start": None,
        "q_end": None,
        "ssc_start": None,
        "ssc_end": None,
    }

    q_raw_t = q_raw_v = None
    s_raw_t = s_raw_v = None
    q_daily_t = q_daily_v = np.asarray([], dtype=float)
    s_daily_t = s_daily_v = np.asarray([], dtype=float)

    if discharge_file:
        q_raw_t, q_raw_v, _, q_fill, q_origin, q_quality = self.read_nc_data(discharge_file)
        result["discharge_raw"] = q_raw_v
        result["discharge_fill"] = q_fill
        result["discharge_origin"] = q_origin
        result["discharge_quality"] = q_quality
        q_daily_t, q_daily_v = aggregate_unix_series_to_daily(
            q_raw_t,
            q_raw_v,
            fill_values=(q_fill, FILL_VALUE_FLOAT),
        )

    if ssc_file:
        s_raw_t, s_raw_v, _, s_fill, s_origin, s_quality = self.read_nc_data(ssc_file)
        result["ssc_raw"] = s_raw_v
        result["ssc_fill"] = s_fill
        result["ssc_origin"] = s_origin
        result["ssc_quality"] = s_quality
        s_daily_t, s_daily_v = aggregate_unix_series_to_daily(
            s_raw_t,
            s_raw_v,
            fill_values=(s_fill, FILL_VALUE_FLOAT),
        )

    if q_raw_t is not None and s_raw_t is not None:
        q_valid_t = np.asarray(q_raw_t, dtype=float)
        s_valid_t = np.asarray(s_raw_t, dtype=float)
        q_valid_t = q_valid_t[np.isfinite(q_valid_t)]
        s_valid_t = s_valid_t[np.isfinite(s_valid_t)]

        if q_valid_t.size and s_valid_t.size:
            overlap_start = max(float(np.min(q_valid_t)), float(np.min(s_valid_t)))
            overlap_end = min(float(np.max(q_valid_t)), float(np.max(s_valid_t)))
            if overlap_start <= overlap_end:
                day_start = np.floor(overlap_start / 86400.0) * 86400.0
                day_end = np.floor(overlap_end / 86400.0) * 86400.0
                time_axis, aligned = align_daily_series(
                    {
                        "discharge": (q_daily_t, q_daily_v),
                        "ssc": (s_daily_t, s_daily_v),
                    },
                    start=day_start,
                    end=day_end,
                )
                result["time"] = time_axis
                result["discharge"] = aligned["discharge"]
                result["ssc"] = aligned["ssc"]
            else:
                result["time"] = np.asarray([], dtype=float)
                result["discharge"] = np.asarray([], dtype=float)
                result["ssc"] = np.asarray([], dtype=float)
        else:
            result["time"] = np.asarray([], dtype=float)
            result["discharge"] = np.asarray([], dtype=float)
            result["ssc"] = np.asarray([], dtype=float)
    elif q_raw_t is not None:
        result["time"] = q_daily_t
        result["discharge"] = q_daily_v
        result["ssc"] = np.full(q_daily_t.size, np.nan, dtype=float)
    elif s_raw_t is not None:
        result["time"] = s_daily_t
        result["ssc"] = s_daily_v
        result["discharge"] = np.full(s_daily_t.size, np.nan, dtype=float)

    if result["time"] is not None and len(result["time"]) > 0:
        result["time_coverage_start"] = datetime.utcfromtimestamp(
            float(result["time"][0])
        ).strftime("%Y-%m-%d")
        result["time_coverage_end"] = datetime.utcfromtimestamp(
            float(result["time"][-1])
        ).strftime("%Y-%m-%d")

    before_q = 0 if q_raw_t is None else int(np.size(q_raw_t))
    before_s = 0 if s_raw_t is None else int(np.size(s_raw_t))
    after_q = int(q_daily_t.size)
    after_s = int(s_daily_t.size)
    if before_q > after_q or before_s > after_s:
        print(
            "    Daily aggregation: Q {0}->{1} rows; SSC {2}->{3} rows."
            .format(before_q, after_q, before_s, after_s)
        )

    return result


_impl.HYBAMProcessor.merge_discharge_ssc = _daily_merge_discharge_ssc
HYBAMProcessor = _impl.HYBAMProcessor
globals()["HYBAMProcessor"] = HYBAMProcessor


if __name__ == "__main__":
    _impl.main()
