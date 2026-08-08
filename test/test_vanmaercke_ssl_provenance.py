#!/usr/bin/env python3
"""Regression tests for Vanmaercke SSL provenance flag."""

import sys
import numpy as np
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import (
    FLAG_ESTIMATED,
    FLAG_MISSING,
    FLAG_SUSPECT,
    FLAG_BAD,
)
from code.qc import apply_quality_flag, propagate_derived_flag_from_inputs


def test_valid_sy_area_gives_ssl_flag_estimated():
    """Valid sediment_yield + area -> SSL calculated -> flag=1 (estimated)."""
    sy_val, area = 365.25, 100.0
    ssl_val = sy_val * area / 365.25  # = 100.0 ton/day

    sy_flag = apply_quality_flag(sy_val, "sediment_yield")
    area_flag = apply_quality_flag(area, "upstream_area")
    ssl_flag = propagate_derived_flag_from_inputs(
        derived_value=ssl_val,
        derived_flag=FLAG_ESTIMATED,
        input_flags=[sy_flag, area_flag],
        input_values=[sy_val, area],
    )

    np.testing.assert_equal(ssl_flag, FLAG_ESTIMATED)


def test_missing_ssl_gives_flag_9():
    """Missing/invalid SSL -> flag=9."""
    ssl_flag = propagate_derived_flag_from_inputs(
        derived_value=np.nan,
        derived_flag=FLAG_MISSING,
        input_flags=[0, 0],
        input_values=[365.25, 100.0],
    )
    np.testing.assert_equal(ssl_flag, FLAG_MISSING)


def test_bad_input_propagates_to_ssl():
    """Bad input (negative area) -> propagated to derived SSL flag=3."""
    sy_flag = apply_quality_flag(365.25, "sediment_yield")   # 0 good
    area_flag = apply_quality_flag(-100.0, "upstream_area")  # 3 bad
    ssl_flag = propagate_derived_flag_from_inputs(
        derived_value=100.0,
        derived_flag=FLAG_ESTIMATED,
        input_flags=[sy_flag, area_flag],
        input_values=[365.25, -100.0],
    )
    np.testing.assert_equal(ssl_flag, FLAG_BAD)


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASS {}".format(name))


if __name__ == "__main__":
    main()
