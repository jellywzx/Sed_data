#!/usr/bin/env python3
"""Focused EUSEDcollab unit-conversion regressions."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parent / "process_eusedcollab_to_cf18_wzx.py"
SOURCE_DIR = Path("/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Source/EUSEDcollab")


def _install_import_stubs():
    package = types.ModuleType("code")
    package.__path__ = []
    sys.modules["code"] = package

    constants = types.ModuleType("code.constants")
    constants.FILL_VALUE_FLOAT = np.float32(-9999.0)
    constants.FILL_VALUE_INT = np.int8(-99)
    sys.modules["code.constants"] = constants

    plot = types.ModuleType("code.plot")
    plot.plot_ssc_q_diagnostic = lambda *args, **kwargs: None
    sys.modules["code.plot"] = plot

    qc = types.ModuleType("code.qc")
    qc.apply_hydro_qc_with_provenance = lambda *args, **kwargs: None
    qc.apply_quality_flag = lambda value, name: 9 if pd.isna(value) else (3 if value < 0 else 0)
    qc.apply_quality_flag_array = lambda values, name: np.zeros(len(values), dtype=np.int8)
    qc.build_ssc_q_envelope = lambda *args, **kwargs: None
    qc.check_ssc_q_consistency = lambda *args, **kwargs: (False, np.nan)
    qc.compute_log_iqr_bounds = lambda values, k=1.5: (None, None)
    qc.propagate_ssc_q_inconsistency_to_ssl = lambda *args, **kwargs: None
    sys.modules["code.qc"] = qc

    runtime = types.ModuleType("code.runtime")
    runtime.resolve_output_root = lambda start=None: Path(tempfile.gettempdir()) / "eused_output"
    runtime.resolve_source_root = lambda start=None: Path(tempfile.gettempdir()) / "eused_source"
    sys.modules["code.runtime"] = runtime

    units = types.ModuleType("code.units")
    units.convert_ssl_units_if_needed = lambda *args, **kwargs: None
    sys.modules["code.units"] = units


def _load_module():
    _install_import_stubs()
    spec = importlib.util.spec_from_file_location("eused_processor_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def assert_close(actual, expected, rel=1e-9, abs_tol=1e-9):
    assert math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol), (actual, expected)


def test_kg_m3_to_mg_l():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01"]),
        "Q (m3 d-1)": [86400.0],
        "SSC (kg m-3)": [2.753],
        "SSL (kg d-1)": [1000.0],
    })
    out = mod.detect_and_convert_columns(df)
    assert_close(out.loc[0, "Q"], 1.0)
    assert_close(out.loc[0, "SSC"], 2753.0)
    assert_close(out.loc[0, "SSL"], 1.0)


def test_g_m3_preferred_over_turbidity_and_non_concentration_ssc():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01"]),
        "Q (m3 d-1)": [86400.0],
        "Turbidity (NTU)": [999.0],
        "SSC (g m-3)": [622.975],
        "SSC (kg m-1)": [0.622975],
        "SSL (kg d-1)": [1000.0],
    })
    out = mod.detect_and_convert_columns(df)
    assert_close(out.loc[0, "SSC"], 622.975)


def test_turbidity_is_not_direct_ssc():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01"]),
        "Q (m3 d-1)": [86400.0],
        "Turbidity (NTU)": [50.0],
    })
    out = mod.detect_and_convert_columns(df)
    assert np.isnan(out.loc[0, "SSC"])


def test_monthly_q_ssl_can_derive_ssc():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        q_ssl = source / "Q_SSL"
        q_ssl.mkdir()
        pd.DataFrame({
            "Date (DD/MM/YYYY)": ["2004-01-01"],
            "Q (m3 m-1)": [31.0 * 86400.0],
            "SSL (kg m-1)": [31.0 * 1000.0 * 8.64],
        }).to_csv(q_ssl / "ID_34_Q_SSL_DK.csv", index=False)

        old_source = mod.SOURCE_DIR
        mod.SOURCE_DIR = str(source)
        try:
            out = mod.read_station_data(34, "DK")
        finally:
            mod.SOURCE_DIR = old_source

    assert out is not None
    assert_close(out.loc[0, "Q"], 1.0)
    assert_close(out.loc[0, "SSL"], 8.64)
    assert_close(out.loc[0, "SSC"], 100.0)
    assert bool(out.loc[0, "SSC_derived"])
    assert not bool(out.loc[0, "Q_derived"])
    assert not bool(out.loc[0, "SSL_derived"])


def test_event_duration_uses_dayfirst_dates():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-02-01"]),
        "Start date (DD/MM/YYYY)": ["01/02/2000"],
        "End date (DD/MM/YYYY)": ["03/02/2000"],
        "Q (m3 event-1)": [2.0 * 86400.0],
        "SSC (kg m-3)": [0.1],
        "SSL (kg event-1)": [2.0 * 1000.0 * 8.64],
    })
    out = mod.detect_and_convert_columns(df)
    assert_close(out.loc[0, "Q"], 1.0)
    assert_close(out.loc[0, "SSC"], 100.0)
    assert_close(out.loc[0, "SSL"], 8.64)


def test_timestep_q_ssl_use_interval_seconds():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 00:10:00"]),
        "Event_index": [1, 1],
        "Q (m3 ts-1)": [600.0, 600.0],
        "SSC (kg m-3)": [0.1, 0.1],
        "time_interval (hh:mm:ss)": ["00:10:00", "00:10:00"],
        "SSL (kg ts-1)": [60.0, 60.0],
    })
    out = mod.detect_and_convert_columns(df)
    assert_close(out.loc[0, "Q"], 1.0)
    assert_close(out.loc[0, "SSC"], 100.0)
    assert_close(out.loc[0, "SSL"], 8.64)


def test_real_source_id15_no_1000x_ssc_regression():
    if not SOURCE_DIR.exists():
        return
    old_source = mod.SOURCE_DIR
    mod.SOURCE_DIR = str(SOURCE_DIR)
    try:
        out = mod.read_station_data(15, "GR")
    finally:
        mod.SOURCE_DIR = old_source
    ssc = out["SSC"].replace(mod.FILL_VALUE, np.nan)
    assert_close(np.nanmax(ssc), 2753.0)


def test_real_source_id25_prefers_g_m3_not_turbidity():
    if not SOURCE_DIR.exists():
        return
    old_source = mod.SOURCE_DIR
    mod.SOURCE_DIR = str(SOURCE_DIR)
    try:
        out = mod.read_station_data(25, "SI")
    finally:
        mod.SOURCE_DIR = old_source
    ssc = out["SSC"].replace(mod.FILL_VALUE, np.nan)
    assert_close(np.nanmax(ssc), 622.975225, rel=1e-6)


def test_real_source_timestep_records_are_standardized():
    if not SOURCE_DIR.exists():
        return
    old_source = mod.SOURCE_DIR
    mod.SOURCE_DIR = str(SOURCE_DIR)
    try:
        id5 = mod.read_station_data(5, "BE")
        id6 = mod.read_station_data(6, "BE")
        id10 = mod.read_station_data(10, "FR")
    finally:
        mod.SOURCE_DIR = old_source

    assert np.nanmax(id5["SSC"].replace(mod.FILL_VALUE, np.nan)) < 100000.0
    assert np.nanmax(id6["SSC"].replace(mod.FILL_VALUE, np.nan)) < 500000.0
    assert_close(np.nanmax(id10["SSC"].replace(mod.FILL_VALUE, np.nan)), 5003.058638, rel=1e-6)
    assert np.nanmax(id10["SSL"].replace(mod.FILL_VALUE, np.nan)) < 10000.0


def test_real_source_id20_extreme_values_are_not_silently_reinterpreted():
    if not SOURCE_DIR.exists():
        return
    old_source = mod.SOURCE_DIR
    mod.SOURCE_DIR = str(SOURCE_DIR)
    try:
        out = mod.read_station_data(20, "PL")
    finally:
        mod.SOURCE_DIR = old_source
    ssc = out["SSC"].replace(mod.FILL_VALUE, np.nan)
    assert_close(np.nanmax(ssc), 56178000.0)


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
