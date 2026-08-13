#!/usr/bin/env python3
"""Focused EUSEDcollab unit-conversion regressions."""


import importlib.util
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import netCDF4 as nc

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_ROOT / "EUSEDcollab" / "process_eusedcollab_to_cf18_wzx.py"
SOURCE_DIR = Path("/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Source/EUSEDcollab")


def _load_repo_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_import_stubs():
    package = types.ModuleType("code")
    package.__path__ = [str(SCRIPT_ROOT / "code")]
    sys.modules["code"] = package

    _load_repo_module("code.constants", SCRIPT_ROOT / "code" / "constants.py")

    plot = types.ModuleType("code.plot")
    plot.plot_ssc_q_diagnostic = lambda *args, **kwargs: None
    sys.modules["code.plot"] = plot

    _load_repo_module("code.qc", SCRIPT_ROOT / "code" / "qc.py")

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


def _qc_input_frame(Q, SSC, SSL, *, ssc_derived_mask, ssl_derived_mask=None):
    n = len(Q)
    if ssl_derived_mask is None:
        ssl_derived_mask = np.zeros(n, dtype=bool)
    return pd.DataFrame({
        "date": pd.to_datetime("2000-01-01") + pd.to_timedelta(np.arange(n), unit="D"),
        "Q": np.asarray(Q, dtype=float),
        "SSC": np.asarray(SSC, dtype=float),
        "SSL": np.asarray(SSL, dtype=float),
        "Q_derived": np.zeros(n, dtype=bool),
        "SSC_derived": np.asarray(ssc_derived_mask, dtype=bool),
        "SSL_derived": np.asarray(ssl_derived_mask, dtype=bool),
    })


def test_shared_qc_marks_clean_derived_ssc_estimated():
    Q = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    SSC = np.array([10, 11, 10, 12, 11, 10], dtype=float)
    SSL = Q * SSC * 0.0864
    df = _qc_input_frame(
        Q,
        SSC,
        SSL,
        ssc_derived_mask=[True, False, False, False, False, False],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="clean_derived_ssc",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.loc[0, "Q_flag"]) == int(mod.FLAG_GOOD)
    assert int(df_qc.loc[0, "SSL_flag"]) == int(mod.FLAG_GOOD)
    assert int(df_qc.loc[0, "SSC_flag"]) == int(mod.FLAG_ESTIMATED)


def test_shared_qc_propagates_suspect_ssl_to_derived_ssc():
    Q = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    SSC = np.array([10, 11, 10, 12, 11, 10], dtype=float)
    SSL = np.array([0.864, 1.9008, 2.592, 4.1472, 4.752, 100000.0], dtype=float)
    df = _qc_input_frame(
        Q,
        SSC,
        SSL,
        ssc_derived_mask=[False, False, False, False, False, True],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="suspect_ssl_derived_ssc",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.iloc[-1]["Q_flag"]) == int(mod.FLAG_GOOD)
    assert int(df_qc.iloc[-1]["SSL_flag"]) == int(mod.FLAG_SUSPECT)
    assert int(df_qc.iloc[-1]["SSC_flag"]) == int(mod.FLAG_SUSPECT)


def test_shared_qc_marks_source_and_derived_ssl_records():
    df = _qc_input_frame(
        Q=[1, 2, 3, 4],
        SSC=[10, 10, 10, 10],
        SSL=[0.864, 1.728, 2.592, 3.456],
        ssc_derived_mask=[False, False, False, False],
        ssl_derived_mask=[False, True, False, False],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="mixed_ssl",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.loc[0, "SSL_flag"]) == int(mod.FLAG_GOOD)
    assert int(df_qc.loc[1, "SSL_flag"]) == int(mod.FLAG_ESTIMATED)


def test_shared_qc_does_not_propagate_suspect_q_to_source_ssl():
    df = _qc_input_frame(
        Q=[1, 2, 3, 4, 5, 100000],
        SSC=[10, 10, 10, 10, 10, 10],
        SSL=[100, 100, 100, 100, 100, 100],
        ssc_derived_mask=[False, False, False, False, False, False],
        ssl_derived_mask=[False, False, False, False, False, False],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="source_ssl",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.iloc[-1]["Q_flag"]) == int(mod.FLAG_SUSPECT)
    assert int(df_qc.iloc[-1]["SSL_flag"]) == int(mod.FLAG_GOOD)


def test_shared_qc_propagates_suspect_q_to_derived_ssl():
    df = _qc_input_frame(
        Q=[1, 2, 3, 4, 5, 100000],
        SSC=[10, 10, 10, 10, 10, 10],
        SSL=[100, 100, 100, 100, 100, 100],
        ssc_derived_mask=[False, False, False, False, False, False],
        ssl_derived_mask=[False, False, False, False, False, True],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="derived_ssl",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.iloc[-1]["Q_flag"]) == int(mod.FLAG_SUSPECT)
    assert int(df_qc.iloc[-1]["SSL_flag"]) == int(mod.FLAG_SUSPECT)


def test_shared_qc_does_not_propagate_suspect_ssl_to_source_ssc():
    df = _qc_input_frame(
        Q=[1, 2, 3, 4, 5, 6],
        SSC=[10, 11, 10, 12, 11, 10],
        SSL=[0.864, 1.9008, 2.592, 4.1472, 4.752, 100000.0],
        ssc_derived_mask=[False, False, False, False, False, False],
    )

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        df,
        station_id="synthetic",
        station_name="source_ssc",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert int(df_qc.iloc[-1]["SSL_flag"]) == int(mod.FLAG_SUSPECT)
    assert int(df_qc.iloc[-1]["SSC_flag"]) == int(mod.FLAG_GOOD)


def test_write_netcdf_records_mixed_ssc_ssl_provenance_metadata():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2000-01-02"]),
        "Q": [1.0, 2.0],
        "SSC": [10.0, 20.0],
        "SSL": [0.864, 3.456],
        "Q_derived": [False, False],
        "SSC_derived": [False, True],
        "SSL_derived": [False, True],
    })
    flags = np.array([mod.FLAG_GOOD, mod.FLAG_ESTIMATED], dtype=np.int8)
    step_flags = df.assign(
        Q_flag_qc1_physical=np.array([0, 0], dtype=np.int8),
        Q_flag_qc2_log_iqr=np.array([8, 8], dtype=np.int8),
        SSC_flag_qc1_physical=np.array([0, 0], dtype=np.int8),
        SSC_flag_qc2_log_iqr=np.array([8, 8], dtype=np.int8),
        SSC_flag_qc3_ssc_q=np.array([8, 8], dtype=np.int8),
        SSL_flag_qc1_physical=np.array([0, 0], dtype=np.int8),
        SSL_flag_qc2_log_iqr=np.array([8, 8], dtype=np.int8),
        SSL_flag_qc3_from_ssc_q=np.array([8, 8], dtype=np.int8),
    )
    metadata = {
        "station_name": "synthetic",
        "catchment_id": 1,
        "latitude": 1.0,
        "longitude": 2.0,
        "drainage_area": 3.0,
        "data_type": "Daily",
        "stream_type": "perennial",
        "country": "BE",
        "references": "synthetic",
        "contact_name": np.nan,
        "contact_email": np.nan,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eused.nc"
        mod.write_netcdf(df, metadata, flags, flags, flags, str(out), step_flags=step_flags)
        with nc.Dataset(out) as ds:
            ssc = ds.variables["SSC"]
            ssl = ds.variables["SSL"]

            assert ssc.long_name == "suspended sediment concentration"
            assert ssl.long_name == "suspended sediment load"
            assert "Mixed source-reported and derived" in ssc.source
            assert "Mixed source-reported and derived" in ssl.source
            assert "source-reported SSC" in ssc.provenance
            assert "derive SSC" in ssc.provenance
            assert "source-reported SSL" in ssl.provenance
            assert "derived from Q and SSC" in ssl.comment
            assert "SSC_derived_mask" in ssc.ancillary_variables
            assert "SSL_derived_mask" in ssl.ancillary_variables
            assert "SSC_flag_qc2_log_iqr" in ssc.ancillary_variables
            assert "SSL_flag_qc2_log_iqr" in ssl.ancillary_variables
            np.testing.assert_array_equal(ds.variables["SSC_derived_mask"][:], np.array([0, 1], dtype=np.int8))
            np.testing.assert_array_equal(ds.variables["SSL_derived_mask"][:], np.array([0, 1], dtype=np.int8))
            assert ds.temporal_resolution == "daily"
            step_names = [
                "Q_flag_qc1_physical",
                "Q_flag_qc2_log_iqr",
                "SSC_flag_qc1_physical",
                "SSC_flag_qc2_log_iqr",
                "SSC_flag_qc3_ssc_q",
                "SSL_flag_qc1_physical",
                "SSL_flag_qc2_log_iqr",
                "SSL_flag_qc3_from_ssc_q",
            ]
            for name in step_names:
                assert name in ds.variables
                assert ds.variables[name].shape == ds.variables["time"].shape
            np.testing.assert_array_equal(
                ds.variables["SSL_flag_qc3_from_ssc_q"].flag_values,
                np.array([0, 2, 8, 9], dtype=np.int8),
            )


def test_final_qc_after_daily_aggregation_uses_final_axis():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01 00:00:00", "2000-01-01 12:00:00"]),
        "Q": [1.0, 3.0],
        "SSC": [10.0, 20.0],
        "SSL": [mod.FILL_VALUE, mod.FILL_VALUE],
        "Q_derived": [False, False],
        "SSC_derived": [False, False],
        "SSL_derived": [False, False],
    })
    metadata = {"data_type": "Event data - fixed timestep"}

    normalized, aggregated, resolution = mod._normalize_station_resolution(df, metadata)

    assert aggregated
    assert resolution == "daily"
    assert len(normalized) == 1
    assert_close(normalized.loc[0, "Q"], 2.0)
    assert_close(normalized.loc[0, "SSC"], 15.0)
    assert_close(normalized.loc[0, "SSL"], 2.0 * 15.0 * 0.0864)
    assert bool(normalized.loc[0, "SSL_derived"])

    df_qc, _, _ = mod.apply_hydro_qc_with_provenance(
        normalized,
        station_id="synthetic",
        station_name="final_axis",
        iqr_k=mod.QC_IQR_K,
        min_samples_envelope=mod.QC_MIN_SAMPLES_ENVELOPE,
    )

    assert len(df_qc) == 1
    assert int(df_qc.loc[0, "SSL_flag"]) == int(mod.FLAG_ESTIMATED)
    assert int(df_qc.loc[0, "SSL_flag_qc1_physical"]) == int(mod.FLAG_GOOD)
    for col in [
        "Q_flag_qc1_physical",
        "Q_flag_qc2_log_iqr",
        "SSC_flag_qc1_physical",
        "SSC_flag_qc2_log_iqr",
        "SSC_flag_qc3_ssc_q",
        "SSL_flag_qc1_physical",
        "SSL_flag_qc2_log_iqr",
        "SSL_flag_qc3_from_ssc_q",
    ]:
        assert len(df_qc[col]) == len(normalized)


def test_temporal_resolution_helper_is_conservative():
    assert not mod._should_daily_aggregate("Monthly data")
    assert mod._resolve_final_temporal_resolution("Monthly data", False) == "monthly"
    assert mod._should_daily_aggregate("Daily data - fixed timestep")
    assert mod._resolve_final_temporal_resolution("Daily data - fixed timestep", True) == "daily"
    assert mod._should_daily_aggregate("Event data - variable timestep")
    assert mod._resolve_final_temporal_resolution("Event data - variable timestep", True) == "daily"
    assert not mod._should_daily_aggregate("Q and rating curve data")
    assert mod._resolve_final_temporal_resolution("Q and rating curve data", False) == "q_and_rating_curve_data"


def test_write_netcdf_keeps_monthly_temporal_resolution():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2000-02-01"]),
        "Q": [1.0, 2.0],
        "SSC": [10.0, 20.0],
        "SSL": [0.864, 3.456],
        "Q_derived": [False, False],
        "SSC_derived": [False, False],
        "SSL_derived": [False, False],
    })
    flags = np.array([mod.FLAG_GOOD, mod.FLAG_GOOD], dtype=np.int8)
    metadata = {
        "station_name": "synthetic_monthly",
        "catchment_id": 2,
        "latitude": 1.0,
        "longitude": 2.0,
        "drainage_area": 3.0,
        "data_type": "Monthly data",
        "temporal_resolution": "monthly",
        "stream_type": "perennial",
        "country": "BE",
        "references": "synthetic",
        "contact_name": np.nan,
        "contact_email": np.nan,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eused_monthly.nc"
        mod.write_netcdf(df, metadata, flags, flags, flags, str(out), step_flags=None)
        with nc.Dataset(out) as ds:
            assert ds.temporal_resolution == "monthly"


def test_seconds_mismatch_heuristic_does_not_modify_source_ssc():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp)
        q_ssl = source / "Q_SSL"
        q_ssl.mkdir()
        dates = pd.date_range("2000-01-01", periods=6, freq="D")
        pd.DataFrame({
            "Date (DD/MM/YYYY)": [d.strftime("%Y-%m-%d") for d in dates],
            "Q (m3 d-1)": np.full(6, 86400.0),
            "SSC (g m-3)": np.full(6, 100.0),
            "SSL (kg d-1)": np.full(6, 0.1),
        }).to_csv(q_ssl / "ID_88_Q_SSL_XX.csv", index=False)

        old_source = mod.SOURCE_DIR
        mod.SOURCE_DIR = str(source)
        try:
            out = mod.read_station_data(88, "XX")
        finally:
            mod.SOURCE_DIR = old_source

    ssc = out["SSC"].replace(mod.FILL_VALUE, np.nan)
    q = out["Q"].replace(mod.FILL_VALUE, np.nan)
    ssl = out["SSL"].replace(mod.FILL_VALUE, np.nan)
    ratio = ssl / (0.0864 * q * ssc)
    assert_close(np.nanmedian(ssc), 100.0)
    assert_close(np.nanmedian(ratio.replace([np.inf, -np.inf], np.nan).dropna()), 1.0 / 86400.0)
    assert not bool(out["SSC_derived"].any())


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


def test_real_source_id20_seconds_mismatch_source_ssc_retained():
    if not SOURCE_DIR.exists():
        return
    old_source = mod.SOURCE_DIR
    mod.SOURCE_DIR = str(SOURCE_DIR)
    try:
        out = mod.read_station_data(20, "PL")
    finally:
        mod.SOURCE_DIR = old_source
    ssc = out["SSC"].replace(mod.FILL_VALUE, np.nan)
    q = out["Q"].replace(mod.FILL_VALUE, np.nan)
    ssl = out["SSL"].replace(mod.FILL_VALUE, np.nan)
    ratio = ssl / (0.0864 * q * ssc)
    assert_close(np.nanmax(ssc), 56178000.0, rel=1e-6)
    assert_close(np.nanmedian(ratio.replace([np.inf, -np.inf], np.nan).dropna()), 1.0 / 86400.0, rel=1e-5)
    assert not bool(out["SSC_derived"].any())


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()
