#!/usr/bin/env python3
"""Regression tests for sediment-oriented station/time eligibility rules.

Validates that:
  - SSC-only records are preserved (not deleted by trim)
  - SSL-only records are preserved
  - Q-only records do NOT determine temporal extent or station entry
  - _classify_records() correctly categorizes records
  - trim_to_valid_data() returns None for all-Q-only stations
"""


import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPT_ROOT / "EUSEDcollab" / "process_eusedcollab_to_cf18_wzx.py"

# -------------------------------------------------------------------
# Stub imports (same pattern as test_process_eusedcollab_units.py)
# -------------------------------------------------------------------

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
    spec = importlib.util.spec_from_file_location("eused_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()

FILL_VALUE = -9999.0   # module-level constant


# -------------------------------------------------------------------
# Helper: build a test dataframe
# -------------------------------------------------------------------

def _make_df(dates, Q, SSC, SSL):
    """Build a dataframe with date, Q, SSC, SSL, Q_derived, SSC_derived, SSL_derived."""
    n = len(dates)
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'Q': np.asarray(Q, dtype=float),
        'SSC': np.asarray(SSC, dtype=float),
        'SSL': np.asarray(SSL, dtype=float),
        'Q_derived': np.zeros(n, dtype=bool),
        'SSC_derived': np.zeros(n, dtype=bool),
        'SSL_derived': np.zeros(n, dtype=bool),
    })


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_trim_preserves_ssc_only():
    """SSC-only records (no Q, no SSL) must survive trim."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01', '2000-03-01'],
        Q=[FILL_VALUE, FILL_VALUE, FILL_VALUE],
        SSC=[15.0, 20.0, 25.0],
        SSL=[FILL_VALUE, FILL_VALUE, FILL_VALUE],
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None, "SSC-only station must not be skipped"
    assert len(result) == 3, f"Expected 3 records, got {len(result)}"
    assert np.allclose(result['SSC'].values, [15.0, 20.0, 25.0])
    print("PASS: SSC-only records preserved")


def test_trim_preserves_ssl_only():
    """SSL-only records (no Q, no SSC) must survive trim."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01'],
        Q=[FILL_VALUE, FILL_VALUE],
        SSC=[FILL_VALUE, FILL_VALUE],
        SSL=[1.5, 2.0],
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None, "SSL-only station must not be skipped"
    assert len(result) == 2, f"Expected 2 records, got {len(result)}"
    assert np.allclose(result['SSL'].values, [1.5, 2.0])
    print("PASS: SSL-only records preserved")


def test_trim_q_only_all_returns_none():
    """All-Q-only, no sediment → trim returns None (station skipped)."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01', '2000-03-01'],
        Q=[1.0, 2.0, 3.0],
        SSC=[FILL_VALUE, FILL_VALUE, FILL_VALUE],
        SSL=[FILL_VALUE, FILL_VALUE, FILL_VALUE],
    )
    result = mod.trim_to_valid_data(df)
    assert result is None, "Q-only station must return None (no sediment)"
    print("PASS: Q-only all returns None")


def test_trim_q_only_edges_do_not_extend_range():
    """Q-only records at edges must not extend temporal coverage."""
    df = _make_df(
        dates=['1999-12-01', '2000-01-01', '2000-02-01', '2000-03-01'],
        Q=[5.0, FILL_VALUE, FILL_VALUE, 3.0],          # Q at edges only
        SSC=[FILL_VALUE, 20.0, 25.0, FILL_VALUE],       # SSC in middle only
        SSL=[FILL_VALUE, FILL_VALUE, FILL_VALUE, FILL_VALUE],
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None, "Station with SSC in middle must survive"
    # Temporal extent must be from SSC (2000-01-01 to 2000-02-01), not Q edges
    assert len(result) == 2, (
        f"Expected 2 records (SSC range only), got {len(result)}"
    )
    assert result.iloc[0]['date'] == pd.Timestamp('2000-01-01'), (
        f"First date should be 2000-01-01, got {result.iloc[0]['date']}"
    )
    assert result.iloc[-1]['date'] == pd.Timestamp('2000-02-01'), (
        f"Last date should be 2000-02-01, got {result.iloc[-1]['date']}"
    )
    print("PASS: Q-only edges do not extend temporal range")


def test_trim_mixed_first_last_from_sediment():
    """Mixed Q/SSC/SSL → first/last valid must come from SSC|SSL, not Q."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01', '2000-03-01', '2000-04-01', '2000-05-01'],
        Q=[1.0, 2.0, FILL_VALUE, 4.0, 5.0],             # Q spans all
        SSC=[FILL_VALUE, 20.0, FILL_VALUE, 40.0, FILL_VALUE],  # SSC in middle
        SSL=[FILL_VALUE, FILL_VALUE, 3.0, FILL_VALUE, FILL_VALUE],  # SSL in middle
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None
    # Sediment range: first=SSC at 2000-02-01, last=SSC at 2000-04-01
    # Note: SSL at 2000-03-01 is between them, so range is 2000-02-01 to 2000-04-01
    assert len(result) == 3, f"Expected 3 records, got {len(result)}"
    assert result.iloc[0]['date'] == pd.Timestamp('2000-02-01')
    assert result.iloc[-1]['date'] == pd.Timestamp('2000-04-01')
    # Q at 2000-01-01 and 2000-05-01 should be trimmed away
    print("PASS: temporal extent from sediment (SSC|SSL), not Q")


def test_classify_records_counts():
    """_classify_records returns correct category counts."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01', '2000-03-01', '2000-04-01', '2000-05-01'],
        Q=[1.0, 2.0, FILL_VALUE, FILL_VALUE, 5.0],
        SSC=[FILL_VALUE, 20.0, 30.0, FILL_VALUE, 50.0],
        SSL=[FILL_VALUE, FILL_VALUE, FILL_VALUE, 4.0, FILL_VALUE],
    )
    counts = mod._classify_records(df)
    assert counts['n_total'] == 5
    assert counts['n_ssc_only'] == 1, f"row 3: SSC-only, got {counts['n_ssc_only']}"   # row idx 2
    assert counts['n_ssl_only'] == 1, f"row 4: SSL-only, got {counts['n_ssl_only']}"   # row idx 3
    assert counts['n_q_only']   == 1, f"row 1: Q-only, got {counts['n_q_only']}"       # row idx 0
    assert counts['n_paired']   == 2, f"rows 2+5: Q+SSC paired, got {counts['n_paired']}"  # row idx 1,4
    assert counts['n_any_sediment'] == 4, f"SSC or SSL at rows 2-5, got {counts['n_any_sediment']}"
    assert counts['n_any_q']    == 3, f"Q at rows 1,2,5, got {counts['n_any_q']}"
    print("PASS: _classify_records counts correct")


def test_classify_records_all_empty():
    """_classify_records on all-FillValue dataframe returns all zeros."""
    df = _make_df(
        dates=['2000-01-01'],
        Q=[FILL_VALUE],
        SSC=[FILL_VALUE],
        SSL=[FILL_VALUE],
    )
    counts = mod._classify_records(df)
    assert counts['n_total'] == 1
    for k in ['n_ssc_only', 'n_ssl_only', 'n_q_only', 'n_paired', 'n_any_sediment', 'n_any_q']:
        assert counts[k] == 0, f"{k} should be 0, got {counts[k]}"
    print("PASS: all-empty classify returns zeros")


def test_ssc_only_passes_derivation_unchanged():
    """read_station_data with SSC-only CSV preserves SSC and does not derive."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "EUSEDcollab"
        q_ssl = source / "Q_SSL"
        q_ssl.mkdir(parents=True)

        pd.DataFrame({
            'Date (DD/MM/YYYY)': ['2000-01-01', '2000-02-01', '2000-03-01'],
            'SSC (g m-3)': [15.0, 20.0, 25.0],
        }).to_csv(q_ssl / "ID_99_Q_SSL_XX.csv", index=False)

        old_source = mod.SOURCE_DIR
        mod.SOURCE_DIR = str(source)
        try:
            out = mod.read_station_data(99, "XX")
        finally:
            mod.SOURCE_DIR = old_source

    assert out is not None, "SSC-only station must not return None"
    assert len(out) == 3, f"Expected 3 records, got {len(out)}"

    ssc = out['SSC'].replace(FILL_VALUE, np.nan)
    assert np.allclose(ssc.values, [15.0, 20.0, 25.0]), f"SSC values changed: {ssc.values}"

    # Q and SSL should be all FillValue (not derived)
    q = out['Q'].replace(FILL_VALUE, np.nan)
    ssl = out['SSL'].replace(FILL_VALUE, np.nan)
    assert q.isna().all(), f"Q should be all-NaN for SSC-only, got {q.values}"
    assert ssl.isna().all(), f"SSL should be all-NaN for SSC-only, got {ssl.values}"

    # Nothing should be marked as derived
    assert not out['Q_derived'].any(), "No Q derivation expected"
    assert not out['SSC_derived'].any(), "No SSC derivation expected"
    assert not out['SSL_derived'].any(), "No SSL derivation expected"

    print("PASS: SSC-only passes derivation unchanged")


def test_trim_preserves_ssc_and_ssl_mixed_no_q():
    """Records with SSC and SSL but no Q must all survive."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01'],
        Q=[FILL_VALUE, FILL_VALUE],
        SSC=[10.0, 20.0],
        SSL=[0.864, 1.728],
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None
    assert len(result) == 2, f"Expected 2 records, got {len(result)}"
    assert np.allclose(result['SSC'].values, [10.0, 20.0])
    assert np.allclose(result['SSL'].values, [0.864, 1.728])
    print("PASS: SSC+SSL mixed (no Q) preserved")


def test_trim_q_inside_sediment_range_kept_as_auxiliary():
    """Q-only records between sediment records should be kept (auxiliary Q)."""
    df = _make_df(
        dates=['2000-01-01', '2000-02-01', '2000-03-01'],
        Q=[FILL_VALUE, 2.0, FILL_VALUE],       # Q only in the middle
        SSC=[15.0, FILL_VALUE, 25.0],           # SSC at edges
        SSL=[FILL_VALUE, FILL_VALUE, FILL_VALUE],
    )
    result = mod.trim_to_valid_data(df)
    assert result is not None
    assert len(result) == 3, (
        f"Interior Q-only record must be kept as auxiliary, got {len(result)}"
    )
    # Q-only record at index 1 must be present
    assert result.iloc[1]['Q'] == 2.0
    assert result.iloc[0]['SSC'] == 15.0
    assert result.iloc[2]['SSC'] == 25.0
    print("PASS: interior Q-only kept as auxiliary within sediment range")


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def main():
    passed = 0
    failed = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
