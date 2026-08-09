#!/usr/bin/env python3
"""
Regression tests for Mekong Delta single-source station processing.

Verifies:
  1. Ratings-only station (SSC) — no fluxes file — is processed successfully
  2. Fluxes-only station (SSL) — no ratings file — is processed successfully
  3. SSC_derived_mask and SSL_derived_mask provenance tracking
  4. Q-only records do NOT cause station retention
  5. NetCDF schema unchanged
"""

import sys
import os
import warnings
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock matplotlib BEFORE any code module tries to import it
import matplotlib as _mpl
_mpl.use('Agg')  # non-interactive backend

# ---------------------------------------------------------------------------
# Mock data helpers
# ---------------------------------------------------------------------------

def make_ratings_csv(filepath, dates, Q_vals, SSC_vals):
    """Create a minimal ratings CSV with Q and SSC."""
    df = pd.DataFrame({
        'Year': [d.year for d in dates],
        'Month': [d.month for d in dates],
        'Day': [d.day for d in dates],
        'Discharge (m3/s)': Q_vals,
        'Section Averaged SSC (mg/l)': SSC_vals,
    })
    df.to_csv(filepath, index=False)
    return filepath


def make_fluxes_csv(filepath, dates, ssl_mt_per_day):
    """Create a minimal fluxes CSV in wide format."""
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    df = pd.DataFrame({
        'Date': [f'{d.day:02d}-{months[d.month-1]}' for d in dates],
        str(dates[0].year): ssl_mt_per_day,
    })
    df.to_csv(filepath, index=False)
    return filepath


# ---------------------------------------------------------------------------
# Test 1: Ratings-only station (SSC source, no fluxes)
# ---------------------------------------------------------------------------

def test_ratings_only_ssc_station():
    """Ratings-only: SSC preserved, SSL derived from Q+SSC, no fluxes file."""
    
    tmpdir = Path(tempfile.mkdtemp(prefix='test_mekong_ratings_only_'))
    
    try:
        # --- Setup mock data ---
        data_dir = tmpdir / 'data'
        data_dir.mkdir()
        out_dir = tmpdir / 'output'
        out_dir.mkdir()
        
        station_id = 'TestSSC'
        rng = np.random.default_rng(42)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        Q_vals = np.abs(np.linspace(800, 1200, 30) + rng.normal(0, 20, 30))
        SSC_vals = np.abs(np.full(30, 123.0) * (1 + rng.normal(0, 0.02, 30)))
        
        # Create ONLY ratings file (no fluxes file)
        make_ratings_csv(data_dir / f'{station_id}ratings.csv', dates, Q_vals, SSC_vals)
        
        # --- Monkey-patch the module ---
        import Mekong_Delta.process_mekong_delta as pmd
        
        orig_source = pmd.SOURCE_DATA_DIR
        orig_target_nc = pmd.TARGET_NC_DIR
        orig_target_csv = pmd.TARGET_CSV_PATH
        orig_stations = pmd.STATIONS
        
        pmd.SOURCE_DATA_DIR = str(data_dir)
        pmd.TARGET_NC_DIR = str(out_dir)
        pmd.TARGET_CSV_PATH = str(out_dir)
        pmd.STATIONS = {
            station_id: {
                'name': 'Test SSC Station', 'Source_ID': station_id,
                'lat': 10.0, 'lon': 105.0, 'river': 'Test River',
                'altitude': np.nan, 'upstream_area': np.nan
            }
        }
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pmd.main()
        finally:
            pmd.SOURCE_DATA_DIR = orig_source
            pmd.TARGET_NC_DIR = orig_target_nc
            pmd.TARGET_CSV_PATH = orig_target_csv
            pmd.STATIONS = orig_stations
        
        # --- Verify NetCDF output ---
        nc_file = out_dir / f'Mekong_Delta_{station_id}.nc'
        assert nc_file.exists(), f"NetCDF not found: {nc_file}"
        
        ds = xr.open_dataset(nc_file)
        
        # SSC: should be preserved from source
        ssc = ds['SSC'].values
        ssc_valid = np.isfinite(ssc) & (ssc > 0)
        assert np.any(ssc_valid), "No valid SSC in ratings-only output"
        print(f"  PASS: SSC has {np.sum(ssc_valid)} valid values")
        
        # SSC_flag: should have good (0) records
        ssc_flag = ds['SSC_flag'].values
        assert np.any(ssc_flag == 0), "No SSC_flag=0 (good) in output"
        print(f"  PASS: SSC_flag has {np.sum(ssc_flag == 0)} good records")
        
        # SSC_derived_mask: source SSC should be 0
        if 'SSC_derived_mask' in ds.variables:
            ssc_dm = ds['SSC_derived_mask'].values
            ssc_dm_valid = ssc_dm[ssc_valid]
            assert np.all(ssc_dm_valid == 0), \
                f"Source SSC should have derived_mask=0, got {set(ssc_dm_valid)}"
            print(f"  PASS: SSC_derived_mask=0 for all valid SSC (source)")
        
        # SSL: should be derived from Q+SSC
        ssl = ds['SSL'].values
        ssl_valid = np.isfinite(ssl) & (ssl > 0)
        assert np.any(ssl_valid), "No valid SSL (should be derived) in output"
        print(f"  PASS: SSL has {np.sum(ssl_valid)} valid (derived) values")
        
        # SSL_derived_mask: derived SSL should be 1
        if 'SSL_derived_mask' in ds.variables:
            ssl_dm = ds['SSL_derived_mask'].values
            ssl_dm_valid = ssl_dm[ssl_valid]
            assert np.all(ssl_dm_valid == 1), \
                f"Derived SSL should have derived_mask=1, got {set(ssl_dm_valid)}"
            print(f"  PASS: SSL_derived_mask=1 for all valid SSL (derived)")
        
        # Q: should be present and valid
        q = ds['Q'].values
        q_valid = np.isfinite(q) & (q > 0)
        assert np.any(q_valid), "No valid Q in ratings-only output"
        print(f"  PASS: Q has {np.sum(q_valid)} valid values")
        
        # Schema check
        for var in ['Q', 'SSC', 'SSL', 'Q_flag', 'SSC_flag', 'SSL_flag']:
            assert var in ds.variables, f"Variable '{var}' missing"
        print(f"  PASS: NetCDF schema intact")
        
        ds.close()
        
        # --- Verify audit CSV ---
        summary_csv = out_dir / 'Mekong_Delta_station_summary.csv'
        assert summary_csv.exists(), f"Summary CSV not found: {summary_csv}"
        summary_df = pd.read_csv(summary_csv)
        assert len(summary_df) == 1, f"Expected 1 station, got {len(summary_df)}"
        assert summary_df['SSC_percent_complete'].iloc[0] > 0, "SSC should be > 0%"
        print(f"  PASS: Summary CSV has SSC_percent_complete={summary_df['SSC_percent_complete'].iloc[0]}")
        
        print(f"\n  ✓ Ratings-only SSC station: ALL CHECKS PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Fluxes-only station (SSL source, no ratings)
# ---------------------------------------------------------------------------

def test_fluxes_only_ssl_station():
    """Fluxes-only: SSL preserved from source, SSC stays NaN (no Q to derive)."""
    
    tmpdir = Path(tempfile.mkdtemp(prefix='test_mekong_fluxes_only_'))
    
    try:
        # --- Setup mock data ---
        data_dir = tmpdir / 'data'
        data_dir.mkdir()
        out_dir = tmpdir / 'output'
        out_dir.mkdir()
        
        station_id = 'TestSSL'
        rng = np.random.default_rng(44)
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        ssl_mt = np.abs(np.linspace(0.004, 0.006, 30) + rng.normal(0, 0.0001, 30))
        
        # Create ONLY fluxes file (no ratings file)
        make_fluxes_csv(data_dir / f'{station_id}fluxes.csv', dates, ssl_mt)
        
        # --- Monkey-patch the module ---
        import Mekong_Delta.process_mekong_delta as pmd
        
        orig_source = pmd.SOURCE_DATA_DIR
        orig_target_nc = pmd.TARGET_NC_DIR
        orig_target_csv = pmd.TARGET_CSV_PATH
        orig_stations = pmd.STATIONS
        
        pmd.SOURCE_DATA_DIR = str(data_dir)
        pmd.TARGET_NC_DIR = str(out_dir)
        pmd.TARGET_CSV_PATH = str(out_dir)
        pmd.STATIONS = {
            station_id: {
                'name': 'Test SSL Station', 'Source_ID': station_id,
                'lat': 10.0, 'lon': 105.0, 'river': 'Test River',
                'altitude': np.nan, 'upstream_area': np.nan
            }
        }
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pmd.main()
        finally:
            pmd.SOURCE_DATA_DIR = orig_source
            pmd.TARGET_NC_DIR = orig_target_nc
            pmd.TARGET_CSV_PATH = orig_target_csv
            pmd.STATIONS = orig_stations
        
        # --- Verify NetCDF output ---
        nc_file = out_dir / f'Mekong_Delta_{station_id}.nc'
        assert nc_file.exists(), f"NetCDF not found: {nc_file}"
        
        ds = xr.open_dataset(nc_file)
        
        # SSL: should be preserved from source
        ssl = ds['SSL'].values
        ssl_valid = np.isfinite(ssl) & (ssl > 0)
        assert np.any(ssl_valid), "No valid SSL in fluxes-only output"
        print(f"  PASS: SSL has {np.sum(ssl_valid)} valid values")
        
        # SSL_derived_mask: source SSL should be 0
        if 'SSL_derived_mask' in ds.variables:
            ssl_dm = ds['SSL_derived_mask'].values
            ssl_dm_valid = ssl_dm[ssl_valid]
            assert np.all(ssl_dm_valid == 0), \
                f"Source SSL should have derived_mask=0, got {set(ssl_dm_valid)}"
            print(f"  PASS: SSL_derived_mask=0 for all valid SSL (source)")
        
        # SSC: should be NaN (can't derive without Q)
        ssc = ds['SSC'].values
        ssc_valid = np.isfinite(ssc) & (ssc > 0)
        assert not np.any(ssc_valid), \
            f"SSC should be all-NaN when no Q, got {np.sum(ssc_valid)} valid"
        print(f"  PASS: SSC is all-NaN (cannot derive without Q)")
        
        # SSC_derived_mask: should not be 1 (no derivation happened)
        if 'SSC_derived_mask' in ds.variables:
            ssc_dm = ds['SSC_derived_mask'].values
            assert not np.any(ssc_dm == 1), \
                "No SSC should be marked as derived"
            print(f"  PASS: SSC_derived_mask never=1 (no derivation possible)")
        
        # Q: should be all-NaN/fill
        q = ds['Q'].values
        q_valid = np.isfinite(q) & (q != -9999.0) & (q > 0)
        assert not np.any(q_valid), "Q should be all-NaN when no ratings file"
        print(f"  PASS: Q is all-NaN (no ratings file)")
        
        # Schema check
        for var in ['Q', 'SSC', 'SSL', 'Q_flag', 'SSC_flag', 'SSL_flag']:
            assert var in ds.variables, f"Variable '{var}' missing"
        print(f"  PASS: NetCDF schema intact")
        
        # SSL_flag check: should have good records
        ssl_flag = ds['SSL_flag'].values
        assert np.any(ssl_flag == 0), "No SSL_flag=0 (good) in output"
        print(f"  PASS: SSL_flag has {np.sum(ssl_flag == 0)} good records")
        
        ds.close()
        
        # --- Verify audit CSV ---
        summary_csv = out_dir / 'Mekong_Delta_station_summary.csv'
        assert summary_csv.exists(), f"Summary CSV not found: {summary_csv}"
        summary_df = pd.read_csv(summary_csv)
        assert len(summary_df) == 1, f"Expected 1 station, got {len(summary_df)}"
        assert summary_df['SSL_percent_complete'].iloc[0] > 0, "SSL should be > 0%"
        print(f"  PASS: Summary CSV has SSL_percent_complete={summary_df['SSL_percent_complete'].iloc[0]}")
        
        print(f"\n  ✓ Fluxes-only SSL station: ALL CHECKS PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Q-only records do not retain station
# ---------------------------------------------------------------------------

def test_q_only_does_not_retain():
    """A station with only Q (zero SSC, zero SSL) should be skipped."""
    
    tmpdir = Path(tempfile.mkdtemp(prefix='test_mekong_q_only_'))
    
    try:
        data_dir = tmpdir / 'data'
        data_dir.mkdir()
        out_dir = tmpdir / 'output'
        out_dir.mkdir()
        
        station_id = 'TestQOnly'
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        Q_vals = np.linspace(100, 200, 10)
        SSC_vals = np.zeros(10)  # All zero SSC
        SSL_mt = np.zeros(10)    # All zero SSL
        
        make_ratings_csv(data_dir / f'{station_id}ratings.csv', dates, Q_vals, SSC_vals)
        make_fluxes_csv(data_dir / f'{station_id}fluxes.csv', dates, SSL_mt)
        
        import Mekong_Delta.process_mekong_delta as pmd
        
        orig_source = pmd.SOURCE_DATA_DIR
        orig_target_nc = pmd.TARGET_NC_DIR
        orig_target_csv = pmd.TARGET_CSV_PATH
        orig_stations = pmd.STATIONS
        
        pmd.SOURCE_DATA_DIR = str(data_dir)
        pmd.TARGET_NC_DIR = str(out_dir)
        pmd.TARGET_CSV_PATH = str(out_dir)
        pmd.STATIONS = {
            station_id: {
                'name': 'Test Q-Only', 'Source_ID': station_id,
                'lat': 10.0, 'lon': 105.0, 'river': 'Test',
                'altitude': np.nan, 'upstream_area': np.nan
            }
        }
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pmd.main()
        finally:
            pmd.SOURCE_DATA_DIR = orig_source
            pmd.TARGET_NC_DIR = orig_target_nc
            pmd.TARGET_CSV_PATH = orig_target_csv
            pmd.STATIONS = orig_stations
        
        # Should NOT create a NetCDF (station skipped — no SSC, no SSL)
        nc_file = out_dir / f'Mekong_Delta_{station_id}.nc'
        assert not nc_file.exists(), \
            f"Q-only station should be skipped, but NetCDF exists: {nc_file}"
        print(f"  PASS: Station with only Q (zero SSC, zero SSL) correctly skipped")
        
        print(f"\n  ✓ Q-only station: ALL CHECKS PASSED")
        return True
        
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Provenance unit-test: derive_ssl_from_q_ssc correctness
# ---------------------------------------------------------------------------

def test_derive_ssl_correctness():
    """Verify calculate_ssl_from_q_ssc formula: SSL = Q * SSC * 0.0864."""
    import Mekong_Delta.process_mekong_delta as pmd
    
    df = pd.DataFrame({
        'Q': [1000.0, 500.0, 0.0, np.nan],
        'SSC': [100.0, 200.0, 50.0, 50.0],
    })
    ssl = pmd.calculate_ssl_from_q_ssc(df)
    
    # Q=1000, SSC=100 → SSL = 1000 * 100 * 0.0864 = 8640
    expected_0 = 1000.0 * 100.0 * 0.0864
    assert np.isclose(ssl.iloc[0], expected_0, rtol=1e-10), \
        f"Expected {expected_0}, got {ssl.iloc[0]}"
    
    # Q=500, SSC=200 → SSL = 500 * 200 * 0.0864 = 8640
    expected_1 = 500.0 * 200.0 * 0.0864
    assert np.isclose(ssl.iloc[1], expected_1, rtol=1e-10), \
        f"Expected {expected_1}, got {ssl.iloc[1]}"
    
    # Q=0 → SSL=0 (valid, unlike division)
    assert ssl.iloc[2] == 0.0, f"Q=0 should produce 0.0, got {ssl.iloc[2]}"
    
    # Q=NaN → should be NaN
    assert np.isnan(ssl.iloc[3]), f"Q=NaN should produce NaN, got {ssl.iloc[3]}"
    
    print(f"  PASS: SSL = {expected_0:.1f} (Q=1000,SSC=100), {expected_1:.1f} (Q=500,SSC=200)")
    print(f"  PASS: Q=0 produces 0.0, Q=NaN produces NaN")
    print(f"\n  ✓ derive_ssl_from_q_ssc: ALL CHECKS PASSED")
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Mekong Delta Single-Source Regression Tests")
    print("=" * 60)
    
    results = {}
    for name, func in sorted(globals().items()):
        if name.startswith('test_') and callable(func):
            print(f"\n{'─'*60}")
            print(f"Running: {name}")
            print(f"{'─'*60}")
            results[name] = func()
    
    print(f"\n{'='*60}")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
    print(f"{'='*60}")
    
    if failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
