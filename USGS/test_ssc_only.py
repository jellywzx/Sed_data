#!/usr/bin/env python3
"""
Minimal regression test: prove that a station with SSC file but NO discharge file
can be processed successfully by process_usgs.process_single_station.

Creates a temporary mock station directory, runs the processor, and verifies:
  1. Station is NOT skipped (status == 'success')
  2. Output NetCDF exists
  3. SSC values are present
  4. Q is all-NaN (no discharge file)
  5. SSL is all-NaN (cannot derive without Q)
  6. SSC_flag has good data
"""

import pandas as pd
import numpy as np
import xarray as xr
import tempfile
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from USGS.process_usgs import process_single_station


def make_mock_sediment_file(station_dir, station_id):
    """Create a minimal sediment CSV with 80154 (SSC) data."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    ssc_vals = [15.2, 18.1, 22.3, 19.8, 25.0, 14.5, 30.1, 16.7, 20.9, 23.4]

    df = pd.DataFrame({
        'datetime': dates.strftime('%Y-%m-%d'),
        'agency_cd': ['USGS'] * 10,
        'site_no': [station_id] * 10,
        '80154_00003': ssc_vals,
        '80154_00003_cd': ['A'] * 10,
    })

    csv_path = station_dir / f"{station_id}_sediment.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_mock_info_file(station_dir, station_id):
    """Create a minimal info CSV matching common_sites_info schema."""
    df = pd.DataFrame({
        'site_no': [station_id],
        'station_nm': ['TEST_SSC_ONLY_STATION'],
        'dec_lat_va': [35.0],
        'dec_long_va': [-90.0],
        'alt_va': [100.0],
        'drain_area_va': [500.0],
    })
    csv_path = station_dir / f"{station_id}_info.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_ssc_only_station():
    """Main test: SSC-only station (no discharge file) must succeed."""

    tmpdir = Path(tempfile.mkdtemp(prefix='test_usgs_ssc_only_'))
    output_dir = tmpdir / 'output'
    output_dir.mkdir()

    try:
        station_id = '99999999'

        station_dir = tmpdir / f'station_{station_id}'
        station_dir.mkdir()

        make_mock_sediment_file(station_dir, station_id)
        info_csv = make_mock_info_file(station_dir, station_id)

        sites_info_df = pd.read_csv(info_csv, dtype={'site_no': str})
        sites_info_df['site_no'] = sites_info_df['site_no'].astype(str)

        # --- Run the processor ---
        args = (station_dir, sites_info_df, output_dir)
        result = process_single_station(args)

        # --- Assertions ---
        assert result['status'] == 'success', \
            f"Expected 'success', got '{result['status']}': {result.get('reason', '')}"
        print(f"PASS: status = {result['status']}")

        nc_file = output_dir / f"USGS_{station_id}.nc"
        assert nc_file.exists(), f"NetCDF not found: {nc_file}"
        print(f"PASS: NetCDF exists at {nc_file}")

        ds = xr.open_dataset(nc_file)

        ssc = ds['SSC'].values
        assert np.any(np.isfinite(ssc) & (ssc > 0)), "No valid SSC values in output"
        print(f"PASS: SSC has {np.sum(np.isfinite(ssc) & (ssc > 0))} valid values")

        q = ds['Q'].values
        q_valid = np.isfinite(q) & (q != -9999.0)
        assert not np.any(q_valid), "Q should be all-NaN/fill when no discharge file"
        print(f"PASS: Q is all-NaN/fill (no discharge file)")

        ssl = ds['SSL'].values
        ssl_valid = np.isfinite(ssl) & (ssl > 0)
        assert not np.any(ssl_valid), "SSL should be all-NaN when Q is missing"
        print(f"PASS: SSL is all-NaN (cannot derive without Q)")

        ssc_flag = ds['SSC_flag'].values
        assert np.any(ssc_flag == 0), "No SSC_flag == 0 (good) in output"
        print(f"PASS: SSC_flag has {np.sum(ssc_flag == 0)} good records")

        assert result['good_count'] > 0, "good_count should be > 0 for SSC-only station"
        print(f"PASS: good_count = {result['good_count']} (SSC eligible)")

        au = result.get('_audit', {})
        assert au.get('has_discharge') == False, "has_discharge should be False"
        assert au.get('ssc_source_count') == 10, f"ssc_source_count should be 10, got {au.get('ssc_source_count')}"
        assert au.get('paired_count') == 0, f"paired_count should be 0, got {au.get('paired_count')}"
        print(f"PASS: audit fields correct: has_discharge={au['has_discharge']}, "
              f"ssc_source={au['ssc_source_count']}, paired={au['paired_count']}, "
              f"retained={au['retained_count']}")

        for var in ['Q', 'SSC', 'SSL']:
            assert var in ds.variables, f"Variable '{var}' missing from output"
        print(f"PASS: NetCDF schema unchanged (Q, SSC, SSL all present)")

        assert ds['SSC'].attrs.get('units') == 'mg L-1', "SSC units changed!"
        assert ds['Q'].attrs.get('units') == 'm3 s-1', "Q units changed!"
        assert ds['SSL'].attrs.get('units') == 'ton day-1', "SSL units changed!"
        print(f"PASS: units unchanged")

        ds.close()

        print(f"\n{'='*60}")
        print(f"ALL TESTS PASSED: SSC-only station (no discharge file) works correctly.")
        print(f"{'='*60}")
        return True

    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    success = test_ssc_only_station()
    sys.exit(0 if success else 1)
