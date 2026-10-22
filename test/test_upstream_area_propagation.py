#!/usr/bin/env python3
"""Regression tests for source upstream-area propagation."""

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FILL_VALUE_FLOAT
from Eurasian_River.process_eurasian_river import write_upstream_area as write_eurasian_area
from GFQA_v2.gfqa_to_netcdf_daily_dualqc import write_upstream_area as write_gfqa_area
from HYBAM.hybam_comprehensive_processor import HYBAMProcessor


def _assert_scalar_area(path, expected):
    with nc.Dataset(path) as ds:
        var = ds.variables['upstream_area']
        assert var.dimensions == ()
        assert var.dtype == np.dtype('float32')
        assert var.units == 'km2'
        assert var.getncattr('_FillValue') == np.float32(FILL_VALUE_FLOAT)
        assert np.isclose(float(var[:]), expected)


def test_hybam_loads_exact_and_legacy_alias_areas(tmp_path):
    source_dir = tmp_path / 'HYBAM' / 'source'
    area_dir = tmp_path / 'HYBAM' / 'HYBAM-old' / 'info' / '15min'
    source_dir.mkdir(parents=True)
    area_dir.mkdir(parents=True)
    (area_dir / 'stNo2xy.txt').write_text(
        'ID,lat,lon,iy,ix,uparea\n'
        '10073500,-10.60762,-73.87119,400,423,200496.69\n'
        '30030000,4.9862,-54.4368,338,502,64678.14\n'
        '30055000,3.8017,-51.8847,344,512,25146.29\n'
        'BAD,-1,-1,0,0,-5\n',
        encoding='utf-8',
    )

    processor = HYBAMProcessor(source_dir, tmp_path / 'intermediate', tmp_path / 'output')

    assert np.isclose(processor.get_upstream_area('10073500'), 200496.69)
    assert np.isclose(processor.get_upstream_area('2604100121'), 64678.14)
    assert np.isclose(processor.get_upstream_area('2604500124'), 25146.29)
    assert processor.get_upstream_area('UNKNOWN') == FILL_VALUE_FLOAT
    assert processor.get_upstream_area('BAD') == FILL_VALUE_FLOAT

    output_path = tmp_path / 'hybam_10073500.nc'
    processor.write_cf18_netcdf(
        station_id='10073500',
        station_name='Lagarto',
        river_name='Ucayali',
        latitude=-10.60762,
        longitude=-73.87119,
        altitude=195.0,
        upstream_area=processor.get_upstream_area('10073500'),
        data_dict={
            'time': np.array([0.0]),
            'discharge': None,
            'ssc': None,
            'time_coverage_start': '1970-01-01',
            'time_coverage_end': '1970-01-01',
        },
        output_file=output_path,
    )
    _assert_scalar_area(output_path, 200496.69)
    with nc.Dataset(output_path) as ds:
        assert 'upstream_area' in ds.variables_provided
        assert ds.variables['upstream_area'].comment == 'Source: CaMa-Flood 15 arc-minute model.'


def test_eurasian_writes_valid_and_missing_scalar_area(tmp_path):
    valid_path = tmp_path / 'eurasian_valid.nc'
    with nc.Dataset(valid_path, 'w') as ds:
        write_eurasian_area(ds, 17700.0)
    _assert_scalar_area(valid_path, 17700.0)

    missing_path = tmp_path / 'eurasian_missing.nc'
    with nc.Dataset(missing_path, 'w') as ds:
        write_eurasian_area(ds, np.nan)
    with nc.Dataset(missing_path) as ds:
        assert np.ma.is_masked(ds.variables['upstream_area'][:])


def test_gfqa_writes_scalar_area_and_legacy_attribute(tmp_path):
    valid_path = tmp_path / 'gfqa_valid.nc'
    with nc.Dataset(valid_path, 'w') as ds:
        write_gfqa_area(ds, '91800')
    _assert_scalar_area(valid_path, 91800.0)
    with nc.Dataset(valid_path) as ds:
        assert ds.getncattr('upstream_area') == 91800.0

    missing_path = tmp_path / 'gfqa_missing.nc'
    with nc.Dataset(missing_path, 'w') as ds:
        write_gfqa_area(ds, 'not available')
    with nc.Dataset(missing_path) as ds:
        assert np.ma.is_masked(ds.variables['upstream_area'][:])
        assert ds.getncattr('upstream_area') == FILL_VALUE_FLOAT
