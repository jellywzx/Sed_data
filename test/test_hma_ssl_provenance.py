#!/usr/bin/env python3
"""Regression tests for HMA SSL provenance branches."""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

plot_stub = types.ModuleType("code.plot")
plot_stub.plot_ssc_q_diagnostic = lambda *args, **kwargs: None
sys.modules["code.plot"] = plot_stub

MODULE_PATH = SCRIPT_ROOT / "HMA" / "convert_to_netcdf_cf18_qc.py"
spec = importlib.util.spec_from_file_location("hma_qc", MODULE_PATH)
hma_qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hma_qc)


def _station_row(qs, sediment_yield):
    return pd.Series(
        {
            "Stations": "Test station (S1)",
            "Longitude": "90.0",
            "Latitude": "30.0",
            "Basin area (km2)": "100.0",
            "Glacier cover (%)": "",
            "Permafrost cover (%)": "",
            "Q (km3/yr)": "1.0",
            "Qs (Mt/yr)": qs,
            "sediment yield\xa0（(t/km2/y)）": sediment_yield,
            "Period for Q": "2000-2010",
            "Period for Qs": "2000-2010",
            "Basin": "Test Basin",
            "Headwaters": "Test River",
            "Notes": "",
        }
    )


def _run_case(row):
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = hma_qc.create_netcdf_for_station(
            row,
            tmpdir,
            "HMA_catchments.csv",
        )
        path = Path(tmpdir) / summary["filename"]
        with Dataset(path) as ds:
            return {
                "summary": summary,
                "SSL": float(ds.variables["SSL"][0]),
                "SSL_flag": int(ds.variables["SSL_flag"][0]),
                "SSL_derived_mask": int(ds.variables["SSL_derived_mask"][0]),
                "SSL_comment": ds.variables["SSL"].comment,
                "SSL_ancillary": ds.variables["SSL"].ancillary_variables,
            }


def test_source_qs_unit_conversion_keeps_good_ssl_flag():
    out = _run_case(_station_row(qs="1.0", sediment_yield=""))

    np.testing.assert_allclose(out["SSL"], hma_qc.convert_Qs_to_SSL(1.0))
    assert out["SSL_flag"] == 0
    assert out["SSL_derived_mask"] == 0
    assert "Source-reported sediment load (Qs)" in out["SSL_comment"]
    assert "SSL_derived_mask" in out["SSL_ancillary"]


def test_sediment_yield_area_derived_ssl_is_estimated():
    out = _run_case(_station_row(qs="", sediment_yield="365.25"))

    np.testing.assert_allclose(out["SSL"], 100.0)
    assert out["SSL_flag"] == 1
    assert out["SSL_derived_mask"] == 1
    assert "Calculated/derived from sediment yield and upstream area" in out["SSL_comment"]


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASS {}".format(name))


if __name__ == "__main__":
    main()
