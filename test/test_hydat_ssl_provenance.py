#!/usr/bin/env python3
"""Regression tests for HYDAT mixed source/derived SSL provenance."""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FLAG_ESTIMATED, FLAG_GOOD, FLAG_SUSPECT  # noqa: E402

metadata_stub = types.ModuleType("code.metadata")
metadata_stub.check_variable_metadata_tiered = lambda *args, **kwargs: ([], [])
sys.modules.setdefault("code.metadata", metadata_stub)

plot_stub = types.ModuleType("code.plot")
plot_stub.plot_ssc_q_diagnostic = lambda *args, **kwargs: None
sys.modules.setdefault("code.plot", plot_stub)


def _load_module(relative_path, module_name):
    path = SCRIPT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if nc is not None:
    stage3 = _load_module("Hydat/3_update_sediment_nc_fixed.py", "hydat_stage3")
    stage4 = _load_module("Hydat/4_process_hydat_cf18.py", "hydat_stage4")
else:
    stage3 = None
    stage4 = None


def _require_netcdf4():
    if nc is None:
        print("SKIP netCDF4 is not installed in this Python environment")
        return False
    return True


def _write_stage3_inputs(tmpdir):
    sed_path = Path(tmpdir) / "HYDAT_TEST_SEDIMENT.nc"
    dis_path = Path(tmpdir) / "HYDAT_TEST.nc"

    with nc.Dataset(sed_path, "w") as ds:
        ds.createDimension("point", 1)
        ds.createDimension("load_time", 2)
        ds.createDimension("ssc_time", 2)
        ds.station_id = "TEST"
        ds.station_name = "TEST RIVER"
        ds.province_territory = "Test Province"

        lat = ds.createVariable("lat", "f4", ("point",))
        lon = ds.createVariable("lon", "f4", ("point",))
        lat[:] = [45.0]
        lon[:] = [-75.0]

        t_load = ds.createVariable("time_sed_load", "f8", ("load_time",))
        t_ssc = ds.createVariable("time_sed_suscon", "f8", ("ssc_time",))
        ssl = ds.createVariable("sediment_load", "f4", ("load_time",))
        ssc = ds.createVariable("suspended_sediment_concentration", "f4", ("ssc_time",))

        t_load[:] = [0.0, 2.0]
        t_ssc[:] = [0.0, 1.0]
        ssl[:] = [123.0, 50.0]
        ssc[:] = [10.0, 20.0]

    with nc.Dataset(dis_path, "w") as ds:
        ds.createDimension("time", 3)
        time = ds.createVariable("time_flow", "f8", ("time",))
        q = ds.createVariable("discharge", "f4", ("time",))
        area = ds.createVariable("drainage_area", "f4")
        time.units = "days since 1970-01-01 00:00:00"
        time[:] = [0.0, 1.0, 2.0]
        q[:] = [100.0, 100.0, 100.0]
        area[:] = 1000.0

    return sed_path, dis_path


def test_stage3_marks_only_imputed_records_and_preserves_source_ssl():
    if not _require_netcdf4():
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        sed_path, dis_path = _write_stage3_inputs(tmpdir)
        out_path = Path(tmpdir) / "HYDAT_TEST_STAGE3.nc"

        assert stage3.update_sediment_file(sed_path, dis_path, out_path)

        with nc.Dataset(out_path) as ds:
            ssl = ds.variables["sediment_load"][:]
            ssc = ds.variables["ssc"][:]
            ssl_derived = ds.variables["SSL_derived"][:]
            ssc_derived = ds.variables["SSC_derived"][:]

        np.testing.assert_allclose(ssl[0], 123.0)
        np.testing.assert_allclose(ssl[1], 100.0 * 20.0 * 0.0864)
        np.testing.assert_allclose(ssc[2], 50.0 / (100.0 * 0.0864))
        np.testing.assert_array_equal(ssl_derived, [0, 1, 0])
        np.testing.assert_array_equal(ssc_derived, [0, 0, 1])


def test_hydat_qc_flags_source_and_derived_records_differently():
    if not _require_netcdf4():
        return
    time = np.arange(3, dtype=float)
    Q = np.array([100.0, 100.0, 100.0])
    SSC = np.array([10.0, 20.0, 50.0 / (100.0 * 0.0864)])
    SSL = np.array([123.0, 100.0 * 20.0 * 0.0864, 50.0])

    qc = stage4.apply_tool_qc(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        station_id="TEST",
        station_name="TEST RIVER",
        SSC_derived_mask=np.array([False, False, True]),
        SSL_derived_mask=np.array([False, True, False]),
        plot_dir=None,
    )

    assert qc is not None
    np.testing.assert_allclose(qc["SSL"][0], 123.0)
    assert qc["SSL_flag"][0] == FLAG_GOOD
    assert qc["SSL_flag"][1] == FLAG_ESTIMATED
    assert qc["SSC_flag"][2] == FLAG_ESTIMATED
    assert qc["SSL_flag"][2] == FLAG_GOOD


def test_suspect_input_propagates_only_to_derived_ssl():
    if not _require_netcdf4():
        return
    Q = np.array([1, 2, 3, 4, 5, 100000], dtype=float)
    SSC = np.array([10, 11, 10, 12, 11, 10], dtype=float)
    SSL = Q * SSC * 0.0864
    time = np.arange(len(Q), dtype=float)

    qc = stage4.apply_tool_qc(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        station_id="TEST",
        station_name="TEST RIVER",
        SSC_derived_mask=np.zeros(len(Q), dtype=bool),
        SSL_derived_mask=np.ones(len(Q), dtype=bool),
        plot_dir=None,
    )

    assert qc is not None
    assert qc["Q_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][0] == FLAG_ESTIMATED


def test_source_ssl_does_not_inherit_derived_ssl_qc3_propagation():
    if not _require_netcdf4():
        return
    Q = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    SSC = np.array([10, 12, 14, 16, 18, 10], dtype=float)
    SSL = np.full(len(Q), 100.0, dtype=float)
    time = np.arange(len(Q), dtype=float)

    qc = stage4.apply_tool_qc(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        station_id="TEST",
        station_name="TEST RIVER",
        SSC_derived_mask=np.zeros(len(Q), dtype=bool),
        SSL_derived_mask=np.zeros(len(Q), dtype=bool),
        plot_dir=None,
    )

    assert qc is not None
    assert qc["SSC_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][-1] == FLAG_GOOD


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASS {}".format(name))


if __name__ == "__main__":
    main()
