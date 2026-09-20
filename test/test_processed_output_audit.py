from pathlib import Path
import csv
import sys

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.audit import collect_dataset_output_audit, write_dataset_output_audit


def _write_station(path, resolution="daily"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", 4)
        ds.temporal_resolution = resolution

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var[:] = np.arange(4, dtype=float)

        values = {
            "Q": np.array([1.0, 2.0, -9999.0, -9999.0]),
            "SSC": np.array([10.0, -9999.0, 20.0, -9999.0]),
            "SSL": np.array([0.864, -9999.0, -9999.0, -9999.0]),
        }
        flags = {
            "Q": np.array([0, 0, 9, 9], dtype=np.int8),
            "SSC": np.array([0, 9, 2, 9], dtype=np.int8),
            "SSL": np.array([1, 9, 9, 9], dtype=np.int8),
        }

        for name in ("Q", "SSC", "SSL"):
            value_var = ds.createVariable(
                name,
                "f4",
                ("time",),
                fill_value=-9999.0,
            )
            value_var[:] = values[name]

            flag_var = ds.createVariable(
                f"{name}_flag",
                "i1",
                ("time",),
                fill_value=9,
            )
            flag_var[:] = flags[name]


def test_collect_dataset_output_audit_classifies_records(tmp_path):
    root = tmp_path / "Output_r"
    _write_station(root / "daily" / "USGS" / "qc" / "USGS_001.nc")

    by_dataset, by_resolution = collect_dataset_output_audit(root)

    assert len(by_dataset) == 1
    row = by_dataset[0]
    assert row["dataset"] == "USGS NWIS"
    assert row["resolution"] == "all"
    assert row["n_files"] == 1
    assert row["nc_time_steps"] == 4
    assert row["sediment_eligible_records"] == 2
    assert row["screened_for_integration_records"] == 2
    assert row["q_only_records"] == 1
    assert row["fully_missing_records"] == 1
    assert row["analysis_ready_sediment_records"] == 1
    assert row["sediment_suspect_bad_only_records"] == 1
    assert row["Q_nonmissing"] == 2
    assert row["SSC_nonmissing"] == 2
    assert row["SSL_nonmissing"] == 1
    assert row["SSC_good"] == 1
    assert row["SSC_suspect"] == 1
    assert row["SSC_missing"] == 2
    assert row["flag_value_mismatch_count"] == 0

    assert len(by_resolution) == 1
    assert by_resolution[0]["resolution"] == "daily"


def test_audit_ignores_intermediate_netcdf_and_writes_csv(tmp_path):
    root = tmp_path / "Output_r"
    _write_station(
        root / "monthly" / "Eurasian_River" / "qc" / "Eurasian_01.nc",
        resolution="monthly",
    )
    _write_station(
        root / "monthly" / "Eurasian_River" / "nc" / "intermediate.nc",
        resolution="monthly",
    )

    paths = write_dataset_output_audit(root)

    assert paths["dataset"].exists()
    assert paths["by_resolution"].exists()

    with paths["dataset"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["dataset"] == "Eurasian River"
    assert int(rows[0]["n_files"]) == 1
    assert int(rows[0]["nc_time_steps"]) == 4
