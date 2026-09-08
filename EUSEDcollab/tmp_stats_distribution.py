#!/usr/bin/env python3
"""Temporary stats: Q/SSC/SSL value distribution across per-station EUSEDcollab qc NetCDF files.

Methodology mirrors Output_r/scripts_basin_test/stats_release/variable_summary.py
(read_numeric_var + numeric_stats conventions), adapted to per-station files:
  - daily:   Output_r/daily/EUSEDcollab/qc/*.nc   (33 stations)
  - monthly: Output_r/monthly/EUSEDcollab/qc/*.nc (211 stations)
"""

# ---- Library path setup: MUST happen before any extension-module imports ----
# (copied from stats_release/variable_summary.py so the wzx conda python can load
#  netCDF4 on nodes with an older system libstdc++)
import os as _os
import ctypes as _ctypes
from pathlib import Path as _Path
_conda_lib = "/share/home/dq134/.conda/envs/wzx/lib"
if _os.path.isdir(_conda_lib):
    _os.environ["LD_LIBRARY_PATH"] = _conda_lib + _os.pathsep + _os.environ.get("LD_LIBRARY_PATH", "")
    try:
        _ctypes.CDLL(str(_Path(_conda_lib) / "libstdc++.so.6"), mode=_ctypes.RTLD_GLOBAL)
    except Exception:
        pass
del _os, _ctypes, _Path, _conda_lib
# ---------------------------------------------------------------------------

import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

VARIABLES = ("Q", "SSC", "SSL")
TIERS = ("all", "flag01")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Script/EUSEDcollab -> sediment_wzx_1111
DIRS = {
    "daily": PROJECT_ROOT / "Output_r" / "daily" / "EUSEDcollab" / "qc",
    "monthly": PROJECT_ROOT / "Output_r" / "monthly" / "EUSEDcollab" / "qc",
}
OUT_CSV = SCRIPT_DIR / "tmp_stats_distribution.csv"


def read_numeric_var(ds, name):
    """Reference recipe (stats_release/release_io.read_numeric_var):
    masked -> NaN, then literal -9999.0 / 1.0e20 -> NaN."""
    arr = np.ma.asarray(ds.variables[name][:]).astype(np.float64)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    for fill in (-9999.0, 1.0e20):
        arr[arr == fill] = np.nan
    return arr.reshape(-1)


def read_flag(ds, name):
    """Reference recipe: missing flags fill with 9 (= missing)."""
    return np.ma.asarray(ds.variables[name][:]).filled(9).reshape(-1)


def numeric_stats(vals):
    """Reference conventions (stats_release/common_stats.numeric_stats):
    drop non-finite first; empty -> all-NaN."""
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {key: np.nan for key in ("mean", "median", "std", "min", "max", "p95", "p99")}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }


def pool_directory(dir_path, var):
    """Pool one variable across all *.nc files in a directory.

    Returns (values, flags, units, n_files)."""
    files = sorted(Path(dir_path).glob("*.nc"))
    values, flags, units = [], [], ""
    for path in files:
        with Dataset(str(path), "r") as ds:
            vals = read_numeric_var(ds, var)
            values.append(vals)
            flag_name = "{}_flag".format(var)
            if flag_name in ds.variables:
                flags.append(read_flag(ds, flag_name))
            else:
                flags.append(np.full(vals.size, 9, dtype=np.int8))
            if not units:
                units = getattr(ds.variables[var], "units", "")
    values = np.concatenate(values) if values else np.asarray([])
    flags = np.concatenate(flags) if flags else np.asarray([])
    return values, flags, units, len(files)


def derived_fraction(dir_path, var):
    """Count present values marked derived=1 via {var}_derived_mask."""
    mask_name = "{}_derived_mask".format(var)
    n_present = n_derived = 0
    for path in sorted(Path(dir_path).glob("*.nc")):
        with Dataset(str(path), "r") as ds:
            if mask_name not in ds.variables:
                continue
            values = read_numeric_var(ds, var)
            mask = np.ma.asarray(ds.variables[mask_name][:]).filled(0).reshape(-1)
            finite = np.isfinite(values)
            n_present += int(np.count_nonzero(finite))
            n_derived += int(np.count_nonzero(finite & (mask == 1)))
    return n_present, n_derived


def _fmt(value):
    """Console formatting: NaN -> '-', else 6 significant digits."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return "{:.6g}".format(value)


def main():
    rows = []
    for dir_label, dir_path in DIRS.items():
        if not dir_path.is_dir():
            print("SKIP (not a directory): {}".format(dir_path))
            continue
        for var in VARIABLES:
            values, flags, units, n_files = pool_directory(dir_path, var)
            finite = np.isfinite(values)
            flag01 = np.isin(flags, [0, 1])
            for tier in TIERS:
                sel = finite if tier == "all" else (finite & flag01)
                stats = numeric_stats(values[sel])
                rows.append({
                    "directory": dir_label,
                    "variable": var,
                    "tier": tier,
                    "n_files": n_files,
                    "n_present": int(np.count_nonzero(finite)),
                    "n_good": int(np.count_nonzero(finite & (flags == 0))),
                    "n_estimated": int(np.count_nonzero(finite & (flags == 1))),
                    "n_suspect": int(np.count_nonzero(finite & (flags == 2))),
                    "count": int(np.count_nonzero(sel)),
                    "median": stats["median"],
                    "p95": stats["p95"],
                    "p99": stats["p99"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "std": stats["std"],
                    "unit": units,
                })

    derived_rows = []
    for dir_label, dir_path in DIRS.items():
        for var in ("SSC", "SSL"):
            n_present, n_derived = derived_fraction(dir_path, var)
            derived_rows.append({
                "directory": dir_label,
                "variable": var,
                "n_present": n_present,
                "n_derived": n_derived,
                "derived_pct": round(100.0 * n_derived / n_present, 6) if n_present else 0.0,
            })

    # ---- console ----
    print("=" * 104)
    print("EUSEDcollab Q/SSC/SSL value distribution (pooled across stations)")
    print("=" * 104)
    header = ("dir", "var", "tier", "count", "median", "p95", "p99", "max", "mean", "min", "std", "unit")
    print("{:<8} {:<4} {:<7} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}  {}".format(*header))
    print("-" * 104)
    for row in rows:
        print(
            "{:<8} {:<4} {:<7} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}  {}".format(
                row["directory"], row["variable"], row["tier"], row["count"],
                _fmt(row["median"]), _fmt(row["p95"]), _fmt(row["p99"]), _fmt(row["max"]),
                _fmt(row["mean"]), _fmt(row["min"]), _fmt(row["std"]), row["unit"],
            )
        )

    print()
    print("Flag counts (present = finite values):")
    print("{:<8} {:<4} {:>10} {:>8} {:>12} {:>9} {:>6}".format(
        "dir", "var", "n_present", "n_good", "n_estimated", "n_suspect", "files"))
    for row in rows:
        if row["tier"] != "all":
            continue
        print("{:<8} {:<4} {:>10} {:>8} {:>12} {:>9} {:>6}".format(
            row["directory"], row["variable"], row["n_present"], row["n_good"],
            row["n_estimated"], row["n_suspect"], row["n_files"]))

    print()
    print("Derived fraction (SSC/SSL derived_mask == 1):")
    print("{:<8} {:<4} {:>10} {:>10} {:>10}".format("dir", "var", "n_present", "n_derived", "pct"))
    for row in derived_rows:
        print("{:<8} {:<4} {:>10} {:>10} {:>10}".format(
            row["directory"], row["variable"], row["n_present"], row["n_derived"],
            "{:.2f}".format(row["derived_pct"])))

    # ---- CSV ----
    csv_fields = [
        "directory", "variable", "tier", "n_files", "n_present", "n_good",
        "n_estimated", "n_suspect", "count", "median", "p95", "p99", "max",
        "mean", "min", "std", "unit",
    ]
    with open(str(OUT_CSV), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("" if (isinstance(value, float) and np.isnan(value)) else value)
                for key, value in row.items()
            })
    print()
    print("CSV written: {}".format(OUT_CSV))


if __name__ == "__main__":
    main()
