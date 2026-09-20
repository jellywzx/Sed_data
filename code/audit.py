"""Audit finalized source-level QC NetCDF outputs.

This module counts post-processing records consistently across datasets. A
record is sediment-eligible when SSC or SSL is non-missing. Q-only and fully
missing time steps are counted as screened for downstream sediment integration.
QC flags 2/3 are retained and are therefore not counted as screened records.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import netCDF4 as nc
import numpy as np

MISSING_FLAG = 9
ANALYSIS_READY_FLAGS = (0, 1)

DATASET_ALIASES = {
    "alideboer": "Ali and De Boer",
    "chaophrayariver": "Chao Phraya River",
    "dethier": "Dethier",
    "eusedcollab": "EUSEDcollab",
    "eurasianriver": "Eurasian River",
    "fukushima": "Fukushima",
    "gfqav2": "GFQA_v2",
    "gsed": "GSED",
    "glorise": "GloRiSe",
    "hma": "HMA",
    "hybam": "HYBAM",
    "huanghe": "Huanghe",
    "hydat": "HYDAT",
    "mekongdelta": "Mekong Delta",
    "milliman": "Milliman",
    "myanmar": "Myanmar Rivers",
    "myanmarrivers": "Myanmar Rivers",
    "nerc": "NERC Avon",
    "nercavon": "NERC Avon",
    "rhine": "Rhine",
    "riversed": "RivSed",
    "rivsed": "RivSed",
    "robotham": "Robotham",
    "shashijianli": "Shashi-Jianli",
    "usgs": "USGS NWIS",
    "usgsnwis": "USGS NWIS",
    "vanmaercke": "Vanmaercke",
    "yajiang": "Yajiang",
    "bayern": "Bayern",
}

RESOLUTION_ALIASES = {
    "daily": "daily",
    "day": "daily",
    "monthly": "monthly",
    "month": "monthly",
    "annual": "annual",
    "annually": "annual",
    "yearly": "annual",
    "year": "annual",
    "climatology": "climatology",
    "climatological": "climatology",
    "annualclimatology": "climatology",
    "annuallyclimatology": "climatology",
}

EXCLUDED_PATH_PARTS = {
    "bs",
    "diagnostic",
    "diagnostics",
    "diagnosticplots",
    "plots",
    "figures",
    "test",
    "tests",
}

AUDIT_COLUMNS = [
    "dataset",
    "resolution",
    "n_files",
    "nc_time_steps",
    "sediment_eligible_records",
    "screened_for_integration_records",
    "q_only_records",
    "fully_missing_records",
    "sediment_eligibility_percent",
    "analysis_ready_sediment_records",
    "sediment_suspect_bad_only_records",
    "Q_nonmissing",
    "SSC_nonmissing",
    "SSL_nonmissing",
    "Q_good",
    "Q_derived",
    "Q_suspect",
    "Q_bad",
    "Q_missing",
    "SSC_good",
    "SSC_derived",
    "SSC_suspect",
    "SSC_bad",
    "SSC_missing",
    "SSL_good",
    "SSL_derived",
    "SSL_suspect",
    "SSL_bad",
    "SSL_missing",
    "flag_value_mismatch_count",
    "files_missing_final_flags",
    "unreadable_files",
]


def _token(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def canonical_dataset_name(value):
    return DATASET_ALIASES.get(_token(value), str(value))


def canonical_resolution(value):
    text = str(value).strip()
    return RESOLUTION_ALIASES.get(_token(text), text.lower() or "unknown")


def _resize_bool(values, n):
    out = np.zeros(n, dtype=bool)
    flat = np.asarray(values, dtype=bool).reshape(-1)
    m = min(n, flat.size)
    out[:m] = flat[:m]
    return out


def _resize_int(values, n):
    out = np.full(n, MISSING_FLAG, dtype=np.int16)
    masked = np.ma.asarray(values).reshape(-1)
    flat = np.asarray(np.ma.filled(masked, MISSING_FLAG)).reshape(-1)
    m = min(n, flat.size)
    if m:
        numeric = np.asarray(flat[:m], dtype=float)
        numeric = np.where(np.isfinite(numeric), numeric, MISSING_FLAG)
        out[:m] = numeric.astype(np.int16)
    return out


def _presence(variable, n):
    if variable is None:
        return np.zeros(n, dtype=bool)

    arr = np.ma.asarray(variable[:]).reshape(-1)
    data = np.asarray(np.ma.filled(arr, np.nan), dtype=float)
    present = (~np.ma.getmaskarray(arr).reshape(-1)) & np.isfinite(data)

    fills = [-9999.0]
    for attr in ("_FillValue", "missing_value"):
        if hasattr(variable, attr):
            try:
                fills.extend(
                    np.asarray(getattr(variable, attr), dtype=float).reshape(-1).tolist()
                )
            except (TypeError, ValueError):
                pass

    for fill in fills:
        if np.isfinite(fill):
            present &= ~np.isclose(data, fill, rtol=1e-5, atol=1e-5)

    return _resize_bool(present, n)


def _flags(ds, name, n):
    flag_name = f"{name}_flag"
    if flag_name not in ds.variables:
        return None
    return _resize_int(ds.variables[flag_name][:], n)


def _record_count(ds):
    if "time" in ds.variables:
        return int(np.asarray(ds.variables["time"][:]).size)

    sizes = []
    for name in ("Q", "SSC", "SSL", "Q_flag", "SSC_flag", "SSL_flag"):
        if name in ds.variables:
            sizes.append(int(np.asarray(ds.variables[name][:]).size))
    return max(sizes) if sizes else 0


def _path_parts(path, root):
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def _dataset_from_path(path, root):
    parts = _path_parts(path, root)
    for part in parts:
        if _token(part) in DATASET_ALIASES:
            return canonical_dataset_name(part)

    tokens = [_token(part) for part in parts]
    if "qc" in tokens and tokens.index("qc") > 0:
        return canonical_dataset_name(parts[tokens.index("qc") - 1])

    return canonical_dataset_name(path.parent.name)


def _resolution_from_path(path, root):
    for part in _path_parts(path, root):
        if _token(part) in RESOLUTION_ALIASES:
            return canonical_resolution(part)
    return "unknown"


def _is_final_qc_file(path, root):
    tokens = [_token(part) for part in _path_parts(path, root)[:-1]]
    return "qc" in tokens and not any(token in EXCLUDED_PATH_PARTS for token in tokens)


def _empty(dataset, resolution):
    row = {column: 0 for column in AUDIT_COLUMNS}
    row["dataset"] = dataset
    row["resolution"] = resolution
    row["sediment_eligibility_percent"] = 0.0
    return row


def audit_netcdf_file(path, output_root):
    """Audit one finalized source-level QC NetCDF file."""
    path = Path(path)
    output_root = Path(output_root)
    row = _empty(
        _dataset_from_path(path, output_root),
        _resolution_from_path(path, output_root),
    )
    row["n_files"] = 1

    try:
        with nc.Dataset(str(path), "r") as ds:
            resolution = getattr(ds, "temporal_resolution", "")
            if str(resolution).strip():
                row["resolution"] = canonical_resolution(resolution)

            n = _record_count(ds)
            row["nc_time_steps"] = n
            if n <= 0:
                return row

            present = {
                name: _presence(ds.variables.get(name), n)
                for name in ("Q", "SSC", "SSL")
            }
            flags = {
                name: _flags(ds, name, n)
                for name in ("Q", "SSC", "SSL")
            }

            sediment = present["SSC"] | present["SSL"]
            q_only = present["Q"] & ~sediment
            fully_missing = ~(present["Q"] | present["SSC"] | present["SSL"])

            row["sediment_eligible_records"] = int(sediment.sum())
            row["q_only_records"] = int(q_only.sum())
            row["fully_missing_records"] = int(fully_missing.sum())
            row["screened_for_integration_records"] = int(
                (q_only | fully_missing).sum()
            )
            row["sediment_eligibility_percent"] = (
                100.0 * int(sediment.sum()) / n
            )

            analysis_ready = {}
            missing_flags = False
            mismatches = 0

            for name in ("Q", "SSC", "SSL"):
                row[f"{name}_nonmissing"] = int(present[name].sum())
                flag = flags[name]

                if flag is None:
                    missing_flags = True
                    analysis_ready[name] = present[name].copy()
                    continue

                for value, label in (
                    (0, "good"),
                    (1, "derived"),
                    (2, "suspect"),
                    (3, "bad"),
                    (9, "missing"),
                ):
                    row[f"{name}_{label}"] = int(np.sum(flag == value))

                mismatches += int(
                    np.sum((flag != MISSING_FLAG) != present[name])
                )
                analysis_ready[name] = (
                    present[name] & np.isin(flag, ANALYSIS_READY_FLAGS)
                )

            sediment_ready = analysis_ready["SSC"] | analysis_ready["SSL"]
            row["analysis_ready_sediment_records"] = int(
                sediment_ready.sum()
            )
            row["sediment_suspect_bad_only_records"] = int(
                np.sum(sediment & ~sediment_ready)
            )
            row["flag_value_mismatch_count"] = mismatches
            row["files_missing_final_flags"] = int(missing_flags)

    except Exception:
        row["unreadable_files"] = 1

    return row


def _sum_rows(rows, dataset, resolution):
    rows = list(rows)
    out = _empty(dataset, resolution)

    for column in AUDIT_COLUMNS:
        if column in {
            "dataset",
            "resolution",
            "sediment_eligibility_percent",
        }:
            continue
        out[column] = sum(
            int(row.get(column, 0) or 0)
            for row in rows
        )

    total = int(out["nc_time_steps"])
    out["sediment_eligibility_percent"] = (
        100.0 * int(out["sediment_eligible_records"]) / total
        if total
        else 0.0
    )
    return out


def collect_dataset_output_audit(output_root, final_qc_only=True):
    """Return dataset and dataset-by-resolution summaries."""
    root = Path(output_root).expanduser().resolve()
    if not root.exists():
        return [], []

    file_rows = []
    for path in sorted(root.rglob("*.nc")):
        if final_qc_only and not _is_final_qc_file(path, root):
            continue
        file_rows.append(audit_netcdf_file(path, root))

    grouped_dataset = defaultdict(list)
    grouped_resolution = defaultdict(list)

    for row in file_rows:
        grouped_dataset[row["dataset"]].append(row)
        grouped_resolution[(row["dataset"], row["resolution"])].append(row)

    by_dataset = [
        _sum_rows(rows, dataset, "all")
        for dataset, rows in sorted(grouped_dataset.items())
    ]
    by_resolution = [
        _sum_rows(rows, dataset, resolution)
        for (dataset, resolution), rows in sorted(grouped_resolution.items())
    ]
    return by_dataset, by_resolution


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()

        for row in rows:
            clean = dict(row)
            clean["sediment_eligibility_percent"] = (
                f"{float(row.get('sediment_eligibility_percent', 0.0)):.4f}"
            )
            writer.writerow(
                {column: clean.get(column, "") for column in AUDIT_COLUMNS}
            )


def write_dataset_output_audit(
    output_root,
    audit_dir=None,
    final_qc_only=True,
):
    """Write Output_r/audit dataset summaries and return their paths."""
    root = Path(output_root).expanduser().resolve()
    audit_dir = (
        Path(audit_dir).expanduser().resolve()
        if audit_dir
        else root / "audit"
    )

    by_dataset, by_resolution = collect_dataset_output_audit(
        root,
        final_qc_only=final_qc_only,
    )

    dataset_csv = audit_dir / "dataset_output_audit.csv"
    resolution_csv = audit_dir / "dataset_output_audit_by_resolution.csv"

    _write_csv(dataset_csv, by_dataset)
    _write_csv(resolution_csv, by_resolution)

    return {
        "dataset": dataset_csv,
        "by_resolution": resolution_csv,
    }
