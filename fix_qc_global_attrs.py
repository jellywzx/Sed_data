#!/usr/bin/env python3
"""Normalize canonical global attributes in-place for qc NetCDF files."""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.dataset_attr_profiles import get_dataset_profile
from code.global_attrs import HISTORY_NOTE, normalize_nc_attrs
from code.runtime import resolve_output_root
from tqdm import tqdm


DEFAULT_WORKERS = 24
REPORT_REL_DIR = Path("scripts_basin_test") / "output"
REPORT_CSV_NAME = "fix_qc_global_attrs_report.csv"
REPORT_TXT_NAME = "fix_qc_global_attrs_summary.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize canonical global attrs in-place for qc NetCDF files.")
    parser.add_argument("--source-root", type=Path, default=resolve_output_root(__file__), help="Source Output_r root")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name to process; can be repeated")
    parser.add_argument("--all", action="store_true", help="Process all datasets under source root")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N", help="Parallel worker count")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="Only process the first N files after filtering")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify NC files; only generate reports")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory for CSV/TXT reports; default is Output_r/scripts_basin_test/output")
    return parser.parse_args()


def _safe_relative_to(path_obj, root_obj):
    try:
        return path_obj.resolve().relative_to(root_obj.resolve())
    except Exception:
        return None


def _iter_qc_nc_files(source_root):
    for path_obj in sorted(source_root.rglob("*.nc")):
        rel = _safe_relative_to(path_obj, source_root)
        if rel is None:
            continue
        parts = rel.parts
        if len(parts) < 4:
            continue
        if parts[2] != "qc":
            continue
        yield {
            "path": path_obj,
            "rel_path": rel,
            "path_resolution": parts[0],
            "dataset": parts[1],
        }


def _filter_targets(records, datasets, process_all, limit):
    if not process_all and not datasets:
        raise SystemExit("Use --all or provide at least one --dataset.")

    dataset_filter = set(item.strip().lower() for item in datasets if str(item).strip())
    if dataset_filter:
        records = [row for row in records if row["dataset"].lower() in dataset_filter]

    if limit and limit > 0:
        records = records[:limit]
    return records


def _normalize_one(item):
    nc_path, dataset_name, path_resolution, dry_run = item
    try:
        profile = get_dataset_profile(dataset_name)
        result = normalize_nc_attrs(
            str(nc_path),
            dataset_name=dataset_name,
            path_resolution=path_resolution,
            history_note=HISTORY_NOTE,
            dry_run=dry_run,
        )
        return {
            "status": "planned" if dry_run else "normalized",
            "path": str(nc_path),
            "dataset": dataset_name,
            "path_resolution": path_resolution,
            "changed": bool(result.get("changed")),
            "changed_keys": result.get("changed_keys", []),
            "missing_after_fix": result.get("missing_after_fix", []),
            "profile_data_source_name": profile.get("data_source_name", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "normalize_error",
            "path": str(nc_path),
            "dataset": dataset_name,
            "path_resolution": path_resolution,
            "changed": False,
            "changed_keys": [],
            "missing_after_fix": [],
            "profile_data_source_name": "",
            "error": str(exc),
        }


def _write_report_csv(report_path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "path_resolution",
        "status",
        "path",
        "changed",
        "changed_keys",
        "missing_after_fix",
        "profile_data_source_name",
        "error",
    ]
    with open(str(report_path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "path_resolution": row.get("path_resolution", ""),
                    "status": row.get("status", ""),
                    "path": row.get("path", ""),
                    "changed": int(bool(row.get("changed"))),
                    "changed_keys": "|".join(row.get("changed_keys", [])),
                    "missing_after_fix": "|".join(row.get("missing_after_fix", [])),
                    "profile_data_source_name": row.get("profile_data_source_name", ""),
                    "error": row.get("error", ""),
                }
            )


def _write_summary_txt(summary_path, rows, dry_run, source_root):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    status_counter = Counter()
    changed_key_counter = Counter()
    missing_key_counter = Counter()
    dataset_counter = defaultdict(Counter)

    for row in rows:
        status_counter[row.get("status", "")] += 1
        dataset_counter[row.get("dataset", "")][row.get("status", "")] += 1
        for key in row.get("changed_keys", []):
            changed_key_counter[key] += 1
        for key in row.get("missing_after_fix", []):
            missing_key_counter[key] += 1

    with open(str(summary_path), "w", encoding="utf-8") as handle:
        handle.write("fix_qc_global_attrs summary\n")
        handle.write("mode            : {0}\n".format("dry-run" if dry_run else "apply"))
        handle.write("source_root     : {0}\n".format(source_root))
        handle.write("total_rows      : {0}\n".format(len(rows)))
        handle.write("\nstatus counts\n")
        for status, count in sorted(status_counter.items()):
            handle.write("  {0:<20s} {1}\n".format(status, count))

        handle.write("\nchanged key counts\n")
        for key, count in changed_key_counter.most_common():
            handle.write("  {0:<30s} {1}\n".format(key, count))

        handle.write("\nmissing-after-fix key counts\n")
        for key, count in missing_key_counter.most_common():
            handle.write("  {0:<30s} {1}\n".format(key, count))

        handle.write("\nper-dataset status counts\n")
        for dataset_name in sorted(dataset_counter):
            handle.write("  [{0}]\n".format(dataset_name))
            for status, count in sorted(dataset_counter[dataset_name].items()):
                handle.write("    {0:<18s} {1}\n".format(status, count))


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    workers = max(1, int(args.workers))

    if not source_root.is_dir():
        raise SystemExit("Source root does not exist: {0}".format(source_root))

    records = list(_iter_qc_nc_files(source_root))
    records = _filter_targets(records, args.dataset, args.all, args.limit)
    if not records:
        raise SystemExit("No qc NetCDF files matched the current selection.")

    print("Source root : {0}".format(source_root))
    print("Files       : {0}".format(len(records)))
    print("Mode        : {0}".format("dry-run" if args.dry_run else "apply"))
    print("\nPhase 1: in-place canonical global-attr normalization")

    normalize_tasks = [
        (row["path"], row["dataset"], row["path_resolution"], args.dry_run)
        for row in records
    ]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_normalize_one, item) for item in normalize_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    results = sorted(results, key=lambda row: (row.get("dataset", ""), row.get("path", "")))

    report_dir = args.report_dir.expanduser().resolve() if args.report_dir else source_root / REPORT_REL_DIR
    report_csv = report_dir / REPORT_CSV_NAME
    report_txt = report_dir / REPORT_TXT_NAME
    _write_report_csv(report_csv, results)
    _write_summary_txt(report_txt, results, args.dry_run, source_root)

    print("\nReports written:")
    print("  {0}".format(report_csv))
    print("  {0}".format(report_txt))


if __name__ == "__main__":
    main()
