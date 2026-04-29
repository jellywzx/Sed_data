"""
Check missing latitude/longitude conditions in RiverSed QC netCDF outputs.

This script reuses the same inventory parsing logic as
`fill_missing_coordinates.py`, so masked values, `_FillValue`, and
`missing_value` are interpreted consistently.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from code.runtime import resolve_output_root
from fill_missing_coordinates import build_inventory

OUTPUT_DATASET_DIR = Path(resolve_output_root(start=__file__) / "daily" / "RiverSed")
OUTPUT_QC_DIR = OUTPUT_DATASET_DIR / "qc"
DEFAULT_REPORT_CSV = OUTPUT_DATASET_DIR / "riversed_missing_coordinates_report.csv"

REPORT_COLUMNS = [
    "file",
    "dataset_branch",
    "station_id",
    "id",
    "comid",
    "reachcode",
    "river_name",
    "vpu_id",
    "rpu_id",
    "orig_lat",
    "orig_lon",
    "missing_lat",
    "missing_lon",
    "missing_both",
    "station_location",
]


def _build_missing_report(inventory_df):
    report_df = inventory_df.copy()
    report_df["missing_lat"] = report_df["orig_lat"].isna()
    report_df["missing_lon"] = report_df["orig_lon"].isna()
    report_df["missing_both"] = report_df["missing_lat"] & report_df["missing_lon"]
    report_df = report_df[report_df["missing_lat"] | report_df["missing_lon"]].copy()
    return report_df[REPORT_COLUMNS]


def _summarize_branch(branch_df):
    missing_lat = int(branch_df["orig_lat"].isna().sum())
    missing_lon = int(branch_df["orig_lon"].isna().sum())
    missing_both = int((branch_df["orig_lat"].isna() & branch_df["orig_lon"].isna()).sum())
    missing_any = int((branch_df["orig_lat"].isna() | branch_df["orig_lon"].isna()).sum())
    valid_both = int((branch_df["orig_lat"].notna() & branch_df["orig_lon"].notna()).sum())
    only_lat = int((branch_df["orig_lat"].isna() & branch_df["orig_lon"].notna()).sum())
    only_lon = int((branch_df["orig_lat"].notna() & branch_df["orig_lon"].isna()).sum())
    return {
        "total": int(len(branch_df)),
        "valid_both": valid_both,
        "missing_any": missing_any,
        "missing_both": missing_both,
        "missing_lat": missing_lat,
        "missing_lon": missing_lon,
        "only_lat": only_lat,
        "only_lon": only_lon,
    }


def _print_summary(inventory_df, missing_df, output_dir, max_examples):
    print("RiverSed missing coordinate summary")
    print(f"  Output directory: {output_dir}")
    print(f"  Total files: {len(inventory_df)}")
    print(f"  Files missing any coordinate: {len(missing_df)}")
    print(f"  Files missing both coordinates: {int(missing_df['missing_both'].sum()) if not missing_df.empty else 0}")

    for branch_name, branch_df in inventory_df.groupby("dataset_branch", sort=False):
        stats = _summarize_branch(branch_df)
        print(f"  [{branch_name}]")
        print(f"    total: {stats['total']}")
        print(f"    valid_both: {stats['valid_both']}")
        print(f"    missing_any: {stats['missing_any']}")
        print(f"    missing_both: {stats['missing_both']}")
        print(f"    missing_lat: {stats['missing_lat']}")
        print(f"    missing_lon: {stats['missing_lon']}")
        print(f"    only_lat_missing: {stats['only_lat']}")
        print(f"    only_lon_missing: {stats['only_lon']}")

    if missing_df.empty:
        return

    print(f"  Example missing files (top {max_examples}):")
    example_df = missing_df.head(max_examples)
    for row in example_df.to_dict("records"):
        status_parts = []
        if row["missing_lat"]:
            status_parts.append("lat")
        if row["missing_lon"]:
            status_parts.append("lon")
        status = "+".join(status_parts)
        print(
            "    - {file} | branch={dataset_branch} | station_id={station_id} | "
            "missing={status} | river_name={river_name} | comid={comid} | "
            "reachcode={reachcode}".format(
                file=row["file"],
                dataset_branch=row["dataset_branch"],
                station_id=row["station_id"],
                status=status,
                river_name=row["river_name"] or "",
                comid=row["comid"] or "",
                reachcode=row["reachcode"] or "",
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check missing latitude/longitude conditions in RiverSed QC netCDF outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_QC_DIR),
        help="Directory containing RiverSed QC netCDF files.",
    )
    parser.add_argument(
        "--report-csv",
        default=str(DEFAULT_REPORT_CSV),
        help="CSV path for rows missing latitude and/or longitude.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="How many missing-file examples to print in the console summary.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write the missing-record report CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    inventory_df = build_inventory(args.output_dir)
    missing_df = _build_missing_report(inventory_df)
    _print_summary(
        inventory_df,
        missing_df,
        output_dir=args.output_dir,
        max_examples=max(0, args.max_examples),
    )

    if not args.no_csv:
        report_path = Path(args.report_csv)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(report_path, index=False)
        print(f"  Missing-record CSV: {report_path}")


if __name__ == "__main__":
    main()
