#!/usr/bin/env python3
"""Backfill GFQA station metadata global attrs into existing NetCDF files."""

import argparse
import re
from pathlib import Path

import netCDF4 as nc
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_METADATA_CSV = PROJECT_ROOT / "Source" / "GFQA_v2" / "sed" / "GEMStat_station_metadata.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "Output_r" / "daily" / "GFQA_v2"
DEFAULT_ORGANIZED_DIR = PROJECT_ROOT / "output_resolution_organized" / "daily"


def clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def first_text(row, columns, default=""):
    for column in columns:
        if column in row.index:
            text = clean_text(row.get(column))
            if text:
                return text
    return default


def station_name_for(row, station_id):
    return first_text(
        row,
        ["Station Identifier", "Station Narrative", "Water Body Name"],
        default=station_id,
    )


def river_name_for(row):
    return first_text(row, ["Water Body Name", "Main Basin"])


def parse_station_id(path):
    match = re.search(r"GFQA_([A-Za-z0-9-]+)$", path.stem)
    return match.group(1) if match else ""


def load_metadata(path):
    df = pd.read_csv(path, sep=";", encoding="iso-8859-1")
    df["GEMS Station Number"] = df["GEMS Station Number"].astype(str).str.strip()
    return df.drop_duplicates("GEMS Station Number").set_index("GEMS Station Number")


def iter_targets(raw_dir, organized_dir):
    targets = []
    if raw_dir.is_dir():
        targets.extend(sorted(raw_dir.glob("GFQA_*.nc")))
        qc_dir = raw_dir / "qc"
        if qc_dir.is_dir():
            targets.extend(sorted(qc_dir.glob("GFQA_*.nc")))
    if organized_dir.is_dir():
        targets.extend(sorted(organized_dir.glob("GFQA_v2_daily_GFQA_*.nc")))
    seen = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def attr_payload(path, metadata):
    station_id = parse_station_id(path)
    if not station_id:
        return None, "unparsed_station_id", {}

    if station_id in metadata.index:
        row = metadata.loc[station_id]
        station_name = station_name_for(row, station_id)
        river_name = river_name_for(row)
    else:
        station_name = station_id
        river_name = ""

    desired = {
        "station_id": station_id,
        "Source_ID": station_id,
        "source_station_id": station_id,
        "station_name": station_name,
        "river_name": river_name,
    }

    status = "metadata_found" if station_id in metadata.index else "metadata_missing"
    return station_id, status, desired


def apply_attrs(path, attrs):
    if not attrs:
        return
    with nc.Dataset(str(path), "a") as ds:
        for key, value in attrs.items():
            setattr(ds, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--organized-dir", type=Path, default=DEFAULT_ORGANIZED_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_csv)
    paths = list(iter_targets(args.raw_dir, args.organized_dir))
    if args.limit > 0:
        paths = paths[: args.limit]

    total = len(paths)
    writable = 0
    missing = 0
    unparsed = 0
    failed = 0

    for idx, path in enumerate(paths, start=1):
        station_id, status, attrs = attr_payload(path, metadata)
        if status == "unparsed_station_id":
            unparsed += 1
            continue
        if status == "metadata_missing":
            missing += 1
        writable += 1
        if not args.dry_run:
            try:
                apply_attrs(path, attrs)
            except Exception as exc:
                failed += 1
                print("FAILED {}: {}".format(path, exc), flush=True)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print("progress {}/{}".format(idx, total), flush=True)

    mode = "dry-run" if args.dry_run else "applied"
    print(
        "GFQA station metadata backfill {} | files={} writable={} metadata_missing={} unparsed={} failed={}".format(
            mode,
            total,
            writable,
            missing,
            unparsed,
            failed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
