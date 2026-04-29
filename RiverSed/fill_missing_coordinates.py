"""
Fill missing RiverSed reach coordinates using local NHDPlus flowlines.

This script is designed for the existing QC netCDF outputs in
Output_r/daily/RiverSed/qc. It builds four CSV products:

1. inventory: every RiverSed netCDF file with parsed identifiers/coordinates
2. reference: the local NHDPlus flowline lookup with representative points
3. candidates: one candidate row for every file that still needs coordinates
4. manual_review: unresolved or non-high-confidence candidates

By default the script only writes the CSV products. Pass --apply to write the
high-confidence coordinate fills back into the netCDF files in place.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from code.runtime import resolve_output_root, resolve_source_root
from convert_to_netcdf import (
    CONUS_LATITUDE_RANGE,
    CONUS_LONGITUDE_RANGE,
    RIVERSED_METADATA_FLOWLINE,
    SCALAR_COORD_FILL_VALUE,
    _coordinates_within_conus,
    _normalize_optional_text,
    _normalize_reach_code,
    _normalize_riversed_id,
    load_riversed_flowline_reference,
)

OUTPUT_QC_DIR = Path(resolve_output_root(start=__file__) / "daily" / "RiverSed" / "qc")
SOURCE_DIR = Path(resolve_source_root(start=__file__) / "RiverSed")

DEFAULT_FLOWLINE_PATH = Path(RIVERSED_METADATA_FLOWLINE)
DEFAULT_INVENTORY_CSV = OUTPUT_QC_DIR / "riversed_coord_fill_inventory.csv"
DEFAULT_REFERENCE_CSV = OUTPUT_QC_DIR / "riversed_coord_fill_reference_flowline.csv"
DEFAULT_CANDIDATES_CSV = OUTPUT_QC_DIR / "riversed_coord_fill_candidates.csv"
DEFAULT_MANUAL_REVIEW_CSV = OUTPUT_QC_DIR / "riversed_coord_fill_manual_review.csv"
INVENTORY_COLUMNS = [
    "file",
    "path",
    "dataset_branch",
    "station_id",
    "id",
    "comid",
    "reachcode",
    "river_name",
    "normalized_river_name",
    "station_location",
    "vpu_id",
    "rpu_id",
    "orig_lat",
    "orig_lon",
    "needs_fill",
]
CANDIDATE_COLUMNS = [
    "file",
    "path",
    "station_id",
    "id",
    "comid",
    "reachcode",
    "river_name",
    "vpu_id",
    "rpu_id",
    "orig_lat",
    "orig_lon",
    "new_lat",
    "new_lon",
    "match_key",
    "match_method",
    "source_dataset",
    "confidence",
    "review_flag",
    "review_reason",
    "reference_id",
    "reference_comid",
    "reference_reachcode",
    "reference_river_name",
]

COMID_PATTERN = re.compile(r"COMID\s+(\d+)", re.IGNORECASE)
REACH_PATTERN = re.compile(r"reach\s+([0-9A-Za-z]+)", re.IGNORECASE)
RIVERSED_STATION_ID_PATTERN = re.compile(r"RiverSed_(\d+)$")


def _read_text_attr(dataset, attr_name):
    value = getattr(dataset, attr_name, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip() if value is not None else ""


def _read_scalar_coordinate(dataset, variable_name):
    if variable_name not in dataset.variables:
        return np.nan

    variable = dataset.variables[variable_name]
    value = np.ma.asarray(variable[:]).reshape(-1)
    if value.size == 0:
        return np.nan
    if np.ma.getmaskarray(value).reshape(-1)[0]:
        return np.nan

    coordinate = float(np.ma.getdata(value[0]))
    fill_values = [SCALAR_COORD_FILL_VALUE]
    for attr_name in ("_FillValue", "missing_value"):
        attr_value = getattr(variable, attr_name, None)
        if attr_value is not None:
            try:
                fill_values.append(float(np.asarray(attr_value).reshape(-1)[0]))
            except Exception:
                pass

    if not np.isfinite(coordinate):
        return np.nan
    if any(np.isclose(coordinate, fill_value) for fill_value in fill_values):
        return np.nan
    return coordinate


def _parse_station_id_to_id(station_id):
    match = RIVERSED_STATION_ID_PATTERN.search(station_id or "")
    return match.group(1) if match else ""


def _parse_comid(station_location):
    match = COMID_PATTERN.search(station_location or "")
    return _normalize_riversed_id(match.group(1)) if match else ""


def _parse_reach_code(station_location):
    match = REACH_PATTERN.search(station_location or "")
    return _normalize_reach_code(match.group(1)) if match else ""


def _normalize_river_name(value):
    text = _normalize_optional_text(value).lower()
    if not text:
        return ""

    text = re.sub(r"\briv\b", "river", text)
    text = re.sub(r"\br\.\b", "river", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_inventory_record(nc_path):
    with nc.Dataset(nc_path, "r") as dataset:
        station_id = _read_text_attr(dataset, "station_id")
        station_location = _read_text_attr(dataset, "station_location")
        river_name = _read_text_attr(dataset, "river_name")
        comid = _normalize_riversed_id(
            _read_text_attr(dataset, "comid") or _parse_comid(station_location)
        )
        reach_code = _normalize_reach_code(
            _read_text_attr(dataset, "reach_code") or _parse_reach_code(station_location)
        )
        vpu_id = _normalize_optional_text(_read_text_attr(dataset, "vpu_id"))
        rpu_id = _normalize_optional_text(_read_text_attr(dataset, "rpu_id"))
        latitude = _read_scalar_coordinate(dataset, "latitude")
        longitude = _read_scalar_coordinate(dataset, "longitude")
        dataset_branch = (
            "riversed_reach"
            if station_id.startswith("RiverSed_")
            else "aquasat"
        )

        return {
            "file": nc_path.name,
            "path": str(nc_path),
            "dataset_branch": dataset_branch,
            "station_id": station_id,
            "id": _parse_station_id_to_id(station_id),
            "comid": comid,
            "reachcode": reach_code,
            "river_name": river_name,
            "normalized_river_name": _normalize_river_name(river_name),
            "station_location": station_location,
            "vpu_id": vpu_id,
            "rpu_id": rpu_id,
            "orig_lat": latitude,
            "orig_lon": longitude,
            "needs_fill": bool(
                dataset_branch == "riversed_reach"
                and (not np.isfinite(latitude) or not np.isfinite(longitude))
            ),
        }


def build_inventory(output_dir):
    records = []
    for nc_path in sorted(Path(output_dir).glob("RiverSed_*.nc")):
        records.append(_extract_inventory_record(nc_path))
    return pd.DataFrame(records, columns=INVENTORY_COLUMNS)


def build_reference_table(flowline_path):
    reference_df = load_riversed_flowline_reference(flowline_path).copy()
    reference_df["normalized_river_name"] = reference_df["river_name"].map(_normalize_river_name)
    reference_df = reference_df.rename(columns={"lat": "rep_lat", "long": "rep_lon"})
    reference_columns = [
        "ID",
        "comid",
        "reach_code",
        "river_name",
        "normalized_river_name",
        "vpu_id",
        "rpu_id",
        "rep_lat",
        "rep_lon",
        "coordinate_source",
        "coordinate_method",
        "coordinate_confidence",
    ]
    return reference_df[reference_columns].copy()


def _row_dict_from_reference(reference_row):
    return {
        "reference_id": _normalize_optional_text(reference_row.get("ID", "")),
        "reference_comid": _normalize_optional_text(reference_row.get("comid", "")),
        "reference_reachcode": _normalize_reach_code(reference_row.get("reach_code", "")),
        "reference_river_name": _normalize_optional_text(reference_row.get("river_name", "")),
        "source_dataset": _normalize_optional_text(reference_row.get("coordinate_source", "")),
    }


def _river_name_mismatch(record, reference_row):
    source_name = record.get("normalized_river_name", "")
    reference_name = _normalize_river_name(reference_row.get("river_name", ""))
    return bool(source_name and reference_name and source_name != reference_name)


def _make_candidate(record, *, reference_row=None, match_key="", match_method="", confidence="", review_flag=False, review_reason=""):
    candidate = {
        "file": record["file"],
        "path": record["path"],
        "station_id": record["station_id"],
        "id": record["id"],
        "comid": record["comid"],
        "reachcode": record["reachcode"],
        "river_name": record["river_name"],
        "vpu_id": record["vpu_id"],
        "rpu_id": record["rpu_id"],
        "orig_lat": record["orig_lat"],
        "orig_lon": record["orig_lon"],
        "new_lat": np.nan,
        "new_lon": np.nan,
        "match_key": match_key,
        "match_method": match_method,
        "source_dataset": "",
        "confidence": confidence,
        "review_flag": bool(review_flag),
        "review_reason": review_reason,
        "reference_id": "",
        "reference_comid": "",
        "reference_reachcode": "",
        "reference_river_name": "",
    }

    if reference_row is None:
        return candidate

    candidate.update(_row_dict_from_reference(reference_row))
    candidate["new_lat"] = float(reference_row["rep_lat"])
    candidate["new_lon"] = float(reference_row["rep_lon"])
    candidate["source_dataset"] = _normalize_optional_text(reference_row.get("coordinate_source", ""))

    if not _coordinates_within_conus(candidate["new_lat"], candidate["new_lon"]):
        candidate["review_flag"] = True
        candidate["review_reason"] = "candidate_outside_conus"

    if _river_name_mismatch(record, reference_row):
        candidate["review_flag"] = True
        candidate["review_reason"] = "river_name_mismatch"

    return candidate


def _filter_reach_candidates(reference_subset, record):
    filtered = reference_subset.copy()
    used_constraint = False

    if record["rpu_id"]:
        filtered = filtered[filtered["rpu_id"] == record["rpu_id"]]
        used_constraint = True
    elif record["vpu_id"]:
        filtered = filtered[filtered["vpu_id"] == record["vpu_id"]]
        used_constraint = True

    normalized_name = record.get("normalized_river_name", "")
    if normalized_name:
        same_name = filtered[filtered["normalized_river_name"] == normalized_name]
        if not same_name.empty:
            filtered = same_name
            used_constraint = True

    if not used_constraint:
        return pd.DataFrame(columns=reference_subset.columns)
    return filtered


def _match_by_reach_code(record, reference_by_reach):
    reach_code = record.get("reachcode", "")
    if not reach_code or reach_code not in reference_by_reach:
        return None

    filtered = _filter_reach_candidates(reference_by_reach[reach_code], record)
    if filtered.empty:
        return None
    if len(filtered) == 1:
        reference_row = filtered.iloc[0]
        return _make_candidate(
            record,
            reference_row=reference_row,
            match_key=f"reachcode={reach_code}",
            match_method="flowline_midpoint_by_reachcode",
            confidence="medium",
        )
    return _make_candidate(
        record,
        match_key=f"reachcode={reach_code}",
        match_method="flowline_midpoint_by_reachcode",
        confidence="medium",
        review_flag=True,
        review_reason="ambiguous_reachcode_match",
    )


def _match_by_river_name(record, reference_by_name):
    normalized_name = record.get("normalized_river_name", "")
    if not normalized_name or normalized_name not in reference_by_name:
        return None

    reference_subset = reference_by_name[normalized_name]
    if record["rpu_id"]:
        filtered = reference_subset[reference_subset["rpu_id"] == record["rpu_id"]]
        match_method = "flowline_midpoint_by_river_name_rpu"
    elif record["vpu_id"]:
        filtered = reference_subset[reference_subset["vpu_id"] == record["vpu_id"]]
        match_method = "flowline_midpoint_by_river_name_vpu"
    else:
        return None

    if filtered.empty:
        return None
    if len(filtered) == 1:
        reference_row = filtered.iloc[0]
        return _make_candidate(
            record,
            reference_row=reference_row,
            match_key=f"river_name={normalized_name}",
            match_method=match_method,
            confidence="low",
        )
    return _make_candidate(
        record,
        match_key=f"river_name={normalized_name}",
        match_method=match_method,
        confidence="low",
        review_flag=True,
        review_reason="ambiguous_river_name_match",
    )


def build_candidate_table(inventory_df, reference_df):
    missing_df = inventory_df[
        (inventory_df["dataset_branch"] == "riversed_reach") & inventory_df["needs_fill"]
    ].copy()

    reference_by_id = reference_df.set_index("ID", drop=False).to_dict("index")
    reference_by_comid = (
        reference_df[reference_df["comid"] != ""]
        .set_index("comid", drop=False)
        .to_dict("index")
    )
    reference_by_reach = {
        reach_code: group.copy()
        for reach_code, group in reference_df[reference_df["reach_code"] != ""].groupby("reach_code", sort=False)
    }
    reference_by_name = {
        normalized_name: group.copy()
        for normalized_name, group in reference_df[reference_df["normalized_river_name"] != ""].groupby(
            "normalized_river_name",
            sort=False,
        )
    }

    candidates = []
    for record in missing_df.to_dict("records"):
        candidate = None

        if record["id"] and record["id"] in reference_by_id:
            reference_row = reference_by_id[record["id"]]
            candidate = _make_candidate(
                record,
                reference_row=reference_row,
                match_key=f"ID={record['id']}",
                match_method="flowline_midpoint_by_id",
                confidence="high",
            )
        elif record["comid"] and record["comid"] in reference_by_comid:
            reference_row = reference_by_comid[record["comid"]]
            candidate = _make_candidate(
                record,
                reference_row=reference_row,
                match_key=f"COMID={record['comid']}",
                match_method="flowline_midpoint_by_comid",
                confidence="high",
            )
        else:
            candidate = _match_by_reach_code(record, reference_by_reach)
            if candidate is None:
                candidate = _match_by_river_name(record, reference_by_name)

        if candidate is None:
            candidate = _make_candidate(
                record,
                confidence="unresolved",
                review_flag=True,
                review_reason="no_local_reference_match",
            )

        candidates.append(candidate)

    return pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)


def build_manual_review_table(candidate_df):
    if candidate_df.empty:
        return candidate_df.copy()

    return candidate_df[
        candidate_df["review_flag"]
        | (~candidate_df["confidence"].isin(["high"]))
        | candidate_df["new_lat"].isna()
        | candidate_df["new_lon"].isna()
    ].copy()


def _write_csv(dataframe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def apply_high_confidence_updates(candidate_df):
    applied = 0
    fill_date = datetime.now().strftime("%Y-%m-%d")

    writable_candidates = candidate_df[
        (candidate_df["confidence"] == "high")
        & (~candidate_df["review_flag"])
        & candidate_df["new_lat"].notna()
        & candidate_df["new_lon"].notna()
    ].copy()

    for record in writable_candidates.to_dict("records"):
        nc_path = Path(record["path"])
        with nc.Dataset(nc_path, "r+") as dataset:
            if "latitude" not in dataset.variables or "longitude" not in dataset.variables:
                raise ValueError(f"Missing latitude/longitude variables in {nc_path}")

            dataset.variables["latitude"][:] = np.float32(record["new_lat"])
            dataset.variables["longitude"][:] = np.float32(record["new_lon"])
            dataset.setncattr("coordinate_source", record["source_dataset"])
            dataset.setncattr("coordinate_method", record["match_method"])
            dataset.setncattr("coordinate_confidence", record["confidence"])
            dataset.setncattr("coordinate_fill_date", fill_date)
        applied += 1

    return applied


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing RiverSed reach coordinates from local NHDPlus flowlines."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_QC_DIR),
        help="Directory containing RiverSed QC netCDF files.",
    )
    parser.add_argument(
        "--flowline-shp",
        default=str(DEFAULT_FLOWLINE_PATH),
        help="Path to nhdplusv2_modified_v1.0.shp.",
    )
    parser.add_argument(
        "--inventory-csv",
        default=str(DEFAULT_INVENTORY_CSV),
        help="Path to write the inventory CSV.",
    )
    parser.add_argument(
        "--reference-csv",
        default=str(DEFAULT_REFERENCE_CSV),
        help="Path to write the reference flowline CSV.",
    )
    parser.add_argument(
        "--candidates-csv",
        default=str(DEFAULT_CANDIDATES_CSV),
        help="Path to write the candidate CSV.",
    )
    parser.add_argument(
        "--manual-review-csv",
        default=str(DEFAULT_MANUAL_REVIEW_CSV),
        help="Path to write the manual review CSV.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write high-confidence coordinate fills back into the netCDF files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    inventory_df = build_inventory(args.output_dir)
    reference_df = build_reference_table(args.flowline_shp)
    candidate_df = build_candidate_table(inventory_df, reference_df)
    manual_review_df = build_manual_review_table(candidate_df)

    _write_csv(inventory_df, args.inventory_csv)
    _write_csv(reference_df, args.reference_csv)
    _write_csv(candidate_df, args.candidates_csv)
    _write_csv(manual_review_df, args.manual_review_csv)

    print("RiverSed coordinate fill summary")
    print(f"  Inventory rows: {len(inventory_df)}")
    print(
        "  Missing reach coordinates: {0}".format(
            int(
                inventory_df[
                    (inventory_df["dataset_branch"] == "riversed_reach")
                    & inventory_df["needs_fill"]
                ].shape[0]
            )
        )
    )
    print(f"  Candidate rows: {len(candidate_df)}")
    print(
        "  High-confidence candidates: {0}".format(
            int((candidate_df["confidence"] == "high").sum()) if not candidate_df.empty else 0
        )
    )
    print(f"  Manual review rows: {len(manual_review_df)}")
    print(
        "  Geographic bounds: lat {0} to {1}, lon {2} to {3}".format(
            CONUS_LATITUDE_RANGE[0],
            CONUS_LATITUDE_RANGE[1],
            CONUS_LONGITUDE_RANGE[0],
            CONUS_LONGITUDE_RANGE[1],
        )
    )

    if args.apply:
        applied = apply_high_confidence_updates(candidate_df)
        print(f"  Applied coordinate fills: {applied}")
    else:
        print("  Apply mode: dry-run (CSV outputs only)")


if __name__ == "__main__":
    main()
