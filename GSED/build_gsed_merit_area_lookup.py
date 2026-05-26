#!/usr/bin/env python3
"""
Export GSED reach geometry hints.

This script intentionally does not perform MERIT matching, upstream basin
tracing, or upstream-area lookup. It only extracts the public GSED reach
midpoint and endpoint candidates needed later by the integration pipeline.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.runtime import resolve_output_root, resolve_source_root


def _load_gsed_helpers():
    helper_path = Path(__file__).with_name("1_process_gsed_cf18.py")
    spec = importlib.util.spec_from_file_location("process_gsed_cf18", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load GSED helper module: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan


def _endpoint_value(endpoint_candidates, index, key):
    try:
        return _safe_float(endpoint_candidates[index].get(key))
    except (IndexError, AttributeError, TypeError):
        return np.nan


def _derive_defaults():
    source_dir = resolve_source_root(__file__) / "GSED" / "GSED"
    output_dir = resolve_output_root(__file__, create=True) / "monthly" / "GSED" / "qc"
    return {
        "csv_file": source_dir / "GSED_Reach_Monthly_SSC.csv",
        "shapefile": source_dir / "GSED_Reach.shp",
        "output_csv": output_dir / "GSED_Reach_geometry_hints.csv",
    }


def build_geometry_hints(csv_file, shapefile, limit=None):
    helpers = _load_gsed_helpers()
    source_df = pd.read_csv(csv_file, usecols=["R_ID"])
    if limit is not None:
        source_df = source_df.head(limit)

    r_ids = [helpers._normalize_gsed_rid(r_id) for r_id in source_df["R_ID"].tolist()]
    metadata = helpers.load_gsed_reach_metadata(shapefile, target_rids=r_ids)

    rows = []
    for r_id in r_ids:
        meta = metadata.get(r_id) or {}
        endpoint_candidates = meta.get("endpoint_candidates") or []
        rows.append(
            {
                "R_ID": r_id,
                "reach_midpoint_lat": _safe_float(meta.get("midpoint_latitude", meta.get("latitude"))),
                "reach_midpoint_lon": _safe_float(meta.get("midpoint_longitude", meta.get("longitude"))),
                "reach_endpoint_1_lat": _endpoint_value(endpoint_candidates, 0, "latitude"),
                "reach_endpoint_1_lon": _endpoint_value(endpoint_candidates, 0, "longitude"),
                "reach_endpoint_2_lat": _endpoint_value(endpoint_candidates, 1, "latitude"),
                "reach_endpoint_2_lon": _endpoint_value(endpoint_candidates, 1, "longitude"),
                "reach_endpoint_candidates_json": json.dumps(
                    endpoint_candidates,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                "reach_coordinate_method": meta.get("coordinate_method", "reach_midpoint"),
                "reach_geometry_source": meta.get("geometry_source", Path(shapefile).name),
                "R_level": meta.get("r_level"),
                "Length": meta.get("reach_length_m"),
                "basin_code_l1": meta.get("basin_code_l1", ""),
                "basin_code_l2": meta.get("basin_code_l2", ""),
                "basin_code_l3": meta.get("basin_code_l3", ""),
                "basin_code_l4": meta.get("basin_code_l4", r_id or ""),
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    defaults = _derive_defaults()
    parser = argparse.ArgumentParser(
        description="Export GSED reach midpoint and endpoint geometry hints."
    )
    parser.add_argument("--csv-file", default=str(defaults["csv_file"]))
    parser.add_argument("--shapefile", default=str(defaults["shapefile"]))
    parser.add_argument("--output-csv", default=str(defaults["output_csv"]))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    hints = build_geometry_hints(
        csv_file=args.csv_file,
        shapefile=args.shapefile,
        limit=args.limit,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    hints.to_csv(output_csv, index=False, float_format="%.8f")
    print(f"Created GSED geometry hint CSV: {output_csv}")
    print(f"Rows: {len(hints)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
