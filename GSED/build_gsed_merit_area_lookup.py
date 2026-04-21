#!/usr/bin/env python3
"""
Build an external R_ID -> upstream_area_km2 lookup for GSED using MERIT Hydro.

The output CSV is designed to be consumed directly by process_gsed_cf18.py
without changing scripts_basin_test.

Matching workflow, end to end
-----------------------------
1. Read the public GSED reach ids (R_ID) from the monthly CSV.
2. Reconstruct both centroid and endpoint candidates from GSED_Reach.shp.
3. Match each endpoint candidate to nearby MERIT reaches and select the
   downstream endpoint proxy by preferring the candidate whose matched MERIT
   reach has the largest uparea.
4. Reuse that already matched MERIT reach directly when tracing the upstream
   basin, instead of asking the tracer to run a second find_best_reach() pass.
5. If all endpoint matches fail, fall back to one centroid-based reach match.
6. Run the shared basin release policy from basin_policy.py so that this lookup
   follows the exact same accept/reject semantics as the downstream basin
   workflow.

The important design choice is that this file still uses MERIT as the final
basin/topology reference. GSED supplies the reach geometry and candidate anchor
points; MERIT still supplies COMID-level upstream area and basin polygons.
"""

import argparse
import multiprocessing as mp
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_WORKER_TRACER = None
_WORKER_CLASSIFY_BASIN_RESULT = None
_WORKER_ROWS_DONE = 0
_WORKER_CLEAR_CACHE_EVERY = None


def _normalize_gsed_rid(value):
    """Normalize a raw GSED reach id to a stable integer-like string."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _derive_basin_info_from_rid(r_id):
    """Recover hierarchical basin prefixes from the public GSED R_ID code."""
    r_id_str = _normalize_gsed_rid(r_id)
    if r_id_str is None:
        return {
            "basin_code_l1": None,
            "basin_code_l2": None,
            "basin_code_l3": None,
            "basin_code_l4": None,
        }

    return {
        "basin_code_l1": r_id_str[:2] if len(r_id_str) >= 2 else r_id_str,
        "basin_code_l2": r_id_str[:5] if len(r_id_str) >= 5 else r_id_str,
        "basin_code_l3": r_id_str[:8] if len(r_id_str) >= 8 else r_id_str,
        "basin_code_l4": r_id_str,
    }


def _read_gsed_dbf_records(dbf_path):
    """Read the minimal GSED DBF attributes without requiring GIS libraries."""
    records = []
    with open(dbf_path, "rb") as handle:
        header = handle.read(32)
        if len(header) < 32:
            raise ValueError(f"Invalid DBF header: {dbf_path}")

        record_count = struct.unpack("<I", header[4:8])[0]
        record_length = struct.unpack("<H", header[10:12])[0]

        fields = []
        while True:
            # DBF field descriptors end with 0x0D. Each descriptor is 32 bytes.
            first = handle.read(1)
            if not first or first == b"\r":
                break
            descriptor = first + handle.read(31)
            name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", "ignore")
            fields.append((name, descriptor[11:12].decode("ascii", "ignore"), descriptor[16], descriptor[17]))

        for _ in range(record_count):
            record = handle.read(record_length)
            if not record:
                break

            deleted = record[:1] == b"*"
            row = {"_deleted": deleted}
            offset = 1

            for name, field_type, length, decimals in fields:
                raw = record[offset:offset + length]
                offset += length
                text = raw.decode("latin1", "ignore").strip()

                # The lookup only needs the reach id, hierarchy level, and length.
                if name not in {"R_ID", "R_level", "Length"}:
                    continue

                if not text:
                    row[name] = None
                    continue

                if field_type in {"N", "F"}:
                    try:
                        value = float(text)
                        if decimals == 0:
                            value = int(value)
                        row[name] = value
                    except ValueError:
                        row[name] = text
                else:
                    row[name] = text

            records.append(row)

    return records


def _empty_reach_info():
    """Return the failed-match structure used by basin_tracer.find_best_reach."""
    return {
        "COMID": None,
        "uparea": np.nan,
        "distance": np.nan,
        "pfaf_code": None,
        "match_quality": "failed",
        "area_error": np.nan,
    }


def _build_missing_coords_row(r_id_str):
    """Return the fixed CSV row used when a GSED reach has no usable anchor."""
    return {
        "R_ID": r_id_str,
        "upstream_area_km2": np.nan,
        "merit_comid": np.nan,
        "merit_pfaf_code": "",
        "merit_distance_m": np.nan,
        "merit_match_quality": "missing_gsed_coords",
        "merit_method": "",
        "merit_n_upstream_reaches": 0,
        "merit_point_in_local": False,
        "merit_point_in_basin": False,
        "merit_basin_status": "unresolved",
        "merit_basin_flag": "no_match",
        "merit_lookup_accept": False,
        "gsed_anchor_source": "",
        "gsed_endpoint_match_count": 0,
        "gsed_lat": np.nan,
        "gsed_lon": np.nan,
        **_derive_basin_info_from_rid(r_id_str),
    }


def _safe_float(value, default=np.nan):
    """Coerce loose numeric values to float, or return the requested default."""
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def _has_valid_coordinate_pair(lat, lon):
    """Return True when both coordinates are finite scalars."""
    return pd.notna(lat) and pd.notna(lon)


def _is_valid_reach_info(reach_info):
    """Return True when a reach-matching result contains a usable COMID."""
    if not isinstance(reach_info, dict):
        return False
    comid = reach_info.get("COMID")
    return comid is not None and not pd.isna(comid)


def _extract_polyline_representatives(record_content):
    """Extract centroid and unique part endpoints from a polyline record."""
    if len(record_content) < 44:
        return None, None, []

    shape_type = struct.unpack("<i", record_content[:4])[0]
    if shape_type == 0:
        return None, None, []
    if shape_type != 3:
        raise ValueError(f"Unsupported shapefile geometry type: {shape_type}")

    # PolyLine record layout: bbox + numParts + numPoints + parts[] + points[].
    num_parts = struct.unpack("<i", record_content[36:40])[0]
    num_points = struct.unpack("<i", record_content[40:44])[0]
    parts_offset = 44
    points_offset = parts_offset + 4 * num_parts
    points_end = points_offset + num_points * 16

    if points_end > len(record_content):
        raise ValueError("Corrupted polyline record in shapefile.")
    if num_points == 0:
        return None, None, []

    part_starts = [
        struct.unpack(
            "<i",
            record_content[parts_offset + i * 4: parts_offset + (i + 1) * 4],
        )[0]
        for i in range(num_parts)
    ]
    if not part_starts:
        part_starts = [0]
    part_starts.append(num_points)

    lons = []
    lats = []
    for i in range(num_points):
        x, y = struct.unpack(
            "<2d",
            record_content[points_offset + i * 16: points_offset + (i + 1) * 16],
        )
        lons.append(x)
        lats.append(y)

    endpoint_candidates = []
    seen = set()
    for part_start, part_end in zip(part_starts[:-1], part_starts[1:]):
        if part_end <= part_start:
            continue
        for point_idx in (part_start, part_end - 1):
            lat = float(lats[point_idx])
            lon = float(lons[point_idx])
            key = (round(lat, 12), round(lon, 12))
            if key in seen:
                continue
            seen.add(key)
            endpoint_candidates.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                }
            )

    return float(np.mean(lats)), float(np.mean(lons)), endpoint_candidates


def _extract_polyline_centroid(record_content):
    """Approximate a reach centroid by averaging all polyline vertices."""
    centroid_lat, centroid_lon, _ = _extract_polyline_representatives(record_content)
    return centroid_lat, centroid_lon


def load_gsed_reach_metadata(shapefile_path, target_rids=None):
    """Load GSED reach geometry anchors and basic metadata from the shapefile."""
    shapefile_path = Path(shapefile_path)
    dbf_records = _read_gsed_dbf_records(shapefile_path.with_suffix(".dbf"))
    target_rids = {_normalize_gsed_rid(r_id) for r_id in target_rids} if target_rids else None
    metadata = {}

    with open(shapefile_path, "rb") as handle:
        handle.read(100)  # ESRI shapefile fixed-length file header.

        for dbf_row in dbf_records:
            # In the ESRI shapefile format, the .dbf attribute table and the .shp
            # geometry records are aligned by record order. The loop below keeps
            # them in lockstep so we can recover coordinates without geopandas.
            record_header = handle.read(8)
            if not record_header:
                break

            content_length_words = struct.unpack(">i", record_header[4:8])[0]
            content = handle.read(content_length_words * 2)

            if dbf_row.get("_deleted"):
                continue

            r_id_str = _normalize_gsed_rid(dbf_row.get("R_ID"))
            if r_id_str is None:
                continue
            if target_rids and r_id_str not in target_rids:
                continue

            # The public release does not carry an explicit outlet point, so we
            # reconstruct both:
            #   - a centroid-like fallback point; and
            #   - all unique part endpoints, which are later tested against
            #     MERIT to infer the downstream anchor.
            centroid_lat, centroid_lon, endpoint_candidates = _extract_polyline_representatives(content)
            metadata[r_id_str] = {
                "r_id_str": r_id_str,
                "r_level": int(dbf_row["R_level"]) if dbf_row.get("R_level") is not None else None,
                "reach_length_m": float(dbf_row["Length"]) if dbf_row.get("Length") is not None else None,
                "latitude": centroid_lat,
                "longitude": centroid_lon,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "endpoint_candidates": endpoint_candidates,
                **_derive_basin_info_from_rid(r_id_str),
            }

    return metadata


def _resolve_gsed_anchor(tracer, meta):
    """Choose the final GSED anchor and matched MERIT reach for one reach."""
    endpoint_matches = []
    endpoint_candidates = meta.get("endpoint_candidates") or []

    for candidate_index, endpoint in enumerate(endpoint_candidates):
        lat = endpoint.get("latitude")
        lon = endpoint.get("longitude")
        if not _has_valid_coordinate_pair(lat, lon):
            continue

        reach_info = tracer.find_best_reach(float(lon), float(lat), reported_area=None)
        if not _is_valid_reach_info(reach_info):
            continue

        endpoint_matches.append(
            {
                "candidate_index": candidate_index,
                "latitude": float(lat),
                "longitude": float(lon),
                "reach_info": reach_info,
            }
        )

    if endpoint_matches:
        endpoint_matches.sort(
            key=lambda item: (
                -_safe_float(item["reach_info"].get("uparea"), default=-np.inf),
                _safe_float(item["reach_info"].get("distance"), default=np.inf),
                item["candidate_index"],
            )
        )
        best_endpoint = endpoint_matches[0]
        return {
            "anchor_source": "downstream_endpoint",
            "endpoint_match_count": len(endpoint_matches),
            "latitude": best_endpoint["latitude"],
            "longitude": best_endpoint["longitude"],
            "reach_info": best_endpoint["reach_info"],
        }

    centroid_lat = meta.get("centroid_latitude", meta.get("latitude"))
    centroid_lon = meta.get("centroid_longitude", meta.get("longitude"))
    reach_info = _empty_reach_info()
    if _has_valid_coordinate_pair(centroid_lat, centroid_lon):
        reach_info = tracer.find_best_reach(float(centroid_lon), float(centroid_lat), reported_area=None)

    return {
        "anchor_source": "centroid_fallback",
        "endpoint_match_count": 0,
        "latitude": centroid_lat,
        "longitude": centroid_lon,
        "reach_info": reach_info,
    }


def _point_is_covered_by_geometries(geometries, point):
    """Return True when any geometry in the series covers the point."""
    if point is None or geometries is None or len(geometries) == 0:
        return False

    try:
        covered = geometries.covers(point)
        if hasattr(covered, "any"):
            return bool(covered.any())
        return bool(covered)
    except Exception:
        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            try:
                if geom.covers(point):
                    return True
            except Exception:
                continue
        return False


def _build_upstream_catchment_lookup_result(tracer, lon, lat, reach_info):
    """Build the tracer result needed by this CSV without merging polygons."""
    result = {
        "geometry": None,
        "geometry_local": None,
        "basin_area": np.nan,
        "basin_id": None,
        "match_quality": "failed",
        "area_error": np.nan,
        "uparea_merit": np.nan,
        "pfaf_code": None,
        "distance": np.nan,
        "method": None,
        "n_upstream_reaches": 0,
        "point_in_local": False,
        "point_in_basin": False,
    }

    if not isinstance(reach_info, dict):
        return result

    basin_id = reach_info.get("COMID")
    pfaf_code = reach_info.get("pfaf_code")
    if basin_id is None or not pfaf_code:
        return result

    try:
        basin_id = int(basin_id)
    except (TypeError, ValueError):
        return result

    def _coerce_float_or_nan(value):
        try:
            number = float(value)
            return number if np.isfinite(number) else np.nan
        except Exception:
            return np.nan

    uparea = _coerce_float_or_nan(reach_info.get("uparea"))
    distance = _coerce_float_or_nan(reach_info.get("distance"))
    area_error = _coerce_float_or_nan(reach_info.get("area_error"))

    result["basin_id"] = basin_id
    result["match_quality"] = str(reach_info.get("match_quality", "failed"))
    result["pfaf_code"] = str(pfaf_code)
    result["uparea_merit"] = uparea
    result["area_error"] = area_error
    result["basin_area"] = uparea
    result["distance"] = distance

    point = None
    if pd.notna(lon) and pd.notna(lat):
        from shapely.geometry import Point

        point = Point(float(lon), float(lat))

    local_region = str(basin_id)[:2]
    local_cat_gdf = tracer._load_level2_catchments(local_region)
    if local_cat_gdf is not None and basin_id in local_cat_gdf.index:
        local_geometries = local_cat_gdf.loc[[basin_id], "geometry"]
        result["geometry_local"] = True
        result["point_in_local"] = _point_is_covered_by_geometries(local_geometries, point)

    upstream_comids = tracer.trace_upstream_reaches(
        basin_id,
        str(pfaf_code),
    )
    result["n_upstream_reaches"] = len(upstream_comids)

    if upstream_comids:
        comids_by_region = {}
        for comid in upstream_comids:
            region = str(comid)[:2]
            comids_by_region.setdefault(region, []).append(comid)

        has_upstream_geometry = False
        point_in_basin = False
        for region, comids in comids_by_region.items():
            cat_gdf = tracer._load_level2_catchments(region)
            if cat_gdf is None:
                continue

            valid_mask = cat_gdf.index.isin(comids)
            if not valid_mask.any():
                continue

            geometries = cat_gdf.loc[valid_mask, "geometry"]
            if len(geometries) == 0:
                continue

            has_upstream_geometry = True
            if point is not None and not point_in_basin:
                point_in_basin = _point_is_covered_by_geometries(geometries, point)

        if has_upstream_geometry:
            result["geometry"] = True
            result["method"] = "upstream_traced"
            result["point_in_basin"] = point_in_basin
        else:
            result["method"] = "area_buffer_fallback"

    return result


def _resolve_defaults():
    """Resolve the default source, output, and MERIT tracer locations."""
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "Source"
    gsed_source_dir = source_root / "GSED" / "GSED"
    merit_dir = source_root.parent.parent / "MERIT_Hydro_v07_Basins_v01_bugfix1"
    basin_tracer_dir = project_root / "Output_r" / "scripts_basin_test"
    return {
        "csv_file": gsed_source_dir / "GSED_Reach_Monthly_SSC.csv",
        "shapefile": gsed_source_dir / "GSED_Reach.shp",
        "output_csv": gsed_source_dir / "GSED_Reach_upstream_area.csv",
        "merit_dir": merit_dir,
        "basin_tracer_dir": basin_tracer_dir,
    }


def _load_basin_tracer(basin_tracer_dir):
    """Import the downstream MERIT tracer only when the CLI actually runs."""
    basin_tracer_dir = Path(basin_tracer_dir)
    if str(basin_tracer_dir) not in sys.path:
        sys.path.insert(0, str(basin_tracer_dir))

    try:
        from basin_tracer import UpstreamBasinTracer
        from basin_policy import classify_basin_result
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import basin_tracer dependencies. Please run this script "
            "in the same environment used by scripts_basin_test "
            "(needs geopandas, pyogrio, shapely, pyproj)."
        ) from exc

    return UpstreamBasinTracer, classify_basin_result


def _build_lookup_row(r_id_str, meta, tracer, classify_basin_result):
    """Build one output row for a single GSED reach id."""
    meta = meta or _derive_basin_info_from_rid(r_id_str)
    centroid_lat = meta.get("centroid_latitude", meta.get("latitude"))
    centroid_lon = meta.get("centroid_longitude", meta.get("longitude"))
    endpoint_candidates = meta.get("endpoint_candidates") or []
    has_valid_endpoint = any(
        _has_valid_coordinate_pair(candidate.get("latitude"), candidate.get("longitude"))
        for candidate in endpoint_candidates
    )
    has_valid_centroid = _has_valid_coordinate_pair(centroid_lat, centroid_lon)

    if not has_valid_endpoint and not has_valid_centroid:
        return _build_missing_coords_row(r_id_str)

    # Stage 1: resolve the final GSED anchor. Endpoints are tested first;
    # if none yields a MERIT reach match, we fall back to the centroid.
    anchor_result = _resolve_gsed_anchor(tracer, meta)
    anchor_lat = anchor_result.get("latitude")
    anchor_lon = anchor_result.get("longitude")
    reach_info = anchor_result.get("reach_info", _empty_reach_info())

    # Stage 2: trace the basin directly from the already matched reach,
    # skipping the second find_best_reach() call inside get_upstream_basin().
    # This lookup only needs CSV diagnostics, so we avoid the heavy polygon
    # union step and compute coverage flags directly from catchment members.
    basin_result = _build_upstream_catchment_lookup_result(
        tracer,
        float(anchor_lon) if pd.notna(anchor_lon) else np.nan,
        float(anchor_lat) if pd.notna(anchor_lat) else np.nan,
        reach_info,
    )
    # basin_policy converts the raw tracer diagnostics into release-facing
    # labels. For GSED this still matters because the product represents a
    # reach footprint rather than a bank gauge, so local-catchment coverage
    # remains an important acceptance signal.
    basin_status, basin_flag = classify_basin_result(
        basin_id=basin_result["basin_id"],
        match_quality=basin_result["match_quality"],
        distance_m=basin_result.get("distance", np.nan),
        source_name="GSED",
        point_in_local=basin_result.get("point_in_local", False),
        point_in_basin=basin_result.get("point_in_basin", False),
    )
    merit_method = basin_result.get("method")
    # Only keep basin areas that are both "resolved" under the shared
    # policy and traced from real upstream polygons rather than fallback
    # buffers. In other words:
    #   - the matched MERIT reach passed the shared acceptance rules; and
    #   - the basin geometry came from explicit upstream tracing instead of
    #     a circular area buffer used only as a last-resort placeholder.
    # process_gsed_cf18.py can later filter on this flag directly.
    merit_accept = (
        basin_status == "resolved"
        and merit_method == "upstream_traced"
    )
    return {
        "R_ID": r_id_str,
        # This is the MERIT uparea attached to the matched COMID, i.e.
        # the drainage area implied by the selected MERIT reach.
        "upstream_area_km2": basin_result["uparea_merit"],
        "merit_comid": basin_result["basin_id"] if basin_result["basin_id"] is not None else np.nan,
        "merit_pfaf_code": basin_result["pfaf_code"] if basin_result["pfaf_code"] is not None else "",
        "merit_distance_m": basin_result["distance"],
        "merit_match_quality": basin_result["match_quality"],
        "merit_method": merit_method if merit_method is not None else "",
        "merit_n_upstream_reaches": basin_result.get("n_upstream_reaches", 0),
        "merit_point_in_local": bool(basin_result.get("point_in_local", False)),
        "merit_point_in_basin": bool(basin_result.get("point_in_basin", False)),
        "merit_basin_status": basin_status,
        "merit_basin_flag": basin_flag,
        "merit_lookup_accept": merit_accept,
        "gsed_anchor_source": anchor_result.get("anchor_source", ""),
        "gsed_endpoint_match_count": int(anchor_result.get("endpoint_match_count", 0)),
        "gsed_lat": _safe_float(anchor_lat),
        "gsed_lon": _safe_float(anchor_lon),
        **_derive_basin_info_from_rid(r_id_str),
    }


def _clear_worker_cache_if_needed(force=False):
    """Bound worker memory growth by periodically releasing tracer caches."""
    global _WORKER_ROWS_DONE

    if _WORKER_TRACER is None:
        return
    if force:
        _WORKER_TRACER.clear_cache()
        return
    if _WORKER_CLEAR_CACHE_EVERY and _WORKER_ROWS_DONE % _WORKER_CLEAR_CACHE_EVERY == 0:
        _WORKER_TRACER.clear_cache()


def _init_lookup_worker(merit_dir, basin_tracer_dir, clear_cache_every=None):
    """Initialize one worker-local MERIT tracer for process-based parallelism."""
    global _WORKER_TRACER, _WORKER_CLASSIFY_BASIN_RESULT
    global _WORKER_ROWS_DONE, _WORKER_CLEAR_CACHE_EVERY

    UpstreamBasinTracer, classify_basin_result = _load_basin_tracer(basin_tracer_dir)
    _WORKER_TRACER = UpstreamBasinTracer(str(merit_dir))
    _WORKER_CLASSIFY_BASIN_RESULT = classify_basin_result
    _WORKER_ROWS_DONE = 0
    _WORKER_CLEAR_CACHE_EVERY = (
        max(1, int(clear_cache_every))
        if clear_cache_every is not None
        else None
    )


def _build_lookup_row_worker(task):
    """Process-pool wrapper that returns one row while preserving input order."""
    global _WORKER_ROWS_DONE

    row_index = task["row_index"]
    r_id_str = task["r_id_str"]
    meta = task["meta"]

    try:
        row = _build_lookup_row(
            r_id_str,
            meta,
            _WORKER_TRACER,
            _WORKER_CLASSIFY_BASIN_RESULT,
        )
    finally:
        _WORKER_ROWS_DONE += 1
        _clear_worker_cache_if_needed()

    return row_index, row


def build_lookup(
    csv_file,
    shapefile,
    merit_dir,
    basin_tracer_dir,
    limit=None,
    workers=1,
    chunksize=None,
    clear_cache_every=None,
    max_tasks_per_child=None,
):
    """Build the GSED R_ID -> MERIT upstream-area lookup table."""
    df = pd.read_csv(csv_file, usecols=["R_ID"])
    if limit is not None:
        df = df.head(limit)

    # target_rids preserves the original CSV row order, because downstream code
    # expects a simple lookup keyed by the public GSED R_ID values.
    target_rids = [_normalize_gsed_rid(r_id) for r_id in df["R_ID"].tolist()]
    # Metadata loading is deduplicated internally through a set, so repeated
    # R_ID values in the monthly CSV do not trigger repeated shapefile parsing.
    reach_metadata = load_gsed_reach_metadata(shapefile, target_rids=target_rids)

    tasks = [
        {
            "row_index": idx,
            "r_id_str": r_id_str,
            "meta": reach_metadata.get(r_id_str),
        }
        for idx, r_id_str in enumerate(target_rids)
    ]

    if workers is None:
        workers = 1
    workers = max(1, int(workers))
    if chunksize is None:
        chunksize = max(1, len(tasks) // max(1, workers * 8))
    else:
        chunksize = max(1, int(chunksize))
    if workers > 1 and clear_cache_every is None:
        clear_cache_every = 8
    elif clear_cache_every is not None:
        clear_cache_every = max(1, int(clear_cache_every))
    if workers > 1 and max_tasks_per_child is None:
        # Pool.imap submits chunked work items, so "1" here means one chunk
        # per child before recycling the process and all cached GeoDataFrames.
        max_tasks_per_child = 1
    elif max_tasks_per_child is not None:
        max_tasks_per_child = max(1, int(max_tasks_per_child))

    rows = [None] * len(tasks)

    if workers == 1 or len(tasks) <= 1:
        UpstreamBasinTracer, classify_basin_result = _load_basin_tracer(basin_tracer_dir)
        tracer = UpstreamBasinTracer(str(merit_dir))
        for idx, task in enumerate(tasks, start=1):
            rows[task["row_index"]] = _build_lookup_row(
                task["r_id_str"],
                task["meta"],
                tracer,
                classify_basin_result,
            )
            if idx % 200 == 0:
                print(f"Matching GSED reaches: {idx}/{len(tasks)}")
    else:
        print(
            "Matching GSED reaches with "
            f"{workers} workers (chunksize={chunksize}, "
            f"clear_cache_every={clear_cache_every}, "
            f"max_tasks_per_child={max_tasks_per_child})"
        )
        mp_context = mp.get_context("spawn")
        with mp_context.Pool(
            processes=workers,
            initializer=_init_lookup_worker,
            initargs=(str(merit_dir), str(basin_tracer_dir), clear_cache_every),
            maxtasksperchild=max_tasks_per_child,
        ) as pool:
            for completed, (row_index, row) in enumerate(
                pool.imap(_build_lookup_row_worker, tasks, chunksize=chunksize),
                start=1,
            ):
                rows[row_index] = row
                if completed % 200 == 0:
                    print(f"Matching GSED reaches: {completed}/{len(tasks)}")

    return pd.DataFrame(rows)


def parse_args():
    """Parse CLI arguments with project-local defaults."""
    defaults = _resolve_defaults()
    parser = argparse.ArgumentParser(
        description="Build a GSED R_ID -> MERIT upstream area lookup table."
    )
    parser.add_argument("--csv-file", default=str(defaults["csv_file"]))
    parser.add_argument("--shapefile", default=str(defaults["shapefile"]))
    parser.add_argument("--merit-dir", default=os.environ.get("MERIT_DIR", str(defaults["merit_dir"])))
    parser.add_argument("--basin-tracer-dir", default=str(defaults["basin_tracer_dir"]))
    parser.add_argument("--output-csv", default=str(defaults["output_csv"]))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of worker processes for MERIT matching/tracing. Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Process-pool chunksize. Defaults to an automatic value based on row count and worker count.",
    )
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=None,
        help="Clear one worker's MERIT cache every N rows. Defaults to 8 when workers > 1.",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=None,
        help="Recycle one worker after this many chunked tasks. Defaults to 1 when workers > 1.",
    )
    return parser.parse_args()


def main():
    """CLI entrypoint: build the lookup table and write it to disk."""
    args = parse_args()

    print(f"Reading GSED IDs from: {args.csv_file}")
    print(f"Reading GSED reach geometry from: {args.shapefile}")
    print(f"Using MERIT Hydro directory: {args.merit_dir}")
    print(f"Using basin tracer from: {args.basin_tracer_dir}")
    print(f"Using worker processes: {args.workers}")

    lookup_df = build_lookup(
        csv_file=args.csv_file,
        shapefile=args.shapefile,
        merit_dir=args.merit_dir,
        basin_tracer_dir=args.basin_tracer_dir,
        limit=args.limit,
        workers=args.workers,
        chunksize=args.chunksize,
        clear_cache_every=args.clear_cache_every,
        max_tasks_per_child=args.max_tasks_per_child,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_csv(output_csv, index=False, float_format="%.6f")

    matched = lookup_df["upstream_area_km2"].notna().sum()
    print(f"Created lookup CSV: {output_csv}")
    print(f"Matched areas: {matched}/{len(lookup_df)}")
    print("Match quality summary:")
    print(lookup_df["merit_match_quality"].fillna("missing").value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
