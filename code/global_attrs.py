"""Shared canonical global-attribute helpers for QC NetCDF files."""

import math
import re
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from code.dataset_attr_profiles import get_dataset_profile


CONVENTIONS_VALUE = "CF-1.8, ACDD-1.3"
HISTORY_NOTE = "[fix_qc_global_attrs] canonical global attrs normalized in place"

CANONICAL_ATTR_ORDER = [
    "Conventions",
    "title",
    "history",
    "summary",
    "comment",
    "processing_level",
    "date_created",
    "date_modified",
    "featureType",
    "station_id",
    "station_name",
    "river_name",
    "station_location",
    "geographic_coverage",
    "country",
    "continent_region",
    "geospatial_lat_min",
    "geospatial_lat_max",
    "geospatial_lon_min",
    "geospatial_lon_max",
    "geospatial_vertical_min",
    "geospatial_vertical_max",
    "upstream_area",
    "temporal_resolution",
    "temporal_span",
    "time_coverage_start",
    "time_coverage_end",
    "observation_type",
    "variables_provided",
    "data_limitations",
    "source",
    "data_source_name",
    "source_data_link",
    "references",
    "creator_institution",
    "creator_name",
    "creator_email",
]

ATTR_PRIORITY_MAP = {
    "Conventions": [],
    "title": ["title"],
    "history": ["history"],
    "summary": ["summary"],
    "comment": ["comment"],
    "processing_level": ["processing_level"],
    "date_created": ["date_created"],
    "date_modified": ["date_modified"],
    "featureType": ["featureType"],
    "station_id": ["station_id", "Source_ID", "source_id", "location_id"],
    "station_name": ["station_name"],
    "river_name": ["river_name"],
    "station_location": ["station_location"],
    "geographic_coverage": ["geographic_coverage", "Geographic_Coverage"],
    "country": ["country"],
    "continent_region": ["continent_region"],
    "geospatial_lat_min": ["geospatial_lat_min"],
    "geospatial_lat_max": ["geospatial_lat_max"],
    "geospatial_lon_min": ["geospatial_lon_min"],
    "geospatial_lon_max": ["geospatial_lon_max"],
    "geospatial_vertical_min": ["geospatial_vertical_min"],
    "geospatial_vertical_max": ["geospatial_vertical_max"],
    "upstream_area": ["upstream_area"],
    "temporal_resolution": ["temporal_resolution", "Temporal_Resolution", "time_resolution", "resolution"],
    "temporal_span": ["temporal_span", "Temporal_Span", "measurement_period"],
    "time_coverage_start": ["time_coverage_start", "data_period_start"],
    "time_coverage_end": ["time_coverage_end", "data_period_end"],
    "observation_type": ["observation_type", "type", "Type"],
    "variables_provided": ["variables_provided", "Variables_Provided"],
    "data_limitations": ["data_limitations"],
    "source": ["source"],
    "data_source_name": ["data_source_name", "Data_Source_Name", "dataset_name"],
    "source_data_link": ["source_data_link", "source_url", "sediment_data_source", "discharge_data_source"],
    "creator_institution": ["creator_institution", "contributor_institution", "institution", "insitiution"],
    "creator_name": ["creator_name", "contributor_name"],
    "creator_email": ["creator_email", "contributor_email"],
}

REFERENCE_KEYS = [
    "references",
    "reference",
    "Reference",
    "Reference1",
    "reference1",
    "reference2",
]

OPTIONAL_PASSTHROUGH = {
    "station_name_chinese",
    "river_name_chinese",
    "reach_id",
    "reach_length_m",
}

TIME_VAR_NAMES = ["time", "Time", "t", "datetime", "date"]
LAT_VAR_NAMES = ["lat", "latitude", "Latitude", "LAT"]
LON_VAR_NAMES = ["lon", "longitude", "Longitude", "LON"]
ALT_VAR_NAMES = ["altitude", "Altitude", "elevation", "Elevation"]
UPSTREAM_AREA_VAR_NAMES = ["upstream_area", "drainage_area", "basin_area", "catchment_area"]
PREFERRED_DATA_VARS = ["Q", "SSC", "SSL", "altitude", "upstream_area", "sediment_yield", "reach_id", "reach_length_m"]


def _stringify_attr(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in ("none", "nan"):
        return ""
    return text


def _first_nonempty(existing, candidates):
    for key in candidates:
        value = existing.get(key, "")
        if value:
            return value
    return ""


def _merge_references(existing):
    parts = []
    for key in REFERENCE_KEYS:
        value = existing.get(key, "")
        if value and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def _safe_relative_to(path_obj, root_obj):
    try:
        return path_obj.resolve().relative_to(root_obj.resolve())
    except Exception:
        return None


def _guess_dataset_name_from_path(nc_path):
    path_obj = Path(nc_path).resolve()
    parts = path_obj.parts
    if "qc" in parts:
        idx = parts.index("qc")
        if idx >= 1:
            return parts[idx - 1]
    return ""


def _guess_path_resolution_from_path(nc_path):
    path_obj = Path(nc_path).resolve()
    parts = path_obj.parts
    if "qc" in parts:
        idx = parts.index("qc")
        if idx >= 2:
            return parts[idx - 2]
    return ""


def _read_var_values(var):
    try:
        if hasattr(var, "values"):
            return var.values
    except Exception:
        pass
    try:
        return var[:]
    except Exception:
        return None


def _get_var_attr(var, attr_name, default=None):
    try:
        if hasattr(var, "attrs"):
            attrs = getattr(var, "attrs", {})
            if attr_name in attrs:
                return attrs.get(attr_name)
    except Exception:
        pass
    try:
        return getattr(var, attr_name)
    except Exception:
        return default


def _flatten_numeric_values(var):
    try:
        data = np.ma.asarray(_read_var_values(var)).filled(np.nan)
    except Exception:
        return []
    try:
        values = np.asarray(data, dtype="float64").ravel()
    except Exception:
        return []

    fill_candidates = [
        _get_var_attr(var, "_FillValue", None),
        _get_var_attr(var, "missing_value", None),
    ]
    for fill_value in fill_candidates:
        try:
            if fill_value is None:
                continue
            values[values == float(fill_value)] = np.nan
        except Exception:
            continue
    values = values[np.isfinite(values)]
    return values.tolist()


def _extract_numeric_stats(ds, candidate_names):
    for name in candidate_names:
        if name not in ds.variables:
            continue
        values = _flatten_numeric_values(ds.variables[name])
        if values:
            return {"name": name, "min": min(values), "max": max(values), "first": values[0]}
    return {"name": "", "min": None, "max": None, "first": None}


def _normalize_datetime(value):
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        pass
    try:
        ts = pd.Timestamp(str(value))
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _extract_time_bounds(ds):
    for name in TIME_VAR_NAMES:
        if name not in ds.variables:
            continue
        var = ds[name]
        try:
            raw = np.asarray(_read_var_values(var)).ravel()
        except Exception:
            continue
        if raw.size < 1:
            continue

        timestamps = []
        for item in raw:
            ts = _normalize_datetime(item)
            if ts is not None:
                timestamps.append(ts)

        if timestamps:
            return {
                "start": min(timestamps),
                "end": max(timestamps),
                "name": name,
            }
    return {"start": None, "end": None, "name": ""}


def _format_timestamp(value):
    if value is None:
        return ""
    ts = _normalize_datetime(value)
    if ts is None:
        return _stringify_attr(value)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.isoformat()


def _build_temporal_span(start_text, end_text):
    start_text = _stringify_attr(start_text)
    end_text = _stringify_attr(end_text)
    if not start_text and not end_text:
        return ""
    if start_text and end_text and start_text != end_text:
        return "{0} to {1}".format(start_text, end_text)
    return start_text or end_text


def _derive_variables_provided(variable_names):
    selected = [name for name in PREFERRED_DATA_VARS if name in variable_names]
    return ", ".join(selected)


def _infer_source_data_link(existing, profile):
    candidate = _first_nonempty(existing, ATTR_PRIORITY_MAP["source_data_link"])
    if candidate:
        return candidate

    reference_text = _merge_references(existing)
    texts = [reference_text, existing.get("comment", ""), existing.get("summary", ""), profile.get("source_data_link", "")]
    for text in texts:
        text = _stringify_attr(text)
        if not text:
            continue
        url_match = re.search(r"https?://\S+", text)
        if url_match:
            return url_match.group(0).rstrip(").,;")
        doi_match = re.search(r"(10\.\d{4,9}/\S+)", text)
        if doi_match:
            doi = doi_match.group(1).rstrip(").,;")
            return "https://doi.org/{0}".format(doi)
    return ""


def _infer_observation_type(existing, profile):
    value = _first_nonempty(existing, ATTR_PRIORITY_MAP["observation_type"])
    if value:
        return value

    default_value = _stringify_attr(profile.get("default_observation_type", ""))
    if default_value:
        return default_value

    joined = " ".join(
        [
            existing.get("source", ""),
            existing.get("title", ""),
            existing.get("comment", ""),
            existing.get("summary", ""),
        ]
    ).lower()
    if "satellite" in joined:
        return "Satellite"
    if "model" in joined:
        return "Model"
    if "station" in joined or "in-situ" in joined or "insitu" in joined:
        return "In-situ station data"
    return ""


def _infer_geographic_coverage(existing, lat_stats, lon_stats, profile):
    value = _first_nonempty(existing, ATTR_PRIORITY_MAP["geographic_coverage"])
    if value:
        return value

    default_value = _stringify_attr(profile.get("default_geographic_coverage", ""))
    if default_value:
        return default_value

    if lat_stats["first"] is None or lon_stats["first"] is None:
        return ""
    return "Station location at ({0}, {1})".format(
        _format_coord(lat_stats["first"], positive_suffix="N", negative_suffix="S"),
        _format_coord(lon_stats["first"], positive_suffix="E", negative_suffix="W"),
    )


def _format_coord(value, positive_suffix, negative_suffix):
    try:
        numeric = float(value)
    except Exception:
        return _stringify_attr(value)
    suffix = positive_suffix if numeric >= 0 else negative_suffix
    return "{0:.4f}°{1}".format(abs(numeric), suffix)


def _infer_station_name(existing):
    value = _first_nonempty(existing, ATTR_PRIORITY_MAP["station_name"])
    if value:
        return value
    station_id = _first_nonempty(existing, ATTR_PRIORITY_MAP["station_id"])
    return station_id


def _infer_summary(existing, data_source_name, station_id, title, profile):
    value = _first_nonempty(existing, ATTR_PRIORITY_MAP["summary"])
    if value:
        return value
    default_value = _stringify_attr(profile.get("default_summary", ""))
    if default_value:
        return default_value.format(
            data_source_name=data_source_name or profile.get("data_source_name", ""),
            station_id=station_id,
            title=title,
        )
    if title:
        return title
    if data_source_name and station_id:
        return "{0} observations for station {1}.".format(data_source_name, station_id)
    return ""


def _infer_comment(existing, profile):
    value = _first_nonempty(existing, ATTR_PRIORITY_MAP["comment"])
    if value:
        return value
    return _stringify_attr(profile.get("default_comment", ""))


def read_nc_context(nc_path, dataset_name="", path_resolution=""):
    """Read stable context from an NC file without modifying it."""
    dataset_name = dataset_name or _guess_dataset_name_from_path(nc_path)
    path_resolution = path_resolution or _guess_path_resolution_from_path(nc_path)
    profile = get_dataset_profile(dataset_name)

    open_attempts = [
        {"engine": "h5netcdf"},
        {"engine": None},
    ]
    last_error = None
    ds = None
    for attempt in open_attempts:
        try:
            if attempt["engine"]:
                ds = xr.open_dataset(str(nc_path), engine=attempt["engine"])
            else:
                ds = xr.open_dataset(str(nc_path))
            break
        except Exception as exc:
            last_error = exc
            ds = None
    if ds is None:
        raise last_error

    with ds:
        existing = dict((str(key), _stringify_attr(value)) for key, value in ds.attrs.items())
        variables = list(ds.variables.keys())
        time_bounds = _extract_time_bounds(ds)
        lat_stats = _extract_numeric_stats(ds, LAT_VAR_NAMES)
        lon_stats = _extract_numeric_stats(ds, LON_VAR_NAMES)
        alt_stats = _extract_numeric_stats(ds, ALT_VAR_NAMES)
        upstream_stats = _extract_numeric_stats(ds, UPSTREAM_AREA_VAR_NAMES)

    return {
        "path": str(nc_path),
        "dataset_name": dataset_name,
        "path_resolution": path_resolution,
        "profile": profile,
        "existing": existing,
        "variables": variables,
        "time_bounds": time_bounds,
        "lat_stats": lat_stats,
        "lon_stats": lon_stats,
        "alt_stats": alt_stats,
        "upstream_stats": upstream_stats,
    }


def build_canonical_attrs(context):
    """Build canonical attributes from file context and dataset defaults."""
    existing = dict(context["existing"])
    profile = dict(context["profile"])
    variables = set(context["variables"])
    time_bounds = context["time_bounds"]
    lat_stats = context["lat_stats"]
    lon_stats = context["lon_stats"]
    alt_stats = context["alt_stats"]
    upstream_stats = context["upstream_stats"]

    attrs = {}
    attrs["Conventions"] = CONVENTIONS_VALUE

    title = _first_nonempty(existing, ATTR_PRIORITY_MAP["title"])
    if not title:
        title = "{0} station data".format(profile.get("data_source_name", "") or context["dataset_name"])
    attrs["title"] = title

    start_text = _first_nonempty(existing, ATTR_PRIORITY_MAP["time_coverage_start"]) or _format_timestamp(time_bounds["start"])
    end_text = _first_nonempty(existing, ATTR_PRIORITY_MAP["time_coverage_end"]) or _format_timestamp(time_bounds["end"])
    station_id = _first_nonempty(existing, ATTR_PRIORITY_MAP["station_id"])

    data_source_name = _first_nonempty(existing, ATTR_PRIORITY_MAP["data_source_name"])
    if not data_source_name:
        data_source_name = _stringify_attr(profile.get("data_source_name", "")) or context["dataset_name"]

    attrs["station_id"] = station_id
    attrs["station_name"] = _infer_station_name(existing)
    attrs["river_name"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["river_name"])
    attrs["station_location"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["station_location"])
    attrs["summary"] = _infer_summary(existing, data_source_name, station_id, title, profile)
    attrs["comment"] = _infer_comment(existing, profile)
    attrs["processing_level"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["processing_level"]) or _stringify_attr(
        profile.get("default_processing_level", "")
    )

    attrs["date_created"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["date_created"]) or _first_nonempty(
        existing, ATTR_PRIORITY_MAP["date_modified"]
    )
    attrs["date_modified"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["date_modified"]) or attrs["date_created"]
    attrs["featureType"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["featureType"]) or _stringify_attr(
        profile.get("default_feature_type", "")
    )

    attrs["geographic_coverage"] = _infer_geographic_coverage(existing, lat_stats, lon_stats, profile)
    attrs["country"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["country"])
    attrs["continent_region"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["continent_region"])
    attrs["geospatial_lat_min"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_lat_min"]) or _stringify_attr(
        lat_stats["min"]
    )
    attrs["geospatial_lat_max"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_lat_max"]) or _stringify_attr(
        lat_stats["max"]
    )
    attrs["geospatial_lon_min"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_lon_min"]) or _stringify_attr(
        lon_stats["min"]
    )
    attrs["geospatial_lon_max"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_lon_max"]) or _stringify_attr(
        lon_stats["max"]
    )
    attrs["geospatial_vertical_min"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_vertical_min"]) or _stringify_attr(
        alt_stats["min"]
    )
    attrs["geospatial_vertical_max"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["geospatial_vertical_max"]) or _stringify_attr(
        alt_stats["max"]
    )
    attrs["upstream_area"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["upstream_area"]) or _stringify_attr(
        upstream_stats["first"]
    )

    attrs["temporal_resolution"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["temporal_resolution"])
    attrs["time_coverage_start"] = start_text
    attrs["time_coverage_end"] = end_text
    attrs["temporal_span"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["temporal_span"]) or _build_temporal_span(
        start_text, end_text
    )

    attrs["observation_type"] = _infer_observation_type(existing, profile)
    attrs["variables_provided"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["variables_provided"]) or _derive_variables_provided(
        variables
    )
    attrs["data_limitations"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["data_limitations"])
    attrs["source"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["source"]) or _stringify_attr(profile.get("default_source", ""))
    attrs["data_source_name"] = data_source_name
    attrs["source_data_link"] = _infer_source_data_link(existing, profile)
    attrs["references"] = _merge_references(existing)
    attrs["creator_institution"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["creator_institution"]) or _stringify_attr(
        profile.get("creator_institution", "")
    )
    attrs["creator_name"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["creator_name"]) or _stringify_attr(
        profile.get("creator_name", "")
    )
    attrs["creator_email"] = _first_nonempty(existing, ATTR_PRIORITY_MAP["creator_email"]) or _stringify_attr(
        profile.get("creator_email", "")
    )

    for key in CANONICAL_ATTR_ORDER:
        attrs[key] = _stringify_attr(attrs.get(key, ""))
    return attrs


def _append_history(existing_history, history_note):
    existing_history = _stringify_attr(existing_history)
    history_note = _stringify_attr(history_note)
    if not history_note:
        return existing_history
    if history_note in existing_history:
        return existing_history
    if not existing_history:
        return history_note
    return "{0}\n{1}".format(existing_history, history_note)


def validate_canonical_attrs(attrs):
    missing = []
    for key in CANONICAL_ATTR_ORDER:
        if key in OPTIONAL_PASSTHROUGH:
            continue
        if not _stringify_attr(attrs.get(key, "")):
            missing.append(key)
    return missing


def apply_canonical_attrs(nc_path, attrs, dry_run=False, history_note=""):
    """Apply canonical attributes to an NC file and return change details."""
    changed_keys = []
    target_history = ""

    with _open_netcdf_for_attrs(str(nc_path), "r") as ds:
        existing = _read_dataset_attrs(ds)
        target_history = _append_history(existing.get("history", attrs.get("history", "")), history_note)
        attrs = dict(attrs)
        attrs["history"] = target_history
        for key in CANONICAL_ATTR_ORDER:
            if _stringify_attr(existing.get(key, "")) != _stringify_attr(attrs.get(key, "")):
                changed_keys.append(key)

    if dry_run:
        return {
            "changed": bool(changed_keys),
            "changed_keys": changed_keys,
            "missing_after_fix": validate_canonical_attrs(attrs),
        }

    with _open_netcdf_for_attrs(str(nc_path), "a") as ds:
        for key in changed_keys:
            _set_dataset_attr(ds, key, _stringify_attr(attrs.get(key, "")))
        if "history" not in changed_keys and history_note and history_note not in _stringify_attr(_read_dataset_attrs(ds).get("history", "")):
            _set_dataset_attr(ds, "history", target_history)
            if "history" not in changed_keys:
                changed_keys.append("history")

    return {
        "changed": bool(changed_keys),
        "changed_keys": changed_keys,
        "missing_after_fix": validate_canonical_attrs(attrs),
    }


def normalize_nc_attrs(nc_path, dataset_name="", path_resolution="", history_note="", dry_run=False):
    """Normalize global attrs for a single NC file."""
    context = read_nc_context(nc_path, dataset_name=dataset_name, path_resolution=path_resolution)
    attrs = build_canonical_attrs(context)
    result = apply_canonical_attrs(nc_path, attrs, dry_run=dry_run, history_note=history_note)
    result.update(
        {
            "path": str(nc_path),
            "dataset_name": context["dataset_name"],
            "path_resolution": context["path_resolution"],
        }
    )
    return result


@contextmanager
def _open_netcdf_for_attrs(nc_path, mode):
    try:
        import netCDF4 as nc4

        ds = nc4.Dataset(str(nc_path), mode)
        try:
            yield ds
        finally:
            ds.close()
        return
    except Exception:
        pass

    import h5netcdf

    ds = h5netcdf.File(str(nc_path), mode)
    try:
        yield ds
    finally:
        ds.close()


def _read_dataset_attrs(ds):
    try:
        if hasattr(ds, "ncattrs"):
            return dict((key, _stringify_attr(getattr(ds, key, ""))) for key in ds.ncattrs())
    except Exception:
        pass
    try:
        return dict((str(key), _stringify_attr(value)) for key, value in ds.attrs.items())
    except Exception:
        return {}


def _set_dataset_attr(ds, key, value):
    if hasattr(ds, "setncattr"):
        ds.setncattr(key, value)
    else:
        ds.attrs[key] = value
