#!/usr/bin/env python3
"""
Quality Control and CF-1.8/ACDD-1.3 Standardization for Yellow River Sediment Data.

This script:
1. Reads annual NetCDF files from source/HuangHe/netcdf
2. Performs quality control checks and adds quality flags
3. Writes TWO standardized outputs for each station:
   - annual: keeps the original yearly sequence
   - climatology: collapses annual values into one climatological annual mean
4. Generates two station summary CSVs
5. Saves outputs to:
   - Output_r/annual/Huanghe
   - Output_r/climatology/Huanghe
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import argparse
import os
import glob
import sys
from pathlib import Path
import zipfile
import re
from xml.etree import ElementTree as ET

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from code.qc import apply_quality_flag, compute_log_iqr_bounds
from code.runtime import ensure_directory, resolve_output_root, resolve_source_root
from code.validation import require_existing_directory


OUTPUT_TIME_UNITS = "days since 1970-01-01 00:00:00"
OUTPUT_TIME_CALENDAR = "gregorian"
SOURCE_EXCEL_FILENAME = "黄河流域泥沙观测数据.xlsx"


def normalize_station_name(name):
    """Normalize station name for lookup."""
    if name is None:
        return ""
    return str(name).strip()


def normalize_period_text(text):
    """Normalize full-width brackets in period text."""
    if text is None:
        return ""
    s = str(text).strip()
    return s.replace("（", "(").replace("）", ")")


def canonical_time_span(period_text, fallback):
    """Convert text like '(1950-2015)' to '1950-2015'."""
    p = normalize_period_text(period_text)
    m = re.search(r"(\d{4}\s*-\s*\d{4})", p)
    if m:
        return m.group(1).replace(" ", "")
    return fallback


def years_from_time_span(time_span, fallback_start, fallback_end):
    """Parse start/end years from time span like '1950-2015'."""
    span = canonical_time_span(time_span, fallback=f"{fallback_start}-{fallback_end}")
    m = re.match(r"^(\d{4})-(\d{4})$", span)
    if m:
        return int(m.group(1)), int(m.group(2))
    return int(fallback_start), int(fallback_end)


def try_parse_float(value):
    """Try converting a value to float, return np.nan on failure."""
    try:
        return float(str(value).strip())
    except Exception:
        return np.nan


def excel_col_to_num(col_letters):
    """Convert Excel column letters to 1-based column number."""
    n = 0
    for ch in col_letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def parse_sheet_cells(xlsx_zip, sheet_xml_path, shared_strings):
    """Parse one worksheet into {row_number: {col_number: cell_value}}."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(xlsx_zip.read(sheet_xml_path))
    rows = {}
    for row in root.iter(f"{ns}row"):
        rid = int(row.attrib["r"])
        row_map = {}
        for c in row.findall(f"{ns}c"):
            cell_ref = c.attrib.get("r", "")
            m = re.match(r"([A-Z]+)(\d+)", cell_ref)
            if m is None:
                continue
            col = excel_col_to_num(m.group(1))
            cell_type = c.attrib.get("t")
            value_node = c.find(f"{ns}v")
            if cell_type == "inlineStr":
                inline_node = c.find(f"{ns}is/{ns}t")
                if inline_node is not None and inline_node.text is not None:
                    row_map[col] = inline_node.text
                continue
            if value_node is None:
                continue
            raw = value_node.text
            if cell_type == "s":
                row_map[col] = shared_strings[int(raw)]
            else:
                row_map[col] = raw
        if row_map:
            rows[rid] = row_map
    return rows


def extract_sheet_climatology(rows):
    """Extract station long-term mean SSC from one parsed sheet."""
    station_row = None
    climatology_row = None
    for rid, row in rows.items():
        first_col = normalize_station_name(row.get(1, ""))
        if first_col == "水文控制站":
            station_row = rid
        if first_col.startswith("年均含沙量"):
            climatology_row = rid
    if station_row is None or climatology_row is None:
        return {}

    period_row = climatology_row + 1
    out = {}
    station_cells = rows.get(station_row, {})
    climatology_cells = rows.get(climatology_row, {})
    period_cells = rows.get(period_row, {})
    for col, station_name in station_cells.items():
        if col < 3:
            continue
        station_name = normalize_station_name(station_name)
        if not station_name:
            continue
        clim_val_kg = try_parse_float(climatology_cells.get(col, np.nan))
        if not np.isfinite(clim_val_kg):
            continue
        period = normalize_period_text(period_cells.get(col, ""))
        out[station_name] = {
            "ssc_kg_m3": float(clim_val_kg),
            "period": period,
        }
    return out


def load_climatology_lookup_from_excel(excel_file):
    """Load long-term mean SSC lookup from the source workbook."""
    excel_file = Path(excel_file)
    if not excel_file.exists():
        print(f"WARNING: Climatology source workbook not found: {excel_file}")
        return {}

    lookup = {}
    with zipfile.ZipFile(str(excel_file), "r") as xlsx_zip:
        shared_strings_xml = ET.fromstring(xlsx_zip.read("xl/sharedStrings.xml"))
        shared_strings = []
        for si in shared_strings_xml:
            text = "".join(
                (t.text or "") for t in si.iter() if t.tag.endswith("}t")
            )
            shared_strings.append(text)

        for sheet_path in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
            if sheet_path not in xlsx_zip.namelist():
                continue
            rows = parse_sheet_cells(
                xlsx_zip=xlsx_zip,
                sheet_xml_path=sheet_path,
                shared_strings=shared_strings,
            )
            lookup.update(extract_sheet_climatology(rows))

    # Name alias for one known station variant in the dataset.
    if "状头" in lookup and "狱头" not in lookup:
        lookup["狱头"] = dict(lookup["状头"])
    if "狱头" in lookup and "状头" not in lookup:
        lookup["状头"] = dict(lookup["狱头"])

    print(f"Loaded long-term climatology values for {len(lookup)} stations from Excel.")
    return lookup


def resolve_climatology_from_lookup(climatology_lookup, station_name, station_name_chinese):
    """Return climatology value from source Excel lookup if available."""
    keys = [
        normalize_station_name(station_name_chinese),
        normalize_station_name(station_name),
    ]
    for key in keys:
        if key and key in climatology_lookup:
            raw = climatology_lookup[key]
            ssc_kg_m3 = try_parse_float(raw.get("ssc_kg_m3", np.nan))
            if np.isfinite(ssc_kg_m3):
                period = normalize_period_text(raw.get("period", ""))
                return float(ssc_kg_m3 * 1000.0), period, key
    return None, "", ""


def read_scalar_variable(var):
    """Read a scalar NetCDF variable safely as float."""
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        if np.all(arr.mask):
            return np.nan
        arr = arr.filled(np.nan)
    arr = np.asarray(arr).squeeze()
    if arr.size == 0:
        return np.nan
    return float(arr)


def read_array_variable(var):
    """Read a NetCDF variable safely as a NumPy array."""
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr)


def write_standardized_file(
    output_file,
    time_vals_out,
    ssc_vals_out,
    ssc_flag_out,
    lon,
    lat,
    alt,
    upstream_area,
    station_id,
    station_name,
    station_name_chinese,
    river_name,
    river_name_chinese,
    original_time_range,
    ssc_start_date,
    ssc_end_date,
    temporal_resolution,
    observation_type,
    title,
    summary,
    ssc_comment,
    global_comment,
):
    """Write one standardized NetCDF file."""

    with nc.Dataset(output_file, "w", format="NETCDF4") as ds:
        # Dimensions
        ds.createDimension("time", None)

        # Coordinate variables
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = OUTPUT_TIME_UNITS
        time_var.calendar = OUTPUT_TIME_CALENDAR
        time_var.axis = "T"
        time_var[:] = np.asarray(time_vals_out, dtype=np.float64)

        lat_var = ds.createVariable("lat", "f4")
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = np.float32(lat)

        lon_var = ds.createVariable("lon", "f4")
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = np.float32(lon)

        alt_var = ds.createVariable("altitude", "f4", fill_value=-9999.0)
        alt_var.long_name = "station elevation above sea level"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.positive = "up"
        alt_var.comment = "Source: Not available in original dataset."
        if np.isnan(alt):
            alt_var[:] = -9999.0
        else:
            alt_var[:] = np.float32(alt)

        area_var = ds.createVariable("upstream_area", "f4", fill_value=-9999.0)
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.comment = (
            "Source: Original data provided by Yellow River Sediment Bulletin (2015-2019)."
        )
        if np.isnan(upstream_area):
            area_var[:] = -9999.0
        else:
            area_var[:] = np.float32(upstream_area)

        # Data variable
        ssc_var = ds.createVariable(
            "SSC",
            "f4",
            ("time",),
            fill_value=-9999.0,
            zlib=True,
            complevel=4,
        )
        ssc_var.long_name = "suspended sediment concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "time lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag"
        ssc_var.comment = ssc_comment
        ssc_var[:] = np.asarray(ssc_vals_out, dtype=np.float32)

        # QC variable
        ssc_flag_var = ds.createVariable(
            "SSC_flag",
            "b",
            ("time",),
            fill_value=9,
            zlib=True,
            complevel=4,
        )
        ssc_flag_var.long_name = "quality flag for suspended sediment concentration"
        ssc_flag_var.standard_name = "status_flag"
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = (
            "good_data estimated_data suspect_data bad_data missing_data"
        )
        ssc_flag_var.comment = (
            "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), "
            "3=Bad (e.g., negative), 9=Missing in source."
        )
        ssc_flag_var[:] = np.asarray(ssc_flag_out, dtype=np.int8)

        # Global attributes
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = title
        ds.summary = summary

        ds.source = "in-situ station data"
        ds.data_source_name = "Yellow River Sediment Bulletin Dataset"
        ds.station_name = station_name
        ds.station_name_chinese = station_name_chinese
        ds.river_name = river_name
        ds.river_name_chinese = river_name_chinese
        ds.Source_ID = station_id

        ds.Type = "in-situ"
        ds.featureType = "timeSeries"
        ds.observation_type = observation_type

        ds.Variables_Provided = "SSC"
        ds.Number_of_data = str(len(time_vals_out))

        ds.Reference = (
            "Zhang Yaonan, Kang jianfang, Liu chun. (2021). Data on Sediment Observation in the "
            "Yellow River Basin from 2015 to 2019. National Cryosphere Desert Data Center. "
            "https://doi.org/10.12072/ncdc.YRiver.db0054.2021"
        )
        ds.source_data_link = "https://doi.org/10.12072/ncdc.YRiver.db0054.2021"

        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"

        ds.time_coverage_start = f"{ssc_start_date}-01-01T00:00:00"
        ds.time_coverage_end = f"{ssc_end_date}-12-31T23:59:59"
        ds.temporal_span = original_time_range
        ds.temporal_resolution = temporal_resolution

        ds.geospatial_lat_min = float(lat)
        ds.geospatial_lat_max = float(lat)
        ds.geospatial_lon_min = float(lon)
        ds.geospatial_lon_max = float(lon)
        if not np.isnan(alt) and alt != -9999.0:
            ds.geospatial_vertical_min = float(alt)
            ds.geospatial_vertical_max = float(alt)

        ds.geographic_coverage = "Yellow River Basin, China"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.history = (
            f"{current_time}: Quality controlled and standardized to CF-1.8/ACDD-1.3. "
            f"Added quality flags and generated {temporal_resolution} product. "
            f"Script: qc_and_standardize.py"
        )

        ds.date_created = datetime.now().strftime("%Y-%m-%d")
        ds.date_modified = datetime.now().strftime("%Y-%m-%d")
        ds.processing_level = "Quality controlled and standardized"

        ds.comment = global_comment
        ds.data_limitations = (
            "Only annual average SSC available; no discharge (Q) or sediment load (SSL) data in original dataset."
        )


def build_station_info(
    station_name,
    station_id,
    river_name,
    lon,
    lat,
    alt,
    upstream_area,
    original_time_range,
    temporal_resolution,
    ssc_start_date,
    ssc_end_date,
    ssc_percent_complete,
    qc_counts,
):
    """Build one station summary dictionary."""
    return {
        "station_name": station_name,
        "Source_ID": station_id,
        "river_name": river_name,
        "longitude": lon,
        "latitude": lat,
        "altitude": alt if not np.isnan(alt) else "N/A",
        "upstream_area": upstream_area if not np.isnan(upstream_area) else "N/A",
        "Data Source Name": "Yellow River Sediment Bulletin Dataset",
        "Type": "in-situ",
        "Temporal Resolution": temporal_resolution,
        "Temporal Span": original_time_range,
        "Variables Provided": "SSC",
        "Geographic Coverage": "Yellow River Basin, China",
        "Reference/DOI": "https://doi.org/10.12072/ncdc.YRiver.db0054.2021",
        "Q_start_date": "N/A",
        "Q_end_date": "N/A",
        "Q_percent_complete": "N/A",
        "SSC_start_date": ssc_start_date,
        "SSC_end_date": ssc_end_date,
        "SSC_percent_complete": ssc_percent_complete,
        "SSL_start_date": "N/A",
        "SSL_end_date": "N/A",
        "SSL_percent_complete": "N/A",
        "SSC_n_total": qc_counts["total"],
        "SSC_n_good": qc_counts["good"],
        "SSC_n_suspect": qc_counts["suspect"],
        "SSC_n_bad": qc_counts["bad"],
        "SSC_n_missing": qc_counts["missing"],
    }


def standardize_netcdf_file(input_file, output_dir_ann, output_dir_clim, climatology_lookup):
    """
    Standardize a single annual NetCDF file and generate:
    1) annual output
    2) climatology output
    """

    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read input file
    with nc.Dataset(input_file, "r") as ds_in:
        lon = read_scalar_variable(ds_in.variables["longitude"])
        lat = read_scalar_variable(ds_in.variables["latitude"])
        alt = read_scalar_variable(ds_in.variables["altitude"])
        upstream_area = read_scalar_variable(ds_in.variables["upstream_area"])

        time_vals_in = read_array_variable(ds_in.variables["time"]).astype(np.float64)
        ssc_vals = read_array_variable(ds_in.variables["ssc"]).astype(np.float64)

        station_id = str(getattr(ds_in, "station_id"))
        station_name = str(getattr(ds_in, "station_name"))
        station_name_chinese = str(getattr(ds_in, "station_name_chinese"))
        river_name = str(getattr(ds_in, "river_name"))
        river_name_chinese = str(getattr(ds_in, "river_name_chinese"))
        original_time_range = getattr(ds_in, "original_time_range", "2015-2019")

        time_units_in = ds_in.variables["time"].units
        time_calendar_in = getattr(ds_in.variables["time"], "calendar", "gregorian")

    # QC checks
    SSC_flag = np.array([apply_quality_flag(v, "SSC") for v in ssc_vals], dtype=np.int8)

    valid_ssc = ssc_vals[(SSC_flag == 0) & np.isfinite(ssc_vals) & (ssc_vals > 0)]

    if len(valid_ssc) < 5:
        print(
            f"  ℹ️  Station {station_id}: valid SSC samples = {len(valid_ssc)} < 5, log-IQR QC skipped."
        )
    else:
        lower, upper = compute_log_iqr_bounds(valid_ssc)
        if lower is not None:
            for i, v in enumerate(ssc_vals):
                if (
                    SSC_flag[i] == 0
                    and np.isfinite(v)
                    and v > 0
                    and (v < lower or v > upper)
                ):
                    SSC_flag[i] = np.int8(2)

    # QC summary for annual sequence
    n_total = len(SSC_flag)
    n_good = int(np.sum(SSC_flag == 0))
    n_suspect = int(np.sum(SSC_flag == 2))
    n_bad = int(np.sum(SSC_flag == 3))
    n_missing = int(np.sum(SSC_flag == 9))
    ssc_percent_complete_ann = 100.0 * n_good / n_total if n_total > 0 else 0.0

    qc_counts_ann = {
        "total": int(n_total),
        "good": n_good,
        "suspect": n_suspect,
        "bad": n_bad,
        "missing": n_missing,
    }

    print(
        f"  QC summary:\n"
        f"    total samples : {qc_counts_ann['total']}\n"
        f"    good (flag=0) : {qc_counts_ann['good']}\n"
        f"    suspect (2)   : {qc_counts_ann['suspect']}\n"
        f"    bad (3)       : {qc_counts_ann['bad']}\n"
        f"    missing (9)   : {qc_counts_ann['missing']}"
    )

    # Decode input dates
    dates = nc.num2date(time_vals_in, units=time_units_in, calendar=time_calendar_in)
    if len(dates) == 0:
        raise ValueError(f"No valid time values found in {input_file}")

    ssc_start_date = int(dates[0].year)
    ssc_end_date = int(dates[-1].year)

    # Annual output: keep original yearly sequence, but normalize to standard output units/calendar
    ann_time_vals = np.array(
        nc.date2num(dates, units=OUTPUT_TIME_UNITS, calendar=OUTPUT_TIME_CALENDAR),
        dtype=np.float64,
    )
    ann_ssc_vals = np.array(ssc_vals, dtype=np.float32)
    ann_flag_vals = np.array(SSC_flag, dtype=np.int8)

    # Climatology output: prefer long-term mean from source workbook.
    clim_ssc_source = "fallback_annual_mean"
    clim_source_period = original_time_range
    clim_temporal_span = original_time_range
    clim_ssc_val_from_lookup, clim_source_period_lookup, matched_station = (
        resolve_climatology_from_lookup(
            climatology_lookup=climatology_lookup,
            station_name=station_name,
            station_name_chinese=station_name_chinese,
        )
    )
    if clim_ssc_val_from_lookup is not None:
        clim_ssc_vals = np.array([clim_ssc_val_from_lookup], dtype=np.float32)
        clim_flag_vals = np.array([0], dtype=np.int8)
        ssc_percent_complete_clim = 100.0
        qc_counts_clim = {
            "total": 1,
            "good": 1,
            "suspect": 0,
            "bad": 0,
            "missing": 0,
        }
        clim_ssc_source = "source_long_term_mean"
        if clim_source_period_lookup:
            clim_source_period = clim_source_period_lookup
        clim_temporal_span = canonical_time_span(
            clim_source_period, fallback=original_time_range
        )
        print(
            f"  Climatology source: source workbook long-term mean ({matched_station}, {clim_source_period})"
        )
    else:
        valid_clim = (SSC_flag == 0) & np.isfinite(ssc_vals)
        if np.any(valid_clim):
            clim_ssc_vals = np.array([np.mean(ssc_vals[valid_clim])], dtype=np.float32)
            clim_flag_vals = np.array([0], dtype=np.int8)
            ssc_percent_complete_clim = 100.0
            qc_counts_clim = {
                "total": 1,
                "good": 1,
                "suspect": 0,
                "bad": 0,
                "missing": 0,
            }
            print(
                "  Climatology source: fallback annual-sequence mean (source workbook value not found)."
            )
        else:
            clim_ssc_vals = np.array([-9999.0], dtype=np.float32)
            clim_flag_vals = np.array([9], dtype=np.int8)
            ssc_percent_complete_clim = 0.0
            qc_counts_clim = {
                "total": 1,
                "good": 0,
                "suspect": 0,
                "bad": 0,
                "missing": 1,
            }
            print(
                "  Climatology source: fallback missing (no source workbook value and no valid annual values)."
            )

    clim_start_year, clim_end_year = years_from_time_span(
        clim_temporal_span, fallback_start=ssc_start_date, fallback_end=ssc_end_date
    )

    clim_year = int(round((clim_start_year + clim_end_year) / 2.0))
    clim_date = datetime(clim_year, 7, 1)
    clim_time_vals = np.array(
        [nc.date2num(clim_date, units=OUTPUT_TIME_UNITS, calendar=OUTPUT_TIME_CALENDAR)],
        dtype=np.float64,
    )

    # Representative print
    valid = SSC_flag == 0
    if np.any(valid):
        ssc_repr = float(np.mean(ssc_vals[valid]))
        flag_repr = 0
    else:
        ssc_repr = float(ssc_vals[0]) if len(ssc_vals) > 0 else np.nan
        flag_repr = int(SSC_flag[0]) if len(SSC_flag) > 0 else 9

    print(f"  Station: {station_name} ({station_name_chinese})")
    print(f"  River: {river_name} ({river_name_chinese})")
    print(f"  Location: {lat:.4f}°N, {lon:.4f}°E")
    print(f"  SSC (representative): {ssc_repr:.2f} mg/L (flag={flag_repr})")

    output_file_ann = os.path.join(output_dir_ann, f"Huanghe_{station_id}_ann.nc")
    output_file_clim = os.path.join(output_dir_clim, f"Huanghe_{station_id}_clim.nc")

    print(f"  Annual output      : {os.path.basename(output_file_ann)}")
    print(f"  Climatology output : {os.path.basename(output_file_clim)}")

    # Write annual file
    write_standardized_file(
        output_file=output_file_ann,
        time_vals_out=ann_time_vals,
        ssc_vals_out=ann_ssc_vals,
        ssc_flag_out=ann_flag_vals,
        lon=lon,
        lat=lat,
        alt=alt,
        upstream_area=upstream_area,
        station_id=station_id,
        station_name=station_name,
        station_name_chinese=station_name_chinese,
        river_name=river_name,
        river_name_chinese=river_name_chinese,
        original_time_range=original_time_range,
        ssc_start_date=ssc_start_date,
        ssc_end_date=ssc_end_date,
        temporal_resolution="annual",
        observation_type="in-situ",
        title="Yellow River suspended sediment concentration (annual mean)",
        summary=(
            f"Suspended sediment concentration data for {station_name} station on the {river_name} "
            f"in the Yellow River Basin, China. This file preserves the original annual mean sequence "
            f"for the {original_time_range} period."
        ),
        ssc_comment=(
            "Source: Original data provided by Yellow River Sediment Bulletin (2015-2019). "
            "Unit conversion: Original unit kg/m³ × 1000 = mg/L. "
            "Each data point represents an annual mean observation for one available year."
        ),
        global_comment=(
            f"Annual mean SSC data for {original_time_range}. "
            f"Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
            f"Note: Discharge and sediment load data are NOT available in the original dataset."
        ),
    )

    # Write climatology file
    write_standardized_file(
        output_file=output_file_clim,
        time_vals_out=clim_time_vals,
        ssc_vals_out=clim_ssc_vals,
        ssc_flag_out=clim_flag_vals,
        lon=lon,
        lat=lat,
        alt=alt,
        upstream_area=upstream_area,
        station_id=station_id,
        station_name=station_name,
        station_name_chinese=station_name_chinese,
        river_name=river_name,
        river_name_chinese=river_name_chinese,
        original_time_range=clim_temporal_span,
        ssc_start_date=clim_start_year,
        ssc_end_date=clim_end_year,
        temporal_resolution="climatology",
        observation_type="in-situ",
        title="Yellow River suspended sediment concentration (climatological annual mean)",
        summary=(
            f"Suspended sediment concentration data for {station_name} station on the {river_name} "
            f"in the Yellow River Basin, China. This file contains one climatological annual mean "
            f"for the climatology period {clim_temporal_span}."
        ),
        ssc_comment=(
            "Source: Original data provided by Yellow River Sediment Bulletin. "
            "Unit conversion: Original unit kg/m³ × 1000 = mg/L. "
            + (
                f"Value represents source long-term mean SSC from {clim_source_period}."
                if clim_ssc_source == "source_long_term_mean"
                else "Value represents the climatological annual mean derived from available annual means."
            )
        ),
        global_comment=(
            (
                f"Climatological annual mean SSC from source long-term mean period {clim_source_period}. "
                if clim_ssc_source == "source_long_term_mean"
                else f"Climatological annual mean SSC derived from annual mean observations over {clim_temporal_span}. "
            )
            + "Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
            + "Note: Discharge and sediment load data are NOT available in the original dataset."
        ),
    )

    # Build station summaries
    station_info_ann = build_station_info(
        station_name=station_name,
        station_id=station_id,
        river_name=river_name,
        lon=lon,
        lat=lat,
        alt=alt,
        upstream_area=upstream_area,
        original_time_range=original_time_range,
        temporal_resolution="annual",
        ssc_start_date=ssc_start_date,
        ssc_end_date=ssc_end_date,
        ssc_percent_complete=ssc_percent_complete_ann,
        qc_counts=qc_counts_ann,
    )

    station_info_clim = build_station_info(
        station_name=station_name,
        station_id=station_id,
        river_name=river_name,
        lon=lon,
        lat=lat,
        alt=alt,
        upstream_area=upstream_area,
        original_time_range=clim_temporal_span,
        temporal_resolution="climatology",
        ssc_start_date=clim_start_year,
        ssc_end_date=clim_end_year,
        ssc_percent_complete=ssc_percent_complete_clim,
        qc_counts=qc_counts_clim,
    )

    return station_info_ann, station_info_clim


def get_representative_time_range(dates):
    """Return start year, end year and formatted time span for an annual sequence."""
    years = [int(d.year) for d in dates]
    start_year = min(years)
    end_year = max(years)
    return start_year, end_year, f"{start_year}-{end_year}"


def generate_annual_metadata_text(station_name, river_name, original_time_range):
    """Return annual metadata text fields."""
    title = "Yellow River suspended sediment concentration (annual mean)"
    summary = (
        f"Suspended sediment concentration data for {station_name} station on the {river_name} "
        f"in the Yellow River Basin, China. This file preserves the original annual mean sequence "
        f"for the {original_time_range} period."
    )
    comment = (
        f"Annual mean SSC data for {original_time_range}. "
        f"Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
        f"Note: Discharge and sediment load data are NOT available in the original dataset."
    )
    return title, summary, comment


def generate_climatology_metadata_text(station_name, river_name, original_time_range):
    """Return climatology metadata text fields."""
    title = "Yellow River suspended sediment concentration (climatological annual mean)"
    summary = (
        f"Suspended sediment concentration data for {station_name} station on the {river_name} "
        f"in the Yellow River Basin, China. This file contains one climatological annual mean "
        f"for the climatology period {clim_temporal_span}."
    )
    ssc_comment = (
        "Source: Original data provided by Yellow River Sediment Bulletin (2015-2019). "
        "Unit conversion: Original unit kg/m³ × 1000 = mg/L. "
        f"Value represents the climatological annual mean derived from available annual means over {original_time_range}."
    )
    global_comment = (
        f"Climatological annual mean SSC derived from annual mean observations over {clim_temporal_span}. "
        f"Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
        f"Note: Discharge and sediment load data are NOT available in the original dataset."
    )
    return title, summary, ssc_comment, global_comment


def get_scalar_value(ds, candidates):
    """Get scalar value from candidate variable names."""
    for name in candidates:
        if name in ds.variables:
            return read_scalar_variable(ds.variables[name])
    return np.nan


def repair_one_annual_file_and_write_climatology(
    annual_file, clim_output_dir, clim_suffix, climatology_lookup
):
    """Repair annual metadata in-place and write a new climatology file."""
    annual_file = Path(annual_file)
    with nc.Dataset(str(annual_file), "r+") as ds_ann:
        if "time" not in ds_ann.variables:
            raise KeyError("Missing required variable: time")
        if "SSC" not in ds_ann.variables:
            raise KeyError("Missing required variable: SSC")
        if "SSC_flag" not in ds_ann.variables:
            raise KeyError("Missing required variable: SSC_flag")

        time_var = ds_ann.variables["time"]
        time_vals_in = read_array_variable(time_var).astype(np.float64)
        time_units_in = getattr(time_var, "units", OUTPUT_TIME_UNITS)
        time_calendar_in = getattr(time_var, "calendar", OUTPUT_TIME_CALENDAR)
        dates = nc.num2date(time_vals_in, units=time_units_in, calendar=time_calendar_in)
        if len(dates) == 0:
            raise ValueError("No valid time values found")

        ssc_start_date, ssc_end_date, inferred_time_range = get_representative_time_range(dates)
        original_time_range = str(getattr(ds_ann, "temporal_span", inferred_time_range)).strip()
        if not original_time_range:
            original_time_range = inferred_time_range

        station_id = str(
            getattr(ds_ann, "Source_ID", getattr(ds_ann, "station_id", annual_file.stem))
        )
        station_name = str(getattr(ds_ann, "station_name", "Unknown Station"))
        station_name_chinese = str(getattr(ds_ann, "station_name_chinese", station_name))
        river_name = str(getattr(ds_ann, "river_name", "Unknown River"))
        river_name_chinese = str(getattr(ds_ann, "river_name_chinese", river_name))

        lon = get_scalar_value(ds_ann, ["lon", "longitude"])
        lat = get_scalar_value(ds_ann, ["lat", "latitude"])
        alt = get_scalar_value(ds_ann, ["altitude"])
        upstream_area = get_scalar_value(ds_ann, ["upstream_area"])

        ssc_vals = read_array_variable(ds_ann.variables["SSC"]).astype(np.float64)
        ssc_flag_vals = read_array_variable(ds_ann.variables["SSC_flag"]).astype(np.int8)

        # Repair annual semantics in global metadata.
        annual_title, annual_summary, annual_comment = generate_annual_metadata_text(
            station_name=station_name,
            river_name=river_name,
            original_time_range=original_time_range,
        )
        ds_ann.temporal_resolution = "annual"
        ds_ann.observation_type = "in-situ"
        ds_ann.title = annual_title
        ds_ann.summary = annual_summary
        ds_ann.comment = annual_comment
        ds_ann.date_modified = datetime.now().strftime("%Y-%m-%d")
        prev_history = str(getattr(ds_ann, "history", "")).strip()
        repair_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        repair_line = (
            f"{repair_stamp}: Repaired annual metadata semantics "
            f"(temporal_resolution, observation_type, title, summary, comment). "
            f"Script: qc_and_standardize.py"
        )
        if prev_history:
            ds_ann.history = f"{prev_history}\n{repair_line}"
        else:
            ds_ann.history = repair_line

    clim_ssc_source = "fallback_annual_mean"
    clim_source_period = original_time_range
    clim_temporal_span = original_time_range
    clim_ssc_val_from_lookup, clim_source_period_lookup, _ = resolve_climatology_from_lookup(
        climatology_lookup=climatology_lookup,
        station_name=station_name,
        station_name_chinese=station_name_chinese,
    )
    if clim_ssc_val_from_lookup is not None:
        clim_ssc_vals = np.array([clim_ssc_val_from_lookup], dtype=np.float32)
        clim_flag_vals = np.array([0], dtype=np.int8)
        clim_ssc_source = "source_long_term_mean"
        if clim_source_period_lookup:
            clim_source_period = clim_source_period_lookup
        clim_temporal_span = canonical_time_span(
            clim_source_period, fallback=original_time_range
        )
    else:
        valid_clim = (ssc_flag_vals == 0) & np.isfinite(ssc_vals)
        if np.any(valid_clim):
            clim_ssc_vals = np.array([np.mean(ssc_vals[valid_clim])], dtype=np.float32)
            clim_flag_vals = np.array([0], dtype=np.int8)
        else:
            clim_ssc_vals = np.array([-9999.0], dtype=np.float32)
            clim_flag_vals = np.array([9], dtype=np.int8)

    clim_start_year, clim_end_year = years_from_time_span(
        clim_temporal_span, fallback_start=ssc_start_date, fallback_end=ssc_end_date
    )

    clim_year = int(round((clim_start_year + clim_end_year) / 2.0))
    clim_date = datetime(clim_year, 7, 1)
    clim_time_vals = np.array(
        [nc.date2num(clim_date, units=OUTPUT_TIME_UNITS, calendar=OUTPUT_TIME_CALENDAR)],
        dtype=np.float64,
    )

    if not clim_suffix:
        raise ValueError("clim_suffix cannot be empty")
    normalized_suffix = clim_suffix if clim_suffix.startswith("_") else f"_{clim_suffix}"
    output_file_clim = Path(clim_output_dir) / f"Huanghe_{station_id}{normalized_suffix}.nc"

    clim_title, clim_summary, clim_ssc_comment, clim_global_comment = generate_climatology_metadata_text(
        station_name=station_name,
        river_name=river_name,
        original_time_range=clim_temporal_span,
    )
    if clim_ssc_source == "source_long_term_mean":
        clim_ssc_comment = (
            "Source: Original data provided by Yellow River Sediment Bulletin. "
            "Unit conversion: Original unit kg/m³ × 1000 = mg/L. "
            f"Value represents source long-term mean SSC from {clim_source_period}."
        )
        clim_global_comment = (
            f"Climatological annual mean SSC from source long-term mean period {clim_source_period}. "
            "Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
            "Note: Discharge and sediment load data are NOT available in the original dataset."
        )

    write_standardized_file(
        output_file=str(output_file_clim),
        time_vals_out=clim_time_vals,
        ssc_vals_out=clim_ssc_vals,
        ssc_flag_out=clim_flag_vals,
        lon=lon,
        lat=lat,
        alt=alt,
        upstream_area=upstream_area,
        station_id=station_id,
        station_name=station_name,
        station_name_chinese=station_name_chinese,
        river_name=river_name,
        river_name_chinese=river_name_chinese,
        original_time_range=clim_temporal_span,
        ssc_start_date=clim_start_year,
        ssc_end_date=clim_end_year,
        temporal_resolution="climatology",
        observation_type="in-situ",
        title=clim_title,
        summary=clim_summary,
        ssc_comment=clim_ssc_comment,
        global_comment=clim_global_comment,
    )
    return station_id, str(output_file_clim)


def run_repair_existing_annual(args):
    """Repair annual files and rebuild climatology files from repaired annual products."""
    annual_dir = require_existing_directory(
        Path(args.annual_dir),
        description="annual standardized NetCDF directory",
    )
    clim_output_dir = ensure_directory(Path(args.clim_output_dir))
    os.makedirs(clim_output_dir, exist_ok=True)

    source_excel_file = resolve_source_root(start=__file__) / "HuangHe" / SOURCE_EXCEL_FILENAME
    climatology_lookup = load_climatology_lookup_from_excel(source_excel_file)

    annual_files = sorted(glob.glob(str(annual_dir / "*_ann.nc")))
    if len(annual_files) == 0:
        print(f"ERROR: No annual NetCDF files found in {annual_dir}")
        return 1

    print("=" * 80)
    print("Repair Existing Annual Metadata + Rebuild Climatology")
    print("=" * 80)
    print(f"Annual input directory        : {annual_dir}")
    print(f"New climatology output dir    : {clim_output_dir}")
    print(f"New climatology filename suffix: {args.clim_suffix}")
    print(f"Total annual files found      : {len(annual_files)}")
    print()

    processed_count = 0
    error_count = 0
    failed_items = []

    for annual_file in annual_files:
        print(f"Processing: {os.path.basename(annual_file)}")
        try:
            station_id, clim_file = repair_one_annual_file_and_write_climatology(
                annual_file=annual_file,
                clim_output_dir=clim_output_dir,
                clim_suffix=args.clim_suffix,
                climatology_lookup=climatology_lookup,
            )
            processed_count += 1
            print(f"  Repaired annual metadata for station: {station_id}")
            print(f"  Created new climatology file       : {os.path.basename(clim_file)}")
        except Exception as e:
            error_count += 1
            failed_items.append((annual_file, str(e)))
            print(f"  ERROR: {e}")
        print()

    print("=" * 80)
    print("Repair/Rebuild Summary")
    print("=" * 80)
    print(f"Total files repaired successfully: {processed_count}")
    print(f"Total files failed              : {error_count}")
    print(f"Annual input directory          : {annual_dir}")
    print(f"New climatology output directory: {clim_output_dir}")
    if failed_items:
        print()
        print("Failed files:")
        for annual_file, reason in failed_items:
            print(f"  - {annual_file}: {reason}")
    print("=" * 80)
    print()
    return 0 if error_count == 0 else 1


def run_standard_pipeline():
    """Main processing function for source-to-standardized pipeline."""

    print("=" * 80)
    print("Yellow River Sediment Data - QC and CF-1.8 Standardization")
    print("=" * 80)
    print()

    # Paths
    input_dir = require_existing_directory(
        resolve_source_root(start=__file__) / "HuangHe" / "netcdf",
        description="HuangHe intermediate NetCDF directory",
    )

    output_dir_ann = ensure_directory(
        resolve_output_root(start=__file__) / "annual" / "Huanghe"
    )
    output_dir_clim = ensure_directory(
        resolve_output_root(start=__file__) / "climatology" / "Huanghe"
    )
    source_excel_file = resolve_source_root(start=__file__) / "HuangHe" / SOURCE_EXCEL_FILENAME
    climatology_lookup = load_climatology_lookup_from_excel(source_excel_file)

    os.makedirs(output_dir_ann, exist_ok=True)
    os.makedirs(output_dir_clim, exist_ok=True)

    # Get all NetCDF files
    input_files = sorted(glob.glob(str(input_dir / "HuangHe_*.nc")))

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    station_info_list_ann = []
    station_info_list_clim = []
    processed_count = 0
    error_count = 0

    for input_file in input_files:
        try:
            station_info_ann, station_info_clim = standardize_netcdf_file(
                input_file=input_file,
                output_dir_ann=output_dir_ann,
                output_dir_clim=output_dir_clim,
                climatology_lookup=climatology_lookup,
            )
            station_info_list_ann.append(station_info_ann)
            station_info_list_clim.append(station_info_clim)
            processed_count += 1
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print()
    print("=" * 80)
    print("Generating Station Summary CSVs")
    print("=" * 80)
    print()

    df_ann = pd.DataFrame(station_info_list_ann)
    df_clim = pd.DataFrame(station_info_list_clim)

    column_order = [
        "station_name",
        "Source_ID",
        "river_name",
        "longitude",
        "latitude",
        "altitude",
        "upstream_area",
        "Data Source Name",
        "Type",
        "Temporal Resolution",
        "Temporal Span",
        "Variables Provided",
        "Geographic Coverage",
        "Reference/DOI",
        "Q_start_date",
        "Q_end_date",
        "Q_percent_complete",
        "SSC_start_date",
        "SSC_end_date",
        "SSC_percent_complete",
        "SSL_start_date",
        "SSL_end_date",
        "SSL_percent_complete",
        "SSC_n_total",
        "SSC_n_good",
        "SSC_n_suspect",
        "SSC_n_bad",
        "SSC_n_missing",
    ]

    if len(df_ann) > 0:
        df_ann = df_ann[column_order]
    if len(df_clim) > 0:
        df_clim = df_clim[column_order]

    csv_file_ann = os.path.join(output_dir_ann, "Huanghe_station_summary_annual.csv")
    csv_file_clim = os.path.join(
        output_dir_clim, "Huanghe_station_summary_climatology.csv"
    )

    df_ann.to_csv(csv_file_ann, index=False)
    df_clim.to_csv(csv_file_clim, index=False)

    print(f"Annual station summary saved to      : {csv_file_ann}")
    print(f"Climatology station summary saved to : {csv_file_clim}")
    print(f"Total annual stations                : {len(df_ann)}")
    print(f"Total climatology stations           : {len(df_clim)}")
    print()

    if len(df_ann) > 0:
        print("Global QC summary (annual SSC):")
        print(df_ann[["SSC_n_good", "SSC_n_suspect", "SSC_n_bad", "SSC_n_missing"]].sum())
        print()

    if len(df_clim) > 0:
        print("Global QC summary (climatology SSC):")
        print(
            df_clim[["SSC_n_good", "SSC_n_suspect", "SSC_n_bad", "SSC_n_missing"]].sum()
        )
        print()

    print("=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total files processed successfully: {processed_count}")
    print(f"Errors encountered               : {error_count}")
    print(f"Annual output directory          : {output_dir_ann}")
    print(f"Climatology output directory     : {output_dir_clim}")
    print()

    print("=" * 80)
    print("Output Description")
    print("=" * 80)
    print("Annual files:")
    print("  - Keep original yearly sequence (e.g., 2015-07-01 ... 2019-07-01)")
    print('  - Global attribute temporal_resolution = "annual"')
    print()
    print("Climatology files:")
    print("  - Collapse annual series into one climatological annual mean value")
    print('  - Global attribute temporal_resolution = "climatology"')
    print("=" * 80)
    print()


def parse_args():
    """Parse command-line arguments."""
    default_annual_dir = resolve_output_root(start=__file__) / "annual" / "Huanghe"
    default_clim_dir = resolve_output_root(start=__file__) / "climatology" / "Huanghe"

    parser = argparse.ArgumentParser(
        description=(
            "Yellow River SSC QC/standardization tool. "
            "Default mode converts source annual files into annual+climatology standardized outputs. "
            "Repair mode fixes existing annual metadata and rebuilds climatology products."
        )
    )
    parser.add_argument(
        "--repair-existing-annual",
        action="store_true",
        help=(
            "Repair annual metadata semantics in existing *_ann.nc files and rebuild "
            "new climatology files from repaired annual data."
        ),
    )
    parser.add_argument(
        "--annual-dir",
        default=str(default_annual_dir),
        help="Directory containing existing annual files (default: Output_r/annual/Huanghe).",
    )
    parser.add_argument(
        "--clim-output-dir",
        default=str(default_clim_dir),
        help=(
            "Directory to write rebuilt climatology files "
            "(default: Output_r/climatology/Huanghe)."
        ),
    )
    parser.add_argument(
        "--clim-suffix",
        default="_clim_new",
        help="Suffix for new climatology filename, e.g., _clim_new.",
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    if args.repair_existing_annual:
        return run_repair_existing_annual(args)
    run_standard_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
