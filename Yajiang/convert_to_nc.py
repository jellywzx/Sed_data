#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Yajiang English table (one row = one record) to NetCDF
Output: one NetCDF file per row.

Default behavior:
- Write coordinates + station metadata
- Write SSC and Discharge
- Optionally include water-chemistry vars (T, EC, TDS, pH, DO, NTU)

Dependencies: pandas, numpy, netCDF4
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
from pathlib import Path

# =======================
# User config
# =======================
INPUT_FILE = "/mnt/d/sediment_wzx_1111/Source/Yajiang/data_en.xlsx"
OUTPUT_DIR = "/mnt/d/sediment_wzx_1111/Output_r/daily/Yajiang/nc"

# If False: only keep SSC + Discharge (recommended for your purpose)
# If True : also save T/EC/TDS/pH/DO/NTU into nc
INCLUDE_WATER_CHEM = False

FILL_VALUE_FLOAT = np.float32(-9999.0)

# =======================
# Helpers
# =======================
def to_nan(val):
    """Convert '/', blank, NA-like to np.nan; otherwise keep."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["/", "NA", "N.A.", "nan", "NaN", ""]:
        return np.nan
    return val

def clean_float(val):
    """Convert to float if possible; else nan."""
    val = to_nan(val)
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().replace(",", ""))
    except Exception:
        return np.nan

def parse_date_yyyymmdd(val):
    """Parse YYYYMMDD into datetime."""
    val = to_nan(val)
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def safe_str(x):
    x = to_nan(x)
    if pd.isna(x):
        return "N.A."
    return str(x).strip()

def create_one_nc(row, out_path: Path):
    # Required columns (English) - try multiple variations
    site = safe_str(row.get("Site", row.get("Site Name", "")))
    sid  = safe_str(row.get("ID", row.get("No.", row.get("Station ID", ""))))

    lon = clean_float(row.get("Longitude (°E)", row.get("Longitude", row.get("Lon", None))))
    lat = clean_float(row.get("Latitude (°N)", row.get("Latitude", row.get("Lat", None))))
    elev = clean_float(row.get("Elevation (m)", row.get("Elevation", None)))

    date_str = str(row.get("Date (YYYYMMDD)", row.get("Date", row.get("Time", ""))))
    date_dt = parse_date_yyyymmdd(date_str)
    if date_dt is None:
        return False, f"Invalid date: {date_str}"
    if np.isnan(lon) or np.isnan(lat):
        return False, f"Invalid coordinates: lon={lon}, lat={lat}"

    ssc = clean_float(row.get("Suspended Sediment Concentration (g/L)", row.get("SSC (g/L)", row.get("SSC", row.get("Suspended Sediment", None)))))
    q   = clean_float(row.get("Discharge (m³/s)", row.get("Discharge", row.get("Q", row.get("Flow", None)))))

    # Create NetCDF
    ds = nc.Dataset(out_path, "w", format="NETCDF4")
    try:
        # Dimensions
        ds.createDimension("time", 1)
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)

        # time
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var.standard_name = "time"
        time_var.long_name = "time"
        time_var[:] = nc.date2num([date_dt], time_var.units, time_var.calendar)

        # lat/lon
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name = "station latitude"
        lat_var[:] = lat

        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name = "station longitude"
        lon_var[:] = lon

        # elevation (scalar)
        elev_var = ds.createVariable("elevation", "f4")
        elev_var.units = "m"
        elev_var.standard_name = "height_above_mean_sea_level"
        elev_var.long_name = "station elevation"
        elev_var.assignValue(elev if not np.isnan(elev) else np.nan)

        # Variables you likely want
        ssc_var = ds.createVariable("SSC", "f4", ("time", "lat", "lon"), fill_value=FILL_VALUE_FLOAT)
        ssc_var.units = "g L-1"
        ssc_var.long_name = "Suspended sediment concentration"
        ssc_var[0, 0, 0] = ssc if not np.isnan(ssc) else FILL_VALUE_FLOAT

        q_var = ds.createVariable("Q", "f4", ("time", "lat", "lon"), fill_value=FILL_VALUE_FLOAT)
        q_var.units = "m3 s-1"
        q_var.standard_name = "river_discharge"
        q_var.long_name = "River discharge"
        q_var[0, 0, 0] = q if not np.isnan(q) else FILL_VALUE_FLOAT

        # Optional derived SSL (ton/day): SSL = Q * SSC * 86.4  (since SSC g/L = kg/m3)
        ssl_var = ds.createVariable("SSL", "f4", ("time", "lat", "lon"), fill_value=FILL_VALUE_FLOAT)
        ssl_var.units = "ton day-1"
        ssl_var.long_name = "Suspended sediment load"
        ssl_var.comment = "Derived when both Q and SSC exist: SSL(ton/day) = Q(m3/s) * SSC(g/L) * 86.4"
        if (not np.isnan(q)) and (not np.isnan(ssc)):
            ssl_var[0, 0, 0] = np.float32(q * ssc * 86.4)
        else:
            ssl_var[0, 0, 0] = FILL_VALUE_FLOAT

        # Optionally include water chemistry
        if INCLUDE_WATER_CHEM:
            # Expect columns: T (°C), EC, TDS, pH, DO (%), NTU
            chem_map = {
                "T": ("T (°C)", "degC", "Water temperature"),
                "EC": ("EC", "uS cm-1", "Electrical conductivity"),
                "TDS": ("TDS", "mg L-1", "Total dissolved solids"),
                "pH": ("pH", "1", "pH"),
                "DO": ("DO (%)", "%", "Dissolved oxygen saturation"),
                "NTU": ("NTU", "NTU", "Turbidity"),
            }
            for vname, (col, unit, lname) in chem_map.items():
                v = clean_float(row.get(col))
                var = ds.createVariable(vname, "f4", ("time", "lat", "lon"), fill_value=FILL_VALUE_FLOAT)
                var.units = unit
                var.long_name = lname
                var[0, 0, 0] = v if not np.isnan(v) else FILL_VALUE_FLOAT

        # Global attributes
        ds.Conventions = "CF-1.8"
        ds.title = f"Yajiang field record: {site} ({sid})"
        ds.source = "English table (one row per record)"
        ds.history = f"Created on {datetime.now():%Y-%m-%d %H:%M:%S}"
        ds.station_id = sid
        ds.station_name = site
        ds.time_coverage_start = date_dt.strftime("%Y-%m-%d")
        ds.time_coverage_end = date_dt.strftime("%Y-%m-%d")

        return True, "Success"

    finally:
        ds.close()

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(INPUT_FILE)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    # Read table
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(in_path, header=0)
    else:
        df = pd.read_csv(in_path, header=0)

    # Remove empty rows
    df = df.dropna(how='all')
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row preview:")
    if len(df) > 0:
        print(df.iloc[0].to_dict())
    
    success_count = 0
    skip_count = 0
    error_messages = {}

    print("\nProcessing rows...")
    for idx, row in df.iterrows():
        site = safe_str(row.get("Site"))
        sid = safe_str(row.get("ID"))
        date_str = str(row.get("Date (YYYYMMDD)", "NA"))

        # Output filename
        safe_sid = sid.replace("/", "_").replace(" ", "_")
        safe_date = date_str.replace(" ", "")
        out_file = out_dir / f"Yajiang_{sid}.nc"

        success, message = create_one_nc(row, out_file)

        if success:
            success_count += 1
            print(f"  ✓ Row {idx+1}: {site} ({sid}) {date_str}")
        else:
            skip_count += 1
            error_messages[f"row{idx}_{sid}_{date_str}"] = message
            if out_file.exists():
                out_file.unlink()
            print(f"  ✗ Row {idx+1}: {message}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows processed: {len(df)}")
    print(f"Successfully created: {success_count}")
    print(f"Skipped: {skip_count}")

    if skip_count > 0:
        print("\nSkipped reasons:")
        reason_counts = {}
        for _, reason in error_messages.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nOutput dir: {out_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
