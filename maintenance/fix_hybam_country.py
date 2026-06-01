#!/usr/bin/env python3
"""Fix HYBAM NC files with per-station country/continent_region/iso_a3."""
import netCDF4 as nc, glob, os

ROOT = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r"

MAP = {
    "4071002205": ("Bolivia", "South America", "BOL"),
    "15900000": ("Brazil", "South America", "BRA"),
    "10064000": ("Peru", "South America", "PER"),
    "50800000": ("Republic of the Congo", "Africa", "COG"),
    "14710000": ("Brazil", "South America", "BRA"),
    "40800000": ("Venezuela", "South America", "VEN"),
    "15860000": ("Brazil", "South America", "BRA"),
    "10080900": ("Ecuador", "South America", "ECU"),
    "17730000": ("Brazil", "South America", "BRA"),
    "10073500": ("Peru", "South America", "PER"),
    "2604100121": ("Suriname", "South America", "SUR"),
    "14100000": ("Brazil", "South America", "BRA"),
    "17050001": ("Brazil", "South America", "BRA"),
    "15400000": ("Brazil", "South America", "BRA"),
    "15275100": ("Bolivia", "South America", "BOL"),
    "2604500124": ("French Guiana", "South America", "GUF"),
    "14420000": ("Brazil", "South America", "BRA"),
}

for f in sorted(glob.glob(os.path.join(ROOT, "*/HYBAM/qc/*.nc"))):
    sid = os.path.basename(f).replace("HYBAM_", "").replace(".nc", "")
    info = MAP.get(sid)
    if info:
        ds = nc.Dataset(f, "r+")
        ds.country = info[0]
        ds.continent_region = info[1]
        ds.iso_a3 = info[2]
        ds.close()
        print(f"HYBAM {sid}: {info[0]}, {info[1]}, {info[2]}")
    else:
        print(f"HYBAM {sid}: NOT FOUND in map")
