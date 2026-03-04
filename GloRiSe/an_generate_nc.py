#!/usr/bin/env python
"""
Generate CF-compliant NetCDF files from GloRiSe sediment database.

Extended version:
Exports ObservationType='an' data from SedimentDatabase_ME_Nut.csv into NetCDF files,
including matched Lat/Lon from SedimentDatabase_Locations.csv.
Only keeps TSS_mg_L, Discharge_m3_s, Sand_perc, Silt_perc, Clay_perc.
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from pathlib import Path
import re
import bibtexparser

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
BASE_DIR = Path(r'D:\sediment_data\Source\GloRiSe')
REF_FILE = BASE_DIR / 'SedimentDatabase_ref.xlsx'
LOC_FILE = BASE_DIR / 'SedimentDatabase_Locations.csv'
ID_FILE = BASE_DIR / 'SedimentDatabase_ID.xlsx'
ME_FILE = BASE_DIR / 'SedimentDatabase_ME_Nut.csv'
BIB_FILE = BASE_DIR / 'References_RiSe.bib'


OUTPUT_DIR = BASE_DIR / 'netcdf_output_an'
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARY_FILE = OUTPUT_DIR / 'GloRiSe_an_all.csv'

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def clean_latex_text(text):
    text = re.sub(r'\\v\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r"\\\'\{([a-zA-Z])\}", r'\1', text)
    text = re.sub(r'\\"\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r'\\`\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r'\\\^\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r'\\~\{([a-zA-Z])\}', r'\1', text)
    text = re.sub(r'\\([a-zA-Z])', r'\1', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\', '', text)
    return text

def load_bibtex_references(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)
    citations = {}
    for entry in bib_database.entries:
        citation_key = entry.get('ID', '')
        entry_type = entry.get('ENTRYTYPE', 'article')
        authors = entry.get('author', '')
        year = entry.get('year', 'n.d.')
        title = clean_latex_text(entry.get('title', 'Untitled'))
        citations[citation_key] = f"{authors} ({year}). {title}."
    return citations

# ---------------------------------------------------------------------
# Create NetCDF for ObservationType='an'
# ---------------------------------------------------------------------
def create_netcdf_for_an_samples(location_id, an_data, location_info, df_id, summary_rows):
    """Generate CF-1.8-compliant NetCDF with QC (using 'sample' dimension)."""

    # --- 仅保留同时有 Q 与 TSS 的样本 ---
    an_data_filtered = an_data[
        an_data["TSS_mg_L"].notna() & an_data["Discharge_m3_s"].notna()
    ].copy()

    if an_data_filtered.empty:
        print(f"  Skipping {location_id}: No valid records (TSS & Q both required)")
        return False

    # --- 计算 SSL (t/day) ---
    an_data_filtered["SSL"] = (
        an_data_filtered["TSS_mg_L"].astype(float)
        * an_data_filtered["Discharge_m3_s"].astype(float)
        * 0.0864
    )

    # --- 经纬度 ---
    lat_val = float(location_info["Lat_deg"])
    lon_val = float(location_info["Lon_deg"])

    # --- 输出文件路径 ---
    filename = OUTPUT_DIR / f"GloRiSe_an_{location_id}.nc"
    ds = Dataset(filename, "w", format="NETCDF4")

    try:
        # === 定义维度 ===
        ds.createDimension("sample", len(an_data_filtered))
        ds.createDimension("lat", 1)
        ds.createDimension("lon", 1)

        lat = ds.createVariable("lat", "f4", ("lat",))
        lon = ds.createVariable("lon", "f4", ("lon",))

        lat.standard_name = "latitude"
        lon.standard_name = "longitude"
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        lat[:] = lat_val
        lon[:] = lon_val

        # === 数据变量 ===
        def make_var(name, data, long_name, units, flag_name):
            var = ds.createVariable(name, "f4", ("sample", "lat", "lon"), fill_value=-9999.0)
            var[:] = data.fillna(-9999).values.reshape(-1, 1, 1)
            var.standard_name = long_name
            var.units = units
            var.ancillary_variables = flag_name
            var.coordinates = "lat lon"
            return var

        ds_Q = make_var("Q", an_data_filtered["Discharge_m3_s"],
                        "water_volume_transport_in_river_channel", "m3 s-1", "Q_flag")
        ds_SSC = make_var("SSC", an_data_filtered["TSS_mg_L"],
                          "mass_concentration_of_suspended_matter_in_water", "mg L-1", "SSC_flag")
        ds_SSL = make_var("SSL", an_data_filtered["SSL"],
                          "suspended_sediment_load", "ton day-1", "SSL_flag")

        # === QC 检查 ===
        def qc_check(data, vmin, vmax):
            flags = np.full(len(data), 0, dtype=np.int8)
            vals = data.values
            flags[(vals == -9999) | np.isnan(vals)] = 9
            flags[vals < vmin] = 3
            flags[vals > vmax] = 2
            return flags

        q_flag = qc_check(an_data_filtered["Discharge_m3_s"], 0, 300000)
        ssc_flag = qc_check(an_data_filtered["TSS_mg_L"], 0, 3000)
        ssl_flag = qc_check(an_data_filtered["SSL"], 0, np.inf)

        
        # --- 保存到 CSV 汇总列表 ---
        summary_df = pd.DataFrame({
            "Location_ID": location_id,
            "Lat_deg": lat_val,
            "Lon_deg": lon_val,
            "TSS_mg_L": an_data_filtered["TSS_mg_L"].values,
            "Discharge_m3_s": an_data_filtered["Discharge_m3_s"].values,
            "SSL": an_data_filtered["SSL"].values,
            "Q_flag": q_flag,
            "SSC_flag": ssc_flag,
            "SSL_flag": ssl_flag,
            "Sand_perc": an_data_filtered.get("Sand_perc", np.nan).values,
            "Silt_perc": an_data_filtered.get("Silt_perc", np.nan).values,
            "Clay_perc": an_data_filtered.get("Clay_perc", np.nan).values,
        })
        summary_rows.append(summary_df)


        # === Flag 变量 ===
        def make_flag(name, flag):
            fvar = ds.createVariable(name, "i1", ("sample",), fill_value=9)
            fvar.long_name = f"quality flag for {name.replace('_flag','')}"
            fvar.standard_name = "status_flag"
            fvar.flag_values = np.array([0,1,2,3,9], dtype=np.int8)
            fvar.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
            fvar[:] = flag

        make_flag("Q_flag", q_flag)
        make_flag("SSC_flag", ssc_flag)
        make_flag("SSL_flag", ssl_flag)

        # === 粒径变量 ===
        for col in ["Sand_perc", "Silt_perc", "Clay_perc"]:
            if col in an_data_filtered.columns:
                v = ds.createVariable(col, "f4", ("sample", "lat", "lon"), fill_value=-9999.0)
                v[:] = an_data_filtered[col].fillna(-9999).values.reshape(-1, 1, 1)
                v.units = "%"
                v.long_name = f"{col.replace('_',' ').capitalize()}"
                v.coordinates = "lat lon"

        # === 全局属性 ===
        ds.title = f"GloRiSe analytical data (ObservationType='an') for station {location_id}"
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.history = f"Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: QC applied following GloRiSe standard."
        ds.processing_level = "Quality controlled and standardized"
        ds.creator_name = "Zhongwang Wei"
        ds.creator_institution = "Sun Yat-sen University, China"

        print(f"  ✓ Processed {filename.name}: {len(an_data_filtered)} samples, lat={lat_val}, lon={lon_val}, QC applied, SSL included")
        return True

    finally:
        ds.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading data files...")

    # Load datasets
    df_loc = pd.read_csv(LOC_FILE)
    df_me = pd.read_csv(ME_FILE)
    df_id = pd.read_excel(ID_FILE)

    print(f"Loaded {len(df_loc)} locations")
    print(f"Loaded {len(df_me)} measurements")

    # Filter ObservationType='an'
    df_an = df_me[df_me['Observationtype'].astype(str).str.lower() == 'an'].copy()

    if df_an.empty:
        print("No 'an' samples found.")
        return

    if 'Location_ID' not in df_an.columns:
        raise KeyError("'Location_ID' column not found in SedimentDatabase_ME_Nut.csv")

    # Match locations by Location_ID to get lat/lon
    df_merged = pd.merge(df_an, df_loc[['Location_ID', 'Lat_deg', 'Lon_deg', 'Elevation_masl']],
                         on='Location_ID', how='inner')

    unique_locations_an = df_merged['Location_ID'].dropna().unique()
    print(f"Found {len(df_merged)} 'an' samples from {len(unique_locations_an)} locations.")
    
    summary_rows = []

    for loc_id in unique_locations_an:
        loc_data = df_merged[df_merged['Location_ID'] == loc_id]
        loc_info = loc_data.iloc[0]  # now already contains lat/lon
        if pd.isna(loc_info["Lat_deg"]) or pd.isna(loc_info["Lon_deg"]):
            print(f"  Skipping {loc_id}: Missing coordinates")
            continue
        create_netcdf_for_an_samples(loc_id, loc_data, loc_info, df_id, summary_rows)
    # === 输出汇总 CSV ===
    if summary_rows:
        df_summary = pd.concat(summary_rows, ignore_index=True)
        df_summary.to_csv(SUMMARY_FILE, index=False)
        print(f"\n✓ Saved summary CSV with {len(df_summary)} samples: {SUMMARY_FILE}")
    else:
        print("\nNo valid samples to save.")

    print("\nAll processing complete!")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
