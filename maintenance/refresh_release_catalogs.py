#!/usr/bin/env python3
"""Refresh source_station_catalog.csv and station_catalog.csv from fixed NC files.

The source_station_catalog.csv references NC files in
output_resolution_organized/ which have filenames like:
  {Dataset}_{resolution}_{StationName}.nc

The fixed QC files in Output_r/ have filenames like:
  {StationName}.nc (without the resolution prefix)
  
This script builds a lookup from QC files, then maps organized paths
to QC paths using the filename convention.
"""
import pandas as pd
import os, csv, re
import netCDF4 as nc
from pathlib import Path

ROOT = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r"
RELEASE_DIR = os.path.join(ROOT, "scripts_basin_test/output/sed_reference_release")
SOURCE_CATALOG = os.path.join(RELEASE_DIR, "source_station_catalog.csv")
STATION_CATALOG = os.path.join(RELEASE_DIR, "station_catalog.csv")

RESOLUTION_KEYWORDS = ('daily_', 'monthly_', 'annual_', 'climatology_', 'other_')


def _build_qc_lookup():
    """Build dict: basename(fname) -> full QC path for all QC NC files."""
    lookup = {}
    for res_dir in os.listdir(ROOT):
        res_path = os.path.join(ROOT, res_dir)
        if not os.path.isdir(res_path) or res_dir.startswith('_'):
            continue
        for ds_name in os.listdir(res_path):
            qc_dir = os.path.join(res_path, ds_name, 'qc')
            if not os.path.isdir(qc_dir):
                continue
            for fname in os.listdir(qc_dir):
                if fname.endswith('.nc'):
                    lookup[fname] = os.path.join(qc_dir, fname)
    return lookup


def _organized_to_station_fname(organized_path):
    """Convert organized filename to station filename.
    
    'Dataset_Resolution_Station.nc' -> 'Station.nc'
    
    The resolution is always one of: daily_, monthly_, annual_, climatology_, other_
    We find it by scanning for known resolution keywords.
    """
    fname = os.path.basename(organized_path)
    if not fname.endswith('.nc'):
        return None
    
    for kw in RESOLUTION_KEYWORDS:
        idx = fname.find('_' + kw[0:-1] + '_')  # '_annual_', '_daily_', etc.
        if idx >= 0:
            # Everything after the resolution keyword
            return fname[idx + len(kw):]  # kw already has trailing _
    
    # Fallback: look for "_{resolution}_" in the filename
    parts = fname.replace('.nc', '').split('_')
    for i, part in enumerate(parts):
        if part.lower() in ('daily', 'monthly', 'annual', 'climatology', 'other'):
            if i + 1 < len(parts):
                return '_'.join(parts[i+1:]) + '.nc'
    
    return None


def _read_geo_from_qc(qc_lookup, organized_path):
    """Read geo attributes from QC file matching the organized path."""
    station_fname = _organized_to_station_fname(organized_path)
    if not station_fname:
        return ('', '', '', '')
    
    qc_path = qc_lookup.get(station_fname)
    if not qc_path:
        return ('', '', '', '')
    
    try:
        ds = nc.Dataset(qc_path, 'r')
        country = getattr(ds, 'country', '') or ''
        continent = getattr(ds, 'continent_region', '') or ''
        geo = getattr(ds, 'geographic_coverage', '') or ''
        iso = getattr(ds, 'iso_a3', '') or ''
        ds.close()
        return (country, continent, geo, iso)
    except Exception:
        return ('', '', '', '')


def update_source_catalog(qc_lookup):
    with open(SOURCE_CATALOG, 'r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    
    if not reader:
        print("Empty source catalog!")
        return [], []
    
    header = reader[0]
    rows = reader[1:]
    total = len(rows)
    
    col_idx = {name: i for i, name in enumerate(header)}
    path_col = col_idx.get('source_station_paths', -1)
    
    countries_found = 0
    
    for i, row in enumerate(rows):
        path = row[path_col].strip() if path_col >= 0 and path_col < len(row) else ''
        if not path:
            continue
        
        country, continent, geo, iso = _read_geo_from_qc(qc_lookup, path)
        
        row[col_idx['country']] = country
        row[col_idx['continent_region']] = continent
        row[col_idx['geographic_coverage']] = geo
        row[col_idx['iso_a3']] = iso
        
        if country:
            countries_found += 1
            row[col_idx['geo_attribute_source']] = 'source_nc_global_attrs'
            row[col_idx['geo_attribute_confidence']] = 'high'
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{total} ({countries_found} with country)...")
    
    with open(SOURCE_CATALOG, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"  Done: {countries_found}/{total} rows with country data")
    return rows, header


def update_station_catalog(updated_rows, header):
    src_df = pd.DataFrame(updated_rows, columns=header)
    station_df = pd.read_csv(STATION_CATALOG, keep_default_na=False)
    
    group_keys = ['cluster_uid', 'resolution']
    geo_cols = ['country', 'continent_region', 'geographic_coverage', 'iso_a3',
                'geo_attribute_source', 'geo_attribute_confidence']
    
    for col in geo_cols:
        if col not in src_df.columns or col not in station_df.columns:
            continue
        grouped = src_df.groupby(group_keys)[col].apply(
            lambda x: '|'.join(sorted(set(str(v) for v in x if v and str(v).strip())))
        ).reset_index().rename(columns={col: col + '_new'})
        
        station_df = station_df.merge(grouped, on=group_keys, how='left', sort=False)
        merged_col = col + '_new'
        if merged_col in station_df.columns:
            old = station_df[col].fillna('')
            new = station_df[merged_col].fillna('')
            station_df[col] = new.where(new != '', old)
            station_df.drop(columns=[merged_col], inplace=True)
    
    station_df.to_csv(STATION_CATALOG, index=False)
    total = len(station_df)
    filled = station_df['country'].fillna('').str.strip().ne('').sum() if 'country' in station_df.columns else 0
    print(f"  station_catalog: {total} rows, {filled} with country")


def main():
    print("Building QC file lookup...")
    qc_lookup = _build_qc_lookup()
    print(f"  Found {len(qc_lookup)} QC files")
    
    print("\n=== Updating source_station_catalog.csv ===")
    updated_rows, header = update_source_catalog(qc_lookup)
    
    print("\n=== Updating station_catalog.csv ===")
    update_station_catalog(updated_rows, header)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
