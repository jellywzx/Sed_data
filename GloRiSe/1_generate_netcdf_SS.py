#!/usr/bin/env python
"""
Generate CF-compliant NetCDF files from GloRiSe sediment database.

This script reads data from multiple Excel files, integrates them by station,
and generates one NetCDF file per station containing Discharge and TSS time series.

IMPORTANT (2026-08): sediment-eligible filtering now requires ONLY valid TSS/SSC
(Sampletype=="SS" + TSS_mg_L not null). Discharge (Q) is NOT a gate-keeping
criterion — SSC-only records are preserved and Q is written as _FillValue when
absent.  This ensures TSS-only stations are not dropped during NetCDF generation.
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset, date2num
from datetime import datetime
import os
from pathlib import Path
import bibtexparser
import re
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = CURRENT_DIR.parent
CODE_DIR = SCRIPT_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from runtime import ensure_directory, resolve_source_root
from validation import require_existing_file

# File paths
BASE_DIR = resolve_source_root(start=__file__) / 'GloRiSe'
REF_FILE = BASE_DIR / 'SedimentDatabase_ref.xlsx'
LOC_FILE = BASE_DIR / 'SedimentDatabase_Locations.xlsx'
ID_FILE = BASE_DIR / 'SedimentDatabase_ID.xlsx'
ME_FILE = BASE_DIR / 'SedimentDatabase_ME_Nut.xlsx'
BIB_FILE = BASE_DIR / 'References_RiSe.bib'

# Output directory
OUTPUT_DIR = ensure_directory(BASE_DIR / 'netcdf_output_SS')

def clean_latex_text(text):
    """
    Remove LaTeX special characters and formatting from text.
    """
    # Remove common LaTeX formatting commands
    text = re.sub(r'\\v\{([a-zA-Z])\}', r'\1', text)  # \v{c} -> c
    text = re.sub(r"\\'\{([a-zA-Z])\}", r'\1', text)  # \'{e} -> e
    text = re.sub(r'\\"\{([a-zA-Z])\}', r'\1', text)   # \"{o} -> o
    text = re.sub(r'\\`\{([a-zA-Z])\}', r'\1', text)   # \`{a} -> a
    text = re.sub(r'\\\^\{([a-zA-Z])\}', r'\1', text)  # \^{e} -> e
    text = re.sub(r'\\~\{([a-zA-Z])\}', r'\1', text)   # \~{n} -> n
    text = re.sub(r'\\([a-zA-Z])', r'\1', text)        # \c -> c
    text = re.sub(r'[{}]', '', text)                   # Remove remaining braces
    text = re.sub(r'\\', '', text)                     # Remove backslashes
    return text

def format_author_list(author_str):
    """
    Format author list in APA style.
    Returns formatted author string.
    """
    if not author_str:
        return "Unknown"

    # Split by 'and'
    authors = re.split(r'\s+and\s+', author_str)
    formatted_authors = []

    for author in authors:
        # Split by comma: "Last, First Middle"
        parts = [p.strip() for p in author.split(',')]
        if len(parts) >= 2:
            last_name = clean_latex_text(parts[0])
            first_names = clean_latex_text(parts[1])
            # Get initials
            initials = '. '.join([name[0] for name in first_names.split() if name]) + '.'
            formatted_authors.append(f"{last_name}, {initials}")
        else:
            formatted_authors.append(clean_latex_text(parts[0]))

    # Format according to APA: up to 20 authors
    if len(formatted_authors) == 1:
        return formatted_authors[0]
    elif len(formatted_authors) == 2:
        return f"{formatted_authors[0]}, & {formatted_authors[1]}"
    elif len(formatted_authors) <= 20:
        return ', '.join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    else:
        # More than 20 authors: first 19, ..., last
        return ', '.join(formatted_authors[:19]) + f", ... {formatted_authors[-1]}"

def load_bibtex_references(bib_file):
    """
    Load BibTeX file and create a dictionary mapping citation keys to full APA-formatted citations.
    """
    with open(bib_file, 'r', encoding='utf-8') as f:
        bib_database = bibtexparser.load(f)

    citations = {}
    for entry in bib_database.entries:
        citation_key = entry.get('ID', '')
        entry_type = entry.get('ENTRYTYPE', 'article')

        # Format authors
        author_str = entry.get('author', '')
        authors = format_author_list(author_str)

        # Get year
        year = entry.get('year', 'n.d.')

        # Get title and clean it
        title = clean_latex_text(entry.get('title', 'Untitled'))
        # Remove extra curly braces from title
        title = re.sub(r'^\{|\}$', '', title)

        # Build citation based on entry type
        if entry_type == 'article':
            journal = clean_latex_text(entry.get('journal', ''))
            volume = entry.get('volume', '')
            number = entry.get('number', '')
            pages = entry.get('pages', '')

            citation = f"{authors} ({year}). {title}. {journal}"
            if volume:
                citation += f", {volume}"
            if number:
                citation += f"({number})"
            if pages:
                citation += f", {pages}"
            citation += "."

        elif entry_type == 'book':
            publisher = clean_latex_text(entry.get('publisher', ''))
            citation = f"{authors} ({year}). {title}. {publisher}."

        elif entry_type == 'inproceedings' or entry_type == 'conference':
            booktitle = clean_latex_text(entry.get('booktitle', ''))
            pages = entry.get('pages', '')
            citation = f"{authors} ({year}). {title}. In {booktitle}"
            if pages:
                citation += f" (pp. {pages})"
            citation += "."

        elif entry_type == 'phdthesis' or entry_type == 'mastersthesis':
            school = clean_latex_text(entry.get('school', ''))
            thesis_type = 'Doctoral dissertation' if entry_type == 'phdthesis' else 'Master\'s thesis'
            citation = f"{authors} ({year}). {title} [{thesis_type}]. {school}."

        else:
            # Generic format for other types
            citation = f"{authors} ({year}). {title}."

        citations[citation_key] = citation

    return citations

def format_citation_from_source(citation_str, bib_citations):
    """
    Convert citation string from database to full APA format using BibTeX data.

    Parameters:
    -----------
    citation_str : str
        Citation string from the database (e.g., "Rousseau et al. 2019")
    bib_citations : dict
        Dictionary mapping BibTeX keys to formatted citations
    """
    if pd.isna(citation_str):
        return "Unknown"

    # Split multiple citations
    citations = re.split(r',\s*', citation_str)
    formatted_citations = []

    for cite in citations:
        # Try to match with BibTeX entries
        matched = False
        cite_clean = cite.strip()

        # Extract author and year from citation string
        # Common patterns: "Author et al. YYYY", "Author YYYY", "Author & Author YYYY"
        year_match = re.search(r'\d{4}', cite_clean)
        if year_match:
            year = year_match.group()
            # Get first author's last name
            author = cite_clean.replace(year, '').strip()
            # Remove 'et al.', '&', and other extras
            first_author = re.split(r'\s+et\s+al\.?|\s+&\s+', author)[0].strip()

            # Search in BibTeX for matching entry
            for key, bib_cite in bib_citations.items():
                # Check if year matches and first author's last name is in citation
                if f"({year})" in bib_cite and first_author in bib_cite:
                    formatted_citations.append(bib_cite)
                    matched = True
                    break

        # If no match found, keep original
        if not matched:
            formatted_citations.append(cite_clean)

    return ' '.join(formatted_citations)

def parse_date(row):
    """
    Parse date from Day, Month, Year, Hour, Minute columns.
    Returns datetime object or None if date cannot be parsed.
    """
    try:
        year = int(row['Year']) if pd.notna(row['Year']) else None
        month = int(row['Month']) if pd.notna(row['Month']) else None
        day = int(row['Day']) if pd.notna(row['Day']) else 15  # Default to mid-month if day is missing
        hour = int(row['Hour']) if pd.notna(row['Hour']) else 0
        minute = int(row['Minute']) if pd.notna(row['Minute']) else 0

        if year is None or month is None:
            return None

        return datetime(year, month, day, hour, minute)
    except (ValueError, OverflowError):
        return None

def create_netcdf_for_station(location_id, station_data, location_info, citation_info):
    """
    Create a CF-compliant NetCDF file for a single station.

    Parameters:
    -----------
    location_id : str
        The location ID for the station
    station_data : DataFrame
        Combined data with columns: datetime, Discharge_m3_s, TSS_mg_L
        Discharge_m3_s may contain NaN for TSS-only records (written as _FillValue).
    location_info : dict
        Station metadata (Lat_deg, Lon_deg, Elevation_masl, Country, Observations)
    citation_info : str
        Citation string
    """
    # Skip if required fields are missing
    if pd.isna(location_info['Lat_deg']) or pd.isna(location_info['Lon_deg']):
        print(f"  Skipping {location_id}: Missing coordinates")
        return False

    if station_data.empty:
        print(f"  Skipping {location_id}: No data")
        return False

    # Sort by datetime
    station_data = station_data.sort_values('datetime')

    # Create NetCDF file with GloRiSe_ prefix
    filename = OUTPUT_DIR / f'GloRiSe_{location_id}.nc'
    nc = Dataset(filename, 'w', format='NETCDF4')

    try:
        # Create dimensions
        time_dim = nc.createDimension('time', len(station_data))
        lat_dim = nc.createDimension('latitude', 1)
        lon_dim = nc.createDimension('longitude', 1)

        # Create coordinate variable for time
        times = nc.createVariable('time', 'f8', ('time',))
        times.units = 'days since 1970-01-01 00:00:00'
        times.calendar = 'gregorian'
        times.standard_name = 'time'
        times.long_name = 'time'
        times.axis = 'T'

        # Convert datetimes to numeric values
        time_values = date2num(station_data['datetime'].tolist(),
                               units=times.units,
                               calendar=times.calendar)
        times[:] = time_values

        # Create latitude coordinate variable
        lat = nc.createVariable('latitude', 'f4', ('latitude',))
        lat.standard_name = 'latitude'
        lat.long_name = 'latitude'
        lat.units = 'degrees_north'
        lat.axis = 'Y'
        lat[:] = float(location_info['Lat_deg'])

        # Create longitude coordinate variable
        lon = nc.createVariable('longitude', 'f4', ('longitude',))
        lon.standard_name = 'longitude'
        lon.long_name = 'longitude'
        lon.units = 'degrees_east'
        lon.axis = 'X'
        lon[:] = float(location_info['Lon_deg'])

        # --- Discharge: keep source value when present; write _FillValue when missing ---
        discharge_vals = station_data['Discharge_m3_s'].values.astype(np.float32).copy()
        # Replace NaN (Q-missing records) with _FillValue
        nan_mask = pd.isna(station_data['Discharge_m3_s'].values)
        discharge_vals = np.where(nan_mask, np.float32(-9999.0), discharge_vals)

        discharge = nc.createVariable('Discharge_m3_s', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
        discharge.standard_name = 'water_volume_transport_in_river_channel'
        discharge.long_name = 'River discharge'
        discharge.units = 'm3 s-1'
        discharge.coordinates = 'time latitude longitude'
        discharge.comment = ('Source-reported discharge. '
                             'Set to _FillValue (-9999) when discharge was not reported '
                             'alongside the suspended-sediment observation.')
        discharge[:, 0, 0] = discharge_vals

        # --- TSS/SSC: always present for valid sediment-eligible records ---
        # For GloRiSe SS (suspended-sediment) samples the database field TSS_mg_L
        # represents suspended-sediment concentration (SSC).  We keep the source
        # column name for provenance transparency.
        tss_vals = station_data['TSS_mg_L'].values.astype(np.float32)

        tss = nc.createVariable('TSS_mg_L', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
        tss.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        tss.long_name = 'Total Suspended Sediment concentration'
        tss.units = 'mg L-1'
        tss.coordinates = 'time latitude longitude'
        tss.comment = ('Source-reported suspended-sediment concentration (TSS/SSC). '
                       'For SS sample type this is the suspended sediment concentration '
                       'as reported by the data originator.')
        tss[:, 0, 0] = tss_vals

        # Add global attributes
        nc.title = f'River sediment and discharge data for station {location_id}'
        nc.institution = 'GloRiSe - Global River Sediment Database'
        nc.source = 'GloRiSe v1.1'
        nc.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        nc.references = citation_info
        nc.Conventions = 'CF-1.8'

        # Station metadata
        nc.location_id = location_id
        nc.latitude = float(location_info['Lat_deg'])
        nc.longitude = float(location_info['Lon_deg'])

        if pd.notna(location_info['Elevation_masl']):
            nc.elevation = float(location_info['Elevation_masl'])

        if pd.notna(location_info['Country']):
            nc.country = str(location_info['Country'])

        if pd.notna(location_info['Observations']):
            nc.observations = str(location_info['Observations'])

        # Count how many records have Q
        n_q_present = int(np.sum(discharge_vals != -9999.0))
        n_total = len(discharge_vals)
        print(f"  Created {filename.name}: {n_total} records ({n_q_present} with Q, "
              f"{n_total - n_q_present} SSC-only)")
        return True

    finally:
        nc.close()

def main():
    """Main processing function."""
    print("Loading data files...")

    for path in (REF_FILE, LOC_FILE, ID_FILE, ME_FILE, BIB_FILE):
        require_existing_file(path, description=f"GloRiSe source file {path.name}")

    # Load BibTeX references
    print("Loading BibTeX references...")
    bib_citations = load_bibtex_references(BIB_FILE)
    print(f"Loaded {len(bib_citations)} BibTeX entries")

    # Read all data files
    df_ref = pd.read_excel(REF_FILE)
    df_loc = pd.read_excel(LOC_FILE)
    df_id = pd.read_excel(ID_FILE)
    df_me = pd.read_excel(ME_FILE)

    print(f"Loaded {len(df_ref)} references")
    print(f"Loaded {len(df_loc)} locations")
    print(f"Loaded {len(df_id)} sample IDs")
    print(f"Loaded {len(df_me)} measurements")

    # Merge ID data with ME data
    print("\nMerging data...")
    df_merged = pd.merge(df_id, df_me, on='Sample_ID', how='inner')
    print(f"Merged dataset has {len(df_merged)} records")

    # =========================================================================
    # FILTERING (FIXED 2026-08):
    #   OLD: required BOTH Discharge_m3_s AND TSS_mg_L non-null + Sampletype=="SS"
    #        -> this dropped TSS-only observations
    #   NEW: only requires TSS_mg_L non-null + Sampletype=="SS"
    #        -> TSS-only records are preserved; Q is written as _FillValue when absent
    # =========================================================================

    # Step A: All SS-sample records that have valid TSS (the sediment-eligible set)
    df_filtered = df_merged[
        (pd.notna(df_merged['TSS_mg_L'])) &
        (df_merged['Sampletype'] == "SS")
    ].copy()

    # Step B: Subset that also has Discharge (for audit comparison only)
    df_q_tss = df_filtered[pd.notna(df_filtered['Discharge_m3_s'])]

    # ---- AUDIT ----
    n_all_ss = len(df_filtered)
    n_q_tss = len(df_q_tss)
    n_tss_only = n_all_ss - n_q_tss
    n_total_merged = len(df_merged)

    n_stations_all = df_filtered['Location_ID'].nunique()
    n_stations_q_tss = df_q_tss['Location_ID'].nunique()
    n_stations_tss_only = n_stations_all - n_stations_q_tss

    print(f"\n{'='*60}")
    print(f"AUDIT — Sediment-eligible filtering")
    print(f"{'='*60}")
    print(f"  Total merged records:               {n_total_merged:>8}")
    print(f"  All SS records w/ valid TSS:        {n_all_ss:>8}")
    print(f"    of which Q+TSS paired:            {n_q_tss:>8}")
    print(f"    of which TSS-only (Q missing):    {n_tss_only:>8}")
    print(f"  Unique locations (all SS+TSS):      {n_stations_all:>8}")
    print(f"    of which have >=1 Q+TSS record:   {n_stations_q_tss:>8}")
    print(f"    of which TSS-only only:           {n_stations_tss_only:>8}")
    print(f"{'='*60}\n")

    # Parse dates on the full sediment-eligible dataframe
    print("Parsing dates...")
    df_filtered['datetime'] = df_filtered.apply(parse_date, axis=1)

    # Filter out records without valid dates
    n_before_date_filter = len(df_filtered)
    df_filtered = df_filtered[pd.notna(df_filtered['datetime'])]
    n_date_dropped = n_before_date_filter - len(df_filtered)
    if n_date_dropped > 0:
        print(f"  Dropped {n_date_dropped} records with invalid/unparseable dates")
    print(f"Records with valid dates: {len(df_filtered)}")

    # unique_locations derived from sediment-eligible records (NOT Q+TSS pairs)
    unique_locations = df_filtered['Location_ID'].unique()
    print(f"\nProcessing {len(unique_locations)} unique locations...")

    # Identify TSS-only locations (zero Q records) for regression test
    tss_only_loc_ids = set()
    for loc_id in unique_locations:
        loc_data = df_filtered[df_filtered['Location_ID'] == loc_id]
        if not loc_data['Discharge_m3_s'].notna().any():
            tss_only_loc_ids.add(loc_id)

    print(f"  (of which {len(tss_only_loc_ids)} are TSS-only locations with no Q records)\n")

    # Process each location
    processed_count = 0
    skipped_count = 0
    tss_only_processed = 0  # counter for regression test

    for location_id in unique_locations:
        # Get all sediment-eligible data for this location
        location_data = df_filtered[df_filtered['Location_ID'] == location_id].copy()

        # Get location metadata
        loc_info = df_loc[df_loc['Location_ID'] == location_id]
        if loc_info.empty:
            print(f"  Skipping {location_id}: No location metadata")
            skipped_count += 1
            continue

        loc_info = loc_info.iloc[0]

        # Get citation and format it
        citation_raw = loc_info['Citation'] if pd.notna(loc_info['Citation']) else 'Unknown'
        citation = format_citation_from_source(citation_raw, bib_citations)

        # Prepare data for NetCDF — keep ALL TSS records
        # Discharge_m3_s may be NaN for TSS-only records (handled in create_netcdf)
        station_data = location_data[['datetime', 'Discharge_m3_s', 'TSS_mg_L']].copy()

        # Create NetCDF file
        if create_netcdf_for_station(location_id, station_data, loc_info, citation):
            processed_count += 1
            if location_id in tss_only_loc_ids:
                tss_only_processed += 1
        else:
            skipped_count += 1
            if location_id in tss_only_loc_ids:
                print(f"  *** WARNING: REGRESSION RISK — TSS-only location {location_id} was skipped!")

    # ---- FINAL AUDIT & REGRESSION TEST ----
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"  Successfully processed: {processed_count} stations")
    print(f"  Skipped:                {skipped_count} stations")
    print(f"  Output directory:       {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Regression test: verify TSS-only locations produced output
    print(f"\n{'='*60}")
    print(f"REGRESSION TEST — TSS-only station NetCDF generation")
    print(f"{'='*60}")
    if len(tss_only_loc_ids) == 0:
        print("  SKIPPED: No TSS-only locations found in dataset")
        print("    (all locations have at least one Q+TSS record)")
    else:
        print(f"  TSS-only locations in source:  {len(tss_only_loc_ids)}")
        print(f"  TSS-only locations processed:  {tss_only_processed}")
        if tss_only_processed > 0:
            print(f"  PASSED: {tss_only_processed}/{len(tss_only_loc_ids)} "
                  f"TSS-only locations generated NetCDF successfully")
        else:
            print(f"  FAILED: 0/{len(tss_only_loc_ids)} TSS-only locations "
                  f"generated NetCDF — regression detected!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
