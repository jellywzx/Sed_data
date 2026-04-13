#!/usr/bin/env python
"""
Generate CF-compliant NetCDF files from GloRiSe sediment database.

This script reads data from multiple Excel files, integrates them by station,
and generates one NetCDF file per station containing Discharge and TSS time series.
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
    text = re.sub(r"\\\'\{([a-zA-Z])\}", r'\1', text)  # \'{e} -> e
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

        # Create Discharge variable with time, lat, lon dimensions
        discharge = nc.createVariable('Discharge_m3_s', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
        discharge.standard_name = 'water_volume_transport_in_river_channel'
        discharge.long_name = 'River discharge'
        discharge.units = 'm3 s-1'
        discharge.coordinates = 'time latitude longitude'
        # Reshape data to (time, 1, 1)
        discharge[:, 0, 0] = station_data['Discharge_m3_s'].values

        # Create TSS variable with time, lat, lon dimensions
        tss = nc.createVariable('TSS_mg_L', 'f4', ('time', 'latitude', 'longitude'), fill_value=-9999.0)
        tss.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        tss.long_name = 'Total Suspended Sediment concentration'
        tss.units = 'mg L-1'
        tss.coordinates = 'time latitude longitude'
        # Reshape data to (time, 1, 1)
        tss[:, 0, 0] = station_data['TSS_mg_L'].values

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

        print(f"  Created {filename.name}: {len(station_data)} records")
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

    # Filter records that have both Discharge and TSS
    df_filtered = df_merged[
        (pd.notna(df_merged['Discharge_m3_s'])) &
        (pd.notna(df_merged['TSS_mg_L'])) &
        (df_merged['Sampletype'] == "SS")
    ].copy()
    print(f"Records with both Discharge and TSS: {len(df_filtered)}")

    # Parse dates
    print("\nParsing dates...")
    df_filtered['datetime'] = df_filtered.apply(parse_date, axis=1)

    # Filter out records without valid dates
    df_filtered = df_filtered[pd.notna(df_filtered['datetime'])]
    print(f"Records with valid dates: {len(df_filtered)}")

    # Get unique locations
    unique_locations = df_filtered['Location_ID'].unique()
    print(f"\nProcessing {len(unique_locations)} unique locations...")

    # Process each location
    processed_count = 0
    skipped_count = 0

    for location_id in unique_locations:
        # Get all data for this location
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

        # Prepare data for NetCDF
        station_data = location_data[['datetime', 'Discharge_m3_s', 'TSS_mg_L']].copy()

        # Create NetCDF file
        if create_netcdf_for_station(location_id, station_data, loc_info, citation):
            processed_count += 1
        else:
            skipped_count += 1

    print(f"\n" + "="*60)
    print(f"Processing complete!")
    print(f"  Successfully processed: {processed_count} stations")
    print(f"  Skipped: {skipped_count} stations")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    main()
