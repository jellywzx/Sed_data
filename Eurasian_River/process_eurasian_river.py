#!/usr/bin/env python3
"""
This script processes the Eurasian River dataset. It performs the following steps:
1.  Reads station metadata from 'station_locations.xls' and 'meta.xlsx'.
2.  Reads discharge data from '.dat' and '.txt' files.
3.  Reads sediment flux data from 'Sediment_Flux_Data.xls'.
4.  Merges discharge and sediment data for each station.
5.  Performs quality control and flags the data.
6.  Creates CF-1.8 compliant NetCDF files for each station.
7.  Generates a summary CSV file with metadata for all stations.
"""

import pandas as pd
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import os
import re
import glob
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl,
    apply_quality_flag_array,        
    apply_hydro_qc_with_provenance, 
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
)
# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SOURCE_DIR = os.path.join(BASE_DIR, "Source", "Eurasian_River")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output_r", "monthly", "Eurasian_River", "qc")
SCRIPT_DIR = os.path.join(BASE_DIR, "Script", "Dataset", "Eurasian_River")

# --- Helper Functions ---
def _count_flags(arr, values=(0, 1, 2, 3, 9)):
    a = np.asarray(arr, dtype=np.int8)
    return {int(v): int(np.sum(a == np.int8(v))) for v in values}

def _qc_counts_flat(df):
    """
    Return flat QC counters for final flags (and step flags if present).
    """
    out = {}
    out["QC_n_days"] = int(len(df))

    # final flags
    for var in ["Q", "SSC", "SSL"]:
        col = f"{var}_flag"
        if col in df.columns:
            c = _count_flags(df[col].values)
            out[f"{var}_final_good"]    = c.get(0, 0)
            out[f"{var}_final_est"]     = c.get(1, 0)
            out[f"{var}_final_suspect"] = c.get(2, 0)
            out[f"{var}_final_bad"]     = c.get(3, 0)
            out[f"{var}_final_missing"] = c.get(9, 0)
        else:
            out[f"{var}_final_good"] = out[f"{var}_final_est"] = 0
            out[f"{var}_final_suspect"] = out[f"{var}_final_bad"] = 0
            out[f"{var}_final_missing"] = int(len(df))

    # step/provenance (only if columns exist)
    step_cols = [
        ("Q_flag_qc1_physical",   "Q_qc1",   (0, 3, 9)),
        ("Q_flag_qc2_log_iqr",    "Q_qc2",   (0, 2, 8, 9)),
        ("Q_flag_qc3_ssc_q",      "Q_qc3",   (0, 2, 8, 9)),
        ("SSC_flag_qc1_physical", "SSC_qc1", (0, 3, 9)),
        ("SSC_flag_qc2_log_iqr",  "SSC_qc2", (0, 2, 8, 9)),
        ("SSC_flag_qc3_ssc_q",    "SSC_qc3", (0, 2, 8, 9)),
        ("SSL_flag_qc1_physical", "SSL_qc1", (0, 3, 9)),
        ("SSL_flag_qc2_log_iqr",  "SSL_qc2", (0, 2, 8, 9)),
        ("SSL_flag_qc3_from_ssc_q","SSL_qc3",(0, 2, 8, 9)),
    ]
    for col, prefix, values in step_cols:
        if col in df.columns:
            c = _count_flags(df[col].values, values=values)
            out[f"{prefix}_pass"]        = c.get(0, 0)
            out[f"{prefix}_suspect"]     = c.get(2, 0)
            out[f"{prefix}_not_checked"] = c.get(8, 0)
            out[f"{prefix}_missing"]     = c.get(9, 0)

    return out

def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_tool_qc(time, Q, SSC, SSL, station_id, station_name, plot_dir=None):
    """
    tool.py end-to-end QC pipeline WITH step-level provenance flags.

    Returns:
        qc (dict): trimmed arrays + final flags + step flags (+optional bounds)
        qc_report (dict): station-level summary counters
    """
    # --- force 1D ---
    time = np.atleast_1d(np.asarray(time)).reshape(-1)
    Q    = np.atleast_1d(np.asarray(Q, dtype=float)).reshape(-1)
    SSC  = np.atleast_1d(np.asarray(SSC, dtype=float)).reshape(-1)
    SSL  = np.atleast_1d(np.asarray(SSL, dtype=float)).reshape(-1)

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None, None
    time, Q, SSC, SSL = time[:n], Q[:n], SSC[:n], SSL[:n]

    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        # 重要：按你“单位换算仍算 independent”的规则
        Q_is_independent=True,
        SSC_is_independent=False,   # 这里 SSC = SSL/(Q*0.0864) 是派生量
        SSL_is_independent=True,    # SSL_kg_s -> SSL(ton/day) 属于单位换算，仍 True
        ssl_is_derived_from_q_ssc=False,

        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )
    if qc is None:
        return None, None

    # --- qc_report: station-level final flag stats ---
    def _count(arr, values=(0, 1, 2, 3, 9)):
        arr = np.asarray(arr, dtype=np.int8)
        return {int(v): int(np.sum(arr == np.int8(v))) for v in values}

    qc_report = {
        "Source_ID": station_id,
        "station_name": station_name,
        "n_total": int(len(qc["Q_flag"])),
        "Q_flag_counts": _count(qc["Q_flag"]),
        "SSC_flag_counts": _count(qc["SSC_flag"]),
        "SSL_flag_counts": _count(qc["SSL_flag"]),
    }

    return qc, qc_report

def read_discharge_data():
    """Reads all discharge data from .dat and .txt files."""
    discharge_data = {}
    files = glob.glob(os.path.join(SOURCE_DIR, "*.dat")) + glob.glob(os.path.join(SOURCE_DIR, "*.dat.txt")) + glob.glob(os.path.join(SOURCE_DIR, "*.txt"))

    for f in files:
        if 'readme' in f:
            continue

        river_name, meta, data = parse_discharge_file(f)

        if river_name:
            if river_name not in discharge_data:
                discharge_data[river_name] = {'meta': meta, 'data': {}}
            discharge_data[river_name]['data'].update(data)
            # Add meta if it's not already there
            if not discharge_data[river_name]['meta']:
                 discharge_data[river_name]['meta'] = meta

    return discharge_data

def print_qc_summary(river_name, station_id, n, skipped_log_iqr, skipped_ssc_q, q, q_flag, ssc, ssc_flag, ssl, ssl_flag, nc_path):
    print(f"\nProcessing: {river_name} ({station_id}) +")
    if skipped_log_iqr:
        print(f"  - Sample size = {n} < 5, log-IQR statistical QC skipped.")
    if skipped_ssc_q:
        print(f"  - Sample size = {n} < 5, SSC-Q consistency check and diagnostic plot skipped.")
    print(f"  - Created: {nc_path}")
    print(f"    Q:   {q:.2f} m3/s (flag={int(q_flag)})")
    print(f"    SSC: {ssc:.2f} mg/L (flag={int(ssc_flag)})")
    print(f"    SSL: {ssl:.2f} ton/day (flag={int(ssl_flag)})")

def parse_discharge_file(filepath):
    """Parses a discharge file (.dat or .txt)."""
    filename = os.path.basename(filepath)
    river_name = ""
    metadata = {}
    data = {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    if filename.endswith('.dat') or filename.endswith('.dat.txt'):
        river_name = filename.split('.')[0].replace('Eurasian_River_RUS-', '').replace('_', ' ')
        if 'Kolyma 1' in river_name: river_name = 'Kolyma_1'
        if 'Kolyma 2' in river_name: river_name = 'Kolyma_2'
        if 'Yana 1' in river_name: river_name = 'Yana_1'
        if 'Yana 2' in river_name: river_name = 'Yana_2'
        if 'S. Dvina' in river_name: river_name = 'Severnaya Dvina'

        metadata, data = parse_dat_content(lines)

    elif filename.endswith('.txt') and (filename.startswith('Eurasian_River_RUS-Omoloy') or filename.startswith('Eurasian_River_RUS-Pur')):
        river_name = filename.split('.')[0].replace('Eurasian_River_RUS-', '')
        metadata, data = parse_special_txt_content(lines)

    return river_name, metadata, data

def parse_dat_content(lines):
    """Parses the content of a .dat file."""
    metadata = {}
    data = {}
    data_start_line = 0

    for i, line in enumerate(lines):
        if 'station code:' in line.lower():
            metadata['station_code'] = lines[i + 1].strip()
        elif 'r-arcticnet id:' in line.lower():
            metadata['station_id'] = lines[i + 1].strip()
        elif 'latitude:' in line.lower():
            try:
                metadata['latitude'] = float(lines[i + 1].strip())
            except (ValueError, IndexError):
                pass
        elif 'longitude:' in line.lower():
            try:
                metadata['longitude'] = float(lines[i + 1].strip())
            except (ValueError, IndexError):
                pass
        elif 'drainage area:' in line.lower():
            match = re.search(r'([\d.]+)', lines[i + 1])
            if match:
                metadata['drainage_area'] = float(match.group(1))

        if metadata.get('station_code') and line.strip().startswith(metadata.get('station_code')):
            parts = line.strip().split()
            if len(parts) >= 2 and '-' in parts[1]:
                data_start_line = i
                break

    for line in lines[data_start_line:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                year, month = map(int, parts[1].split('-'))
                discharge = float(parts[2])
                data[(year, month)] = discharge
            except (ValueError, IndexError):
                continue
    return metadata, data

def parse_special_txt_content(lines):
    """Parses the content of special .txt files (Omoloy, Pur)."""
    metadata = {}
    data = {}
    for i, line in enumerate(lines):
        if i == 0:  # Latitude
            metadata['latitude'] = float(line.split('°')[0].split(':')[1].strip().replace('Â', ''))
        elif i == 1:  # Longitude
            metadata['longitude'] = float(line.split('°')[0].split(':')[1].strip().replace('Â', ''))
        elif i == 4:  # Drainage Area
            metadata['drainage_area'] = float(line.split(':')[1].split('km2')[0].strip())
        elif i == 7:  # Gauge Altitude
            metadata['altitude'] = float(line.split(':')[1].strip())
        elif i >= 10:
            break

    for line in lines[11:]:
        if line.strip() and not line.startswith('Point_ID'):
            parts = line.strip().split('\t')
            if len(parts) > 3:
                try:
                    year = int(parts[2].strip())
                    for month in range(1, 13):
                        if len(parts) > month + 2:
                            discharge_val = parts[month + 2].strip()
                            if discharge_val:
                                discharge = float(discharge_val)
                                data[(year, month)] = discharge
                except (ValueError, IndexError):
                    continue
    return metadata, data

def read_sediment_data():
    """Reads sediment flux data from the Excel file."""
    sediment_path = os.path.join(SOURCE_DIR, "Sediment_Flux_Data.xls")
    xl = pd.ExcelFile(sediment_path)
    sediment_data = {}
    sheet_to_river = {
        'Alazeya': 'Alazeya', 'Anabar': 'Anabar', 'Indigirka': 'Indigirka',
        'Kolyma1': 'Kolyma_1', 'Kolyma2': 'Kolyma_2', 'Lena': 'Lena',
        'Mezen': "Mezen'", 'Ob': 'Ob', 'Olenek': 'Olenek',
        'Omoloy': 'Omoloy', 'Onega': 'Onega', 'Pechora': 'Pechora',
        'Pur': 'Pur', 'S. Dvina': 'Severnaya Dvina', 'Taz': 'Taz',
        'Yana1': 'Yana_1', 'Yana2': 'Yana_2', 'Yenisey': 'Yenisey'
    }

    for sheet_name in xl.sheet_names:
        river_name = sheet_to_river.get(sheet_name, sheet_name)
        df = xl.parse(sheet_name, skiprows=2)
        df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df = df[df['Year'] != 'Year'] # Remove header row
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)

        sediment_data[river_name] = df

    return sediment_data

def main():
    """Main processing function."""
    create_output_dir()
    DIAG_DIR = os.path.join(OUTPUT_DIR, "diagnostic")
    os.makedirs(DIAG_DIR, exist_ok=True)

    discharge_data = read_discharge_data()
    sediment_data = read_sediment_data()
    summary_data = []
    qc_reports = []

    # River name mapping
    river_map = {
        "Mezen'": "Mezen"
    }

    for river_name, dis_data in discharge_data.items():
        print(f"Processing {river_name}...")

        # Find corresponding sediment data
        sed_df = sediment_data.get(river_name)
        if sed_df is None:
            # Try mapping
            mapped_river_name = river_map.get(river_name)
            if mapped_river_name:
                sed_df = sediment_data.get(mapped_river_name)

        if sed_df is None:
            print(f"  - Sediment data not found for {river_name}. Skipping.")
            continue

        # Combine data
        dis_years = [y for y, m in dis_data['data'].keys()]
        if not dis_years:
            continue
        all_years = sorted(list(set(dis_years) | set(sed_df['Year'])))
        if not all_years:
            continue

        start_year = min(all_years)
        end_year = max(all_years)

        # Create a dataframe to hold the merged data
        time_index = pd.to_datetime([f'{y}-{m}-15' for y in range(start_year, end_year + 1) for m in range(1, 13)])
        df = pd.DataFrame(index=time_index)

        # Add discharge data
        df['Q'] = np.nan
        for (year, month), value in dis_data['data'].items():
            df.loc[f'{year}-{month}-15', 'Q'] = value

        # Add sediment flux data
        df['SSL_kg_s'] = np.nan
        for _, row in sed_df.iterrows():
            year = row['Year']
            for i, month_name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                month = i + 1
                if pd.notna(row[month_name]):
                    df.loc[f'{year}-{month}-15', 'SSL_kg_s'] = row[month_name]

        # --- Calculations ---
        # SSL (ton/day) = SSL (kg/s) * 86400 / 1000
        df['SSL'] = df['SSL_kg_s'] * 86.4

        # SSC (mg/L) = SSL (ton/day) / (Q (m3/s) * 0.0864)
        df['SSC'] = df['SSL'] / (df['Q'] * 0.0864)

        # --- Keep only overlapping data (intersection) ---
        df = df.dropna(subset=['Q', 'SSL_kg_s'], how='any')

        if df.empty:
            print(f"  - No overlapping data for {river_name}. Skipping.")
            continue
        # 先确定 station_id（后面 QC/print 都要用）
        station_id = dis_data['meta'].get('station_id', dis_data['meta'].get('station_code', river_name))
        station_name = dis_data['meta'].get('station_name', river_name)

        # time base（你注释写的是 since 1970-01-01）
        base = pd.Timestamp("1970-01-01")

        # --- Quality Control and Flagging ---  
        # 1) QC1-array：显式调用（用于 qc=None 的 fallback，同时也写成 step flag）
        Q_flag_qc1   = apply_quality_flag_array(df["Q"].values,   "Q")
        SSC_flag_qc1 = apply_quality_flag_array(df["SSC"].values, "SSC")
        SSL_flag_qc1 = apply_quality_flag_array(df["SSL"].values, "SSL")

        # 2) time days since 1970-01-01
        time_days = ((df.index - base) / np.timedelta64(1, "D")).astype(float).to_numpy()

        # 3) tool.py pipeline
        qc, qc_report = apply_tool_qc(
            time=time_days,
            Q=df["Q"].values,
            SSC=df["SSC"].values,
            SSL=df["SSL"].values,
            station_id=station_id,
            station_name=station_name,
            plot_dir=os.path.join(OUTPUT_DIR, "diagnostic_plots"),
        )

        if qc is None:
            # 退回 QC1（同时补齐 step flags，缺失的 step 用 9）
            df["Q_flag"]   = Q_flag_qc1
            df["SSC_flag"] = SSC_flag_qc1
            df["SSL_flag"] = SSL_flag_qc1

            df["Q_flag_qc1_physical"]   = Q_flag_qc1
            df["SSC_flag_qc1_physical"] = SSC_flag_qc1
            df["SSL_flag_qc1_physical"] = SSL_flag_qc1

            # qc2/qc3 不跑 -> 统一 9
            for col in [
                "Q_flag_qc2_log_iqr", "Q_flag_qc3_ssc_q_envelope",
                "SSC_flag_qc2_log_iqr", "SSC_flag_qc3_ssc_q_envelope",
                "SSL_flag_qc2_log_iqr", "SSL_flag_qc3_propagated_from_ssc_q",
            ]:
                df[col] = np.int8(9)

            ssc_q_bounds = None
        else:
            # qc 已经 trimmed(valid_time)，用 qc["time"] 找回 df 索引
            qc_time = base + pd.to_timedelta(qc["time"], unit="D")
            df = df.loc[qc_time].copy()

            # final flags
            df["Q_flag"]   = qc["Q_flag"]
            df["SSC_flag"] = qc["SSC_flag"]
            df["SSL_flag"] = qc["SSL_flag"]
            # ✅ 把 step/provenance flags 也落到 df 里（没有就先跳过）
            for k in [
                "Q_flag_qc1_physical", "Q_flag_qc2_log_iqr", "Q_flag_qc3_ssc_q",
                "SSC_flag_qc1_physical","SSC_flag_qc2_log_iqr","SSC_flag_qc3_ssc_q",
                "SSL_flag_qc1_physical","SSL_flag_qc2_log_iqr","SSL_flag_qc3_from_ssc_q",
            ]:
                if k in qc:
                    df[k] = qc[k]

            # step/provenance flags：把 qc 里所有 *_flag_qc* 都塞回 df
            for k, v in qc.items():
                if isinstance(v, np.ndarray) and ("flag_qc" in k):
                    df[k] = v.astype(np.int8)

            ssc_q_bounds = qc.get("ssc_q_bounds", None)

            # 收集 qc_report（后面要写 qc_results_summary.csv）
            qc_reports.append(qc_report)

        print(
            f"[QC] {station_id} QC1(good cnt): "
            f"Q={(Q_flag_qc1==0).sum()}, SSC={(SSC_flag_qc1==0).sum()}, SSL={(SSL_flag_qc1==0).sum()} | "
            f"Final(good cnt): Q={(df['Q_flag'].to_numpy()==0).sum()}, "
            f"SSC={(df['SSC_flag'].to_numpy()==0).sum()}, SSL={(df['SSL_flag'].to_numpy()==0).sum()}"
        )


        # --- Time Trimming ---
        df.dropna(how='all', subset=['Q', 'SSL', 'SSC'], inplace=True)
        if df.empty:
            print(f"  - No overlapping data for {river_name}. Skipping.")
            continue

        # --- Create NetCDF ---
        station_id = dis_data['meta'].get('station_id', dis_data['meta'].get('station_code', river_name))
        nc_filename = os.path.join(OUTPUT_DIR, f"Eurasian_River_{station_id}.nc")
        with Dataset(nc_filename, 'w', format='NETCDF4') as nc:
            # Dimensions
            nc.createDimension('time', None)
            nc.createDimension('lat', 1)
            nc.createDimension('lon', 1)

            # Coordinates
            time = nc.createVariable('time', 'f8', ('time',))
            time.units = f"days since {df.index.min().strftime('%Y-%m-%d')} 00:00:00"
            time.calendar = 'gregorian'
            time.standard_name = 'time'
            time.long_name = 'time'
            time[:] = (df.index - df.index.min()).days.values

            lat = nc.createVariable('lat', 'f4', ('lat',))
            lat.units = 'degrees_north'
            lat.standard_name = 'latitude'
            lat.long_name = 'station latitude'
            lat[:] = dis_data['meta'].get('latitude', np.nan)

            lon = nc.createVariable('lon', 'f4', ('lon',))
            lon.units = 'degrees_east'
            lon.standard_name = 'longitude'
            lon.long_name = 'station longitude'
            lon[:] = dis_data['meta'].get('longitude', np.nan)

            # Global Attributes
            nc.Conventions = "CF-1.8, ACDD-1.3"
            nc.title = "River sediment flux data for station RUS-Anabar"
            nc.insitiution = "Eurasian Arctic River Database"
            nc.dataset_name = "Eurasian River Historical Sediment Flux Data"
            nc.station_name = dis_data['meta'].get('station_name', river_name)
            nc.river_name = river_name
            nc.source_id = station_id
            nc.type = "In-situ station data"
            nc.temporal_resolution = "monthly"
            nc.temporal_span = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
            nc.geographic_coverage = f"{river_name} River Basin, Russia"
            # nc.variables_provided = "altitude, upstream_area, Q, SSC, SSL, station_name, river_name, Source_ID"
            nc.reference1 = "Holmes, R. M., McClelland, J. W., Peterson, B. J., Shiklomanov, I. A., Shiklomanov, A. I., Zhulidov, A. V., ... & Bobrovitskaya, N. N. (2002). A circumpolar perspective on fluvial sediment flux to the Arctic Ocean. Global biogeochemical cycles, 16(4), 45-1."
            nc.reference2 = "Holmes, R., Peterson, B. (2009). Eurasian River Historical Nutrient and Sediment Flux Data. Version 1.0. NSF NCAR Earth Observing Laboratory. https://doi.org/10.5065/D6F769PB. Accessed 17 Oct 2025."
            nc.comment = "Original data: Monthly sediment flux data. TSS concentration and discharge data not available for this dataset. Processed: Sediment load calculated from sediment flux data. SSC derived from: SSC = sediment_load / (discharge × 86.4)"
            nc.discharge_data_source = "https://www.r-arcticnet.sr.unh.edu/v4.0/ViewPoint.pl?Point=5951"
            nc.sediment_data_source = "https://doi.org/10.5065/D6F769PB"
            nc.creator_name = "Zhongwang Wei"
            nc.creator_email = "weizhw6@mail.sysu.edu.cn"
            nc.creator_institution = "Sun Yat-sen University, China"
            nc.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by process_eurasian_river.py"

            # Data Variables
            q_var = nc.createVariable('Q', 'f4', ('time', 'lat', 'lon'), fill_value=FILL_VALUE_FLOAT)
            q_var.units = 'm3 s-1'
            q_var.long_name = 'River Discharge'
            q_var.standard_name = 'river_discharge'
            q_var.ancillary_variables = 'Q_flag'
            q_var[:,0,0] = df['Q'].fillna(-9999.0).values

            q_flag_var = nc.createVariable('Q_flag', 'b', ('time', 'lat', 'lon'),fill_value=FILL_VALUE_INT)
            q_flag_var.long_name = 'Quality flag for River Discharge'
            q_flag_var.flag_values = [0, 1, 2, 3, 9]
            q_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
            q_flag_var[:,0,0] = df['Q_flag'].values

            ssc_var = nc.createVariable('SSC', 'f4', ('time', 'lat', 'lon'), fill_value=FILL_VALUE_FLOAT)
            ssc_var.units = 'mg L-1'
            ssc_var.long_name = 'Suspended Sediment Concentration'
            ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
            ssc_var.ancillary_variables = 'SSC_flag'
            ssc_var[:,0,0] = df['SSC'].fillna(FILL_VALUE_FLOAT).values

            ssc_flag_var = nc.createVariable('SSC_flag', 'b', ('time', 'lat', 'lon'))
            ssc_flag_var.long_name = 'Quality flag for Suspended Sediment Concentration'
            ssc_flag_var.flag_values = [0, 1, 2, 3, 9]
            ssc_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
            ssc_flag_var[:,0,0] = df['SSC_flag'].values

            ssl_var = nc.createVariable('SSL', 'f4', ('time', 'lat', 'lon'), fill_value=-9999.0)
            ssl_var.units = 'ton day-1'
            ssl_var.long_name = 'Suspended Sediment Load'
            ssl_var.standard_name = 'suspended_sediment_load'
            ssl_var.ancillary_variables = 'SSL_flag'
            ssl_var[:,0,0] = df['SSL'].fillna(FILL_VALUE_FLOAT).values

            ssl_flag_var = nc.createVariable('SSL_flag', 'b', ('time', 'lat', 'lon'))
            ssl_flag_var.long_name = 'Quality flag for Suspended Sediment Load'
            ssl_flag_var.flag_values = [0, 1, 2, 3, 9]
            ssl_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
            ssl_flag_var[:,0,0] = df['SSL_flag'].values
            # ---------- Step/provenance flags (optional but recommended) ----------
            def _write_step_flag(varname):
                if varname in df.columns:
                    v = nc.createVariable(varname, 'b', ('time', 'lat', 'lon'), fill_value=FILL_VALUE_INT)
                    v.long_name = f"Provenance step flag: {varname}"
                    v.flag_values = [0, 1, 2, 3, 9]
                    v.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
                    v[:, 0, 0] = df[varname].fillna(9).astype(np.int8).values

            # Q steps
            _write_step_flag("Q_flag_qc1_physical")
            _write_step_flag("Q_flag_qc2_log_iqr")
            _write_step_flag("Q_flag_qc3_ssc_q_envelope")

            # SSC steps
            _write_step_flag("SSC_flag_qc1_physical")
            _write_step_flag("SSC_flag_qc2_log_iqr")
            _write_step_flag("SSC_flag_qc3_ssc_q_envelope")

            # SSL steps
            _write_step_flag("SSL_flag_qc1_physical")
            _write_step_flag("SSL_flag_qc2_log_iqr")
            _write_step_flag("SSL_flag_qc3_propagated_from_ssc_q")

        print(f"  - Created {nc_filename}")

        n = len(df)
        skipped_log_iqr = (n < 5) 
        skipped_ssc_q = (n < 5) or (ssc_q_bounds is None)

        last = df.iloc[-1]
        print_qc_summary(
            river_name=river_name,
            station_id=station_id,
            n=n,
            skipped_log_iqr=skipped_log_iqr,
            skipped_ssc_q=skipped_ssc_q,
            q=float(last['Q']), q_flag=int(last['Q_flag']),
            ssc=float(last['SSC']), ssc_flag=int(last['SSC_flag']),
            ssl=float(last['SSL']), ssl_flag=int(last['SSL_flag']),
            nc_path=nc_filename
        )


        # ==========================================================
        # Post-write CF-1.8 / ACDD-1.3 compliance check
        # ==========================================================
        # errors, warnings = check_nc_completeness(nc_filename)

        # if errors:
        #     print("  ❌ CF/ACDD compliance FAILED:")
        #     for e in errors:
        #         print(f"     - {e}")
        #     raise RuntimeError(
        #         f"NetCDF compliance check failed for {nc_filename}"
        #     )

        # if warnings:
        #     print("  ⚠️ CF/ACDD compliance warnings:")
        #     for w in warnings:
        #         print(f"     - {w}")


        # =====================================
        # SSC–Q diagnostic plot
        # =====================================
        diag_png = os.path.join(
            DIAG_DIR,
            f"SSC_Q_{station_id}.png"
        )

        plot_ssc_q_diagnostic(
            time=df.index.to_pydatetime(),
            Q=df['Q'].values,
            SSC=df['SSC'].values,
            Q_flag=df['Q_flag'].values,
            SSC_flag=df['SSC_flag'].values,
            ssc_q_bounds=ssc_q_bounds,
            station_id=station_id,
            station_name=dis_data['meta'].get('station_name', river_name),
            out_png=diag_png
        )


        # --- Generate Summary ---
        # --- Generate Summary (metadata + completeness + QC counts) ---
        summary = {
            "Source_ID": station_id,
            "station_name": dis_data["meta"].get("station_name", river_name),
            "river_name": river_name,
            "longitude": dis_data["meta"].get("longitude", np.nan),
            "latitude": dis_data["meta"].get("latitude", np.nan),
            "altitude": dis_data["meta"].get("altitude", np.nan),
            "upstream_area": dis_data["meta"].get("drainage_area", np.nan),

            "Q_start_date": df["Q"].first_valid_index().strftime("%Y-%m-%d") if df["Q"].first_valid_index() else None,
            "Q_end_date": df["Q"].last_valid_index().strftime("%Y-%m-%d") if df["Q"].last_valid_index() else None,
            "Q_percent_complete": (df["Q"].count() / len(df)) * 100 if len(df) > 0 else 0,

            "SSC_start_date": df["SSC"].first_valid_index().strftime("%Y-%m-%d") if df["SSC"].first_valid_index() else None,
            "SSC_end_date": df["SSC"].last_valid_index().strftime("%Y-%m-%d") if df["SSC"].last_valid_index() else None,
            "SSC_percent_complete": (df["SSC"].count() / len(df)) * 100 if len(df) > 0 else 0,

            "SSL_start_date": df["SSL"].first_valid_index().strftime("%Y-%m-%d") if df["SSL"].first_valid_index() else None,
            "SSL_end_date": df["SSL"].last_valid_index().strftime("%Y-%m-%d") if df["SSL"].last_valid_index() else None,
            "SSL_percent_complete": (df["SSL"].count() / len(df)) * 100 if len(df) > 0 else 0,
        }

        # ✅ 关键：塞入 QC_n_days + final/step flag 计数（扁平化列）
        summary.update(_qc_counts_flat(df))
        summary_data.append(summary)

    # Write summary CSV

    summary_csv = os.path.join(OUTPUT_DIR, "Eurasian_River_station_summary.csv")
    qc_csv = os.path.join(OUTPUT_DIR, "Eurasian_River_qc_results_summary.csv")

    generate_csv_summary_tool(summary_data, summary_csv)
    generate_qc_results_csv_tool(summary_data, qc_csv)

    print(f"\nCreated summary file: {summary_csv}")
    print(f"Created QC results file: {qc_csv}")

if __name__ == "__main__":
    main()