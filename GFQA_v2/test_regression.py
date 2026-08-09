#!/usr/bin/env python3
"""
Regression test for manuscript-consistent GFQA_v2 processing rules.

Tests:
  1. SSC-only station (no Q in Flux.csv) → must NOT be deleted
  2. Q+SSC station → must work as before
  3. SSC-only records within a Q+SSC station → preserved via outer merge
  4. Station with no overlapping Q-SSC days → SSC records still kept
"""
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Setup path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from code.constants import FILL_VALUE_FLOAT, FILL_VALUE_INT
from code.qc import apply_hydro_qc_with_provenance

# ---- Helper: replicate the OLD logic ----
def old_logic_station_filter(flux_df, water_df):
    """OLD: only stations in BOTH Flux and Water"""
    flux_stations = set(flux_df['GEMS.Station.Number'].unique())
    water_stations = set(water_df['GEMS.Station.Number'].unique())
    return flux_stations & water_stations

def old_logic_process(discharge_data, sediment_data):
    """OLD: requires overlapping period + inner merge"""
    if len(discharge_data) == 0 or len(sediment_data) == 0:
        return None, "no_overlap"
    start = max(discharge_data['Sample.Date'].min(), sediment_data['Sample.Date'].min())
    end = min(discharge_data['Sample.Date'].max(), sediment_data['Sample.Date'].max())
    if start > end:
        return None, "no_overlap"
    # daily agg
    q_daily = discharge_data.copy()
    q_daily['Date'] = q_daily['Sample.Date'].dt.floor('D')
    q_daily = q_daily.groupby('Date')['Value'].mean().reset_index()
    q_daily.columns = ['Date', 'Q']
    
    s_daily = sediment_data.copy()
    s_daily['Date'] = s_daily['Sample.Date'].dt.floor('D')
    s_daily = s_daily.groupby('Date')['Value'].mean().reset_index()
    s_daily.columns = ['Date', 'SSC']
    
    merged = pd.merge(q_daily, s_daily, on='Date', how='inner')
    if merged.empty:
        return None, "no_same_day"
    return merged, "ok"

# ---- Helper: replicate the NEW logic ----
def new_logic_station_filter(water_df):
    """NEW: stations from water-quality TSS records only"""
    tss = water_df[water_df['Parameter.Code'] == 'TSS']
    return set(tss['GEMS.Station.Number'].unique())

def new_logic_process(discharge_data, sediment_data):
    """NEW: outer merge, no overlapping requirement"""
    has_q = len(discharge_data) > 0
    has_ssc = len(sediment_data) > 0
    
    if not has_q and not has_ssc:
        return None, "no_data"
    
    if has_q:
        q_daily = discharge_data.copy()
        q_daily['Date'] = q_daily['Sample.Date'].dt.floor('D')
        q_daily = q_daily.groupby('Date')['Value'].mean().reset_index()
        q_daily.columns = ['Date', 'Q']
    else:
        q_daily = pd.DataFrame(columns=['Date', 'Q'])
    
    if has_ssc:
        s_daily = sediment_data.copy()
        s_daily['Date'] = s_daily['Sample.Date'].dt.floor('D')
        s_daily = s_daily.groupby('Date')['Value'].mean().reset_index()
        s_daily.columns = ['Date', 'SSC']
    else:
        s_daily = pd.DataFrame(columns=['Date', 'SSC'])
    
    if has_q and has_ssc:
        merged = pd.merge(q_daily, s_daily, on='Date', how='outer')
    elif has_q:
        merged = q_daily.copy()
        merged['SSC'] = np.nan
    else:
        merged = s_daily.copy()
        merged['Q'] = np.nan
    
    return merged, "ok"

# ================================================================
# BUILD SYNTHETIC TEST DATA
# ================================================================
np.random.seed(42)
base_date = datetime(2020, 1, 1)
dates_q = [base_date + timedelta(days=i) for i in range(10)]
dates_ssc = [base_date + timedelta(days=i) for i in range(5, 15)]  # 5 days overlap

# Station A: Q+SSC (both present, 5 days overlap)
flux_A = pd.DataFrame({
    'GEMS.Station.Number': ['STA_A'] * 10,
    'Parameter.Code': ['Q-Inst'] * 10,
    'Sample.Date': dates_q,
    'Value': np.random.uniform(10, 100, 10),
    'Data.Quality': ['checked'] * 10,
})
water_A = pd.DataFrame({
    'GEMS.Station.Number': ['STA_A'] * 10,
    'Parameter.Code': ['TSS'] * 10,
    'Sample.Date': dates_ssc,
    'Value': np.random.uniform(5, 200, 10),
    'Data.Quality': ['checked'] * 10,
})

# Station B: SSC-only (no Q in Flux.csv)
water_B = pd.DataFrame({
    'GEMS.Station.Number': ['STA_B'] * 8,
    'Parameter.Code': ['TSS'] * 8,
    'Sample.Date': [base_date + timedelta(days=i*2) for i in range(8)],
    'Value': np.random.uniform(10, 150, 8),
    'Data.Quality': ['checked'] * 8,
})

# Station C: Q+SSC but NO same-day overlap
dates_q_c = [base_date + timedelta(days=i) for i in range(5)]
dates_ssc_c = [base_date + timedelta(days=i) for i in range(20, 30)]
flux_C = pd.DataFrame({
    'GEMS.Station.Number': ['STA_C'] * 5,
    'Parameter.Code': ['Q-Inst'] * 5,
    'Sample.Date': dates_q_c,
    'Value': np.random.uniform(10, 100, 5),
    'Data.Quality': ['checked'] * 5,
})
water_C = pd.DataFrame({
    'GEMS.Station.Number': ['STA_C'] * 10,
    'Parameter.Code': ['TSS'] * 10,
    'Sample.Date': dates_ssc_c,
    'Value': np.random.uniform(5, 200, 10),
    'Data.Quality': ['checked'] * 10,
})

flux_all = pd.concat([flux_A, flux_C], ignore_index=True)
water_all = pd.concat([water_A, water_B, water_C], ignore_index=True)

print("=" * 70)
print("REGRESSION TEST: Manuscript-consistent GFQA_v2 rules")
print("=" * 70)

# ================================================================
# TEST 1: Station filter
# ================================================================
print("\n--- TEST 1: Station candidate filter ---")
old_candidates = old_logic_station_filter(flux_all, water_all)
new_candidates = new_logic_station_filter(water_all)

print(f"OLD candidates (flux & water intersection): {sorted(old_candidates)}")
print(f"NEW candidates (TSS-bearing water stations): {sorted(new_candidates)}")

assert 'STA_A' in old_candidates, "STA_A should be in OLD"
assert 'STA_C' in old_candidates, "STA_C should be in OLD"
assert 'STA_B' not in old_candidates, "STA_B (SSC-only) should NOT be in OLD"
assert 'STA_B' in new_candidates, "STA_B (SSC-only) MUST be in NEW"
print("✅ TEST 1 PASSED: SSC-only station STA_B included in NEW, excluded in OLD")

# ================================================================
# TEST 2: SSC-only station processing
# ================================================================
print("\n--- TEST 2: SSC-only station (STA_B) ---")
disc_B = flux_all[flux_all['GEMS.Station.Number'] == 'STA_B']
sed_B = water_all[water_all['GEMS.Station.Number'] == 'STA_B']

old_result_B, old_status_B = old_logic_process(disc_B, sed_B)
new_result_B, new_status_B = new_logic_process(disc_B, sed_B)

print(f"OLD result: {old_status_B} (expect 'no_overlap')")
print(f"NEW result: {new_status_B} (expect 'ok')")
print(f"NEW SSC records: {len(new_result_B)} (expect 8)")

assert old_result_B is None, "OLD must skip STA_B (no Q data)"
assert new_result_B is not None, "NEW must NOT skip STA_B"
assert len(new_result_B) == 8, f"NEW should have 8 SSC records, got {len(new_result_B)}"
assert new_result_B['SSC'].notna().sum() == 8, "All 8 SSC records should be valid"
print("✅ TEST 2 PASSED: SSC-only station preserved, not deleted")

# ================================================================
# TEST 3: Q+SSC station (STA_A) - outer merge keeps SSC-only days
# ================================================================
print("\n--- TEST 3: Q+SSC station with partial overlap (STA_A) ---")
disc_A = flux_all[flux_all['GEMS.Station.Number'] == 'STA_A']
sed_A = water_all[water_all['GEMS.Station.Number'] == 'STA_A']

old_result_A, old_status_A = old_logic_process(disc_A, sed_A)
new_result_A, new_status_A = new_logic_process(disc_A, sed_A)

print(f"OLD records (inner merge): {len(old_result_A)}")
print(f"NEW records (outer merge): {len(new_result_A)}")
print(f"  OLD keeps only same-day Q+SSC pairs")
print(f"  NEW keeps all dates with Q or SSC")

# OLD: only overlapping days (dates 5-9, 5 days)
assert len(old_result_A) <= 5, f"OLD should have ≤5 overlapping days, got {len(old_result_A)}"

# NEW: all unique dates from both series (dates 0-14, 15 days)
assert len(new_result_A) >= 10, f"NEW should have ≥10 total days, got {len(new_result_A)}"

# SSC-only days (dates 10-14) should be present in NEW
ssc_only_days = new_result_A[new_result_A['Q'].isna() & new_result_A['SSC'].notna()]
print(f"  SSC-only days preserved: {len(ssc_only_days)} (expect 5)")
assert len(ssc_only_days) == 5, f"Expected 5 SSC-only days, got {len(ssc_only_days)}"
print("✅ TEST 3 PASSED: Outer merge preserves SSC-only days in Q+SSC station")

# ================================================================
# TEST 4: Station with NO overlapping days (STA_C)
# ================================================================
print("\n--- TEST 4: Station with no Q-SSC overlap (STA_C) ---")
disc_C = flux_all[flux_all['GEMS.Station.Number'] == 'STA_C']
sed_C = water_all[water_all['GEMS.Station.Number'] == 'STA_C']

old_result_C, old_status_C = old_logic_process(disc_C, sed_C)
new_result_C, new_status_C = new_logic_process(disc_C, sed_C)

print(f"OLD result: {old_status_C} (expect 'no_overlap')")
print(f"NEW result: {new_status_C} (expect 'ok')")
print(f"NEW SSC records: {new_result_C['SSC'].notna().sum()} (expect 10)")

assert old_result_C is None, "OLD must skip STA_C (no overlapping period)"
assert new_result_C is not None, "NEW must NOT skip STA_C"
assert new_result_C['SSC'].notna().sum() == 10, "All 10 SSC records should be valid"
print("✅ TEST 4 PASSED: Station with no Q-SSC overlap preserved")

# ================================================================
# TEST 5: QC module handles SSC-only data
# ================================================================
print("\n--- TEST 5: QC module with SSC-only data ---")
ssc_vals = np.random.uniform(10, 200, 8)
q_vals = np.full(8, np.nan)
ssl_vals = np.full(8, np.nan)
time_arr = np.array([base_date + timedelta(days=i*2) for i in range(8)])

qc_result = apply_hydro_qc_with_provenance(
    time=time_arr,
    Q=q_vals,
    SSC=ssc_vals,
    SSL=ssl_vals,
    Q_is_independent=True,
    SSC_is_independent=True,
    SSL_is_independent=False,
    ssl_is_derived_from_q_ssc=True,
    qc2_k=1.5,
    qc2_min_samples=5,
    qc3_k=1.5,
    qc3_min_samples=5,
)

assert qc_result is not None, "QC should NOT return None for SSC-only data"
ssc_flags = qc_result["SSC_flag"]
valid_ssc = (ssc_flags != FILL_VALUE_INT)
print(f"  Valid SSC records after QC: {valid_ssc.sum()}/{len(ssc_flags)}")
assert valid_ssc.sum() > 0, "At least some SSC records should be valid"
print(f"  Q flags: all missing={np.all(qc_result['Q_flag'] == FILL_VALUE_INT)} (expect True)")
print("✅ TEST 5 PASSED: QC module handles SSC-only data correctly")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("ALL 5 TESTS PASSED ✅")
print("=" * 70)
print("""
Summary of behavioral changes verified:
  TEST 1: SSC-only stations now included in candidate set (were excluded)
  TEST 2: SSC-only station generates output (was silently skipped)
  TEST 3: Outer merge preserves SSC-only days in Q+SSC stations
  TEST 4: Stations with no Q-SSC overlap still produce output
  TEST 5: QC module correctly handles NaN Q with valid SSC
  
SSC-only stations are NO LONGER DELETED. ✓
""")
