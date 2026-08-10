#!/usr/bin/env python3
"""Tests for daily aggregation of sub-daily Q, SSC, SSL observations.

Covers:
  Test  1: 240 ~6-min observations -> 1 daily record
  Test  2: minute-scale irregular observations -> 1 daily record
  Test  3: 2-3 irregular observations -> 1 daily record
  Test  4: duplicate identical timestamps -> collapsed before daily mean
  Test  5: two days, each with multiple obs -> exactly 2 daily records
  Test  6: HYBAM-style: Q and SSC timestamps offset -> merged by day, no duplicates
  Test  7: HYBAM: SSC day D, Q only D+1 -> no cross-day pairing
  Test  8: SSL_daily = Q_daily * SSC_daily * 0.0864 exactly
  Test  9: SSC-only day -> sediment observation preserved
  Test 10: final daily output: max count per station+date = 1
"""



import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from code.daily_aggregation import (
    collapse_duplicate_timestamps,
    aggregate_daily,
    _worst_flag,
)

FLAG_GOOD = np.int8(0)
FLAG_ESTIMATED = np.int8(1)
FLAG_SUSPECT = np.int8(2)
FLAG_BAD = np.int8(3)
FLAG_MISSING = np.int8(9)
FILL_VALUE = -9999.0


def _epoch_seconds(*dt_args):
    """Convert (year, month, day, hour, min, sec) to UTC epoch seconds."""
    dt = datetime(*dt_args, tzinfo=timezone.utc)
    return (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()


def assert_close(a, b, rel=1e-9):
    assert math.isclose(float(a), float(b), rel_tol=rel), f"{a} != {b}"


# ---------------------------------------------------------------------------
# Test 1: 240 ~6-min observations -> 1 daily record
# ---------------------------------------------------------------------------
def test_1_high_freq_six_min():
    """240 sub-daily observations at ~6 min interval -> 1 daily record."""
    n = 240
    base = _epoch_seconds(2000, 1, 15, 0, 0, 0)
    times = np.array([base + i * 360.0 for i in range(n)])  # ~6 min intervals
    Q = np.full(n, 100.0, dtype=float) + np.random.default_rng(42).normal(0, 5, n)
    SSC = np.full(n, 50.0, dtype=float) + np.random.default_rng(43).normal(0, 2, n)
    SSL = np.full(n, np.nan)

    qf = np.full(n, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(n, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(n, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1, f"Expected 1 daily record, got {len(result['time'])}"
    assert_close(np.mean(Q), result["Q"][0], rel=1e-6)
    assert_close(np.mean(SSC), result["SSC"][0], rel=1e-6)
    print("PASS test_1_high_freq_six_min")


# ---------------------------------------------------------------------------
# Test 2: minute-scale irregular observations -> 1 daily record
# ---------------------------------------------------------------------------
def test_2_minute_scale_irregular():
    """Minute-scale irregular observations -> 1 daily record."""
    base = _epoch_seconds(2000, 1, 15, 0, 0, 0)
    # Random sub-daily times within one day
    rng = np.random.default_rng(99)
    offsets = np.sort(rng.uniform(0, 86400, size=50))
    times = base + offsets

    Q = np.full(50, 80.0, dtype=float)
    SSC = np.full(50, 30.0, dtype=float)
    SSL = np.full(50, np.nan)

    qf = np.full(50, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(50, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(50, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1, f"Expected 1 daily record, got {len(result['time'])}"
    assert_close(80.0, result["Q"][0])
    assert_close(30.0, result["SSC"][0])
    print("PASS test_2_minute_scale_irregular")


# ---------------------------------------------------------------------------
# Test 3: 2-3 irregular observations -> 1 daily record
# ---------------------------------------------------------------------------
def test_3_few_irregular_obs():
    """2-3 irregular observations in one day -> 1 daily record."""
    base = _epoch_seconds(2000, 3, 10, 0, 0, 0)
    times = np.array([base + 3600, base + 39600, base + 79200])  # 1h, 11h, 22h
    Q = np.array([50.0, 60.0, 55.0])
    SSC = np.array([100.0, 110.0, 105.0])
    SSL = np.full(3, np.nan)

    qf = np.full(3, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(3, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(3, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1, f"Expected 1 daily record, got {len(result['time'])}"
    assert_close(np.mean([50.0, 60.0, 55.0]), result["Q"][0])
    assert_close(np.mean([100.0, 110.0, 105.0]), result["SSC"][0])
    print("PASS test_3_few_irregular_obs")


# ---------------------------------------------------------------------------
# Test 4: duplicate identical timestamps -> collapsed before daily mean
# ---------------------------------------------------------------------------
def test_4_duplicate_timestamps():
    """Duplicate timestamp rows -> collapsed, no extra weighting."""
    base = _epoch_seconds(2000, 5, 1, 12, 0, 0)
    # Two unique timestamps, one duplicated 3x
    times = np.array([base, base, base, base + 14400])  # noon, noon, noon, 4pm
    Q = np.array([100.0, 100.0, 100.0, 200.0])
    SSC = np.array([50.0, 50.0, 50.0, 100.0])
    SSL = np.full(4, np.nan)

    qf = np.full(4, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(4, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(4, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    # After collapse: noon=100, 4pm=200, average=150
    # NOT: (100+100+100+200)/4 = 125
    assert len(result["time"]) == 1
    assert_close(150.0, result["Q"][0])  # mean(100, 200) = 150
    assert_close(75.0, result["SSC"][0])  # mean(50, 100) = 75
    print("PASS test_4_duplicate_timestamps")


def test_4b_duplicate_different_values():
    """Duplicate timestamps with different values -> mean of that timestamp."""
    base = _epoch_seconds(2000, 5, 1, 12, 0, 0)
    times = np.array([base, base])
    Q = np.array([100.0, 120.0])
    SSC = np.array([50.0, 60.0])
    SSL = np.full(2, np.nan)

    qf = np.full(2, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(2, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(2, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1
    assert_close(110.0, result["Q"][0])  # mean(100, 120)
    assert_close(55.0, result["SSC"][0])  # mean(50, 60)
    print("PASS test_4b_duplicate_different_values")


# ---------------------------------------------------------------------------
# Test 5: two days, each with multiple obs -> exactly 2 daily records
# ---------------------------------------------------------------------------
def test_5_two_days_multi_obs():
    """Two separate days, each with multiple obs -> exactly 2 records."""
    base_day1 = _epoch_seconds(2000, 1, 1, 0, 0, 0)
    base_day2 = _epoch_seconds(2000, 1, 2, 0, 0, 0)

    times = np.array([
        base_day1 + 3600, base_day1 + 43200, base_day1 + 79200,  # day 1
        base_day2 + 7200, base_day2 + 36000, base_day2 + 72000,  # day 2
    ])
    Q = np.array([100.0, 120.0, 110.0, 200.0, 220.0, 210.0])
    SSC = np.array([50.0, 55.0, 52.0, 100.0, 105.0, 102.0])
    SSL = np.full(6, np.nan)

    qf = np.full(6, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(6, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(6, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 2, f"Expected 2 daily records, got {len(result['time'])}"
    # Values should not mix across days
    q_vals = sorted(result["Q"])
    ssc_vals = sorted(result["SSC"])
    assert_close(110.0, q_vals[0])  # mean(100, 120, 110) = 110
    assert_close(210.0, q_vals[1])  # mean(200, 220, 210) = 210
    print("PASS test_5_two_days_multi_obs")


# ---------------------------------------------------------------------------
# Test 6: HYBAM-style Q/SSC timestamp offset -> merged by day
# ---------------------------------------------------------------------------
def test_6_hybam_offset_timestamps():
    """Q and SSC at different sub-daily times -> separately aggregated, merged by day."""
    base = _epoch_seconds(2000, 6, 15, 0, 0, 0)
    # Q at 00:00, 06:00, 12:00, 18:00
    q_times = base + np.array([0, 21600, 43200, 64800])
    # SSC at 03:00, 09:00, 15:00, 21:00
    ssc_times = base + np.array([10800, 32400, 54000, 75600])

    # Combine: this is what HYBAM currently does (merged time axis)
    all_times = np.sort(np.concatenate([q_times, ssc_times]))
    n = len(all_times)
    Q_arr = np.full(n, np.nan)
    SSC_arr = np.full(n, np.nan)
    SSL_arr = np.full(n, np.nan)
    qf_arr = np.full(n, FLAG_MISSING, dtype=np.int8)
    sscf_arr = np.full(n, FLAG_MISSING, dtype=np.int8)
    sslf_arr = np.full(n, FLAG_MISSING, dtype=np.int8)

    for i, t in enumerate(all_times):
        if t in q_times:
            Q_arr[i] = 100.0
            qf_arr[i] = FLAG_GOOD
        if t in ssc_times:
            SSC_arr[i] = 50.0
            sscf_arr[i] = FLAG_GOOD

    result = aggregate_daily(all_times, Q_arr, SSC_arr, SSL_arr, qf_arr, sscf_arr, sslf_arr)

    # Should produce exactly 1 daily record (not 8)
    assert len(result["time"]) == 1, f"Expected 1 daily record, got {len(result['time'])}"
    assert_close(100.0, result["Q"][0])
    assert_close(50.0, result["SSC"][0])
    # SSL should be recalculated
    assert_close(100.0 * 50.0 * 0.0864, result["SSL"][0])
    print("PASS test_6_hybam_offset_timestamps")


# ---------------------------------------------------------------------------
# Test 7: HYBAM SSC day D, Q only day D+1 -> no cross-day pairing
# ---------------------------------------------------------------------------
def test_7_hybam_no_cross_day_pairing():
    """SSC on day D, Q only on D+1 -> no false same-day pair created."""
    day1 = _epoch_seconds(2000, 7, 1, 12, 0, 0)
    day2 = _epoch_seconds(2000, 7, 2, 12, 0, 0)

    times = np.array([day1, day2])
    Q_arr = np.array([np.nan, 100.0])
    SSC_arr = np.array([50.0, np.nan])
    SSL_arr = np.full(2, np.nan)

    qf_arr = np.array([FLAG_MISSING, FLAG_GOOD], dtype=np.int8)
    sscf_arr = np.array([FLAG_GOOD, FLAG_MISSING], dtype=np.int8)
    sslf_arr = np.full(2, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q_arr, SSC_arr, SSL_arr, qf_arr, sscf_arr, sslf_arr)

    # Should produce 2 daily records: day1=SSC only, day2=Q only
    assert len(result["time"]) == 2, f"Expected 2 daily records, got {len(result['time'])}"

    # Find each day's record
    for i in range(2):
        dt = datetime.fromtimestamp(result["time"][i], tz=timezone.utc)
        if dt.day == 1:
            assert_close(50.0, result["SSC"][i])  # SSC preserved
            assert not np.isfinite(result["Q"][i])  # No Q
            assert not np.isfinite(result["SSL"][i])  # No SSL (no Q to pair)
        elif dt.day == 2:
            assert_close(100.0, result["Q"][i])
            assert not np.isfinite(result["SSC"][i])
    print("PASS test_7_hybam_no_cross_day_pairing")


# ---------------------------------------------------------------------------
# Test 8: SSL_daily = Q_daily * SSC_daily * 0.0864 exactly
# ---------------------------------------------------------------------------
def test_8_ssl_formula_exact():
    """SSL_daily must equal Q_daily * SSC_daily * 0.0864."""
    base = _epoch_seconds(2000, 8, 1, 0, 0, 0)
    n = 10
    times = base + np.arange(n) * 7200.0  # every 2 hours
    Q = np.array([50.0, 52.0, 48.0, 51.0, 53.0, 49.0, 50.0, 51.0, 52.0, 48.0])
    SSC = np.array([100.0, 102.0, 98.0, 101.0, 103.0, 99.0, 100.0, 101.0, 102.0, 98.0])
    SSL = np.full(n, 999999.0)  # Some garbage values that should be ignored

    qf = np.full(n, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(n, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(n, FLAG_GOOD, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    q_mean = np.mean(Q)
    ssc_mean = np.mean(SSC)
    expected_ssl = q_mean * ssc_mean * 0.0864

    assert len(result["time"]) == 1
    assert_close(q_mean, result["Q"][0])
    assert_close(ssc_mean, result["SSC"][0])
    assert_close(expected_ssl, result["SSL"][0])

    # Verify it is NOT mean(subdaily SSL) = mean(Q_i * SSC_i * 0.0864)
    wrong_ssl = np.mean(Q * SSC * 0.0864)
    assert not math.isclose(expected_ssl, wrong_ssl, rel_tol=1e-12), \
        "SSL should NOT be mean of sub-daily SSL products"
    print("PASS test_8_ssl_formula_exact")


# ---------------------------------------------------------------------------
# Test 9: SSC-only day preserved
# ---------------------------------------------------------------------------
def test_9_ssc_only_day_preserved():
    """SSC-only day -> sediment observation not deleted."""
    base = _epoch_seconds(2000, 9, 1, 12, 0, 0)
    times = np.array([base])
    Q = np.array([np.nan])
    SSC = np.array([50.0])
    SSL = np.full(1, np.nan)

    qf = np.array([FLAG_MISSING], dtype=np.int8)
    sscf = np.array([FLAG_GOOD], dtype=np.int8)
    sslf = np.array([FLAG_MISSING], dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1, "SSC-only day should produce a record, not be deleted"
    assert_close(50.0, result["SSC"][0])
    print("PASS test_9_ssc_only_day_preserved")


def test_9b_direct_ssl_only_day_preserved():
    """Direct source SSL-only day preserved (no Q, no SSC)."""
    base = _epoch_seconds(2000, 9, 2, 12, 0, 0)
    times = np.array([base])
    Q = np.array([np.nan])
    SSC = np.array([np.nan])
    SSL = np.array([100.0])

    qf = np.array([FLAG_MISSING], dtype=np.int8)
    sscf = np.array([FLAG_MISSING], dtype=np.int8)
    sslf = np.array([FLAG_GOOD], dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 1, "SSL-only day should produce a record"
    assert_close(100.0, result["SSL"][0])
    assert result["SSL_flag"][0] == FLAG_GOOD  # Direct source, not derived
    print("PASS test_9b_direct_ssl_only_day_preserved")


# ---------------------------------------------------------------------------
# Test 10: group by station+date -> max count = 1
# ---------------------------------------------------------------------------
def test_10_no_duplicate_dates():
    """Final output grouped by station + calendar_date has max count = 1."""
    rng = np.random.default_rng(777)
    times = []
    Q_vals, SSC_vals, SSL_vals = [], [], []
    qf_vals, sscf_vals, sslf_vals = [], [], []

    # Generate 20 days of data, 1-10 observations per day
    expected_days = set()
    for day in range(1, 21):
        date_str = f"2000-01-{day:02d}"
        expected_days.add(date_str)
        base = _epoch_seconds(2000, 1, day, 0, 0, 0)
        n_obs = rng.integers(1, 11)
        for _ in range(n_obs):
            offset = rng.uniform(0, 86400)
            times.append(base + offset)
            Q_vals.append(rng.uniform(10, 200))
            SSC_vals.append(rng.uniform(5, 100))
            SSL_vals.append(np.nan)
            qf_vals.append(0)
            sscf_vals.append(0)
            sslf_vals.append(9)

    times = np.array(times)
    Q_arr = np.array(Q_vals)
    SSC_arr = np.array(SSC_vals)
    SSL_arr = np.array(SSL_vals)
    qf = np.array(qf_vals, dtype=np.int8)
    sscf = np.array(sscf_vals, dtype=np.int8)
    sslf = np.array(sslf_vals, dtype=np.int8)

    result = aggregate_daily(times, Q_arr, SSC_arr, SSL_arr, qf, sscf, sslf)

    # Check no duplicate dates
    result_dates = []
    for t in result["time"]:
        result_dates.append(datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d'))

    from collections import Counter
    date_counts = Counter(result_dates)
    max_count = max(date_counts.values())
    assert max_count == 1, f"Max count per date = {max_count}, expected 1"
    # Also check we didn't lose days
    assert len(result_dates) == 20, f"Expected 20 days, got {len(result_dates)}"

    print("PASS test_10_no_duplicate_dates")


# ---------------------------------------------------------------------------
# Test 11: flag propagation - good inputs -> derived SSL = estimated
# ---------------------------------------------------------------------------
def test_11_flag_propagation_derived_ssl():
    """Q,SSC both good -> SSL should be estimated(1)."""
    base = _epoch_seconds(2000, 10, 1, 12, 0, 0)
    times = np.array([base, base + 3600])
    Q = np.array([100.0, 100.0])
    SSC = np.array([50.0, 50.0])
    SSL = np.full(2, np.nan)

    qf = np.full(2, FLAG_GOOD, dtype=np.int8)
    sscf = np.full(2, FLAG_GOOD, dtype=np.int8)
    sslf = np.full(2, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert result["SSL_flag"][0] == FLAG_ESTIMATED
    assert result["SSL_derived_mask"][0] == True
    print("PASS test_11_flag_propagation_derived_ssl")


# ---------------------------------------------------------------------------
# Test 12: flag propagation - suspect Q -> suspect SSL
# ---------------------------------------------------------------------------
def test_12_flag_propagation_suspect_q_to_ssl():
    """Suspect Q input -> derived SSL should be suspect(2)."""
    base = _epoch_seconds(2000, 11, 1, 12, 0, 0)
    times = np.array([base])
    Q = np.array([100.0])
    SSC = np.array([50.0])
    SSL = np.full(1, np.nan)

    qf = np.array([FLAG_SUSPECT], dtype=np.int8)
    sscf = np.array([FLAG_GOOD], dtype=np.int8)
    sslf = np.array([FLAG_MISSING], dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    # Derived SSL from suspect Q -> suspect
    assert result["SSL_flag"][0] == FLAG_SUSPECT
    print("PASS test_12_flag_propagation_suspect_q_to_ssl")


# ---------------------------------------------------------------------------
# Test 13: fill values excluded from mean
# ---------------------------------------------------------------------------
def test_13_fill_values_excluded():
    """-9999 fill values must not enter arithmetic mean."""
    base = _epoch_seconds(2000, 12, 1, 12, 0, 0)
    times = np.array([base, base + 3600, base + 7200])
    Q = np.array([100.0, -9999.0, 200.0])
    SSC = np.array([50.0, -9999.0, 100.0])
    SSL = np.full(3, -9999.0)

    qf = np.array([FLAG_GOOD, FLAG_MISSING, FLAG_GOOD], dtype=np.int8)
    sscf = np.array([FLAG_GOOD, FLAG_MISSING, FLAG_GOOD], dtype=np.int8)
    sslf = np.full(3, FLAG_MISSING, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    # Mean should be (100+200)/2 = 150, NOT (100-9999+200)/3
    assert len(result["time"]) == 1
    assert_close(150.0, result["Q"][0])
    assert_close(75.0, result["SSC"][0])
    print("PASS test_13_fill_values_excluded")


# ---------------------------------------------------------------------------
# Test 14: collapse_duplicate_timestamps standalone
# ---------------------------------------------------------------------------
def test_14_collapse_duplicates_standalone():
    """Standalone duplicate collapse with mixed values."""
    base = _epoch_seconds(2000, 1, 1, 0, 0, 0)
    ts = np.array([base, base, base + 3600, base + 3600, base + 7200])
    vals = np.array([10.0, 20.0, 30.0, 30.0, 40.0])
    flags = np.full(5, FLAG_GOOD, dtype=np.int8)

    unique_ts, collapsed, coll_flags = collapse_duplicate_timestamps(ts, vals, flags)

    assert len(unique_ts) == 3
    # First timestamp: mean(10, 20) = 15
    # Second: mean(30, 30) = 30 (same values)
    # Third: 40
    expected = {base: 15.0, base + 3600: 30.0, base + 7200: 40.0}
    for i in range(3):
        assert_close(expected[unique_ts[i]], collapsed[i])
    print("PASS test_14_collapse_duplicates_standalone")


# ---------------------------------------------------------------------------
# Test 15: worst_flag helper
# ---------------------------------------------------------------------------
def test_15_worst_flag():
    assert _worst_flag([0, 0, 0]) == 0
    assert _worst_flag([0, 1, 0]) == 1
    assert _worst_flag([0, 2, 1]) == 2
    assert _worst_flag([0, 3, 2, 1]) == 3
    assert _worst_flag([9, 0, 1]) == 9
    assert _worst_flag([9, 9, 9]) == 9
    assert _worst_flag([]) == 9
    print("PASS test_15_worst_flag")


# ---------------------------------------------------------------------------
# Test 16: multi-day HYBAM-style with SSC-only days
# ---------------------------------------------------------------------------
def test_16_multi_day_hybam_mixed():
    """3 days: day1 Q+SSC, day2 SSC only, day3 Q+SSC -> correct output."""
    day1 = _epoch_seconds(2001, 1, 1, 12, 0, 0)
    day2 = _epoch_seconds(2001, 1, 2, 12, 0, 0)
    day3 = _epoch_seconds(2001, 1, 3, 12, 0, 0)

    times = np.array([day1, day1 + 3600, day2, day3, day3 + 7200])
    Q = np.array([100.0, 120.0, np.nan, 200.0, 210.0])
    SSC = np.array([50.0, 55.0, 30.0, 100.0, 105.0])
    SSL = np.full(5, np.nan)

    qf = np.array([0, 0, 9, 0, 0], dtype=np.int8)
    sscf = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    sslf = np.full(5, 9, dtype=np.int8)

    result = aggregate_daily(times, Q, SSC, SSL, qf, sscf, sslf)

    assert len(result["time"]) == 3, f"Expected 3 days, got {len(result['time'])}"

    for i in range(3):
        dt = datetime.fromtimestamp(result["time"][i], tz=timezone.utc)
        if dt.day == 1:
            assert_close(110.0, result["Q"][i])
            assert_close(52.5, result["SSC"][i])
        elif dt.day == 2:
            assert not np.isfinite(result["Q"][i])
            assert_close(30.0, result["SSC"][i])
            assert not np.isfinite(result["SSL"][i])
        elif dt.day == 3:
            assert_close(205.0, result["Q"][i])
            assert_close(102.5, result["SSC"][i])
    print("PASS test_16_multi_day_hybam_mixed")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()


if __name__ == "__main__":
    main()
