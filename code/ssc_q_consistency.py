"""
SSC-Q hydrological consistency quality control module

Purpose
-------
Identify SSC values that are inconsistent with site-specific SSC–Q
relationships under similar flow conditions, following USGS principles.

Design principles
-----------------
- Station-wise only
- Flow-conditioned comparison
- Log-space analysis
- Flag-based (no automatic deletion)

Author: Zhongwang Wei
"""

import numpy as np


def flow_bins(q, n_bins=3):
    """
    Assign flow bins based on quantiles.

    Returns
    -------
    bins : np.ndarray of int
        Bin index for each Q value.
    """
    q = np.asarray(q, dtype=float)
    bins = np.full(len(q), -1, dtype=int)

    valid = np.isfinite(q) & (q > 0)
    if valid.sum() < 5:
        return bins

    quantiles = np.quantile(q[valid], np.linspace(0, 1, n_bins + 1))
    for i in range(n_bins):
        mask = (q >= quantiles[i]) & (q <= quantiles[i + 1])
        bins[mask] = i

    return bins


def detect_ssc_q_outliers(
    q,
    ssc,
    log_threshold=1.0,
    min_samples_per_bin=3,
):
    """
    Detect SSC-Q inconsistency outliers.

    Parameters
    ----------
    q : array-like
        Discharge values.
    ssc : array-like
        Suspended sediment concentration values.
    log_threshold : float
        Threshold in log10 space (default = 1.0 ~ one order of magnitude).
    min_samples_per_bin : int
        Minimum samples required per flow bin.

    Returns
    -------
    flags : np.ndarray of bool
        True where SSC-Q inconsistency is detected.
    """
    q = np.asarray(q, dtype=float)
    ssc = np.asarray(ssc, dtype=float)

    flags = np.zeros(len(q), dtype=bool)

    valid = np.isfinite(q) & np.isfinite(ssc) & (q > 0) & (ssc > 0)
    if valid.sum() < 5:
        return flags

    bins = flow_bins(q)

    for b in np.unique(bins[bins >= 0]):
        idx = np.where((bins == b) & valid)[0]
        if len(idx) < min_samples_per_bin:
            continue

        log_ssc = np.log10(ssc[idx])
        med = np.median(log_ssc)

        deviation = np.abs(log_ssc - med)
        flags[idx] = deviation > log_threshold

    return flags


def apply_ssc_q_qc(
    q,
    ssc,
    ssc_flags,
    suspect_flag=2,
    bad_flag=3,
):
    """
    Apply SSC-Q hydrological consistency QC.

    Logic
    -----
    - Only update values currently flagged as good (0)
    - Promote to suspect (2) if SSC-Q inconsistency detected
    - Do not override bad data (3)

    Returns
    -------
    new_flags : np.ndarray
    ssc_q_flags : np.ndarray of bool
    """
    q = np.asarray(q, dtype=float)
    ssc = np.asarray(ssc, dtype=float)
    ssc_flags = np.asarray(ssc_flags, dtype=int)

    ssc_q_flags = detect_ssc_q_outliers(q, ssc)
    new_flags = ssc_flags.copy()

    for i in range(len(new_flags)):
        if ssc_flags[i] == 0 and ssc_q_flags[i]:
            new_flags[i] = suspect_flag

    return new_flags, ssc_q_flags

