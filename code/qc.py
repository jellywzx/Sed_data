# core/qc.py
import numpy as np
import pandas as pd


def apply_quality_flag(value):
    if pd.isna(value) or np.isnan(value):
        return np.int8(9)
    if value < 0:
        return np.int8(3)
    return np.int8(0)


def compute_log_iqr_bounds(values, k=1.5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 5:
        return None, None

    logv = np.log10(values)
    q1 = np.percentile(logv, 25)
    q3 = np.percentile(logv, 75)
    iqr = q3 - q1

    return 10 ** (q1 - k * iqr), 10 ** (q3 + k * iqr)


def apply_ssl_log_iqr_flag(ssl_mt_yr, bounds, base_flag):
    if base_flag != 0 or bounds is None or bounds[0] is None:
        return base_flag

    lower, upper = bounds
    if ssl_mt_yr > 0 and (ssl_mt_yr < lower or ssl_mt_yr > upper):
        return np.int8(2)

    return base_flag
