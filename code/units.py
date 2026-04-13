import numpy as np

from constants import DAYS_PER_JULIAN_YEAR, SSC_DISCHARGE_TO_SSL_FACTOR


def _coerce_finite(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def calculate_discharge(runoff_mm_yr, area_km2):
    runoff = _coerce_finite(runoff_mm_yr)
    area = _coerce_finite(area_km2)
    if runoff is None or area is None:
        return np.nan
    return runoff * area / 31557.6


def calculate_ssl_from_mt_yr(sediment_mt_yr):
    sediment = _coerce_finite(sediment_mt_yr)
    if sediment is None:
        return np.nan
    return sediment * 1e6 / DAYS_PER_JULIAN_YEAR


def calculate_ssc(ssl_ton_day, discharge_m3s):
    ssl = _coerce_finite(ssl_ton_day)
    discharge = _coerce_finite(discharge_m3s)
    if ssl is None or discharge is None or discharge <= 0:
        return np.nan
    return ssl / (discharge * SSC_DISCHARGE_TO_SSL_FACTOR)
