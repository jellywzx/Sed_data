# core/units.py
import numpy as np


def calculate_discharge(runoff_mm_yr, area_km2):
    if runoff_mm_yr is None or area_km2 is None:
        return np.nan
    return runoff_mm_yr * area_km2 / 31557.6


def calculate_ssl_from_mt_yr(sediment_mt_yr):
    if sediment_mt_yr is None:
        return np.nan
    return sediment_mt_yr * 1e6 / 365.0


def calculate_ssc(ssl_ton_day, discharge_m3s):
    if ssl_ton_day is None or discharge_m3s is None or discharge_m3s <= 0:
        return np.nan
    return ssl_ton_day / (discharge_m3s * 86.4)
