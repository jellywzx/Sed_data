# core/geo.py
import numpy as np
import pandas as pd
import re


def parse_dms_to_decimal(dms_str):
    if pd.isna(dms_str):
        return np.nan

    parts = re.findall(r'(\d+)', str(dms_str))
    if len(parts) >= 2:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        return degrees + minutes / 60.0 + seconds / 3600.0

    return np.nan

