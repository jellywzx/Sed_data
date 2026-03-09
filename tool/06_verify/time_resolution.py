"""
时间分辨率分类：根据时间间隔推断 frequency。
供 verify_time_resolution 等脚本调用。
"""
import numpy as np
import pandas as pd


def classify_frequency(time_values):
    """根据时间间隔推断频率。"""
    if len(time_values) < 2:
        return "single_point"

    diffs = np.diff(time_values) / np.timedelta64(1, "h")
    median_diff = np.median(diffs)

    if median_diff < 2:
        return "hourly"
    elif median_diff < 36:
        return "daily"
    elif median_diff < 24 * 45:
        return "monthly"
    elif median_diff < 24 * 120:
        return "quarterly"
    elif median_diff < 24 * 500:
        return "annual"
    else:
        return "irregular"
