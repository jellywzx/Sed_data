import pandas as pd
import numpy as np
import re
import os
import xarray as xr


FILL_VALUE_FLOAT = np.float32(-9999.0)
FILL_VALUE_INT = np.int8(9)

#=====================================
# time unit conversion
#====================================

def parse_dms_to_decimal(dms_str):
    """
    Convert degrees, minutes, seconds (DMS) to decimal degrees.

    Parameters
    ----------
    dms_str : str
        DMS string in format "DD° MM′ SS″"

    Returns
    -------
    float
        Decimal degrees
    """
    if pd.isna(dms_str):
        return np.nan

    parts = re.findall(r'(\d+)', str(dms_str))
    if len(parts) >= 2:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2]) if len(parts) > 2 else 0.0
        decimal = degrees + minutes/60.0 + seconds/3600.0
        return decimal
    return np.nan


def parse_period(period_str):
    """
    Parse period of record string.

    Parameters
    ----------
    period_str : str
        Period string in format "YYYY-YYYY"

    Returns
    -------
    tuple
        (start_year, end_year) or (None, None) if parsing fails
    """
    if pd.isna(period_str):
        return None, None

    period_str = period_str.replace('–', '-').replace('—', '-')
    parts = period_str.split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


#=====================================
# variable calculations
#====================================

def calculate_discharge(runoff_mm_yr, area_km2):
    """
    Calculate river discharge from runoff and drainage area.

    Formula:
    - Volume (m³/yr) = runoff (mm/yr) × area (km²) × 1000
      (because 1 mm over 1 km² = 1000 m³)
    - Q (m³/s) = Volume / (365.25 days × 86400 s/day)
    - Q (m³/s) = runoff × area × 1000 / 31,557,600
    - Q (m³/s) = runoff × area / 31,557.6
    """
    if pd.isna(runoff_mm_yr) or pd.isna(area_km2):
        return np.nan
    return runoff_mm_yr * area_km2 / 31557.6


def calculate_ssl_from_mt_yr(sediment_mt_yr):
    """
    Convert sediment load from Mt/yr to ton/day.

    Formula:
    - 1 Mt = 10⁶ ton
    - 1 year = 365.25 days
    - SSL (ton/day) = SSL (Mt/yr) × 10⁶ / 365.25
    """
    if pd.isna(sediment_mt_yr):
        return np.nan
    return sediment_mt_yr * 1e6 / 365.25

def convert_ssl_units_if_needed(ssl_da: xr.DataArray):
    """
    如果 sediment_flux 单位为 kg/s，则转换成 ton/day
    - kg/s × 86400 s/day / 1000 kg/ton = ton/day
    - 若无单位或已经是 ton/day，则不改值，只标准化 units 属性
    """
    units = str(getattr(ssl_da, "units", "")).lower()

    if "kg/s" in units or "kg s-1" in units or "kg s^-1" in units:
        LOGGER.info("Converting sediment_flux units from kg/s to ton/day")
        data = ssl_da.values.astype(float)
        converted = data * 86400.0 / 1000.0
        da = ssl_da.copy(data=converted)
        da.attrs["units"] = "ton day-1"
        return da
    else:
        # 不需要转换，统一写成 ton day-1 或沿用原值
        da = ssl_da.copy()
        if units == "" or "ton" in units:
            da.attrs["units"] = "ton day-1"
        return da


def calculate_ssc(ssl_ton_day, discharge_m3s):
    """
    Calculate suspended sediment concentration from sediment load and discharge.

    Formula (derived from standard sediment transport equation):
    - SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4
      where 86.4 = 86400 s/day × 1000 L/m³ × 10⁻⁶ ton/mg
    - Therefore: SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 86.4)
    """
    if pd.isna(ssl_ton_day) or pd.isna(discharge_m3s) or discharge_m3s <= 0:
        return np.nan
    return ssl_ton_day / (discharge_m3s * 0.0864)


#=====================================
# quality control
#====================================

def compute_log_iqr_bounds(values, k=1.5):
    """
    Compute log-space IQR bounds for positive values.

    Parameters
    ----------
    values : array-like
        Input values (must be > 0)
    k : float
        IQR multiplier

    Returns
    -------
    tuple
        (lower_bound, upper_bound) in original space
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) < 5:
        return None, None

    logv = np.log10(values)
    q1 = np.percentile(logv, 25)
    q3 = np.percentile(logv, 75)
    iqr = q3 - q1

    #反log
    lower = 10 ** (q1 - k * iqr)
    upper = 10 ** (q3 + k * iqr)

    return lower, upper

def apply_quality_flag(value, variable_name):
    """
    Apply quality flag based only on missing values and physical impossibility.

    Flag values:
    - 0 = Good data
    - 3 = Bad data (physically impossible)
    - 9 = Missing data

    Notes
    -----
    - No magnitude-based (threshold) screening is applied.
    - Statistical outlier detection, if any, is handled separately.
    """

    # Missing
    if pd.isna(value) or np.isnan(value):
        return np.int8(9)

    # Physical impossibility
    if value < 0:
        return np.int8(3)

    # Otherwise good
    return np.int8(0)

def build_ssc_q_envelope(
    Q_m3s,
    SSC_mgL,
    k=1.5,
    min_samples=5
):
    """
    Build a dataset-level SSC–Q consistency envelope in log–log space
    using standardized physical variables.

    Parameters
    ----------
    Q_m3s : array-like
        River discharge in m3/s (standard unit)
    SSC_mgL : array-like
        Suspended sediment concentration in mg/L (standard unit)
    k : float, optional
        IQR multiplier for envelope width (default: 1.5)
    min_samples : int, optional
        Minimum number of valid samples required

    Returns
    -------
    dict or None
        Dictionary with keys:
        - 'coef'  : linear coefficients [slope, intercept] in log–log space
        - 'lower' : lower residual bound
        - 'upper' : upper residual bound

        Returns None if insufficient valid samples.
    """

    Q = np.asarray(Q_m3s, dtype=float)
    SSC = np.asarray(SSC_mgL, dtype=float)

    valid = (
        np.isfinite(Q)
        & np.isfinite(SSC)
        & (Q > 0)
        & (SSC > 0)
    )

    if valid.sum() < min_samples:
        return None

    logQ = np.log10(Q[valid])
    logSSC = np.log10(SSC[valid])

    # Fit central trend (log–log)
    coef = np.polyfit(logQ, logSSC, 1)
    logSSC_pred = np.polyval(coef, logQ)

    # Residual-based IQR envelope
    resid = logSSC - logSSC_pred
    q1, q3 = np.percentile(resid, [25, 75])
    iqr = q3 - q1

    return {
        "coef": coef,
        "lower": q1 - k * iqr,
        "upper": q3 + k * iqr,
    }



def check_ssc_q_consistency(
    Q,
    SSC,
    Q_flag,
    SSC_flag,
    ssc_q_bounds
    ):
    """
    Check hydrological consistency between SSC and Q using a dataset-level
    log–log SSC–Q envelope.

    Parameters
    ----------
    Q : float
        Discharge (m3/s)
    SSC : float
        Suspended sediment concentration (mg/L)
    Q_flag : int
        Quality flag for Q
    SSC_flag : int
        Quality flag for SSC
    ssc_q_bounds : dict
        Dictionary with keys:
        - 'coef': linear coefficients in log–log space
        - 'lower': lower residual bound
        - 'upper': upper residual bound

    Returns
    -------
    bool
        True if SSC is hydrologically inconsistent with Q, otherwise False
    """
    # Default
    resid = np.nan

    # Only check when both variables are valid and good
    if (
        ssc_q_bounds is None
        or Q_flag != 0
        or SSC_flag != 0
        or pd.isna(Q)
        or pd.isna(SSC)
        or Q <= 0
        or SSC <= 0
    ):
        return False, resid

    logQ = np.log10(Q)
    logSSC = np.log10(SSC)

    coef = ssc_q_bounds["coef"]
    logSSC_expected = coef[0] * logQ + coef[1]

    resid = logSSC - logSSC_expected

    is_inconsistent = (
            resid < ssc_q_bounds["lower"]
            or resid > ssc_q_bounds["upper"]
        )

    return is_inconsistent, resid

#=====================================
# plot_session
#====================================

import matplotlib.pyplot as plt

def plot_ssc_q_diagnostic(
    time,
    Q,
    SSC,
    Q_flag,
    SSC_flag,
    ssc_q_bounds,
    station_id,
    station_name,
    out_png,
):
    """
    Create and save a station-level SSC–Q diagnostic plot.

    Parameters
    ----------
    time : array-like (datetime64)
    Q : array-like (m3/s)
    SSC : array-like (mg/L)
    Q_flag : array-like
    SSC_flag : array-like
    ssc_q_bounds : dict
        Output of build_ssc_q_envelope
    station_id : str
    station_name : str
    out_png : str
        Path to output PNG
    """

    if ssc_q_bounds is None:
        return

    time = np.asarray(time)
    Q = np.asarray(Q, dtype=float)
    SSC = np.asarray(SSC, dtype=float)
    Q_flag = np.asarray(Q_flag)
    SSC_flag = np.asarray(SSC_flag)

    valid = (
        np.isfinite(Q)
        & np.isfinite(SSC)
        & (Q > 0)
        & (SSC > 0)
    )

    if valid.sum() < 5:
        return

    # log–log values
    logQ = np.log10(Q[valid])
    logSSC = np.log10(SSC[valid])

    coef = ssc_q_bounds["coef"]
    logSSC_pred = coef[0] * logQ + coef[1]
    resid = logSSC - logSSC_pred

    # Good vs suspect
    good = (Q_flag[valid] == 0) & (SSC_flag[valid] == 0)
    suspect = (SSC_flag[valid] == 2)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False
    )

    # =========================
    # Panel A: SSC–Q log–log
    # =========================
    ax = axes[0]

    ax.scatter(
        logQ[good], logSSC[good],
        s=20, c="tab:blue", alpha=0.7, label="Good"
    )
    ax.scatter(
        logQ[suspect], logSSC[suspect],
        s=20, c="tab:red", alpha=0.7, label="Suspect"
    )

    # Envelope lines
    x_line = np.linspace(logQ.min(), logQ.max(), 200)
    y_mid = coef[0] * x_line + coef[1]
    y_low = y_mid + ssc_q_bounds["lower"]
    y_up = y_mid + ssc_q_bounds["upper"]

    ax.plot(x_line, y_mid, "k-", lw=2, label="Median trend")
    ax.plot(x_line, y_low, "k--", lw=1)
    ax.plot(x_line, y_up, "k--", lw=1)

    ax.set_xlabel("log10(Q) [m³/s]")
    ax.set_ylabel("log10(SSC) [mg/L]")
    ax.legend(frameon=False)
    ax.set_title(f"SSC–Q diagnostic: {station_name} ({station_id})")

    # =========================
    # Panel B: Residual vs time
    # =========================
    ax2 = axes[1]

    ax2.scatter(
        time[valid][good], resid[good],
        s=15, c="tab:blue", alpha=0.7
    )
    ax2.scatter(
        time[valid][suspect], resid[suspect],
        s=15, c="tab:red", alpha=0.7
    )

    ax2.axhline(0, color="k", lw=1)
    ax2.axhline(ssc_q_bounds["lower"], color="k", ls="--")
    ax2.axhline(ssc_q_bounds["upper"], color="k", ls="--")

    ax2.set_ylabel("Residual (log SSC)")
    ax2.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)




#=====================================
# summary CSV generation
#====================================

def generate_station_summary_csv(station_data, output_dir):
    """Generate a CSV summary file of station metadata and data completeness."""

    csv_file = os.path.join(output_dir, 'ALi_De_Boer_station_summary.csv')

    summary_data = []
    for data in station_data:
        # For climatology data, completeness is either 100% (good data) or 0% (not good)
        Q_complete = 100.0 if data['Q_flag'] == 0 else 0.0
        SSC_complete = 100.0 if data['SSC_flag'] == 0 else 0.0
        SSL_complete = 100.0 if data['SSL_flag'] == 0 else 0.0

        # Date formatting
        Q_start = str(data['start_year']) if data['start_year'] else "N/A"
        Q_end = str(data['end_year']) if data['end_year'] else "N/A"

        # Temporal span
        temporal_span = f"{Q_start}-{Q_end}" if Q_start != "N/A" and Q_end != "N/A" else "N/A"

        # Variables provided (based on data availability)
        vars_provided = []
        if Q_complete > 0:
            vars_provided.append('Q')
        if SSC_complete > 0:
            vars_provided.append('SSC')
        if SSL_complete > 0:
            vars_provided.append('SSL')
        vars_str = ', '.join(vars_provided) if vars_provided else "N/A"

        summary_data.append({
            'station_name': data['station_name'],
            'Source_ID': data['source_id'],
            'river_name': data['river_name'],
            'longitude': f"{data['longitude']:.6f}" if not pd.isna(data['longitude']) else "N/A",
            'latitude': f"{data['latitude']:.6f}" if not pd.isna(data['latitude']) else "N/A",
            'altitude': f"{data['altitude']:.1f}" if not pd.isna(data['altitude']) else "N/A",
            'upstream_area': f"{data['upstream_area']:.1f}" if not pd.isna(data['upstream_area']) else "N/A",
            'Data Source Name': 'ALi_De_Boer Dataset',
            'Type': 'In-situ',
            'Temporal Resolution': 'climatology',
            'Temporal Span': temporal_span,
            'Variables Provided': vars_str,
            'Geographic Coverage': 'Upper Indus River Basin, Northern Pakistan',
            'Reference/DOI': 'https://doi.org/10.1016/j.jhydrol.2006.10.013',
            'Q_start_date': Q_start,
            'Q_end_date': Q_end,
            'Q_percent_complete': f"{Q_complete:.1f}",
            'SSC_start_date': Q_start,
            'SSC_end_date': Q_end,
            'SSC_percent_complete': f"{SSC_complete:.1f}",
            'SSL_start_date': Q_start,
            'SSL_end_date': Q_end,
            'SSL_percent_complete': f"{SSL_complete:.1f}",
        })

    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_file, index=False, encoding='utf-8')

    print(f"\nCreated station summary CSV: {csv_file}")

    return csv_file

