import pandas as pd
import numpy as np
import numpy.ma as ma
import re
import os
import xarray as xr


FILL_VALUE_FLOAT = np.float32(-9999.0)
FILL_VALUE_INT = np.int8(9)
NOT_CHECKED_INT = np.int8(8)
ESTIMATED_INT = np.int8(1)  # derived/estimated data


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
    if ma.isMaskedArray(values):
        values = ma.filled(values, np.nan)
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


def apply_log_iqr_screening(
    values,
    base_flag,
    k=1.5,
    min_samples=5,
    suspect_flag=np.int8(2),
    pass_flag=np.int8(0),
    missing_flag=FILL_VALUE_INT,
    not_checked_flag=NOT_CHECKED_INT,
):
    """
    Apply log-IQR screening WITHOUT overriding upstream QC failures.

    This helper is designed to be reusable across datasets/pipelines:
    - It only evaluates points where base_flag == pass_flag (default 0),
      so upstream flags like "bad=3" will never be overwritten as "suspect=2".
    - It distinguishes missing vs not_checked at the step level:
      - missing_flag (default 9): value is missing/fill/NaN
      - not_checked_flag (default 8): not evaluated (e.g., <=0, failed upstream QC,
        insufficient samples, or bounds not computed)

    Parameters
    ----------
    values : array-like
        Data values.
    base_flag : array-like (int)
        Upstream QC flag array (same length as values).
    k : float
        IQR multiplier in log space.
    min_samples : int
        Minimum number of evaluable samples required to compute bounds.

    Returns
    -------
    step_flag : np.ndarray (int8)
        Step-level provenance flag (pass/suspect/not_checked/missing).
    updated_flag : np.ndarray (int8)
        Updated final flag array (base_flag with suspect applied where appropriate).
    bounds : tuple
        (lower, upper) bounds in original space; (None, None) if not computed.
    """
    if ma.isMaskedArray(values):
        values = ma.filled(values, np.nan)
    v = np.asarray(values, dtype=float)
    f = np.asarray(base_flag, dtype=np.int8)
    n = len(v)

    step_flag = np.full(n, not_checked_flag, dtype=np.int8)

    # Missing detection: respect upstream missing flag and also guard against NaNs/fill.
    missing_mask = (
        (f == missing_flag)
        | ~np.isfinite(v)
        | np.isclose(v, FILL_VALUE_FLOAT, rtol=1e-5, atol=1e-5)
    )
    step_flag[missing_mask] = missing_flag

    # Evaluate only where upstream says "good" and value is strictly positive
    eval_mask = (f == pass_flag) & np.isfinite(v) & (v > 0)

    if eval_mask.sum() < int(min_samples):
        return step_flag, f.copy(), (None, None)

    lower, upper = compute_log_iqr_bounds(v[eval_mask], k=k)
    if lower is None:
        return step_flag, f.copy(), (None, None)

    step_flag[eval_mask] = pass_flag
    outlier_mask = eval_mask & ((v < lower) | (v > upper))
    step_flag[outlier_mask] = suspect_flag

    updated_flag = f.copy()
    # Key rule: only downgrade points that are currently "good"
    updated_flag[outlier_mask] = suspect_flag

    return step_flag, updated_flag, (lower, upper)

def apply_qc2_log_iqr_if_independent(
    values,
    base_flag,
    is_independent: bool,
    *,
    k=1.5,
    min_samples=5,
    pass_flag=np.int8(0),
    suspect_flag=np.int8(2),
    estimated_flag=ESTIMATED_INT,
    bad_flag=np.int8(3),
    missing_flag=FILL_VALUE_INT,
    not_checked_flag=NOT_CHECKED_INT,
    fill_value_float=FILL_VALUE_FLOAT,
):
    """
    QC2: log-IQR screening applied ONLY to independent observations.

    If is_independent == True:
        - run apply_log_iqr_screening(values, base_flag)
        - return (qc2_step_flag, updated_flag, bounds)

    If is_independent == False (derived / estimated variable):
        - QC2 is NOT applied:
            qc2_step_flag:
                - missing -> 9
                - otherwise -> 8 (not_checked)
        - final flag:
            - downgrade ONLY "good(0)" points to "estimated(1)"
            - keep suspect(2)/bad(3)/missing(9) unchanged
        - bounds -> (None, None)

    Notes
    -----
    - base_flag is the cumulative flag from upstream steps (typically QC1 or QC1+QCX).
    - This function NEVER overwrites upstream bad/missing/suspect.
    """

    # normalize
    if ma.isMaskedArray(values):
        values = ma.filled(values, np.nan)
    v = np.asarray(values, dtype=float)
    f = np.asarray(base_flag, dtype=np.int8)
    n = len(v)

    # missing detection (same spirit as apply_log_iqr_screening)
    missing_mask = (
        (f == missing_flag)
        | ~np.isfinite(v)
        | np.isclose(v, float(fill_value_float), rtol=1e-5, atol=1e-5)
    )

    # -------------------------
    # Case A: independent -> real QC2
    # -------------------------
    if bool(is_independent):
        qc2_step_flag, updated_flag, bounds = apply_log_iqr_screening(
            values=v,
            base_flag=f,
            k=k,
            min_samples=min_samples,
            suspect_flag=suspect_flag,
            pass_flag=pass_flag,
            missing_flag=missing_flag,
            not_checked_flag=not_checked_flag,
        )
        return qc2_step_flag, updated_flag, bounds

    # -------------------------
    # Case B: derived -> no QC2, mark estimated
    # -------------------------
    qc2_step_flag = np.full(n, not_checked_flag, dtype=np.int8)
    qc2_step_flag[missing_mask] = missing_flag

    updated_flag = f.copy()

    # Only turn "good(0)" into "estimated(1)" for non-missing points
    mark_est_mask = (updated_flag == pass_flag) & (~missing_mask)

    # Important: do NOT overwrite suspect(2)/bad(3)/missing(9)
    updated_flag[mark_est_mask] = estimated_flag

    return qc2_step_flag, updated_flag, (None, None)


def apply_quality_flag_array(values, variable_name=""):
    """
    Vectorized wrapper for apply_quality_flag (returns int8 flag array).

    This exists so dataset scripts can reuse the exact same QC1 logic without
    re-implementing list comprehensions.
    """
    if ma.isMaskedArray(values):
        values = ma.filled(values, np.nan)
    arr = np.asarray(values)
    return np.array([apply_quality_flag(v, variable_name) for v in arr], dtype=np.int8)


def apply_hydro_qc_with_provenance(
    time,
    Q,
    SSC,
    SSL,
    *,
    Q_is_independent: bool = True,
    SSC_is_independent: bool = True,
    SSL_is_independent: bool = False,
    ssl_is_derived_from_q_ssc: bool = True,
    qc2_k: float = 1.5,
    qc2_min_samples: int = 5,
    qc3_k: float = 1.5,
    qc3_min_samples: int = 5,
):
    """
    End-to-end QC pipeline (QC1/QC2/QC3) with step-level provenance flags.

    Designed for reuse across datasets that provide:
    - Q   : discharge (independent or not)
    - SSC : suspended sediment concentration (independent or not)
    - SSL : suspended sediment load (often derived from Q×SSC)

    Flag conventions
    ----------------
    Final flags (Q_flag/SSC_flag/SSL_flag):
    - 0 good, 1 estimated (typically derived), 2 suspect, 3 bad, 9 missing

    Step flags:
    - QC1 physical: 0 pass, 3 bad, 9 missing
    - QC2 log-IQR: 0 pass, 2 suspect, 8 not_checked, 9 missing
    - QC3 SSC–Q:  0 pass, 2 suspect, 8 not_checked, 9 missing
    - QC3 SSL propagation: 2 propagated, 0 not_propagated, 8 not_checked, 9 missing

    Returns
    -------
    dict or None
        Dict contains trimmed arrays (valid_time) and all flags.
        Returns None if no valid time remains.
    """
    time = np.asarray(time)
    n = len(time)

    # Normalize values for numeric operations (preserve NaNs)
    def _to_float_array(x):
        if ma.isMaskedArray(x):
            x = ma.filled(x, np.nan)
        return np.asarray(x, dtype=float)

    Qv = _to_float_array(Q)
    SSCv = _to_float_array(SSC)
    SSLv = _to_float_array(SSL)

    # -----------------------------
    # QC1. Physical feasibility / missing
    # -----------------------------
    Q_flag_qc1_physical = apply_quality_flag_array(Qv, "Q")
    SSC_flag_qc1_physical = apply_quality_flag_array(SSCv, "SSC")
    SSL_flag_qc1_physical = apply_quality_flag_array(SSLv, "SSL")

    Q_flag = Q_flag_qc1_physical.copy()
    SSC_flag = SSC_flag_qc1_physical.copy()
    SSL_flag = SSL_flag_qc1_physical.copy()

    # -----------------------------
    # QC2. log-IQR screening (only for independent observations)
    # -----------------------------
    Q_flag_qc2_log_iqr, Q_flag, _ = apply_qc2_log_iqr_if_independent(
        values=Qv,
        base_flag=Q_flag,
        is_independent=Q_is_independent,
        k=qc2_k,
        min_samples=qc2_min_samples,
    )
    SSC_flag_qc2_log_iqr, SSC_flag, _ = apply_qc2_log_iqr_if_independent(
        values=SSCv,
        base_flag=SSC_flag,
        is_independent=SSC_is_independent,
        k=qc2_k,
        min_samples=qc2_min_samples,
    )
    SSL_flag_qc2_log_iqr, SSL_flag, _ = apply_qc2_log_iqr_if_independent(
        values=SSLv,
        base_flag=SSL_flag,
        is_independent=SSL_is_independent,
        k=qc2_k,
        min_samples=qc2_min_samples,
    )

    # -----------------------------
    # QC3. SSC–Q consistency + propagate to SSL if derived
    # -----------------------------
    SSC_flag_qc3_ssc_q = np.full(n, NOT_CHECKED_INT, dtype=np.int8)
    SSL_flag_qc3_from_ssc_q = np.full(n, NOT_CHECKED_INT, dtype=np.int8)

    # Mark missing explicitly (distinguish from not_checked)
    SSC_flag_qc3_ssc_q[SSC_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT
    SSL_flag_qc3_from_ssc_q[SSL_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT

    env_mask = (
        (Q_flag == 0)
        & (SSC_flag == 0)
        & np.isfinite(Qv)
        & np.isfinite(SSCv)
        & (Qv > 0)
        & (SSCv > 0)
    )

    Q_env = np.where(env_mask, Qv, np.nan)
    SSC_env = np.where(env_mask, SSCv, np.nan)
    ssc_q_bounds = build_ssc_q_envelope(Q_env, SSC_env, k=qc3_k, min_samples=qc3_min_samples)

    if ssc_q_bounds is not None:
        SSC_flag_qc3_ssc_q[env_mask] = np.int8(0)

        for i in np.where(env_mask)[0]:
            inconsistent, _ = check_ssc_q_consistency(
                Qv[i],
                SSCv[i],
                Q_flag[i],
                SSC_flag[i],
                ssc_q_bounds,
            )

            if not inconsistent:
                continue

            # SSC downgrade
            SSC_flag_qc3_ssc_q[i] = np.int8(2)
            SSC_flag[i] = np.int8(2)

            # Propagate to SSL (optional)
            prev_ssl_flag = SSL_flag[i]
            SSL_flag[i] = propagate_ssc_q_inconsistency_to_ssl(
                inconsistent=True,
                Q=Qv[i],
                SSC=SSCv[i],
                SSL=SSLv[i],
                Q_flag=Q_flag[i],
                SSC_flag=np.int8(0),  # use pre-downgrade SSC state for propagation logic
                SSL_flag=prev_ssl_flag,
                ssl_is_derived_from_q_ssc=ssl_is_derived_from_q_ssc,
            )

            # Record whether propagation actually downgraded SSL to suspect
            SSL_flag_qc3_from_ssc_q[i] = (
                np.int8(2)
                if (prev_ssl_flag in (np.int8(0), ESTIMATED_INT) and SSL_flag[i] == np.int8(2))
                else np.int8(0)
            )

    # -----------------------------
    # Valid-time mask: keep days where ANY variable is non-missing
    # -----------------------------
    valid_time = (Q_flag != FILL_VALUE_INT) | (SSC_flag != FILL_VALUE_INT) | (SSL_flag != FILL_VALUE_INT)
    if not np.any(valid_time):
        return None

    return {
        "time": time[valid_time],
        "Q": Qv[valid_time],
        "SSC": SSCv[valid_time],
        "SSL": SSLv[valid_time],
        "Q_flag": Q_flag[valid_time],
        "SSC_flag": SSC_flag[valid_time],
        "SSL_flag": SSL_flag[valid_time],
        # Step-level provenance flags
        "Q_flag_qc1_physical": Q_flag_qc1_physical[valid_time],
        "SSC_flag_qc1_physical": SSC_flag_qc1_physical[valid_time],
        "SSL_flag_qc1_physical": SSL_flag_qc1_physical[valid_time],
        "Q_flag_qc2_log_iqr": Q_flag_qc2_log_iqr[valid_time],
        "SSC_flag_qc2_log_iqr": SSC_flag_qc2_log_iqr[valid_time],
        "SSL_flag_qc2_log_iqr": SSL_flag_qc2_log_iqr[valid_time],
        "SSC_flag_qc3_ssc_q": SSC_flag_qc3_ssc_q[valid_time],
        "SSL_flag_qc3_from_ssc_q": SSL_flag_qc3_from_ssc_q[valid_time],
        # Extra (useful for plotting/debug)
        "ssc_q_bounds": ssc_q_bounds,
    }


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

    # Robust missing/invalid handling:
    # - netCDF often yields masked values (np.ma.masked / MaskedConstant)
    # - some datasets may contain non-numeric objects
    if value is None or ma.is_masked(value):
        return np.int8(9)

    # Convert to float safely; non-convertible values are treated as missing
    try:
        v = float(value)
    except Exception:
        return np.int8(9)

    # Missing / fill / non-finite
    if not np.isfinite(v):
        return np.int8(9)
    if np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5):
        return np.int8(9)

    # Physical impossibility
    if v < 0:
        return np.int8(3)

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


def propagate_ssc_q_inconsistency_to_ssl(
    inconsistent,
    Q,
    SSC,
    SSL,
    Q_flag,
    SSC_flag,
    SSL_flag,
    ssl_is_derived_from_q_ssc
):
    """
    Propagate SSC–Q hydrological inconsistency to derived sediment load (SSL)
    ONLY when SSL is computed from Q and SSC.

    Conditions
    ----------
    Propagation is applied ONLY if:
    1. SSC–Q inconsistency is detected
    2. Q, SSC, and SSL are all present and valid
    3. SSL is derived from Q and SSC (not independently observed)

    Rationale
    ---------
    SSL derived from Q × SSC inherits uncertainties from both variables.
    If SSC is hydrologically inconsistent with Q, SSL cannot be considered 'good'.

    Parameters
    ----------
    inconsistent : bool
        Output from check_ssc_q_consistency
    Q, SSC, SSL : float
        Data values
    Q_flag, SSC_flag, SSL_flag : int
        Quality flags
    ssl_is_derived_from_q_ssc : bool
        True if SSL is computed from Q and SSC

    Returns
    -------
    int
        Updated SSL_flag
    """

    # --------------------------------------------------
    # Condition 0: must be flagged inconsistent
    # --------------------------------------------------
    if not inconsistent:
        return SSL_flag

    # --------------------------------------------------
    # Condition 1: SSL must be derived, not observed
    # --------------------------------------------------
    if not ssl_is_derived_from_q_ssc:
        return SSL_flag

    # --------------------------------------------------
    # Condition 2: all variables must exist and be valid
    # --------------------------------------------------
    if (
        Q_flag != 0
        or SSC_flag != 0
        or SSL_flag == FILL_VALUE_INT
        or pd.isna(Q)
        or pd.isna(SSC)
        or pd.isna(SSL)
        or Q <= 0
        or SSC <= 0
    ):
        return SSL_flag

    # --------------------------------------------------
    # Propagation: downgrade SSL from good → suspect
    # --------------------------------------------------
    if SSL_flag in (np.int8(0), ESTIMATED_INT):
        return np.int8(2)

    return SSL_flag

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

