import pandas as pd
import numpy as np
import numpy.ma as ma
import re
import os
import xarray as xr
from netCDF4 import Dataset
from collections import Counter
import logging
LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # 防止在 notebook/重复 import 时重复加 handler
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

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

def check_nc_completeness(
    nc_path,
    required_vars=None,
    required_attrs=None,
    strict=False,
):
    """
    Check whether a NetCDF dataset contains the expected variables and global attributes.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file to validate.
    required_vars : list[str] or None
        Explicit list of variables that must exist. Defaults to standard variables
        used across dataset processors.
    required_attrs : list[str] or None
        Explicit list of global attributes that must exist. Defaults to common
        CF/ACDD metadata fields used in dataset processors.
    strict : bool
        If True, missing extended attributes are treated as errors. If False,
        they are reported as warnings.

    Returns
    -------
    tuple[list[str], list[str]]
        (errors, warnings)
    """
    default_var_requirements = {
        "time": ["units", "calendar"],
        "lat": ["units"],
        "lon": ["units"],
        "Q": ["units", "long_name", "ancillary_variables"],
        "Q_flag": ["flag_values", "flag_meanings"],
        "SSC": ["units", "long_name", "ancillary_variables"],
        "SSC_flag": ["flag_values", "flag_meanings"],
        "SSL": ["units", "long_name", "ancillary_variables"],
        "SSL_flag": ["flag_values", "flag_meanings"],
    }

    default_required_attrs = [
        "Conventions",
        "title",
        "summary",
        "source",
        "data_source_name",
        "variables_provided",
        "number_of_data",
        "reference",
        "source_data_link",
        "creator_name",
        "creator_email",
        "creator_institution",
        "processing_level",
        "comment",
    ]

    strict_attrs = [
        "temporal_resolution",
        "time_coverage_start",
        "time_coverage_end",
        "geographic_coverage",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
    ]

    errors = []
    warnings = []

    required_vars = required_vars or list(default_var_requirements.keys())
    required_attrs = required_attrs or default_required_attrs

    with Dataset(nc_path, mode="r") as ds:
        available_attrs = set(ds.ncattrs())

        for attr in required_attrs:
            if attr not in available_attrs:
                errors.append(f"Missing global attribute: {attr}")

        for attr in strict_attrs:
            if attr not in available_attrs:
                message = f"Missing global attribute: {attr}"
                if strict:
                    errors.append(message)
                else:
                    warnings.append(message)

        for var_name in required_vars:
            if var_name not in ds.variables:
                errors.append(f"Missing variable: {var_name}")
                continue

            attrs_required = default_var_requirements.get(var_name, [])
            var_attrs = set(ds.variables[var_name].ncattrs())
            for attr in attrs_required:
                if attr not in var_attrs:
                    errors.append(
                        f"Variable '{var_name}' missing attribute: {attr}"
                    )

        if "variables_provided" in available_attrs:
            provided = [
                item.strip()
                for item in str(ds.getncattr("variables_provided")).split(",")
                if item.strip()
            ]
            for var_name in provided:
                if var_name not in ds.variables:
                    errors.append(
                        "variables_provided lists missing variable: "
                        f"{var_name}"
                    )

    return errors, warnings

def check_variable_metadata_tiered(
    nc_path,
    *,
    tier: str = "basic",
    extra_requirements: dict | None = None,
    treat_empty_as_missing: bool = True,
    strict_empty: bool = False,
):
    """
    Tiered variable metadata completeness checker.

    Tiers
    -----
    basic:
        - Minimal, low false-positive checks.
        - Focus on: time/lat/lon units + time calendar
                  Q/SSC/SSL units + long_name + ancillary_variables
                  flag vars flag_values + flag_meanings
    recommended:
        - Adds common CF/ACDD-friendly attributes that improve interoperability.
        - Focus on: coordinates, standard_name presence (not validation),
                    axis for coordinate vars, fill_value presence for flags
    strict:
        - Stronger expectations; still no external CF standard_name table.
        - Adds: valid_range (lat/lon), positive for altitude if present,
                comment presence for data vars, standard_name presence for data vars
                (as a requirement; can be strict in some pipelines)

    Parameters
    ----------
    nc_path : str or Path
        NetCDF file to check.
    tier : {"basic","recommended","strict"}
    extra_requirements : dict or None
        Additional/override requirements, merged into tier rules.
        Format: {var_name: {"require": [...], "warn": [...]} }
        - "require" missing -> errors
        - "warn" missing -> warnings
    treat_empty_as_missing : bool
        Whether empty strings / None-like values are treated as missing.
    strict_empty : bool
        If True, empty values are treated as errors (when the key exists).
        If False, empty values become warnings (unless tier/attr is required and you want to enforce).

    Returns
    -------
    errors : list[str]
    warnings : list[str]
    """
    tier = str(tier).lower().strip()
    if tier not in {"basic", "recommended", "strict"}:
        raise ValueError("tier must be one of: 'basic', 'recommended', 'strict'")

    # -----------------------------
    # Tier rule definitions
    # -----------------------------
    # Each var has:
    #   - require: missing -> error
    #   - warn:    missing -> warning
    # Note: "existence" of a variable is checked if it appears in any rule.
    BASIC = {
        "time": {"require": ["units", "calendar"], "warn": ["long_name"]},
        "lat": {"require": ["units"], "warn": ["standard_name", "long_name"]},
        "lon": {"require": ["units"], "warn": ["standard_name", "long_name"]},
        "Q": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "Q_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name"]},
        "SSC": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "SSC_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name"]},
        "SSL": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "SSL_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name"]},
    }

    RECOMMENDED = {
        # coordinates & axis are very common; still keep as warnings unless strict.
        "time": {"require": ["units", "calendar"], "warn": ["axis", "standard_name", "long_name"]},
        "lat": {"require": ["units"], "warn": ["standard_name", "axis", "valid_range", "long_name"]},
        "lon": {"require": ["units"], "warn": ["standard_name", "axis", "valid_range", "long_name"]},
        "Q": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "Q_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        "SSC": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "SSC_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        "SSL": {"require": ["units", "long_name", "ancillary_variables"], "warn": ["standard_name", "coordinates", "comment"]},
        "SSL_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        # optional station scalar vars if present (warn only)
        "altitude": {"require": [], "warn": ["units", "positive", "long_name", "standard_name"]},
        "upstream_area": {"require": [], "warn": ["units", "long_name"]},
    }

    STRICT = {
        # Make more things "require"
        "time": {"require": ["units", "calendar"], "warn": ["axis", "standard_name", "long_name"]},
        "lat": {"require": ["units", "standard_name"], "warn": ["axis", "valid_range", "long_name"]},
        "lon": {"require": ["units", "standard_name"], "warn": ["axis", "valid_range", "long_name"]},
        "Q": {"require": ["units", "long_name", "ancillary_variables", "coordinates"], "warn": ["standard_name", "comment"]},
        "Q_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        "SSC": {"require": ["units", "long_name", "ancillary_variables", "coordinates"], "warn": ["standard_name", "comment"]},
        "SSC_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        "SSL": {"require": ["units", "long_name", "ancillary_variables", "coordinates"], "warn": ["standard_name", "comment"]},
        "SSL_flag": {"require": ["flag_values", "flag_meanings"], "warn": ["long_name", "standard_name", "_FillValue"]},
        # If altitude exists, enforce core attrs
        "altitude": {"require": [], "warn": []},  # handled dynamically below
        "upstream_area": {"require": [], "warn": ["units", "long_name"]},
    }

    rules = BASIC if tier == "basic" else RECOMMENDED if tier == "recommended" else STRICT

    # Merge extra_requirements (override/extend)
    if extra_requirements:
        for vname, spec in extra_requirements.items():
            if vname not in rules:
                rules[vname] = {"require": [], "warn": []}
            for k in ("require", "warn"):
                if k in spec and spec[k]:
                    # merge unique
                    merged = list(dict.fromkeys(list(rules[vname].get(k, [])) + list(spec[k])))
                    rules[vname][k] = merged

    errors = []
    warnings = []

    def _is_empty(val) -> bool:
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    with Dataset(str(nc_path), mode="r") as ds:
        # Dynamic strict rules for altitude if present
        if tier == "strict" and "altitude" in ds.variables:
            STRICT["altitude"] = {"require": ["units", "long_name"], "warn": ["positive", "standard_name", "comment"]}

        # ---------- existence check ----------
        for vname in rules.keys():
            # Only require existence for core vars; for optional vars we can allow missing:
            # Heuristic: if a var has any "require" attrs, treat var itself as required.
            var_is_required = len(rules[vname].get("require", [])) > 0
            if vname not in ds.variables:
                if var_is_required:
                    errors.append(f"Missing required variable: {vname}")
                else:
                    # optional variable absent -> ok
                    continue

        # ---------- attribute check ----------
        for vname, spec in rules.items():
            if vname not in ds.variables:
                continue

            var = ds.variables[vname]
            present_attrs = set(var.ncattrs())

            for attr in spec.get("require", []):
                if attr not in present_attrs:
                    errors.append(f"Variable '{vname}' missing required attribute: {attr}")
                    continue

                if treat_empty_as_missing:
                    try:
                        val = getattr(var, attr)
                    except Exception:
                        val = None
                    if _is_empty(val):
                        msg = f"Variable '{vname}' required attribute '{attr}' is empty/None"
                        if strict_empty:
                            errors.append(msg)
                        else:
                            warnings.append(msg)

            for attr in spec.get("warn", []):
                if attr not in present_attrs:
                    warnings.append(f"Variable '{vname}' missing recommended attribute: {attr}")
                    continue

                if treat_empty_as_missing:
                    try:
                        val = getattr(var, attr)
                    except Exception:
                        val = None
                    if _is_empty(val):
                        warnings.append(f"Variable '{vname}' recommended attribute '{attr}' is empty/None")

        # ---------- light sanity checks (non-fatal) ----------
        # Units should exist and be non-empty for common vars
        for vname in ("Q", "SSC", "SSL", "lat", "lon", "time"):
            if vname in ds.variables and "units" in ds.variables[vname].ncattrs():
                u = str(getattr(ds.variables[vname], "units", "")).strip()
                if u == "":
                    warnings.append(f"Variable '{vname}' has empty 'units' attribute")

    return errors, warnings



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
def summarize_warning_types(stations_info):
    """
    Summarize and print warning types across stations.

    Parameters
    ----------
    stations_info : list[dict]
        Station metadata dictionaries containing "warnings" (pipe-delimited).

    Returns
    -------
    collections.Counter
        Warning counts by warning message.
    """
    counter = Counter()

    for station in stations_info:
        warnings = station.get("warnings")
        if warnings:
            for warning in warnings.split(" | "):
                if warning:
                    counter[warning] += 1

    print("\nMetadata warning type summary:")
    for warning, count in counter.most_common():
        print(f"  {count:4d} × {warning}")

    return counter


def generate_csv_summary(stations_info, output_csv, column_order=None):
    """Generate a station metadata summary CSV."""
    print(f"\n生成CSV摘要文件: {output_csv}")

    if not stations_info:
        print("  ⚠ 警告: 无站点信息可写入CSV")
        return

    df = pd.DataFrame(stations_info)

    if column_order is None:
        column_order = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'altitude', 'upstream_area', 'Data Source Name', 'Type',
            'Temporal Resolution', 'Temporal Span', 'Variables Provided',
            'Geographic Coverage', 'Reference/DOI',
            'Q_start_date', 'Q_end_date', 'Q_percent_complete',
            'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
            'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
        ]

    ordered_cols = [col for col in column_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    if ordered_cols:
        df = df[ordered_cols + remaining_cols]

    df.to_csv(output_csv, index=False)

    print(f"  ✓ CSV文件已生成: {len(df)} 个站点")


def generate_qc_results_csv(stations_info, output_csv, preferred_cols=None):
    """Generate per-station QC results summary CSV."""
    print(f"\n生成站点QC结果汇总CSV: {output_csv}")

    if not stations_info:
        print("  ⚠ 警告: 无站点信息可写入CSV")
        return

    df = pd.DataFrame(stations_info)

    if preferred_cols is None:
        preferred_cols = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'QC_n_days',
            'Q_final_good', 'Q_final_estimated', 'Q_final_suspect', 'Q_final_bad', 'Q_final_missing',
            'SSC_final_good', 'SSC_final_estimated', 'SSC_final_suspect', 'SSC_final_bad', 'SSC_final_missing',
            'SSL_final_good', 'SSL_final_estimated', 'SSL_final_suspect', 'SSL_final_bad', 'SSL_final_missing',
            'Q_qc1_pass', 'Q_qc1_bad', 'Q_qc1_missing',
            'SSC_qc1_pass', 'SSC_qc1_bad', 'SSC_qc1_missing',
            'SSL_qc1_pass', 'SSL_qc1_bad', 'SSL_qc1_missing',
            'Q_qc2_pass', 'Q_qc2_suspect', 'Q_qc2_not_checked', 'Q_qc2_missing',
            'SSC_qc2_pass', 'SSC_qc2_suspect', 'SSC_qc2_not_checked', 'SSC_qc2_missing',
            'SSL_qc2_pass', 'SSL_qc2_suspect', 'SSL_qc2_not_checked', 'SSL_qc2_missing',
            'SSC_qc3_pass', 'SSC_qc3_suspect', 'SSC_qc3_not_checked', 'SSC_qc3_missing',
            'SSL_qc3_not_propagated', 'SSL_qc3_propagated', 'SSL_qc3_not_checked', 'SSL_qc3_missing',
        ]

    cols = [col for col in preferred_cols if col in df.columns]
    if cols:
        df = df[cols]

    df.to_csv(output_csv, index=False)
    print(f"  ✓ QC结果CSV已生成: {len(df)} 个站点")


def generate_warning_summary_csv(stations_info, output_csv):
    """Generate a warning summary CSV for stations with metadata warnings."""
    if not stations_info:
        print("⚠ No station info available for warning summary.")
        return

    rows = []
    for station in stations_info:
        if station.get("n_warnings", 0) > 0:
            rows.append({
                "station_name": station.get("station_name", ""),
                "Source_ID": station.get("Source_ID", ""),
                "n_warnings": station.get("n_warnings", 0),
                "warnings": station.get("warnings", ""),
            })

    if not rows:
        print("✓ No warnings found across all stations.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("n_warnings", ascending=False, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✓ Warning summary CSV written: {output_csv} ({len(df)} stations)")

def generate_station_summary_csv(
    stations_info,
    output_dir,
    *,
    raw_df=None,
    stations=None,
    event_col="event",
    year_col="year",
    vars_cols=("Q", "SSC", "SSL"),
    output_filename=None,
    dataset_name=None,
    data_type="In-situ station data",
    temporal_resolution="daily",
    geographic_coverage=None,
    reference_doi=None,
    extra_columns=None,
    include_all_stations=True,
    encoding="utf-8",
):
    """
    Combined station summary CSV (Coverage + QC usability) - generic version.
    This function REPLACES the old generate_station_summary_csv.

    What it outputs
    ---------------
    For each station, the CSV contains:
    (A) Coverage/availability metrics (from raw_df, NaN-based):
        - <var>_cov_start_date, <var>_cov_end_date, <var>_cov_percent_complete
    (B) QC usability metrics (from stations_info, QC-based):
        - <var>_percent_complete  (good fraction after QC; if final counts exist)
        - <var>_final_* counts if present in stations_info

    Parameters
    ----------
    stations_info : list[dict]
        Per-station dicts (built in your process scripts). Can be empty if you
        still want coverage-only output (when raw_df+stations provided).
    output_dir : str
        Output directory for CSV.
    raw_df : pd.DataFrame or None
        Raw per-record table (after cleaning/unit conversion), used to compute
        coverage completeness like generate_summary_csv did.
        Must include columns: event_col, year_col, and vars in vars_cols.
    stations : dict or None
        Station metadata mapping event_id -> {"station_id","lat","lon","river"}.
        Same structure as parse_station_metadata() output in your process.py.
    include_all_stations : bool
        If True and raw_df+stations are provided, include stations even if they
        are not present in stations_info (e.g., no sediment stations).
    """

    import os
    import pandas as pd
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Helpers
    # -------------------------
    def _get(d, keys, default=None):
        for k in keys:
            if k in d:
                v = d.get(k)
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == "":
                    continue
                return v
        return default

    def _to_float(x):
        try:
            if x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _to_int(x):
        try:
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return None
            return int(float(x))
        except Exception:
            return None

    def _fmt_coord(x, nd=6):
        v = _to_float(x)
        if not np.isfinite(v):
            return ""
        return f"{v:.{nd}f}"

    def _fmt_num(x, nd=1):
        v = _to_float(x)
        if not np.isfinite(v):
            return ""
        return f"{v:.{nd}f}"

    def _calc_qc_good_percent(d, var_prefix):
        """
        QC percent complete = good / (good+estimated+suspect+bad+missing),
        when final counts exist. Otherwise fallback:
        - if <var>_percent_complete exists, use it
        - else if <var>_flag exists: 0 -> 100, else -> 0
        """
        good = _to_int(_get(d, [f"{var_prefix}_final_good"]))
        est  = _to_int(_get(d, [f"{var_prefix}_final_estimated"]))
        sus  = _to_int(_get(d, [f"{var_prefix}_final_suspect"]))
        bad  = _to_int(_get(d, [f"{var_prefix}_final_bad"]))
        miss = _to_int(_get(d, [f"{var_prefix}_final_missing"]))

        if good is not None and miss is not None:
            total = sum([x for x in [good, est, sus, bad, miss] if x is not None])
            if total > 0:
                return 100.0 * good / total
            return 0.0

        direct = _get(d, [f"{var_prefix}_percent_complete", f"{var_prefix}_pct_complete"])
        if direct is not None:
            try:
                return float(direct)
            except Exception:
                pass

        flag = _get(d, [f"{var_prefix}_flag", f"{var_prefix.lower()}_flag"])
        try:
            return 100.0 if int(flag) == 0 else 0.0
        except Exception:
            return 0.0

    def _infer_vars_provided(qp, sp, lp):
        vars_ = []
        if qp > 0:
            vars_.append("Q")
        if sp > 0:
            vars_.append("SSC")
        if lp > 0:
            vars_.append("SSL")
        return ", ".join(vars_) if vars_ else ""

    # -------------------------
    # 1) Build coverage table from raw_df + stations (like old generate_summary_csv)
    # -------------------------
    coverage_by_sourceid = {}
    if raw_df is not None and stations is not None:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame or None.")
        for event_id, meta in stations.items():
            sid = str(meta.get("station_id", "")).strip()
            if sid == "":
                continue
            source_id = sid.replace(".", "_")

            sub = raw_df[raw_df[event_col] == event_id]
            if sub.empty:
                continue

            n_total = len(sub)

            cov = {
                "QC_n_days_raw": int(n_total),
                "Temporal Span (raw)": "",
            }

            # overall span
            try:
                y_all = pd.to_numeric(sub[year_col], errors="coerce")
                if np.isfinite(y_all).any():
                    cov["Temporal Span (raw)"] = f"{int(np.nanmin(y_all))}-{int(np.nanmax(y_all))}"
            except Exception:
                pass

            for v in vars_cols:
                if v not in sub.columns:
                    cov[f"{v}_cov_start_date"] = ""
                    cov[f"{v}_cov_end_date"] = ""
                    cov[f"{v}_cov_percent_complete"] = "0.0"
                    continue

                ser = sub[v]
                valid = ser.dropna()
                if valid.empty:
                    cov[f"{v}_cov_start_date"] = ""
                    cov[f"{v}_cov_end_date"] = ""
                    cov[f"{v}_cov_percent_complete"] = "0.0"
                else:
                    idx = valid.index
                    try:
                        y = pd.to_numeric(sub.loc[idx, year_col], errors="coerce")
                        cov[f"{v}_cov_start_date"] = int(np.nanmin(y)) if np.isfinite(y).any() else ""
                        cov[f"{v}_cov_end_date"] = int(np.nanmax(y)) if np.isfinite(y).any() else ""
                    except Exception:
                        cov[f"{v}_cov_start_date"] = ""
                        cov[f"{v}_cov_end_date"] = ""

                    cov_pct = 100.0 * len(valid) / n_total if n_total > 0 else 0.0
                    cov[f"{v}_cov_percent_complete"] = f"{cov_pct:.1f}"

            # also store basic meta
            cov.update({
                "station_name": sid,
                "Source_ID": source_id,
                "river_name": meta.get("river", ""),
                "longitude": meta.get("lon", np.nan),
                "latitude": meta.get("lat", np.nan),
            })

            coverage_by_sourceid[source_id] = cov

    # -------------------------
    # 2) Build QC table from stations_info (like old station_summary_csv)
    # -------------------------
    qc_by_sourceid = {}
    if stations_info:
        for d in stations_info:
            sid = _get(d, ["Source_ID", "source_id", "SourceID", "id"], "")
            if sid == "":
                # fallback: from station_name
                name = _get(d, ["station_name", "station", "name"], "")
                sid = str(name).replace(".", "_") if name else ""
            if sid == "":
                continue

            qc_by_sourceid[sid] = d

    # -------------------------
    # 3) Merge keys: either from coverage-only, qc-only, or both
    # -------------------------
    all_source_ids = set()
    all_source_ids.update(coverage_by_sourceid.keys())
    all_source_ids.update(qc_by_sourceid.keys())

    if not include_all_stations and stations_info:
        # only output stations that appear in stations_info
        all_source_ids = set(qc_by_sourceid.keys())

    # If neither provided, nothing to do
    if not all_source_ids:
        print("  ⚠ generate_station_summary_csv: no stations to write.")
        return ""

    rows = []
    for source_id in sorted(all_source_ids):
        cov = coverage_by_sourceid.get(source_id, {})
        qc  = qc_by_sourceid.get(source_id, {})

        station_name = _get(qc, ["station_name", "station", "name"], _get(cov, ["station_name"], ""))
        river_name   = _get(qc, ["river_name", "river"], _get(cov, ["river_name"], ""))

        lon = _get(qc, ["longitude", "lon"], _get(cov, ["longitude"], np.nan))
        lat = _get(qc, ["latitude", "lat"], _get(cov, ["latitude"], np.nan))
        alt = _get(qc, ["altitude", "elevation", "alt"], np.nan)
        area = _get(qc, ["upstream_area", "drainage_area", "area_km2"], np.nan)

        # QC-based completeness (% good after QC)
        q_qc_pct   = _calc_qc_good_percent(qc, "Q")
        ssc_qc_pct = _calc_qc_good_percent(qc, "SSC")
        ssl_qc_pct = _calc_qc_good_percent(qc, "SSL")

        # Variables provided: prefer explicit -> else infer from QC pct (or coverage if QC missing)
        vars_col = _get(qc, ["Variables Provided", "variables_provided"], "")
        if not vars_col:
            # if QC data absent, infer from coverage pct
            q_cov_pct = float(_get(cov, ["Q_cov_percent_complete"], "0.0") or "0.0")
            s_cov_pct = float(_get(cov, ["SSC_cov_percent_complete"], "0.0") or "0.0")
            l_cov_pct = float(_get(cov, ["SSL_cov_percent_complete"], "0.0") or "0.0")
            base_q = q_qc_pct if qc else q_cov_pct
            base_s = ssc_qc_pct if qc else s_cov_pct
            base_l = ssl_qc_pct if qc else l_cov_pct
            vars_col = _infer_vars_provided(base_q, base_s, base_l)

        # Temporal span: prefer raw coverage span -> else from qc start/end
        temporal_span = _get(cov, ["Temporal Span (raw)"], "")
        if not temporal_span:
            start_year = _get(qc, ["start_year", "Q_start_date"], "")
            end_year   = _get(qc, ["end_year", "Q_end_date"], "")
            if str(start_year) != "" and str(end_year) != "":
                temporal_span = f"{start_year}-{end_year}"
            else:
                temporal_span = _get(qc, ["Temporal Span", "temporal_span"], "")

        geo_cov = geographic_coverage if geographic_coverage is not None else _get(
            qc, ["Geographic Coverage", "geographic_coverage"], ""
        )
        ref = reference_doi if reference_doi is not None else _get(
            qc, ["Reference/DOI", "reference", "doi", "source_data_link"], ""
        )
        ds_name = dataset_name if dataset_name is not None else _get(
            qc, ["Data Source Name", "data_source_name"], ""
        )

        row = {
            "station_name": station_name,
            "Source_ID": source_id,
            "river_name": river_name,
            "longitude": _fmt_coord(lon),
            "latitude": _fmt_coord(lat),
            "altitude": _fmt_num(alt, nd=1),
            "upstream_area": _fmt_num(area, nd=1),

            "Data Source Name": ds_name,
            "Type": data_type,
            "Temporal Resolution": temporal_resolution,
            "Temporal Span": temporal_span,
            "Variables Provided": vars_col,
            "Geographic Coverage": geo_cov,
            "Reference/DOI": ref,

            # --- QC usability (good %) ---
            "Q_percent_complete": f"{q_qc_pct:.1f}",
            "SSC_percent_complete": f"{ssc_qc_pct:.1f}",
            "SSL_percent_complete": f"{ssl_qc_pct:.1f}",
        }

        # --- Coverage (raw non-NaN %) ---
        for v in vars_cols:
            row[f"{v}_cov_start_date"] = _get(cov, [f"{v}_cov_start_date"], "")
            row[f"{v}_cov_end_date"] = _get(cov, [f"{v}_cov_end_date"], "")
            row[f"{v}_cov_percent_complete"] = _get(cov, [f"{v}_cov_percent_complete"], "0.0")

        # Optional: keep QC final counts if present (very useful)
        for v in vars_cols:
            for k in ("good", "estimated", "suspect", "bad", "missing"):
                kk = f"{v}_final_{k}"
                if kk in qc:
                    row[kk] = qc.get(kk)

        # Optional: include QC_n_days if present (from qc dict)
        if "QC_n_days" in qc:
            row["QC_n_days"] = qc.get("QC_n_days")
        if "QC_n_days_raw" in cov:
            row["Raw_n_records"] = cov.get("QC_n_days_raw")

        # Extra columns passthrough
        if extra_columns:
            for c in extra_columns:
                if c not in row:
                    val = _get(qc, [c], None)
                    if val is None:
                        val = _get(cov, [c], None)
                    if val is not None:
                        row[c] = val

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Column order: core -> QC -> coverage -> extras
    base_cols = [
        "station_name", "Source_ID", "river_name", "longitude", "latitude",
        "altitude", "upstream_area",
        "Data Source Name", "Type", "Temporal Resolution", "Temporal Span",
        "Variables Provided", "Geographic Coverage", "Reference/DOI",
        "Q_percent_complete", "SSC_percent_complete", "SSL_percent_complete",
    ]
    cov_cols = []
    for v in vars_cols:
        cov_cols += [f"{v}_cov_start_date", f"{v}_cov_end_date", f"{v}_cov_percent_complete"]
    qc_count_cols = []
    for v in vars_cols:
        qc_count_cols += [f"{v}_final_good", f"{v}_final_estimated", f"{v}_final_suspect", f"{v}_final_bad", f"{v}_final_missing"]

    front = [c for c in base_cols if c in df_out.columns]
    mid = [c for c in qc_count_cols if c in df_out.columns] + [c for c in ("QC_n_days", "Raw_n_records") if c in df_out.columns]
    back = [c for c in cov_cols if c in df_out.columns]
    remaining = [c for c in df_out.columns if c not in (front + mid + back)]

    df_out = df_out[front + mid + back + remaining]

    # Output filename
    if output_filename is None:
        base = "station_summary_combined.csv"
        if dataset_name and isinstance(dataset_name, str) and dataset_name.strip():
            base = f"{dataset_name.replace(' ', '_')}_station_summary_combined.csv"
        output_filename = base

    out_csv = os.path.join(output_dir, output_filename)
    df_out.to_csv(out_csv, index=False, encoding=encoding)

    print(f"  ✓ Combined station summary CSV written: {out_csv} ({len(df_out)} stations)")
    return out_csv
