import numpy as np
import numpy.ma as ma

from constants import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    FLAG_BAD,
    FLAG_ESTIMATED,
    FLAG_GOOD,
    FLAG_MISSING,
    FLAG_NOT_CHECKED,
    FLAG_SUSPECT,
)


def _as_float_array(values):
    if ma.isMaskedArray(values):
        values = ma.filled(values, np.nan)
    return np.asarray(values, dtype=float)


def apply_quality_flag(value, variable_name=None):
    """
    Shared QC1 implementation.

    The optional ``variable_name`` argument is accepted for compatibility with
    the historical ``tool.py`` signature used by dataset scripts.
    """
    del variable_name

    if value is None or ma.is_masked(value):
        return FLAG_MISSING

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return FLAG_MISSING

    if not np.isfinite(numeric):
        return FLAG_MISSING
    if np.isclose(numeric, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5):
        return FLAG_MISSING
    if numeric < 0:
        return FLAG_BAD
    return FLAG_GOOD


def apply_quality_flag_array(values, variable_name=""):
    arr = _as_float_array(values)
    return np.array(
        [apply_quality_flag(value, variable_name=variable_name) for value in arr],
        dtype=np.int8,
    )


def compute_log_iqr_bounds(values, k=1.5):
    values = _as_float_array(values)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 5:
        return None, None

    logv = np.log10(values)
    q1 = np.percentile(logv, 25)
    q3 = np.percentile(logv, 75)
    iqr = q3 - q1

    return 10 ** (q1 - k * iqr), 10 ** (q3 + k * iqr)


def apply_ssl_log_iqr_flag(ssl_mt_yr, bounds, base_flag):
    if base_flag != FLAG_GOOD or bounds is None or bounds[0] is None:
        return base_flag

    lower, upper = bounds
    if ssl_mt_yr > 0 and (ssl_mt_yr < lower or ssl_mt_yr > upper):
        return FLAG_SUSPECT

    return base_flag


def apply_log_iqr_screening(
    values,
    base_flag,
    k=1.5,
    min_samples=5,
    suspect_flag=FLAG_SUSPECT,
    pass_flag=FLAG_GOOD,
    missing_flag=FILL_VALUE_INT,
    not_checked_flag=FLAG_NOT_CHECKED,
):
    v = _as_float_array(values)
    f = np.asarray(base_flag, dtype=np.int8)
    n = len(v)

    step_flag = np.full(n, not_checked_flag, dtype=np.int8)
    missing_mask = (
        (f == missing_flag)
        | ~np.isfinite(v)
        | np.isclose(v, FILL_VALUE_FLOAT, rtol=1e-5, atol=1e-5)
    )
    step_flag[missing_mask] = missing_flag

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
    updated_flag[outlier_mask] = suspect_flag
    return step_flag, updated_flag, (lower, upper)


def apply_qc2_log_iqr_if_independent(
    values,
    base_flag,
    is_independent,
    *,
    k=1.5,
    min_samples=5,
    pass_flag=FLAG_GOOD,
    suspect_flag=FLAG_SUSPECT,
    estimated_flag=FLAG_ESTIMATED,
    missing_flag=FILL_VALUE_INT,
    not_checked_flag=FLAG_NOT_CHECKED,
    fill_value_float=FILL_VALUE_FLOAT,
):
    v = _as_float_array(values)
    f = np.asarray(base_flag, dtype=np.int8)
    n = len(v)

    missing_mask = (
        (f == missing_flag)
        | ~np.isfinite(v)
        | np.isclose(v, float(fill_value_float), rtol=1e-5, atol=1e-5)
    )

    if bool(is_independent):
        return apply_log_iqr_screening(
            values=v,
            base_flag=f,
            k=k,
            min_samples=min_samples,
            suspect_flag=suspect_flag,
            pass_flag=pass_flag,
            missing_flag=missing_flag,
            not_checked_flag=not_checked_flag,
        )

    qc2_step_flag = np.full(n, not_checked_flag, dtype=np.int8)
    qc2_step_flag[missing_mask] = missing_flag

    updated_flag = f.copy()
    mark_est_mask = (updated_flag == pass_flag) & (~missing_mask)
    updated_flag[mark_est_mask] = estimated_flag
    return qc2_step_flag, updated_flag, (None, None)


def build_ssc_q_envelope(Q_m3s, SSC_mgL, k=1.5, min_samples=5):
    Q = _as_float_array(Q_m3s)
    SSC = _as_float_array(SSC_mgL)

    valid = np.isfinite(Q) & np.isfinite(SSC) & (Q > 0) & (SSC > 0)
    if valid.sum() < min_samples:
        return None

    logQ = np.log10(Q[valid])
    logSSC = np.log10(SSC[valid])
    coef = np.polyfit(logQ, logSSC, 1)
    logSSC_pred = np.polyval(coef, logQ)

    resid = logSSC - logSSC_pred
    q1, q3 = np.percentile(resid, [25, 75])
    iqr = q3 - q1

    return {
        "coef": coef,
        "lower": q1 - k * iqr,
        "upper": q3 + k * iqr,
    }


def check_ssc_q_consistency(Q, SSC, Q_flag, SSC_flag, ssc_q_bounds):
    resid = np.nan
    if (
        ssc_q_bounds is None
        or Q_flag != FLAG_GOOD
        or SSC_flag != FLAG_GOOD
        or not np.isfinite(Q)
        or not np.isfinite(SSC)
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
        resid < ssc_q_bounds["lower"] or resid > ssc_q_bounds["upper"]
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
    ssl_is_derived_from_q_ssc,
):
    if not inconsistent:
        return SSL_flag
    if not ssl_is_derived_from_q_ssc:
        return SSL_flag
    if (
        Q_flag != FLAG_GOOD
        or SSC_flag != FLAG_GOOD
        or SSL_flag == FILL_VALUE_INT
        or not np.isfinite(Q)
        or not np.isfinite(SSC)
        or not np.isfinite(SSL)
        or Q <= 0
        or SSC <= 0
    ):
        return SSL_flag
    if SSL_flag in (FLAG_GOOD, FLAG_ESTIMATED):
        return FLAG_SUSPECT
    return SSL_flag


def apply_hydro_qc_with_provenance(
    time,
    Q,
    SSC,
    SSL,
    *,
    Q_is_independent=True,
    SSC_is_independent=True,
    SSL_is_independent=False,
    ssl_is_derived_from_q_ssc=True,
    qc2_k=1.5,
    qc2_min_samples=5,
    qc3_k=1.5,
    qc3_min_samples=5,
):
    time = np.asarray(time)
    n = len(time)

    Qv = _as_float_array(Q)
    SSCv = _as_float_array(SSC)
    SSLv = _as_float_array(SSL)

    Q_flag_qc1_physical = apply_quality_flag_array(Qv, "Q")
    SSC_flag_qc1_physical = apply_quality_flag_array(SSCv, "SSC")
    SSL_flag_qc1_physical = apply_quality_flag_array(SSLv, "SSL")

    Q_flag = Q_flag_qc1_physical.copy()
    SSC_flag = SSC_flag_qc1_physical.copy()
    SSL_flag = SSL_flag_qc1_physical.copy()

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

    SSC_flag_qc3_ssc_q = np.full(n, FLAG_NOT_CHECKED, dtype=np.int8)
    SSL_flag_qc3_from_ssc_q = np.full(n, FLAG_NOT_CHECKED, dtype=np.int8)
    SSC_flag_qc3_ssc_q[SSC_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT
    SSL_flag_qc3_from_ssc_q[SSL_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT

    env_mask = (
        (Q_flag == FLAG_GOOD)
        & (SSC_flag == FLAG_GOOD)
        & np.isfinite(Qv)
        & np.isfinite(SSCv)
        & (Qv > 0)
        & (SSCv > 0)
    )
    ssc_q_bounds = build_ssc_q_envelope(
        np.where(env_mask, Qv, np.nan),
        np.where(env_mask, SSCv, np.nan),
        k=qc3_k,
        min_samples=qc3_min_samples,
    )

    if ssc_q_bounds is not None:
        SSC_flag_qc3_ssc_q[env_mask] = FLAG_GOOD
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

            SSC_flag_qc3_ssc_q[i] = FLAG_SUSPECT
            SSC_flag[i] = FLAG_SUSPECT

            prev_ssl_flag = SSL_flag[i]
            SSL_flag[i] = propagate_ssc_q_inconsistency_to_ssl(
                inconsistent=True,
                Q=Qv[i],
                SSC=SSCv[i],
                SSL=SSLv[i],
                Q_flag=Q_flag[i],
                SSC_flag=FLAG_GOOD,
                SSL_flag=prev_ssl_flag,
                ssl_is_derived_from_q_ssc=ssl_is_derived_from_q_ssc,
            )
            SSL_flag_qc3_from_ssc_q[i] = (
                FLAG_SUSPECT
                if (prev_ssl_flag in (FLAG_GOOD, FLAG_ESTIMATED) and SSL_flag[i] == FLAG_SUSPECT)
                else FLAG_GOOD
            )

    valid_time = (
        (Q_flag != FILL_VALUE_INT)
        | (SSC_flag != FILL_VALUE_INT)
        | (SSL_flag != FILL_VALUE_INT)
    )
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
        "Q_flag_qc1_physical": Q_flag_qc1_physical[valid_time],
        "SSC_flag_qc1_physical": SSC_flag_qc1_physical[valid_time],
        "SSL_flag_qc1_physical": SSL_flag_qc1_physical[valid_time],
        "Q_flag_qc2_log_iqr": Q_flag_qc2_log_iqr[valid_time],
        "SSC_flag_qc2_log_iqr": SSC_flag_qc2_log_iqr[valid_time],
        "SSL_flag_qc2_log_iqr": SSL_flag_qc2_log_iqr[valid_time],
        "SSC_flag_qc3_ssc_q": SSC_flag_qc3_ssc_q[valid_time],
        "SSL_flag_qc3_from_ssc_q": SSL_flag_qc3_from_ssc_q[valid_time],
        "ssc_q_bounds": ssc_q_bounds,
    }
