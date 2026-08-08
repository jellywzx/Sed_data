#!/usr/bin/env python3
"""Regression tests for mixed source/derived provenance in shared QC."""

import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import (  # noqa: E402
    FLAG_BAD,
    FLAG_ESTIMATED,
    FLAG_GOOD,
    FLAG_MISSING,
    FLAG_NOT_CHECKED,
    FLAG_SUSPECT,
)
from code.qc import apply_hydro_qc_with_provenance  # noqa: E402


def _time(n):
    return np.datetime64("2000-01-01") + np.arange(n).astype("timedelta64[D]")


def _run(
    Q,
    SSC,
    SSL,
    *,
    SSC_derived_mask=None,
    SSL_derived_mask=None,
    SSC_is_independent=True,
    SSL_is_independent=True,
    ssl_is_derived_from_q_ssc=False,
    qc2_min_samples=5,
):
    qc = apply_hydro_qc_with_provenance(
        time=_time(len(Q)),
        Q=np.asarray(Q, dtype=float),
        SSC=np.asarray(SSC, dtype=float),
        SSL=np.asarray(SSL, dtype=float),
        Q_is_independent=True,
        SSC_is_independent=SSC_is_independent,
        SSL_is_independent=SSL_is_independent,
        SSC_derived_mask=SSC_derived_mask,
        SSL_derived_mask=SSL_derived_mask,
        ssl_is_derived_from_q_ssc=ssl_is_derived_from_q_ssc,
        qc2_k=1.5,
        qc2_min_samples=qc2_min_samples,
        qc3_k=1.5,
        qc3_min_samples=20,
    )
    assert qc is not None
    return qc


def test_source_ssl_record_stays_good_with_explicit_source_mask():
    qc = _run(
        Q=[1, 2, 3, 4],
        SSC=[10, 10, 10, 10],
        SSL=[1, 2, 3, 4],
        SSL_derived_mask=[False, False, False, False],
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
    )

    assert int(qc["SSL_flag"][0]) == int(FLAG_GOOD)
    assert bool(qc["SSL_derived_mask"][0]) is False


def test_derived_ssl_record_is_estimated_when_inputs_are_good():
    qc = _run(
        Q=[1, 2, 3, 4],
        SSC=[10, 10, 10, 10],
        SSL=[1, 2, 3, 4],
        SSL_derived_mask=[False, True, False, False],
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
    )

    assert int(qc["SSL_flag"][1]) == int(FLAG_ESTIMATED)
    assert bool(qc["SSL_derived_mask"][1]) is True


def test_mixed_ssl_provenance_flags_source_derived_and_missing_records():
    qc = _run(
        Q=[1, 2, 3, 4],
        SSC=[10, 10, 10, 10],
        SSL=[1, 2, 3, np.nan],
        SSL_derived_mask=[False, False, True, False],
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
    )

    np.testing.assert_array_equal(
        qc["SSL_flag"],
        np.array([FLAG_GOOD, FLAG_GOOD, FLAG_ESTIMATED, FLAG_MISSING], dtype=np.int8),
    )


def test_source_ssl_does_not_receive_suspect_q_flag_propagation():
    qc = _run(
        Q=[1, 2, 3, 4, 5, 100000],
        SSC=[10, 10, 10, 10, 10, 10],
        SSL=[100, 100, 100, 100, 100, 100],
        SSL_derived_mask=[False, False, False, False, False, False],
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
    )

    assert int(qc["Q_flag"][-1]) == int(FLAG_SUSPECT)
    assert int(qc["SSL_flag"][-1]) == int(FLAG_GOOD)


def test_derived_ssl_receives_suspect_q_flag_propagation():
    qc = _run(
        Q=[1, 2, 3, 4, 5, 100000],
        SSC=[10, 10, 10, 10, 10, 10],
        SSL=[100, 100, 100, 100, 100, 100],
        SSL_derived_mask=[False, False, False, False, False, True],
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
    )

    assert int(qc["Q_flag"][-1]) == int(FLAG_SUSPECT)
    assert int(qc["SSL_flag"][-1]) == int(FLAG_SUSPECT)


def test_derived_ssl_receives_bad_ssc_flag_propagation():
    qc = _run(
        Q=[1, 2, 3, 4],
        SSC=[10, 10, 10, -1],
        SSL=[1, 2, 3, 4],
        SSL_derived_mask=[False, False, False, True],
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
    )

    assert int(qc["SSC_flag"][-1]) == int(FLAG_BAD)
    assert int(qc["SSL_flag"][-1]) == int(FLAG_BAD)


def test_mixed_ssc_provenance_flags_source_and_derived_records():
    qc = _run(
        Q=[1, 2, 3, 4],
        SSC=[10, 11, 12, 13],
        SSL=[1, 2, 3, 4],
        SSC_derived_mask=[False, True, False, False],
        SSC_is_independent=False,
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
    )

    assert int(qc["SSC_flag"][0]) == int(FLAG_GOOD)
    assert int(qc["SSC_flag"][1]) == int(FLAG_ESTIMATED)
    assert bool(qc["SSC_derived_mask"][0]) is False
    assert bool(qc["SSC_derived_mask"][1]) is True


def test_qc2_skips_derived_records_in_mixed_provenance():
    qc = _run(
        Q=[1, 1, 1, 1, 1, 1],
        SSC=[10, 10, 10, 10, 10, 10],
        SSL=[10, 11, 12, 13, 14, 1e9],
        SSL_derived_mask=[False, False, False, False, False, True],
        SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
    )

    assert int(qc["SSL_flag_qc2_log_iqr"][-1]) == int(FLAG_NOT_CHECKED)
    assert int(qc["SSL_flag"][-1]) == int(FLAG_ESTIMATED)


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASS {}".format(name))


if __name__ == "__main__":
    main()
