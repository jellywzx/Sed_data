#!/usr/bin/env python3
"""Regression tests for derived SSL flag propagation in shared QC."""

import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FLAG_ESTIMATED, FLAG_SUSPECT  # noqa: E402
from code.qc import apply_hydro_qc_with_provenance  # noqa: E402


def _run_case(Q, SSC):
    Q = np.asarray(Q, dtype=float)
    SSC = np.asarray(SSC, dtype=float)
    SSL = Q * SSC * 0.0864
    time = np.datetime64("2000-01-01") + np.arange(len(Q)).astype("timedelta64[D]")
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )
    assert qc is not None
    return qc


def test_q_suspect_propagates_to_derived_ssl():
    qc = _run_case(
        Q=[1, 2, 3, 4, 5, 100000],
        SSC=[10, 11, 10, 12, 11, 10],
    )
    assert qc["Q_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][0] == FLAG_ESTIMATED


def test_ssc_suspect_propagates_to_derived_ssl():
    qc = _run_case(
        Q=[1, 2, 3, 4, 5, 6],
        SSC=[10, 11, 10, 12, 11, 1000000],
    )
    assert qc["SSC_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][-1] == FLAG_SUSPECT
    assert qc["SSL_flag"][0] == FLAG_ESTIMATED


def main():
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print("PASS {}".format(name))


if __name__ == "__main__":
    main()
