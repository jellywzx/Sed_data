#!/usr/bin/env python3
"""Regression tests for Mekong Delta SSC provenance preservation."""

import sys
import numpy as np
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from code.constants import FLAG_GOOD, FLAG_ESTIMATED
from code.qc import apply_hydro_qc_with_provenance


def test_source_ssc_preserved_flag_good():
    """Source SSC~123 mg/L -> preserved, SSC_derived=False, flag=0."""
    np.random.seed(42)
    n = 30
    Q = np.abs(np.linspace(800, 1200, n) + np.random.normal(0, 20, n))
    # Tiny noise on source SSC so QC2 IQR doesn't flag all as outlier
    SSC_source = np.full(n, 123.0) * np.abs(1 + np.random.normal(0, 0.02, n))
    SSL = SSC_source * Q * 0.0864 * np.abs(1 + np.random.normal(0, 0.03, n))
    
    ssc_derived_mask = np.zeros(n, dtype=bool)
    
    # Verify formula-derived SSC (from SSL and Q) would differ from 123
    ssc_derived_from_formula = SSL / (Q * 0.0864)
    assert not np.allclose(ssc_derived_from_formula, 123.0, rtol=0.01),         f'Derived SSC ({ssc_derived_from_formula[0]:.1f}) != 123'
    
    qc = apply_hydro_qc_with_provenance(
        time=np.arange(n, dtype=float),
        Q=Q, SSC=SSC_source, SSL=SSL,
        Q_is_independent=True, SSC_is_independent=True, SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
        SSC_derived_mask=ssc_derived_mask,
    )
    assert qc is not None
    assert (qc['SSC_derived_mask'] == 0).all()
    assert (qc['SSC_flag'] == FLAG_GOOD).all(),         f'Source SSC should be flag=0, got {set(qc["SSC_flag"])}'
    np.testing.assert_allclose(qc['SSC'], SSC_source, rtol=1e-6)
    print('PASS test_source_ssc_preserved_flag_good')
    print(f'  SSC~123 mg/L, SSC_derived=False, SSC_flag=0')
    print(f'  Derived from formula would be ~{ssc_derived_from_formula[0]:.1f}, source SSC intact')


def test_source_ssc_missing_derived_flag_estimated():
    """Source SSC missing, Q/SSL valid -> derived, SSC_derived=True, flag=1."""
    np.random.seed(42)
    n = 30
    Q = np.abs(np.linspace(800, 1200, n) + np.random.normal(0, 20, n))
    SSL = np.abs(np.linspace(4000, 6000, n) + np.random.normal(0, 100, n))
    with np.errstate(divide='ignore', invalid='ignore'):
        SSC = SSL / (Q * 0.0864)
    ssc_derived_mask = np.ones(n, dtype=bool)
    
    qc = apply_hydro_qc_with_provenance(
        time=np.arange(n, dtype=float),
        Q=Q, SSC=SSC, SSL=SSL,
        Q_is_independent=True, SSC_is_independent=True, SSL_is_independent=True,
        ssl_is_derived_from_q_ssc=False,
        SSC_derived_mask=ssc_derived_mask,
    )
    assert qc is not None
    assert (qc['SSC_derived_mask'] == 1).all()
    assert (qc['SSC_flag'] == FLAG_ESTIMATED).all(),         f'Derived SSC should be flag=1, got {set(qc["SSC_flag"])}'
    expected = SSL / (Q * 0.0864)
    np.testing.assert_allclose(qc['SSC'], expected, rtol=1e-10)
    print('PASS test_source_ssc_missing_derived_flag_estimated')
    print(f'  SSC_derived=True, SSC_flag=1, value matches formula')


def main():
    for name, func in sorted(globals().items()):
        if name.startswith('test_') and callable(func):
            func()
            print()

if __name__ == '__main__':
    main()
