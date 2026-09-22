#!/usr/bin/env python3
"""Regression checks for the SSL QC3 propagation flag contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_constants_module():
    spec = importlib.util.spec_from_file_location(
        "sed_data_constants", ROOT / "code" / "constants.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ssl_qc3_propagation_uses_suspect_flag_not_estimated_flag():
    constants = _load_constants_module()

    assert int(constants.FLAG_ESTIMATED) == 1
    assert int(constants.FLAG_SUSPECT) == 2
    assert int(constants.QC3_SSL_FROM_SSC_Q_PROPAGATED_FLAG) == int(constants.FLAG_SUSPECT)
    assert int(constants.QC3_SSL_FROM_SSC_Q_PROPAGATED_FLAG) != int(constants.FLAG_ESTIMATED)


def test_ssl_qc3_flag_values_and_meanings_are_canonical():
    constants = _load_constants_module()

    assert constants.QC3_SSL_FROM_SSC_Q_FLAG_VALUES.tolist() == [0, 2, 8, 9]
    assert (
        constants.QC3_SSL_FROM_SSC_Q_FLAG_MEANINGS
        == "not_propagated propagated not_checked missing"
    )


def test_workflow_doc_repeats_the_same_contract():
    workflow = (ROOT / "docs" / "workflow.md").read_text(encoding="utf-8")

    assert "`SSL_flag_qc3_from_ssc_q`" in workflow
    assert "0=not_propagated, 2=propagated, 8=not_checked, 9=missing" in workflow
    assert "不要使用 `1` 表示 propagated" in workflow
