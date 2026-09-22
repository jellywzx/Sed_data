#!/usr/bin/env python3
"""Regression test for the canonical GloRiSe release pipeline."""

from run_pipeline import PIPELINES


def test_glorise_pipeline_is_ss_only():
    stages = [stage["script"] for stage in PIPELINES["GloRiSe"]["stages"]]

    assert stages == [
        "GloRiSe/1_generate_netcdf_SS.py",
        "GloRiSe/2_qc_and_standardize_glorise.py",
    ]
    assert all("_BS.py" not in script for script in stages)
