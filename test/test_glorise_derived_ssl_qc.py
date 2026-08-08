#!/usr/bin/env python3
"""Minimal regression test for GloRiSe derived SSL QC semantics."""

import importlib.util
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

plot_stub = types.ModuleType("code.plot")
plot_stub.plot_ssc_q_diagnostic = lambda *args, **kwargs: None
sys.modules["code.plot"] = plot_stub

MODULE_PATH = SCRIPT_ROOT / "GloRiSe" / "2_qc_and_standardize_glorise.py"
spec = importlib.util.spec_from_file_location("glorise_qc", MODULE_PATH)
glorise_qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glorise_qc)


def test_valid_derived_ssl_is_estimated():
    q = np.array([10.0], dtype=float)
    ssc = np.array([25.0], dtype=float)
    ssl = q * ssc * 0.0864

    with redirect_stdout(StringIO()):
        result = glorise_qc.apply_tool_qc(q, ssc, ssl)
    ssl_flag = result[2]

    assert float(ssl[0]) == 21.6
    assert int(ssl_flag[0]) == 1


def main():
    test_valid_derived_ssl_is_estimated()
    print("PASS test_valid_derived_ssl_is_estimated")


if __name__ == "__main__":
    main()
