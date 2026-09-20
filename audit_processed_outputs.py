#!/usr/bin/env python3
"""Generate post-processing record eligibility/QC summaries from Output_r."""

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from audit import write_dataset_output_audit
from runtime import resolve_output_root


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit final QC NetCDF outputs by dataset. Counts sediment-eligible "
            "records (SSC or SSL present), Q-only/fully-missing records screened "
            "from downstream sediment integration, and final QC-flag composition."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output_r root. Defaults to the repository/runtime Output_r path.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="Directory for audit CSVs. Defaults to <output-root>/audit.",
    )
    parser.add_argument(
        "--include-non-qc",
        action="store_true",
        help="Also scan NetCDF files outside final qc directories (normally not recommended).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else resolve_output_root(__file__)
    )
    paths = write_dataset_output_audit(
        output_root,
        audit_dir=args.audit_dir,
        final_qc_only=not args.include_non_qc,
    )

    print("Processed-output audit complete:")
    print(f"  dataset summary      : {paths['dataset']}")
    print(f"  by-resolution summary: {paths['by_resolution']}")
    print()
    print("Interpretation:")
    print("  retained = sediment_eligible_records (SSC or SSL is non-missing)")
    print("  screened = q_only_records + fully_missing_records")
    print("  QC flags 2/3 remain in the NetCDF and are NOT counted as screened records")
    print("  raw-source parsing/aggregation losses are not inferred by this audit")


if __name__ == "__main__":
    main()
