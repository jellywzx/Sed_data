#!/usr/bin/env python3
"""
Orchestrator: run_pipeline.py → fix_qc_global_attrs.py

先运行指定数据集的标准化处理流程（run_pipeline.py），
再对生成的 qc NetCDF 文件进行全局属性修正（fix_qc_global_attrs.py）。

用法示例
---------
  # 处理指定数据集（与 run_pipeline.py 语法一致）
  python run_pipeline_and_fix_attrs.py USGS
  python run_pipeline_and_fix_attrs.py USGS GSED HYBAM

  # 处理所有数据集
  python run_pipeline_and_fix_attrs.py --all

  # 预览，不实际执行
  python run_pipeline_and_fix_attrs.py USGS --dry-run

  # 自定义输出目录
  python run_pipeline_and_fix_attrs.py USGS --output-root /path/to/custom_output

  # 传递给 fix_qc_global_attrs.py 的额外参数：
  #   --workers N        并行进程数（默认 32）
  #   --limit N          最多处理的文件数
  #   --report-dir PATH  报告输出目录
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipeline then fix qc global attributes for selected datasets."
    )

    # Dataset selection (same as run_pipeline.py)
    parser.add_argument("datasets", nargs="*", help="Dataset name(s) to process")
    parser.add_argument("--all", action="store_true", help="Run all datasets")

    # Common passthrough options
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Override Output_r root (passed to both scripts)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")

    # run_pipeline.py specific
    parser.add_argument("--include-optional", action="store_true",
                        help="Include optional stages in run_pipeline.py")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable for run_pipeline stage scripts")

    # fix_qc_global_attrs.py specific
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel worker count for fix_qc_global_attrs.py")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max files to process in fix_qc_global_attrs.py")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Report directory for fix_qc_global_attrs.py")

    return parser.parse_args()


def run_step(cmd, label, dry_run):
    """Run a subprocess command with labeled output."""
    print(f"\n{'='*60}")
    print(f"  [{label}]")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry-run, skipped)")
        return True

    result = subprocess.run(cmd, cwd=str(SCRIPT_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: [{label}] failed with exit code {result.returncode}")
        return False
    print(f"  [{label}] completed successfully")
    return True


def build_run_pipeline_cmd(args):
    """Build command list for run_pipeline.py."""
    cmd = [args.python, "run_pipeline.py"]

    # Dataset selection
    if args.all:
        cmd.append("--all")
    else:
        cmd.extend(args.datasets)

    # Optional passthrough flags
    if args.include_optional:
        cmd.append("--include-optional")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.output_root:
        cmd.extend(["--output-root", str(args.output_root)])

    return cmd


def build_fix_attrs_cmd(args):
    """Build command list for fix_qc_global_attrs.py."""
    cmd = [sys.executable, "fix_qc_global_attrs.py"]

    # Dataset selection
    if args.all:
        cmd.append("--all")
    else:
        for ds in args.datasets:
            cmd.extend(["--dataset", ds])

    # Passthrough flags
    if args.dry_run:
        cmd.append("--dry-run")
    if args.output_root:
        cmd.extend(["--source-root", str(args.output_root)])
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.report_dir:
        cmd.extend(["--report-dir", str(args.report_dir)])

    return cmd


def main():
    args = parse_args()

    # Validate: need --all or dataset names
    if not args.all and not args.datasets:
        raise SystemExit(
            "Specify dataset names, or use --all / --list.\n"
            "  e.g.  python run_pipeline_and_fix_attrs.py USGS\n"
            "        python run_pipeline_and_fix_attrs.py --all"
        )

    # Step 1: run_pipeline.py
    pipeline_cmd = build_run_pipeline_cmd(args)
    ok = run_step(pipeline_cmd, "run_pipeline.py", args.dry_run)
    if not ok:
        print("\nPipeline step failed. Skipping attribute fix step.")
        sys.exit(1)

    # Step 2: fix_qc_global_attrs.py
    fix_cmd = build_fix_attrs_cmd(args)
    ok = run_step(fix_cmd, "fix_qc_global_attrs.py", args.dry_run)
    if not ok:
        print("\nAttribute fix step failed. Check the report for details.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  All steps completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
