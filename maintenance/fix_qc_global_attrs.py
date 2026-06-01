#!/usr/bin/env python3
"""
批量修正 qc NetCDF 文件的全局属性，使其符合标准化规则。

该脚本遍历指定目录下所有 qc 子目录中的 .nc 文件，调用 normalize_nc_attrs()
对每个文件的全局属性进行原地标准化（如补充缺失属性、修正属性值等），
并生成 CSV 和 TXT 格式的处理报告。

运行环境要求
-------------
- Python >= 3.8
- 依赖包: tqdm, netCDF4 (被 normalize_nc_attrs 间接使用)
- 同级目录下需存在 code/ 模块包（dataset_attr_profiles, global_attrs, runtime）
- 运行前请确保当前 Python 环境已安装所需依赖：
    pip install tqdm netCDF4

用法示例
---------
# 处理所有数据集（默认读取脚本所在目录的 Output_r 上级目录）
python fix_qc_global_attrs.py --all

# 仅处理指定数据集（可重复使用 --dataset）
python fix_qc_global_attrs.py --dataset SED

# 指定 source root 和数据集的组合
python fix_qc_global_attrs.py --source-root /path/to/Output_r --dataset SED

# 仅预览变更，不实际修改文件
python fix_qc_global_attrs.py --dataset SED --dry-run

# 限制处理文件数 + 控制并行度
python fix_qc_global_attrs.py --dataset SED --limit 50 --workers 8

# 指定报告输出目录
python fix_qc_global_attrs.py --dataset SED --report-dir /path/to/reports

参数说明
---------
--source-root     Output_r 根目录路径（默认由 resolve_output_root 自动推导）
--dataset         要处理的数据集名称（可重复）；需配合 --all 或至少一个 --dataset 使用
--all             处理 --source-root 下所有数据集
--workers         并行进程数，默认 32
--limit           最多处理的文件数（过滤后），0 表示不限制
--dry-run         仅生成报告，不修改 .nc 文件
--report-dir      报告输出目录（默认 Output_r/scripts_basin_test/output）

输出说明 & 运行阶段
--------------------
脚本运行时会依次打印以下信息，对应三个处理阶段：

1. 启动信息（脚本刚运行时打印）
   - Source root : 待处理的 Output_r 根目录
   - Files       : 匹配到的 .nc 文件总数
   - Mode        : dry-run（预览不动文件）或 apply（实际修改）
   - Phase 1     : 开始执行全局属性标准化

2. 处理进度条（Phase 1 期间动态显示）
   - Processing  : 并行处理各 .nc 文件的实时进度条

3. 报告输出（处理完成后打印）
   - Reports written : 生成的 CSV 明细报告和 TXT 汇总报告路径

输出文件说明
-------------
- CSV 报告：每条记录一行，包含数据集、状态、是否变更、变更的属性键、修复后仍缺失的属性等
- TXT 汇总：统计各状态数量、高频变更属性、高频缺失属性、按数据集统计的状态分布
- dry-run 模式下 CSV 的 status 列为 "planned"，否则为 "normalized" 或 "normalize_error"
"""

import argparse

import argparse
import csv
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from code.dataset_attr_profiles import get_dataset_profile
from code.global_attrs import HISTORY_NOTE, normalize_nc_attrs
from code.runtime import resolve_output_root
from tqdm import tqdm


DEFAULT_WORKERS = 32
REPORT_REL_DIR = Path("scripts_basin_test") / "output"
REPORT_CSV_NAME = "fix_qc_global_attrs_report.csv"
REPORT_TXT_NAME = "fix_qc_global_attrs_summary.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize canonical global attrs in-place for qc NetCDF files.")
    parser.add_argument("--source-root", type=Path, default=resolve_output_root(__file__), help="Source Output_r root")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name to process; can be repeated")
    parser.add_argument("--all", action="store_true", help="Process all datasets under source root")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, metavar="N", help="Parallel worker count")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="Only process the first N files after filtering")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify NC files; only generate reports")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory for CSV/TXT reports; default is Output_r/scripts_basin_test/output")
    return parser.parse_args()


def _safe_relative_to(path_obj, root_obj):
    try:
        return path_obj.resolve().relative_to(root_obj.resolve())
    except Exception:
        return None


def _iter_qc_nc_files(source_root):
    for path_obj in sorted(source_root.rglob("*.nc")):
        rel = _safe_relative_to(path_obj, source_root)
        if rel is None:
            continue
        parts = rel.parts
        if len(parts) < 4:
            continue
        if parts[2] != "qc":
            continue
        yield {
            "path": path_obj,
            "rel_path": rel,
            "path_resolution": parts[0],
            "dataset": parts[1],
        }


def _filter_targets(records, datasets, process_all, limit):
    if not process_all and not datasets:
        raise SystemExit("Use --all or provide at least one --dataset.")

    dataset_filter = set(item.strip().lower() for item in datasets if str(item).strip())
    if dataset_filter:
        records = [row for row in records if row["dataset"].lower() in dataset_filter]

    if limit and limit > 0:
        records = records[:limit]
    return records


def _normalize_one(item):
    nc_path, dataset_name, path_resolution, dry_run = item
    try:
        profile = get_dataset_profile(dataset_name)
        result = normalize_nc_attrs(
            str(nc_path),
            dataset_name=dataset_name,
            path_resolution=path_resolution,
            history_note=HISTORY_NOTE,
            dry_run=dry_run,
        )
        return {
            "status": "planned" if dry_run else "normalized",
            "path": str(nc_path),
            "dataset": dataset_name,
            "path_resolution": path_resolution,
            "changed": bool(result.get("changed")),
            "changed_keys": result.get("changed_keys", []),
            "removed_keys": result.get("removed_keys", []),
            "missing_after_fix": result.get("missing_after_fix", []),
            "profile_data_source_name": profile.get("data_source_name", ""),
            "new_dataset_name": result.get("new_dataset_name", ""),
            "new_data_source_name": result.get("new_data_source_name", ""),
            "old_dataset_name": result.get("old_dataset_name", ""),
            "old_data_source_name": result.get("old_data_source_name", ""),
            "old_source": result.get("old_source", ""),
            "new_source": result.get("new_source", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "normalize_error",
            "path": str(nc_path),
            "dataset": dataset_name,
            "path_resolution": path_resolution,
            "changed": False,
            "changed_keys": [],
            "removed_keys": [],
            "missing_after_fix": [],
            "profile_data_source_name": "",
            "new_dataset_name": "",
            "new_data_source_name": "",
            "old_dataset_name": "",
            "old_data_source_name": "",
            "old_source": "",
            "new_source": "",
            "error": str(exc),
        }


def _write_report_csv(report_path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "path_resolution",
        "status",
        "path",
        "changed",
        "changed_keys",
        "removed_keys",
        "missing_after_fix",
        "profile_data_source_name",
        "old_dataset_name",
        "new_dataset_name",
        "old_data_source_name",
        "new_data_source_name",
        "old_source",
        "new_source",
        "error",
    ]
    with open(str(report_path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row.get("dataset", ""),
                    "path_resolution": row.get("path_resolution", ""),
                    "status": row.get("status", ""),
                    "path": row.get("path", ""),
                    "changed": int(bool(row.get("changed"))),
                    "changed_keys": "|".join(row.get("changed_keys", [])),
                    "removed_keys": "|".join(row.get("removed_keys", [])),
                    "missing_after_fix": "|".join(row.get("missing_after_fix", [])),
                    "profile_data_source_name": row.get("profile_data_source_name", ""),
                    "old_dataset_name": row.get("old_dataset_name", ""),
                    "new_dataset_name": row.get("new_dataset_name", ""),
                    "old_data_source_name": row.get("old_data_source_name", ""),
                    "new_data_source_name": row.get("new_data_source_name", ""),
                    "old_source": row.get("old_source", ""),
                    "new_source": row.get("new_source", ""),
                    "error": row.get("error", ""),
                }
            )


def _write_summary_txt(summary_path, rows, dry_run, source_root):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    status_counter = Counter()
    changed_key_counter = Counter()
    removed_key_counter = Counter()
    missing_key_counter = Counter()
    dataset_counter = defaultdict(Counter)
    dataset_name_values = defaultdict(set)
    data_source_name_values = defaultdict(set)

    for row in rows:
        status_counter[row.get("status", "")] += 1
        dataset_counter[row.get("dataset", "")][row.get("status", "")] += 1
        for key in row.get("changed_keys", []):
            changed_key_counter[key] += 1
        for key in row.get("removed_keys", []):
            removed_key_counter[key] += 1
        for key in row.get("missing_after_fix", []):
            missing_key_counter[key] += 1
        if row.get("new_dataset_name", ""):
            dataset_name_values[row.get("dataset", "")].add(row.get("new_dataset_name", ""))
        if row.get("new_data_source_name", ""):
            data_source_name_values[row.get("dataset", "")].add(row.get("new_data_source_name", ""))

    with open(str(summary_path), "w", encoding="utf-8") as handle:
        handle.write("fix_qc_global_attrs summary\n")
        handle.write("mode            : {0}\n".format("dry-run" if dry_run else "apply"))
        handle.write("source_root     : {0}\n".format(source_root))
        handle.write("total_rows      : {0}\n".format(len(rows)))
        handle.write("\nstatus counts\n")
        for status, count in sorted(status_counter.items()):
            handle.write("  {0:<20s} {1}\n".format(status, count))

        handle.write("\nchanged key counts\n")
        for key, count in changed_key_counter.most_common():
            handle.write("  {0:<30s} {1}\n".format(key, count))

        handle.write("\nremoved key counts\n")
        for key, count in removed_key_counter.most_common():
            handle.write("  {0:<30s} {1}\n".format(key, count))

        handle.write("\nmissing-after-fix key counts\n")
        for key, count in missing_key_counter.most_common():
            handle.write("  {0:<30s} {1}\n".format(key, count))

        handle.write("\nper-dataset canonical names\n")
        for dataset_name in sorted(dataset_name_values):
            handle.write(
                "  [{0}] dataset_name={1} data_source_name={2}\n".format(
                    dataset_name,
                    " | ".join(sorted(dataset_name_values[dataset_name])),
                    " | ".join(sorted(data_source_name_values.get(dataset_name, []))),
                )
            )

        handle.write("\nper-dataset status counts\n")
        for dataset_name in sorted(dataset_counter):
            handle.write("  [{0}]\n".format(dataset_name))
            for status, count in sorted(dataset_counter[dataset_name].items()):
                handle.write("    {0:<18s} {1}\n".format(status, count))


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    workers = max(1, int(args.workers))

    if not source_root.is_dir():
        raise SystemExit("Source root does not exist: {0}".format(source_root))

    records = list(_iter_qc_nc_files(source_root))
    records = _filter_targets(records, args.dataset, args.all, args.limit)
    if not records:
        raise SystemExit("No qc NetCDF files matched the current selection.")

    print("Source root : {0}".format(source_root))
    print("Files       : {0}".format(len(records)))
    print("Mode        : {0}".format("dry-run" if args.dry_run else "apply"))
    print("\nPhase 1: in-place canonical global-attr normalization")

    normalize_tasks = [
        (row["path"], row["dataset"], row["path_resolution"], args.dry_run)
        for row in records
    ]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_normalize_one, item) for item in normalize_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    results = sorted(results, key=lambda row: (row.get("dataset", ""), row.get("path", "")))

    report_dir = args.report_dir.expanduser().resolve() if args.report_dir else source_root / REPORT_REL_DIR
    report_csv = report_dir / REPORT_CSV_NAME
    report_txt = report_dir / REPORT_TXT_NAME
    _write_report_csv(report_csv, results)
    _write_summary_txt(report_txt, results, args.dry_run, source_root)

    print("\nReports written:")
    print("  {0}".format(report_csv))
    print("  {0}".format(report_txt))


if __name__ == "__main__":
    main()
