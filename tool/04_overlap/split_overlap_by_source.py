#!/usr/bin/env python3
"""
将 overlap_for_manual_choice.csv 按 source（数据集）列拆分为多个 CSV：
每个 source 一个文件：overlap_for_manual_choice_{source}.csv，放在同目录下。

流式读取，适用于大文件。
用法：
  python split_overlap_by_source.py [--input overlap_for_manual_choice.csv] [--out-dir 同目录]
"""

import argparse
import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"
DEFAULT_INPUT = OUTPUT_R_ROOT / "output/03_merge" / "overlap_for_manual_choice.csv"


def safe_filename(s):
    """将 source 转为安全文件名：去掉或替换非法字符。"""
    s = re.sub(r'[\\/:*?"<>|]', "_", str(s))
    return s.strip() or "unknown"


def main():
    ap = argparse.ArgumentParser(description="Split overlap_for_manual_choice.csv by source")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="输入 CSV 路径")
    ap.add_argument("--out-dir", "-o", default=None, help="输出目录，默认与输入文件同目录")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print("Error: file not found: {}".format(inp))
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    header = None
    source_index = None  # source 列下标
    writers = {}  # source -> csv.writer
    files = {}   # source -> open file handle

    try:
        with open(inp, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    if "source" in header:
                        source_index = header.index("source")
                    else:
                        source_index = 4  # 默认第 5 列
                    continue
                if not row or len(row) <= source_index:
                    continue
                source = row[source_index].strip()
                if not source:
                    source = "unknown"
                name = safe_filename(source)
                if name not in writers:
                    path = out_dir / "overlap_for_manual_choice_{}.csv".format(name)
                    fp = open(path, "w", encoding="utf-8", newline="")
                    files[name] = fp
                    w = csv.writer(fp)
                    w.writerow(header)
                    writers[name] = w
                writers[name].writerow(row)
        for fp in files.values():
            fp.close()
        print("Done. Wrote {} files to {}:".format(len(files), out_dir))
        for name in sorted(files.keys()):
            p = out_dir / "overlap_for_manual_choice_{}.csv".format(name)
            print("  {}".format(p.name))
        return 0
    except Exception as e:
        for fp in files.values():
            try:
                fp.close()
            except Exception:
                pass
        print("Error: {}".format(e))
        return 1


if __name__ == "__main__":
    exit(main())
