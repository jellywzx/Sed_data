#!/usr/bin/env python3
"""
第四步：将 overlap_for_manual_choice.csv 中同一 (cluster_id, resolution, date) 的多行合并为一行，
输出 overlap_resolved.csv，供 s5 使用。

合并策略（--strategy）：
  first            每个键保留第一次出现的行（默认）
  source_priority  按优先级保留：内置 BUILDIN_SOURCE_PRIORITY 或 --sources，未传 --sources 时用内置
  mean             对 Q,SSC,SSL 取均值
  median           对 Q,SSC,SSL 取中位数

内置 source 优先级（BUILDIN_SOURCE_PRIORITY）：修改下面列表即可，从左到右优先级从高到低。
用法：
  python s4_merge_overlap_by_cluster.py
  python s4_merge_overlap_by_cluster.py --strategy source_priority
  python s4_merge_overlap_by_cluster.py --strategy source_priority --sources USGS,HYDAT,RiverSed
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"
DEFAULT_INPUT = OUTPUT_R_ROOT / "output/03_merge" / "overlap_for_manual_choice.csv"
DEFAULT_OUTPUT = OUTPUT_R_ROOT / "output/03_merge" / "overlap_resolved.csv"

# 内置 source 优先级：从左到右优先级从高到低；空列表表示不启用优先级（等同 first）
BUILDIN_SOURCE_PRIORITY = [
    "USGS",
    "HYDAT",
    "RiverSed",
    "EUSEDcollab",
    "GFQA_v2",
    "Robotham",
    "GSED",
    "GloRiSe_SS",
    "GloRiSe_BS",
    "HYBAM",
    "Yajiang",
    "Rhine",
    "Dethier",
    "Vanmaercke",
    "Milliman",
    "GloRiSe",
]


def main():
    ap = argparse.ArgumentParser(description="Step 4: merge overlap_for_manual_choice by (cluster_id, resolution, date)")
    ap.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="输入 overlap_for_manual_choice.csv")
    ap.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT), help="输出 overlap_resolved.csv")
    ap.add_argument("--strategy", choices=["first", "source_priority", "mean", "median"], default="source_priority",
                    help="合并策略；默认 source_priority 使用内置 BUILDIN_SOURCE_PRIORITY")
    ap.add_argument("--sources", type=str, default="",
                    help="source_priority 时的优先级列表（逗号分隔）；空则使用脚本内 BUILDIN_SOURCE_PRIORITY")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="mean/median 时分块行数")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.is_file():
        print("Error: not found: {}".format(inp))
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.sources.strip():
        source_priority = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        source_priority = list(BUILDIN_SOURCE_PRIORITY)
    if args.strategy == "source_priority" and source_priority:
        print("Source priority ({}): {}".format(len(source_priority), ", ".join(source_priority[:5]) + (" ..." if len(source_priority) > 5 else "")))
    strategy = args.strategy

    if strategy in ("first", "source_priority"):
        selected = {}
        header = None
        n_read = 0
        use_resolution = False
        with open(inp, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames)
            if "source" not in header:
                print("Error: CSV must have column 'source'")
                return 1
            use_resolution = "resolution" in header
            for row in reader:
                n_read += 1
                if n_read % 1_000_000 == 0:
                    print("  read {} rows, {} keys ...".format(n_read, len(selected)))
                try:
                    cid = int(row.get("cluster_id", 0))
                    date_val = row.get("date", "")
                    res_val = str(row.get("resolution", "")).strip() if use_resolution else ""
                except (ValueError, TypeError):
                    continue
                key = (cid, res_val, date_val) if use_resolution else (cid, date_val)
                row["cluster_id"] = cid
                if key not in selected:
                    selected[key] = row
                    continue
                if strategy == "first":
                    continue
                cur = selected[key]
                cur_src = cur.get("source", "")
                new_src = row.get("source", "")
                cur_rank = len(source_priority) + 1
                new_rank = len(source_priority) + 1
                if cur_src in source_priority:
                    cur_rank = source_priority.index(cur_src)
                if new_src in source_priority:
                    new_rank = source_priority.index(new_src)
                if new_rank < cur_rank:
                    selected[key] = row

        print("Resolved {} keys from {} rows.".format(len(selected), n_read))
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for key in sorted(selected.keys()):
                w.writerow(selected[key])
        print("Wrote {}.".format(out))
        return 0

    chunks = pd.read_csv(inp, chunksize=args.chunk, dtype={"cluster_id": int}, on_bad_lines="skip")
    agg_dict = {}
    use_resolution = False
    for i, df in enumerate(chunks):
        if len(df) == 0:
            continue
        use_resolution = "resolution" in df.columns
        for col in ["cluster_id", "date", "Q", "SSC", "SSL"]:
            if col not in df.columns:
                print("Error: CSV must have columns cluster_id, date, Q, SSC, SSL")
                return 1
        df["date"] = df["date"].astype(str)
        if use_resolution:
            df["resolution"] = df["resolution"].astype(str)
        gb = df.groupby(["cluster_id", "resolution", "date"] if use_resolution else ["cluster_id", "date"], dropna=False)
        for key, grp in gb:
            if use_resolution:
                cid, res, d = key
                key = (int(cid), str(res), str(d))
            else:
                cid, d = key
                key = (int(cid), "", str(d))
            if key not in agg_dict:
                agg_dict[key] = {"q": [], "ssc": [], "ssl": [], "first": grp.iloc[0].to_dict()}
            for _, r in grp.iterrows():
                agg_dict[key]["q"].append(r.get("Q", np.nan))
                agg_dict[key]["ssc"].append(r.get("SSC", np.nan))
                agg_dict[key]["ssl"].append(r.get("SSL", np.nan))
        print("  chunk {}: {} rows, {} keys.".format(i + 1, len(df), len(agg_dict)))

    func = np.nanmean if strategy == "mean" else np.nanmedian
    rows = []
    for (cid, res, date_val), v in agg_dict.items():
        first = v["first"]
        row = {
            "cluster_id": cid,
            "lat": first.get("lat"),
            "lon": first.get("lon"),
            "date": date_val,
            "resolution": res if res else first.get("resolution", ""),
            "source": "merged_{}".format(strategy),
            "Q": func(v["q"]) if v["q"] else np.nan,
            "SSC": func(v["ssc"]) if v["ssc"] else np.nan,
            "SSL": func(v["ssl"]) if v["ssl"] else np.nan,
            "path": first.get("path", ""),
        }
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["cluster_id", "resolution", "date"] if use_resolution else ["cluster_id", "date"]).reset_index(drop=True)
    out_df.to_csv(out, index=False)
    print("Resolved {} keys. Wrote {}.".format(len(out_df), out))
    return 0


if __name__ == "__main__":
    exit(main())
