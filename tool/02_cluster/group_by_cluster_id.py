#!/usr/bin/env python3
"""
按 cluster_id 整理 clustered_stations.csv：
每行一个 cluster_id，列出该 cluster 包含的 path、source、lat、lon 等摘要。

输入/输出路径默认相对 Output_r 数据根目录（Script/tool 同级的 Output_r）。
"""

import argparse
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_R_ROOT = SCRIPT_DIR.parent.parent.parent / "Output_r"


def main():
    p = argparse.ArgumentParser(description="按 cluster_id 整理表格，每行体现该 cluster 包含的数据")
    p.add_argument(
        "--input",
        default=str(OUTPUT_R_ROOT / "output/02_cluster/clustered_stations.csv"),
        help="clustered_stations.csv 路径",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_R_ROOT / "output/02_cluster/clusters_summary.csv"),
        help="输出 CSV 路径",
    )
    p.add_argument(
        "--sep",
        default=" | ",
        help="同一 cluster 内多条 path 之间的分隔符",
    )
    args = p.parse_args()

    def _resolve(p):
        p = Path(p)
        return p.resolve() if p.is_absolute() else (OUTPUT_R_ROOT / p).resolve()
    inp = _resolve(args.input)
    out = _resolve(args.output)

    if not inp.exists():
        raise SystemExit(f"输入文件不存在: {inp}")

    df = pd.read_csv(inp)
    if "cluster_id" not in df.columns:
        raise SystemExit("输入 CSV 需包含列 cluster_id")

    # 按 cluster_id 分组，每组内把 path/source/lat/lon 汇总成一行
    grouped = df.groupby("cluster_id", sort=True)

    rows = []
    for cid, grp in grouped:
        paths = grp["path"].astype(str).tolist()
        sources = grp["source"].astype(str).tolist()
        lats = grp["lat"].tolist()
        lons = grp["lon"].tolist()

        # 每行：cluster_id, station_count, paths, sources, lat_lon_pairs
        rows.append({
            "cluster_id": cid,
            "station_count": len(grp),
            "paths": args.sep.join(paths),
            "sources": args.sep.join(sources),
            "lat_lon_pairs": args.sep.join(f"({a:.4f},{b:.4f})" for a, b in zip(lats, lons)),
        })

    out_df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8")
    print(f"已写入: {out}  共 {len(out_df)} 个 cluster")


if __name__ == "__main__":
    main()
