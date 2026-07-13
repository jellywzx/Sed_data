from pathlib import Path
import pandas as pd

qc_dir = Path("/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/daily/HYDAT/qc")
summary_csv = qc_dir / "HYDAT_qc_results_summary.csv"

df = pd.read_csv(summary_csv)

# 防止某些列不存在时报错
for col in [
    "Q_qc1_bad", "SSC_qc1_bad", "SSL_qc1_bad",
    "Q_qc1_missing", "SSC_qc1_missing", "SSL_qc1_missing",
    "Q_qc2_suspect", "SSC_qc2_suspect",
    "SSC_qc3_suspect", "SSL_qc3_propagated",
    "QC_n_days"
]:
    if col not in df.columns:
        df[col] = 0

# 定义三个阶段是否真的“筛选/标记”了数据
# QC1: 物理检查阶段，优先看 bad；如果你想把 missing 也作为 QC1 展示，可加入 missing
df["qc1_affected"] = (
    df["Q_qc1_bad"] + df["SSC_qc1_bad"] + df["SSL_qc1_bad"]
)

df["qc1_affected_with_missing"] = (
    df["Q_qc1_bad"] + df["SSC_qc1_bad"] + df["SSL_qc1_bad"]
    + df["Q_qc1_missing"] + df["SSC_qc1_missing"] + df["SSL_qc1_missing"]
)

# QC2: log-IQR 统计异常值
df["qc2_affected"] = (
    df["Q_qc2_suspect"] + df["SSC_qc2_suspect"]
)

# QC3: SSC-Q 水文一致性，以及传播到 SSL 的 suspect
df["qc3_affected"] = (
    df["SSC_qc3_suspect"] + df["SSL_qc3_propagated"]
)

# 严格筛选：三个 QC 阶段都有非缺失类筛选结果
strict = df[
    (df["qc1_affected"] > 0) &
    (df["qc2_affected"] > 0) &
    (df["qc3_affected"] > 0)
].copy()

# 如果 strict 为空，使用更适合示意图的宽松筛选：
# QC1 中允许用 missing/bad 共同展示“物理/缺失检查”
relaxed = df[
    (df["qc1_affected_with_missing"] > 0) &
    (df["qc2_affected"] > 0) &
    (df["qc3_affected"] > 0)
].copy()

candidates = strict if len(strict) > 0 else relaxed

# 排序：优先选择记录长、三个阶段都有较多标记点的站点
candidates["total_qc_affected"] = (
    candidates["qc1_affected_with_missing"]
    + candidates["qc2_affected"]
    + candidates["qc3_affected"]
)

candidates = candidates.sort_values(
    ["total_qc_affected", "qc3_affected", "qc2_affected", "QC_n_days"],
    ascending=False
)

cols = [
    "Source_ID", "station_name", "river_name",
    "longitude", "latitude", "upstream_area",
    "QC_n_days",
    "qc1_affected", "qc1_affected_with_missing",
    "qc2_affected", "qc3_affected",
    "Q_qc1_bad", "SSC_qc1_bad", "SSL_qc1_bad",
    "Q_qc2_suspect", "SSC_qc2_suspect",
    "SSC_qc3_suspect", "SSL_qc3_propagated",
]

print(candidates[cols].head(20))

out_csv = qc_dir / "HYDAT_qc_demo_station_candidates.csv"
candidates[cols].to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
