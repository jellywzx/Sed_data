import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. 手动整理你刚才跑出来的数据 ===
#（也可以直接从 station_data 或 summary CSV 读）

data = [
    ("Kharmong", 488.10, 1551.61, 0),
    ("Yugo", 346.75, 2842.06, 0),
    ("Shigar", 207.15, 2569.87, 0),
    ("Kachura", 1056.76, 2401.88, 0),
    ("Dainyor Bridge", 335.20, 4197.30, 0),
    ("Gilgit", 282.85, 672.19, 0),
    ("Alam Bridge", 646.56, 2685.75, 0),
    ("Partab Bridge", 1769.61, 2476.52, 0),
    ("Doyian", 129.56, 415.80, 2),   # suspect
    ("Shatial Bridge", 2013.56, 1866.45, 0),
    ("Barsin", 1777.88, 2504.21, 0),
    ("Karora", 20.83, 304.31, 2),    # suspect
    ("Besham Qila", 2413.44, 2552.44, 0),
    ("Daggar", 5.67, 1677.83, 0),
    ("Darband", 2448.27, 3722.42, 0),
    ("Phulra", 21.10, 3604.09, 0),
    ("Thapla", 30.95, 2968.72, 0),
]

df = pd.DataFrame(data, columns=["station", "Q", "SSC", "SSC_flag"])

# === 2. 拟合 SSC–Q 关系（log–log） ===
mask = (df["Q"] > 0) & (df["SSC"] > 0)
logQ = np.log10(df.loc[mask, "Q"])
logSSC = np.log10(df.loc[mask, "SSC"])

coef = np.polyfit(logQ, logSSC, 1)
logSSC_pred = np.polyval(coef, logQ)
resid = logSSC - logSSC_pred

q1, q3 = np.percentile(resid, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# 画 envelope
Q_line = np.logspace(np.log10(df["Q"].min()), np.log10(df["Q"].max()), 100)
logSSC_line = coef[0] * np.log10(Q_line) + coef[1]

SSC_lower = 10 ** (logSSC_line + lower)
SSC_upper = 10 ** (logSSC_line + upper)

# === 3. 开始画图 ===
plt.figure(figsize=(6, 5))

# 所有站点
plt.scatter(
    df["Q"], df["SSC"],
    c="lightgray", s=50, label="All stations"
)

# suspect 站点
sus = df["SSC_flag"] == 2
plt.scatter(
    df.loc[sus, "Q"], df.loc[sus, "SSC"],
    c="red", s=60, label="SSC–Q suspect"
)

# 拟合线
plt.plot(Q_line, 10 ** logSSC_line, "k-", label="SSC–Q trend")

# envelope
plt.plot(Q_line, SSC_lower, "k--", linewidth=1)
plt.plot(Q_line, SSC_upper, "k--", linewidth=1)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Discharge Q (m³ s⁻¹)")
plt.ylabel("Suspended sediment concentration (mg L⁻¹)")
plt.title("SSC–Q hydrological consistency check (Ali & De Boer)")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('ssc_q_hydrological_consistency.png', dpi=300)
plt.close()
