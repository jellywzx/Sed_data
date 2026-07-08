# Script 数据集处理流程规范

本文档归纳 `Script/` 下所有数据源处理脚本需要遵循的**统一规范**，确保最终输出的 QC NetCDF 能被 `scripts_basin_test/` 正确接收并整合。

---

## 1. 总体目标

每个数据源的处理脚本应当完成以下任务：

1. **读取原始数据**：从 Source/ 下的对应目录读取原始数据（CSV、Excel、已有 NetCDF 等）
2. **变量标准化**：将 Q、SSC、SSL 转换为标准单位
3. **质量控制**：调用 `code/qc.apply_hydro_qc_with_provenance` 执行统一 QC
4. **NetCDF 输出**：写入 CF-1.8 / ACDD-1.3 兼容的标准化 NetCDF 文件
5. **摘要输出**：生成站点汇总 CSV

---

## 2. 变量与单位标准

所有数据集必须输出以下三个核心变量，**单位必须统一**：

| 变量 | 全称 | 标准单位 | 常见来源单位 |
|------|------|---------|-------------|
| `Q` | Discharge (river flow) | **m3 s-1** | m³/s, m³/day, m³/month, cfs |
| `SSC` | Suspended Sediment Concentration | **mg L-1** | kg/m³ (×1000 → mg/L), g/m³ (×1 → mg/L) |
| `SSL` | Suspended Sediment Load | **ton day-1** | kg/day (÷1000 → ton/day), ton/day |

### SSL 派生公式

当 SSL 不是直接观测值而是从 Q 和 SSC 计算得到时，使用统一公式：

```
SSL (ton/day) = Q (m3/s) × SSC (mg/L) × 0.0864
```

> 因子 `0.0864 = 86400 s/day × 1000 L/m³ ÷ 1e9 mg/ton`

### SSL 计算必须保护 NaN/非正值

所有数据集在计算 SSL 时必须避免在无效数据上产生结果：

```python
valid = np.isfinite(Q) & np.isfinite(SSC) & (Q >= 0) & (SSC >= 0)
SSL = np.full(len(Q), np.nan, dtype=float)
SSL[valid] = Q[valid] * SSC[valid] * 0.0864
```

已经统一应用此修正的数据集：EUSEDcollab, USGS, GFQA_v2, HYBAM。

---

## 3. 质量控制（QC）规范

### 3.1 推荐调用路径

所有数据集应尽量复用 `code/qc.py` 中的共享 QC 函数：

```python
from code.qc import apply_hydro_qc_with_provenance

qc_result = apply_hydro_qc_with_provenance(
    time=time_array,
    Q=Q_array, SSC=SSC_array, SSL=SSL_array,
    station_id=station_id,
    station_name=station_name,
    iqr_k=1.5,
    min_samples_envelope=5,
)
```

### 3.2 QC 输出变量

`apply_hydro_qc_with_provenance` 返回以下逐步 QC 标志数组：

| 阶段 | 变量 | 含义 | 标志值 |
|------|------|------|-------|
| QC1 - Physical | `Q_flag_qc1_physical` | 物理范围检查（负值→bad, 缺失→missing） | 0=good, 3=bad, 9=missing |
| QC1 - Physical | `SSC_flag_qc1_physical` | 同上 | 同上 |
| QC1 - Physical | `SSL_flag_qc1_physical` | 同上 | 同上 |
| QC2 - Log-IQR | `Q_flag_qc2_log_iqr` | 对数IQR异常值检测 | 0=pass, 2=suspect, 8=not_checked, 9=missing |
| QC2 - Log-IQR | `SSC_flag_qc2_log_iqr` | 同上 | 同上 |
| QC2 - Log-IQR | `SSL_flag_qc2_log_iqr` | 同上 | 同上 |
| QC3 - SSC-Q | `SSC_flag_qc3_ssc_q` | SSC-Q 一致性检查 | 0=pass, 2=suspect, 8=not_checked, 9=missing |
| QC3 - SSL propagation | `SSL_flag_qc3_from_ssc_q` | SSC-Q 不一致传播到由 Q 和 SSC 派生的 SSL | 0=not_propagated, 2=propagated, 8=not_checked, 9=missing |

**Canonical rule:** `SSL_flag_qc3_from_ssc_q` 的 propagated 状态必须使用 `2`。这里的 `2` 与最终质量标志中的 `suspect` 一致，表示“SSC-Q 不一致已传播到 SSL，因此 SSL 应被视为 suspect”。不要使用 `1` 表示 propagated；`1` 只保留给最终变量标志中的 `estimated/derived`。代码层面的统一定义见 `code/constants.py` 中的 `QC3_SSL_FROM_SSC_Q_FLAG_VALUES`、`QC3_SSL_FROM_SSC_Q_FLAG_MEANINGS` 和 `QC3_SSL_FROM_SSC_Q_PROPAGATED_FLAG`。

### 3.3 NetCDF 必须写出的 QC 变量

**所有 QC NetCDF 文件必须同时输出最终标志和逐步 QC 标志**（`second_stage_contract.md` 第3节要求）：

| 必须输出 | 可选输出（在源数据存在时） |
|---------|----------------------|
| `Q_flag` | `Q_flag_qc1_physical` |
| `SSC_flag` | `Q_flag_qc2_log_iqr` |
| `SSL_flag` | `SSC_flag_qc1_physical` |
| `Q_flag_qc1_physical` | `SSC_flag_qc2_log_iqr` |
| `Q_flag_qc2_log_iqr` | `SSC_flag_qc3_ssc_q` |
| `SSC_flag_qc1_physical` | `SSL_flag_qc1_physical` |
| `SSC_flag_qc2_log_iqr` | `SSL_flag_qc2_log_iqr` |
| `SSC_flag_qc3_ssc_q` | `SSL_flag_qc3_from_ssc_q` |
| `SSL_flag_qc1_physical` | |
| `SSL_flag_qc2_log_iqr` | |
| `SSL_flag_qc3_from_ssc_q` | |

> **当前状态**：仅 HYBAM、Eurasian_River、bayern、Chao_Phraya 写入了逐步QC标志。
> 其余 14 个数据集（EUSEDcollab、USGS、GFQA_v2 等）虽然调用了共享 QC 函数，但写入 NetCDF 时只写了最终 Q_flag/SSC_flag/SSL_flag——**这是一个已知缺口**。逐步 QC 标志的补充写入计划见 [issue 补充计划](#)。

### 3.4 derived/estimated 标记

当通过 `SSL = Q × SSC × 0.0864` 恒等式对缺失变量做**逆向推导**时（即不是正向计算 SSL，而是反推 Q 或 SSC），必须分别标记：

- `Q_derived` — Q 是通过 `Q = SSL / (SSC × 0.0864)` 反算得出的
- `SSC_derived` — SSC 是通过 `SSC = SSL / (Q × 0.0864)` 反算得出的
- `SSL_derived` — SSL 是通过 `SSL = Q × SSC × 0.0864` 正算得出的

这些标记应传入 QC 函数的 `flag_estimated_mask` 参数，使对应的 derived 值被标记为 `flag=1 (estimated)`。

> 注：目前仅 EUSEDcollab 需要此机制（三个变量互有缺失需双向推导）。其他数据集的 SSL 总是有完整 Q 和 SSC 才计算，不需要 derived 标记。

---

## 4. NetCDF 全局属性标准

### 4.1 必须写出的标准属性

参考 `modify_global_attribute.md`，所有 QC NetCDF 必须包含以下属性。**属性名必须精确匹配**：

#### 站点身份
| 属性名 | 说明 |
|--------|------|
| `station_id` | **标准站点标识键名**（不是 `Source_ID`、`source_id` 等） |
| `station_name` | 站点名称 |
| `river_name` | 河流名称（无则填空字符串） |

#### 地理空间
| 属性名 | 说明 |
|--------|------|
| `country` | 国家名称 |
| `continent_region` | 大洲/区域 |
| `geospatial_lat_min` / `geospatial_lat_max` | 纬度范围 |
