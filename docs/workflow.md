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
| `geospatial_lon_min` / `geospatial_lon_max` | 经度范围 |
| `upstream_area` | 上游集水面积 (km²) |
| `geographic_coverage` | 文字描述的地理覆盖范围 |

#### 时间
| 属性名 | 说明 |
|--------|------|
| `temporal_resolution` | 时间分辨率（daily/monthly/annual/climatology） |
| `temporal_span` | 覆盖期字符串，如 `"2000-01-01 to 2020-12-31"` |
| `time_coverage_start` | 起始日期 |
| `time_coverage_end` | 结束日期 |

#### 数据来源与引用
| 属性名 | 说明 |
|--------|------|
| `data_source_name` | 数据集名称 |
| `source_data_link` | 原始数据链接 |
| `creator_institution` | 创建机构 |
| `creator_name` | 创建者 |
| `creator_email` | 创建者邮箱 |
| `references` | 参考文献（多个用 ` | ` 拼接） |
| `source` | 数据来源简述 |

#### CF/ACDD 惯例
| 属性名 | 说明 |
|--------|------|
| `Conventions` | 强制覆写为 `CF-1.8, ACDD-1.3` |
| `title` | 数据集标题 |
| `summary` | 数据集摘要 |
| `history` | 处理历史 |
| `date_created` | 创建日期 |
| `date_modified` | 修改日期 |
| `processing_level` | 处理等级 |
| `featureType` | 特征类型 |
| `comment` | 补充说明 |
| `variables_provided` | 提供的变量列表 |

### 4.2 向后兼容属性

在写入标准 `station_id` 的同时，建议保留旧的兼容属性名（如 `Source_ID`），确保老版本的 `scripts_basin_test` 仍可解析：

```python
ds.station_id = str(station_id)   # 标准键名（必须）
ds.Source_ID = str(station_id)    # 兼容键名（建议保留）
```

> 当前处理脚本使用的站点 ID 键名不一：`station_id`（GSED、Huanghe、RiverSed、Yajiang、Vanmaercke、Fukushima、Hydat、bayern）、`Source_ID`（HYBAM、Mekong_Delta、Shashi_Jianli、Eurasian_River、GFQA_v2、Robotham、Yajiang）、或两者兼有。建议统一写 `station_id`。

---

## 5. NetCDF 变量标准

### 5.1 坐标变量

| 变量名 | 维度 | 类型 | units |
|--------|------|------|-------|
| `time` | (time) | f8 | `days since 1970-01-01 00:00:00` |
| `lat` | 标量 | f4 | `degrees_north` |
| `lon` | 标量 | f4 | `degrees_east` |
| `altitude` | 标量 | f4 | `m` |
| `upstream_area` | 标量 | f4 | `km2` |

### 5.2 数据变量

| 变量名 | 维度 | 类型 | units | ancillary_variables |
|--------|------|------|-------|-------------------|
| `Q` | (time) | f4 | `m3 s-1` | `Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr` |
| `SSC` | (time) | f4 | `mg L-1` | `SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q` |
| `SSL` | (time) | f4 | `ton day-1` | `SSL_flag SSL_flag_qc1_physical SSL_flag_qc2_log_iqr SSL_flag_qc3_from_ssc_q` |

### 5.3 QC 标志变量

最终 QC 标志变量使用 `int8` 类型，共享相同的 flag_values/flag_meanings：

```python
flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
```

逐步 QC 标志不使用同一套含义。尤其是 `SSL_flag_qc3_from_ssc_q` 必须使用以下独立 contract：

```python
flag_values = np.array([0, 2, 8, 9], dtype=np.int8)
flag_meanings = 'not_propagated propagated not_checked missing'
```

其中 `2=propagated`，表示 SSC-Q inconsistency 已传播到派生 SSL，因此 SSL 的最终标志也应为 suspect。不要将 `1` 用于 propagated，因为 `1` 只表示最终变量值为 estimated/derived。

---

## 6. FillValue 约定

| 变量类型 | FillValue |
|---------|-----------|
| 浮点变量 (Q, SSC, SSL) | `-9999.0` (对应 `code.constants.FILL_VALUE_FLOAT`) |
| 整型变量 (flags) | `9` (对应 `code.constants.FILL_VALUE_INT`，与 `FLAG_MISSING` 一致) |

数据流中应使用 `np.nan` 进行计算，最终写入 NetCDF 时转为 FillValue。

---

## 7. QC 数据流规范

```
原始数据
    │
    ▼
[单位转换]  → Q: m³/s, SSC: mg/L, SSL: ton/day
    │
    ▼
[缺失推导]  → 仅 EUSEDcollab: 用 SSL=Q×SSC×0.0864 双向推导，标记 Q_derived/SSC_derived/SSL_derived
    │
    ▼
[apply_hydro_qc_with_provenance]
    │  ├─ QC1: Physical (flag bad/missing)
    │  ├─ QC2: Log-IQR outlier (flag suspect)
    │  └─ QC3: SSC-Q envelope + SSL propagation (flag suspect; step flag uses 2=propagated)
    │
    ▼
[trim_to_valid_data]  → 裁剪到有效数据范围
    │
    ▼
[write_netcdf]  → 写 CF-1.8 兼容 NetCDF，包括 final flags + stepwise flags
    │
    ▼
[generate_summary_csv]  → 站点汇总表
```

---

## 8. 时间分辨率约定

| 实际时间分辨率 | 输出 temporal_resolution | s2 输出目录 |
|---------------|------------------------|------------|
| 小时 | `daily` | `daily` |
| 日 | `daily` | `daily` |
| 月 | `monthly` | `monthly` |
| 季度 | `monthly` | `monthly` |
| 年 | `annual` | `annual` |
| 多年平均/气候态 | `climatology` | `climatology` |

输出目录名必须与 `temporal_resolution` 全局属性一致。

---

## 9. 输出路径约定

```
Output_r/
  {resolution}/           ← daily / monthly / annual / climatology
    {dataset_name}/       ← EUSEDcollab / USGS / HYBAM / …
      qc/                 ← 标准化 NetCDF 输出目录
        EUSEDcollab_*.nc
        …
```

---

## 10. 当前差距清单

| 规范要求 | 已符合的数据集 | 有差距的数据集 |
|---------|--------------|--------------|
| SSL 公式 0.0864 | EUSEDcollab, USGS, GFQA, HYBAM, Hydat | — |
| NaN guard in SSL calc | EUSEDcollab, USGS, GFQA, HYBAM | — |
| 逐步 QC 标志写入 NC | HYBAM, Eurasian_River, bayern, Chao_Phraya | **14 个数据集**（EUSEDcollab, USGS, GFQA, Dethier, Fukushima, Hydat, Mekong_Delta, Myanmar, NERC, Rhine, Robotham, Shashi_Jianli, Yajiang, ALi_De_Boer） |
| `station_id` 标准键名 | GSED, Huanghe, RiverSed, Yajiang, Vanmaercke, Fukushima, Hydat, bayern | **需补充** station_id（HYBAM, Eurasian_River, GFQA, EUSEDcollab 等主要使用 Source_ID） |
| `ancillary_variables` 关联完整 | HYBAM | **其余数据集**（EUSEDcollab 只关联 Q_flag/SSC_flag/SSL_flag） |

---

## 11. 数据集处理模式速查

| 模式 | 代表数据集 | 特征 |
|------|-----------|------|
| **单脚本一体化** | HYBAM, GFQA, HMA, Rhine, EUSEDcollab | 1 个脚本完成读取→QC→标准化→输出 |
| **两步式** | Huanghe, Vanmaercke, bayern | convert + qc_and_standardize 分离 |
| **多步流水线** | Hydat (4步), Milliman (5步), GloRiSe (4步) | 流程拆分更细 |
| **主处理+验证** | GSED, Myanmar, Mekong_Delta | 主脚本 + 独立验证/绘图脚本 |
