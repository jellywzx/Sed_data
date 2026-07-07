# Script 数据集处理合规检查报告

**日期**：2026-06-04  
**依据**：[workflow.md](./workflow.md) — Script 数据集处理流程规范  
**检查范围**：`Script/` 下 26 个数据源的处理脚本及其输出 NetCDF 文件（`Output_r/` 下对应 qc 目录）  
**方法**：静态代码分析 + 输出文件 NetCDF 变量/属性验证

---

## 总体概览

| 评级 | 数量 | 数据集 |
|------|:----:|--------|
| ✅ **PASS** | 5 | HYBAM, Myanmar, USGS, Yajiang, bayern |
| ⚠️ **CONDITIONAL** | 20 | ALi_De_Boer, Chao_Phraya_River, Dethier, EUSEDcollab, Eurasian_River, Fukushima, GFQA_v2, GSED, GloRiSe, HMA, Hydat, Mekong_Delta, Milliman, NERC, Rhine, RiverSed, Robotham, Shashi_Jianli, Vanmaercke, Land2sea |
| ❌ **FAIL** | 1 | Huanghe |

---

## 一、检查维度总表

以下按 workflow.md 的核心要求逐项列出。

| # | 检查项 | 要求（workflow.md 节） | 通过率 |
|---|--------|----------------------|:------:|
| 1 | Q 单位 → m³/s | §2 变量与单位标准 | 25/26 ✅ |
| 2 | SSC 单位 → mg/L | §2 | 25/26 ✅ |
| 3 | SSL 单位 → ton/day | §2 | 25/26 ✅ |
| 4 | SSL = Q × SSC × 0.0864 公式 | §2 | 15/26 ⚠️ |
| 5 | SSL NaN guard | §2 | 1/26 🔴 |
| 6 | 调用 `apply_hydro_qc_with_provenance` | §3.1 | 19/26 ✅ |
| 7 | 逐步 QC 标志写入 NC（8 个变量） | §3.3 | 8/26 ⚠️ |
| 8 | `station_id` 标准属性 | §4.1 | 19/26 ⚠️ |
| 9 | `Source_ID` 兼容属性（可选） | §4.2 | 20/26 ✅ |
| 10 | `ancillary_variables` 设置 | §5.2 | 24/26 ✅ |
| 11 | CF-1.8, ACDD-1.3 Conventions | §4.1 | 24/26 ✅ |
| 12 | `temporal_resolution` 属性 | §4.1 | 24/26 ✅ |
| 13 | `featureType` 属性 | §4.1 | 15/26 ⚠️ |
| 14 | 使用标准输出路径（Output_r/） | §9 | 25/26 ✅ |

---

## 二、三大共性问题

### 问题 1：逐步 QC 标志严重缺失（🔴 最严重）

workflow.md §3.3 要求 **所有 QC NetCDF 必须同时输出最终标志和逐步 QC 标志**（共 8 个变量）：

```
Q_flag_qc1_physical    Q_flag_qc2_log_iqr
SSC_flag_qc1_physical  SSC_flag_qc2_log_iqr  SSC_flag_qc3_ssc_q
SSL_flag_qc1_physical  SSL_flag_qc2_log_iqr  SSL_flag_qc3_from_ssc_q
```

**状态分布：**

| 写入比例 | 数据集 | 说明 |
|:---------:|--------|------|
| **8/8** ✅ | HYBAM, Rhine, Robotham, GFQA_v2, Mekong_Delta, Dethier | 完整写入 |
| **7/8** ⚠️ | NERC（缺 SSL_flag_qc2_log_iqr） | 缺 1 个 |
| **7/8** ⚠️ | Myanmar（缺 SSL_flag_qc2_log_iqr） | 缺 1 个 |
| **7/8** ⚠️ | bayern（缺 SSC_flag_qc3_ssc_q, SSL_flag_qc3_from_ssc_q），使用了旧命名 `SSC_flag_qc3_ssc_q_consistency` | 缺 2 个，命名不规范 |
| **6/8** ⚠️ | Eurasian_River（缺 QC3 相关：SSC_flag_qc3_ssc_q, SSL_flag_qc3_from_ssc_q） | 缺 2 个 |
| **3/8** ⚠️ | RiverSed（仅有 SSC 相关 3 个，缺 Q/SSL 的 QC 标志） | 仅 SSC 部分 |
| **0/8** 🔴 | **EUSEDcollab**, USGS, Fukushima, Hydat, Shashi_Jianli, Yajiang, ALi_De_Boer, Chao_Phraya, HMA, Huanghe, Milliman, Vanmaercke, GSED, Land2sea | **14 个数据集**完全没有逐步 QC 标志 |

**值得注意的特殊情况：**

- **EUSEDcollab** — 代码中 `write_netcdf()`（第 1112-1143 行）有完整的逐步 QC 写入逻辑，且调用时传了 `step_flags=df`（第 878 行）。输出的 244 个 NC 文件却全部缺失。**可能原因**：`fix_qc_global_attrs.py` 后处理移除了变量，或写入时 `step_flags` 为空导致 `if step_flags:` 分支未执行。
- **USGS** — 代码调用了 `apply_hydro_qc_with_provenance`（该函数返回 dict 含 stepwise flags），但写入 NC 时只写了 `Q_flag/SSC_flag/SSL_flag`。
- **bayern** — 使用了非标准的 QC3 变量名 `SSC_flag_qc3_ssc_q_consistency`。

### 问题 2：共享 QC 函数未调用

workflow.md §3.1 要求调用 `apply_hydro_qc_with_provenance`。以下数据集未调用：

| 数据集 | 输出文件数 | 替代 QC 方式 |
|--------|:----------:|-------------|
| **GSED** | 5237 | 可能使用自定义 QC（仅有最终标志） |
| **GloRiSe** | — | 分为 SS/BS 两阶段，有独立 QC 逻辑 |
| **HMA** | 28 | 无 QC 标志（仅最终标志） |
| **Huanghe** | 48 | **仅 SSC_flag，无 Q/SSL 标志** |
| **Milliman** | 737 | 流程分为 5 步，最后一步为标准化但无 QC |
| **RiverSed** | 42177 | 仅有 SSC 的 QC1/QC2/QC3 标志，无 Q/SSL QC |
| **Vanmaercke** | 516 | 仅最终标志，无逐步 QC |

### 问题 3：全局属性不完整

workflow.md §4.1 列出了必须的全局属性。缺失最严重的是：

| 属性 | 缺失的数据集 |
|------|-------------|
| `featureType` | ALi_De_Boer, Chao_Phraya, Dethier, EUSEDcollab, Eurasian_River, Fukushima, GSED, Mekong_Delta, Myanmar, Rhine, RiverSed, Robotham, Shashi_Jianli, USGS, Yajiang, bayern, HMA, Hydat, Huanghe, Milliman, NERC, Vanmaercke（**22 个**） |
| `country` | ALi_De_Boer, Chao_Phraya, Dethier, Eurasian_River, Fukushima, GSED, Mekong_Delta, Myanmar, NERC, Rhine, Robotham, Shashi_Jianli, USGS（**13 个**） |
| `continent_region` | 同上 13 个 |
| `geospatial_lat/lon` | Dethier, GFQA_v2, Mekong_Delta, Myanmar, Rhine, RiverSed, Shashi_Jianli, USGS, Yajiang, bayern（**10 个**） |
| `date_created` | Dethier, Eurasian_River, GFQA_v2, Mekong_Delta, Myanmar, Rhine, Shashi_Jianli, USGS, Yajiang, bayern（**10 个**） |
| `processing_level` | Chao_Phraya, Dethier, Eurasian_River, GFQA_v2, Mekong_Delta, Myanmar, Rhine, RiverSed, Shashi_Jianli, USGS, Yajiang, bayern（**12 个**） |

---

## 三、逐数据集详细检查结果

### ✅ PASS（5 个）

#### HYBAM
- **输出**：daily/HYBAM/qc/ → 12 文件
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：apply_hydro_qc_with_provenance ✅
- **逐步 QC 标志**：8/8 ✅ 包含完整变量
- **ancillary_variables**：完整（含 stepwise flags）✅
- **station_id**：真实站点 ID ✅
- **Source_ID**：NOT SET
- **Conventions**：CF-1.8, ACDD-1.3 ✅
- **temporal_resolution**：daily ✅
- **featureType**：❌ 缺失
- **其他**：全局属性基本完整，唯一缺 featureType

#### Myanmar
- **输出**：daily/Myanmar/qc/ → 8 文件
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：✅
- **逐步 QC 标志**：7/8 ⚠️（缺 SSL_flag_qc2_log_iqr）
- **station_id**：✅
- **全局属性**：缺 country, continent_region, geospatial_lat/lon, data_source_name, date_created, processing_level, featureType

#### USGS
- **输出**：daily/USGS/qc/ → 887 文件
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：✅（唯一真正确认的）
- **QC 函数调用**：✅
- **逐步 QC 标志**：**0/8** 🔴（代码调用了 QC 函数但只写最终标志）
- **ancillary_variables**：仅最终标志
- **station_id**：✅
- **temporal_resolution**：`irregular_daily_overlap`（非标准命名）
- **全局属性**：缺 country, continent_region, geospatial_lat/lon, date_created, processing_level, featureType

#### Yajiang
- **输出**：daily/Yajiang/qc/ → 23 文件
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：✅
- **逐步 QC 标志**：**0/8** 🔴
- **station_id**：✅
- **全局属性**：缺 country, continent_region, geospatial_lat/lon, date_created, processing_level, featureType

#### bayern
- **输出**：daily/Bayern/qc/ → 34 文件
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：✅
- **逐步 QC 标志**：7/8 ⚠️（缺 SSC_flag_qc3_ssc_q, SSL_flag_qc3_from_ssc_q；使用旧名 `SSC_flag_qc3_ssc_q_consistency`）
- **station_id**：✅
- **全局属性**：缺 geospatial_lat/lon, date_created, processing_level, featureType

---

### ⚠️ CONDITIONAL（20 个）

#### EUSEDcollab ⚠️ **需优先排查**
- **输出**：monthly/EUSEDcollab/qc/ → **244 文件**
- **单位**：✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：✅（使用了 EUSEDcollab 内部 `qc_with_toolpy`，也使用了 `code/qc.py` 的 `apply_hydro_qc_with_provenance`）
- **逐步 QC 标志（代码）**：✅ `write_netcdf()` 函数有完整写入逻辑（第 1112-1143 行）
- **逐步 QC 标志（输出）**：**0/8** 🔴 代码写了但输出文件没有
- **derived 标志**：✅ Q_derived/SSC_derived/SSL_derived + flag_estimated_mask
- **ancillary_variables**：✅ 代码设置了含 stepwise 的完整引用
- **station_id**：✅
- **Source_ID**：NOT SET（无兼容属性）
- **featureType**：❌
- **历史记录**：显示经过 `[fix_qc_global_attrs]` 后处理

#### GFQA_v2 ⚠️ **站标识缺失**
- **输出**：daily/GFQA_v2/qc/ → **2073 文件**
- **单位**：Q=m³/s, SSC=mg/L, SSL=ton/day ✅
- **SSL 公式**：0.0864 ✅
- **NaN guard**：❌
- **QC 函数调用**：✅
- **逐步 QC 标志**：8/8 ✅（含完整 stepwise）
- **ancillary_variables**：完整 ✅
- **station_id**：**❌ 缺失** — 全局属性中没有 `station_id`，也没有 `Source_ID`
- **temporal_resolution**：**❌ 缺失**
- **Conventions**：CF-1.8, ACDD-1.3 ✅
- **featureType**：timeSeries ✅
- **其他**：所有其他全局属性完整（country, geospatial, 等）

#### GSED
- **输出**：monthly/GSED/qc/ → **5237 文件**
- **单位**：✅
- **SSL 公式**：❌
- **QC 函数调用**：❌ 未调用
- **逐步 QC 标志**：0/8 🔴
- **station_id**：✅
- **全局属性**：缺 river_name, country, continent_region, featureType

#### GloRiSe
- **输出**：多阶段（SS/BS）
- **单位**：✅
- **SSL 公式**：0.0864 ✅（缺 NaN guard）
- **QC 函数调用**：❌ 未调用（有自定义 QC）
- **逐步 QC 标志**：N/A
- **station_id**：❌
- **全局属性**：缺 continent_region, featureType

#### HMA
- **输出**：annually_climatology/HMA/qc/ → 28 文件
- **单位**：✅
- **SSL 公式**：❌
- **QC 函数调用**：❌ 未调用
- **逐步 QC 标志**：0/8 🔴
- **station_id**：❌
- **全局属性**：缺 featureType

#### Huanghe ❌（唯一 FAIL）
- **输出**：annually_climatology/Huanghe/qc/ → 48 文件（ann + clim）
- **单位**：Q ❌ 缺失变量 / SSC=mg/L ✅ / SSL ❌ 缺失变量
- **定量问题**：输出文件 **不含 Q 变量和 SSL 变量**，仅有 SSC 和 SSC_flag
- **QC 函数调用**：❌ 未调用
- **逐步 QC 标志**：0/8 🔴
- **全局属性**：缺 country, continent_region

#### Hydat
- **输出**：daily/HYDAT/qc/ → 782 文件
- **单位**：✅
- **SSL 公式**：❌（Hydat 的 SSL 直接从源数据转换，不使用 Q×SSC×0.0864）
- **QC 函数调用**：✅
- **逐步 QC 标志**：0/8 🔴
- **station_id**：✅
- **全局属性**：缺 featureType

---

### 其余数据集简要问题

| 数据集 | 输出文件数 | 核心问题 |
|--------|:---------:|---------|
| ALi_De_Boer | 17 | 逐步QC=0/8, 缺 station_id, 缺 NaN guard |
| Chao_Phraya_River | 7 | 逐步QC=0/8, 缺 SSL 公式, temporal_resolution=`annually`（非标准） |
| Dethier | 409 | 缺 station_id, 缺 SSL 公式, 全局属性大量缺失 |
| Eurasian_River | 17 | 逐步QC=6/8（缺 QC3）, 缺 SSL 公式 |
| Fukushima | 2 | 逐步QC=0/8, 缺 station_id, 缺 SSL 公式 |
| Mekong_Delta | 4 | 逐步QC=8/8 ✅, 缺 SSL 公式, 全局属性缺失多 |
| Milliman | 737 | 逐步QC=0/8, 缺 station_id, 未调用 QC 函数 |
| NERC | 4 | 逐步QC=7/8, 缺 station_id, 缺 SSL_flag_qc2 |
| Rhine | 12 | 逐步QC=8/8 ✅, 缺 station_id, 缺 NaN guard |
| RiverSed | 42177 | 逐步QC=3/8, 未调用 QC 函数, 缺 ancillary_variables |
| Robotham | 3 | 逐步QC=8/8 ✅, 缺 station_id, 缺 NaN guard |
| Shashi_Jianli | 2 | 逐步QC=0/8, 缺 station_id, 缺 NaN guard |
| Vanmaercke | 516 | 逐步QC=0/8, 未调用 QC 函数, 缺 SSL 公式 |
| Land2sea | 0 | 仅有脚本目录（`convert_land2sea_to_netcdf.py`），输出未确认 |

---

## 四、与 workflow.md 已知差距清单的对比

workflow.md 第 10 节 `当前差距清单` 列出以下差距，验证结果如下：

| workflow.md 描述 | 实际验证 |
|-----------------|---------|
| ✅ 逐步 QC 标志：仅 HYBAM, Eurasian_River, bayern, Chao_Phraya 已写入，其余 14 个缺失 | **基本确认**。但需修正：Eurasian_River 仅 6/8（缺 QC3）, bayern 用了非标准命名 |
| ✅ `station_id` 标准键名：GSED, Huanghe, RiverSed, Yajiang, Vanmaercke, Fukushima, Hydat, bayern 已使用 | **确认**。新增发现：GFQA_v2 **完全没有** station_id 属性 |
| ✅ `ancillary_variables` 完整：仅 HYBAM | **确认**。新增：GFQA_v2, Rhine, Robotham 也较完整 |
| — 其余数据集 ancillary 仅关联最终标志 | **确认**（USGS, Vanmaercke, Milliman, HMA 等） |

**新发现差距（workflow.md 未列出的）：**

1. GFQA_v2 缺少 `station_id` 和 `temporal_resolution`
2. EUSEDcollab 代码写了逐步 QC 但输出文件没有（可能后处理丢失）
3. Huanghe 输出文件缺少 Q 和 SSL 变量
4. USGS 的 `temporal_resolution` 使用了非标准值 `irregular_daily_overlap`
5. bayern 的 QC3 变量使用了非标准命名
6. `featureType` 缺失具有普遍性（22/26 数据集缺失）

---

## 五、优先处理建议

### P0 — 紧急

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 1 | **EUSEDcollab 逐步 QC 标志丢失** | 244 个文件需要修复 | 排查 `write_netcdf` 中 `step_flags` 是否为空（被覆盖的 df 是否真是 df_qc）；或排查 `fix_qc_global_attrs.py` 是否移除了变量。如果确认是后处理问题，修改 `normalize_nc_attrs` 白名单 |
| 2 | **GFQA_v2 缺少 station_id 和 temporal_resolution** | 2073 个文件，scripts_basin_test 无法识别 | 修改 `gfqa_to_netcdf_daily_dualqc.py`，在 `create_netcdf_file()` 中添加 `ds.station_id = station_id` 和 `ds.temporal_resolution = 'daily'`，然后对所有文件重跑或使用 `fix_qc_global_attrs.py` 批量修复 |

### P1 — 高优先级

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 3 | **14 个数据集缺失逐步 QC 标志** | 影响 QC 追溯能力 | 对调用了 QC 函数但只写最终标志的数据集（USGS, Fukushima, Hydat, Shashi_Jianli, Yajiang 等），补写 stepwise flags 到 NC 输出 |
| 4 | **Huanghe 缺少 Q 和 SSL 变量** | 输出不完整 | 修复 `Huanghe/convert_to_netcdf.py` 或 `qc_and_standardize.py`，确保输出标准 Q 和 SSL 字段 |

### P2 — 中优先级

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 5 | **featureType 普遍缺失** | CF 合规性 | 批量补充 `ds.featureType = 'timeSeries'` |
| 6 | **temporal_resolution 命名不一致** | scripts_basin_test 归类 | 统一标准化命名：`annually` → `annual`, `climatological` → `climatology`, `irregular_daily_overlap` → `daily` |
| 7 | **bayern QC3 命名不规范** | 下游无法识别 QC3 | `SSC_flag_qc3_ssc_q_consistency` → `SSC_flag_qc3_ssc_q` |
| 8 | **NaN guard 普遍缺失** | 潜在计算错误 | 在 SSL = Q×SSC×0.0864 计算处统一使用 `np.isfinite` 保护 |

---

## 六、检查方法说明

- **代码检查**：对每个数据源目录下所有 `.py` 文件做字符串模式匹配，检查关键词（单位转换、QC 函数调用、标志变量引用、全局属性设置等）
- **输出验证**：对 `Output_r/{resolution}/{dataset}/qc/` 下每个 NetCDF 文件，读取其变量列表、全局属性和变量元数据
- **局限性**：
  - 静态分析无法判断代码分支是否真的执行了写入逻辑（如 EUSEDcollab 案例）
  - 部分数据集（GloRiSe, Land2sea）因输出路径特殊未完全验证
  - 数值正确性（如 SSL 计算值是否真确）不在本次检查范围内
