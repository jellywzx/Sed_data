# Sed_data — 全球河流泥沙数据处理系统

本项目是一套面向全球河流悬沙浓度（SSC）、悬沙通量（SSL）及流量（Q）数据的处理与质量控制（QC）系统，涵盖 27 个区域/全球数据集，最终输出符合 CF-1.8 / ACDD-1.3 标准的 NetCDF 文件。

---

## 目录结构

```
Sed_data/
├── tool.py                        # 核心工具库（1,729 行）
├── code/                          # 共享工具模块
│   ├── constants.py               # 质控标志值与填充值常量
│   ├── geo.py                     # 地理坐标转换
│   ├── time_utils.py              # 时间解析工具
│   ├── units.py                   # 单位换算函数
│   ├── qc.py                      # 核心质控标志逻辑
│   ├── ssc_q_consistency.py       # 水文一致性检查
│   ├── metadata.py                # CF/ACDD 元数据构建器
│   ├── cf_writer.py               # 通用 NetCDF 写入器
│   └── adapter.py                 # 数据集输入适配器（开发中，存在问题，详见"已知问题"）
│
├── ALi_De_Boer/                   # 印度河（Ali & De Boer 2007）
├── Chao_Phraya_River/             # 湄南河
├── Dethier/                       # 冰川融水河流（Dethier）
├── EUSEDcollab/                   # 欧洲泥沙协作数据库
├── Eurasian_River/                # 欧亚多站点河流
├── Fukushima/                     # 福岛核事故后泥沙数据
├── GFQA_v2/                       # 全球流量与水质档案 v2
├── GSED/                          # 全球泥沙数据集
├── GloRiSe/                       # 全球河流泥沙数据库
├── HMA/                           # 高山亚洲冰川河流
├── HYBAM/                         # 亚马逊河（HYBAM 网络）
├── Huanghe/                       # 黄河
├── Hydat/                         # 加拿大水文测量数据
├── Land2sea/                      # 陆海通量模型输出
├── Mekong_Delta/                  # 湄公河三角洲站点
├── Milliman/                      # Milliman & Meade 全球汇编
├── Myanmar/                       # 缅甸河流站点
├── NERC/                          # 英国 NERC 环境数据
├── Rhine/                         # 莱茵河
├── RiverSed/                      # RiverSed 汇编
├── Robotham/                      # Robotham 数据集
├── Shashi_Jianli/                 # 长江（沙市—监利河段）
├── USGS/                          # 美国地质调查局（NWIS）
├── Vanmaercke/                    # Vanmaercke 泥沙数据
├── Yajiang/                       # 雅砻江
└── bayern/                        # 巴伐利亚河流网络
```

---

## 核心模块（`/code/`）

| 文件 | 行数 | 功能说明 |
|------|------|---------|
| `constants.py` | 11 | 标志常量：`FLAG_GOOD=0`、`FLAG_SUSPECT=2`、`FLAG_BAD=3`、`FLAG_MISSING=9` |
| `geo.py` | 20 | 度分秒（DMS）→ 十进制度坐标转换 |
| `time_utils.py` | 22 | 时间段字符串解析、气候态时间编码 |
| `units.py` | 21 | 流量、SSL、SSC 单位换算 |
| `qc.py` | 37 | `apply_quality_flag()`、`compute_log_iqr_bounds()`、`apply_ssl_log_iqr_flag()` |
| `ssc_q_consistency.py` | 130 | 流量分箱、SSC-Q 异常检测、标志传播 |
| `metadata.py` | 379 | 构建 CF-1.8 NetCDF 的维度/变量/全局属性字典 |
| `cf_writer.py` | 75 | 通用 NetCDF4 文件写入器 |
| `adapter.py` | 146 | 输入适配器（**存在问题，见"已知问题"第1条**） |

---

## 核心工具库（`tool.py`）

`tool.py` 是所有数据集处理脚本最常用的文件，按功能分为六个部分：

### 第一部分 — 解析（第 24–80 行）
- `parse_dms_to_decimal(dms_str)` — 度分秒坐标 → 十进制度
- `parse_period(period_str)` — `"YYYY-YYYY"` 字符串 → 起止年份

### 第二部分 — 变量计算（第 82–150 行）
- `calculate_discharge(runoff_mm_yr, area_km2)` → 流量 Q（m³/s）
- `calculate_ssl_from_mt_yr(sediment_mt_yr)` → 悬沙通量 SSL（ton/day）
- `convert_ssl_units_if_needed(ssl_da)` — 归一化 sediment_flux 单位
- `calculate_ssc(ssl_ton_day, discharge_m3s)` → 悬沙浓度 SSC（mg/L）

### 第三部分 — 质量控制（第 152–785 行）
- `compute_log_iqr_bounds(values, k=1.5)` — 对数空间 IQR 异常值边界
- `apply_log_iqr_screening(data, variable_name, bounds, k)` — QC2 统计筛查
- `apply_quality_flag_array(values, variable_name)` — 批量标志赋值
- `apply_hydro_qc_with_provenance(df, station_id, diagnostic_dir)` — 带逐步溯源的多层级统一 QC 流程
- `build_ssc_q_envelope(q, ssc)` — 构建流量分层 SSC 包络线
- `check_ssc_q_consistency(q, ssc, ssc_flag)` — 水文合理性检验
- `propagate_ssc_q_inconsistency_to_ssl(ssc_q_flags, ssl_flags)` — SSC 标志传播至 SSL

### 第四部分 — 元数据验证（第 902–1096 行）
- `check_variable_metadata_tiered(nc, variable_name, requirements, tier)` — 三级 CF/ACDD 合规性检查
- `check_nc_completeness(nc_path, variables, min_pct)` — NetCDF 数据覆盖率验证

### 第五部分 — 诊断绘图（第 1096–1224 行）
- `plot_ssc_q_diagnostic(df, station_id, diagnostic_dir)` — 带流量分箱的 SSC-Q 散点图

### 第六部分 — CSV 汇总输出（第 1224–1729 行）
- `generate_station_summary_csv(stations_info, output_dir, dataset_name)` — 站点综合汇总表
- `generate_qc_results_csv(stations_info, output_csv, preferred_cols)` — 逐变量质控标志统计表
- `generate_warning_summary_csv(stations_info, output_csv)` — 质控告警汇总表

---

## 数据处理流程

### 各数据集独立处理

每个数据集文件夹独立处理，通用流程如下：

```
原始数据（Excel / CSV / ASCII）
        │
        ▼
解析与字段标准化
        │  （数据集专属脚本）
        ▼
计算衍生变量
        │  Q（m³/s）= 径流量 × 集水面积 / 31,557.6
        │  SSL（ton/day）= 泥沙（Mt/yr）× 10⁶ / 365.25
        │  SSC（mg/L）= SSL / (Q × 0.0864)
        ▼
QC 第一层 — 物理合理性检验
        │  负值 / NaN → FLAG_BAD(3) / FLAG_MISSING(9)
        ▼
QC 第二层 — 统计异常值检测
        │  对数空间 IQR（k=1.5）
        │  极端异常值 → FLAG_SUSPECT(2)
        ▼
QC 第三层 — 水文一致性检验
        │  SSC-Q 流量分箱 → 不一致 → FLAG_SUSPECT(2)
        │  SSC 标志传播 → SSL 标志
        ▼
构建 CF/ACDD 元数据
        │  （metadata.py）
        ▼
写出 NetCDF 文件
        │  （cf_writer.py）
        ▼
输出：station_XXXXX.nc（CF-1.8 / ACDD-1.3）
     station_summary.csv
     qc_results.csv
```

### 质控标志说明

| 标志名称 | 数值 | 含义 |
|---------|------|------|
| `FLAG_GOOD` | 0 | 通过所有质控检查 |
| `FLAG_ESTIMATED` | 1 | 衍生或插值数据 |
| `FLAG_SUSPECT` | 2 | 统计异常值或 SSC-Q 不一致 |
| `FLAG_BAD` | 3 | 负值或物理上不可能的值 |
| `FLAG_NOT_CHECKED` | 8 | 数据量不足，未执行此 QC 步骤 |
| `FLAG_MISSING` | 9 | NaN 或填充值 |

---

## 数据集清单

| 数据集 | 处理方式 | 主要脚本 | 备注 |
|--------|---------|---------|------|
| ALi_De_Boer | 单脚本 | `process_data_tool.py` | 印度河；Excel 输入 |
| Chao_Phraya_River | 单脚本 | `process_chao_phraya.py` | |
| Dethier | 单脚本 | `process_dethier_tool.py` | 冰川融水河流 |
| EUSEDcollab | 单脚本 | `process_eusedcollab_to_cf18_wzx.py` | 欧洲数据库 |
| Eurasian_River | 单脚本 | `process_eurasian_river.py` | |
| Fukushima | 单脚本 | `fukushima_qc_and_cf_enhancement.py` | 核事故后数据 |
| GFQA_v2 | 单脚本 | `gfqa_to_netcdf_daily_dualqc.py` | 双层 QC |
| GSED | 单脚本 | `process_gsed_cf18.py` | 集成 ShapeFile |
| GloRiSe | 多阶段（4步） | `1_generate_netcdf_SS.py` + 3 个 QC 脚本 | SS / BS / 营养盐 |
| HMA | 单脚本 | `convert_to_netcdf_cf18_qc.py` | 高山亚洲 |
| HYBAM | 单脚本 | `hybam_comprehensive_processor.py` | 亚马逊；脚本约 42K 行 |
| Huanghe | 两阶段 | `convert_to_netcdf.py` → `qc_and_standardize.py` | 黄河 |
| Hydat | 四阶段 | `1_` → `2_` → `3_` → `4_process_hydat_cf18.py` | 加拿大站点 |
| Land2sea | 单脚本 | `convert_land2sea_to_netcdf.py` | 模型输出 |
| Mekong_Delta | 规范入口 + 辅助脚本 | `process_mekong_delta.py` | `convert_to_netcdf*.py`、`verify_qc.py`、`summarize_data.py` 为辅助/遗留脚本 |
| Milliman | 五阶段 | `1_convert_to_netcdf.py` → … → `5_qc.py` | 全球汇编 |
| Myanmar | 单脚本 | `convert_to_netcdf.py` | 含验证与汇总脚本 |
| NERC | 单脚本 | `convert_NERC_to_netcdf.py` | 含精化与验证步骤 |
| Rhine | 单脚本 | `process_rhine.py` | 正则表达式解析带引号字段 |
| RiverSed | 单脚本 | `convert_to_netcdf.py` | |
| Robotham | 单脚本 | `convert_to_netcdf_v2.py` | |
| Shashi_Jianli | 单脚本 | `process_shashi_jianli.py` | 长江沙市—监利河段 |
| USGS | 三阶段 | `process_usgs.py` + existing + merge | NWIS；CFS → CMS 换算 |
| Vanmaercke | 两阶段 | `convert_to_netcdf.py` → `qc_and_standardize.py` | |
| Yajiang | 两阶段 | `convert_to_nc.py` → `process_yajiang.py` | 冰川融水河流 |
| bayern | 两阶段 | `convert_bayern_to_netcdf.py` → `qc_and_standardize.py` | 含 5 个诊断绘图脚本 |

---

## 运行方式

现在提供了顶层运行器 `run_pipeline.py`，推荐优先通过它查看和执行数据集流程：

```bash
cd Script
python run_pipeline.py --list
python run_pipeline.py GloRiSe --dry-run
python run_pipeline.py Milliman
python run_pipeline.py --all
```

多阶段执行顺序见 [PIPELINES.md](./PIPELINES.md)。

如需自定义路径，可通过命令行或环境变量覆盖：

```bash
python run_pipeline.py GloRiSe \
  --source-root /path/to/Source \
  --output-root /path/to/Output_r
```

---

## 本次修复（2026-04-13）

### 1. `code/adapter.py` 已恢复可用

- 移除了不存在的 `core.*` 依赖残留。
- 补上了缺失的 `numpy` / `pandas` 导入。
- 新增 Excel 输入列校验，缺列时给出明确错误。

### 2. QC 共享实现已统一到 `code/qc.py`

- `code/qc.py` 现在提供权威的 QC1 / QC2 / QC3 共享实现。
- `tool.py` 中的同名接口改为兼容包装层，继续服务现有数据集脚本。
- `apply_quality_flag()` 现兼容 `tool.py` 旧签名，消除了接口不一致问题。

### 3. 共享单位换算已统一

- `code/units.py` 与 `tool.py` 统一采用 `365.25` 天/年。
- 同步修复了共享 `calculate_ssc()` 中的因子错误，现统一使用 `0.0864`。
- `metadata.py` 与本文档中的 SSC 公式说明已同步更新。

### 4. `tool.py` 的核心职责已下沉到 `/code/`

- 解析、变量计算和 QC 的权威逻辑已经迁移到 `code/` 模块。
- `tool.py` 目前主要承担向后兼容入口的职责。
- 文件体量仍然偏大，但测试与复用现在可以优先基于 `/code/` 进行。

### 5. 已新增全局调度脚本

- 顶层新增 `run_pipeline.py`。
- 支持 `--list`、`--dry-run`、按数据集运行和全量运行。

### 6. 输出路径现在可配置

- 新增 `code/runtime.py` 统一解析 `Source` / `Output_r` 根目录。
- 关键多阶段脚本已接入统一路径解析。
- `run_pipeline.py` 支持 `--output-root` 和 `SEDIMENT_OUTPUT_ROOT`。

### 7. 已补充输入预检工具

- 新增 `code/validation.py`，提供文件存在性、目录存在性和表字段校验。
- `adapter.py`、`Vanmaercke/convert_to_netcdf.py` 以及若干关键入口脚本已接入预检。

### 8. 多阶段流程已有统一说明

- 新增 [PIPELINES.md](./PIPELINES.md) 记录规范执行顺序。
- `run_pipeline.py --list` 可直接查看所有数据集的规范入口与阶段顺序。
