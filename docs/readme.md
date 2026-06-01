# Sed_data — 全球河流泥沙数据处理系统

本项目是一套面向全球河流悬沙浓度（SSC）、悬沙通量（SSL）及流量（Q）数据的处理与质量控制（QC）系统，涵盖 27 个区域/全球数据集，最终输出符合 CF-1.8 / ACDD-1.3 标准的 NetCDF 文件。

---

## 目录结构

```
Sed_data/
├── tool.py                        # 兼容性入口（80 行，纯转发层）
├── run_pipeline.py                # 全局调度脚本
├── PIPELINES.md                   # 多阶段数据集执行顺序说明
├── code/                          # 共享工具模块（权威实现）
│   ├── __init__.py                # 注册为 Python 包
│   ├── constants.py               # 质控标志值与物理常量
│   ├── geo.py                     # 地理坐标转换
│   ├── time_utils.py              # 时间解析工具
│   ├── units.py                   # 单位换算函数
│   ├── qc.py                      # 质控逻辑（QC1 / QC2 / QC3，权威实现）
│   ├── metadata.py                # CF/ACDD 元数据构建与验证
│   ├── output.py                  # CSV 汇总输出函数
│   ├── plot.py                    # 诊断绘图函数
│   ├── validation.py              # 输入文件与字段校验
│   └── runtime.py                 # 路径解析（支持环境变量覆盖）
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

所有共享逻辑的权威实现均在 `/code/` 目录下。数据集脚本直接从 `code.xxx` 导入，`tool.py` 仅作向后兼容的转发层保留。

| 文件 | 行数 | 功能说明 |
|------|------|---------|
| `constants.py` | ~25 | 标志常量（`FLAG_GOOD=0`、`FLAG_SUSPECT=2`、`FLAG_BAD=3`、`FLAG_MISSING=9`）及物理常量（`DAYS_PER_JULIAN_YEAR=365.25`、`SSC_DISCHARGE_TO_SSL_FACTOR` 等） |
| `geo.py` | 20 | 度分秒（DMS）→ 十进制度坐标转换 |
| `time_utils.py` | 22 | 时间段字符串解析、气候态时间编码 |
| `units.py` | ~80 | 流量、SSL、SSC 单位换算；`convert_ssl_units_if_needed()` |
| `qc.py` | 411 | QC1 物理检验、QC2 对数IQR筛查、QC3 SSC-Q水文一致性；`apply_hydro_qc_with_provenance()` 统一流程 |
| `metadata.py` | 559 | CF-1.8 NetCDF 元数据构建；`check_variable_metadata_tiered()` 三级合规检查 |
| `output.py` | 545 | 站点汇总 CSV、QC 结果 CSV、告警汇总 CSV 输出函数 |
| `plot.py` | 81 | `plot_ssc_q_diagnostic()` SSC-Q 散点图（带流量分箱） |
| `validation.py` | ~170 | 文件/目录存在性检查、DataFrame 字段校验、`read_excel_validated()`、`check_nc_completeness()` |
| `runtime.py` | 59 | 路径解析：自动定位 `Source/` 和 `Output_r/`，支持环境变量 `SEDIMENT_SOURCE_ROOT` / `SEDIMENT_OUTPUT_ROOT` 覆盖 |

---

## `tool.py` 说明

`tool.py` 现在是一个 **80 行的纯兼容转发层**，不含任何实现代码。它将所有函数从 `code/` 对应模块重新导出，使现有数据集脚本无需修改即可继续工作。

新开发的脚本应直接从 `code.xxx` 导入，例如：

```python
from code.qc import apply_hydro_qc_with_provenance
from code.units import calculate_ssl_from_mt_yr
from code.output import generate_station_summary_csv
```

---

## 数据处理流程

### 各数据集独立处理

每个数据集文件夹独立处理，通用流程如下：

```
原始数据（Excel / CSV / ASCII）
        │
        ▼
解析与字段标准化（含输入列校验）
        │  code/validation.py + 数据集专属脚本
        ▼
计算衍生变量
        │  Q（m³/s）= 径流量 × 集水面积 / 31,557.6
        │  SSL（ton/day）= 泥沙（Mt/yr）× 10⁶ / 365.25
        │  SSC（mg/L）= SSL / (Q × 0.0864)
        ▼
QC 第一层 — 物理合理性检验          [code/qc.py]
        │  负值 / NaN → FLAG_BAD(3) / FLAG_MISSING(9)
        ▼
QC 第二层 — 统计异常值检测          [code/qc.py]
        │  对数空间 IQR（k=1.5）
        │  极端异常值 → FLAG_SUSPECT(2)
        ▼
QC 第三层 — 水文一致性检验          [code/qc.py]
        │  SSC-Q 幂律回归包络线 → 不一致 → FLAG_SUSPECT(2)
        │  SSC 标志传播 → SSL 标志
        ▼
构建 CF/ACDD 元数据                 [code/metadata.py]
        ▼
写出 NetCDF 文件（各脚本直接调用 netCDF4）
        ▼
输出：station_XXXXX.nc（CF-1.8 / ACDD-1.3）
     station_summary.csv
     qc_results.csv
     warning_summary.csv
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
| GloRiSe | 四阶段 | `1_generate_netcdf_SS.py` → `2_qc_and_standardize_glorise.py` → `3_generate_nc_BS.py` → `4_qc_and_standardize_BS.py` | SS / BS / 营养盐 |
| HMA | 单脚本 | `convert_to_netcdf_cf18_qc.py` | 高山亚洲 |
| HYBAM | 单脚本 | `hybam_comprehensive_processor.py` | 亚马逊 |
| Huanghe | 两阶段 | `convert_to_netcdf.py` → `qc_and_standardize.py` | 黄河 |
| Hydat | 四阶段 | `1_` → `2_` → `3_` → `4_process_hydat_cf18.py` | 加拿大站点 |
| Land2sea | 单脚本 | `convert_land2sea_to_netcdf.py` | 模型输出 |
| Mekong_Delta | 单脚本 | `process_mekong_delta.py` | 其他脚本为辅助/遗留 |
| Milliman | 五阶段 | `1_convert_to_netcdf.py` → … → `5_qc_and_standardize.py` | 全球汇编 |
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

通过顶层运行器 `run_pipeline.py` 查看和执行数据集流程：

```bash
# 列出所有数据集及其规范入口
python run_pipeline.py --list

# 预览某数据集的执行步骤（不实际运行）
python run_pipeline.py GloRiSe --dry-run

# 运行单个数据集
python run_pipeline.py Milliman

# 运行全部数据集
python run_pipeline.py --all
```

自定义输出路径：

```bash
# 方式一：命令行参数
python run_pipeline.py GloRiSe \
  --source-root /path/to/Source \
  --output-root /path/to/Output_r

# 方式二：环境变量（对所有脚本生效）
export SEDIMENT_OUTPUT_ROOT=/path/to/Output_r
export SEDIMENT_SOURCE_ROOT=/path/to/Source
python run_pipeline.py --all
```

多阶段数据集的详细执行顺序见 [PIPELINES.md](./PIPELINES.md)。

---

## 待处理事项

| 项目 | 说明 |
|------|------|
| 删除 `code/ssc_q_consistency.py` | 无任何脚本引用，功能已由 `code/qc.py` 覆盖，可直接删除 |
| 删除 `code/cf_writer.py` | 无任何脚本引用，所有数据集直接使用 `netCDF4`，可直接删除 |
| 删除 `code/adapter.py` | ALi_De_Boer 专属逻辑，`process_data_tool.py` 已完整覆盖，建议删除并在 `process_data_tool.py` 中补充 `read_excel_validated` 列校验 |
| 删除 `modify_plan_0413.md` | 临时工作文档，整理完成后可删除 |

