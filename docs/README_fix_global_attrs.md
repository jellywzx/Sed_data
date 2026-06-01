# Script — QC NetCDF 全局属性标准化

## 背景

对 QC NetCDF 文件的全局属性做规范化处理，包括：
- 统一规范 key（如 `references` → `reference`）
- 删除遗留别名（如 `Source_ID`、`data_period_start`、`measurement_period` 等）
- 合并同族属性值（如多个 `reference*` 属性合并为一个 `reference`）

## 修改的文件

| 文件 | 说明 |
|------|------|
| `code/global_attrs.py` | 核心逻辑：规范 key 列表、合并规则、待删除属性列表 |
| `apply_fix_attrs.py` | 对全部 QC 文件执行 `normalize_nc_attrs()` 的入口脚本 |
| `rename_number_observations.py` | 将 `number_of_observations` 重命名为 `number_of_data` |

### 关联的外部文件

| 文件 | 说明 |
|------|------|
| `/share/home/dq134/full_scan_parallel.py` | 全量扫描审计脚本（DUP_FAMILIES 已同步更新） |
| `/share/home/dq134/phase1_sample_audit.py` | 采样审计脚本（DUP_FAMILIES 已同步更新） |

## 包含的变更

### `code/global_attrs.py`（4 处改动）

1. **`CANONICAL_ATTR_ORDER`**：`"references"` → `"reference"`
2. **`REFERENCE_KEYS`**：`"reference"` 提到首位作为规范 key
3. **`build_canonical_attrs`**：输出 attr 改为 `attrs["reference"]`
4. **`LEGACY_GLOBAL_ATTRS_TO_REMOVE`**：追加 12 项遗留属性

### 清理的属性

| 属性 | 规范替代 | 影响文件数 |
|------|----------|-----------|
| `references` | → `reference` | ~12,599 |
| `Reference` | → `reference` | 同族合并 |
| `Reference1` | → `reference` | 409 |
| `reference1` | → `reference` | 19 |
| `reference2` | → `reference` | 19 |
| `Source_ID` | → `station_id` | 8,188 |
| `Station_ID` | → `station_id` | 0 |
| `location_id` | → `station_id` | 828 |
| `data_period_start` | → `time_coverage_start` | 9,236 |
| `data_period_end` | → `time_coverage_end` | 9,236 |
| `measurement_period` | → `temporal_span` | 516 |
| `source_url` | → `source_data_link` | 12 |
| `number_of_observations` | → `number_of_data` | 1,313 |

## 使用方式

重新生成 QC 数据后，按顺序执行：

```bash
# 1. 确保在 fix_global_attrs 分支
cd /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script
git checkout fix_global_attrs

# 2. 标准化全局属性
ssh node113 \
  /share/home/dq134/.conda/envs/wzx/bin/python3 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/apply_fix_attrs.py

# 3. 重命名 number_of_observations → number_of_data
ssh node113 \
  /share/home/dq134/.conda/envs/wzx/bin/python3 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/rename_number_observations.py

# 4. 跑下游 basin pipeline
cd /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/scripts_basin_test
python run_s1_s8_basin_pipeline.py --start-at s2 --end-at s8
```

## 为什么用 spawn 模式？

`fix_qc_global_attrs.py` 内部使用 `ProcessPoolExecutor`，在 node113 上因 GPFS 文件系统与 fork 的兼容性问题导致 worker 进程全部卡死。`apply_fix_attrs.py` 改用 `multiprocessing.Pool(spawn)` 绕过此问题，功能完全相同。
