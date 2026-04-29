# RiverSed 坐标补全工作流程

## 1. 目标

这份文档说明 RiverSed 数据集中缺失经纬度的河段文件，如何利用本地
NHDPlus flowline 数据进行补全，以及对应的脚本、匹配规则、质控逻辑和
产出文件。

当前关注的对象主要是这类文件：

- `RiverSed_RiverSed_*.nc`

它们通常具有以下特征：

- `river_name` 存在
- `station_location` 中包含 `COMID` 和 `reach ID`
- `latitude` / `longitude` 为缺测值（原文件中常表现为 `_FillValue=-9999`）

而像下面这类 Aquasat 文件通常已经有真实经纬度：

- `RiverSed_0801478-CG-2.nc`


## 2. 相关脚本

当前这条处理链由两个脚本组成：

### 2.1 `convert_to_netcdf.py`

路径：

- `Script/RiverSed/convert_to_netcdf.py`

作用：

- 负责从源数据重新生成 RiverSed / Aquasat 的 QC netCDF
- 在 RiverSed 分支中，现已接入本地 flowline 几何代表点计算
- 如果整套数据重跑，这个脚本会直接把 RiverSed reach 的坐标写进新生成的 nc

### 2.2 `fill_missing_coordinates.py`

路径：

- `Script/RiverSed/fill_missing_coordinates.py`

作用：

- 面向已经存在的 `Output_r/daily/RiverSed/qc/*.nc`
- 扫描缺失坐标的 RiverSed reach 文件
- 构建 inventory / reference / candidates / manual review 表
- 默认只输出 CSV，不修改 nc
- 传入 `--apply` 时，只把高置信度结果写回现有 nc


## 3. 数据来源

坐标补全当前采用 **offline-first** 策略，优先使用本地数据源。

### 3.1 主数据源：`nhdplusv2_modified_v1.0.shp`

路径：

- `Source/RiverSed/nhdplusv2_modified_v1.0.shp`

用途：

- 提供 RiverSed reach 对应的 flowline 几何
- 用于生成“河段代表坐标”

使用字段：

- `ID`
- `COMID`
- `GNIS_NA`
- `REACHCO`
- `VPUID`
- `RPUID`
- `geometry`

### 3.2 元数据查找表：`nhdplusv2_modified_v1.0.dbf`

路径：

- `Source/RiverSed/nhdplusv2_modified_v1.0.dbf`

用途：

- 提供 RiverSed 观测记录与河段属性之间的 `ID` 键值映射
- 为 `convert_to_netcdf.py` 中的 RiverSed 分支提供元数据连接基础

### 3.3 不作为主数据源的文件

#### `nhdplusv2_polygons_v1.0.shp`

用途：

- 仅适合做 catchment 辅助检查
- 不适合直接作为河段“站点坐标”来源

原因：

- polygon 代表点不一定落在河道上
- 对当前 RiverSed reach 语义，flowline 中点更合理

#### `COMID_ID.csv`

用途：

- 只能作为辅助排查
- 不作为最终几何来源

原因：

- 一个 `ID` 可能对应多个 `COMID`
- 无法单独稳定生成唯一坐标


## 4. 核心思路

当前 RiverSed 缺失坐标文件的补全主路径是：

1. 从 nc 中解析出 `ID`
2. 用 `ID` 连接本地 flowline
3. 若 `ID` 缺失或未命中，再尝试 `COMID`
4. 从命中的 flowline 计算代表点
5. 对坐标做美国本土范围质控
6. 将结果标记为高置信度，并写入 nc 或输出候选表

这条路径的核心特点是：

- 不做模糊地名匹配作为主流程
- 不做新的空间最近邻匹配
- 依赖既有 RiverSed `ID -> NHDPlus flowline` 关系
- 目标是稳定、可重复、可追溯


## 5. 详细工作流程

### 5.1 在 `convert_to_netcdf.py` 中的流程

当整套 RiverSed 数据重跑时，坐标补全流程已经内置在转换脚本中。

### 步骤 1：读取源观测数据

- 读取 `RiverSed_USA_V1.1.txt`
- 保留：
  - `ID`
  - `date`
  - `time`
  - `tss`
  - `elevation`

### 步骤 2：读取 DBF 元数据

- 从 `nhdplusv2_modified_v1.0.dbf` 读取：
  - `ID`
  - `COMID`
  - `GNIS_NA`
  - `REACHCO`
  - `VPUID`
  - `RPUID`
  - `TtDASKM`

### 步骤 3：读取 flowline 几何并生成代表点

- 读取 `nhdplusv2_modified_v1.0.shp`
- 保留：
  - `ID`
  - `COMID`
  - `GNIS_NA`
  - `REACHCO`
  - `VPUID`
  - `RPUID`
  - `geometry`

处理逻辑：

1. 先把几何投影到 `EPSG:5070`
2. 沿 flowline 取 50% 线长位置的点
3. 再转回 `EPSG:4326`
4. 输出：
   - `lat`
   - `long`

补充属性：

- `coordinate_source = "nhdplusv2_modified_v1.0.shp"`
- `coordinate_method = "flowline_midpoint_by_id"`
- `coordinate_confidence = "high"`

### 步骤 4：把代表点并回 RiverSed 元数据表

- 通过 `ID` 将 flowline 代表点合并回 DBF 元数据表
- 这样 RiverSed 的 metadata 除了河段属性，还会带：
  - `lat`
  - `long`
  - `coordinate_source`
  - `coordinate_method`
  - `coordinate_confidence`

### 步骤 5：把元数据合并到 RiverSed 观测表

- 用 `ID` 将 RiverSed 观测与元数据表做 `m:1` 合并
- 合并完成后，每一条 RiverSed 观测记录都会带上：
  - `river_name`
  - `comid`
  - `reach_code`
  - `vpu_id`
  - `rpu_id`
  - `upstream_area`
  - `lat`
  - `long`
  - `coordinate_*`

### 步骤 6：按站点输出 netCDF

在写出单个 netCDF 时：

- `latitude` / `longitude` 直接写入 flowline 代表点坐标
- 新增全局属性：
  - `coordinate_source`
  - `coordinate_method`
  - `coordinate_confidence`
  - `coordinate_fill_date`


### 5.2 在 `fill_missing_coordinates.py` 中的流程

这个脚本用于处理已经存在的 QC netCDF。

### 步骤 1：扫描现有 netCDF，生成 inventory

遍历：

- `Output_r/daily/RiverSed/qc/RiverSed_*.nc`

提取字段：

- `file`
- `path`
- `dataset_branch`
- `station_id`
- `id`
- `comid`
- `reachcode`
- `river_name`
- `normalized_river_name`
- `station_location`
- `vpu_id`
- `rpu_id`
- `orig_lat`
- `orig_lon`
- `needs_fill`

其中：

- `id` 来自 `station_id` 中的 `RiverSed_<ID>`
- `comid` 和 `reachcode` 优先读全局属性；若缺失，则从 `station_location` 解析
- `needs_fill=True` 表示当前文件是 RiverSed reach 且经纬度缺失

### 步骤 2：构建 reference flowline 表

从 `nhdplusv2_modified_v1.0.shp` 生成一张标准化 reference 表，字段包括：

- `ID`
- `comid`
- `reach_code`
- `river_name`
- `normalized_river_name`
- `vpu_id`
- `rpu_id`
- `rep_lat`
- `rep_lon`
- `coordinate_source`
- `coordinate_method`
- `coordinate_confidence`

### 步骤 3：按优先级生成候选坐标

候选生成顺序：

1. `ID`
2. `COMID`
3. `REACHCO + basin constraint`
4. `river_name + RPU/VPU`

只有前一层未命中时，才进入下一层。

#### Pass 1：按 `ID`

- 如果 nc 能解析出 `id`
- 且 reference 中存在相同 `ID`
- 则直接命中

输出：

- `match_key = "ID=<id>"`
- `match_method = "flowline_midpoint_by_id"`
- `confidence = "high"`

#### Pass 2：按 `COMID`

- 若 `ID` 未命中，再尝试 `comid`

输出：

- `match_key = "COMID=<comid>"`
- `match_method = "flowline_midpoint_by_comid"`
- `confidence = "high"`

#### Pass 3：按 `REACHCO`

仅在 `ID` 和 `COMID` 都无效时使用。

规则：

- `reachcode` 必须按字符串处理
- 保留前导零
- 必须额外加约束：
  - 优先 `rpu_id`
  - 否则 `vpu_id`
  - 若 `river_name` 存在，则再要求标准化后名称一致

结果：

- 唯一命中时，记为 `medium`
- 多候选时，进入人工复核

#### Pass 4：按 `river_name`

仅在前几步都失败时使用。

规则：

- 先对河名标准化：
  - lowercase
  - 去空白
  - 压缩空格
  - 常见缩写统一
- 再按：
  - `rpu_id`
  - 或 `vpu_id`
  约束候选集

结果：

- 唯一命中：`low`
- 多候选：进入人工复核


## 6. 质控规则

### 6.1 坐标范围检查

自动候选必须落在美国本土合理范围内：

- `lon` 在 `[-125, -66]`
- `lat` 在 `[24, 50]`

如果超出范围：

- 不自动写回
- 标记 `review_flag=True`
- `review_reason="candidate_outside_conus"`

### 6.2 河名一致性检查

若源 nc 已有 `river_name`，且与 reference 中 `river_name` 标准化后不一致：

- 标记 `review_flag=True`
- `review_reason="river_name_mismatch"`

### 6.3 自动写回阈值

当前规则：

- 仅 `confidence="high"` 且 `review_flag=False` 的记录允许自动写回
- `medium` 和 `low` 一律保留在候选表或人工复核表中


## 7. 产出文件

运行 `fill_missing_coordinates.py` 后，会在 QC 输出目录生成以下文件：

### 7.1 Inventory

- `riversed_coord_fill_inventory.csv`

作用：

- 记录每一个 RiverSed/Aquasat netCDF 的解析结果

### 7.2 Reference

- `riversed_coord_fill_reference_flowline.csv`

作用：

- 保存本地 flowline 的标准化代表点参考表

### 7.3 Candidates

- `riversed_coord_fill_candidates.csv`

作用：

- 对每个待补坐标文件给出一个候选结果

关键字段：

- `new_lat`
- `new_lon`
- `match_key`
- `match_method`
- `source_dataset`
- `confidence`
- `review_flag`
- `review_reason`

### 7.4 Manual Review

- `riversed_coord_fill_manual_review.csv`

作用：

- 收集未命中、低置信度、冲突或异常候选


## 8. 写回规则

当 `fill_missing_coordinates.py` 使用 `--apply` 时：

- 只处理 `confidence="high"` 且 `review_flag=False` 的候选
- 更新 nc 中的：
  - `latitude`
  - `longitude`
- 同时新增或更新全局属性：
  - `coordinate_source`
  - `coordinate_method`
  - `coordinate_confidence`
  - `coordinate_fill_date`

默认不加 `--apply` 时：

- 只输出 CSV
- 不修改任何 nc 文件


## 9. 推荐执行方式

### 9.1 只生成清单和候选表

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/fill_missing_coordinates.py
```

适用场景：

- 想先看 inventory / candidates / manual review
- 不想直接改 nc

### 9.2 直接写回高置信度坐标

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/fill_missing_coordinates.py \
  --apply
```

适用场景：

- 已确认当前主路径可靠
- 只希望自动写回 `high` 置信度结果

### 9.3 整套重跑 RiverSed 生成新 nc

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/convert_to_netcdf.py
```

适用场景：

- 希望在生成阶段就把坐标补进去
- 希望新生成的 RiverSed reach 文件天然带上 `coordinate_*` 元数据


## 10. 示例

样例文件：

- `Output_r/daily/RiverSed/qc/RiverSed_RiverSed_1.nc`

现有标识：

- `station_id = RiverSed_1`
- `river_name = Allagash River`
- `station_location = Allagash River (COMID 717072, reach 01010002000001)`

补全逻辑：

1. 从 `station_id` 解析出 `ID = 1`
2. 在 flowline reference 中命中 `ID = 1`
3. 生成 flowline 中点坐标
4. 输出：
   - `match_method = flowline_midpoint_by_id`
   - `confidence = high`


## 11. 已知边界与注意事项

### 11.1 `REACHCO` 不能直接转数值

例如：

- 正确：`01010002000001`
- 错误：`1010002000001`

如果把 `REACHCO` 当数值处理，会丢失前导零，导致匹配失败。

### 11.2 `river_name` 不是主键

不能把河名模糊匹配当主路径，因为：

- 同名河流很多
- 同一个名字可能跨不同 basin
- 只靠名称容易误匹配

### 11.3 polygon 不作为主写回坐标

`nhdplusv2_polygons_v1.0.shp` 可以用于辅助手工核查，但不作为主坐标来源。

### 11.4 Aquasat 文件不应被覆盖

Aquasat 站点通常已经有真实坐标，坐标补全脚本只应该处理：

- `dataset_branch == "riversed_reach"`
- 且 `needs_fill == True`


## 12. 总结

当前 RiverSed 的坐标补全已经形成两条稳定路径：

1. **生成阶段补全**  
   通过 `convert_to_netcdf.py` 在重新生成 nc 时直接写入 reach 代表点坐标。

2. **后处理阶段补全**  
   通过 `fill_missing_coordinates.py` 对现有 qc netCDF 做 inventory、候选生成和高置信度写回。

当前主方法是：

- 本地 `ID` / `COMID` 直连 `nhdplusv2_modified_v1.0.shp`
- 使用投影后 flowline 50% 线长位置作为代表点
- 只自动写回高置信度结果

这套流程的优点是：

- 不依赖外部网络
- 结果可重复
- 匹配逻辑清晰
- 坐标来源可追溯
- 适合批量处理当前 RiverSed reach 文件
