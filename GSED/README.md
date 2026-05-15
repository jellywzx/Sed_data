# GSED 源数据流域信息输出说明

本文档说明当前 GSED 处理流程如何保留源数据中的 reach / basin 层级信息，以及为什么输出中的 `upstream_area` 不再来自 MERIT Hydro 匹配。

## 1. 当前结论

当前 `1_process_gsed_cf18.py` 只使用 GSED 公开源数据提供的信息：

```text
GSED_Reach_Monthly_SSC.csv
    -> R_ID 和月度 SSC 时间序列

GSED_Reach.dbf / GSED_Reach.shp
    -> R_ID, R_level, Length, reach geometry
    -> 由 geometry 计算 reach midpoint lat/lon
```

公开 GSED 属性表不包含真实的 upstream drainage area 字段。`GSED_Reach.dbf` 当前只有：

```text
OBJECTID
R_ID
R_level
Length
```

因此当前主流程不再读取 `GSED_Reach_upstream_area.csv`，也不再使用 MERIT Hydro / basin tracing 结果回填 `upstream_area`。

## 2. R_ID 派生的层级字段

GSED 源数据没有单独提供 basin name 或 catchment name。为了让下游流程仍能保留 GSED 的层级编码，代码从公开 `R_ID` 中派生以下字段：

```python
basin_code_l1 = R_ID[:2]
basin_code_l2 = R_ID[:5]
basin_code_l3 = R_ID[:8]
basin_code_l4 = R_ID
```

写入 NetCDF 时继续使用兼容字段名：

```text
reach_code = basin_code_l4
vpu_id     = basin_code_l1
rpu_id     = basin_code_l3
```

示例：

```text
R_ID = 10101010300

basin_code_l1 = 10
basin_code_l2 = 10101
basin_code_l3 = 10101010
basin_code_l4 = 10101010300
reach_code    = 10101010300
vpu_id        = 10
rpu_id        = 10101010
```

这些字段的唯一来源是 GSED 的 `R_ID`，不是 MERIT 或其他外部匹配结果。

## 3. upstream_area 输出规则

当前输出仍保留 NetCDF 标量变量：

```text
upstream_area
```

但它统一写为缺失值：

```text
_FillValue = FILL_VALUE_FLOAT
value      = FILL_VALUE_FLOAT
```

原因是公开 GSED 源数据不提供流域面积；为了避免把外部匹配面积误认为源数据，主流程不会再从任何 lookup 表读取或合并面积。

summary CSV 中的兼容列也保留：

```text
upstream_area_km2
upstream_area_source
```

其中 `upstream_area_km2` 写为缺失，`upstream_area_source` 写为空字符串。

## 4. 历史 MERIT lookup 脚本

`2_build_gsed_merit_area_lookup.py` 和 `build_gsed_merit_area_lookup_flowchart.md` 保留为历史工具，用于说明过去如何通过 MERIT Hydro 构建 `R_ID -> upstream_area_km2` lookup。

当前 GSED 主处理流程已经停用这一路径：

```text
1_process_gsed_cf18.py
    -> 不查找 GSED_Reach_upstream_area.*
    -> 不读取 GSED_UPSTREAM_AREA_FILE
    -> 不合并 merit_lookup_accept / upstream_area_km2
    -> upstream_area 固定写为缺失值
```

## 5. 当前输出包含的源数据信息

每个 GSED NetCDF 文件保留：

```text
Source_ID / station_id / station_name / reach_id
reach_level
reach_length_m
lat / lon
basin_code_l1
basin_code_l2
basin_code_l3
basin_code_l4
reach_code
vpu_id
rpu_id
```

其中 `lat/lon` 为 reach 几何 midpoint，`reach_level` 和 `reach_length_m` 来自 `GSED_Reach.dbf`，层级字段来自 `R_ID` 前缀。
