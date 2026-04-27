# RiverSed 数据处理工作流程说明

## 1. 脚本作用

RiverSed 这条处理链目前包含两个核心脚本：

- `convert_to_netcdf.py`
- `fill_missing_coordinates.py`

它们共同的目标是把两套卫星反演悬浮泥沙数据统一整理成
“每个站点/河段一个 netCDF 文件”的格式，并尽可能为 RiverSed reach
补充合理的代表坐标：

- `Aquasat_TSS_v1.1.csv`
- `RiverSed_USA_V1.1.txt`

脚本会完成以下事情：

- 读取源数据
- 统一字段名和时间格式
- 将 RiverSed 观测与修改后的 NHDPlus 元数据表进行匹配
- 从本地 NHDPlus flowline 生成 RiverSed reach 的代表坐标
- 把同一天的多条观测聚合成日尺度 SSC
- 做基础 SSC 质量控制
- 输出每个站点/河段的 netCDF
- 生成站点汇总 CSV 和 QC 汇总 CSV

这个脚本输出的是一个“只有 SSC、没有实测流量 Q”的产品，因此：

- `SSC` 来自卫星反演的 TSS/SSC
- `Q` 在输出里全部写成缺测
- `SSL` 因为缺少 `Q` 无法计算，所以也全部写成缺测


## 2. 输入文件

脚本会自动解析 `Source/RiverSed` 目录，主要依赖下面四个文件：

- `Aquasat_TSS_v1.1.csv`
- `RiverSed_USA_V1.1.txt`
- `nhdplusv2_modified_v1.0.dbf`
- `nhdplusv2_modified_v1.0.shp`

其中最关键的是后两个 NHDPlus 文件：

- `nhdplusv2_modified_v1.0.dbf` 负责提供 `ID -> 河段属性` 的键值映射
- `nhdplusv2_modified_v1.0.shp` 负责提供 flowline 几何，用于生成 RiverSed
  reach 的代表坐标


## 3. 输出文件

脚本会输出：

- 每个站点/河段一个 `.nc` 文件
- `RiverSed_station_summary.csv`
- `RiverSed_qc_results_summary.csv`

注意：

- 当前代码里 `OUTPUT_NC_DIR` 被设置成了 `OUTPUT_QC_DIR`
- 也就是说 `.nc` 文件和两个 CSV 目前会写到同一个输出目录里


## 4. 总体工作流程

脚本主流程可以概括为：

1. 解析源数据和输出目录。
2. 读取 Aquasat 数据，只保留实际要用的列。
3. 读取 RiverSed 观测数据，只保留实际要用的列。
4. 从修改后的 NHDPlus DBF 中读取河段/流域元数据。
5. 从修改后的 NHDPlus flowline shapefile 中读取河段几何。
6. 对 RiverSed 观测表、DBF 元数据表和 flowline 表中的 `ID` 做统一规范化。
7. 检查每一个 RiverSed 观测 `ID` 是否都能在 DBF 里找到。
8. 基于 flowline 几何生成代表点坐标，并并回 RiverSed 元数据表。
9. 将 RiverSed 观测与扩展后的元数据按 `ID` 连接。
10. 按 `station_id` 建立站点索引。
11. 并行处理每一个站点/河段。
12. 将同一天的多条观测聚合成日均 SSC。
13. 对 SSC 做质量控制。
14. 写出单站点 netCDF。
15. 汇总站点级元数据和 QC 统计。
16. 写出 CSV 汇总文件。
17. 打印进度条和各阶段耗时。


## 5. Aquasat 分支在做什么

Aquasat 的处理相对直接，但现在也会顺手保留后续会用到的站点元数据：

- 读取 `SiteID`、`date`、`value`、`lat`、`long`、`elevation`
- 额外读取 `GNIS_NAME`、`COMID`、`REACHCODE`、`RPUID`、`VPUID`
- 把 `SiteID` 重命名成 `station_id`
- 把 `value` 重命名成 `tss`
- 把 `GNIS_NAME` 重命名成 `river_name`
- 解析时间
- 丢掉缺少 `station_id` 或 `tss` 的行

之后每个 `station_id` 会单独进入后续的日尺度聚合、QC 和 netCDF 写出。


## 6. RiverSed 分支在做什么

RiverSed 的处理比 Aquasat 多出一段“河段/流域匹配 + 坐标补全”：

- 只读取 `ID`、`date`、`time`、`tss`、`elevation`
- 读取 `nhdplusv2_modified_v1.0.dbf`
- 读取 `nhdplusv2_modified_v1.0.shp`
- 规范化两边的 `ID`
- 检查 RiverSed 里的每个 `ID` 是否都存在于 DBF
- 根据 flowline 几何生成每个 reach 的代表坐标
- 将 `date` 和 `time` 合成为一个完整时间戳
- 用 `ID` 把 RiverSed 观测与 DBF 元数据连接起来
- 生成 `station_id = "RiverSed_" + ID`

连接成功后，每条 RiverSed 观测就不再只是一个数值点，而是带上了：

- `river_name`
- `comid`
- `reach_code`
- `vpu_id`
- `rpu_id`
- `upstream_area`
- `lat`
- `long`
- `coordinate_source`
- `coordinate_method`
- `coordinate_confidence`


## 7. 流域匹配与坐标补全部分详解

这一部分是整个脚本里最关键、也最容易误解的部分。

### 7.1 这里做的不是新的运行时河网匹配

脚本**没有**在运行时做新的最近河段搜索，也没有拿原始点位经纬度去和河网
重新做空间匹配。

脚本当前做的是：

- 使用 RiverSed 观测表中的 `ID`
- 去匹配一个已经提前准备好的、修改过的 NHDPlus DBF 表
- 在 `ID` 已经确定的前提下，再读取对应 flowline 几何，生成这条 reach 的
  代表坐标

也就是说，这里的主逻辑仍然是**键值匹配**，不是**重新决定河段归属的几何匹配**。

更准确地说：

- 运行前，某个上游流程已经把 RiverSed 的 `ID` 和 NHDPlus 河段关系准备好了
- 本脚本只是在运行时把这个关系读出来，并严格地应用到输出产品上


### 7.2 DBF 文件的角色

`nhdplusv2_modified_v1.0.dbf` 是一张查找表，它把 RiverSed 的 `ID`
映射到水文/河网属性。

代码里通过 `RIVERSED_METADATA_FIELD_MAP` 把 DBF 字段映射为内部字段名：

- `ID -> ID`
- `COMID -> comid`
- `GNIS_NA -> river_name`
- `REACHCO -> reach_code`
- `VPUID -> vpu_id`
- `RPUID -> rpu_id`
- `TtDASKM -> upstream_area`

这些字段后面会被写到：

- netCDF 全局属性
- netCDF 的 `upstream_area`
- 站点汇总 CSV
- QC 汇总 CSV


### 7.3 flowline shapefile 的角色

`nhdplusv2_modified_v1.0.shp` 的作用不是重新找“属于哪条河”，而是在
`ID` 已经确定之后，为这条 reach 生成一个合理的代表坐标。

这个 shapefile 主要提供：

- `ID`
- `COMID`
- `GNIS_NA`
- `REACHCO`
- `VPUID`
- `RPUID`
- `geometry`

其中真正用来补坐标的是 `geometry`。

当前实现会：

1. 读取 flowline 几何
2. 先投影到 `EPSG:5070`
3. 沿 flowline 取 50% 线长位置
4. 再投回 `EPSG:4326`
5. 生成 `lat` / `long`

这样得到的不是“测站实测 GPS 点”，而是“这条 RiverSed reach 在河网上的代表位置”。


### 7.4 为什么一定要做 ID 规范化

RiverSed 观测表和 DBF 表里的 `ID`，虽然逻辑上表示的是同一个对象，
但实际存储时很容易出现格式不一致。

例如同一个 ID 可能被存成：

- `123`
- `123.0`
- `"123 "`
- 浮点数
- 字符串

如果直接拿这些原始值做连接，就可能出现“明明是同一个 ID，但因为文本格式不同而
匹配失败”的情况。

脚本中的 `_normalize_riversed_id()` 就是为了解决这个问题。它会：

- 把空值变成空字符串
- 去掉首尾空白
- 如果是整数形式的数字，就统一转换成整数样式的字符串
- 否则保留清洗后的文本

这一步非常关键，因为后面的匹配是否稳定，几乎完全取决于这里。


### 7.5 为什么要在 merge 前做完整性检查

脚本在真正合并之前，会先比较两边的 `ID` 集合：

- RiverSed 观测表中的全部 `ID`
- DBF 元数据中的全部 `ID`

如果 RiverSed 里有任何一个 `ID` 在 DBF 中找不到，脚本会直接报错停止。

这样设计的目的是避免“静默丢匹配”。

如果不这么做，就可能发生：

- 某些观测写进 netCDF 了，但没有正确的河段身份
- `comid` 缺失
- `vpu_id` / `rpu_id` 缺失
- `upstream_area` 缺失
- 最终输出看起来成功了，但实际上元数据是不完整的

所以这里采用的是“宁可提前失败，也不输出部分错误结果”的策略。


### 7.6 为什么 DBF 里重复的 ID 会被拒绝

DBF 读完之后，脚本还会检查 `ID` 是否唯一。

原因是这里的设计假设是：

- 一条 RiverSed 观测
- 只能对应一个唯一的河段元数据记录

代码里的 merge 是：

```python
df.merge(metadata_df, on="ID", how="left", validate="m:1")
```

这里的 `m:1` 表示：

- 左表可以有多条观测
- 右表对应同一个 `ID` 必须只有一条元数据

如果右表也有重复 `ID`，那么一条观测就可能匹配到多个河段记录，
这在水文意义上是不可接受的，所以脚本会直接报错。


### 7.7 这一步在水文意义上代表什么

从概念上看，RiverSed 分支做的是三件事：

1. RiverSed 观测表提供“卫星反演出来的 SSC/TSS 数值”
2. 修改后的 NHDPlus DBF 提供“这个 ID 对应哪一条河段、属于哪个分区、流域面积是多少”
3. 脚本用 `ID` 把这两部分绑定起来

因此，这里的“流域匹配”其实更准确地说是：

- `RiverSed ID -> 已预处理好的 NHDPlus 河段/流域元数据`

它不是实时空间分析，而是把已有的河网归属关系稳定地写进结果产品。


### 7.8 匹配成功后，字段在下游如何使用

匹配成功后，下面这些字段会进入后续产品：

- `river_name`
- `comid`
- `reach_code`
- `vpu_id`
- `rpu_id`
- `upstream_area`
- `latitude`
- `longitude`
- `coordinate_source`
- `coordinate_method`
- `coordinate_confidence`

这些字段会被用在：

- netCDF 全局属性
- `upstream_area` 变量
- 汇总 CSV
- `Geographic Coverage` 文本

其中 `Geographic Coverage` 是由下面两部分拼出来的：

- `VPUID=<vpu_id>`
- `RPUID=<rpu_id>`


### 7.9 这段逻辑的边界条件

当前实现有几个重要边界：

- 它依赖修改后的 DBF 已经正确表达了 RiverSed 与 NHDPlus 的对应关系
- 它使用 flowline 几何生成代表坐标，但不重新判定 reach 归属
- 它不使用原始经纬度重新匹配河段
- 它要求 `ID` 在元数据表中唯一且完备

如果未来想改成真正的运行时空间匹配，就需要完全不同的实现方式，比如：

- 基于河网几何做空间相交
- 基于点位坐标映射到最近河段
- 使用 shapefile / geopackage / geopandas 等空间数据流程

而不是当前这种表驱动的 `ID` join。


## 8. 日尺度聚合和 QC

每个站点/河段在写出 netCDF 前，会先做如下处理：

1. 时间戳向下取整到天
2. 同一天的多条观测取平均，生成一个日尺度 SSC
3. 执行 SSC 质量控制
4. 将 bad 和 missing 的 SSC 置为 `NaN`
5. 把最终 flag 和中间 flag 一起写进 netCDF

这里的 QC 逻辑分三层：

- QC1：物理合理性检查
  - 负值记为 bad
  - 缺测和填充值记为 missing
- QC2：log-IQR 异常值筛查
  - 如果样本足够，就把离群值记为 suspect
- QC3：SSC-Q 一致性检查
  - 当前这个脚本没有 `Q`，所以这一步基本记为 not_checked

最终 `SSC_flag` 的含义是：

- `0 = good`
- `2 = suspect`
- `3 = bad`
- `9 = missing`


## 9. netCDF 写出逻辑

每个站点/河段都会输出成一个文件：

- `RiverSed_<safe_station_id>.nc`

其中主要变量包括：

- `time`
- `latitude`
- `longitude`
- `altitude`
- `upstream_area`
- `Q`
- `Q_flag`
- `SSC`
- `SSC_flag`
- `SSL`
- `SSL_flag`
- 各个中间 SSC QC flag

这里要特别注意：

- RiverSed reach 在当前实现下会尽量写入代表坐标
- `Q` 整列都是缺测
- `SSL` 整列都是缺测
- 真正有观测值的是 `SSC`

另外，RiverSed reach 在坐标可用时还会带上这些全局属性：

- `coordinate_source`
- `coordinate_method`
- `coordinate_confidence`
- `coordinate_fill_date`


## 10. 并行处理和进度条

脚本不会对每个站点反复扫整张表，而是先构建：

- `station_id -> 行号列表`

这样做有两个好处：

- 避免重复布尔筛选整张 DataFrame
- 并行调度时只需要传站点键，不需要传整块大数据

在 Linux 的 `fork` 模式下，worker 进程会复用父进程里已经读好的
DataFrame，从而减少进程间传输开销。

运行时主进程会显示：

- 当前阶段进度条
- 已完成站点数
- 成功/失败数
- 已耗时
- 预计剩余时间

最后还会打印：

- 读取数据耗时
- 构建分组耗时
- Aquasat 处理耗时
- RiverSed 处理耗时
- CSV 写出耗时
- 总耗时


## 11. RiverSed 的站点筛选规则

RiverSed 在正式处理前会先过滤一次：

- 只有观测数不少于 5 条的 `station_id` 才会进入导出流程

这个阈值是在按 `station_id` 分组后应用的。


## 12. 汇总 CSV 是怎么来的

脚本在生成每个 netCDF 的同时，也会构建一个 `stations_info` 字典，
里面保存：

- 站点级元数据
- 时间跨度
- 地理覆盖信息
- 各类 QC flag 计数

最后再统一把 `stations_info` 写成：

- 站点汇总 CSV
- QC 结果汇总 CSV

所以这两个 CSV 不是额外重新扫原始数据得到的，而是跟 netCDF 生成过程
同步积累出来的。


## 13. 维护时最值得关注的地方

如果以后要修改这份脚本，最值得重点看的是：

- `RIVERSED_METADATA_FIELD_MAP`
- `RIVERSED_FLOWLINE_FIELD_MAP`
- `_normalize_riversed_id()`
- `_normalize_reach_code()`
- `load_riversed_flowline_reference()`
- `load_riversed_station_metadata()`
- `load_riversed_data()`
- `df.merge(..., validate="m:1")`

因为这几个位置共同决定了：

- RiverSed 和 NHDPlus 元数据能不能正确对上
- 是否会出现漏匹配
- 是否会出现重复匹配
- 下游 netCDF 和 CSV 里的流域信息是否可信
- RiverSed reach 的代表坐标是否仍然稳定、合理

如果这几处逻辑发生变化，建议一定重新检查：

- `ID` 是否仍然唯一
- 所有 `ID` 是否仍能覆盖
- `comid / reach_code / vpu_id / rpu_id / upstream_area` 是否仍然正确进入结果
- `latitude / longitude / coordinate_*` 是否仍然正确进入结果


## 14. 现有 QC netCDF 的坐标补全流程

上面 1 到 13 节主要解释的是：

- 如何从源表重新生成 RiverSed / Aquasat netCDF

但在实际工作中，经常还会遇到另一类需求：

- 现有 `Output_r/daily/RiverSed/qc` 里已经有一批 netCDF
- 其中一部分 RiverSed reach 文件没有真实经纬度
- 希望尽可能不重跑全流程，直接对现有 qc 输出做坐标补全

这时使用的是：

- `fill_missing_coordinates.py`


## 15. `fill_missing_coordinates.py` 在做什么

这个脚本是一个后处理工具，专门用于：

- 扫描现有 `RiverSed_*.nc`
- 识别哪些 RiverSed reach 文件仍然缺坐标
- 构建可追踪的参考表和候选表
- 只把高置信度坐标写回 nc

它的输入不是原始 CSV，而是现有的 qc netCDF。

### 15.1 inventory 表怎么来的

脚本会遍历：

- `Output_r/daily/RiverSed/qc/RiverSed_*.nc`

并从每个文件中提取：

- `station_id`
- `station_location`
- `river_name`
- `comid`
- `reachcode`
- `vpu_id`
- `rpu_id`
- `latitude`
- `longitude`

其中：

- `id` 从 `station_id` 里的 `RiverSed_<ID>` 解析
- `comid` / `reachcode` 优先读全局属性
- 如果全局属性里没有，就从 `station_location` 文本里解析

脚本还会专门处理 netCDF 中的 `_FillValue` 和 `missing_value`，把 masked
坐标正确识别成缺测，而不是误判成 `0.0`。

### 15.2 reference 表怎么来的

reference 表来自：

- `nhdplusv2_modified_v1.0.shp`

它会先走和主流程一致的代表点生成逻辑：

1. 读取 flowline
2. 规范化 `ID`、`COMID`、`REACHCO`
3. 投影到 `EPSG:5070`
4. 沿 flowline 取 50% 线长位置
5. 投回 `EPSG:4326`
6. 生成 `rep_lat` / `rep_lon`

然后保存成标准化 reference 表，字段包括：

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

### 15.3 候选坐标是如何生成的

候选生成顺序固定为：

1. `ID`
2. `COMID`
3. `REACHCO + basin constraint`
4. `river_name + RPU/VPU`

并且只有前一层失败时才会进入下一层。

#### 第一层：按 `ID`

这是最强匹配。

- 如果 nc 能解析出 `ID`
- 且 reference 里存在相同 `ID`
- 就直接命中

输出：

- `match_method = flowline_midpoint_by_id`
- `confidence = high`

#### 第二层：按 `COMID`

如果 `ID` 没命中，再试 `COMID`。

输出：

- `match_method = flowline_midpoint_by_comid`
- `confidence = high`

#### 第三层：按 `REACHCO`

只有 `ID` 和 `COMID` 都失败时才会使用。

这里有两个关键点：

- `REACHCO` 必须作为字符串处理
- 必须保留前导零

同时还必须加约束：

- 优先 `rpu_id`
- 否则 `vpu_id`
- 若 `river_name` 存在，则要求标准化后名称一致

结果：

- 唯一命中：`confidence = medium`
- 多候选：进入人工复核

#### 第四层：按 `river_name`

这是最低优先级的自动候选来源。

脚本会先标准化河名：

- 全部转小写
- 去掉多余空格
- 统一常见缩写

然后按：

- `rpu_id`
- 或 `vpu_id`

约束候选集合。

结果：

- 唯一命中：`confidence = low`
- 多候选：进入人工复核


## 16. 如何避免补错坐标

当前坐标补全不追求“能补多少补多少”，而是优先保证可靠性。

### 16.1 坐标范围检查

候选坐标必须落在美国本土合理范围内：

- `lon` 在 `[-125, -66]`
- `lat` 在 `[24, 50]`

超出范围就不自动写回。

### 16.2 河名一致性检查

如果源 nc 已有 `river_name`，而候选 reference 的 `river_name` 标准化后与之
不一致，脚本会打上：

- `review_flag = True`
- `review_reason = "river_name_mismatch"`

### 16.3 自动写回阈值

当前规则是：

- 只有 `confidence = high`
- 且 `review_flag = False`
- 且 `new_lat / new_lon` 非空

才允许自动写回。

所以：

- `medium`
- `low`
- `unresolved`

默认都不会直接改 nc。


## 17. 后处理脚本会产出什么

运行 `fill_missing_coordinates.py` 后，会在 qc 输出目录生成：

### 17.1 Inventory

- `riversed_coord_fill_inventory.csv`

记录所有相关 netCDF 的解析结果。

### 17.2 Reference

- `riversed_coord_fill_reference_flowline.csv`

记录本地 flowline 参考点表。

### 17.3 Candidates

- `riversed_coord_fill_candidates.csv`

记录每个待补文件的候选坐标、匹配方式和置信度。

关键字段包括：

- `new_lat`
- `new_lon`
- `match_key`
- `match_method`
- `source_dataset`
- `confidence`
- `review_flag`
- `review_reason`

### 17.4 Manual Review

- `riversed_coord_fill_manual_review.csv`

收集低置信度、冲突、未命中或异常候选。


## 18. 如何执行

### 18.1 只生成表，不改 nc

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/fill_missing_coordinates.py
```

适合：

- 想先看候选结果
- 想先抽样检查

### 18.2 只写回高置信度结果

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/fill_missing_coordinates.py \
  --apply
```

适合：

- 已确认当前主路径稳定
- 只想自动写回 `high` 置信度结果

### 18.3 整套重跑 RiverSed

```bash
/share/home/dq134/.conda/envs/wzx/bin/python3.9 \
  /share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/convert_to_netcdf.py
```

适合：

- 希望新生成的 RiverSed reach 文件在生成阶段就带上代表坐标
- 希望 `coordinate_*` 元数据直接进入最终输出


## 19. 一个具体例子

以：

- `Output_r/daily/RiverSed/qc/RiverSed_RiverSed_1.nc`

为例，这个文件原本可能只有：

- `station_id = RiverSed_1`
- `river_name = Allagash River`
- `station_location = Allagash River (COMID 717072, reach 01010002000001)`
- `latitude / longitude = missing`

补全逻辑是：

1. 从 `station_id` 解析出 `ID = 1`
2. 到 flowline reference 中查 `ID = 1`
3. 找到对应的 NHDPlus flowline
4. 取这条 flowline 的 50% 线长位置
5. 得到代表坐标
6. 标记：
   - `coordinate_method = flowline_midpoint_by_id`
   - `coordinate_confidence = high`

这说明补上的坐标不是“实测站点 GPS”，而是：

- 这条 RiverSed reach 在河网上的代表位置


## 20. 最后怎么理解这套坐标

这套流程补出来的经纬度，本质上表示的是：

- `RiverSed ID -> 对应 NHDPlus flowline -> 这条河段的代表点`

所以它的语义是：

- 适合表示 reach 级别位置
- 适合用于地理索引、绘图、统计和元数据完整性

但它不等同于：

- 人工采样站 GPS 点
- 传感器安装点
- 河段起点或终点

如果把这个区别记住，整套 RiverSed 坐标补全逻辑就会非常清楚。
