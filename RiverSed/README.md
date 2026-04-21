# RiverSed `convert_to_netcdf.py` 工作流程说明

## 1. 脚本作用

`convert_to_netcdf.py` 的目标是把两套卫星反演悬浮泥沙数据统一整理成
“每个站点/河段一个 netCDF 文件”的格式：

- `Aquasat_TSS_v1.1.csv`
- `RiverSed_USA_V1.1.txt`

脚本会完成以下事情：

- 读取源数据
- 统一字段名和时间格式
- 将 RiverSed 观测与修改后的 NHDPlus 元数据表进行匹配
- 把同一天的多条观测聚合成日尺度 SSC
- 做基础 SSC 质量控制
- 输出每个站点/河段的 netCDF
- 生成站点汇总 CSV 和 QC 汇总 CSV

这个脚本输出的是一个“只有 SSC、没有实测流量 Q”的产品，因此：

- `SSC` 来自卫星反演的 TSS/SSC
- `Q` 在输出里全部写成缺测
- `SSL` 因为缺少 `Q` 无法计算，所以也全部写成缺测


## 2. 输入文件

脚本会自动解析 `Source/RiverSed` 目录，主要依赖下面三个文件：

- `Aquasat_TSS_v1.1.csv`
- `RiverSed_USA_V1.1.txt`
- `nhdplusv2_modified_v1.0.dbf`

其中最关键的是第三个 DBF 文件。它不是普通附加信息，而是 RiverSed
这条处理链里“河段/流域身份信息”的核心映射表。


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
5. 对 RiverSed 观测表和 DBF 元数据表中的 `ID` 做统一规范化。
6. 检查每一个 RiverSed 观测 `ID` 是否都能在 DBF 里找到。
7. 将 RiverSed 观测与 DBF 元数据按 `ID` 连接。
8. 按 `station_id` 建立站点索引。
9. 并行处理每一个站点/河段。
10. 将同一天的多条观测聚合成日均 SSC。
11. 对 SSC 做质量控制。
12. 写出单站点 netCDF。
13. 汇总站点级元数据和 QC 统计。
14. 写出 CSV 汇总文件。
15. 打印进度条和各阶段耗时。


## 5. Aquasat 分支在做什么

Aquasat 的处理相对直接：

- 只读取 `SiteID`、`date`、`value`、`lat`、`long`、`elevation`
- 把 `SiteID` 重命名成 `station_id`
- 把 `value` 重命名成 `tss`
- 解析时间
- 丢掉缺少 `station_id` 或 `tss` 的行

之后每个 `station_id` 会单独进入后续的日尺度聚合、QC 和 netCDF 写出。


## 6. RiverSed 分支在做什么

RiverSed 的处理比 Aquasat 多出一段“河段/流域匹配”：

- 只读取 `ID`、`date`、`time`、`tss`、`elevation`
- 读取 `nhdplusv2_modified_v1.0.dbf`
- 规范化两边的 `ID`
- 检查 RiverSed 里的每个 `ID` 是否都存在于 DBF
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


## 7. 流域匹配部分详解

这一部分是整个脚本里最关键、也最容易误解的部分。

### 7.1 这里做的不是运行时空间匹配

脚本**没有**在运行时做新的 GIS 空间相交，也没有拿经纬度去和河网重新做
空间匹配。

脚本当前做的是：

- 使用 RiverSed 观测表中的 `ID`
- 去匹配一个已经提前准备好的、修改过的 NHDPlus DBF 表

也就是说，这里是**键值匹配**，不是**几何匹配**。

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


### 7.3 为什么一定要做 ID 规范化

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


### 7.4 为什么要在 merge 前做完整性检查

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


### 7.5 为什么 DBF 里重复的 ID 会被拒绝

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


### 7.6 这一步在水文意义上代表什么

从概念上看，RiverSed 分支做的是三件事：

1. RiverSed 观测表提供“卫星反演出来的 SSC/TSS 数值”
2. 修改后的 NHDPlus DBF 提供“这个 ID 对应哪一条河段、属于哪个分区、流域面积是多少”
3. 脚本用 `ID` 把这两部分绑定起来

因此，这里的“流域匹配”其实更准确地说是：

- `RiverSed ID -> 已预处理好的 NHDPlus 河段/流域元数据`

它不是实时空间分析，而是把已有的河网归属关系稳定地写进结果产品。


### 7.7 匹配成功后，字段在下游如何使用

匹配成功后，下面这些字段会进入后续产品：

- `river_name`
- `comid`
- `reach_code`
- `vpu_id`
- `rpu_id`
- `upstream_area`

这些字段会被用在：

- netCDF 全局属性
- `upstream_area` 变量
- 汇总 CSV
- `Geographic Coverage` 文本

其中 `Geographic Coverage` 是由下面两部分拼出来的：

- `VPUID=<vpu_id>`
- `RPUID=<rpu_id>`


### 7.8 这段逻辑的边界条件

当前实现有几个重要边界：

- 它依赖修改后的 DBF 已经正确表达了 RiverSed 与 NHDPlus 的对应关系
- 它不检查几何对象本身
- 它不使用经纬度重新匹配河段
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

- `Q` 整列都是缺测
- `SSL` 整列都是缺测
- 真正有观测值的是 `SSC`


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
- `_normalize_riversed_id()`
- `load_riversed_station_metadata()`
- `load_riversed_data()`
- `df.merge(..., validate="m:1")`

因为这几个位置共同决定了：

- RiverSed 和 NHDPlus 元数据能不能正确对上
- 是否会出现漏匹配
- 是否会出现重复匹配
- 下游 netCDF 和 CSV 里的流域信息是否可信

如果这几处逻辑发生变化，建议一定重新检查：

- `ID` 是否仍然唯一
- 所有 `ID` 是否仍能覆盖
- `comid / reach_code / vpu_id / rpu_id / upstream_area` 是否仍然正确进入结果
