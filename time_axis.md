# `sed_data` 数据集处理与时间轴规则说明

本文档根据 `Sed_data` 与 `sed_data_integration` 当前连接仓库中的处理脚本整理，重点说明各数据集在进入标准化 NetCDF 产品前后的处理流程、时间轴定义、时区/地方时处理现状、质量控制与变量统一规则。

> 结论先行：当前代码体系总体将时间轴标准化为 CF/ACDD 风格的 `time` 坐标变量，通常使用 `float64` 数值、`gregorian` 日历，以及 `days since 1970-01-01 00:00:00` 作为统一时间单位。对于 climatology/多年平均数据，`time` 不是实际观测时刻，而是观测期中间年的 7 月 1 日。代码中没有明确声明某个数据集的原始时间为地方时；但 Fukushima 与 RiverSed 含有小时/分钟或 date+time 信息，当前被当作 naive datetime 处理，没有显式时区转换。

---

## 1. 总体标准化目标

### 1.1 标准 NetCDF 时间坐标

大多数标准化脚本将时间坐标写为：

```python
 time = f8(time)
 time.standard_name = "time"
 time.long_name = "time" 或 "representative time of climatological mean"
 time.units = "days since 1970-01-01 00:00:00"
 time.calendar = "gregorian"
 time.axis = "T"
```

该规则在 HYDAT、HYBAM、GloRiSe、Milliman、ALi_De_Boer、Vanmaercke、Yajiang 等处理脚本中反复出现。少数脚本会使用源起始日期作为单位基准，例如 Fukushima 当前输出使用 `days since {t0} 00:00:00`，但仍保留 `calendar='gregorian'` 和 `axis='T'`。

### 1.2 坐标变量和数据变量

标准化后的文件通常包含：

- `time`：一维时间坐标。
- `lat`/`lon` 或 `latitude`/`longitude`：站点标量坐标，单位为 `degrees_north`、`degrees_east`。
- `altitude`：若可用则写入站点高程；不可用时填充 `-9999.0`。
- `upstream_area`：若可用则写入流域面积。
- `Q` 或 `discharge`：河流流量，标准单位通常为 `m3 s-1`。
- `SSC` 或 `ssc`：悬浮泥沙浓度，标准单位通常为 `mg L-1`。
- `SSL` 或 `sediment_load`：悬浮泥沙通量/负荷，标准单位通常为 `ton day-1`。
- 对应质量标记变量：`Q_flag`、`SSC_flag`、`SSL_flag`，以及部分数据集中的分步 QC 标记。

### 1.3 质量标记约定

多数脚本采用以下质量码：

| flag | 含义 |
|---:|---|
| 0 | good_data，通过 QC |
| 1 | estimated_data，估算或插补 |
| 2 | suspect_data，可疑值或统计异常 |
| 3 | bad_data，物理不可能值，例如负值 |
| 9 | missing_data，源数据缺测 |

部分分步 QC 使用 `8` 表示 `not_checked`，尤其是在样本量不足或变量缺失导致无法进行某一步检查时。

---

## 2. 时间轴处理总规则

### 2.1 普通时间序列

普通逐日、逐月或逐年观测序列通常执行以下步骤：

1. 读取源时间字段或源 NetCDF 时间变量。
2. 如果源时间单位不同，先用 `netCDF4.num2date` 转为 datetime。
3. 再用 `date2num(..., units="days since 1970-01-01 00:00:00", calendar="gregorian")` 转为统一数值。
4. 写入 `time` 变量。
5. 根据有效 Q/SSC/SSL 或有效 SSC 记录确定时间覆盖范围。

### 2.2 多变量时间轴合并

代码里存在两种主要策略：

| 策略 | 适用场景 | 说明 |
|---|---|---|
| 时间点并集 | HYDAT sediment + discharge 合并 | 将 sediment load、SSC、flow 的所有时间点合并为唯一排序时间轴，缺测填充。 |
| 共同覆盖区间 + 映射 | HYBAM discharge + SSC 合并 | 取 discharge 与 SSC 的重叠时间范围，以 discharge 时间为基准，将 SSC 映射到最近 1 天内的 discharge 时间点。 |

### 2.3 climatology/多年平均时间轴

对于多年平均或 climatology 数据，`time` 坐标不是瞬时观测时间，而是代表性时间戳：

```python
mid_year = (start_year + end_year) // 2
representative_time = datetime(mid_year, 7, 1)
```

也就是说，若源记录期为 `1957-2017`，代表时间为 `1987-07-01`。

这种规则主要用于：

- ALi_De_Boer
- Milliman
- GloRiSe annually_climatology
- Vanmaercke
- 其他长期平均型数据集

---

## 3. 地方时 / 时区处理现状

当前代码中没有发现明确写入或声明如下信息：

- `source_time_zone = ...`
- `timezone = ...`
- `tz_localize(...)`
- `tz_convert(...)`
- `UTC` 到地方时或地方时到 `UTC` 的显式转换

因此，除 HYBAM 这种明确使用 Unix 秒并用 UTC 解释的情况外，含日期或小时信息的数据基本按 **naive datetime** 处理。

### 3.1 明确更像 UTC 的数据集：HYBAM

HYBAM 源 NetCDF 中时间变量名为 `Date`，代码注释将其视为 Unix seconds，并使用 `datetime.utcfromtimestamp(...)` 计算时间覆盖范围。因此 HYBAM 时间更接近 UTC/Unix 秒语义。

### 3.2 最需要注意的潜在地方时数据集：Fukushima

Fukushima DOI00147 源表包含 `yyyy, mm, dd, hh, min` 字段，代码直接用这些字段构造 `pd.to_datetime`。随后进行日平均聚合：

```python
data['datetime'] = pd.to_datetime({
    'year': data[3].astype(int),
    'month': data[4].astype(int),
    'day': data[5].astype(int),
    'hour': data[6].astype(int),
    'minute': data[7].astype(int)
})

daily = df.set_index('datetime').resample('D').mean()
```

这意味着：

- 如果原始 DOI00147 时间实际为日本当地时间，当前代码没有显式记录这一点。
- 日平均边界按 naive datetime 的自然日处理。
- 输出时还会将时间 `floor('D')` 到日尺度，因此小时分钟信息不会保留在最终时间轴中。

### 3.3 RiverSed / Aquasat

RiverSed 源数据包含分开的 `date` 和 `time` 字段，代码将其拼接为：

```python
df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
```

随后会按日聚合：

```python
tss_df['date'] = tss_df['date'].dt.floor('D')
tss_daily = tss_df.groupby('date', as_index=False)['tss'].mean()
```

因此 RiverSed 也没有显式时区转换。若源 `time` 字段为当地时间，当前代码没有记录或转换。

### 3.4 建议增加的元数据

建议对含小时/分钟的原始数据增加以下全局属性：

```text
source_time_zone = "unknown" 或 "Asia/Tokyo" / "local station time"
time_zone_handling = "source timestamps treated as naive local time before daily aggregation"
time_aggregation_boundary = "local-naive calendar day"
```

如果后续需要统一到 UTC，则应在读取阶段显式处理，例如 Fukushima 可考虑：

```python
data["datetime"] = (
    data["datetime"]
    .dt.tz_localize("Asia/Tokyo")
    .dt.tz_convert("UTC")
)
```

但必须先决定聚合顺序：

- 先按地方日聚合，再转 UTC：适合保留当地水文日/采样日含义。
- 先转 UTC，再按 UTC 日聚合：适合全球统一日边界，但可能改变部分样本归属日期。

---

## 4. 各数据集处理流程

## 4.1 HYDAT

### 输入

HYDAT 处理包括 sediment 文件和 discharge/waterlevel 文件。sediment 侧可能包含：

- `time_sed_load`
- `sediment_load`
- `time_sed_suscon`
- `suspended_sediment_concentration`

flow 侧包含：

- `time_flow`
- `discharge`

### 时间轴处理

HYDAT 合并脚本将所有可用时间变量合并为统一时间轴：

```python
all_times = []
if time_sed_load is not None:
    all_times.append(time_sed_load)
if time_sed_suscon is not None:
    all_times.append(time_sed_suscon)
all_times.append(time_flow)

time_combined = np.unique(np.concatenate(all_times))
time_combined.sort()
```

随后：

1. 按统一时间轴初始化 `sediment_load`、`ssc`、`discharge`。
2. 将各变量映射到统一时间轴。
3. 对缺失项填充 `-9999.0`。
4. 若有 Q 和 SSL 但缺 SSC，则反推 SSC。
5. 若有 Q 和 SSC 但缺 SSL，则计算 SSL。
6. 将原始时间单位转换为 `days since 1970-01-01 00:00:00`。

### 变量计算

HYDAT 脚本中的关系式包括：

```text
sediment_load = discharge × ssc × 86.4
```

但在不同单位语境下，代码中也出现了 `/1000` 或 `*1000` 的处理。因此审核时应注意源 `ssc` 单位到底是 `mg L-1` 还是其他单位。

### 输出

输出采用：

- `time`: `f8(time)`
- `units = "days since 1970-01-01 00:00:00"`
- `calendar = "gregorian"`
- `axis = "T"`
- `discharge`, `ssc`, `sediment_load`
- scalar `latitude`, `longitude`, `altitude`, `upstream_area`

---

## 4.2 HYBAM

### 输入

HYBAM 站点目录通常包含 discharge 与 SSC 两类 NetCDF 文件。时间变量为 `Date`。

### 时间轴处理

HYBAM 将 `Date` 视为 Unix seconds。若 discharge 和 SSC 同时存在：

1. 找到 discharge 时间中落在 SSC 时间范围内的片段。
2. 以 discharge 时间轴为基准。
3. 将 SSC 映射到最近的 discharge 时间点。
4. 只有最近时间差小于 86400 秒，即 1 天，才写入 SSC。

若只有 discharge 或只有 SSC，则使用单一变量自身时间轴。

### 时间覆盖

时间覆盖范围通过：

```python
datetime.utcfromtimestamp(result['time'][0])
datetime.utcfromtimestamp(result['time'][-1])
```

生成，说明 HYBAM 时间语义更接近 UTC。

### 变量处理

- Q 保持为 `m3 s-1`。
- SSC 保持为 `mg L-1`。
- SSL 由 Q 和 SSC 推导。

代码中实际用于 mg/L 的日负荷公式为：

```text
SSL(ton/day) = Q(m3/s) × SSC(mg/L) × 0.0864
```

文档注释中有时写为 `86.4`，这通常对应 SSC 为 `g/L` 或单位解释不同的情况。建议统一说明和代码实现。

### QC

HYBAM 使用 `apply_hydro_qc_with_provenance`，输出：

- 最终标记：`Q_flag`, `SSC_flag`, `SSL_flag`
- 物理检查：`*_flag_qc1_physical`
- log-IQR 检查：`*_flag_qc2_log_iqr`
- SSC-Q 一致性：`SSC_flag_qc3_ssc_q`
- SSL 从 SSC-Q 异常传播：`SSL_flag_qc3_from_ssc_q`

---

## 4.3 Fukushima / DOI00147 Niida River

### 输入

Fukushima 源 Excel 每个 sheet 中包含：

- DOI / DID / station
- `yyyy`, `mm`, `dd`, `hh`, `min`
- latitude / longitude
- sampling depth
- discharge, unit `m3/s`
- SSC, unit `g/L`
- SSC uncertainty, unit `g/L`

### 时间处理

代码将年月日时分直接构造成 datetime：

```python
data['datetime'] = pd.to_datetime({
    'year': data[3].astype(int),
    'month': data[4].astype(int),
    'day': data[5].astype(int),
    'hour': data[6].astype(int),
    'minute': data[7].astype(int)
})
```

随后进行日平均：

```python
daily = df.set_index('datetime').resample('D').mean()
```

### 单位与变量计算

- SSC 原始单位为 `g/L`。
- 标准输出中 SSC 转为 `mg/L`：`ssc_mg_L = ssc * 1000`。
- SSL 使用原始 `g/L` 计算：

```text
SSL(ton/day) = Q(m3/s) × SSC(g/L) × 86.4
```

### 时间轴注意事项

Fukushima 当前没有显式时区处理。若源时间是日本地方时，则当前代码会：

1. 以 naive datetime 解释源时间；
2. 按 naive calendar day 聚合；
3. 在 NetCDF 输出中只保留日尺度时间；
4. 不记录源时区。

建议新增 `source_time_zone` 与 `time_zone_handling` 元数据。

---

## 4.4 RiverSed / Aquasat

### 输入

该脚本同时处理 Aquasat 与 RiverSed CSV。总体流程为：

1. 读取 Aquasat 源行并统一为 station/date/tss schema。
2. 读取 RiverSed 源行。
3. 读取修改后的 NHDPlus DBF 查找表。
4. 按 RiverSed ID 合并 reach/basin 元数据。
5. 从 flowline 几何中提取代表性 reach 坐标。
6. 按站点/河段分组。
7. 聚合为 daily SSC 时间序列。
8. 对 SSC 做 QC。
9. 输出每站点/每河段 NetCDF。

### RiverSed 时间处理

RiverSed 源 `date` 和 `time` 分开存储。代码将二者拼接：

```python
df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
```

随后输出前折叠到日尺度：

```python
tss_df['date'] = tss_df['date'].dt.floor('D')
tss_daily = tss_df.groupby('date', as_index=False)['tss'].mean()
```

代码明确说明输出是 sparse daily data，不补全无观测日。

### 变量与限制

RiverSed/Aquasat 主要提供 satellite-derived TSS/SSC：

- 有 SSC/TSS。
- 没有 in-situ discharge Q。
- 没有 SSL。
- 输出不是 gap-filled 日序列，仅包含有观测的日期。

### QC

对 satellite-only SSC：

1. QC1：物理可行性，缺测、负值等。
2. QC2：log-IQR 异常筛查，样本量足够时执行。
3. QC3：SSC-Q 一致性由于没有 Q，通常标记为 not_checked。
4. 最终 SSC flag 综合 QC1/QC2。

---

## 4.5 NERC

### 输入

NERC 脚本读取 CSV，并解析 `Date` 字段。代码使用 `dayfirst=True`，说明源日期格式按日-月-年解释。

### 时间处理

NERC 只处理日期，不处理小时或时区。处理流程为：

1. 找出有效数据的起止年份。
2. 将数据裁剪到起止年份的完整年份范围。
3. 用 `1970-01-01` 作为 reference date。
4. 计算 `df['time'] = (df['date'] - reference_date).dt.total_seconds() / 86400.0`。
5. 写入 `days since 1970-01-01 00:00:00`。

### 输出

输出变量包括：

- `Q`
- `SSC`
- `SSL`
- `Q_flag`, `SSC_flag`, `SSL_flag`
- `lat`, `lon`, `altitude`, `upstream_area`

NERC 对地方时问题基本不适用，因为最终只保留日期。

---

## 4.6 GloRiSe 普通时间序列

### 输入

GloRiSe 普通处理脚本将 station-level 数据表按 `datetime` 排序，并生成 NetCDF。

### 时间处理

时间使用：

```python
time_values = date2num(
    station_data['datetime'].tolist(),
    units='days since 1970-01-01 00:00:00',
    calendar='gregorian'
)
```

输出 `time` 变量具有：

- `standard_name = 'time'`
- `long_name = 'time'`
- `axis = 'T'`

### 变量

普通 GloRiSe 输出中变量名可能是：

- `Discharge_m3_s`
- `TSS_mg_L`

后续 QC/标准化步骤会统一到 `Q`、`SSC`、`SSL`。

---

## 4.7 GloRiSe annually_climatology

### 输入

该脚本处理 `Output_r/annually_climatology/GloRiSe` 下的 NetCDF，目标是：

1. 统一变量名：支持 `Discharge_m3_s/TSS_mg_L` 或 `Q/SSC/SSL`。
2. 统一为 `Q`, `SSC`, `SSL`。
3. 用 `apply_quality_flag` 做 climatology-safe QC。
4. 输出单个代表性 climatology 记录。

### 时间处理

如果输入包含时间轴：

1. 用输入 `time` 变量反解年份。
2. 取最小年份和最大年份作为源期起止。
3. 用 `climatology_mid_datetime(start_year, end_year)` 生成中间年 7 月 1 日。
4. 输出为一个 `time` 记录。

输出 `time`：

```text
long_name = "representative time of climatological mean"
comment = "It is set to July 1 of the middle year of the source period."
```

### 变量推导

若有 Q 和 SSC 但缺 SSL：

```text
SSL = Q × SSC × 0.0864
```

若有 Q 和 SSL 但缺 SSC，则通过 `calculate_ssc(SSL, Q)` 反推 SSC。

---

## 4.8 Milliman

### 输入

Milliman 处理对象是长期平均/多年平均类型数据。初始转换中会把：

- discharge 从 `km3/yr` 转为 `m3/s`
- sediment flux 从 `Mt/yr` 转为 `ton/day`
- SSC 由 SSL 和 Q 计算

### 时间处理

标准化脚本优先读取 station-specific observation period，例如：

- `period`
- `measurement_period`
- `temporal_span`
- `original_time_range`
- `time_period`

若能解析出 `start_year` 和 `end_year`，则：

```text
representative_time = July 1 of middle year
```

并写入：

```text
time_coverage_start = "{start_year}-01-01"
time_coverage_end = "{end_year}-12-31"
temporal_span = "{start_year}-{end_year}"
temporal_resolution = "climatology"
```

若无法解析观测期，则保留原代表时间，并写明：

```text
station-specific observation period unavailable
time coordinate is representative only
```

### 输出

输出采用 CF-1.8/ACDD-1.3，时间变量为：

```text
standard_name = "time"
long_name = "representative time of climatological mean"
units = "days since 1970-01-01 00:00:00"
calendar = "gregorian"
axis = "T"
```

---

## 4.9 ALi_De_Boer

### 时间处理

ALi_De_Boer 也是 climatology/多年平均型数据。处理逻辑为：

1. 解析 `start_year`、`end_year`。
2. 取中间年：`mid_year = (start_year + end_year) // 2`。
3. 使用 `datetime(mid_year, 7, 1)` 作为代表时间。
4. 写入 `days since 1970-01-01 00:00:00`。

输出 `time` 的注释明确说明：该时间是记录期中点，仅代表多年平均数据，不是瞬时观测。

---

## 4.10 Vanmaercke

### 输入和时间处理

Vanmaercke 的处理逻辑也依赖 period 字段：

- 若识别出年份范围，则以该范围中间年 7 月 1 日作为代表时间。
- 若只识别出单一年份，则使用该年 7 月 1 日。
- 若无法解析 period，则退回原 time 变量，并注明这只是 representative time。

### 输出

输出与其他 climatology 数据一致，使用：

```text
units = "days since 1970-01-01 00:00:00"
calendar = "gregorian"
```

---

## 4.11 Yajiang

### 输入

Yajiang 脚本读取行级日期字段，例如：

- `Date (YYYYMMDD)`
- `Date`
- `Time`

并通过 `parse_date_yyyymmdd` 解析。

### 时间处理

每条记录输出一个时间点：

```python
time_var[:] = nc.date2num([date_dt], time_var.units, time_var.calendar)
```

其中：

```text
units = "days since 1970-01-01 00:00:00"
calendar = "gregorian"
standard_name = "time"
long_name = "time"
```

该脚本只处理日期级信息，未见时区处理。

---

## 5. 发布/集成层时间检查

`sed_data_integration` 发布层会读取核心产品和 catalog 的时间范围，并比较：

- master
- daily matrix
- monthly matrix
- annual matrix
- station catalog
- source station catalog
- climatology product

如果发现时间范围不一致，发布脚本会判定为 mixed-run outputs，并提示需要从 s1 到 s8 全链重跑。这一步用于避免不同运行批次的产品混在同一 release 中。

---

## 6. 主要风险点

### 6.1 地方时未显式记录

Fukushima 与 RiverSed 最值得关注，因为它们含小时/分钟或 date+time 信息，但当前没有：

- 时区声明；
- 地方时到 UTC 的转换；
- 聚合日边界说明。

### 6.2 `86.4` 与 `0.0864` 的单位语义需统一

泥沙负荷公式取决于 SSC 单位：

| SSC 单位 | SSL 公式 |
|---|---|
| `g/L` | `SSL(ton/day) = Q × SSC × 86.4` |
| `mg/L` | `SSL(ton/day) = Q × SSC × 0.0864` |

代码中不同数据集可能因为源单位不同使用不同系数。建议在每个输出变量 `SSL.comment` 中明确源 SSC 单位与换算公式。

### 6.3 时间单位字符串轻微不一致

大多数脚本使用：

```text
days since 1970-01-01 00:00:00
```

但部分读取默认或 fallback 会使用：

```text
days since 1970-01-01
```

二者通常等价，但如果要做严格字符串一致性检查，建议统一为带时间部分的版本。

### 6.4 climatology 时间容易被误读

climatology 的 `time` 不是真实观测日期，而是代表性日期。建议在集成产品和 README 中反复强调：

```text
For climatology records, time is a representative timestamp only, set to July 1 of the middle year of the observation/source period where available.
```

---

## 7. 建议改进清单

### 7.1 增加统一时间元数据

建议所有输出 NetCDF 添加：

```text
time_encoding = "numeric days"
time_reference = "1970-01-01 00:00:00"
time_calendar = "gregorian"
time_axis_type = "observation" 或 "representative_climatology"
```

### 7.2 对含小时源数据添加时区说明

建议 Fukushima：

```text
source_time_zone = "Asia/Tokyo" 或 "unknown"
time_zone_handling = "source timestamps treated as naive local time before daily aggregation"
time_aggregation_boundary = "source local calendar day"
```

建议 RiverSed：

```text
source_time_zone = "unknown / source-provided local time"
time_zone_handling = "date+time parsed as naive datetime; daily aggregation before export"
time_aggregation_boundary = "naive calendar day"
```

### 7.3 统一 SSL 公式注释

建议所有脚本按 SSC 单位写清：

```text
If SSC is mg/L: SSL = Q * SSC * 0.0864
If SSC is g/L:  SSL = Q * SSC * 86.4
```

### 7.4 对 climatology 文件增加更强提示

建议增加：

```text
time_is_representative = "true"
representative_time_method = "July 1 of middle year of source/observation period"
```

### 7.5 发布层增加时区完整性检查

可在发布验证中检查：

- 含小时级源数据是否有 `source_time_zone`。
- climatology 是否有 `time_is_representative`。
- daily 数据是否说明 `time_aggregation_boundary`。

---

## 8. 简表：各数据集时间处理对比

| 数据集 | 时间来源 | 标准化时间单位 | 时间轴类型 | 是否可能涉及地方时 | 当前是否显式处理时区 |
|---|---|---|---|---|---|
| HYDAT | `time_sed_load`, `time_sed_suscon`, `time_flow` | `days since 1970-01-01 00:00:00` | 观测时间并集 | 低 | 否 |
| HYBAM | Unix seconds `Date` | `days since 1970-01-01 00:00:00` | 观测时间，UTC 语义 | 低 | 使用 `utcfromtimestamp` |
| Fukushima | `yyyy/mm/dd/hh/min` | 部分以源起始日为基准 | 日平均观测时间 | 高 | 否 |
| RiverSed | `date + time` | daily 输出 | sparse daily SSC | 中/高 | 否 |
| Aquasat | `date` | daily 输出 | sparse daily SSC | 低/中 | 否 |
| NERC | `Date` | `days since 1970-01-01 00:00:00` | 日期级观测时间 | 低 | 否 |
| GloRiSe time series | `datetime` | `days since 1970-01-01 00:00:00` | 观测时间 | 未明确 | 否 |
| GloRiSe climatology | 输入时间范围 | `days since 1970-01-01 00:00:00` | representative climatology | 不适用 | 不适用 |
| Milliman | observation period / fallback representative time | `days since 1970-01-01 00:00:00` | representative climatology | 不适用 | 不适用 |
| ALi_De_Boer | start/end year | `days since 1970-01-01 00:00:00` | representative climatology | 不适用 | 不适用 |
| Vanmaercke | period year/range | `days since 1970-01-01 00:00:00` | representative climatology | 不适用 | 不适用 |
| Yajiang | `YYYYMMDD` 日期 | `days since 1970-01-01 00:00:00` | 日期级观测时间 | 低 | 否 |

---

## 9. 总结

`sed_data` 当前处理体系已经形成比较清晰的标准化框架：

1. 普通观测数据统一为 CF 风格 `time` 轴。
2. 多变量数据按数据集特性合并时间轴。
3. climatology 数据使用中间年 7 月 1 日作为代表时间。
4. Q/SSC/SSL 统一到常用水文泥沙单位。
5. QC flags 与分步 QC provenance 被逐步加入 NetCDF。

最需要进一步完善的是时间语义元数据，尤其是 Fukushima 和 RiverSed 这类包含小时级源时间的数据。当前代码没有证明它们是地方时，但也没有排除；更稳妥的做法是在输出中明确记录源时区未知或源地方时处理方式，并明确日聚合边界。

