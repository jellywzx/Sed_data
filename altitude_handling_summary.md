# Script 中其他数据集对 altitude 的处理方式

本文整理 `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script` 下其他数据集处理代码中关于 `altitude` / `elevation` 的处理方式，并与 GFQA_v2 当前写法作对照。

## 总体规律

大多数数据集把 `altitude` 写成 NetCDF 里的标量变量，变量名通常为 `altitude`，单位为 `m`，并设置：

- `standard_name = "altitude"`
- `long_name` 通常为 `"station elevation above sea level"` 或 `"station altitude above sea level"`
- 常见属性包括 `positive = "up"`
- 如果原始数据没有高程，则写 `-9999.0`、`np.nan` 或对应的 `fill_value`

很多标准化后的时间序列变量也会把 `altitude` 放进 `coordinates` 属性，例如：

```python
q_var.coordinates = "time lat lon altitude"
ssc_var.coordinates = "time lat lon altitude"
ssl_var.coordinates = "time lat lon altitude"
```

GFQA_v2 的 `gfqa_to_netcdf_daily_dualqc_test.py` 当前做法与这个主流模式一致：创建标量变量 `altitude`，并将 `Q`、`SSC`、`SSL` 的 `coordinates` 改为包含 `altitude`。

## 有真实高程来源的数据集

### USGS

从站点信息字段 `alt_va` 读取高程，并将英尺转换为米：

```python
ds['altitude'] = ((), station_info['alt_va'] * FEET_TO_METERS if pd.notna(station_info['alt_va']) else np.nan)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/USGS/process_usgs.py:405`

### ALi_De_Boer

从表格列 `Elevation (masl)` 读取高程：

```python
elevation = row['Elevation (masl)']
```

然后写入 NetCDF 标量变量 `altitude`：

```python
alt_var = nc.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station elevation above sea level'
alt_var.units = 'm'
alt_var.positive = 'up'
alt_var.comment = 'Source: Original data provided by Ali & De Boer (2007).'
if not pd.isna(elevation):
    alt_var[:] = elevation
```

如果高程有效，还会写：

```python
nc.geospatial_vertical_min = float(elevation)
nc.geospatial_vertical_max = float(elevation)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/ALi_De_Boer/process_data_tool.py:81`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/ALi_De_Boer/process_data_tool.py:287`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/ALi_De_Boer/process_data_tool.py:421`

### bayern

从元数据行 `Pegelnullpunktshöhe` 解析高程，支持逗号小数：

```python
elif 'Pegelnullpunktshöhe' in line:
    parts = line.split(';')[1].strip('"').split()
    try:
        metadata['altitude'] = float(parts[0].replace(',', '.'))
    except:
        metadata['altitude'] = np.nan
```

写入 NetCDF：

```python
alt_var = dataset.createVariable('altitude', 'f4')
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station altitude above sea level'
alt_var.units = 'm'
alt_var[:] = metadata.get('altitude', np.nan)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/bayern/convert_bayern_to_netcdf.py:57`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/bayern/convert_bayern_to_netcdf.py:342`

### RiverSed

从合并后的元数据列 `elevation` 取第一个有效数值：

```python
altitude = _first_valid_numeric(tss_df, "elevation")
```

然后写成 `altitude` 变量，缺失时写 `-9999.0`：

```python
alt_var = ds.createVariable('altitude', 'f4')
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station altitude above sea level'
alt_var.units = 'm'
alt_var[:] = altitude if not np.isnan(altitude) else -9999.0
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/convert_to_netcdf.py:939`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/RiverSed/convert_to_netcdf.py:1010`

### HYBAM

从 `STATION_INFO` 里的 `alt` 字段读取：

```python
info = STATION_INFO.get(station_id, None)
if info:
    latitude = info["lat"]
    longitude = info["lon"]
    altitude = info["alt"]
else:
    latitude = FILL_VALUE_FLOAT
    longitude = FILL_VALUE_FLOAT
    altitude = FILL_VALUE_FLOAT
```

如果 `altitude is not None`，则创建 `altitude` 变量：

```python
if altitude is not None:
    alt_var = ds.createVariable('altitude', 'f4', zlib=True)
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station elevation above sea level'
    alt_var.units = 'm'
    alt_var.positive = 'up'
    alt_var[:] = altitude
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/HYBAM/hybam_comprehensive_processor.py:449`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/HYBAM/hybam_comprehensive_processor.py:743`

### Eurasian_River

解析文本中的 `Gauge Altitude` 到 `metadata['altitude']`：

```python
elif i == 7:  # Gauge Altitude
    metadata['altitude'] = float(line.split(':')[1].strip())
```

但在当前代码中，没有看到它实际创建 NetCDF 的 `altitude` 变量；该值主要进入 summary：

```python
"altitude": dis_data["meta"].get("altitude", np.nan)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Eurasian_River/process_eurasian_river.py:272`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Eurasian_River/process_eurasian_river.py:664`

## 没有高程来源的数据集

这些脚本会显式创建 `altitude` 变量，但填缺测值，并在 `comment` 里说明原始数据没有高程。

### Chao_Phraya_River

```python
alt = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE)
alt.standard_name = "altitude"
alt.long_name = "station elevation above sea level"
alt.units = "m"
alt.positive = "up"
alt.comment = "Source: Not available in original dataset."
alt[:] = FILL_VALUE
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Chao_Phraya_River/process_chao_phraya.py:288`

### EUSEDcollab

```python
alt_var = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE_FLOAT)
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station elevation above sea level'
alt_var.units = 'm'
alt_var.positive = 'up'
alt_var.comment = 'Source: Not available in EUSEDcollab metadata.'
alt_var[:] = FILL_VALUE
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/EUSEDcollab/process_eusedcollab_to_cf18_wzx.py:870`

### Milliman

```python
alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
alt_var.long_name = "station elevation above sea level"
alt_var.standard_name = "altitude"
alt_var.units = "m"
alt_var.positive = "up"
alt_var.comment = "Source: Not available in Milliman database."
alt_var[:] = -9999.0
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Milliman/5_qc_and_standardize.py:165`

### HMA

```python
# Altitude (not in CSV, set to missing)
alt_var = nc.createVariable('altitude', 'f4', fill_value=-9999.0)
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station elevation above sea level'
alt_var.units = 'm'
alt_var.positive = 'up'
alt_var.comment = 'Source: Not available in original dataset.'
alt_var[:] = -9999.0
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/HMA/convert_to_netcdf_cf18_qc.py:444`

### Myanmar

```python
alt_var = ds.createVariable('altitude', 'f4', fill_value=fill_value)
alt_var.long_name = 'station altitude'
alt_var.units = 'm'
alt_var.missing_value = fill_value
alt_var[:] = fill_value
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Myanmar/convert_to_netcdf.py:274`

### Shashi_Jianli

```python
ds['altitude'] = ((), np.nan, {
    'long_name': 'station altitude',
    'standard_name': 'altitude',
    'units': 'm',
    'comment': 'Not available in source data'
})
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Shashi_Jianli/process_shashi_jianli.py:405`

### Vanmaercke

```python
alt_var = dataset.createVariable('altitude', 'f4')
alt_var.units = 'm'
alt_var.long_name = 'altitude'
alt_var.standard_name = 'altitude'
alt_var.comment = 'Not available in source dataset'
alt_var.assignValue(np.nan)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Vanmaercke/convert_to_netcdf.py:159`

### Land2sea

```python
alt_var = ds.createVariable('altitude', 'f4')
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station altitude above sea level'
alt_var.units = 'm'
alt_var[:] = np.nan  # Not available in Land2Sea
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Land2sea/convert_land2sea_to_netcdf.py:158`

### NERC

站点元数据里直接把 altitude 设为 `-9999.0`：

```python
'altitude': -9999.0,  # Not provided in data files
```

写出时：

```python
alt_var = ncfile.createVariable('altitude', 'f4', fill_value=-9999.0)
alt_var.standard_name = 'altitude'
alt_var.long_name = 'station elevation above sea level'
alt_var.units = 'm'
alt_var.positive = 'up'
alt_var.comment = 'Source: Not provided in original dataset.'
alt_var[:] = metadata['altitude']
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/NERC/convert_NERC_to_netcdf.py:60`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/NERC/convert_NERC_to_netcdf.py:430`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/NERC/convert_NERC_to_netcdf.py:491`

### Mekong_Delta

站点元数据里 `altitude` 是 `np.nan`：

```python
'altitude': np.nan
```

写出时，如果不是 NaN 就写值，否则在部分脚本中写 `fill_value`：

```python
alt_var = ds.createVariable('altitude', 'f4')
alt_var.long_name = "station altitude"
alt_var.standard_name = "altitude"
alt_var.units = "m"
alt_var.missing_value = fill_value
if not np.isnan(station_meta['altitude']):
    alt_var[:] = station_meta['altitude']
else:
    alt_var[:] = fill_value
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Mekong_Delta/process_mekong_delta.py:63`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Mekong_Delta/process_mekong_delta.py:434`

### Huanghe

原始转换脚本里 altitude 不可用，写 NaN；标准化脚本中则按是否为 NaN 写值或 `-9999.0`：

```python
alt_var = ds.createVariable("altitude", "f4", fill_value=-9999.0)
alt_var.long_name = "station elevation above sea level"
alt_var.standard_name = "altitude"
alt_var.units = "m"
alt_var.positive = "up"
alt_var.comment = "Source: Not available in original dataset."
if np.isnan(alt):
    alt_var[:] = -9999.0
else:
    alt_var[:] = np.float32(alt)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Huanghe/qc_and_standardize.py:287`

## 从已有 NetCDF 继承 altitude 的做法

一些 QC 或 standardize 脚本不是重新从原始表读高程，而是从已有 NetCDF 中读取 `altitude`，再写入标准化输出。

### USGS existing NetCDF

```python
new_ds['altitude'] = ((), ds.altitude.item() if 'altitude' in ds else np.nan)
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/USGS/process_existing_usgs_netcdf.py:98`

### HYDAT

先读取已有变量：

```python
altitude = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else -9999.0
```

再写出标准化变量：

```python
var_alt = ds_out.createVariable('altitude', 'f4', fill_value=-9999.0)
var_alt.standard_name = 'altitude'
var_alt.long_name = 'station elevation above sea level'
var_alt.units = 'm'
var_alt.positive = 'up'
var_alt.comment = 'Source: HYDAT database.'
var_alt[:] = altitude
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Hydat/4_process_hydat_cf18.py:264`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Hydat/4_process_hydat_cf18.py:361`

### bayern / GloRiSe / Vanmaercke QC 脚本

这些脚本同样会读取输入文件里的 `altitude`，再标准化写出：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/bayern/qc_and_standardize.py:174`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/GloRiSe/2_qc_and_standardize_glorise.py:214`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/GloRiSe/4_qc_and_standardize_BS.py:250`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Vanmaercke/qc_and_standardize.py:72`

## 不太统一的地方

### Yajiang

初始转换脚本写的是变量 `elevation`，不是 `altitude`：

```python
elev = clean_float(row.get("Elevation (m)", row.get("Elevation", None)))
```

```python
elev_var = ds.createVariable("elevation", "f4")
elev_var.units = "m"
elev_var.standard_name = "height_above_mean_sea_level"
elev_var.long_name = "station elevation"
elev_var.assignValue(elev if not np.isnan(elev) else np.nan)
```

但后续处理脚本读取的是 `altitude`：

```python
new_ds['altitude'] = ((), ds.altitude.item() if 'altitude' in ds else np.nan)
```

因此这里存在 `elevation` 与 `altitude` 命名不一致的问题。

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Yajiang/convert_to_nc.py:90`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Yajiang/convert_to_nc.py:132`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Yajiang/process_yajiang.py:519`

### GloRiSe

早期生成脚本把 `Elevation_masl` 写成全局属性 `nc.elevation`，不是 `altitude` 变量：

```python
if pd.notna(location_info['Elevation_masl']):
    nc.elevation = float(location_info['Elevation_masl'])
```

但后续 QC 脚本主要读取 `altitude` 变量：

```python
alt = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else np.nan
```

因此这里也存在全局属性 `elevation` 与变量 `altitude` 的衔接问题。

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/GloRiSe/1_generate_netcdf_SS.py:318`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/GloRiSe/2_qc_and_standardize_glorise.py:214`

### Fukushima

`altitude` 变量创建代码被注释掉了：

```python
# alt_var = dataset.createVariable('altitude', 'f4')
# alt_var.standard_name = 'altitude'
# alt_var.long_name = 'station elevation above sea level'
# alt_var.units = 'm'
# alt_var.positive = 'up'
# alt_var._FillValue = -9999.0
# alt_var.comment = f'Sampling depth: {depth} m below surface. Negative values indicate below water surface.'
# alt_var[:] = -depth  # Negative for below surface
```

NetCDF 元数据里说明站点 altitude 暂无可靠来源：

```python
"Station altitude and upstream drainage area are not included in the current "
"version due to lack of reliable source information. These variables will be "
"added in future releases when available."
```

但 summary 中使用了负的采样深度作为 `altitude`：

```python
'altitude': -data['depth'].iloc[0],  # Negative for below surface
```

位置：

- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Fukushima/fukushima_qc_and_cf_enhancement.py:507`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Fukushima/fukushima_qc_and_cf_enhancement.py:596`
- `/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script/Fukushima/fukushima_qc_and_cf_enhancement.py:831`

## 与 GFQA_v2 的对照结论

`gfqa_to_netcdf_daily_dualqc.py` 原始版本只把高程写成全局属性：

```python
ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
```

而 `gfqa_to_netcdf_daily_dualqc_test.py` 的新版本做了以下增强：

1. 从多个可能列名读取高程：`Elevation`、`Elevation (m)`、`Elevation_m`、`Altitude`、`Altitude (m)`、`Altitude_m`、`altitude`、`elevation`
2. 创建 NetCDF 标量变量 `altitude`
3. 设置 `units = "m"`、`standard_name = "altitude"`、`long_name = "station altitude above mean sea level"`、`positive = "up"`
4. 将 `Q`、`SSC`、`SSL` 的 `coordinates` 改为包含 `altitude`
5. 全局属性 `ds.altitude` 也复用同一个解析后的 `altitude` 值

这比原始 GFQA 脚本更接近项目里大多数标准化脚本的写法，也更利于 CF-1.8 风格的数据变量描述。

一个小注意点是：当前 `get_station_altitude()` 找到第一个存在的高程列后就立即返回。如果该列存在但某个站点值为空，它不会继续尝试后面的备选列。如果 GEMStat 元数据可能存在多列混用，可以考虑改成“当前列解析有效才返回，否则继续检查下一列”。
