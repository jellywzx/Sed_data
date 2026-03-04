#!/usr/bin/env python3
"""
GloRiSe annually_climatology NetCDF: 变量统一 + 质量控制

对 Output_r/annually_climatology/GloRiSe 下的 nc 进行：
1. 变量统一：支持 Discharge_m3_s/TSS_mg_L 或 Q/SSC/SSL，统一为 Q、SSC、SSL
2. 质量控制：仅调用 tool.apply_quality_flag 与 tool.calculate_ssc

流程参照 1_generate_netcdf_SS.py 与 2_qc_and_standardize_glorise.py，
但 QC 仅使用 apply_quality_flag 和 calculate_ssc。
"""

import os
import sys
import numpy as np
import netCDF4 as nc4
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
from tool import FILL_VALUE_FLOAT, FILL_VALUE_INT, apply_quality_flag, calculate_ssc

# 路径配置
INPUT_DIR = Path('/media/zhwei/data02/weizx/sediment_wzx_1111/Output_r/annually_climatology/GloRiSe')
OUTPUT_DIR = INPUT_DIR / 'qc'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 与 2_qc 一致的数据源信息
DATA_SOURCE = {
    'name': 'GloRiSe Dataset',
    'full_name': 'Global River Sediment Database v1.1',
    'type': 'In-situ',
    'temporal_resolution': 'annually climatology',
    'reference': 'Müller, G., Middelburg, J. J., and Sluijs, A.: Introducing GloRiSe – a global database on river sediment composition, Earth Syst. Sci. Data, 13, 3565–3575, https://doi.org/10.5194/essd-13-3565-2021, 2021.',
    'data_link': 'https://doi.org/10.5281/zenodo.4485795',
    'creator_name': 'Zhongwang Wei',
    'creator_email': 'weizhw6@mail.sysu.edu.cn',
    'creator_institution': 'Sun Yat-sen University, China',
}


def _read_var_squeezed(ds, name, default=None):
    """从 Dataset 读取变量并压成 1D（便于标量/时间序列/样本维统一处理）。"""
    if name not in ds.variables:
        return default
    v = ds.variables[name][:]
    return np.asarray(v).flatten()


def _scalar_float(x, default=np.nan):
    """将可能为数组的值转为 Python float，避免 truth value of array 错误。"""
    try:
        a = np.asarray(x, dtype=np.float64).flatten()
    except Exception:
        return default
    if a.size == 0:
        return default
    v = a.flat[0]
    try:
        if np.ma.is_masked(v):
            return default
    except Exception:
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _scalar_bool(x):
    """将可能为 0 维数组的布尔值转为 Python bool。"""
    if hasattr(x, 'shape') and x.shape != ():
        return bool(np.any(x))  # 多元素时用 any
    return bool(x)


def _get_coord(ds, name, default=np.nan):
    """读取标量坐标 (lat/lon/altitude 等)。"""
    if name not in ds.variables:
        return default
    v = ds.variables[name][:]
    return _scalar_float(np.asarray(v).flatten()[0], default=default)


def _infer_and_unify_variables(ds):
    """
    从 nc 中识别 Q/SSC/SSL（或 Discharge_m3_s, TSS_mg_L），统一为 Q, SSC, SSL 的 1D 数组。
    缺失的用 Q*SSC*0.0864=SSL 或 calculate_ssc(SSL,Q) 推导。
    返回 (Q, SSC, SSL) 及 time_dim_name（用于写回时的维度名）。
    """
    # 识别时间/样本维度名
    dims = list(ds.dimensions.keys())
    time_dim_name = 'time' if 'time' in dims else ('sample' if 'sample' in dims else None)
    if not time_dim_name:
        # 单变量可能是 (lat, lon)，当作 1 个样本
        n = 1
    else:
        n = len(ds.dimensions[time_dim_name])

    # 可能的名字映射（不用 "x or y"，否则 x 为数组时会触发 truth value of array 错误）
    q_raw = _read_var_squeezed(ds, 'Q')
    if q_raw is None:
        q_raw = _read_var_squeezed(ds, 'Discharge_m3_s')
    ssc_raw = _read_var_squeezed(ds, 'SSC')
    if ssc_raw is None:
        ssc_raw = _read_var_squeezed(ds, 'TSS_mg_L')
    ssl_raw = _read_var_squeezed(ds, 'SSL')

    fill = float(FILL_VALUE_FLOAT)
    Q = np.full(n, fill, dtype=np.float32)
    SSC = np.full(n, fill, dtype=np.float32)
    SSL = np.full(n, fill, dtype=np.float32)

    if q_raw is not None and len(q_raw) >= n:
        Q[:] = q_raw[:n]
    if ssc_raw is not None and len(ssc_raw) >= n:
        SSC[:] = ssc_raw[:n]
    if ssl_raw is not None and len(ssl_raw) >= n:
        SSL[:] = ssl_raw[:n]

    # 缺失值用 NaN 便于推导
    Q_nan = np.where((Q == fill) | np.isnan(Q), np.nan, Q)
    SSC_nan = np.where((SSC == fill) | np.isnan(SSC), np.nan, SSC)
    SSL_nan = np.where((SSL == fill) | np.isnan(SSL), np.nan, SSL)

    # 统一：优先用 Q、SSC 算 SSL；若缺 SSC 但有 SSL 和 Q，用 calculate_ssc 反推 SSC
    # 全部用 _scalar_float / _scalar_bool 避免 sample>1 时 "truth value of array" 错误
    Q_flat = np.asarray(Q_nan, dtype=np.float64).flatten()
    SSC_flat = np.asarray(SSC_nan, dtype=np.float64).flatten()
    SSL_flat = np.asarray(SSL_nan, dtype=np.float64).flatten()
    Q_arr = np.asarray(Q, dtype=np.float64).flatten()
    SSC_arr = np.asarray(SSC, dtype=np.float64).flatten()
    SSL_arr = np.asarray(SSL, dtype=np.float64).flatten()
    for i in range(n):
        q = _scalar_float(Q_flat.flat[i] if i < Q_flat.size else np.nan)
        ssc = _scalar_float(SSC_flat.flat[i] if i < SSC_flat.size else np.nan)
        ssl = _scalar_float(SSL_flat.flat[i] if i < SSL_flat.size else np.nan)
        q_ok = _scalar_bool(np.isfinite(q))
        ssc_ok = _scalar_bool(np.isfinite(ssc))
        ssl_ok = _scalar_bool(np.isfinite(ssl))
        if q_ok and ssc_ok and (not ssl_ok):
            SSL[i] = np.float32(q * ssc * 0.0864)
        elif q_ok and ssl_ok and (not ssc_ok):
            ssc_calc = calculate_ssc(ssl, q)
            if _scalar_bool(np.isfinite(ssc_calc)):
                SSC[i] = np.float32(ssc_calc)
        # 若原位置是 fill，保持 fill（SSC/SSL 用当前值，可能已被上面更新）
        qi = _scalar_float(Q_arr.flat[i] if i < Q_arr.size else np.nan)
        if not _scalar_bool(np.isfinite(qi)):
            Q[i] = fill
        si = _scalar_float(np.asarray(SSC).flat[i] if i < np.asarray(SSC).size else np.nan)
        if not _scalar_bool(np.isfinite(si)):
            SSC[i] = fill
        li = _scalar_float(np.asarray(SSL).flat[i] if i < np.asarray(SSL).size else np.nan)
        if not _scalar_bool(np.isfinite(li)):
            SSL[i] = fill

    return Q, SSC, SSL, time_dim_name, n


def apply_qc_flags_only(Q, SSC, SSL):
    """
    仅使用 tool.apply_quality_flag 对 Q、SSC、SSL 打质量码。
    返回 (Q_flag, SSC_flag, SSL_flag) 为 int8 数组。
    使用 .flat[i] 取标量，兼容 sample>1 时多组数据。
    """
    Q = np.asarray(Q, dtype=np.float64).flatten()
    SSC = np.asarray(SSC, dtype=np.float64).flatten()
    SSL = np.asarray(SSL, dtype=np.float64).flatten()
    n = len(Q)
    q_flag = np.array([apply_quality_flag(_scalar_float(Q.flat[i]), 'Q') for i in range(n)], dtype=np.int8)
    ssc_flag = np.array([apply_quality_flag(_scalar_float(SSC.flat[i]), 'SSC') for i in range(n)], dtype=np.int8)
    ssl_flag = np.array([apply_quality_flag(_scalar_float(SSL.flat[i]), 'SSL') for i in range(n)], dtype=np.int8)
    return q_flag, ssc_flag, ssl_flag


def process_one_file(input_path):
    """
    处理单个 nc：变量统一 + 仅 apply_quality_flag/calculate_ssc 的 QC，写出到 OUTPUT_DIR。
    返回 True 成功，False 跳过/失败。
    """
    input_path = Path(input_path)
    if not input_path.suffix == '.nc' or input_path.name.startswith('.'):
        return False

    # 输出文件名：保持原名写入 qc 目录
    output_path = OUTPUT_DIR / input_path.name

    ds_in = nc4.Dataset(input_path, 'r')
    try:
        Q, SSC, SSL, time_dim_name, n = _infer_and_unify_variables(ds_in)
        if n == 0:
            print(f"  跳过（无有效维度）: {input_path.name}")
            return False

        # 质量控制：仅 apply_quality_flag（calculate_ssc 已在变量统一时用于推导 SSC）
        q_flag, ssc_flag, ssl_flag = apply_qc_flags_only(Q, SSC, SSL)

        lat = _get_coord(ds_in, 'lat') if 'lat' in ds_in.variables else _get_coord(ds_in, 'latitude')
        lon = _get_coord(ds_in, 'lon') if 'lon' in ds_in.variables else _get_coord(ds_in, 'longitude')
        alt = _get_coord(ds_in, 'altitude')
        upstream_area = _get_coord(ds_in, 'upstream_area')
        if not _scalar_bool(np.isfinite(alt)):
            alt = np.nan
        if not _scalar_bool(np.isfinite(upstream_area)):
            upstream_area = np.nan

        # 确定写回维度：与 2_qc 一致用 (time,) 或 (sample,) 标量坐标
        if time_dim_name == 'time':
            dim_list = ['time']
            time_var = ds_in.variables['time']
            time_vals = time_var[:]
            time_units = getattr(time_var, 'units', 'days since 1970-01-01 00:00:00')
            time_calendar = getattr(time_var, 'calendar', 'gregorian')
        elif time_dim_name == 'sample':
            dim_list = ['sample']
            time_vals = np.arange(n, dtype=np.float64)
            time_units = 'days since 1970-01-01 00:00:00'
            time_calendar = 'gregorian'
        else:
            dim_list = ['time']
            time_vals = np.array([0.0])
            time_units = 'days since 1970-01-01 00:00:00'
            time_calendar = 'gregorian'

        ds_out = nc4.Dataset(output_path, 'w', format='NETCDF4')

        # 维度：仅时间/样本维，lat/lon 为标量（与 2_qc 一致）
        ds_out.createDimension(dim_list[0], n)

        # 坐标变量：标量 lat, lon
        t_var = ds_out.createVariable(dim_list[0], 'f8', (dim_list[0],))
        t_var.standard_name = 'time'
        t_var.long_name = 'time'
        t_var.units = time_units
        t_var.calendar = time_calendar
        t_var.axis = 'T'
        t_var[:] = time_vals

        lat_var = ds_out.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
        lat_var[:] = np.float32(lat)

        lon_var = ds_out.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
        lon_var[:] = np.float32(lon)

        alt_var = ds_out.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var[:] = np.float32(alt)

        area_var = ds_out.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var[:] = np.float32(upstream_area)

        # 数据变量：统一为 (time,) 或 (sample,) 一维，与 2_qc 一致
        q_var = ds_out.createVariable('Q', 'f4', (dim_list[0],), fill_value=-9999.0, zlib=True, complevel=4)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = f'{dim_list[0]} lat lon altitude'
        q_var.ancillary_variables = 'Q_flag'
        q_var[:] = Q

        ssc_var = ds_out.createVariable('SSC', 'f4', (dim_list[0],), fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = f'{dim_list[0]} lat lon altitude'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var[:] = SSC

        ssl_var = ds_out.createVariable('SSL', 'f4', (dim_list[0],), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = f'{dim_list[0]} lat lon altitude'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'SSL = Q × SSC × 0.0864'
        ssl_var[:] = SSL

        q_flag_var = ds_out.createVariable('Q_flag', 'i1', (dim_list[0],), fill_value=9, zlib=True, complevel=4)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var[:] = q_flag

        ssc_flag_var = ds_out.createVariable('SSC_flag', 'i1', (dim_list[0],), fill_value=9, zlib=True, complevel=4)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var[:] = ssc_flag

        ssl_flag_var = ds_out.createVariable('SSL_flag', 'i1', (dim_list[0],), fill_value=9, zlib=True, complevel=4)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var[:] = ssl_flag

        # 若原文件有 Sand/Silt/Clay 等，复制到输出（保持兼容）
        for vname in ('Sand_perc', 'Silt_perc', 'Clay_perc'):
            if vname in ds_in.variables:
                vin = ds_in.variables[vname]
                data = np.asarray(vin[:]).flatten()[:n]
                if vname not in ds_out.variables:
                    vout = ds_out.createVariable(vname, vin.dtype, (dim_list[0],), fill_value=getattr(vin, '_FillValue', -9999.0))
                    for attr in ('units', 'long_name', 'standard_name'):
                        if hasattr(vin, attr):
                            setattr(vout, attr, getattr(vin, attr))
                    vout.coordinates = f'{dim_list[0]} lat lon'
                ds_out.variables[vname][:] = data

        # 全局属性（CF-1.8 / ACDD-1.3 风格，与 2_qc 一致）
        station_id = input_path.stem.replace('GloRiSe_', '').replace('GloRiSe_an_', '')
        ds_out.Conventions = 'CF-1.8, ACDD-1.3'
        ds_out.title = f'Harmonized GloRiSe annually climatology: {station_id}'
        ds_out.summary = f'River discharge and suspended sediment (annually climatology), variable unified and QC applied (apply_quality_flag + calculate_ssc only).'
        ds_out.data_source_name = DATA_SOURCE['name']
        ds_out.source_data_type = DATA_SOURCE['type']
        ds_out.source = f"{DATA_SOURCE['full_name']} - annually climatology, QC applied"
        ds_out.station_name = station_id
        ds_out.Source_ID = station_id
        ds_out.temporal_resolution = DATA_SOURCE['temporal_resolution']
        ds_out.geospatial_lat_min = lat
        ds_out.geospatial_lat_max = lat
        ds_out.geospatial_lon_min = lon
        ds_out.geospatial_lon_max = lon
        ds_out.reference = DATA_SOURCE['reference']
        ds_out.source_data_link = DATA_SOURCE['data_link']
        ds_out.creator_name = DATA_SOURCE['creator_name']
        ds_out.creator_email = DATA_SOURCE['creator_email']
        ds_out.creator_institution = DATA_SOURCE['creator_institution']
        ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
        ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
        ds_out.processing_level = 'Variable unified and QC (apply_quality_flag, calculate_ssc)'
        history_entry = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            "Variable unification and QC (tool.apply_quality_flag, tool.calculate_ssc only). "
            "Script: qc_glorise_annually_climatology.py"
        )
        ds_out.history = history_entry

        ds_out.close()
        ds_in.close()
        print(f"  已处理: {input_path.name} -> {output_path.name}")
        return True
    except Exception as e:
        ds_in.close()
        print(f"  错误 {input_path.name}: {e}")
        return False


def main():
    print("=" * 70)
    print("GloRiSe annually_climatology: 变量统一 + 质量控制 (apply_quality_flag, calculate_ssc)")
    print("=" * 70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    nc_files = sorted([f for f in INPUT_DIR.glob('*.nc') if f.is_file()])
    if not nc_files:
        print("未找到任何 .nc 文件。")
        return

    print(f"共 {len(nc_files)} 个 nc 文件。\n")
    ok = 0
    for fp in nc_files:
        if process_one_file(fp):
            ok += 1
    print("\n" + "=" * 70)
    print(f"完成: 成功 {ok}/{len(nc_files)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
