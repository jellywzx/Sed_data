#!/usr/bin/env python3
"""
将全局属性添加为独立变量到 Milliman NetCDF 文件

将以下全局属性转换为变量：
- sediment_concentration_mg_L → SSC (mg/L)
- drainage_area_km2 → drainage_area (km²)

这样可以更方便地使用标准工具访问这些数据

Author: Claude Code
Date: 2025-10-19
"""

import netCDF4 as nc
import numpy as np
import glob
import os
from datetime import datetime
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = CURRENT_DIR.parent
CODE_DIR = SCRIPT_ROOT / 'code'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from runtime import resolve_source_root
from validation import require_existing_directory


def add_variables_to_file(filepath):
    """
    向单个 NetCDF 文件添加变量

    Returns:
        dict: 处理状态信息
    """
    try:
        # 以追加模式打开文件
        ds = nc.Dataset(filepath, 'a')

        # 检查是否已经有这些变量（避免重复添加）
        has_ssc = 'SSC' in ds.variables
        has_drainage = 'drainage_area' in ds.variables

        added_vars = []

        # 1. 添加 SSC (Suspended Sediment Concentration) 变量
        if not has_ssc and 'sediment_concentration_mg_L' in ds.ncattrs():
            ssc_value_str = ds.sediment_concentration_mg_L

            # 处理可能的 'N/A' 或空值
            if ssc_value_str and ssc_value_str != 'N/A':
                try:
                    ssc_value = float(ssc_value_str)

                    # 创建 SSC 变量
                    ssc_var = ds.createVariable('SSC', 'f4', ('time', 'latitude', 'longitude'),
                                                fill_value=-9999.0)
                    ssc_var.long_name = 'Suspended Sediment Concentration'
                    ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                    ssc_var.units = 'mg L-1'
                    ssc_var.coordinates = 'time latitude longitude'
                    ssc_var.description = 'Long-term average suspended sediment concentration'

                    # 填充数据
                    n_times = len(ds.dimensions['time'])
                    ssc_data = np.full((n_times, 1, 1), ssc_value, dtype=np.float32)
                    ssc_var[:] = ssc_data

                    added_vars.append('SSC')
                except ValueError:
                    pass  # 无法转换为浮点数，跳过

        # 2. 添加 drainage_area 变量
        if not has_drainage and 'drainage_area_km2' in ds.ncattrs():
            area_value_str = ds.drainage_area_km2

            # 处理可能的 'N/A' 或空值
            if area_value_str and area_value_str != 'N/A':
                try:
                    area_value = float(area_value_str)

                    # 创建 drainage_area 变量（标量变量）
                    area_var = ds.createVariable('drainage_area', 'f4', ())
                    area_var.long_name = 'Drainage basin area'
                    area_var.standard_name = 'drainage_basin_area'
                    area_var.units = 'km2'
                    area_var.description = 'Total drainage basin area upstream of the station'

                    # 填充数据
                    area_var[:] = area_value

                    added_vars.append('drainage_area')
                except ValueError:
                    pass  # 无法转换为浮点数，跳过

        # 添加处理历史
        if added_vars:
            history_entry = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Added variables from global attributes: {', '.join(added_vars)}"

            if hasattr(ds, 'history'):
                ds.history = ds.history + history_entry
            else:
                ds.history = history_entry

        ds.close()

        return {
            'status': 'success' if added_vars else 'no_action',
            'file': os.path.basename(filepath),
            'added_vars': added_vars
        }

    except Exception as e:
        return {
            'status': 'error',
            'file': os.path.basename(filepath),
            'error': str(e)
        }


def verify_variables(filepath):
    """
    验证文件中的变量
    """
    ds = nc.Dataset(filepath, 'r')

    result = {
        'file': os.path.basename(filepath),
        'variables': list(ds.variables.keys()),
        'has_SSC': 'SSC' in ds.variables,
        'has_drainage_area': 'drainage_area' in ds.variables
    }

    if result['has_SSC']:
        result['SSC_value'] = ds.variables['SSC'][0,0,0] if ds.variables['SSC'].ndim == 3 else ds.variables['SSC'][:]
        result['SSC_units'] = ds.variables['SSC'].units

    if result['has_drainage_area']:
        result['drainage_area_value'] = float(ds.variables['drainage_area'][:])
        result['drainage_area_units'] = ds.variables['drainage_area'].units

    ds.close()
    return result


def main():
    """主函数：批量添加变量到所有 Milliman NetCDF 文件"""

    print("="*70)
    print("向 Milliman NetCDF 文件添加变量")
    print("="*70)

    # 设置路径
    netcdf_dir = require_existing_directory(
        resolve_source_root(start=__file__) / "Milliman" / "netcdf_output",
        description="Milliman intermediate NetCDF directory",
    )

    # 查找所有 Milliman 文件
    milliman_files = glob.glob(os.path.join(netcdf_dir, "Milliman_*.nc"))

    print(f"\n找到 {len(milliman_files)} 个 Milliman NetCDF 文件")

    if len(milliman_files) == 0:
        print("错误：未找到文件！")
        return

    # 统计变量
    success_count = 0
    no_action_count = 0
    error_count = 0

    ssc_added = 0
    drainage_added = 0

    results = []

    print("\n开始添加变量...")
    print("-" * 70)

    # 处理每个文件
    for i, filepath in enumerate(milliman_files):
        result = add_variables_to_file(filepath)
        results.append(result)

        if result['status'] == 'success':
            success_count += 1
            if 'SSC' in result['added_vars']:
                ssc_added += 1
            if 'drainage_area' in result['added_vars']:
                drainage_added += 1

            if success_count <= 5:  # 只显示前5个
                print(f"✓ 已添加: {result['file']}")
                print(f"  变量: {', '.join(result['added_vars'])}")
        elif result['status'] == 'no_action':
            no_action_count += 1
        elif result['status'] == 'error':
            error_count += 1
            print(f"✗ 错误: {result['file']} - {result['error']}")

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(milliman_files)} 文件处理完成...")

    print("-" * 70)
    print("\n添加完成！")
    print("="*70)
    print(f"统计结果:")
    print(f"  成功添加变量:  {success_count} 文件")
    print(f"    - SSC:           {ssc_added} 文件")
    print(f"    - drainage_area: {drainage_added} 文件")
    print(f"  无需操作:      {no_action_count} 文件")
    print(f"  错误:          {error_count} 文件")
    print(f"  总计:          {len(milliman_files)} 文件")
    print("="*70)

    # 验证添加（抽样检查）
    print("\n验证添加结果（抽样检查前5个已添加变量的文件）...")
    print("-" * 70)

    success_files = [r for r in results if r['status'] == 'success']
    sample_files = [milliman_files[results.index(r)] for r in success_files[:5]]

    for filepath in sample_files:
        verify_result = verify_variables(filepath)
        print(f"\n✓ {verify_result['file']}")
        print(f"  变量列表: {', '.join(verify_result['variables'])}")
        if verify_result['has_SSC']:
            print(f"  SSC: {verify_result['SSC_value']:.2f} {verify_result['SSC_units']}")
        if verify_result['has_drainage_area']:
            print(f"  drainage_area: {verify_result['drainage_area_value']:.2f} {verify_result['drainage_area_units']}")

    # 显示详细示例
    print("\n" + "="*70)
    print("详细示例 (Agri 河):")
    print("="*70)

    # 查找 Agri 文件
    agri_file = [f for f in milliman_files if 'Agri_MILLIMAN-705' in f]
    if agri_file:
        ds = nc.Dataset(agri_file[0], 'r')
        print(f"\n文件: {os.path.basename(agri_file[0])}")

        print(f"\n所有变量:")
        for var_name in ds.variables.keys():
            var = ds.variables[var_name]
            print(f"  {var_name}: {var.dimensions}, units={var.units if hasattr(var, 'units') else 'N/A'}")

        print(f"\n数据值:")
        print(f"  TSS (泥沙通量):     {ds.variables['TSS'][0,0,0]:.4f} {ds.variables['TSS'].units}")
        if 'SSC' in ds.variables:
            if ds.variables['SSC'].ndim == 3:
                print(f"  SSC (泥沙浓度):     {ds.variables['SSC'][0,0,0]:.2f} {ds.variables['SSC'].units}")
            else:
                print(f"  SSC (泥沙浓度):     {ds.variables['SSC'][:]:.2f} {ds.variables['SSC'].units}")
        if 'drainage_area' in ds.variables:
            print(f"  drainage_area (流域面积): {ds.variables['drainage_area'][:]:.2f} {ds.variables['drainage_area'].units}")
        if 'Discharge' in ds.variables:
            print(f"  Discharge (径流):   {ds.variables['Discharge'][0,0,0]:.2f} {ds.variables['Discharge'].units}")

        ds.close()

    print("\n" + "="*70)
    print("所有变量添加完成！")
    print("="*70)


if __name__ == "__main__":
    main()
