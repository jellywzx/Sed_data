#!/usr/bin/env python3
"""
将 Milliman NetCDF 文件的单位转换为日值

转换内容：
1. Discharge: km³/yr → m³/s (通过计算转换)
2. TSS: Mt/yr → ton/day (通过计算转换)

转换公式：
- Discharge (m³/s) = Discharge (km³/yr) × 10⁹ / (365.25 × 24 × 3600)
- TSS (ton/day) = TSS (Mt/yr) × 10⁶ / 365.25

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


def convert_single_file(filepath):
    """
    转换单个 NetCDF 文件的单位

    Returns:
        dict: 处理结果
    """
    try:
        ds = nc.Dataset(filepath, 'a')  # 追加模式

        result = {
            'file': os.path.basename(filepath),
            'status': 'no_change',
            'discharge_converted': False,
            'tss_converted': False,
            'discharge_old': None,
            'discharge_new': None,
            'tss_old': None,
            'tss_new': None
        }

        # 1. 转换 Discharge: km³/yr → m³/s
        if 'Discharge' in ds.variables:
            discharge_var = ds.variables['Discharge']

            # 检查当前单位
            if hasattr(discharge_var, 'units'):
                current_units = discharge_var.units

                # 如果是 km3 yr-1 或 m3 s-1，需要处理
                if 'km3' in current_units.lower() or 'km³' in current_units:
                    # 读取数据（km³/yr）
                    discharge_data = discharge_var[:]

                    # 保存原始值（用于报告）
                    valid_mask = discharge_data != -9999.0
                    if np.any(valid_mask):
                        result['discharge_old'] = float(discharge_data[valid_mask][0])

                    # 转换：km³/yr → m³/s
                    # 1 km³ = 10⁹ m³
                    # 1 year = 365.25 × 24 × 3600 seconds = 31,557,600 s
                    seconds_per_year = 365.25 * 24 * 3600

                    # 只转换有效数据（不是填充值）
                    discharge_data_new = discharge_data.copy()
                    discharge_data_new[valid_mask] = discharge_data[valid_mask] * 1e9 / seconds_per_year

                    # 更新数据
                    discharge_var[:] = discharge_data_new

                    # 更新属性
                    discharge_var.units = 'm3 s-1'
                    discharge_var.long_name = 'River discharge'
                    discharge_var.standard_name = 'water_volume_transport_in_river_channel'

                    result['discharge_converted'] = True
                    result['status'] = 'converted'
                    if np.any(valid_mask):
                        result['discharge_new'] = float(discharge_data_new[valid_mask][0])

                elif 'm3 s-1' in current_units or 'm³/s' in current_units:
                    # 已经是 m³/s，但需要检查数值是否正确
                    # 如果数值很大（>100000），说明可能还是 km³/yr 但单位标记错了
                    discharge_data = discharge_var[:]
                    valid_mask = discharge_data != -9999.0

                    if np.any(valid_mask):
                        max_val = np.max(discharge_data[valid_mask])

                        # Amazon 河的 Q 约 200,000 m³/s
                        # 如果数值 < 50,000，很可能是 km³/yr 被错误标记为 m³/s
                        if max_val < 50000:
                            result['discharge_old'] = float(discharge_data[valid_mask][0])

                            # 转换
                            seconds_per_year = 365.25 * 24 * 3600
                            discharge_data_new = discharge_data.copy()
                            discharge_data_new[valid_mask] = discharge_data[valid_mask] * 1e9 / seconds_per_year

                            discharge_var[:] = discharge_data_new

                            result['discharge_converted'] = True
                            result['status'] = 'converted'
                            result['discharge_new'] = float(discharge_data_new[valid_mask][0])

        # 2. 转换 TSS: Mt/yr → ton/day
        if 'TSS' in ds.variables:
            tss_var = ds.variables['TSS']

            if hasattr(tss_var, 'units'):
                current_units = tss_var.units

                # 如果是 Mt yr-1
                if 'Mt' in current_units and 'yr' in current_units:
                    # 读取数据（Mt/yr）
                    tss_data = tss_var[:]

                    # 保存原始值
                    valid_mask = tss_data != -9999.0
                    if np.any(valid_mask):
                        result['tss_old'] = float(tss_data[valid_mask][0])

                    # 转换：Mt/yr → ton/day
                    # 1 Mt = 10⁶ ton
                    # 1 year = 365.25 days
                    days_per_year = 365.25

                    tss_data_new = tss_data.copy()
                    tss_data_new[valid_mask] = tss_data[valid_mask] * 1e6 / days_per_year

                    # 更新数据
                    tss_var[:] = tss_data_new

                    # 更新属性
                    tss_var.units = 'ton day-1'
                    tss_var.long_name = 'Total Suspended Sediment flux'
                    tss_var.standard_name = 'sediment_flux'

                    result['tss_converted'] = True
                    result['status'] = 'converted'
                    if np.any(valid_mask):
                        result['tss_new'] = float(tss_data_new[valid_mask][0])

        # 添加历史记录
        if result['status'] == 'converted':
            changes = []
            if result['discharge_converted']:
                changes.append('Discharge: km³/yr → m³/s')
            if result['tss_converted']:
                changes.append('TSS: Mt/yr → ton/day')

            history_entry = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Unit conversion applied - {', '.join(changes)}"

            if hasattr(ds, 'history'):
                ds.history = ds.history + history_entry
            else:
                ds.history = history_entry

        ds.close()
        return result

    except Exception as e:
        return {
            'file': os.path.basename(filepath),
            'status': 'error',
            'error': str(e)
        }


def main():
    """主函数：批量转换所有 Milliman NetCDF 文件"""

    print("=" * 70)
    print("转换 Milliman NetCDF 文件单位")
    print("=" * 70)
    print("\n转换内容:")
    print("  1. Discharge: km³/yr → m³/s")
    print("  2. TSS: Mt/yr → ton/day")
    print()

    # 设置路径
    netcdf_dir = require_existing_directory(
        resolve_source_root(start=__file__) / "Milliman" / "netcdf_output",
        description="Milliman intermediate NetCDF directory",
    )

    # 查找所有 Milliman 文件
    milliman_files = glob.glob(os.path.join(netcdf_dir, "Milliman_*.nc"))

    print(f"找到 {len(milliman_files)} 个 Milliman NetCDF 文件")

    if len(milliman_files) == 0:
        print("错误：未找到文件！")
        return

    # 统计变量
    converted_count = 0
    no_change_count = 0
    error_count = 0

    discharge_converted = 0
    tss_converted = 0

    results = []

    print("\n开始转换单位...")
    print("-" * 70)

    # 处理每个文件
    for i, filepath in enumerate(milliman_files):
        result = convert_single_file(filepath)
        results.append(result)

        if result['status'] == 'converted':
            converted_count += 1
            if result['discharge_converted']:
                discharge_converted += 1
            if result['tss_converted']:
                tss_converted += 1

            # 显示前5个转换结果
            if converted_count <= 5:
                print(f"\n✓ 已转换: {result['file']}")
                if result['discharge_converted']:
                    print(f"  Discharge: {result['discharge_old']:.2f} km³/yr → {result['discharge_new']:.2f} m³/s")
                if result['tss_converted']:
                    print(f"  TSS: {result['tss_old']:.4f} Mt/yr → {result['tss_new']:.2f} ton/day")

        elif result['status'] == 'no_change':
            no_change_count += 1

        elif result['status'] == 'error':
            error_count += 1
            print(f"✗ 错误: {result['file']} - {result['error']}")

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"\n  进度: {i+1}/{len(milliman_files)} 文件处理完成...")

    print("\n" + "-" * 70)
    print("转换完成！")
    print("=" * 70)
    print(f"统计结果:")
    print(f"  成功转换:      {converted_count} 文件")
    print(f"    - Discharge: {discharge_converted} 文件")
    print(f"    - TSS:       {tss_converted} 文件")
    print(f"  无需转换:      {no_change_count} 文件")
    print(f"  错误:          {error_count} 文件")
    print(f"  总计:          {len(milliman_files)} 文件")
    print("=" * 70)

    # 验证转换（抽样检查）
    print("\n验证转换结果（抽样检查前5个文件）...")
    print("-" * 70)

    converted_files = [r for r in results if r['status'] == 'converted']
    sample_results = converted_files[:5]

    for result in sample_results:
        print(f"\n✓ {result['file']}")
        if result['discharge_converted']:
            print(f"  Discharge: {result['discharge_old']:.2f} km³/yr → {result['discharge_new']:.2f} m³/s")
            # 验证转换
            expected = result['discharge_old'] * 1e9 / (365.25 * 24 * 3600)
            print(f"  验证: {expected:.2f} m³/s (误差: {abs(result['discharge_new'] - expected):.6f})")
        if result['tss_converted']:
            print(f"  TSS: {result['tss_old']:.4f} Mt/yr → {result['tss_new']:.2f} ton/day")
            # 验证转换
            expected = result['tss_old'] * 1e6 / 365.25
            print(f"  验证: {expected:.2f} ton/day (误差: {abs(result['tss_new'] - expected):.6f})")

    # 详细示例
    print("\n" + "=" * 70)
    print("详细示例验证 (Amazon 河):")
    print("=" * 70)

    # 查找 Amazon 文件
    amazon_file = [f for f in milliman_files if 'Amazon_MILLIMAN-1375' in f]
    if amazon_file:
        ds = nc.Dataset(amazon_file[0], 'r')
        print(f"\n文件: {os.path.basename(amazon_file[0])}")

        if 'Discharge' in ds.variables:
            discharge = ds.variables['Discharge'][0, 0, 0]
            discharge_units = ds.variables['Discharge'].units
            print(f"\nDischarge: {discharge:,.2f} {discharge_units}")
            print(f"  转换验证: 6300 km³/yr × 10⁹ / 31,557,600 s = {6300 * 1e9 / 31557600:,.2f} m³/s")

        if 'TSS' in ds.variables:
            tss = ds.variables['TSS'][0, 0, 0]
            tss_units = ds.variables['TSS'].units
            print(f"\nTSS: {tss:,.2f} {tss_units}")
            print(f"  转换验证: 1200 Mt/yr × 10⁶ / 365.25 = {1200 * 1e6 / 365.25:,.2f} ton/day")

        print(f"\n历史记录:")
        if hasattr(ds, 'history'):
            for line in ds.history.split('\n'):
                if line.strip():
                    print(f"  • {line}")

        ds.close()

    print("\n" + "=" * 70)
    print("所有单位转换完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
