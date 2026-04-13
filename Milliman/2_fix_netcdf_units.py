#!/usr/bin/env python3
"""
修正 Milliman NetCDF 文件中的 TSS 单位标记错误

问题：TSS 变量实际是年泥沙通量 (Mt/yr)，但被错误标记为浓度 (mg/L)
解决：批量修正所有 Milliman_*.nc 文件的 TSS 变量元数据

Author: Claude Code
Date: 2025-10-19
"""

import netCDF4 as nc
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


def fix_single_file(filepath):
    """
    修正单个 NetCDF 文件的 TSS 单位

    Returns:
        dict: 修正状态信息
    """
    try:
        # 以追加模式打开文件
        ds = nc.Dataset(filepath, 'a')

        # 获取当前信息
        tss_var = ds.variables['TSS']
        old_units = tss_var.units
        old_long_name = tss_var.long_name
        old_standard_name = tss_var.standard_name
        tss_value = tss_var[0, 0, 0]

        # 判断是否需要修正（只修正被错误标记为浓度的文件）
        needs_fix = False
        if old_units == 'mg L-1':
            # Milliman 文件的 TSS 都应该是通量 (Mt/yr)
            needs_fix = True

        if needs_fix:
            # 修正单位和描述
            tss_var.units = 'Mt yr-1'
            tss_var.long_name = 'Total Suspended Sediment flux'
            tss_var.standard_name = 'sediment_flux'

            # 添加修正历史记录
            history_entry = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Corrected TSS units from 'mg L-1' to 'Mt yr-1' (TSS represents annual sediment flux, not concentration)"

            if hasattr(ds, 'history'):
                ds.history = ds.history + history_entry
            else:
                ds.history = f"Created on {ds.history if hasattr(ds, 'history') else 'unknown'}" + history_entry

            ds.close()

            return {
                'status': 'fixed',
                'file': os.path.basename(filepath),
                'old_units': old_units,
                'new_units': 'Mt yr-1',
                'tss_value': tss_value
            }
        else:
            ds.close()
            return {
                'status': 'no_fix_needed',
                'file': os.path.basename(filepath),
                'units': old_units,
                'tss_value': tss_value
            }

    except Exception as e:
        return {
            'status': 'error',
            'file': os.path.basename(filepath),
            'error': str(e)
        }


def verify_fix(filepath):
    """
    验证文件是否已正确修复
    """
    ds = nc.Dataset(filepath, 'r')
    tss_var = ds.variables['TSS']

    result = {
        'file': os.path.basename(filepath),
        'units': tss_var.units,
        'long_name': tss_var.long_name,
        'standard_name': tss_var.standard_name,
        'value': tss_var[0, 0, 0],
        'correct': tss_var.units == 'Mt yr-1'
    }

    ds.close()
    return result


def main():
    """主函数：批量修正所有 Milliman NetCDF 文件"""

    print("="*70)
    print("修正 Milliman NetCDF 文件 TSS 单位标记")
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
    fixed_count = 0
    no_fix_needed_count = 0
    error_count = 0

    results = []

    print("\n开始修正...")
    print("-" * 70)

    # 处理每个文件
    for i, filepath in enumerate(milliman_files):
        result = fix_single_file(filepath)
        results.append(result)

        if result['status'] == 'fixed':
            fixed_count += 1
            if fixed_count <= 5:  # 只显示前5个
                print(f"✓ 已修正: {result['file']}")
                print(f"  {result['old_units']} → {result['new_units']} (值: {result['tss_value']:.4f})")
        elif result['status'] == 'no_fix_needed':
            no_fix_needed_count += 1
        elif result['status'] == 'error':
            error_count += 1
            print(f"✗ 错误: {result['file']} - {result['error']}")

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(milliman_files)} 文件处理完成...")

    print("-" * 70)
    print("\n修正完成！")
    print("="*70)
    print(f"统计结果:")
    print(f"  已修正:        {fixed_count} 文件")
    print(f"  无需修正:      {no_fix_needed_count} 文件")
    print(f"  错误:          {error_count} 文件")
    print(f"  总计:          {len(milliman_files)} 文件")
    print("="*70)

    # 验证修正（抽样检查）
    print("\n验证修正结果（抽样检查前10个已修正的文件）...")
    print("-" * 70)

    fixed_files = [r for r in results if r['status'] == 'fixed']
    sample_files = [milliman_files[results.index(r)] for r in fixed_files[:10]]

    all_correct = True
    for filepath in sample_files:
        verify_result = verify_fix(filepath)
        status = "✓" if verify_result['correct'] else "✗"
        print(f"{status} {verify_result['file']}: {verify_result['units']}")
        if not verify_result['correct']:
            all_correct = False

    print("-" * 70)

    if all_correct:
        print("\n✓ 验证通过！所有抽样文件的单位都已正确修正为 'Mt yr-1'")
    else:
        print("\n✗ 验证失败！部分文件修正不成功")

    # 显示修正示例
    if fixed_count > 0:
        print("\n" + "="*70)
        print("修正示例 (Agri 河):")
        print("="*70)

        # 查找 Agri 文件
        agri_file = [f for f in milliman_files if 'Agri_MILLIMAN-705' in f]
        if agri_file:
            ds = nc.Dataset(agri_file[0], 'r')
            print(f"\n文件: {os.path.basename(agri_file[0])}")
            print(f"\nTSS 变量:")
            print(f"  值:          {ds.variables['TSS'][0,0,0]:.4f}")
            print(f"  单位:        {ds.variables['TSS'].units}")
            print(f"  长名称:      {ds.variables['TSS'].long_name}")
            print(f"  标准名称:    {ds.variables['TSS'].standard_name}")
            print(f"\n全局属性:")
            print(f"  泥沙浓度:    {ds.sediment_concentration_mg_L} mg/L")
            print(f"  径流量:      {ds.variables['Discharge'][0,0,0]:.2f} m3/s")
            ds.close()

    print("\n" + "="*70)
    print("所有文件已修正完成！")
    print("="*70)


if __name__ == "__main__":
    main()
