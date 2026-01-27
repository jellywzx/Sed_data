#!/usr/bin/env python3
"""
HYDAT数据集全面质量控制和CF-1.8标准化处理脚本

功能包括:
1. 数据内容验证与质量标志 (Data Validation & Flagging)
2. 元数据标准化 (CF-1.8 Compliant Metadata)
3. 物理规律检查与标记
4. 时间截取和无效站点删除
5. 数据溯源信息追加

作者: Zhongwang Wei
日期: 2025-10-26
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) #得到当前脚本所在目录
#设置相对路径第一步是利用dirname得到当前脚本所在目录，然后利用os.path.join()函数和获取其他目录的路径。
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..')) 
if PARENT_DIR not in sys.path: #sys.path：这是一个列表，存储了 Python 查找模块 / 包的路径。
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    check_nc_completeness,
    check_variable_metadata_tiered,
    # add_global_attributes,
    propagate_ssc_q_inconsistency_to_ssl,
    apply_hydro_qc_with_provenance,
    summarize_warning_types as summarize_warning_types_tool,
    generate_csv_summary as generate_csv_summary_tool,
    generate_qc_results_csv as generate_qc_results_csv_tool,
    generate_warning_summary_csv as generate_warning_summary_csv_tool,
)

def apply_tool_qc(
    time,
    Q,
    SSC,
    SSL,
    station_id,
    station_name,
    plot_dir=None,
):
    """
    Apply QC using tool.py end-to-end pipeline WITH step-level provenance flags.
    Also fixes valid-time logic using value-based missing detection.
    """

    # 调用 tool.py 的通用 QC（QC1/QC2/QC3 + provenance）
    # new
    qc = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )

    if qc is None:
        return None

    # =============================
    # ✅ 修正 valid time 逻辑（关键）
    # =============================
    # 不仅看 flag，还要用“值是否为 NaN / fill”来判定是否缺失
    def _present(v, f):
        v = np.asarray(v, dtype=float)
        f = np.asarray(f, dtype=np.int8)
        return (
            (f != FILL_VALUE_INT)  # final flag 不是 9
            & np.isfinite(v)
            & (~np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5))
        )

    present_Q   = _present(qc["Q"], qc["Q_flag"])
    present_SSC = _present(qc["SSC"], qc["SSC_flag"])
    present_SSL = _present(qc["SSL"], qc["SSL_flag"])

    valid_time = present_Q | present_SSC | present_SSL
    if not np.any(valid_time):
        return None

    # 按修正后的 valid_time 再裁一遍（包括分步 flags）
    for k in list(qc.keys()):
        if isinstance(qc[k], np.ndarray) and len(qc[k]) == len(valid_time):
            qc[k] = qc[k][valid_time]

    # =============================
    # 诊断图（可选）
    # =============================
    if plot_dir is not None and qc.get("ssc_q_bounds") is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_ssc_q_diagnostic(
            time=pd.to_datetime(qc["time"], unit="D", origin="1970-01-01"),
            Q=qc["Q"],
            SSC=qc["SSC"],
            Q_flag=qc["Q_flag"],
            SSC_flag=qc["SSC_flag"],
            ssc_q_bounds=qc["ssc_q_bounds"],
            station_id=station_id,
            station_name=station_name,
            out_png=plot_dir / f"{station_id}_ssc_q.png",
        )

    return qc

class HYDATQualityControl: 
    """HYDAT批量数据质量控制和标准化处理类"""

    def __init__(self, input_dir, output_dir):
        """
        初始化

        Parameters:
        -----------
        input_dir : str or Path
            输入NetCDF文件目录
        output_dir : str or Path
            输出NetCDF文件目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 物理阈值设置
        # self.Q_extreme_high = 100000.0  # m3/s - 极端高值
        # self.SSC_extreme_high = 3000.0  # mg/L - 极端高值
        # self.SSC_min = 0.1  # mg/L - 最小物理有效值

        # 统计信息
        self.stats = {
            'total_stations': 0,
            'processed_stations': 0,
            'removed_stations': 0,
            'stations_info': []
        }

    def _count_flags(self, f):
        f = np.asarray(f, dtype=np.int8)
        return {
            "good": int(np.sum(f == 0)),
            "estimated": int(np.sum(f == 1)),
            "suspect": int(np.sum(f == 2)),
            "bad": int(np.sum(f == 3)),
            "missing": int(np.sum(f == FILL_VALUE_INT)),
        }


    def calculate_completeness(self, data_array, flag_array, start_date, end_date):
        """
        计算数据完整性（Good data的百分比）

        Parameters:
        -----------
        data_array : numpy.ndarray
            数据数组
        flag_array : numpy.ndarray
            质量标志数组
        start_date, end_date : pd.Timestamp
            起止日期

        Returns:
        --------
        percent_complete : float
            完整性百分比
        """
        # 计算时间范围内的总天数
        total_days = (end_date - start_date).days + 1

        # 计算Good data的天数（flag == 0）
        good_data_count = np.sum(flag_array == 0)

        if total_days > 0:
            return (good_data_count / total_days) * 100.0
        else:
            return 0.0
        
    def _count_final_flags(self, f):
        f = np.asarray(f, dtype=np.int8)
        return {
            "good": int(np.sum(f == 0)),
            "estimated": int(np.sum(f == 1)),
            "suspect": int(np.sum(f == 2)),
            "bad": int(np.sum(f == 3)),
            "missing": int(np.sum(f == FILL_VALUE_INT)),  # 9
        }

    def _count_step_flags(self, f, mapping: dict):
        """
        mapping: { "col_suffix": flag_value }
        e.g. {"pass":0, "suspect":2, "not_checked":8, "missing":9}
        """
        f = np.asarray(f, dtype=np.int8)
        out = {}
        for name, val in mapping.items():
            out[name] = int(np.sum(f == np.int8(val)))
        return out


    def process_station(self, input_file): #处理单个站点文件
        """
        处理单个站点文件

        Parameters:
        -----------
        input_file : Path
            输入NetCDF文件路径

        Returns:
        --------
        success : bool
            是否处理成功
        station_info : dict
            站点信息字典
        """
        print(f"处理站点: {input_file.name}")

        try:
            with nc.Dataset(input_file, 'r') as ds_in:
                # 读取基本信息
                station_id = ds_in.station_id if hasattr(ds_in, 'station_id') else ''
                station_name = ds_in.station_name if hasattr(ds_in, 'station_name') else ''
                province = ds_in.province_territory if hasattr(ds_in, 'province_territory') else ''

                # 读取坐标 (兼容不同的变量名)
                if 'latitude' in ds_in.variables:
                    lat = float(ds_in.variables['latitude'][:])
                elif 'lat' in ds_in.variables:
                    lat = float(ds_in.variables['lat'][:])
                else:
                    raise ValueError("Cannot find latitude variable")

                if 'longitude' in ds_in.variables:
                    lon = float(ds_in.variables['longitude'][:])
                elif 'lon' in ds_in.variables:
                    lon = float(ds_in.variables['lon'][:])
                else:
                    raise ValueError("Cannot find longitude variable")

                # 读取其他标量
                altitude = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else -9999.0
                upstream_area = float(ds_in.variables['upstream_area'][:]) if 'upstream_area' in ds_in.variables else -9999.0

                # 读取时间序列数据
                time = ds_in.variables['time'][:]

                # 读取 Q (discharge) - 兼容不同变量名
                if 'discharge' in ds_in.variables:
                    Q = ds_in.variables['discharge'][:]
                elif 'Q' in ds_in.variables:
                    Q = ds_in.variables['Q'][:]
                else:
                    raise ValueError("Cannot find discharge variable")

                # 读取 SSC - 兼容不同变量名
                if 'ssc' in ds_in.variables:
                    SSC = ds_in.variables['ssc'][:]
                elif 'SSC' in ds_in.variables:
                    SSC = ds_in.variables['SSC'][:]
                else:
                    raise ValueError("Cannot find SSC variable")

                # 读取 SSL - 兼容不同变量名
                if 'sediment_load' in ds_in.variables:
                    SSL = ds_in.variables['sediment_load'][:]
                elif 'SSL' in ds_in.variables:
                    SSL = ds_in.variables['SSL'][:]
                else:
                    raise ValueError("Cannot find sediment load variable")

                qc = apply_tool_qc(
                    time=time,
                    Q=Q,
                    SSC=SSC,
                    SSL=SSL,
                    station_id=station_id,
                    station_name=station_name,
                    plot_dir=self.output_dir / "diagnostic_plots",
                )

                if qc is None:
                    print(f"  ⚠ No valid data after QC, skip station {station_id}")
                    return False, None

                time = qc["time"] 
                Q = qc["Q"]
                SSC = qc["SSC"]
                SSL = qc["SSL"]
                Q_flag = qc["Q_flag"]
                SSC_flag = qc["SSC_flag"]
                SSL_flag = qc["SSL_flag"] #这段在返回的字典里面取出这些变量进行下面工作

                
                # 计算时间范围
                reference_date = pd.Timestamp('1970-01-01')
                start_date = reference_date + pd.Timedelta(days=float(time[0]))
                end_date = reference_date + pd.Timedelta(days=float(time[-1]))

                # 计算完整性
                Q_completeness = self.calculate_completeness(Q, Q_flag, start_date, end_date)
                SSC_completeness = self.calculate_completeness(SSC, SSC_flag, start_date, end_date)
                SSL_completeness = self.calculate_completeness(SSL, SSL_flag, start_date, end_date)

                # 创建输出文件
                output_file = self.output_dir / f"HYDAT_{station_id}.nc" #self.output_dir是在初始化类的时候定义的输出目录

                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    # 创建维度
                    ds_out.createDimension('time', len(time))

                    # 创建时间变量
                    var_time = ds_out.createVariable('time', 'f8', ('time',)) #f8表示64位浮点数
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.units = 'days since 1970-01-01 00:00:00'
                    var_time.calendar = 'gregorian'
                    var_time.axis = 'T' #T 表示时间轴
                    var_time[:] = time

                    # 创建坐标变量 (标量)
                    var_lat = ds_out.createVariable('lat', 'f4')#f4表示32位浮点数
                    var_lat.standard_name = 'latitude'
                    var_lat.long_name = 'station latitude'
                    var_lat.units = 'degrees_north'
                    var_lat.axis = 'Y'
                    var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
                    var_lat[:] = lat

                    var_lon = ds_out.createVariable('lon', 'f4')
                    var_lon.standard_name = 'longitude'
                    var_lon.long_name = 'station longitude'
                    var_lon.units = 'degrees_east'
                    var_lon.axis = 'X'
                    var_lon.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
                    var_lon[:] = lon

                    # 创建其他标量变量
                    var_alt = ds_out.createVariable('altitude', 'f4', fill_value=-9999.0)
                    var_alt.standard_name = 'altitude'
                    var_alt.long_name = 'station elevation above sea level'
                    var_alt.units = 'm'
                    var_alt.positive = 'up'
                    var_alt.comment = 'Source: HYDAT database.'
                    var_alt[:] = altitude

                    var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=-9999.0)
                    var_area.long_name = 'upstream drainage area'
                    var_area.units = 'km2'
                    var_area.comment = 'Source: HYDAT database.'
                    var_area[:] = upstream_area

                    # 创建数据变量 Q
                    var_Q = ds_out.createVariable('Q', 'f4', ('time',),
                                                   fill_value=-9999.0, zlib=True, complevel=4)
                    var_Q.standard_name = 'water_volume_transport_in_river_channel'
                    var_Q.long_name = 'river discharge'
                    var_Q.units = 'm3 s-1'
                    var_Q.coordinates = 'time lat lon'
                    var_Q.ancillary_variables = 'Q_flag'
                    var_Q.comment = 'Source: Original data from HYDAT database.'
                    var_Q[:] = Q

                    # Q质量标志
                    var_Q_flag = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_Q_flag.long_name = 'quality flag for river discharge'
                    var_Q_flag.standard_name = 'status_flag'
                    var_Q_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_Q_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_Q_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_Q_flag[:] = Q_flag

                    # 创建数据变量 SSC
                    var_SSC = ds_out.createVariable('SSC', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSC.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                    var_SSC.long_name = 'suspended sediment concentration'
                    var_SSC.units = 'mg L-1'
                    var_SSC.coordinates = 'time lat lon'
                    var_SSC.ancillary_variables = 'SSC_flag'
                    var_SSC.comment = 'Source: Original data from HYDAT database.'
                    var_SSC[:] = SSC

                    # SSC质量标志
                    var_SSC_flag = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSC_flag.long_name = 'quality flag for suspended sediment concentration'
                    var_SSC_flag.standard_name = 'status_flag'
                    var_SSC_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSC_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSC_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSC_flag[:] = SSC_flag

                    # 创建数据变量 SSL
                    var_SSL = ds_out.createVariable('SSL', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSL.long_name = 'suspended sediment load'
                    var_SSL.units = 'ton day-1'
                    var_SSL.coordinates = 'time lat lon'
                    var_SSL.ancillary_variables = 'SSL_flag'
                    var_SSL.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4, where 86.4 = 86400 s/day × 10⁻⁶ ton/mg × 1000 L/m³.'
                    var_SSL[:] = SSL

                    # SSL质量标志
                    var_SSL_flag = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSL_flag.long_name = 'quality flag for suspended sediment load'
                    var_SSL_flag.standard_name = 'status_flag'
                    var_SSL_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSL_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSL_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSL_flag[:] = SSL_flag

                    # 设置全局属性
                    ds_out.Conventions = 'CF-1.8, ACDD-1.3'
                    ds_out.title = 'Harmonized Global River Discharge and Sediment'
                    ds_out.summary = f'River discharge and suspended sediment data for station {station_name} (ID: {station_id}) from the HYDAT database (Water Survey of Canada). This dataset contains daily observations of discharge, suspended sediment concentration, and sediment load with quality control flags.'
                    ds_out.source = 'In-situ station data'
                    ds_out.data_source_name = 'HYDAT Dataset'
                    ds_out.station_name = station_name
                    river_name = station_name.split(' AT ')[0] if ' AT ' in station_name else station_name.split(' NEAR ')[0] if ' NEAR ' in station_name else ''
                    ds_out.river_name = river_name
                    ds_out.location_id = station_id
                    ds_out.type = 'In-situ station data'
                    ds_out.temporal_resolution = 'daily'
                    ds_out.temporal_span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    ds_out.variables_provided = 'altitude, upstream_area, Q, SSC, SSL'
                    ds_out.geographic_coverage = f"{province}, Canada"
                    ds_out.country = 'Canada'
                    ds_out.continent_region = 'North America'
                    ds_out.time_coverage_start = start_date.strftime('%Y-%m-%d')
                    ds_out.time_coverage_end = end_date.strftime('%Y-%m-%d')
                    ds_out.number_of_data = '1'
                    ds_out.reference = 'HYDAT - Canadian Hydrometric Database, Water Survey of Canada'
                    ds_out.source_data_link = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
                    ds_out.creator_name = 'Zhongwang Wei'
                    ds_out.creator_email = 'weizhw6@mail.sysu.edu.cn'
                    ds_out.creator_institution = 'Sun Yat-sen University, China'
                    ds_out.geospatial_lat_min = lat
                    ds_out.geospatial_lat_max = lat
                    ds_out.geospatial_lon_min = lon
                    ds_out.geospatial_lon_max = lon

                    # 数据溯源历史记录
                    history_entry = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"Converted from HYDAT database to CF-1.8 compliant NetCDF format. "
                        f"Applied quality control checks including physical constraint validation "
                        f"(Q range check, SSC range check, SSL negative check). "
                        f"Trimmed data to valid time range from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}. "
                        f"Script: process_hydat_cf18.py"
                    )
                    ds_out.history = history_entry
                    ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
                    ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
                    ds_out.processing_level = 'Quality controlled and standardized'
                    ds_out.comment = (
                        "Quality flags: 0=good, 1=estimated (derived), 2=suspect, 3=bad, 9=missing. "
                        "QC1: physical feasibility; QC2: log-IQR screening (independent variables only); "
                        "QC3: SSC–Q consistency and propagation to derived SSL."
                    )

                # ==========================================================
                # NetCDF completeness check (CF-1.8 / ACDD-1.3)
                # ==========================================================

                errors, warnings = check_nc_completeness(
                    output_file,
                    strict=False   # ← 建议先用 False
                )
                var_errs, var_warns = check_variable_metadata_tiered(output_file, tier="recommended")
                errors.extend(var_errs)
                warnings.extend(var_warns)

                if errors:
                    print(f"  ✗ NetCDF completeness check FAILED for {station_id}")
                    for e in errors:
                        print(f"    ERROR: {e}")

                    # 删除不合格文件（强一致性）
                    try:
                        output_file.unlink()
                        print(f"    → Invalid NetCDF removed: {output_file.name}")
                    except Exception:
                        pass

                    return False, None

                if warnings:
                    print(f"  ⚠ NetCDF completeness warnings for {station_id}: {len(warnings)}")
                    for w in warnings:
                        print(f"    WARNING: {w}")

                    # 把 warning 写入 history（非常加分）
                    with nc.Dataset(output_file, "a") as ds_out:
                        ds_out.history = (
                            ds_out.history
                            + f"; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                            + f"Completeness check warnings ({len(warnings)}): "
                            + "; ".join(warnings[:3])
                        )

                station_warnings = warnings.copy() if warnings else []

                # 收集站点信息用于CSV
                station_info = {
                    'station_name': station_name,
                    'Source_ID': station_id,
                    'river_name': river_name,
                    'longitude': lon,
                    'latitude': lat,
                    'altitude': altitude if altitude != -9999.0 else np.nan,
                    'upstream_area': upstream_area if upstream_area != -9999.0 else np.nan,
                    'Data Source Name': 'HYDAT Dataset',
                    'Type': 'In-situ',
                    'Temporal Resolution': 'daily',
                    'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'Variables Provided': 'Q, SSC, SSL',
                    'Geographic Coverage': f"{province}, Canada",
                    'Reference/DOI': 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html',
                    'Q_start_date': start_date.year,
                    'Q_end_date': end_date.year,
                    'Q_percent_complete': round(Q_completeness, 2),
                    'SSC_start_date': start_date.year,
                    'SSC_end_date': end_date.year,
                    'SSC_percent_complete': round(SSC_completeness, 2),
                    'SSL_start_date': start_date.year,
                    'SSL_end_date': end_date.year,
                    'SSL_percent_complete': round(SSL_completeness, 2)
                }
                # ==========================================================
                # ✅ QC统计（最终 + 分步）
                # ==========================================================
                station_info["QC_n_days"] = int(len(time))

                # ---- Final flags count ----
                q_cnt   = self._count_final_flags(Q_flag)
                ssc_cnt = self._count_final_flags(SSC_flag)
                ssl_cnt = self._count_final_flags(SSL_flag)

                station_info.update({
                    "Q_final_good": q_cnt["good"],
                    "Q_final_estimated": q_cnt["estimated"],
                    "Q_final_suspect": q_cnt["suspect"],
                    "Q_final_bad": q_cnt["bad"],
                    "Q_final_missing": q_cnt["missing"],

                    "SSC_final_good": ssc_cnt["good"],
                    "SSC_final_estimated": ssc_cnt["estimated"],
                    "SSC_final_suspect": ssc_cnt["suspect"],
                    "SSC_final_bad": ssc_cnt["bad"],
                    "SSC_final_missing": ssc_cnt["missing"],

                    "SSL_final_good": ssl_cnt["good"],
                    "SSL_final_estimated": ssl_cnt["estimated"],
                    "SSL_final_suspect": ssl_cnt["suspect"],
                    "SSL_final_bad": ssl_cnt["bad"],
                    "SSL_final_missing": ssl_cnt["missing"],
                })

                # ---- Step flags count (QC1/QC2/QC3) ----
                # QC1 step flag: 0 pass, 3 bad, 9 missing
                qc1_map = {"pass": 0, "bad": 3, "missing": 9}

                # QC2 step flag: 0 pass, 2 suspect, 8 not_checked, 9 missing
                qc2_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}

                # QC3 SSC–Q step: 0 pass, 2 suspect, 8 not_checked, 9 missing
                qc3_map = {"pass": 0, "suspect": 2, "not_checked": 8, "missing": 9}

                # QC3 SSL propagation step: 2 propagated, 0 not_propagated, 8 not_checked, 9 missing
                qc3_ssl_map = {"not_propagated": 0, "propagated": 2, "not_checked": 8, "missing": 9}

                # 注意：这些 key 只有在你用 apply_hydro_qc_with_provenance 时才会存在
                if "Q_flag_qc1_physical" in qc:
                    c = self._count_step_flags(qc["Q_flag_qc1_physical"], qc1_map)
                    station_info.update({f"Q_qc1_{k}": v for k, v in c.items()})

                if "SSC_flag_qc1_physical" in qc:
                    c = self._count_step_flags(qc["SSC_flag_qc1_physical"], qc1_map)
                    station_info.update({f"SSC_qc1_{k}": v for k, v in c.items()})

                if "SSL_flag_qc1_physical" in qc:
                    c = self._count_step_flags(qc["SSL_flag_qc1_physical"], qc1_map)
                    station_info.update({f"SSL_qc1_{k}": v for k, v in c.items()})

                if "Q_flag_qc2_log_iqr" in qc:
                    c = self._count_step_flags(qc["Q_flag_qc2_log_iqr"], qc2_map)
                    station_info.update({f"Q_qc2_{k}": v for k, v in c.items()})

                if "SSC_flag_qc2_log_iqr" in qc:
                    c = self._count_step_flags(qc["SSC_flag_qc2_log_iqr"], qc2_map)
                    station_info.update({f"SSC_qc2_{k}": v for k, v in c.items()})

                if "SSL_flag_qc2_log_iqr" in qc:
                    c = self._count_step_flags(qc["SSL_flag_qc2_log_iqr"], qc2_map)
                    station_info.update({f"SSL_qc2_{k}": v for k, v in c.items()})

                if "SSC_flag_qc3_ssc_q" in qc:
                    c = self._count_step_flags(qc["SSC_flag_qc3_ssc_q"], qc3_map)
                    station_info.update({f"SSC_qc3_{k}": v for k, v in c.items()})

                if "SSL_flag_qc3_from_ssc_q" in qc:
                    c = self._count_step_flags(qc["SSL_flag_qc3_from_ssc_q"], qc3_ssl_map)
                    station_info.update({f"SSL_qc3_{k}": v for k, v in c.items()})


                station_info.update({
                        "n_warnings": len(station_warnings),
                        "warnings": " | ".join(station_warnings[:5])  # 最多存前5条，防爆
                    })

                self.stats['processed_stations'] += 1
                print(f"  ✓ 成功处理")
                print(f"    时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
                print(f"    完整性: Q={Q_completeness:.1f}%, SSC={SSC_completeness:.1f}%, SSL={SSL_completeness:.1f}%")

                return True, station_info

        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None



    def process_all_stations(self):
        """并行处理所有站点"""

        print(f"\n{'='*80}")
        print(f"HYDAT 数据集质量控制和CF-1.8标准化处理 (并行加速版)")
        print(f"{'='*80}\n")

        input_files = sorted(self.input_dir.glob('HYDAT_*_SEDIMENT.nc'))
        self.stats['total_stations'] = len(input_files)

        print(f"找到 {len(input_files)} 个站点文件")
        print(f"使用 CPU 核心数: {os.cpu_count()} 并行处理\n")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}\n")

        results = []

        # ★★★ 并行执行 ★★★
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor: #ProcessPoolExecutor用于并行处理任务
            future_to_station = {executor.submit(self.process_station, f): f for f in input_files}
            # 收集结果
            for future in as_completed(future_to_station):
                success, station_info = future.result()
                if success and station_info:
                    results.append(station_info)

        # === 更新统计信息 ===
        self.stats['processed_stations'] = len(results)
        self.stats['removed_stations'] = self.stats['total_stations'] - len(results)
        self.stats['stations_info'] = results

        print(f"\n{'='*80}")
        print(f"处理完成! (并行)")
        print(f"{'='*80}")
        print(f"总站点数: {self.stats['total_stations']}")
        print(f"成功处理: {self.stats['processed_stations']}")
        print(f"删除站点: {self.stats['removed_stations']}")
        print(f"{'='*80}\n")

        return self.stats


    def summarize_warning_types(self):
        return summarize_warning_types_tool(self.stats['stations_info'])


    def generate_csv_summary(self, output_csv):
        generate_csv_summary_tool(self.stats['stations_info'], output_csv)

    def generate_qc_results_csv(self, output_csv):
        generate_qc_results_csv_tool(self.stats['stations_info'], output_csv)



def main():
    """主函数"""
    # 设置路径
    BASE_DIR = Path(__file__).resolve().parent          # .../Script/Hydat 之类
    PROJECT_DIR = BASE_DIR.parents[1]                      # 上一级（你之前的 PARENT_DIR 逻辑）

    # 用相对路径替代绝对路径（按你的目录结构改这里）
    input_dir = PROJECT_DIR / "Output_r" / "daily" / "HYDAT" / "sediment_update"
    output_dir = PROJECT_DIR / "Output_r" / "daily" / "HYDAT" / "output_update"

    csv_file = output_dir / 'HYDAT_station_summary.csv'

    # 创建处理对象
    qc = HYDATQualityControl(input_dir, output_dir) #这行是在初始化类，传入输入输出目录。初始化类是为了创建一个类的实例，并为其设置初始状态或属性。
    qc_csv = output_dir / "HYDAT_qc_results_summary.csv"

    # 处理所有站点
    stats = qc.process_all_stations() #调用类的方法处理所有站点

    # 生成CSV摘要
    qc.generate_csv_summary(csv_file)
    qc.generate_qc_results_csv(qc_csv)

    print(f"\n✓ 全部完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  CSV摘要: {csv_file}")


if __name__ == '__main__':
    main()
