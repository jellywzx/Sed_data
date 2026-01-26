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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    NOT_CHECKED_INT,
    apply_quality_flag,
    compute_log_iqr_bounds,
    apply_log_iqr_screening,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    # check_nc_completeness,
    # add_global_attributes,
    propagate_ssc_q_inconsistency_to_ssl
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
    Apply QC using functions from tool.py (local wrapper).

    Rules
    -----
    - Physical checks: apply_quality_flag
    - Statistical screening: log-IQR (Q, SSC only)
    - Hydrological consistency: SSC–Q envelope
    - Valid time: keep days where ANY of Q/SSC/SSL is not missing
    """

    n = len(time)

    # -----------------------------
    # 1. Physical QC (baseline)
    # -----------------------------
    # Step-level provenance flags (for tracing which QC step triggered a downgrade)
    # QC1: physical feasibility / missing / fill value
    Q_flag_qc1_physical = np.array([apply_quality_flag(v, "Q") for v in Q], dtype=np.int8)
    SSC_flag_qc1_physical = np.array([apply_quality_flag(v, "SSC") for v in SSC], dtype=np.int8)
    SSL_flag_qc1_physical = np.array([apply_quality_flag(v, "SSL") for v in SSL], dtype=np.int8)

    # Final flags start from physical QC
    Q_flag = Q_flag_qc1_physical.copy()
    SSC_flag = SSC_flag_qc1_physical.copy()
    SSL_flag = SSL_flag_qc1_physical.copy()

    # -----------------------------
    # 2. log-IQR screening
    #    (only independent vars)
    # -----------------------------
    # QC2: statistical screening (log-IQR)
    # Use tool.py helper so QC2 never overrides upstream "bad=3".
    #!!!需要手动修改
    Q_flag_qc2_log_iqr, Q_flag, _ = apply_qc2_log_iqr_if_independent(
        values=Q,
        base_flag=Q_flag,
        is_independent=True,   # Q 是独立观测
    )

    SSC_flag_qc2_log_iqr, SSC_flag, _ = apply_qc2_log_iqr_if_independent(
        values=SSC,
        base_flag=SSC_flag,
        is_independent=True,   # SSC 是独立观测
    )

    # 如果你的 SSL 是派生量（Q×SSC），就这样：
    SSL_flag_qc2_log_iqr, SSL_flag, _ = apply_qc2_log_iqr_if_independent(
        values=SSL,
        base_flag=SSL_flag,
        is_independent=False,  # SSL 是派生量 -> QC2 不做，但标 estimated
    )

    # -----------------------------
    # 3. SSC–Q consistency check
    # -----------------------------
    # QC3: hydrological consistency (SSC–Q envelope)
    # Only build envelope from data that survived QC1+QC2 (both good).
    SSC_flag_qc3_ssc_q = np.full(n, NOT_CHECKED_INT, dtype=np.int8)    # 8 = not checked
    SSL_flag_qc3_from_ssc_q = np.full(n, NOT_CHECKED_INT, dtype=np.int8)  # 8 = not checked / not applicable
    # Mark missing explicitly
    SSC_flag_qc3_ssc_q[SSC_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT
    SSL_flag_qc3_from_ssc_q[SSL_flag_qc1_physical == FILL_VALUE_INT] = FILL_VALUE_INT

    env_mask = (Q_flag == 0) & (SSC_flag == 0) & np.isfinite(Q) & np.isfinite(SSC) & (Q > 0) & (SSC > 0)
    Q_env = np.where(env_mask, Q, np.nan)
    SSC_env = np.where(env_mask, SSC, np.nan)
    ssc_q_bounds = build_ssc_q_envelope(Q_env, SSC_env)

    if ssc_q_bounds is not None:
        eval_mask = env_mask.copy()
        SSC_flag_qc3_ssc_q[eval_mask] = 0  # checked and passed by default

        for i in np.where(eval_mask)[0]:
            inconsistent, _ = check_ssc_q_consistency(
                Q[i], SSC[i],
                Q_flag[i], SSC_flag[i],
                ssc_q_bounds
            )

            if inconsistent:
                SSC_flag_qc3_ssc_q[i] = 2
                SSC_flag[i] = 2

                # Propagate inconsistency to derived SSL (optional)
                prev_ssl_flag = SSL_flag[i]
                SSL_flag[i] = propagate_ssc_q_inconsistency_to_ssl(
                    inconsistent=inconsistent,
                    Q=Q[i],
                    SSC=SSC[i],
                    SSL=SSL[i],
                    Q_flag=Q_flag[i],
                    SSC_flag=0,  # use pre-downgrade state for propagation logic
                    SSL_flag=prev_ssl_flag,
                    ssl_is_derived_from_q_ssc=True  # ← 这里你可以控制
                )
                # record whether propagation actually changed SSL quality
                SSL_flag_qc3_from_ssc_q[i] = 2 if (prev_ssl_flag == 0 and SSL_flag[i] == 2) else 0


    # -----------------------------
    # 4. Valid-time mask
    #    ANY variable non-missing
    # -----------------------------
    valid_time = (
        (Q_flag != FILL_VALUE_INT)
        | (SSC_flag != FILL_VALUE_INT)
        | (SSL_flag != FILL_VALUE_INT)
    )

    if not np.any(valid_time):
        return None

    time = time[valid_time]
    Q = Q[valid_time]
    SSC = SSC[valid_time]
    SSL = SSL[valid_time]
    Q_flag = Q_flag[valid_time]
    SSC_flag = SSC_flag[valid_time]
    SSL_flag = SSL_flag[valid_time]
    Q_flag_qc1_physical = Q_flag_qc1_physical[valid_time]
    SSC_flag_qc1_physical = SSC_flag_qc1_physical[valid_time]
    SSL_flag_qc1_physical = SSL_flag_qc1_physical[valid_time]
    Q_flag_qc2_log_iqr = Q_flag_qc2_log_iqr[valid_time]
    SSC_flag_qc2_log_iqr = SSC_flag_qc2_log_iqr[valid_time]
    SSC_flag_qc3_ssc_q = SSC_flag_qc3_ssc_q[valid_time]
    SSL_flag_qc3_from_ssc_q = SSL_flag_qc3_from_ssc_q[valid_time]

    # -----------------------------
    # 5. Diagnostic plot (optional)
    # -----------------------------
    if plot_dir is not None and ssc_q_bounds is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_ssc_q_diagnostic(
            time=pd.to_datetime(time, unit="D", origin="1970-01-01"),
            Q=Q,
            SSC=SSC,
            Q_flag=Q_flag,
            SSC_flag=SSC_flag,
            ssc_q_bounds=ssc_q_bounds,
            station_id=station_id,
            station_name=station_name,
            out_png=plot_dir / f"{station_id}_ssc_q.png",
        )

    return {
        "time": time,
        "Q": Q,
        "SSC": SSC,
        "SSL": SSL,
        "Q_flag": Q_flag,
        "SSC_flag": SSC_flag,
        "SSL_flag": SSL_flag,
        # Step-level provenance flags
        "Q_flag_qc1_physical": Q_flag_qc1_physical,
        "SSC_flag_qc1_physical": SSC_flag_qc1_physical,
        "SSL_flag_qc1_physical": SSL_flag_qc1_physical,
        "Q_flag_qc2_log_iqr": Q_flag_qc2_log_iqr,
        "SSC_flag_qc2_log_iqr": SSC_flag_qc2_log_iqr,
        "SSC_flag_qc3_ssc_q": SSC_flag_qc3_ssc_q,
        "SSL_flag_qc3_from_ssc_q": SSL_flag_qc3_from_ssc_q,
    }


class HYDATQualityControl:
    """HYDAT数据质量控制和标准化处理类"""

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

        # 统计信息
        self.stats = {
            'total_stations': 0,
            'processed_stations': 0,
            'removed_stations': 0,
            'stations_info': []
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

    def process_station(self, input_file):
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
                SSL_flag = qc["SSL_flag"]
                Q_flag_qc1_physical = qc["Q_flag_qc1_physical"]
                SSC_flag_qc1_physical = qc["SSC_flag_qc1_physical"]
                SSL_flag_qc1_physical = qc["SSL_flag_qc1_physical"]
                Q_flag_qc2_log_iqr = qc["Q_flag_qc2_log_iqr"]
                SSC_flag_qc2_log_iqr = qc["SSC_flag_qc2_log_iqr"]
                SSC_flag_qc3_ssc_q = qc["SSC_flag_qc3_ssc_q"]
                SSL_flag_qc3_from_ssc_q = qc["SSL_flag_qc3_from_ssc_q"]

                
                # 计算时间范围
                reference_date = pd.Timestamp('1970-01-01')
                start_date = reference_date + pd.Timedelta(days=float(time[0]))
                end_date = reference_date + pd.Timedelta(days=float(time[-1]))

                # 计算完整性
                Q_completeness = self.calculate_completeness(Q, Q_flag, start_date, end_date)
                SSC_completeness = self.calculate_completeness(SSC, SSC_flag, start_date, end_date)
                SSL_completeness = self.calculate_completeness(SSL, SSL_flag, start_date, end_date)

                # ==========================================================
                # Per-station QC results summary (counts) for traceability
                # ==========================================================
                n_days = int(len(time))

                def _count_flags(arr, code_to_name):
                    arr = np.asarray(arr)
                    out = {}
                    for code, name in code_to_name.items():
                        out[name] = int(np.sum(arr == code))
                    return out

                final_map = {0: "good", 2: "suspect", 3: "bad", 9: "missing"}
                step_map = {0: "pass", 2: "suspect", 8: "not_checked", 9: "missing"}
                ssl_qc3_map = {0: "not_propagated", 2: "propagated", 8: "not_checked", 9: "missing"}

                # 创建输出文件
                output_file = self.output_dir / f"HYDAT_{station_id}.nc"

                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    # 创建维度
                    ds_out.createDimension('time', len(time))

                    # 创建时间变量
                    var_time = ds_out.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.units = 'days since 1970-01-01 00:00:00'
                    var_time.calendar = 'gregorian'
                    var_time.axis = 'T'
                    var_time[:] = time

                    # 创建坐标变量 (标量)
                    var_lat = ds_out.createVariable('lat', 'f4')
                    var_lat.standard_name = 'latitude'
                    var_lat.long_name = 'station latitude'
                    var_lat.units = 'degrees_north'
                    var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
                    var_lat[:] = lat

                    var_lon = ds_out.createVariable('lon', 'f4')
                    var_lon.standard_name = 'longitude'
                    var_lon.long_name = 'station longitude'
                    var_lon.units = 'degrees_east'
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
                    var_Q.ancillary_variables = 'Q_flag Q_flag_qc1_physical Q_flag_qc2_log_iqr'
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

                    # Q - QC1 physical provenance flag
                    var_Q_flag_qc1 = ds_out.createVariable('Q_flag_qc1_physical', 'i1', ('time',), fill_value=np.int8(9))
                    var_Q_flag_qc1.long_name = 'QC1 physical check flag for river discharge'
                    var_Q_flag_qc1.standard_name = 'status_flag'
                    var_Q_flag_qc1.flag_values = np.array([0, 3, 9], dtype=np.int8)
                    var_Q_flag_qc1.flag_meanings = 'pass bad_data missing_data'
                    var_Q_flag_qc1.comment = 'QC step 1 (physical): 3=physically impossible (e.g., negative), 9=missing/fill.'
                    var_Q_flag_qc1[:] = Q_flag_qc1_physical

                    # Q - QC2 log-IQR provenance flag
                    var_Q_flag_qc2 = ds_out.createVariable('Q_flag_qc2_log_iqr', 'i1', ('time',), fill_value=np.int8(9))
                    var_Q_flag_qc2.long_name = 'QC2 log-IQR screening flag for river discharge'
                    var_Q_flag_qc2.standard_name = 'status_flag'
                    var_Q_flag_qc2.flag_values = np.array([0, 2, 8, 9], dtype=np.int8)
                    var_Q_flag_qc2.flag_meanings = 'pass suspect_data not_checked missing_data'
                    var_Q_flag_qc2.comment = 'QC step 2 (statistical): 2=outside log-IQR bounds; 8=not checked (e.g., failed upstream QC, non-positive, or insufficient samples); 9=missing.'
                    var_Q_flag_qc2[:] = Q_flag_qc2_log_iqr

                    # 创建数据变量 SSC
                    var_SSC = ds_out.createVariable('SSC', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSC.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                    var_SSC.long_name = 'suspended sediment concentration'
                    var_SSC.units = 'mg L-1'
                    var_SSC.coordinates = 'time lat lon'
                    var_SSC.ancillary_variables = 'SSC_flag SSC_flag_qc1_physical SSC_flag_qc2_log_iqr SSC_flag_qc3_ssc_q'
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

                    # SSC - QC1 physical provenance flag
                    var_SSC_flag_qc1 = ds_out.createVariable('SSC_flag_qc1_physical', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSC_flag_qc1.long_name = 'QC1 physical check flag for suspended sediment concentration'
                    var_SSC_flag_qc1.standard_name = 'status_flag'
                    var_SSC_flag_qc1.flag_values = np.array([0, 3, 9], dtype=np.int8)
                    var_SSC_flag_qc1.flag_meanings = 'pass bad_data missing_data'
                    var_SSC_flag_qc1.comment = 'QC step 1 (physical): 3=physically impossible (e.g., negative), 9=missing/fill.'
                    var_SSC_flag_qc1[:] = SSC_flag_qc1_physical

                    # SSC - QC2 log-IQR provenance flag
                    var_SSC_flag_qc2 = ds_out.createVariable('SSC_flag_qc2_log_iqr', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSC_flag_qc2.long_name = 'QC2 log-IQR screening flag for suspended sediment concentration'
                    var_SSC_flag_qc2.standard_name = 'status_flag'
                    var_SSC_flag_qc2.flag_values = np.array([0, 2, 8, 9], dtype=np.int8)
                    var_SSC_flag_qc2.flag_meanings = 'pass suspect_data not_checked missing_data'
                    var_SSC_flag_qc2.comment = 'QC step 2 (statistical): 2=outside log-IQR bounds; 8=not checked (e.g., failed upstream QC, non-positive, or insufficient samples); 9=missing.'
                    var_SSC_flag_qc2[:] = SSC_flag_qc2_log_iqr

                    # SSC - QC3 SSC–Q envelope provenance flag
                    var_SSC_flag_qc3 = ds_out.createVariable('SSC_flag_qc3_ssc_q', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSC_flag_qc3.long_name = 'QC3 SSC–Q consistency check flag for suspended sediment concentration'
                    var_SSC_flag_qc3.standard_name = 'status_flag'
                    var_SSC_flag_qc3.flag_values = np.array([0, 2, 8, 9], dtype=np.int8)
                    var_SSC_flag_qc3.flag_meanings = 'pass suspect_data not_checked missing_data'
                    var_SSC_flag_qc3.comment = 'QC step 3 (hydrological): 2=SSC inconsistent with Q based on station-level SSC–Q envelope; 8=not checked (e.g., envelope unavailable or failed upstream QC); 9=missing.'
                    var_SSC_flag_qc3[:] = SSC_flag_qc3_ssc_q

                    # 创建数据变量 SSL
                    var_SSL = ds_out.createVariable('SSL', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSL.long_name = 'suspended sediment load'
                    var_SSL.units = 'ton day-1'
                    var_SSL.coordinates = 'time lat lon'
                    var_SSL.ancillary_variables = 'SSL_flag SSL_flag_qc1_physical SSL_flag_qc3_from_ssc_q'
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

                    # SSL - QC1 physical provenance flag
                    var_SSL_flag_qc1 = ds_out.createVariable('SSL_flag_qc1_physical', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSL_flag_qc1.long_name = 'QC1 physical check flag for suspended sediment load'
                    var_SSL_flag_qc1.standard_name = 'status_flag'
                    var_SSL_flag_qc1.flag_values = np.array([0, 3, 9], dtype=np.int8)
                    var_SSL_flag_qc1.flag_meanings = 'pass bad_data missing_data'
                    var_SSL_flag_qc1.comment = 'QC step 1 (physical): 3=physically impossible (e.g., negative), 9=missing/fill.'
                    var_SSL_flag_qc1[:] = SSL_flag_qc1_physical

                    # SSL - QC3 propagation provenance flag (from SSC–Q inconsistency)
                    var_SSL_flag_qc3 = ds_out.createVariable('SSL_flag_qc3_from_ssc_q', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSL_flag_qc3.long_name = 'QC3 propagated flag for suspended sediment load from SSC–Q inconsistency'
                    var_SSL_flag_qc3.standard_name = 'status_flag'
                    var_SSL_flag_qc3.flag_values = np.array([0, 2, 8, 9], dtype=np.int8)
                    var_SSL_flag_qc3.flag_meanings = 'not_propagated propagated not_checked missing_data'
                    var_SSL_flag_qc3.comment = 'QC step 3 (hydrological): 2=SSL downgraded because SSC is inconsistent with Q and SSL is treated as derived from Q×SSC; 0=inconsistency detected but SSL not downgraded; 8=not checked/not applicable; 9=missing.'
                    var_SSL_flag_qc3[:] = SSL_flag_qc3_from_ssc_q

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
                    ds_out.Type = 'In-situ station data'
                    ds_out.Temporal_Resolution = 'daily'
                    ds_out.Temporal_Span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    ds_out.Variables_Provided = 'altitude, upstream_area, Q, SSC, SSL'
                    ds_out.Geographic_Coverage = f"{province}, Canada"
                    ds_out.country = 'Canada'
                    ds_out.continent_region = 'North America'
                    ds_out.time_coverage_start = start_date.strftime('%Y-%m-%d')
                    ds_out.time_coverage_end = end_date.strftime('%Y-%m-%d')
                    ds_out.number_of_data = '1'
                    ds_out.Reference = 'HYDAT - Canadian Hydrometric Database, Water Survey of Canada'
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
                    # ds_out.comment = (
                    #     f"Data quality flags indicate reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
                    #     f"Quality control applied: Q<0 flagged as bad, Q=0 flagged as suspect, Q>{self.Q_extreme_high} flagged as suspect; "
                    #     f"SSC<0 flagged as bad, SSC<{self.SSC_min} or SSC>{self.SSC_extreme_high} flagged as suspect; "
                    #     f"SSL<0 flagged as bad."
                    # )
                # ==========================================================
                # NetCDF completeness check (CF-1.8 / ACDD-1.3)
                # ==========================================================
                # errors, warnings = check_nc_completeness(output_file, strict=True)

                # if errors:
                #     print(f"  ✗ NetCDF completeness check FAILED for {station_id}")
                #     for e in errors:
                #         print(f"    ERROR: {e}")

                #     # 可选：删除不合格文件
                #     try:
                #         output_file.unlink()
                #         print(f"    → Invalid NetCDF removed: {output_file.name}")
                #     except Exception:
                #         pass

                #     return False, None

                #     if warnings:
                #         with nc.Dataset(output_file, "a") as ds_out:
                #             ds_out.history = (
                #                 ds_out.history
                #                 + f"; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                #                 f"Completeness check warnings: {len(warnings)} issues"
                #             )



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
                    'SSL_percent_complete': round(SSL_completeness, 2),

                    # QC results (final flags)
                    'QC_n_days': n_days,
                    **{f"Q_final_{k}": v for k, v in _count_flags(Q_flag, final_map).items()},
                    **{f"SSC_final_{k}": v for k, v in _count_flags(SSC_flag, final_map).items()},
                    **{f"SSL_final_{k}": v for k, v in _count_flags(SSL_flag, final_map).items()},

                    # QC step provenance (QC1/QC2/QC3)
                    **{f"Q_qc1_{k}": v for k, v in _count_flags(Q_flag_qc1_physical, {0: "pass", 3: "bad", 9: "missing"}).items()},
                    **{f"SSC_qc1_{k}": v for k, v in _count_flags(SSC_flag_qc1_physical, {0: "pass", 3: "bad", 9: "missing"}).items()},
                    **{f"SSL_qc1_{k}": v for k, v in _count_flags(SSL_flag_qc1_physical, {0: "pass", 3: "bad", 9: "missing"}).items()},

                    **{f"Q_qc2_{k}": v for k, v in _count_flags(Q_flag_qc2_log_iqr, step_map).items()},
                    **{f"SSC_qc2_{k}": v for k, v in _count_flags(SSC_flag_qc2_log_iqr, step_map).items()},
                    **{f"SSC_qc3_{k}": v for k, v in _count_flags(SSC_flag_qc3_ssc_q, step_map).items()},
                    **{f"SSL_qc3_{k}": v for k, v in _count_flags(SSL_flag_qc3_from_ssc_q, ssl_qc3_map).items()},
                }

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
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_station = {executor.submit(self.process_station, f): f for f in input_files}

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

    def generate_csv_summary(self, output_csv):
        """生成CSV站点摘要文件"""
        print(f"\n生成CSV摘要文件: {output_csv}")

        if not self.stats['stations_info']:
            print("  ⚠ 警告: 无站点信息可写入CSV")
            return

        df = pd.DataFrame(self.stats['stations_info'])

        # 按指定顺序排列列
        column_order = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'altitude', 'upstream_area', 'Data Source Name', 'Type',
            'Temporal Resolution', 'Temporal Span', 'Variables Provided',
            'Geographic Coverage', 'Reference/DOI',
            'Q_start_date', 'Q_end_date', 'Q_percent_complete',
            'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
            'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
        ]

        df = df[column_order]
        df.to_csv(output_csv, index=False)

        print(f"  ✓ CSV文件已生成: {len(df)} 个站点")

    def generate_qc_results_csv(self, output_csv):
        """输出每一个站点的质量控制结果（按flag计数汇总）"""
        print(f"\n生成站点QC结果汇总CSV: {output_csv}")

        if not self.stats['stations_info']:
            print("  ⚠ 警告: 无站点信息可写入CSV")
            return

        df = pd.DataFrame(self.stats['stations_info'])

        # Prefer a stable, explicit subset if available
        preferred_cols = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'QC_n_days',
            'Q_final_good', 'Q_final_suspect', 'Q_final_bad', 'Q_final_missing',
            'SSC_final_good', 'SSC_final_suspect', 'SSC_final_bad', 'SSC_final_missing',
            'SSL_final_good', 'SSL_final_suspect', 'SSL_final_bad', 'SSL_final_missing',
            'Q_qc1_pass', 'Q_qc1_bad', 'Q_qc1_missing',
            'SSC_qc1_pass', 'SSC_qc1_bad', 'SSC_qc1_missing',
            'SSL_qc1_pass', 'SSL_qc1_bad', 'SSL_qc1_missing',
            'Q_qc2_pass', 'Q_qc2_suspect', 'Q_qc2_not_checked', 'Q_qc2_missing',
            'SSC_qc2_pass', 'SSC_qc2_suspect', 'SSC_qc2_not_checked', 'SSC_qc2_missing',
            'SSC_qc3_pass', 'SSC_qc3_suspect', 'SSC_qc3_not_checked', 'SSC_qc3_missing',
            'SSL_qc3_not_propagated', 'SSL_qc3_propagated', 'SSL_qc3_not_checked', 'SSL_qc3_missing',
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        if cols:
            df = df[cols]

        df.to_csv(output_csv, index=False)
        print(f"  ✓ QC结果CSV已生成: {len(df)} 个站点")


def main():
    """主函数"""
    # 设置路径
    # Project root = .../sediment_wzx_1111 (this file is .../sediment_wzx_1111/Script/Hydat/4_process_hydat_cf18.py)
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / 'Output_r/daily/HYDAT/sediment_update'
    output_dir = project_root / 'Output_r/daily/HYDAT/output_update'
    csv_file = output_dir / 'HYDAT_station_summary.csv'
    qc_csv_file = output_dir / 'HYDAT_station_qc_results.csv'

    # 创建处理对象
    qc = HYDATQualityControl(input_dir, output_dir)

    # 处理所有站点
    stats = qc.process_all_stations()

    # 生成CSV摘要
    qc.generate_csv_summary(csv_file)
    qc.generate_qc_results_csv(qc_csv_file)

    print(f"\n✓ 全部完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  CSV摘要: {csv_file}")
    print(f"  QC结果: {qc_csv_file}")


if __name__ == '__main__':
    main()
