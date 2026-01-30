import os
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
import sys
from datetime import datetime
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from tool import (
    FILL_VALUE_FLOAT,
    FILL_VALUE_INT,
    apply_quality_flag,
    apply_quality_flag_array,                
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    apply_hydro_qc_with_provenance,           
    generate_csv_summary as generate_csv_summary_tool,          
    generate_qc_results_csv as generate_qc_results_csv_tool,
)



# ========= 1. 通用读取函数（关键：正确处理引号和空格） =========
SEP_REGEX = r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)'  # 空白分隔，但忽略引号内部的空格


def read_spm(spm_file):
    """
    读取 data_spm.txt：
    "date" "suspended_sediment_concentration [mg/L]" "discharge [m3/s]" ...
    """
    df = pd.read_csv(
        spm_file,
        sep=SEP_REGEX,
        engine='python',
        quotechar='"'
    )
    # 列名去掉外层引号与空格
    df.columns = df.columns.str.strip().str.strip('"')

    # 去掉值里的引号
    df['date'] = pd.to_datetime(df['date'].str.strip('"'))
    df['station'] = df['station'].str.strip('"')
    df['method'] = df['method'].str.strip('"')
    df['instrument'] = df['instrument'].str.strip('"')

    # 重命名成更干净的名字
    df = df.rename(columns={
        'suspended_sediment_concentration [mg/L]': 'ssc',
        'discharge [m3/s]': 'q',
        'latitude [WGS84]': 'lat',
        'longitude [WGS84]': 'lon'
    })

    # 转成数值（保险起见）
    for col in ['ssc', 'q', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def read_mpm(mpm_file):
    """
    读取 data_mpm.txt：
    "station" "id" "id_lot" "date" "discharge [m3/s]" ... "suspended_sediment_concentration [mg/l]" ...
    """
    df = pd.read_csv(
        mpm_file,
        sep=SEP_REGEX,
        engine='python',
        quotechar='"'
    )
    df.columns = df.columns.str.strip().str.strip('"')

    df['date'] = pd.to_datetime(df['date'].str.strip('"'))
    df['station'] = df['station'].str.strip('"')

    # 重命名
    df = df.rename(columns={
        'suspended_sediment_concentration [mg/l]': 'ssc',
        'discharge [m3/s]': 'q',
        'latitude [WGS84]': 'lat',
        'longitude [WGS84]': 'lon'
    })

    for col in ['ssc', 'q', 'lat', 'lon']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def apply_tool_qc(
    time,
    Q,
    SSC,
    SSL,
    station_id,
    station_name,
    plot_dir=None,
):
    # --- force strict 1D(time) & align length (avoid len() of scalar / 0D) ---
    time = np.atleast_1d(np.asarray(time)).reshape(-1)
    Q    = np.atleast_1d(np.asarray(Q, dtype=float)).reshape(-1)
    SSC  = np.atleast_1d(np.asarray(SSC, dtype=float)).reshape(-1)
    SSL  = np.atleast_1d(np.asarray(SSL, dtype=float)).reshape(-1)

    n = min(time.size, Q.size, SSC.size, SSL.size)
    if n == 0:
        return None, None
    time, Q, SSC, SSL = time[:n], Q[:n], SSC[:n], SSL[:n]

    # --- tool.py pipeline: QC1/QC2/QC3 + provenance ---
    qc_all = apply_hydro_qc_with_provenance(
        time=time,
        Q=Q,
        SSC=SSC,
        SSL=SSL,
        # source data: Q/SSC independent; SSL derived from Q*SSC
        Q_is_independent=True,
        SSC_is_independent=True,
        SSL_is_independent=False,
        ssl_is_derived_from_q_ssc=True,
        # (可按需调参)
        qc2_k=1.5,
        qc2_min_samples=5,
        qc3_k=1.5,
        qc3_min_samples=5,
    )
    if qc_all is None:
        return None, None

    # --- valid_time: value-based "present" (更稳) ---
    def _present(v, f):
        v = np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)
        f = np.atleast_1d(np.asarray(f, dtype=np.int8)).reshape(-1)
        return (
            (f != FILL_VALUE_INT)
            & np.isfinite(v)
            & (~np.isclose(v, float(FILL_VALUE_FLOAT), rtol=1e-5, atol=1e-5))
        )

    present_Q   = _present(qc_all["Q"],   qc_all["Q_flag"])
    present_SSC = _present(qc_all["SSC"], qc_all["SSC_flag"])
    present_SSL = _present(qc_all["SSL"], qc_all["SSL_flag"])

    valid_time = present_Q | present_SSC | present_SSL
    if not np.any(valid_time):
        return None, None

    # --- trim ALL arrays incl. step/provenance flags ---
    for k in list(qc_all.keys()):
        v = qc_all[k]
        if isinstance(v, np.ndarray) and v.shape[0] == valid_time.shape[0]:
            qc_all[k] = v[valid_time]

    # --- optional diagnostic plot (if envelope/bounds present) ---
    ssc_q_bounds = qc_all.get("ssc_q_bounds", None)
    if plot_dir is not None and ssc_q_bounds is not None:
        try:
            os.makedirs(plot_dir, exist_ok=True)
            out_png = os.path.join(plot_dir, f"{station_id}_ssc_q.png")
            plot_ssc_q_diagnostic(
                time=qc_all.get("time", time[valid_time]),
                Q=qc_all["Q"],
                SSC=qc_all["SSC"],
                Q_flag=qc_all["Q_flag"],
                SSC_flag=qc_all["SSC_flag"],
                ssc_q_bounds=ssc_q_bounds,
                station_id=station_id,
                station_name=station_name,
                out_png=out_png,
            )
        except Exception:
            pass

    # --- only keep array-like items (avoid pandas “Mixing dicts with non-Series” error) ---
    qc = {k: v for k, v in qc_all.items() if isinstance(v, np.ndarray)}

    # keep the rest of this script compatible: use key 'date'
    if "time" in qc and "date" not in qc:
        qc["date"] = qc.pop("time")

    # --- station qc_report (for generate_qc_results_csv) ---
    def _cnt(arr, v):
        a = np.asarray(arr, dtype=np.int8) if arr is not None else np.asarray([], dtype=np.int8)
        return int(np.sum(a == np.int8(v)))

    def _get(name):
        return qc.get(name, None)

    qf   = _get("Q_flag")
    sscf = _get("SSC_flag")
    sslf = _get("SSL_flag")

    qc_report = {
        "station_name": station_name,
        "Source_ID": station_id,
        "QC_n_days": int(len(qf)) if qf is not None else 0,

        "Q_final_good":       _cnt(qf, 0),
        "Q_final_estimated":  _cnt(qf, 1),
        "Q_final_suspect":    _cnt(qf, 2),
        "Q_final_bad":        _cnt(qf, 3),
        "Q_final_missing":    _cnt(qf, 9),

        "SSC_final_good":      _cnt(sscf, 0),
        "SSC_final_estimated": _cnt(sscf, 1),
        "SSC_final_suspect":   _cnt(sscf, 2),
        "SSC_final_bad":       _cnt(sscf, 3),
        "SSC_final_missing":   _cnt(sscf, 9),

        "SSL_final_good":      _cnt(sslf, 0),
        "SSL_final_estimated": _cnt(sslf, 1),
        "SSL_final_suspect":   _cnt(sslf, 2),
        "SSL_final_bad":       _cnt(sslf, 3),
        "SSL_final_missing":   _cnt(sslf, 9),
    }

    # step flags（存在就统计）
    q_qc1   = _get("Q_flag_qc1_physical")
    ssc_qc1 = _get("SSC_flag_qc1_physical")
    ssl_qc1 = _get("SSL_flag_qc1_physical")
    qc_report.update({
        "Q_qc1_pass":    _cnt(q_qc1, 0),
        "Q_qc1_bad":     _cnt(q_qc1, 3),
        "Q_qc1_missing": _cnt(q_qc1, 9),

        "SSC_qc1_pass":    _cnt(ssc_qc1, 0),
        "SSC_qc1_bad":     _cnt(ssc_qc1, 3),
        "SSC_qc1_missing": _cnt(ssc_qc1, 9),

        "SSL_qc1_pass":    _cnt(ssl_qc1, 0),
        "SSL_qc1_bad":     _cnt(ssl_qc1, 3),
        "SSL_qc1_missing": _cnt(ssl_qc1, 9),
    })

    q_qc2   = _get("Q_flag_qc2_log_iqr")
    ssc_qc2 = _get("SSC_flag_qc2_log_iqr")
    ssl_qc2 = _get("SSL_flag_qc2_log_iqr")
    qc_report.update({
        "Q_qc2_pass":        _cnt(q_qc2, 0),
        "Q_qc2_suspect":     _cnt(q_qc2, 2),
        "Q_qc2_not_checked": _cnt(q_qc2, 8),
        "Q_qc2_missing":     _cnt(q_qc2, 9),

        "SSC_qc2_pass":        _cnt(ssc_qc2, 0),
        "SSC_qc2_suspect":     _cnt(ssc_qc2, 2),
        "SSC_qc2_not_checked": _cnt(ssc_qc2, 8),
        "SSC_qc2_missing":     _cnt(ssc_qc2, 9),

        "SSL_qc2_pass":        _cnt(ssl_qc2, 0),
        "SSL_qc2_suspect":     _cnt(ssl_qc2, 2),
        "SSL_qc2_not_checked": _cnt(ssl_qc2, 8),
        "SSL_qc2_missing":     _cnt(ssl_qc2, 9),
    })

    ssc_qc3 = _get("SSC_flag_qc3_ssc_q")
    ssl_qc3 = _get("SSL_flag_qc3_from_ssc_q")
    qc_report.update({
        "SSC_qc3_pass":        _cnt(ssc_qc3, 0),
        "SSC_qc3_suspect":     _cnt(ssc_qc3, 2),
        "SSC_qc3_not_checked": _cnt(ssc_qc3, 8),
        "SSC_qc3_missing":     _cnt(ssc_qc3, 9),

        # SSL qc3: 0 not_propagated, 1 propagated, 8 not_checked, 9 missing
        "SSL_qc3_not_propagated": _cnt(ssl_qc3, 0),
        "SSL_qc3_propagated":     _cnt(ssl_qc3, 1),
        "SSL_qc3_not_checked":    _cnt(ssl_qc3, 8),
        "SSL_qc3_missing":        _cnt(ssl_qc3, 9),
    })

    return qc, qc_report


def get_rhine_data(input_path):
    spm_file = os.path.join(input_path, 'data_spm.txt')
    mpm_file = os.path.join(input_path, 'data_mpm.txt')

    spm_df = read_spm(spm_file)
    mpm_df = read_mpm(mpm_file)

    return spm_df, mpm_df


# ========= 2. 主处理函数 =========
def process_rhine_data(output_path, source_path):

    os.makedirs(output_path, exist_ok=True)

    spm_df, mpm_df = get_rhine_data(source_path)

    stations = set(spm_df['station'].unique()) | set(mpm_df['station'].unique())

    station_summary_data = []
    qc_results_rows = []

    # 固定时间基准（和你示例一致）
    time_ref = pd.Timestamp("1990-01-01")

    for station_name in sorted(stations):
        print(f"\n=== 处理站点: {station_name} ===")

        spm_station = spm_df[spm_df['station'] == station_name].copy()
        mpm_station = mpm_df[mpm_df['station'] == station_name].copy()

        # ---------- SPM 部分 ----------
        if not spm_station.empty:
            spm_station_clean = spm_station[['date', 'ssc', 'q', 'lat', 'lon']].copy()
            spm_station_clean.columns = ['date', 'ssc', 'q', 'latitude', 'longitude']
        else:
            spm_station_clean = pd.DataFrame(columns=['date', 'ssc', 'q', 'latitude', 'longitude'])

        # ---------- MPM 部分（按 date 聚合） ----------
        if not mpm_station.empty:
            mpm_grouped = mpm_station.groupby('date').agg({
                'q': 'first',
                'ssc': 'mean',
                'lat': 'first',
                'lon': 'first'
            }).reset_index()
            mpm_grouped.columns = ['date', 'q', 'ssc', 'latitude', 'longitude']
        else:
            mpm_grouped = pd.DataFrame(columns=['date', 'q', 'ssc', 'latitude', 'longitude'])

        # ---------- 合并 SPM + MPM ----------
        # 安全拼接：排除空 DataFrame 和所有值均为 NA 的列，避免 pandas FutureWarning
        frames = []
        for df_part in (spm_station_clean, mpm_grouped):
            if df_part is None:
                continue
            if df_part.empty:
                continue
            # 去掉整列均为 NA 的列
            df_part2 = df_part.loc[:, df_part.notna().any(axis=0)]
            if df_part2.empty:
                continue
            frames.append(df_part2)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=['date', 'ssc', 'q', 'latitude', 'longitude'])
        combined = combined.drop_duplicates(subset=['date'])
        combined = combined.sort_values('date')

        # 只要不是整站都空，就继续
        combined['ssc'] = pd.to_numeric(combined['ssc'], errors='coerce')
        combined['q'] = pd.to_numeric(combined['q'], errors='coerce')
        combined['latitude'] = pd.to_numeric(combined['latitude'], errors='coerce')
        combined['longitude'] = pd.to_numeric(combined['longitude'], errors='coerce')

        combined = combined.dropna(subset=['ssc', 'q'], how='all')
        if combined.empty:
            print("  -> 此站点无有效 Q/SSC 数据，跳过。")
            continue

        start_year = combined['date'].dt.year.min()
        end_year = combined['date'].dt.year.max()
        if pd.isna(start_year) or pd.isna(end_year):
            print("  -> 日期全是 NaN，跳过。")
            continue

        start_date = datetime(int(start_year), 1, 1)
        end_date = datetime(int(end_year), 12, 31)

        # ---------- 构建完整逐日时间序列 ----------
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_data = pd.DataFrame({'date': date_range})

        merged = pd.merge(daily_data, combined, on='date', how='left')

        # 计算 SSL（物理公式，保留）
        merged['ssl'] = merged['q'] * merged['ssc'] * 0.0864

        print("  -> Applying tool.py quality control...")

        # 保存经纬度信息以便在 QC 后重新附加（apply_tool_qc 不会返回经纬度）
        pre_qc_latlon = merged[['date', 'latitude', 'longitude']].copy()

        qc, qc_report  = apply_tool_qc(
            time=merged['date'].values,
            Q=merged['q'].values,
            SSC=merged['ssc'].values,
            SSL=merged['ssl'].values,
            station_id=station_name,
            station_name=station_name,
            plot_dir=os.path.join(output_path, "diagnostic_plots"),
        )

        if qc is None:
            print("  -> QC 后无有效数据，跳过该站点。")
            continue

        # QC 输出只包含 time/Q/SSC/SSL 和 flags，重新合并经纬度
        merged = pd.DataFrame(qc)
        n_total = len(merged)

        def _repr(v, f):
            v = np.asarray(v, dtype=float)
            f = np.asarray(f, dtype=np.int8)
            ok = np.isfinite(v) & (v > 0)
            ok_good = ok & (f == 0)
            if np.any(ok_good):
                return float(np.nanmedian(v[ok_good])), 0
            if np.any(ok):
                return float(np.nanmedian(v[ok])), int(np.min(f[ok]))
            return np.nan, 9

        qv, qf   = _repr(merged['Q'].values,   merged['Q_flag'].values)
        sscv, sscf = _repr(merged['SSC'].values, merged['SSC_flag'].values)
        sslv, sslf = _repr(merged['SSL'].values, merged['SSL_flag'].values)


        print(f"  ✔ QC summary ({station_name})")
        print(f"    Samples: {n_total}")
        print(f"    Q   : {qv:.2f} m3/s (flag={qf})")
        print(f"    SSC : {sscv:.2f} mg/L (flag={sscf})")
        print(f"    SSL : {sslv:.2f} ton/day (flag={sslf})")

        if 'date' in merged.columns:
            merged = pd.merge(merged, pre_qc_latlon, on='date', how='left')
        # 标准化列名为脚本其余部分使用的小写形式
        rename_map = {}
        if 'Q' in merged.columns:
            rename_map['Q'] = 'q'
        if 'SSC' in merged.columns:
            rename_map['SSC'] = 'ssc'
        if 'SSL' in merged.columns:
            rename_map['SSL'] = 'ssl'
        if 'Q_flag' in merged.columns:
            rename_map['Q_flag'] = 'q_flag'
        if 'SSC_flag' in merged.columns:
            rename_map['SSC_flag'] = 'ssc_flag'
        if 'SSL_flag' in merged.columns:
            rename_map['SSL_flag'] = 'ssl_flag'
        if rename_map:
            merged.rename(columns=rename_map, inplace=True)
        else:
            # 万一 time 字段命名不同，尝试用第一个列名匹配（保守处理）
            try:
                merged = pd.concat([merged, pre_qc_latlon[['latitude','longitude']].reset_index(drop=True)], axis=1)
            except Exception:
                # 如果仍失败，填充缺省值，避免 KeyError
                merged['latitude'] = np.nan
                merged['longitude'] = np.nan



        # 提前保存经纬度（在 fillna 前）
        lat_series = merged['latitude'].dropna()
        lon_series = merged['longitude'].dropna()
        lat_val = float(lat_series.iloc[0]) if not lat_series.empty else -9999.0
        lon_val = float(lon_series.iloc[0]) if not lon_series.empty else -9999.0
        # --- collect per-station QC results (for QC results CSV) ---
        if qc_report is not None:
            qc_row = {
                'station_name': station_name,
                'Source_ID': station_name,
                'river_name': 'Rhine',
                'longitude': lon_val if lon_val != -9999.0 else '',
                'latitude': lat_val if lat_val != -9999.0 else '',
            }
            qc_row.update(qc_report)
            qc_results_rows.append(qc_row)

        # 填 -9999 用于写 NetCDF
        merged = merged.fillna(-9999.0)

        # ---------- 写 NetCDF ----------
        nc_file_path = os.path.join(output_path, f'Rhine_{station_name}.nc')
        print("  -> 写入:", nc_file_path)

        with nc.Dataset(nc_file_path, 'w', format='NETCDF4') as ds:
            # 维度
            ds.createDimension('time', None)

            # 变量: time
            time_var = ds.createVariable('time', 'f8', ('time',))
            time_var.units = 'days since 1990-01-01 00:00:00'
            time_var.standard_name = 'time'
            time_var.long_name = 'time'
            time_var.calendar = 'gregorian'
            time_var[:] = (merged['date'] - time_ref).dt.total_seconds() / 86400.0

            # 变量: lat / lon（标量）
            lat = ds.createVariable('lat', 'f4')
            lat.units = 'degrees_north'
            lat.standard_name = 'latitude'
            lat.long_name = 'station latitude'
            lon = ds.createVariable('lon', 'f4')
            lon.units = 'degrees_east'
            lon.standard_name = 'longitude'
            lon.long_name = 'station longitude'
            lat[:] = lat_val
            lon[:] = lon_val

            # Q
            q = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
            q.units = 'm3 s-1'
            q.standard_name = 'river_discharge'
            q.long_name = 'River Discharge'
            q.ancillary_variables = 'Q_flag'
            q[:] = merged['q'].values

            q_flag = ds.createVariable('Q_flag', 'b', ('time',))
            q_flag.long_name = "Quality flag for River Discharge"
            q_flag.flag_values = [0, 2, 3, 9]
            q_flag.flag_meanings = "good_data suspect_data bad_data missing_data"
            q_flag[:] = merged['q_flag'].values

            # SSC
            ssc = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
            ssc.units = 'mg L-1'
            ssc.standard_name = 'suspended_sediment_concentration'
            ssc.long_name = 'Suspended Sediment Concentration'
            ssc.ancillary_variables = 'SSC_flag'
            ssc[:] = merged['ssc'].values

            ssc_flag = ds.createVariable('SSC_flag', 'b', ('time',))
            ssc_flag.long_name = "Quality flag for Suspended Sediment Concentration"
            ssc_flag.flag_values = [0, 2, 3, 9]
            ssc_flag.flag_meanings = "good_data suspect_data bad_data missing_data"
            ssc_flag[:] = merged['ssc_flag'].values

            # SSL
            ssl = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
            ssl.units = 'ton day-1'
            ssl.long_name = 'Suspended Sediment Load'
            ssl.ancillary_variables = 'SSL_flag'
            ssl.comment = "Source: Calculated. Formula: SSL = Q * SSC * 0.0864"
            ssl[:] = merged['ssl'].values

            ssl_flag = ds.createVariable('SSL_flag', 'b', ('time',))
            ssl_flag.long_name = "Quality flag for Suspended Sediment Load"
            ssl_flag.flag_values = [0, 3, 9]
            ssl_flag.flag_meanings = "good_data bad_data missing_data"
            ssl_flag[:] = merged['ssl_flag'].values
            # ------------------------------
            # step/provenance flag variables
            # ------------------------------
            def _add_step_flag(name, values, *, flag_values, flag_meanings, long_name):
                v = ds.createVariable(name, 'b', ('time',), fill_value=np.int8(9))
                v.long_name = long_name
                v.standard_name = 'status_flag'
                v.flag_values = np.array(flag_values, dtype='b')
                v.flag_meanings = flag_meanings
                v.missing_value = np.int8(9)
                v[:] = np.asarray(values, dtype=np.int8)
                return v

            # Q step flags
            if 'Q_flag_qc1_physical' in merged.columns:
                _add_step_flag(
                    'Q_flag_qc1_physical', merged['Q_flag_qc1_physical'].fillna(9).values,
                    flag_values=[0, 3, 9],
                    flag_meanings='pass bad missing',
                    long_name='QC1 physical flag for river discharge',
                )
            if 'Q_flag_qc2_log_iqr' in merged.columns:
                _add_step_flag(
                    'Q_flag_qc2_log_iqr', merged['Q_flag_qc2_log_iqr'].fillna(9).values,
                    flag_values=[0, 2, 8, 9],
                    flag_meanings='pass suspect not_checked missing',
                    long_name='QC2 log-IQR flag for river discharge',
                )

            # SSC step flags
            if 'SSC_flag_qc1_physical' in merged.columns:
                _add_step_flag(
                    'SSC_flag_qc1_physical', merged['SSC_flag_qc1_physical'].fillna(9).values,
                    flag_values=[0, 3, 9],
                    flag_meanings='pass bad missing',
                    long_name='QC1 physical flag for suspended sediment concentration',
                )
            if 'SSC_flag_qc2_log_iqr' in merged.columns:
                _add_step_flag(
                    'SSC_flag_qc2_log_iqr', merged['SSC_flag_qc2_log_iqr'].fillna(9).values,
                    flag_values=[0, 2, 8, 9],
                    flag_meanings='pass suspect not_checked missing',
                    long_name='QC2 log-IQR flag for suspended sediment concentration',
                )
            if 'SSC_flag_qc3_ssc_q' in merged.columns:
                _add_step_flag(
                    'SSC_flag_qc3_ssc_q', merged['SSC_flag_qc3_ssc_q'].fillna(9).values,
                    flag_values=[0, 2, 8, 9],
                    flag_meanings='pass suspect not_checked missing',
                    long_name='QC3 SSC-Q consistency flag for suspended sediment concentration',
                )

            # SSL step flags
            if 'SSL_flag_qc1_physical' in merged.columns:
                _add_step_flag(
                    'SSL_flag_qc1_physical', merged['SSL_flag_qc1_physical'].fillna(9).values,
                    flag_values=[0, 3, 9],
                    flag_meanings='pass bad missing',
                    long_name='QC1 physical flag for suspended sediment load',
                )
            if 'SSL_flag_qc2_log_iqr' in merged.columns:
                _add_step_flag(
                    'SSL_flag_qc2_log_iqr', merged['SSL_flag_qc2_log_iqr'].fillna(9).values,
                    flag_values=[0, 2, 8, 9],
                    flag_meanings='pass suspect not_checked missing',
                    long_name='QC2 log-IQR flag for suspended sediment load',
                )
            if 'SSL_flag_qc3_from_ssc_q' in merged.columns:
                _add_step_flag(
                    'SSL_flag_qc3_from_ssc_q', merged['SSL_flag_qc3_from_ssc_q'].fillna(9).values,
                    flag_values=[0, 1, 8, 9],
                    flag_meanings='not_propagated propagated not_checked missing',
                    long_name='QC3 propagated inconsistency flag for suspended sediment load',
                )

            # update ancillary_variables (final + step flags)
            q.ancillary_variables = 'Q_flag'
            if 'Q_flag_qc1_physical' in merged.columns: q.ancillary_variables += ' Q_flag_qc1_physical'
            if 'Q_flag_qc2_log_iqr' in merged.columns:   q.ancillary_variables += ' Q_flag_qc2_log_iqr'

            ssc.ancillary_variables = 'SSC_flag'
            if 'SSC_flag_qc1_physical' in merged.columns: ssc.ancillary_variables += ' SSC_flag_qc1_physical'
            if 'SSC_flag_qc2_log_iqr' in merged.columns:  ssc.ancillary_variables += ' SSC_flag_qc2_log_iqr'
            if 'SSC_flag_qc3_ssc_q' in merged.columns:    ssc.ancillary_variables += ' SSC_flag_qc3_ssc_q'

            ssl.ancillary_variables = 'SSL_flag'
            if 'SSL_flag_qc1_physical' in merged.columns:  ssl.ancillary_variables += ' SSL_flag_qc1_physical'
            if 'SSL_flag_qc2_log_iqr' in merged.columns:   ssl.ancillary_variables += ' SSL_flag_qc2_log_iqr'
            if 'SSL_flag_qc3_from_ssc_q' in merged.columns: ssl.ancillary_variables += ' SSL_flag_qc3_from_ssc_q'

            # 一些全局属性（你可按需改）
            ds.title = 'Harmonized Global River Discharge and Sediment'
            ds.data_source_name = 'Rhine Dataset'
            ds.station_name = station_name
            ds.river_name = 'Rhine'
            ds.source_id = station_name
            ds.type = 'In-situ station data'
            ds.temporal_resolution = 'daily'
            ds.temporal_span = f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
            ds.geographic_coverage = 'Rhine River Basin'
            ds.variables_provided = "Q, SSC, SSL"
            ds.reference = 'Slabon, A., Terweh, S. and Hoffmann, T.O. (2025), Vertical and Lateral Variability of Suspended Sediment Transport in the Rhine River. Hydrological Processes, 39: e70070. https://doi.org/10.1002/hyp.70070'
            ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by process_rhine.py'
            ds.summary = 'This dataset contains daily river discharge and sediment data for the Rhine River.'
            ds.creator_name = 'Zhongwang Wei'
            ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
            ds.creator_institution = 'Sun Yat-sen University, China'
            ds.conventions = 'CF-1.8, ACDD-1.3'


        # errors, warnings_nc = check_nc_completeness(nc_file_path, strict=True)

        # if errors:
        #     print(f"  ✗ Completeness check FAILED for {station_name}")
        #     for e in errors:
        #         print(f"    ERROR: {e}")
        #     os.remove(nc_file_path)
        #     continue

        # if warnings_nc:
        #     print(f"  ⚠ Completeness warnings for {station_name}")
        #     for w in warnings_nc:
        #         print(f"    WARNING: {w}")


        # ---------- 站点统计（可选） ----------
        q_good = merged[merged['q_flag'] == 0]
        ssc_good = merged[merged['ssc_flag'] == 0]
        ssl_good = merged[merged['ssl_flag'] == 0]

        station_summary_data.append({
            'Source_ID': station_name,
            'station_name': station_name,
            'river_name': 'Rhine',
            'longitude': lon_val if lon_val != -9999.0 else '',
            'latitude': lat_val if lat_val != -9999.0 else '',
            'altitude': '',
            'upstream_area': '',
            'Q_start_date': q_good['date'].min().strftime('%Y-%m-%d') if not q_good.empty else '',
            'Q_end_date': q_good['date'].max().strftime('%Y-%m-%d') if not q_good.empty else '',
            'Q_percent_complete': (len(q_good) / len(merged)) * 100 if not merged.empty else 0,
            'SSC_start_date': ssc_good['date'].min().strftime('%Y-%m-%d') if not ssc_good.empty else '',
            'SSC_end_date': ssc_good['date'].max().strftime('%Y-%m-%d') if not ssc_good.empty else '',
            'SSC_percent_complete': (len(ssc_good) / len(merged)) * 100 if not merged.empty else 0,
            'SSL_start_date': ssl_good['date'].min().strftime('%Y-%m-%d') if not ssl_good.empty else '',
            'SSL_end_date': ssl_good['date'].max().strftime('%Y-%m-%d') if not ssl_good.empty else '',
            'SSL_percent_complete': (len(ssl_good) / len(merged)) * 100 if not merged.empty else 0,
        })

    # 站点 summary CSV（用 tool.py 新函数）
    if station_summary_data:
        summary_path = os.path.join(output_path, 'Rhine_station_summary.csv')
        generate_csv_summary_tool(station_summary_data, summary_path)

    # 站点 QC 结果汇总 CSV（用 tool.py 新函数）
    if qc_results_rows:
        qc_path = os.path.join(output_path, 'Rhine_qc_results_summary.csv')
        generate_qc_results_csv_tool(qc_results_rows, qc_path)

    print("\n✅ 所有站点处理完成。")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    process_rhine_data(
        output_path=os.path.join(
            PROJECT_ROOT, "Output_r", "daily", "Rhine", "qc"
        ),
        source_path=os.path.join(
            PROJECT_ROOT, "Source", "Rhine"
        )
    )
