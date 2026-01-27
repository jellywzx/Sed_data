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
    compute_log_iqr_bounds,
    build_ssc_q_envelope,
    check_ssc_q_consistency,
    plot_ssc_q_diagnostic,
    convert_ssl_units_if_needed,
    propagate_ssc_q_inconsistency_to_ssl
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
    """
    Unified QC using tool.py:
    - Physical validity
    - log-IQR screening
    - SSC–Q hydrological consistency
    """

    n = len(time)

    # -----------------------------
    # 1. Physical QC
    # -----------------------------
    Q_flag = np.array([apply_quality_flag(v, "Q") for v in Q], dtype=np.int8)
    SSC_flag = np.array([apply_quality_flag(v, "SSC") for v in SSC], dtype=np.int8)
    SSL_flag = np.array([apply_quality_flag(v, "SSL") for v in SSL], dtype=np.int8)

    # -----------------------------
    # 2. log-IQR screening
    # -----------------------------
    q_bounds = compute_log_iqr_bounds(Q)
    if q_bounds[0] is not None:
        Q_flag[(Q < q_bounds[0]) | (Q > q_bounds[1])] = 2

    ssc_bounds = compute_log_iqr_bounds(SSC)
    if ssc_bounds[0] is not None:
        SSC_flag[(SSC < ssc_bounds[0]) | (SSC > ssc_bounds[1])] = 2

    # -----------------------------
    # 3. SSC–Q consistency
    # -----------------------------
    ssc_q_bounds = build_ssc_q_envelope(Q, SSC)

    if ssc_q_bounds is not None:
        for i in range(n):
            inconsistent, _ = check_ssc_q_consistency(
                Q[i], SSC[i],
                Q_flag[i], SSC_flag[i],
                ssc_q_bounds
            )

            if inconsistent and SSC_flag[i] == 0:
                SSC_flag[i] = 2  # suspect

                SSL_flag[i] = propagate_ssc_q_inconsistency_to_ssl(
                    inconsistent=inconsistent,
                    Q=Q[i],
                    SSC=SSC[i],
                    SSL=SSL[i],
                    Q_flag=Q_flag[i],
                    SSC_flag=SSC_flag[i],
                    SSL_flag=SSL_flag[i],
                    ssl_is_derived_from_q_ssc=True,
                )


    # -----------------------------
    # 4. Keep valid rows
    # -----------------------------
    valid = (
        (Q_flag != FILL_VALUE_INT)
        | (SSC_flag != FILL_VALUE_INT)
        | (SSL_flag != FILL_VALUE_INT)
    )

    if not np.any(valid):
        return None

    # Optional: create diagnostic plot if envelope was constructed
    if plot_dir is not None and ssc_q_bounds is not None:
        try:
            os.makedirs(plot_dir, exist_ok=True)
            out_png = os.path.join(plot_dir, f"{station_id}_ssc_q.png")
            plot_ssc_q_diagnostic(
                time=time,
                Q=Q,
                SSC=SSC,
                Q_flag=Q_flag,
                SSC_flag=SSC_flag,
                ssc_q_bounds=ssc_q_bounds,
                station_id=station_id,
                station_name=station_name,
                out_png=out_png,
            )
        except Exception:
            # Plotting must not break processing
            pass

    return {
        "date": time[valid],
        "Q": Q[valid],
        "SSC": SSC[valid],
        "SSL": SSL[valid],
        "Q_flag": Q_flag[valid],
        "SSC_flag": SSC_flag[valid],
        "SSL_flag": SSL_flag[valid],
    }


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

        qc = apply_tool_qc(
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

    # 站点 summary CSV
    if station_summary_data:
        summary_df = pd.DataFrame(station_summary_data)
        summary_path = os.path.join(output_path, 'Rhine_station_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n站点统计已写入:", summary_path)

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
