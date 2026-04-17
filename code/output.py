import os
from collections import Counter

import numpy as np
import pandas as pd


def summarize_warning_types(stations_info):
    """
    Summarize and print warning types across stations.
    """
    counter = Counter()

    for station in stations_info:
        warnings = station.get("warnings")
        if warnings:
            for warning in warnings.split(" | "):
                if warning:
                    counter[warning] += 1

    print("\nMetadata warning type summary:")
    for warning, count in counter.most_common():
        print("  {0:4d} x {1}".format(count, warning))

    return counter


def generate_csv_summary(stations_info, output_csv, column_order=None):
    """Generate a station metadata summary CSV."""
    print("\n生成CSV摘要文件: {0}".format(output_csv))

    if not stations_info:
        print("  ⚠ 警告: 无站点信息可写入CSV")
        return

    df = pd.DataFrame(stations_info)

    if column_order is None:
        column_order = [
            "station_name",
            "Source_ID",
            "river_name",
            "comid",
            "reach_code",
            "vpu_id",
            "rpu_id",
            "longitude",
            "latitude",
            "altitude",
            "upstream_area",
            "Data Source Name",
            "Type",
            "Temporal Resolution",
            "Temporal Span",
            "Variables Provided",
            "Geographic Coverage",
            "Reference/DOI",
            "Q_start_date",
            "Q_end_date",
            "Q_percent_complete",
            "SSC_start_date",
            "SSC_end_date",
            "SSC_percent_complete",
            "SSL_start_date",
            "SSL_end_date",
            "SSL_percent_complete",
        ]

    ordered_cols = [col for col in column_order if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    if ordered_cols:
        df = df[ordered_cols + remaining_cols]

    df.to_csv(output_csv, index=False)
    print("  ✓ CSV文件已生成: {0} 个站点".format(len(df)))


def generate_qc_results_csv(stations_info, output_csv, preferred_cols=None):
    """Generate per-station QC results summary CSV."""
    print("\n生成站点QC结果汇总CSV: {0}".format(output_csv))

    if not stations_info:
        print("  ⚠ 警告: 无站点信息可写入CSV")
        return

    df = pd.DataFrame(stations_info)

    if preferred_cols is None:
        preferred_cols = [
            "station_name",
            "Source_ID",
            "river_name",
            "comid",
            "reach_code",
            "vpu_id",
            "rpu_id",
            "longitude",
            "latitude",
            "upstream_area",
            "Geographic Coverage",
            "QC_n_days",
            "Q_final_good",
            "Q_final_estimated",
            "Q_final_suspect",
            "Q_final_bad",
            "Q_final_missing",
            "SSC_final_good",
            "SSC_final_estimated",
            "SSC_final_suspect",
            "SSC_final_bad",
            "SSC_final_missing",
            "SSL_final_good",
            "SSL_final_estimated",
            "SSL_final_suspect",
            "SSL_final_bad",
            "SSL_final_missing",
            "Q_qc1_pass",
            "Q_qc1_bad",
            "Q_qc1_missing",
            "SSC_qc1_pass",
            "SSC_qc1_bad",
            "SSC_qc1_missing",
            "SSL_qc1_pass",
            "SSL_qc1_bad",
            "SSL_qc1_missing",
            "Q_qc2_pass",
            "Q_qc2_suspect",
            "Q_qc2_not_checked",
            "Q_qc2_missing",
            "SSC_qc2_pass",
            "SSC_qc2_suspect",
            "SSC_qc2_not_checked",
            "SSC_qc2_missing",
            "SSL_qc2_pass",
            "SSL_qc2_suspect",
            "SSL_qc2_not_checked",
            "SSL_qc2_missing",
            "SSC_qc3_pass",
            "SSC_qc3_suspect",
            "SSC_qc3_not_checked",
            "SSC_qc3_missing",
            "SSL_qc3_not_propagated",
            "SSL_qc3_propagated",
            "SSL_qc3_not_checked",
            "SSL_qc3_missing",
        ]

    cols = [col for col in preferred_cols if col in df.columns]
    if cols:
        df = df[cols]

    df.to_csv(output_csv, index=False)
    print("  ✓ QC结果CSV已生成: {0} 个站点".format(len(df)))


def generate_warning_summary_csv(stations_info, output_csv):
    """Generate a warning summary CSV for stations with metadata warnings."""
    if not stations_info:
        print("⚠ No station info available for warning summary.")
        return

    rows = []
    for station in stations_info:
        if station.get("n_warnings", 0) > 0:
            rows.append(
                {
                    "station_name": station.get("station_name", ""),
                    "Source_ID": station.get("Source_ID", ""),
                    "n_warnings": station.get("n_warnings", 0),
                    "warnings": station.get("warnings", ""),
                }
            )

    if not rows:
        print("✓ No warnings found across all stations.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("n_warnings", ascending=False, inplace=True)
    df.to_csv(output_csv, index=False)
    print("✓ Warning summary CSV written: {0} ({1} stations)".format(output_csv, len(df)))


def generate_station_summary_csv(
    stations_info,
    output_dir,
    *,
    raw_df=None,
    stations=None,
    event_col="event",
    year_col="year",
    vars_cols=("Q", "SSC", "SSL"),
    output_filename=None,
    dataset_name=None,
    data_type="In-situ station data",
    temporal_resolution="daily",
    geographic_coverage=None,
    reference_doi=None,
    extra_columns=None,
    include_all_stations=True,
    encoding="utf-8",
):
    """
    Combined station summary CSV (coverage + QC usability) - generic version.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _get(d, keys, default=None):
        for key in keys:
            if key in d:
                value = d.get(key)
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                return value
        return default

    def _to_float(value):
        try:
            if value is None:
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    def _to_int(value):
        try:
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return None
            return int(float(value))
        except Exception:
            return None

    def _fmt_coord(value, nd=6):
        val = _to_float(value)
        if not np.isfinite(val):
            return ""
        return ("{0:." + str(nd) + "f}").format(val)

    def _fmt_num(value, nd=1):
        val = _to_float(value)
        if not np.isfinite(val):
            return ""
        return ("{0:." + str(nd) + "f}").format(val)

    def _calc_qc_good_percent(d, var_prefix):
        good = _to_int(_get(d, ["{0}_final_good".format(var_prefix)]))
        est = _to_int(_get(d, ["{0}_final_estimated".format(var_prefix)]))
        sus = _to_int(_get(d, ["{0}_final_suspect".format(var_prefix)]))
        bad = _to_int(_get(d, ["{0}_final_bad".format(var_prefix)]))
        miss = _to_int(_get(d, ["{0}_final_missing".format(var_prefix)]))

        if good is not None and miss is not None:
            total = sum(
                [x for x in [good, est, sus, bad, miss] if x is not None]
            )
            if total > 0:
                return 100.0 * good / total
            return 0.0

        direct = _get(
            d,
            [
                "{0}_percent_complete".format(var_prefix),
                "{0}_pct_complete".format(var_prefix),
            ],
        )
        if direct is not None:
            try:
                return float(direct)
            except Exception:
                pass

        flag = _get(
            d,
            [
                "{0}_flag".format(var_prefix),
                "{0}_flag".format(var_prefix.lower()),
            ],
        )
        try:
            return 100.0 if int(flag) == 0 else 0.0
        except Exception:
            return 0.0

    def _infer_vars_provided(qp, sp, lp):
        vars_ = []
        if qp > 0:
            vars_.append("Q")
        if sp > 0:
            vars_.append("SSC")
        if lp > 0:
            vars_.append("SSL")
        return ", ".join(vars_) if vars_ else ""

    coverage_by_sourceid = {}
    if raw_df is not None and stations is not None:
        if not isinstance(raw_df, pd.DataFrame):
            raise TypeError("raw_df must be a pandas DataFrame or None.")
        for event_id, meta in stations.items():
            sid = str(meta.get("station_id", "")).strip()
            if sid == "":
                continue
            source_id = sid.replace(".", "_")

            sub = raw_df[raw_df[event_col] == event_id]
            if sub.empty:
                continue

            n_total = len(sub)
            cov = {"QC_n_days_raw": int(n_total), "Temporal Span (raw)": ""}

            try:
                years_all = pd.to_numeric(sub[year_col], errors="coerce")
                if np.isfinite(years_all).any():
                    cov["Temporal Span (raw)"] = "{0}-{1}".format(
                        int(np.nanmin(years_all)),
                        int(np.nanmax(years_all)),
                    )
            except Exception:
                pass

            for var_name in vars_cols:
                if var_name not in sub.columns:
                    cov["{0}_cov_start_date".format(var_name)] = ""
                    cov["{0}_cov_end_date".format(var_name)] = ""
                    cov["{0}_cov_percent_complete".format(var_name)] = "0.0"
                    continue

                series = sub[var_name]
                valid = series.dropna()
                if valid.empty:
                    cov["{0}_cov_start_date".format(var_name)] = ""
                    cov["{0}_cov_end_date".format(var_name)] = ""
                    cov["{0}_cov_percent_complete".format(var_name)] = "0.0"
                else:
                    idx = valid.index
                    try:
                        years = pd.to_numeric(sub.loc[idx, year_col], errors="coerce")
                        cov["{0}_cov_start_date".format(var_name)] = (
                            int(np.nanmin(years)) if np.isfinite(years).any() else ""
                        )
                        cov["{0}_cov_end_date".format(var_name)] = (
                            int(np.nanmax(years)) if np.isfinite(years).any() else ""
                        )
                    except Exception:
                        cov["{0}_cov_start_date".format(var_name)] = ""
                        cov["{0}_cov_end_date".format(var_name)] = ""

                    cov_pct = 100.0 * len(valid) / n_total if n_total > 0 else 0.0
                    cov["{0}_cov_percent_complete".format(var_name)] = "{0:.1f}".format(cov_pct)

            cov.update(
                {
                    "station_name": sid,
                    "Source_ID": source_id,
                    "river_name": meta.get("river", ""),
                    "longitude": meta.get("lon", np.nan),
                    "latitude": meta.get("lat", np.nan),
                }
            )

            coverage_by_sourceid[source_id] = cov

    qc_by_sourceid = {}
    if stations_info:
        for item in stations_info:
            sid = _get(item, ["Source_ID", "source_id", "SourceID", "id"], "")
            if sid == "":
                name = _get(item, ["station_name", "station", "name"], "")
                sid = str(name).replace(".", "_") if name else ""
            if sid != "":
                qc_by_sourceid[sid] = item

    all_source_ids = set()
    all_source_ids.update(coverage_by_sourceid.keys())
    all_source_ids.update(qc_by_sourceid.keys())

    if not include_all_stations and stations_info:
        all_source_ids = set(qc_by_sourceid.keys())

    if not all_source_ids:
        print("  ⚠ generate_station_summary_csv: no stations to write.")
        return ""

    rows = []
    for source_id in sorted(all_source_ids):
        cov = coverage_by_sourceid.get(source_id, {})
        qc = qc_by_sourceid.get(source_id, {})

        station_name = _get(qc, ["station_name", "station", "name"], _get(cov, ["station_name"], ""))
        river_name = _get(qc, ["river_name", "river"], _get(cov, ["river_name"], ""))

        lon = _get(qc, ["longitude", "lon"], _get(cov, ["longitude"], np.nan))
        lat = _get(qc, ["latitude", "lat"], _get(cov, ["latitude"], np.nan))
        alt = _get(qc, ["altitude", "elevation", "alt"], np.nan)
        area = _get(qc, ["upstream_area", "drainage_area", "area_km2"], np.nan)

        q_qc_pct = _calc_qc_good_percent(qc, "Q")
        ssc_qc_pct = _calc_qc_good_percent(qc, "SSC")
        ssl_qc_pct = _calc_qc_good_percent(qc, "SSL")

        vars_col = _get(qc, ["Variables Provided", "variables_provided"], "")
        if not vars_col:
            q_cov_pct = float(_get(cov, ["Q_cov_percent_complete"], "0.0") or "0.0")
            s_cov_pct = float(_get(cov, ["SSC_cov_percent_complete"], "0.0") or "0.0")
            l_cov_pct = float(_get(cov, ["SSL_cov_percent_complete"], "0.0") or "0.0")
            base_q = q_qc_pct if qc else q_cov_pct
            base_s = ssc_qc_pct if qc else s_cov_pct
            base_l = ssl_qc_pct if qc else l_cov_pct
            vars_col = _infer_vars_provided(base_q, base_s, base_l)

        temporal_span = _get(cov, ["Temporal Span (raw)"], "")
        if not temporal_span:
            start_year = _get(qc, ["start_year", "Q_start_date"], "")
            end_year = _get(qc, ["end_year", "Q_end_date"], "")
            if str(start_year) != "" and str(end_year) != "":
                temporal_span = "{0}-{1}".format(start_year, end_year)
            else:
                temporal_span = _get(qc, ["Temporal Span", "temporal_span"], "")

        geo_cov = (
            geographic_coverage
            if geographic_coverage is not None
            else _get(qc, ["Geographic Coverage", "geographic_coverage"], "")
        )
        ref = (
            reference_doi
            if reference_doi is not None
            else _get(qc, ["Reference/DOI", "reference", "doi", "source_data_link"], "")
        )
        ds_name = (
            dataset_name
            if dataset_name is not None
            else _get(qc, ["Data Source Name", "data_source_name"], "")
        )

        row = {
            "station_name": station_name,
            "Source_ID": source_id,
            "river_name": river_name,
            "longitude": _fmt_coord(lon),
            "latitude": _fmt_coord(lat),
            "altitude": _fmt_num(alt, nd=1),
            "upstream_area": _fmt_num(area, nd=1),
            "Data Source Name": ds_name,
            "Type": data_type,
            "Temporal Resolution": temporal_resolution,
            "Temporal Span": temporal_span,
            "Variables Provided": vars_col,
            "Geographic Coverage": geo_cov,
            "Reference/DOI": ref,
            "Q_percent_complete": "{0:.1f}".format(q_qc_pct),
            "SSC_percent_complete": "{0:.1f}".format(ssc_qc_pct),
            "SSL_percent_complete": "{0:.1f}".format(ssl_qc_pct),
        }

        for var_name in vars_cols:
            row["{0}_cov_start_date".format(var_name)] = _get(
                cov, ["{0}_cov_start_date".format(var_name)], ""
            )
            row["{0}_cov_end_date".format(var_name)] = _get(
                cov, ["{0}_cov_end_date".format(var_name)], ""
            )
            row["{0}_cov_percent_complete".format(var_name)] = _get(
                cov, ["{0}_cov_percent_complete".format(var_name)], "0.0"
            )

        for var_name in vars_cols:
            for suffix in ("good", "estimated", "suspect", "bad", "missing"):
                key = "{0}_final_{1}".format(var_name, suffix)
                if key in qc:
                    row[key] = qc.get(key)

        if "QC_n_days" in qc:
            row["QC_n_days"] = qc.get("QC_n_days")
        if "QC_n_days_raw" in cov:
            row["Raw_n_records"] = cov.get("QC_n_days_raw")

        if extra_columns:
            for col in extra_columns:
                if col not in row:
                    value = _get(qc, [col], None)
                    if value is None:
                        value = _get(cov, [col], None)
                    if value is not None:
                        row[col] = value

        rows.append(row)

    df_out = pd.DataFrame(rows)

    base_cols = [
        "station_name",
        "Source_ID",
        "river_name",
        "longitude",
        "latitude",
        "altitude",
        "upstream_area",
        "Data Source Name",
        "Type",
        "Temporal Resolution",
        "Temporal Span",
        "Variables Provided",
        "Geographic Coverage",
        "Reference/DOI",
        "Q_percent_complete",
        "SSC_percent_complete",
        "SSL_percent_complete",
    ]
    cov_cols = []
    for var_name in vars_cols:
        cov_cols += [
            "{0}_cov_start_date".format(var_name),
            "{0}_cov_end_date".format(var_name),
            "{0}_cov_percent_complete".format(var_name),
        ]
    qc_count_cols = []
    for var_name in vars_cols:
        qc_count_cols += [
            "{0}_final_good".format(var_name),
            "{0}_final_estimated".format(var_name),
            "{0}_final_suspect".format(var_name),
            "{0}_final_bad".format(var_name),
            "{0}_final_missing".format(var_name),
        ]

    front = [col for col in base_cols if col in df_out.columns]
    mid = [col for col in qc_count_cols if col in df_out.columns] + [
        col for col in ("QC_n_days", "Raw_n_records") if col in df_out.columns
    ]
    back = [col for col in cov_cols if col in df_out.columns]
    remaining = [col for col in df_out.columns if col not in (front + mid + back)]

    df_out = df_out[front + mid + back + remaining]

    if output_filename is None:
        base = "station_summary_combined.csv"
        if dataset_name and isinstance(dataset_name, str) and dataset_name.strip():
            base = "{0}_station_summary_combined.csv".format(
                dataset_name.replace(" ", "_")
            )
        output_filename = base

    out_csv = os.path.join(output_dir, output_filename)
    df_out.to_csv(out_csv, index=False, encoding=encoding)

    print(
        "  ✓ Combined station summary CSV written: {0} ({1} stations)".format(
            out_csv, len(df_out)
        )
    )
    return out_csv
