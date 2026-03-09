#!/usr/bin/env python3
"""
在保留原有 Output_r 目录结构的前提下，根据时间分辨率校验结果，
将「所有 qc 文件夹下的 .nc」按检测到的时间分辨率复制到新目录。

新目录结构：
  {out_dir}/
    daily/                    # 检测为 daily 或 hourly
    monthly/                  # 检测为 monthly
    annually_climatology/     # 检测为 annual 或 quarterly
    other/                    # 其余：single_point（且非 long_term_average）、irregular、no_time_var、error 等

single_point 的元数据判断：若 CSV 中 single_point_interpretation 以 long_term_average_ 开头
（由 verify_time_resolution.py 根据 nc 元数据识别为长时间历史平均），则归入 annually_climatology，否则归入 other。

每个分辨率目录下不按数据集分子文件夹，文件名为全库唯一，体现「数据源」和「时间分辨率」：
  {数据源}_{分辨率}_{原文件名无后缀}.nc
  若重名则追加 _2, _3, ...

依赖：需先运行 verify_time_resolution.py 生成 output/05_verify/verify_time_resolution_results.csv。

用法（数据根目录为 Script 同级的 Output_r，可在任意目录运行）：
  python Script/tool/06_verify/reorganize_qc_by_resolution.py
  python Script/tool/06_verify/reorganize_qc_by_resolution.py --out-dir my_reorganized
  python Script/tool/06_verify/reorganize_qc_by_resolution.py -j 16   # 16 线程并行复制
"""

import re
import shutil
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# 数据根目录 Output_r
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent / "Output_r"
# 校验结果 CSV（相对 ROOT_DIR）
VERIFY_CSV = "output/05_verify/verify_time_resolution_results.csv"
# 新目录名（相对 ROOT_DIR），仅包含 qc 下 nc 按分辨率整理后的副本
OUT_DIR = "output_resolution_organized"
# 并行复制时的默认线程数（I/O 为主，线程池即可）
DEFAULT_WORKERS = 8

# detected_frequency -> 新目录下的第一级目录名
FREQ_TO_RESOLUTION = {
    "daily": "daily",
    "hourly": "daily",
    "monthly": "monthly",
    "quarterly": "annually_climatology",
    "annual": "annually_climatology",
}
# 未在上述映射中的 -> "other"；single_point 若 single_point_interpretation 为 long_term_average_* 则改归 annually_climatology（在 main 中处理）


def get_source_from_path(path: str, root_dir: Path) -> str:
    """从相对路径解析数据源，如 daily/GloRiSe/SS/qc/xxx.nc -> GloRiSe_SS。"""
    try:
        p = Path(path).resolve()
        root = root_dir.resolve()
        rel = p.relative_to(root)
        parts = rel.parts
        if "qc" in parts:
            idx = parts.index("qc")
            before = parts[:idx]
            if len(before) >= 2:
                source = "_".join(before[1:])
            else:
                source = before[0] if before else "unknown"
        else:
            source = parts[0] if parts else "unknown"
        return re.sub(r"[^\w\-]", "_", source).strip("_") or "unknown"
    except Exception:
        return "unknown"


def safe_fname_part(s: str) -> str:
    """文件名安全：只保留字母数字下划线横线。"""
    return re.sub(r"[^\w\-]", "_", str(s)).strip("_") or "unknown"


def resolution_from_detected(detected_freq: str) -> str:
    """将 detected_frequency 映射到新目录名：daily / monthly / annually_climatology / other。"""
    if not detected_freq or not isinstance(detected_freq, str):
        return "other"
    d = detected_freq.strip().lower()
    if d in FREQ_TO_RESOLUTION:
        return FREQ_TO_RESOLUTION[d]
    return "other"


def _copy_one(item):
    """单次复制，供线程池调用。返回 (res_dir_name, None) 成功，(res_dir_name, (src, err)) 失败。"""
    src_path, dest_path, res_dir_name = item
    try:
        shutil.copy2(src_path, dest_path)
        return (res_dir_name, None)
    except Exception as e:
        return (res_dir_name, (str(src_path), str(e)))


def main():
    ap = argparse.ArgumentParser(description="按时间分辨率校验结果将 qc 下 nc 复制到新目录（数据源_分辨率_原名）")
    ap.add_argument("--out-dir", "-o", default=OUT_DIR, help=f"新目录名（相对 Output_r），默认 {OUT_DIR}")
    ap.add_argument("--verify-csv", default=VERIFY_CSV, help=f"校验结果 CSV 路径，默认 {VERIFY_CSV}")
    ap.add_argument("--clear", action="store_true", help="复制前清空输出目录下 daily/monthly/annually_climatology/other，避免残留旧分类文件")
    ap.add_argument("--workers", "-j", type=int, default=DEFAULT_WORKERS, metavar="N", help=f"并行复制线程数，默认 {DEFAULT_WORKERS}，设为 1 则串行")
    args = ap.parse_args()

    root_dir = Path(ROOT_DIR).resolve()
    if not root_dir.is_dir():
        print(f"错误：根目录不存在: {root_dir}", file=sys.stderr)
        sys.exit(1)

    verify_path = root_dir / args.verify_csv
    if not verify_path.is_file():
        print(f"错误：未找到校验结果 {verify_path}，请先运行 verify_time_resolution.py", file=sys.stderr)
        sys.exit(1)

    out_base = root_dir / args.out_dir
    for sub in ("daily", "monthly", "annually_climatology", "other"):
        (out_base / sub).mkdir(parents=True, exist_ok=True)

    if args.clear:
        for sub in ("daily", "monthly", "annually_climatology", "other"):
            d = out_base / sub
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
            print(f"已清空: {d}")

    df = pd.read_csv(verify_path)
    for col in ("path", "detected_frequency"):
        if col not in df.columns:
            print(f"错误：CSV 缺少列 {col}", file=sys.stderr)
            sys.exit(1)

    # 只处理路径中包含 qc 的 nc（qc 文件夹下的数据）
    def is_qc_path(path_str):
        if pd.isna(path_str):
            return False
        parts = Path(path_str).parts
        return "qc" in parts

    df = df[df["path"].apply(is_qc_path)].copy()
    df["resolution_dir"] = df["detected_frequency"].apply(resolution_from_detected)

    # single_point 的元数据判断：若 single_point_interpretation 以 long_term_average_ 开头，归入 annually_climatology
    if "single_point_interpretation" in df.columns:
        mask = (df["resolution_dir"] == "other") & (df["detected_frequency"] == "single_point")
        interp = df.loc[mask, "single_point_interpretation"].astype(str)
        to_annual = interp.str.strip().str.startswith("long_term_average_")
        df.loc[to_annual[to_annual].index, "resolution_dir"] = "annually_climatology"

    df["source"] = df["path"].apply(lambda p: get_source_from_path(p, root_dir))
    df["stem"] = df["path"].apply(lambda p: Path(p).stem)
    df["safe_source"] = df["source"].apply(safe_fname_part)
    df["safe_stem"] = df["stem"].apply(safe_fname_part)

    # 已使用的文件名（不含 .nc），按 resolution 目录记录，用于生成唯一名
    used = {}
    for r in ("daily", "monthly", "annually_climatology", "other"):
        used[r] = set()

    copied = {r: 0 for r in ("daily", "monthly", "annually_climatology", "other")}
    skipped = 0
    tasks = []  # (src_path, dest_path, res_dir_name)

    for _, row in df.iterrows():
        src_path = Path(row["path"])
        if not src_path.is_file():
            skipped += 1
            continue
        res_dir_name = row["resolution_dir"]
        res_dir = out_base / res_dir_name
        base = f"{row['safe_source']}_{res_dir_name}_{row['safe_stem']}"
        base_candidate = base
        idx = 2
        while base_candidate in used[res_dir_name]:
            base_candidate = f"{base}_{idx}"
            idx += 1
        used[res_dir_name].add(base_candidate)
        dest_path = res_dir / (base_candidate + ".nc")
        tasks.append((src_path, dest_path, res_dir_name))

    workers = max(1, int(args.workers))
    errors = []
    if workers == 1:
        for item in tasks:
            res_dir_name, err = _copy_one(item)
            if err:
                errors.append(err)
            else:
                copied[res_dir_name] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_copy_one, item): item for item in tasks}
            for fut in as_completed(futures):
                res_dir_name, err = fut.result()
                if err:
                    errors.append(err)
                else:
                    copied[res_dir_name] += 1

    print(f"新目录: {out_base}")
    print(f"已处理 qc 下 nc 数量: {len(df)}（跳过不存在: {skipped}）")
    for r in ("daily", "monthly", "annually_climatology", "other"):
        print(f"  {r}: {copied[r]} 个文件")
    if errors:
        print(f"复制失败 {len(errors)} 个:")
        for p, e in errors[:10]:
            print(f"  {p} -> {e}")
        if len(errors) > 10:
            print(f"  ... 共 {len(errors)} 个")
    else:
        print("全部复制完成。")

    # other 目录的数据集构成说明
    other_df = df[df["resolution_dir"] == "other"]
    if len(other_df) > 0:
        print("\n--- other 目录构成（未归入 daily/monthly/annually_climatology 的文件）---")
        print("按 detected_frequency 统计:")
        for freq, cnt in other_df["detected_frequency"].value_counts().items():
            print(f"  {freq}: {cnt} 个")
        single_in_other = other_df[other_df["detected_frequency"] == "single_point"]
        if len(single_in_other) > 0 and "single_point_interpretation" in other_df.columns:
            print("\nsingle_point 中留在 other 的 single_point_interpretation 统计（前 15 类）:")
            interp = single_in_other["single_point_interpretation"].fillna("").astype(str)
            for val, c in interp.value_counts().head(15).items():
                print(f"  {val or '(空)'}: {c} 个")
        print("---")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
