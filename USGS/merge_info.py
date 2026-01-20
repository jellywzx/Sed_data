import pandas as pd
from pathlib import Path

# === 设置路径 ===
base_dir = Path(r"/mnt/d/sediment_data/Source/USGS/usgs_data_by_station")
output_file = base_dir.parent / "common_sites_info.csv"

# === 查找所有站点文件夹 ===
station_dirs = sorted(base_dir.glob("station_*"))

all_info = []

for station_dir in station_dirs:
    station_id = station_dir.name.split("_")[1]        # 保留前导零（来自文件夹名）
    info_file = station_dir / f"{station_id}_info.csv"
    if info_file.exists():
        try:
            df = pd.read_csv(info_file, low_memory=False,dtype=str)
            if df.empty:
                print(f"⚠️ Empty info file for {station_id}")
                continue

            # --- 记录原始 site_no（若存在），并统一覆盖为 station_id ---
            orig_site_no = None
            if 'site_no' in df.columns:
                # 仅用于比对提示，不改变逻辑
                orig_site_no = df['site_no'].astype(str).iloc[0]

            # 不纠结 dtype/前导零，直接对齐到文件夹名
            df['site_no'] = station_id

            if orig_site_no is not None and orig_site_no != station_id:
                print(f"ℹ️ site_no corrected: '{orig_site_no}' -> '{station_id}' for {station_id}")

            all_info.append(df)
            print(f"✅ Loaded info for station {station_id}")

        except Exception as e:
            print(f"❌ Failed to read {info_file}: {e}")
    else:
        print(f"🚫 No info.csv found for {station_id}")

# === 合并所有 info.csv ===
if all_info:
    combined_df = pd.concat(all_info, ignore_index=True)

    # 可选：去重（以 site_no 为主键）
    if 'site_no' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['site_no'])

    # 可选：统一列名小写，便于后续匹配
    combined_df.columns = combined_df.columns.str.strip()

    combined_df["site_no"] = combined_df["site_no"].astype(str)
    combined_df.to_excel(output_file.with_suffix(".xlsx"), index=False)

    print(f"\n✅ common_sites_info.xlsx saved to: {output_file}")
    print(f"Total stations combined: {len(all_info)}")
else:
    print("\n⚠️ No valid info files found, nothing was saved.")
