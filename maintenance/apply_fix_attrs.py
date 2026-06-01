#!/share/home/dq134/.conda/envs/wzx/bin/python3
"""Apply fix_qc_global_attrs for all QC files (non-dry-run).
Uses 'spawn' to avoid GPFS+fork deadlocks."""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os, sys, glob, time, csv
from collections import Counter

NWORKERS = 8
QC_ROOT = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r"
SCRIPT_DIR = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Script"


def process_one(path):
    """Apply normalize_nc_attrs to a single file."""
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from code.global_attrs import normalize_nc_attrs, HISTORY_NOTE

    try:
        parts = os.path.relpath(path, QC_ROOT).replace(os.sep, "/").split("/")
        resolution = parts[0] if len(parts) >= 1 else ""
        dataset = parts[1] if len(parts) >= 2 else ""
        result = normalize_nc_attrs(
            path, dataset_name=dataset, path_resolution=resolution,
            history_note=HISTORY_NOTE, dry_run=False,
        )
        result["resolution"] = resolution
        result["dataset"] = dataset
        return result
    except Exception as e:
        rel = os.path.relpath(path, QC_ROOT)
        return {
            "path": rel, "dataset": "", "resolution": "",
            "changed": False, "changed_keys": [], "removed_keys": [],
            "error": str(e),
        }


def main():
    t0 = time.time()
    all_files = []
    for res in sorted(os.listdir(QC_ROOT)):
        res_path = os.path.join(QC_ROOT, res)
        if not os.path.isdir(res_path) or res.startswith("_") or res.startswith("."):
            continue
        for ds_name in sorted(os.listdir(res_path)):
            qc_dir = os.path.join(res_path, ds_name, "qc")
            if not os.path.isdir(qc_dir):
                continue
            nc_files = sorted(glob.glob(os.path.join(qc_dir, "*.nc")))
            all_files.extend(nc_files)

    print(f"Source root : {QC_ROOT}")
    print(f"Files       : {len(all_files)}")
    print(f"Mode        : apply")
    print(f"Workers     : {NWORKERS}", flush=True)

    pool = multiprocessing.Pool(NWORKERS)
    results = []
    try:
        for i, r in enumerate(pool.imap_unordered(process_one, all_files, chunksize=10)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  [{time.time()-t0:.1f}s] Processed {i+1}/{len(all_files)}", flush=True)
    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"\nProcessing completed in {elapsed:.1f}s", flush=True)

    changed_items = [r for r in results if r.get("changed")]
    error_items = [r for r in results if r.get("error")]
    ok_items = [r for r in results if not r.get("changed") and not r.get("error")]

    print(f"  Changed: {len(changed_items)} files", flush=True)
    print(f"  Unchanged: {len(ok_items)} files", flush=True)
    if error_items:
        print(f"  Errors: {len(error_items)} files", flush=True)
        for e in error_items[:5]:
            print(f"    ERROR {e.get('path','')}: {e.get('error','')}", flush=True)

    changed_key_counts = Counter()
    removed_key_counts = Counter()
    for r in changed_items:
        for k in r.get("changed_keys", []):
            changed_key_counts[k] += 1
        for k in r.get("removed_keys", []):
            removed_key_counts[k] += 1

    if changed_key_counts:
        print(f"\n  Changed keys:", flush=True)
        for k, c in changed_key_counts.most_common(40):
            print(f"    {k:40s} {c:>6}", flush=True)
    if removed_key_counts:
        print(f"\n  Removed keys:", flush=True)
        for k, c in sorted(removed_key_counts.items(), key=lambda x: -x[1]):
            print(f"    {k:40s} {c:>6}", flush=True)

    csv_path = os.path.join(os.path.dirname(__file__), "apply_fix_attrs_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "dataset", "resolution", "changed", "changed_keys", "removed_keys", "error"])
        for r in results:
            w.writerow([
                r.get("path", ""),
                r.get("dataset", ""),
                r.get("resolution", ""),
                "1" if r.get("changed") else "0",
                "|".join(r.get("changed_keys", [])),
                "|".join(r.get("removed_keys", [])),
                r.get("error", ""),
            ])
    print(f"\nReport written: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
