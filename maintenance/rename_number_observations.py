#!/share/home/dq134/.conda/envs/wzx/bin/python3
"""Rename number_of_observations -> number_of_data across all QC NetCDF files.
Uses 'spawn' to avoid GPFS+fork deadlocks."""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os, sys, glob, time, csv

NWORKERS = 8
QC_ROOT = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r"
OLD_ATTR = "number_of_observations"
NEW_ATTR = "number_of_data"


def process_file(fpath):
    """Rename number_of_observations to number_of_data in a single NC file."""
    import netCDF4 as nc4
    try:
        ds = nc4.Dataset(fpath, "a")
        changed = False
        old_value = None
        nc_attrs = list(ds.ncattrs())
        if OLD_ATTR in nc_attrs:
            old_value = str(getattr(ds, OLD_ATTR))
            ds.delncattr(OLD_ATTR)
            ds.setncattr(NEW_ATTR, old_value)
            changed = True
        ds.close()
        if changed:
            return {"file": fpath, "changed": True, "old_value": old_value}
        return {"file": fpath, "changed": False, "old_value": None}
    except Exception as e:
        return {"file": fpath, "changed": False, "old_value": None, "error": str(e)}


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

    print(f"Total files: {len(all_files)}", flush=True)
    print(f"Workers    : {NWORKERS}", flush=True)

    pool = multiprocessing.Pool(NWORKERS)
    results = []
    try:
        for i, r in enumerate(pool.imap_unordered(process_file, all_files, chunksize=50)):
            results.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  [{time.time()-t0:.1f}s] Processed {i+1}/{len(all_files)}", flush=True)
    finally:
        pool.close()
        pool.join()

    changed = [r for r in results if r["changed"]]
    errors = [r for r in results if r.get("error")]
    unchanged = [r for r in results if not r["changed"] and not r.get("error")]

    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)
    print(f"  Changed ({OLD_ATTR} -> {NEW_ATTR}): {len(changed)} files", flush=True)
    print(f"  No change: {len(unchanged)} files", flush=True)
    if errors:
        print(f"  Errors: {len(errors)} files", flush=True)
        for e in errors[:5]:
            print(f"    {e['file']}: {e['error']}", flush=True)

    csv_path = os.path.join(os.path.dirname(__file__), "rename_number_observations_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "old_value", "error"])
        for r in changed:
            rel = os.path.relpath(r["file"], QC_ROOT)
            w.writerow([rel, "renamed", r["old_value"], ""])
        if errors:
            for e in errors:
                rel = os.path.relpath(e["file"], QC_ROOT)
                w.writerow([rel, "error", "", e.get("error", "")])
    print(f"Report: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
