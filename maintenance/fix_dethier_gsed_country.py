#!/usr/bin/env python3
"""
Reverse-geocode Dethier and GSED NC files using naturalearth_lowres shapefile
to populate country/continent_region/iso_a3 from lat/lon coordinates.

Multi-core: reads coords in parallel -> batch spatial join (single) -> writes in parallel.
"""
import netCDF4 as nc
import geopandas as gpd
import glob, os, time, sys
from shapely.geometry import Point
from multiprocessing import get_context, Pool


ROOT = "/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r"
WORLD_SHP = "/share/home/dq134/.local/share/mamba/envs/delineator310/lib/python3.10/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp"


def _extract_coords(filepath):
    """Read lat/lon from a single NC file. Returns (filepath, lat, lon) or (filepath, None, None)."""
    try:
        ds = nc.Dataset(filepath, "r")
        lat = lon = None
        for vn in ("lat", "latitude"):
            if vn in ds.variables and ds.variables[vn].size > 0:
                lat = float(ds.variables[vn][0])
                break
        if lat is None:
            lat = getattr(ds, "lat", None) or getattr(ds, "latitude", None)
            lat = float(lat) if lat is not None else None

        for vn in ("lon", "longitude"):
            if vn in ds.variables and ds.variables[vn].size > 0:
                lon = float(ds.variables[vn][0])
                break
        if lon is None:
            lon = getattr(ds, "lon", None) or getattr(ds, "longitude", None)
            lon = float(lon) if lon is not None else None

        ds.close()
        if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
            return (filepath, lat, lon)
    except Exception:
        pass
    return (filepath, None, None)


def _write_attrs(args):
    """Write country/continent/iso_a3 to a single NC file. Returns True on success."""
    filepath, country, continent, iso_a3 = args
    if not country:
        return False
    try:
        ds = nc.Dataset(filepath, "r+")
        ds.country = country
        ds.continent_region = continent
        ds.iso_a3 = iso_a3
        ds.close()
        return True
    except Exception:
        return False


def fix_nc_files_parallel(nc_files, world, workers=8, batch_name=""):
    """Parallel read -> single spatial join -> parallel write."""
    if not nc_files:
        return 0

    n = len(nc_files)
    batch_name = batch_name or f"{n} files"

    # Phase 1: Read all coords in parallel
    print(f"  [{batch_name}] Reading coordinates ({n} files, {workers} workers)...")
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        raw = pool.map(_extract_coords, nc_files)
    t1 = time.time()
    print(f"  [{batch_name}] Read {n} files in {t1-t0:.1f}s")

    # Build spatial join input
    valid = [(p, la, lo) for p, la, lo in raw if la is not None]
    if not valid:
        print(f"  [{batch_name}] No valid coordinates found!")
        return 0

    n_valid = len(valid)
    pts = [Point(lo, la) for _, la, lo in valid]
    print(f"  [{batch_name}] Spatial join {n_valid} points with world boundaries...")
    t2 = time.time()
    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    joined = gpd.sjoin(gdf, world[["name", "iso_a3", "continent", "geometry"]],
                       how="left", predicate="within")
    t3 = time.time()
    print(f"  [{batch_name}] Spatial join done in {t3-t2:.1f}s")

    # Phase 2: Write back in parallel
    write_batch = []
    for i, (fp, _, _) in enumerate(valid):
        country = joined.iloc[i].get("name", "")
        continent = joined.iloc[i].get("continent", "")
        iso = joined.iloc[i].get("iso_a3", "")
        write_batch.append((fp, country, continent, iso))

    ocean_pts = sum(1 for _, c, _, _ in write_batch if not c)
    if ocean_pts:
        print(f"  [{batch_name}] {ocean_pts}/{n_valid} points in ocean (country will remain empty)")

    print(f"  [{batch_name}] Writing {n_valid} results ({workers} workers)...")
    with ctx.Pool(workers) as pool:
        results = pool.map(_write_attrs, write_batch)
    t4 = time.time()
    ok = sum(results)
    print(f"  [{batch_name}] Written {ok}/{n} files in {t4-t3:.1f}s")
    return ok


def main():
    # Detect core count
    try:
        workers = len(os.sched_getaffinity(0))
    except AttributeError:
        workers = os.cpu_count() or 4
    workers = max(1, workers - 1)

    print(f"Using {workers} workers")
    print("Loading world boundaries...")
    world = gpd.read_file(WORLD_SHP)
    world = world.to_crs("EPSG:4326")
    print(f"Loaded {len(world)} countries")

    # Dethier
    dethier_files = sorted(glob.glob(os.path.join(ROOT, "*/Dethier/qc/*.nc")))
    if dethier_files:
        print(f"\n=== Dethier ({len(dethier_files)} files) ===")
        ok = fix_nc_files_parallel(dethier_files, world, workers=workers, batch_name="Dethier")
        print(f"  Result: {ok}/{len(dethier_files)}")
    else:
        print("\n=== Dethier: no files found ===")

    # GSED
    gsed_files = sorted(glob.glob(os.path.join(ROOT, "*/GSED/qc/*.nc")))
    if gsed_files:
        print(f"\n=== GSED ({len(gsed_files)} files) ===")
        ok = fix_nc_files_parallel(gsed_files, world, workers=workers, batch_name="GSED")
        print(f"  Result: {ok}/{len(gsed_files)}")
    else:
        print("\n=== GSED: no files found ===")

    print("\nDone!")


if __name__ == "__main__":
    main()
