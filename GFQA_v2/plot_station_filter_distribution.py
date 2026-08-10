#!/usr/bin/env python3
"""
GFQA_v2 station-distribution audit
==================================

Purpose
-------
Visualize and quantify the station-level filtering before/after requiring
both valid discharge (Q-Inst) and valid suspended sediment concentration (TSS).

Definitions
-----------
Valid Q record:
    Flux.csv
    Parameter.Code == "Q-Inst"
    Value can be converted to float and Value >= 0

Valid SSC record:
    Water.csv
    Parameter.Code == "TSS"
    Value can be converted to float and Value >= 0

Before filtering:
    stations with valid Q OR valid SSC

After filtering:
    stations with valid Q AND valid SSC

Map categories
--------------
1. Q only   -> removed
2. SSC only -> removed
3. Q + SSC  -> retained

The script also reports how many raw valid observations are lost solely
because their station is removed by this station-level intersection.

Note
----
This script evaluates the station-intersection stage only.
It does NOT yet count the additional loss caused by:
    - Q/SSC date overlap
    - same-day inner join
    - QC1/QC2/QC3
"""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Repository paths
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

from code.runtime import resolve_source_root, resolve_output_root


SOURCE_DIR = resolve_source_root(start=__file__) / "GFQA_v2" / "sed"

OUTPUT_DIR = (
    resolve_output_root(start=__file__)
    / "daily"
    / "GFQA_v2"
    / "qc"
    / "diagnostic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Optional Cartopy support
# ============================================================

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True

except ImportError:
    HAS_CARTOPY = False
    print(
        "WARNING: cartopy is not installed. "
        "A longitude-latitude scatter plot will be produced without coastlines."
    )


# ============================================================
# Helper functions
# ============================================================

def clean_numeric_series(series):
    """
    Reproduce the effective behavior of clean_value() in the GFQA script.

    - comma decimal separator -> dot
    - non-numeric -> NaN
    - negative -> NaN
    - zero is retained
    """
    values = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    values = values.where(values >= 0)

    return values


def clean_coordinate_series(series):
    """Parse latitude/longitude with comma or dot decimal separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_data():
    """Load exactly the same three source files used by GFQA_v2."""

    print("=" * 70)
    print("Reading GFQA source data")
    print("=" * 70)

    flux_path = SOURCE_DIR / "Flux.csv"
    water_path = SOURCE_DIR / "Water.csv"
    station_path = SOURCE_DIR / "GEMStat_station_metadata.xlsx"

    print(f"Flux:    {flux_path}")
    print(f"Water:   {water_path}")
    print(f"Stations:{station_path}")

    flux_df = pd.read_csv(
        flux_path,
        delimiter=";",
        encoding="iso-8859-1",
        low_memory=False,
    )

    water_df = pd.read_csv(
        water_path,
        delimiter=";",
        encoding="iso-8859-1",
        low_memory=False,
    )

    station_df = pd.read_excel(station_path)

    # Same station-ID cleanup as main GFQA script
    flux_df["GEMS.Station.Number"] = (
        flux_df["GEMS.Station.Number"].astype(str).str.strip()
    )

    water_df["GEMS.Station.Number"] = (
        water_df["GEMS.Station.Number"].astype(str).str.strip()
    )

    station_df["GEMS Station Number"] = (
        station_df["GEMS Station Number"].astype(str).str.strip()
    )

    flux_df["Parameter.Code"] = (
        flux_df["Parameter.Code"].astype(str).str.strip()
    )

    water_df["Parameter.Code"] = (
        water_df["Parameter.Code"].astype(str).str.strip()
    )

    print(f"\nFlux records:   {len(flux_df):,}")
    print(f"Water records:  {len(water_df):,}")
    print(f"Metadata rows:  {len(station_df):,}")

    return flux_df, water_df, station_df


def extract_valid_q_ssc(flux_df, water_df):
    """
    Extract valid Q and SSC source observations.

    Important:
    This applies the same numerical validity criterion as clean_value()
    before station intersection.
    """

    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    q_df = flux_df[
        flux_df["Parameter.Code"] == "Q-Inst"
    ].copy()

    q_df["Clean_Value"] = clean_numeric_series(q_df["Value"])

    q_valid = q_df[
        q_df["Clean_Value"].notna()
    ].copy()

    # --------------------------------------------------------
    # SSC
    # --------------------------------------------------------

    ssc_df = water_df[
        water_df["Parameter.Code"] == "TSS"
    ].copy()

    ssc_df["Clean_Value"] = clean_numeric_series(ssc_df["Value"])

    ssc_valid = ssc_df[
        ssc_df["Clean_Value"].notna()
    ].copy()

    return q_df, ssc_df, q_valid, ssc_valid


def calculate_station_sets(q_valid, ssc_valid):
    """Determine before/after station sets."""

    q_stations = set(
        q_valid["GEMS.Station.Number"].dropna().unique()
    )

    ssc_stations = set(
        ssc_valid["GEMS.Station.Number"].dropna().unique()
    )

    # All stations represented before requiring both variables
    before_stations = q_stations | ssc_stations

    # Stations retained after Q/SSC intersection
    retained_stations = q_stations & ssc_stations

    # Stations removed by intersection
    q_only_stations = q_stations - ssc_stations
    ssc_only_stations = ssc_stations - q_stations

    return {
        "q": q_stations,
        "ssc": ssc_stations,
        "before": before_stations,
        "retained": retained_stations,
        "q_only": q_only_stations,
        "ssc_only": ssc_only_stations,
    }


def prepare_station_metadata(station_df):
    """Clean station latitude/longitude and remove invalid coordinates."""

    meta = station_df.copy()

    meta["Latitude_clean"] = clean_coordinate_series(meta["Latitude"])
    meta["Longitude_clean"] = clean_coordinate_series(meta["Longitude"])

    valid_coordinate = (
        meta["Latitude_clean"].between(-90, 90)
        & meta["Longitude_clean"].between(-180, 180)
    )

    invalid_n = int((~valid_coordinate).sum())

    if invalid_n > 0:
        print(
            f"\nWARNING: {invalid_n} metadata rows have invalid/missing "
            "coordinates and cannot be plotted."
        )

    meta = meta[valid_coordinate].copy()

    # In case metadata contains duplicate station rows
    meta = meta.drop_duplicates(
        subset="GEMS Station Number",
        keep="first",
    )

    return meta


def select_metadata(meta, station_ids):
    """Return metadata rows for a given station set."""

    return meta[
        meta["GEMS Station Number"].isin(station_ids)
    ].copy()


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    flux_df,
    water_df,
    q_df,
    ssc_df,
    q_valid,
    ssc_valid,
    station_sets,
):
    """Print station- and observation-level screening statistics."""

    q_stations = station_sets["q"]
    ssc_stations = station_sets["ssc"]
    before = station_sets["before"]
    retained = station_sets["retained"]
    q_only = station_sets["q_only"]
    ssc_only = station_sets["ssc_only"]

    removed_stations = before - retained

    # Raw valid observations that belong to retained stations
    q_valid_retained = q_valid[
        q_valid["GEMS.Station.Number"].isin(retained)
    ]

    ssc_valid_retained = ssc_valid[
        ssc_valid["GEMS.Station.Number"].isin(retained)
    ]

    n_valid_before = len(q_valid) + len(ssc_valid)
    n_valid_after = len(q_valid_retained) + len(ssc_valid_retained)
    n_valid_removed = n_valid_before - n_valid_after

    # This reproduces the CURRENT process_all_stations() candidate logic:
    # all Flux stations intersect all Water stations, without restricting
    # Parameter.Code or checking Value.
    script_flux_stations = set(
        flux_df["GEMS.Station.Number"].dropna().unique()
    )

    script_water_stations = set(
        water_df["GEMS.Station.Number"].dropna().unique()
    )

    script_common_stations = (
        script_flux_stations & script_water_stations
    )

    print("\n")
    print("=" * 70)
    print("GFQA STATION FILTER AUDIT")
    print("=" * 70)

    print("\n[Raw parameter records]")
    print(f"Q-Inst records before value cleaning : {len(q_df):,}")
    print(f"TSS records before value cleaning    : {len(ssc_df):,}")
    print(f"Valid Q records                      : {len(q_valid):,}")
    print(f"Valid SSC records                    : {len(ssc_valid):,}")

    print("\n[Station counts BEFORE intersection]")
    print(f"Stations with valid Q                : {len(q_stations):,}")
    print(f"Stations with valid SSC              : {len(ssc_stations):,}")
    print(f"Q OR SSC station union               : {len(before):,}")

    print("\n[Station counts AFTER intersection]")
    print(f"Q AND SSC retained stations          : {len(retained):,}")
    print(f"Q-only stations removed              : {len(q_only):,}")
    print(f"SSC-only stations removed            : {len(ssc_only):,}")
    print(f"Total stations removed               : {len(removed_stations):,}")

    if len(before) > 0:
        station_retention = 100 * len(retained) / len(before)
        station_removed_pct = 100 - station_retention

        print(f"Station retention rate               : {station_retention:.2f}%")
        print(f"Station removal rate                 : {station_removed_pct:.2f}%")

    print("\n[Valid raw observations affected by station intersection]")
    print(f"Valid Q observations before          : {len(q_valid):,}")
    print(f"Valid Q observations retained        : {len(q_valid_retained):,}")
    print(
        f"Valid Q observations removed         : "
        f"{len(q_valid) - len(q_valid_retained):,}"
    )

    print(f"Valid SSC observations before        : {len(ssc_valid):,}")
    print(f"Valid SSC observations retained      : {len(ssc_valid_retained):,}")
    print(
        f"Valid SSC observations removed       : "
        f"{len(ssc_valid) - len(ssc_valid_retained):,}"
    )

    print(f"Total valid observations before      : {n_valid_before:,}")
    print(f"Total valid observations retained    : {n_valid_after:,}")
    print(f"Total valid observations removed     : {n_valid_removed:,}")

    if n_valid_before > 0:
        observation_removed_pct = (
            100 * n_valid_removed / n_valid_before
        )

        print(
            f"Observation removal rate             : "
            f"{observation_removed_pct:.2f}%"
        )

    print("\n[Important comparison with current GFQA_v2 code]")
    print(
        "Current process_all_stations() Flux/Water intersection : "
        f"{len(script_common_stations):,}"
    )

    print(
        "Stations with ACTUAL valid Q + SSC                    : "
        f"{len(retained):,}"
    )

    difference = script_common_stations - retained

    print(
        "Current script candidates lacking valid Q and/or SSC   : "
        f"{len(difference):,}"
    )

    print("=" * 70)


def save_summary_csv(q_valid, ssc_valid, station_sets):
    """Save a compact summary table."""

    retained = station_sets["retained"]
    before = station_sets["before"]

    q_after = q_valid[
        q_valid["GEMS.Station.Number"].isin(retained)
    ]

    ssc_after = ssc_valid[
        ssc_valid["GEMS.Station.Number"].isin(retained)
    ]

    rows = [
        {
            "metric": "stations_before_Q_or_SSC",
            "value": len(before),
        },
        {
            "metric": "stations_with_valid_Q",
            "value": len(station_sets["q"]),
        },
        {
            "metric": "stations_with_valid_SSC",
            "value": len(station_sets["ssc"]),
        },
        {
            "metric": "stations_after_Q_and_SSC",
            "value": len(retained),
        },
        {
            "metric": "Q_only_stations_removed",
            "value": len(station_sets["q_only"]),
        },
        {
            "metric": "SSC_only_stations_removed",
            "value": len(station_sets["ssc_only"]),
        },
        {
            "metric": "valid_Q_records_before",
            "value": len(q_valid),
        },
        {
            "metric": "valid_Q_records_after_station_filter",
            "value": len(q_after),
        },
        {
            "metric": "valid_SSC_records_before",
            "value": len(ssc_valid),
        },
        {
            "metric": "valid_SSC_records_after_station_filter",
            "value": len(ssc_after),
        },
    ]

    summary_df = pd.DataFrame(rows)

    output_path = (
        OUTPUT_DIR
        / "GFQA_station_intersection_summary.csv"
    )

    summary_df.to_csv(output_path, index=False)

    print(f"\nSaved summary CSV:")
    print(output_path)


# ============================================================
# Mapping
# ============================================================

def plot_station_distribution(meta, station_sets):
    """
    Plot mutually exclusive station categories.

    Blue circle:
        Q available but SSC absent -> removed

    Orange triangle:
        SSC available but Q absent -> removed

    Red star:
        both Q and SSC available -> retained
    """

    q_only_meta = select_metadata(
        meta,
        station_sets["q_only"],
    )

    ssc_only_meta = select_metadata(
        meta,
        station_sets["ssc_only"],
    )

    retained_meta = select_metadata(
        meta,
        station_sets["retained"],
    )

    # --------------------------------------------------------
    # Check metadata coverage
    # --------------------------------------------------------

    expected_before = len(station_sets["before"])

    plotted_before = len(
        set(q_only_meta["GEMS Station Number"])
        | set(ssc_only_meta["GEMS Station Number"])
        | set(retained_meta["GEMS Station Number"])
    )

    print("\n[Map metadata coverage]")
    print(f"Stations expected before filter : {expected_before:,}")
    print(f"Stations with plottable coords  : {plotted_before:,}")
    print(
        f"Stations missing coordinates    : "
        f"{expected_before - plotted_before:,}"
    )

    # --------------------------------------------------------
    # Cartopy version
    # --------------------------------------------------------

    if HAS_CARTOPY:

        fig = plt.figure(figsize=(16, 8.5))

        ax = plt.axes(
            projection=ccrs.Robinson()
        )

        ax.set_global()

        ax.add_feature(
            cfeature.LAND,
            facecolor="0.94",
            zorder=0,
        )

        ax.add_feature(
            cfeature.OCEAN,
            facecolor="white",
            zorder=0,
        )

        ax.add_feature(
            cfeature.COASTLINE,
            linewidth=0.5,
            edgecolor="0.35",
            zorder=1,
        )

        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.25,
            edgecolor="0.60",
            zorder=1,
        )

        # Q only: filtered out
        ax.scatter(
            q_only_meta["Longitude_clean"],
            q_only_meta["Latitude_clean"],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=0.9,
            alpha=0.75,
            transform=ccrs.PlateCarree(),
            label=(
                f"Q only - removed "
                f"(n={len(station_sets['q_only'])})"
            ),
            zorder=3,
        )

        # SSC only: filtered out
        ax.scatter(
            ssc_only_meta["Longitude_clean"],
            ssc_only_meta["Latitude_clean"],
            s=35,
            marker="^",
            facecolors="none",
            edgecolors="tab:orange",
            linewidths=0.9,
            alpha=0.80,
            transform=ccrs.PlateCarree(),
            label=(
                f"SSC only - removed "
                f"(n={len(station_sets['ssc_only'])})"
            ),
            zorder=3,
        )

        # Both: retained
        ax.scatter(
            retained_meta["Longitude_clean"],
            retained_meta["Latitude_clean"],
            s=42,
            marker="*",
            c="crimson",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.90,
            transform=ccrs.PlateCarree(),
            label=(
                f"Q + SSC - retained "
                f"(n={len(station_sets['retained'])})"
            ),
            zorder=4,
        )

        ax.gridlines(
            linewidth=0.35,
            linestyle="--",
            alpha=0.45,
        )

    # --------------------------------------------------------
    # Plain matplotlib fallback
    # --------------------------------------------------------

    else:

        fig, ax = plt.subplots(
            figsize=(16, 8.5)
        )

        ax.scatter(
            q_only_meta["Longitude_clean"],
            q_only_meta["Latitude_clean"],
            s=30,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            linewidths=0.9,
            alpha=0.75,
            label=(
                f"Q only - removed "
                f"(n={len(station_sets['q_only'])})"
            ),
        )

        ax.scatter(
            ssc_only_meta["Longitude_clean"],
            ssc_only_meta["Latitude_clean"],
            s=35,
            marker="^",
            facecolors="none",
            edgecolors="tab:orange",
            linewidths=0.9,
            alpha=0.80,
            label=(
                f"SSC only - removed "
                f"(n={len(station_sets['ssc_only'])})"
            ),
        )

        ax.scatter(
            retained_meta["Longitude_clean"],
            retained_meta["Latitude_clean"],
            s=42,
            marker="*",
            c="crimson",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.90,
            label=(
                f"Q + SSC - retained "
                f"(n={len(station_sets['retained'])})"
            ),
        )

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.grid(
            linestyle="--",
            linewidth=0.4,
            alpha=0.5,
        )

    # --------------------------------------------------------
    # Figure annotation
    # --------------------------------------------------------

    n_before = len(station_sets["before"])
    n_after = len(station_sets["retained"])
    n_removed = n_before - n_after

    if n_before > 0:
        removal_pct = 100 * n_removed / n_before
    else:
        removal_pct = np.nan

    title = (
        "GFQA Station Distribution Before and After Q-SSC Intersection\n"
        f"Before: {n_before:,} stations   |   "
        f"Retained: {n_after:,}   |   "
        f"Removed: {n_removed:,} ({removal_pct:.1f}%)"
    )

    ax.set_title(
        title,
        fontsize=14,
        pad=16,
    )

    ax.legend(
        loc="lower left",
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()

    output_png = (
        OUTPUT_DIR
        / "GFQA_station_distribution_before_after_intersection.png"
    )

    output_pdf = (
        OUTPUT_DIR
        / "GFQA_station_distribution_before_after_intersection.pdf"
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved station-distribution map:")
    print(output_png)
    print(output_pdf)


# ============================================================
# Main
# ============================================================

def main():

    flux_df, water_df, station_df = load_data()

    q_df, ssc_df, q_valid, ssc_valid = extract_valid_q_ssc(
        flux_df,
        water_df,
    )

    station_sets = calculate_station_sets(
        q_valid,
        ssc_valid,
    )

    meta = prepare_station_metadata(
        station_df
    )

    print_statistics(
        flux_df=flux_df,
        water_df=water_df,
        q_df=q_df,
        ssc_df=ssc_df,
        q_valid=q_valid,
        ssc_valid=ssc_valid,
        station_sets=station_sets,
    )

    save_summary_csv(
        q_valid=q_valid,
        ssc_valid=ssc_valid,
        station_sets=station_sets,
    )

    plot_station_distribution(
        meta=meta,
        station_sets=station_sets,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
