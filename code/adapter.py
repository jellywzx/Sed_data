"""
Adapter for the ALi & De Boer (2007) dataset.

Responsibilities:
- Read raw ALi_De_Boer Excel source
- Map source-specific column names to standard variables
- Call core functions for parsing, calculation, QC, and time semantics
- Assemble per-station records for metadata + cf_writer

This module MUST NOT implement any generic algorithms.
"""

import pandas as pd
import numpy as np

# ---------- core imports (single source of truth) ----------
from core.parsing import (
    parse_dms_to_decimal,
    parse_period,
)

from core.calculation import (
    calculate_discharge,
    calculate_ssl_from_mt_yr,
    calculate_ssc,
)

from core.time import (
    climatology_midpoint_days_since_1970,
)

from core.qc import (
    compute_log_iqr_bounds,
    flag_log_iqr_outlier,
)

from core.constants import (
    FILL_FLOAT,
    FILL_INT,
)


# ==========================================================
# Public API
# ==========================================================
def load_and_process(source_file):
    """
    Load ALi_De_Boer Excel file and return a list of per-station records.

    Each record is a pure-data dictionary (numbers + strings only),
    ready to be passed to metadata.build_metadata().
    """

    # ------------------------------------------------------
    # 1. Read source file
    # ------------------------------------------------------
    df = pd.read_excel(source_file, sheet_name="Sheet2")

    # ------------------------------------------------------
    # 2. Compute dataset-level SSL log-IQR bounds
    #    (ONLY from source-reported sediment load: Mt/yr)
    # ------------------------------------------------------
    ssl_source_mt_yr = df["Sediment（(Mt yr−1)）"].to_numpy()
    ssl_bounds = compute_log_iqr_bounds(ssl_source_mt_yr)

    records = []

    # ------------------------------------------------------
    # 3. Per-station processing
    # ------------------------------------------------------
    for idx, row in df.iterrows():
        source_id = f"ALI{idx + 1:03d}"

        # ---------- coordinates ----------
        latitude = parse_dms_to_decimal(row["Latitude"])
        longitude = parse_dms_to_decimal(row["Longitude"])

        # ---------- temporal coverage ----------
        start_year, end_year = parse_period(row["Period of record"])
        days_since_1970 = climatology_midpoint_days_since_1970(
            start_year, end_year
        )

        # ---------- station properties ----------
        elevation = row["Elevation (masl)"]
        drainage_area = row["Drainage area (km2)"]

        # ---------- source-reported values ----------
        runoff_mm_yr = row["Runoff (mm)"]
        sediment_mt_yr = row["Sediment（(Mt yr−1)）"]
        sediment_yield = row["Sediment（t km−2 yr−1）"]

        # ---------- derived quantities ----------
        Q = calculate_discharge(runoff_mm_yr, drainage_area)
        SSL = calculate_ssl_from_mt_yr(sediment_mt_yr)
        SSC = calculate_ssc(SSL, Q)

        # ---------- flags (default logic in core) ----------
        Q_flag = 0 if np.isfinite(Q) else FILL_INT
        SSC_flag = 0 if np.isfinite(SSC) else FILL_INT
        SSL_flag = 0 if np.isfinite(SSL) else FILL_INT

        # ---------- dataset-specific QC rule ----------
        # ONLY SSL participates in log-IQR screening
        SSL_flag = flag_log_iqr_outlier(
            value=sediment_mt_yr,
            bounds=ssl_bounds,
            default_flag=SSL_flag,
        )

        # ---------- assemble record ----------
        records.append(
            {
                # identifiers
                "station_name": row["Station"],
                "river_name": row["River"],
                "source_id": source_id,

                # coordinates & time
                "latitude": latitude,
                "longitude": longitude,
                "days_since_1970": days_since_1970,
                "start_year": start_year,
                "end_year": end_year,

                # station properties
                "elevation": elevation,
                "drainage_area": drainage_area,

                # hydrology & sediment
                "Q": Q,
                "Q_flag": Q_flag,
                "SSC": SSC,
                "SSC_flag": SSC_flag,
                "SSL": SSL,
                "SSL_flag": SSL_flag,
                "sediment_yield": sediment_yield,

                # provenance (for metadata documentation only)
                "runoff": runoff_mm_yr,
                "sediment_mt_yr": sediment_mt_yr,
            }
        )

    return records
