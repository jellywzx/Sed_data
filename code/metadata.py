"""
Metadata definition for the ALi & De Boer (2007) dataset.

This file contains all dataset-specific scientific semantics, variable
definitions, CF/ACDD attributes, and processing descriptions.

No numerical calculations or I/O operations should appear in this file.
"""

import numpy as np

FILL_FLOAT = -9999.0
FILL_INT = 9


def build_metadata(
    *,
    # ---- time / coordinates ----
    days_since_1970,
    latitude,
    longitude,

    # ---- station properties ----
    elevation,
    drainage_area,

    # ---- hydrology & sediment ----
    Q, Q_flag,
    SSC, SSC_flag,
    SSL, SSL_flag,
    sediment_yield,

    # ---- provenance / identifiers ----
    station_name,
    river_name,
    source_id,

    # ---- source values for documentation ----
    runoff,
    sediment_mt_yr,

    # ---- temporal coverage ----
    start_year,
    end_year,
):
    """
    Build dimension, variable, and global-attribute metadata dictionaries
    for the ALi & De Boer dataset.
    """

    # ==========================================================
    # Dimensions
    # ==========================================================
    dimensions = {
        "time": None
    }

    # ==========================================================
    # Variables
    # ==========================================================
    variables = {

        # ------------------------------------------------------
        # Time coordinate (climatological midpoint)
        # ------------------------------------------------------
        "time": {
            "dtype": "f8",
            "dims": ("time",),
            "fill_value": FILL_FLOAT,
            "data": days_since_1970,
            "attrs": {
                "standard_name": "time",
                "long_name": "representative time of climatological mean",
                "units": "days since 1970-01-01 00:00:00",
                "calendar": "gregorian",
                "axis": "T",
                "comment": (
                    "This time value represents the midpoint of the period of record "
                    "and is used as a representative timestamp for climatological "
                    "(multi-year mean) data. The values in this file are not "
                    "instantaneous observations but averages over the entire "
                    "period of record."
                ),
            },
        },

        # ------------------------------------------------------
        # Latitude (scalar coordinate)
        # ------------------------------------------------------
        "lat": {
            "dtype": "f4",
            "dims": (),
            "fill_value": FILL_FLOAT,
            "data": latitude,
            "attrs": {
                "standard_name": "latitude",
                "long_name": "station latitude",
                "units": "degrees_north",
                "valid_range": np.array([-90.0, 90.0], dtype="f4"),
            },
        },

        # ------------------------------------------------------
        # Longitude (scalar coordinate)
        # ------------------------------------------------------
        "lon": {
            "dtype": "f4",
            "dims": (),
            "fill_value": FILL_FLOAT,
            "data": longitude,
            "attrs": {
                "standard_name": "longitude",
                "long_name": "station longitude",
                "units": "degrees_east",
                "valid_range": np.array([-180.0, 180.0], dtype="f4"),
            },
        },

        # ------------------------------------------------------
        # Altitude (scalar data variable)
        # ------------------------------------------------------
        "altitude": {
            "dtype": "f4",
            "dims": (),
            "fill_value": FILL_FLOAT,
            "data": elevation,
            "attrs": {
                "standard_name": "altitude",
                "long_name": "station elevation above sea level",
                "units": "m",
                "positive": "up",
                "comment": (
                    "Source: Original data provided by Ali & De Boer (2007)."
                ),
            },
        },

        # ------------------------------------------------------
        # Upstream drainage area (scalar data variable)
        # ------------------------------------------------------
        "upstream_area": {
            "dtype": "f4",
            "dims": (),
            "fill_value": FILL_FLOAT,
            "data": drainage_area,
            "attrs": {
                "long_name": "upstream drainage area",
                "units": "km2",
                "comment": (
                    "Source: Original data provided by Ali & De Boer (2007)."
                ),
            },
        },

        # ------------------------------------------------------
        # River discharge (Q)
        # ------------------------------------------------------
        "Q": {
            "dtype": "f4",
            "dims": ("time",),
            "fill_value": FILL_FLOAT,
            "data": Q,
            "attrs": {
                "standard_name": "water_volume_transport_in_river_channel",
                "long_name": "river discharge",
                "units": "m3 s-1",
                "coordinates": "time lat lon",
                "ancillary_variables": "Q_flag",
                "comment": (
                    "Source: Calculated. Formula: Q (m³/s) = runoff (mm/yr) × "
                    "drainage_area (km²) / 31557.6, where the divisor accounts for "
                    "conversion from mm·km²·yr⁻¹ to m³·s⁻¹. "
                    f"Original runoff: {runoff} mm/yr. "
                    "Represents mean annual value over the period of record."
                ),
            },
        },

        # ------------------------------------------------------
        # Discharge quality flag
        # ------------------------------------------------------
        "Q_flag": {
            "dtype": "i1",
            "dims": ("time",),
            "fill_value": FILL_INT,
            "data": Q_flag,
            "attrs": {
                "standard_name": "status_flag",
                "long_name": "quality flag for river discharge",
                "flag_values": np.array([0, 1, 2, 3, 9], dtype="i1"),
                "flag_meanings": (
                    "good_data estimated_data suspect_data "
                    "bad_data missing_data"
                ),
                "comment": (
                    "Flag definitions: 0=Good, 1=Estimated, "
                    "2=Suspect, 3=Bad, 9=Missing in source."
                ),
            },
        },

        # ------------------------------------------------------
        # Suspended sediment concentration (SSC)
        # ------------------------------------------------------
        "SSC": {
            "dtype": "f4",
            "dims": ("time",),
            "fill_value": FILL_FLOAT,
            "data": SSC,
            "attrs": {
                "standard_name": "mass_concentration_of_suspended_matter_in_water",
                "long_name": "suspended sediment concentration",
                "units": "mg L-1",
                "coordinates": "time lat lon",
                "ancillary_variables": "SSC_flag",
                "comment": (
                    "Source: Calculated. Formula: SSC (mg/L) = SSL (ton/day) / "
                    "(Q (m³/s) × 86.4), where 86.4 = 86400 s/day × "
                    "1000 L/m³ × 10⁻⁶ ton/mg. "
                    "Represents mean annual value over the period of record."
                ),
            },
        },

        # ------------------------------------------------------
        # SSC quality flag
        # ------------------------------------------------------
        "SSC_flag": {
            "dtype": "i1",
            "dims": ("time",),
            "fill_value": FILL_INT,
            "data": SSC_flag,
            "attrs": {
                "standard_name": "status_flag",
                "long_name": (
                    "quality flag for suspended sediment concentration"
                ),
                "flag_values": np.array([0, 1, 2, 3, 9], dtype="i1"),
                "flag_meanings": (
                    "good_data estimated_data suspect_data "
                    "bad_data missing_data"
                ),
                "comment": (
                    "Flag definitions: 0=Good, 1=Estimated, "
                    "2=Suspect, 3=Bad, 9=Missing in source."
                ),
            },
        },

        # ------------------------------------------------------
        # Suspended sediment load (SSL)
        # ------------------------------------------------------
        "SSL": {
            "dtype": "f4",
            "dims": ("time",),
            "fill_value": FILL_FLOAT,
            "data": SSL,
            "attrs": {
                "long_name": "suspended sediment load",
                "units": "ton day-1",
                "coordinates": "time lat lon",
                "ancillary_variables": "SSL_flag",
                "comment": (
                    "Source: Calculated. Formula: SSL (ton/day) = "
                    "sediment_load (Mt/yr) × 10⁶ / 365, where 1 Mt = 10⁶ ton. "
                    f"Original sediment load: {sediment_mt_yr} Mt/yr. "
                    "Represents mean annual value over the period of record."
                ),
            },
        },

        # ------------------------------------------------------
        # SSL quality flag
        # ------------------------------------------------------
        "SSL_flag": {
            "dtype": "i1",
            "dims": ("time",),
            "fill_value": FILL_INT,
            "data": SSL_flag,
            "attrs": {
                "standard_name": "status_flag",
                "long_name": "quality flag for suspended sediment load",
                "flag_values": np.array([0, 1, 2, 3, 9], dtype="i1"),
                "flag_meanings": (
                    "good_data estimated_data suspect_data "
                    "bad_data missing_data"
                ),
                "comment": (
                    "Flag definitions: 0=Good, 1=Estimated, "
                    "2=Suspect, 3=Bad, 9=Missing in source. "
                    "A statistical outlier pre-screening was applied only to "
                    "SSL using a log-transformed interquartile range (IQR) "
                    "method based on source-reported annual sediment load "
                    "(Mt yr−1). Values identified as statistical outliers "
                    "were not removed but flagged as suspect (flag=2)."
                ),
            },
        },

        # ------------------------------------------------------
        # Sediment yield
        # ------------------------------------------------------
        "sediment_yield": {
            "dtype": "f4",
            "dims": ("time",),
            "fill_value": FILL_FLOAT,
            "data": sediment_yield,
            "attrs": {
                "long_name": "sediment yield per unit drainage area",
                "units": "t km-2 yr-1",
                "coordinates": "time lat lon",
                "comment": (
                    "Source: Original data provided by Ali & De Boer (2007)."
                ),
            },
        },
    }

    # ==========================================================
    # Global attributes (CF-1.8 / ACDD-1.3)
    # ==========================================================
    global_attrs = {
        "Conventions": "CF-1.8, ACDD-1.3",
        "title": "Harmonized Global River Discharge and Sediment",
        "summary": (
            f"River discharge and suspended sediment data for {station_name} "
            f"station on the {river_name} River in the upper Indus River basin, "
            "northern Pakistan. This dataset contains mean annual values "
            "including discharge, suspended sediment concentration, sediment "
            "load, and sediment yield. Data represents climatological average "
            "over the period of record."
        ),
        "source": "In-situ station data",
        "data_source_name": "ALi_De_Boer Dataset",
        "station_name": station_name,
        "river_name": river_name,
        "Source_ID": source_id,
        "geographic_coverage": (
            "Upper Indus River Basin, Northern Pakistan and Western Himalayas"
        ),
        "temporal_resolution": "climatological",
        "time_coverage_resolution": "climatological",
        "climatology": (
            "This dataset contains climatological (multi-year mean) values "
            "derived from observations over the stated period of record."
        ),
        "variables_provided": (
            "altitude, upstream_area, Q, SSC, SSL, sediment_yield"
        ),
        "number_of_data": "1",
        "reference": (
            "Ali, K. F., & De Boer, D. H. (2007). Spatial patterns and variation "
            "of suspended sediment yield in the upper Indus River basin, "
            "northern Pakistan. Journal of Hydrology, 334(3-4), 368-387. "
            "https://doi.org/10.1016/j.jhydrol.2006.10.013"
        ),
        "source_data_link": (
            "https://doi.org/10.1016/j.jhydrol.2006.10.013"
        ),
        "creator_name": "Zhongwang Wei",
        "creator_email": "weizhw6@mail.sysu.edu.cn",
        "creator_institution": "Sun Yat-sen University, China",
        "processing_level": "Quality controlled and standardized",
        "comment": (
            "Data represents mean annual values calculated from observations "
            "over the period of record. Discharge calculated from runoff and "
            "drainage area. SSC calculated from sediment load and discharge. "
            "Quality flags indicate data reliability: "
            "0=good, 1=estimated, 2=suspect, 3=bad, 9=missing."
        ),
    }

    if start_year and end_year:
        global_attrs["time_coverage_start"] = f"{start_year}-01-01"
        global_attrs["time_coverage_end"] = f"{end_year}-12-31"
        global_attrs["temporal_span"] = f"{start_year}-{end_year}"

    return dimensions, variables, global_attrs

