from pathlib import Path

import pandas as pd
from netCDF4 import Dataset


class InputValidationError(ValueError):
    """Raised when a source file exists but does not match the expected schema."""


def _as_path(path_like):
    return Path(path_like).expanduser().resolve()


def require_existing_file(path_like, *, description="input file"):
    path = _as_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {description} to be a file: {path}")
    return path


def require_existing_directory(path_like, *, description="input directory"):
    path = _as_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Expected {description} to be a directory: {path}")
    return path


def require_dataframe_columns(df, required_columns, *, source_name="DataFrame"):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise InputValidationError(
            f"{source_name} is missing required columns: {', '.join(missing)}"
        )
    return df


def read_excel_validated(
    path_like,
    *,
    sheet_name=0,
    required_columns=None,
    description="Excel source",
    **kwargs,
):
    path = require_existing_file(path_like, description=description)
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    if required_columns:
        require_dataframe_columns(
            df,
            required_columns,
            source_name=f"{path.name} [{sheet_name}]",
        )
    return df


def read_csv_validated(
    path_like,
    *,
    required_columns=None,
    description="CSV source",
    **kwargs,
):
    path = require_existing_file(path_like, description=description)
    df = pd.read_csv(path, **kwargs)
    if required_columns:
        require_dataframe_columns(df, required_columns, source_name=path.name)
    return df


def check_nc_completeness(
    nc_path,
    required_vars=None,
    required_attrs=None,
    strict=False,
):
    """
    Check whether a NetCDF dataset contains the expected variables and
    global attributes.
    """
    default_var_requirements = {
        "time": ["units", "calendar"],
        "lat": ["units"],
        "lon": ["units"],
        "Q": ["units", "long_name", "ancillary_variables"],
        "Q_flag": ["flag_values", "flag_meanings"],
        "SSC": ["units", "long_name", "ancillary_variables"],
        "SSC_flag": ["flag_values", "flag_meanings"],
        "SSL": ["units", "long_name", "ancillary_variables"],
        "SSL_flag": ["flag_values", "flag_meanings"],
    }

    default_required_attrs = [
        "Conventions",
        "title",
        "summary",
        "source",
        "data_source_name",
        "variables_provided",
        "number_of_data",
        "reference",
        "source_data_link",
        "creator_name",
        "creator_email",
        "creator_institution",
        "processing_level",
        "comment",
    ]

    strict_attrs = [
        "temporal_resolution",
        "time_coverage_start",
        "time_coverage_end",
        "geographic_coverage",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
    ]

    errors = []
    warnings = []

    required_vars = required_vars or list(default_var_requirements.keys())
    required_attrs = required_attrs or default_required_attrs

    with Dataset(nc_path, mode="r") as ds:
        available_attrs = set(ds.ncattrs())

        for attr in required_attrs:
            if attr not in available_attrs:
                errors.append("Missing global attribute: {0}".format(attr))

        for attr in strict_attrs:
            if attr not in available_attrs:
                message = "Missing global attribute: {0}".format(attr)
                if strict:
                    errors.append(message)
                else:
                    warnings.append(message)

        for var_name in required_vars:
            if var_name not in ds.variables:
                errors.append("Missing variable: {0}".format(var_name))
                continue

            attrs_required = default_var_requirements.get(var_name, [])
            var_attrs = set(ds.variables[var_name].ncattrs())
            for attr in attrs_required:
                if attr not in var_attrs:
                    errors.append(
                        "Variable '{0}' missing attribute: {1}".format(
                            var_name, attr
                        )
                    )

        if "variables_provided" in available_attrs:
            provided = [
                item.strip()
                for item in str(ds.getncattr("variables_provided")).split(",")
                if item.strip()
            ]
            for var_name in provided:
                if var_name not in ds.variables:
                    errors.append(
                        "variables_provided lists missing variable: "
                        "{0}".format(var_name)
                    )

    return errors, warnings
