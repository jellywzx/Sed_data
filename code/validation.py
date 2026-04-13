from pathlib import Path

import pandas as pd


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
