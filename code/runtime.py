import os
from pathlib import Path


OUTPUT_ROOT_ENV = "SEDIMENT_OUTPUT_ROOT"
SOURCE_ROOT_ENV = "SEDIMENT_SOURCE_ROOT"


def _candidate_paths(start):
    path = Path(start).expanduser().resolve() if start else Path(__file__).resolve()
    if path.is_file():
        path = path.parent
    return [path, *path.parents]


def resolve_script_root(start=None):
    for candidate in _candidate_paths(start):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate Script/ repository root from current path")


def resolve_project_root(start=None):
    return resolve_script_root(start).parent


def resolve_source_root(start=None):
    env_value = os.environ.get(SOURCE_ROOT_ENV)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return resolve_project_root(start) / "Source"


def resolve_output_root(start=None, *, create=False):
    env_value = os.environ.get(OUTPUT_ROOT_ENV)
    if env_value:
        output_root = Path(env_value).expanduser().resolve()
    else:
        output_root = resolve_project_root(start) / "Output_r"
    if create:
        output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def resolve_source_path(*parts, start=None):
    return resolve_source_root(start).joinpath(*parts)


def resolve_output_path(*parts, start=None, create_parent=False):
    path = resolve_output_root(start, create=create_parent).joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(path_like):
    path = Path(path_like).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
