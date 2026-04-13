#!/usr/bin/env python3
"""
Top-level dataset runner for the sediment processing repository.

Features
--------
- Lists canonical stage order for each dataset
- Runs one dataset or all datasets from a single entry point
- Supports dry-run mode for reproducible planning
- Allows overriding the output root via SEDIMENT_OUTPUT_ROOT / --output-root
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from runtime import OUTPUT_ROOT_ENV, SOURCE_ROOT_ENV, resolve_output_root, resolve_project_root, resolve_source_root


PIPELINES = {
    "ALi_De_Boer": {
        "summary": "Ali & De Boer annual climatology conversion",
        "stages": [
            {"script": "ALi_De_Boer/process_data_tool.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Chao_Phraya_River": {
        "summary": "Chao Phraya annual climatology conversion",
        "stages": [
            {"script": "Chao_Phraya_River/process_chao_phraya.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Dethier": {
        "summary": "Dethier glacier-fed rivers",
        "stages": [
            {"script": "Dethier/process_dethier_tool.py", "description": "Canonical end-to-end processor"},
            {"script": "Dethier/analyze_output.py", "description": "Post-run diagnostics", "optional": True},
        ],
    },
    "EUSEDcollab": {
        "summary": "European suspended sediment collaboration database",
        "stages": [
            {"script": "EUSEDcollab/process_eusedcollab_to_cf18_wzx.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Eurasian_River": {
        "summary": "Eurasian monthly sediment compilation",
        "stages": [
            {"script": "Eurasian_River/process_eurasian_river.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Fukushima": {
        "summary": "Fukushima post-accident sediment records",
        "stages": [
            {"script": "Fukushima/fukushima_qc_and_cf_enhancement.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "GFQA_v2": {
        "summary": "Global Flow and Water Quality Archive v2",
        "stages": [
            {"script": "GFQA_v2/gfqa_to_netcdf_daily_dualqc.py", "description": "Convert + dual QC + CF NetCDF"},
        ],
    },
    "GSED": {
        "summary": "Global sediment dataset",
        "stages": [
            {"script": "GSED/process_gsed_cf18.py", "description": "Canonical end-to-end processor"},
            {"script": "GSED/validate_gsed_data.py", "description": "Post-run validator", "optional": True},
        ],
    },
    "GloRiSe": {
        "summary": "Global River Sediment Database daily pipeline",
        "stages": [
            {"script": "GloRiSe/1_generate_netcdf_SS.py", "description": "Generate SS intermediate NetCDF"},
            {"script": "GloRiSe/2_qc_and_standardize_glorise.py", "description": "QC + standardize SS"},
            {"script": "GloRiSe/3_generate_nc_BS.py", "description": "Generate BS intermediate NetCDF"},
            {"script": "GloRiSe/4_qc_and_standardize_BS.py", "description": "QC + standardize BS"},
        ],
    },
    "HMA": {
        "summary": "High Mountain Asia glacier-fed rivers",
        "stages": [
            {"script": "HMA/convert_to_netcdf_cf18_qc.py", "description": "Canonical end-to-end processor"},
            {"script": "HMA/verify_netcdf.py", "description": "Post-run verifier", "optional": True},
        ],
    },
    "HYBAM": {
        "summary": "HYBAM Amazon sediment network",
        "stages": [
            {"script": "HYBAM/hybam_comprehensive_processor.py", "description": "Canonical end-to-end processor"},
        ],
    },
    "Huanghe": {
        "summary": "Yellow River annual climatology pipeline",
        "stages": [
            {"script": "Huanghe/convert_to_netcdf.py", "description": "Generate intermediate NetCDF"},
            {"script": "Huanghe/qc_and_standardize.py", "description": "QC + CF standardization"},
        ],
    },
    "Hydat": {
        "summary": "HYDAT four-stage pipeline",
        "stages": [
            {"script": "Hydat/1_hydat_to_netcdf_fixed.py", "description": "Convert HYDAT MDB to discharge NetCDF"},
            {"script": "Hydat/2_extract_sediment_data_prallel.py", "description": "Extract sediment data by station"},
            {"script": "Hydat/3_update_sediment_nc_fixed.py", "description": "Merge/update sediment NetCDF"},
            {"script": "Hydat/4_process_hydat_cf18.py", "description": "QC + CF standardization"},
        ],
    },
    "Land2sea": {
        "summary": "Land2Sea model output conversion",
        "stages": [
            {"script": "Land2sea/convert_land2sea_to_netcdf.py", "description": "Convert + CF NetCDF"},
        ],
    },
    "Mekong_Delta": {
        "summary": "Canonical QC pipeline plus legacy utilities",
        "stages": [
            {"script": "Mekong_Delta/process_mekong_delta.py", "description": "Canonical end-to-end QC pipeline"},
            {"script": "Mekong_Delta/verify_qc.py", "description": "Post-run validator", "optional": True},
            {"script": "Mekong_Delta/summarize_data.py", "description": "Post-run summarizer", "optional": True},
            {"script": "Mekong_Delta/convert_to_netcdf.py", "description": "Legacy raw converter", "optional": True},
            {"script": "Mekong_Delta/convert_to_netcdf_ratings_only.py", "description": "Legacy ratings-only variant", "optional": True},
        ],
    },
    "Milliman": {
        "summary": "Five-stage Milliman processing pipeline",
        "stages": [
            {"script": "Milliman/1_convert_to_netcdf.py", "description": "Create intermediate NetCDF"},
            {"script": "Milliman/2_fix_netcdf_units.py", "description": "Fix mislabeled TSS units"},
            {"script": "Milliman/3_add_variables_to_netcdf.py", "description": "Add derived variables to intermediates"},
            {"script": "Milliman/4_convert_units_to_daily.py", "description": "Convert discharge/TSS units to daily"},
            {"script": "Milliman/5_qc_and_standardize.py", "description": "QC + CF standardization"},
        ],
    },
    "Myanmar": {
        "summary": "Myanmar station pipeline",
        "stages": [
            {"script": "Myanmar/convert_to_netcdf.py", "description": "Canonical converter"},
            {"script": "Myanmar/verify_myanmar_qc.py", "description": "Post-run validator", "optional": True},
            {"script": "Myanmar/summarize_myanmar_data.py", "description": "Post-run summarizer", "optional": True},
        ],
    },
    "NERC": {
        "summary": "NERC daily river chemistry and sediment pipeline",
        "stages": [
            {"script": "NERC/convert_NERC_to_netcdf.py", "description": "Canonical converter"},
            {"script": "NERC/refine_NERC_processing.py", "description": "Legacy refinement utility", "optional": True},
            {"script": "NERC/validate_nerc_data.py", "description": "Post-run validator", "optional": True},
        ],
    },
    "Rhine": {
        "summary": "Rhine daily parser and QC",
        "stages": [
            {"script": "Rhine/process_rhine.py", "description": "Canonical end-to-end processor"},
        ],
    },
    "RiverSed": {
        "summary": "RiverSed daily compilation",
        "stages": [
            {"script": "RiverSed/convert_to_netcdf.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Robotham": {
        "summary": "Robotham daily sediment records",
        "stages": [
            {"script": "Robotham/convert_to_netcdf_v2.py", "description": "Convert + QC + CF NetCDF"},
        ],
    },
    "Shashi_Jianli": {
        "summary": "Yangtze Shashi-Jianli reach",
        "stages": [
            {"script": "Shashi_Jianli/process_shashi_jianli.py", "description": "Canonical end-to-end processor"},
        ],
    },
    "USGS": {
        "summary": "USGS NWIS daily pipeline plus utilities",
        "stages": [
            {"script": "USGS/process_usgs.py", "description": "Canonical NWIS processor"},
            {"script": "USGS/process_existing_usgs_netcdf.py", "description": "Existing-NetCDF reprocessor", "optional": True},
            {"script": "USGS/merge_info.py", "description": "Metadata merge utility", "optional": True},
        ],
    },
    "Vanmaercke": {
        "summary": "Vanmaercke annual climatology pipeline",
        "stages": [
            {"script": "Vanmaercke/convert_to_netcdf.py", "description": "Generate intermediate NetCDF"},
            {"script": "Vanmaercke/qc_and_standardize.py", "description": "QC + CF standardization"},
        ],
    },
    "Yajiang": {
        "summary": "Yajiang two-stage daily pipeline",
        "stages": [
            {"script": "Yajiang/convert_to_nc.py", "description": "Generate intermediate NetCDF"},
            {"script": "Yajiang/process_yajiang.py", "description": "QC + CF standardization"},
        ],
    },
    "bayern": {
        "summary": "Bayern daily network two-stage pipeline",
        "stages": [
            {"script": "bayern/convert_bayern_to_netcdf.py", "description": "Generate intermediate NetCDF"},
            {"script": "bayern/qc_and_standardize.py", "description": "QC + CF standardization"},
        ],
    },
}


def normalize_dataset_name(name):
    lookup = {dataset.lower(): dataset for dataset in PIPELINES}
    normalized = lookup.get(name.lower())
    if normalized is None:
        valid = ", ".join(PIPELINES.keys())
        raise KeyError(f"Unknown dataset '{name}'. Valid dataset names: {valid}")
    return normalized


def iter_stages(dataset_name, include_optional):
    for stage in PIPELINES[dataset_name]["stages"]:
        if stage.get("optional") and not include_optional:
            continue
        yield stage


def print_pipeline_list(include_optional):
    for dataset_name, config in PIPELINES.items():
        stages = list(iter_stages(dataset_name, include_optional=include_optional))
        optional_count = sum(1 for stage in config["stages"] if stage.get("optional"))
        print(f"{dataset_name}: {config['summary']}")
        for index, stage in enumerate(stages, start=1):
            label = "optional" if stage.get("optional") else "core"
            print(f"  {index}. [{label}] {stage['script']} - {stage['description']}")
        if optional_count and not include_optional:
            print(f"  ... {optional_count} optional stage(s) hidden; use --include-optional to show them")
        print()


def prepare_output_root(output_root):
    """
    Ensure the canonical project-root Output_r path exists.

    Most scripts now honor SEDIMENT_OUTPUT_ROOT directly, but this symlink keeps
    older relative-path scripts working when a custom output root is requested.
    """
    project_root = resolve_project_root(__file__)
    default_output_root = project_root / "Output_r"
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if output_root == default_output_root.resolve():
        default_output_root.mkdir(parents=True, exist_ok=True)
        return

    if default_output_root.exists():
        if default_output_root.is_symlink() and default_output_root.resolve() == output_root:
            return
        raise RuntimeError(
            f"Cannot map custom output root because {default_output_root} already exists. "
            "Remove or rename that path first, or run without --output-root."
        )

    default_output_root.symlink_to(output_root, target_is_directory=True)


def run_stage(script_path, python_executable, env, dry_run):
    cmd = [python_executable, str(script_path)]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(SCRIPT_ROOT), env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run sediment dataset pipelines from one entry point")
    parser.add_argument("datasets", nargs="*", help="Dataset name(s) to run")
    parser.add_argument("--all", action="store_true", help="Run every dataset in manifest order")
    parser.add_argument("--list", action="store_true", help="List dataset pipelines and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--include-optional", action="store_true", help="Include optional utility/validation stages")
    parser.add_argument("--output-root", type=Path, help="Override Output_r root")
    parser.add_argument("--source-root", type=Path, help="Override Source root for migrated scripts")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run stage scripts")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print_pipeline_list(include_optional=args.include_optional)
        return

    selected = []
    if args.all:
        selected = list(PIPELINES.keys())
    elif args.datasets:
        selected = [normalize_dataset_name(name) for name in args.datasets]
    else:
        raise SystemExit("Specify dataset names, or use --all / --list.")

    env = os.environ.copy()
    source_root = args.source_root.expanduser().resolve() if args.source_root else resolve_source_root(__file__)
    output_root = args.output_root.expanduser().resolve() if args.output_root else resolve_output_root(__file__)
    env[SOURCE_ROOT_ENV] = str(source_root)
    env[OUTPUT_ROOT_ENV] = str(output_root)

    prepare_output_root(output_root)

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print()

    for dataset_name in selected:
        print(f"== {dataset_name} ==")
        for index, stage in enumerate(iter_stages(dataset_name, include_optional=args.include_optional), start=1):
            script_path = SCRIPT_ROOT / stage["script"]
            if not script_path.exists():
                raise FileNotFoundError(f"Stage script not found: {script_path}")
            print(f"{index}. {stage['description']}")
            run_stage(script_path, args.python, env, args.dry_run)
        print()


if __name__ == "__main__":
    main()
