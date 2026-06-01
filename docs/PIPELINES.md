# Pipeline Guide

`run_pipeline.py` is the new top-level entry point for dataset execution order and batch runs.

## Usage

```bash
cd Script
python run_pipeline.py --list
python run_pipeline.py GloRiSe --dry-run
python run_pipeline.py Milliman
python run_pipeline.py --all
```

Optional roots:

```bash
python run_pipeline.py GloRiSe \
  --source-root /path/to/Source \
  --output-root /path/to/Output_r
```

## Canonical Multi-Stage Order

| Dataset | Stage Order |
|---|---|
| `GloRiSe` | `1_generate_netcdf_SS.py` → `2_qc_and_standardize_glorise.py` → `3_generate_nc_BS.py` → `4_qc_and_standardize_BS.py` |
| `Hydat` | `1_hydat_to_netcdf_fixed.py` → `2_extract_sediment_data_prallel.py` → `3_update_sediment_nc_fixed.py` → `4_process_hydat_cf18.py` |
| `Milliman` | `1_convert_to_netcdf.py` → `2_fix_netcdf_units.py` → `3_add_variables_to_netcdf.py` → `4_convert_units_to_daily.py` → `5_qc_and_standardize.py` |
| `Huanghe` | `convert_to_netcdf.py` → `qc_and_standardize.py` |
| `Vanmaercke` | `convert_to_netcdf.py` → `qc_and_standardize.py` |
| `Yajiang` | `convert_to_nc.py` → `process_yajiang.py` |
| `bayern` | `convert_bayern_to_netcdf.py` → `qc_and_standardize.py` |

## Canonical Single-Stage Entry Points

These datasets should normally be launched through the following scripts:

| Dataset | Entry Script |
|---|---|
| `ALi_De_Boer` | `ALi_De_Boer/process_data_tool.py` |
| `Chao_Phraya_River` | `Chao_Phraya_River/process_chao_phraya.py` |
| `Dethier` | `Dethier/process_dethier_tool.py` |
| `EUSEDcollab` | `EUSEDcollab/process_eusedcollab_to_cf18_wzx.py` |
| `Eurasian_River` | `Eurasian_River/process_eurasian_river.py` |
| `Fukushima` | `Fukushima/fukushima_qc_and_cf_enhancement.py` |
| `GFQA_v2` | `GFQA_v2/gfqa_to_netcdf_daily_dualqc.py` |
| `GSED` | `GSED/process_gsed_cf18.py` |
| `HMA` | `HMA/convert_to_netcdf_cf18_qc.py` |
| `HYBAM` | `HYBAM/hybam_comprehensive_processor.py` |
| `Land2sea` | `Land2sea/convert_land2sea_to_netcdf.py` |
| `Mekong_Delta` | `Mekong_Delta/process_mekong_delta.py` |
| `Myanmar` | `Myanmar/convert_to_netcdf.py` |
| `NERC` | `NERC/convert_NERC_to_netcdf.py` |
| `Rhine` | `Rhine/process_rhine.py` |
| `RiverSed` | `RiverSed/convert_to_netcdf.py` |
| `Robotham` | `Robotham/convert_to_netcdf_v2.py` |
| `Shashi_Jianli` | `Shashi_Jianli/process_shashi_jianli.py` |
| `USGS` | `USGS/process_usgs.py` |

## Notes

- `run_pipeline.py --list --include-optional` also shows legacy converters, validators, and post-run utilities.
- `Mekong_Delta` keeps several older helper scripts in the folder, but `process_mekong_delta.py` is the canonical end-to-end pipeline.
- `USGS`, `Myanmar`, `NERC`, `GSED`, `HMA`, and `Dethier` include optional verification or utility scripts that are intentionally not part of the default core run.
