# Sediment Source-Data Harmonization and Quality-Control Workflow

This repository contains the **source-level processing workflow** used to prepare river discharge (Q), suspended sediment concentration (SSC), and suspended sediment load (SSL) observations for the dataset described in:

> *A harmonized global station-reference dataset of river discharge, suspended sediment concentration, and suspended sediment load*

This README follows the methodology structure of **Sect. 3.1–3.3** of the manuscript and documents **only the processing implemented in the `Sed_data` repository**.

The repository converts heterogeneous source files into standardized, quality-controlled, source-traceable NetCDF products. Its main responsibilities are:

- metadata harmonization;
- temporal standardization;
- variable mapping and unit conversion;
- derivation of missing sediment variables where appropriate;
- coordinate standardization;
- source-level quality control;
- CF-1.8 / ACDD-1.3 compatible NetCDF output.

---

## Manuscript-to-code map

| Manuscript section | Scope covered by `Sed_data` | Main scripts/modules |
| --- | --- | --- |
| **Sect. 3.1** Metadata, temporal, variable, and unit harmonization | Source-specific parsing, metadata standardization, temporal harmonization, variable mapping, unit conversion, and derivation of missing Q/SSC/SSL variables | `run_pipeline.py`; source-specific processors; `code/time_utils.py`; `code/units.py`; `code/daily_aggregation.py`; `code/metadata.py`; `code/global_attrs.py`; `code/constants.py` |
| **Sect. 3.2** Georeferencing and basin matching | Coordinate parsing, conversion, and standardization are handled in `Sed_data`; basin matching is performed in the companion `sed_data_integration` repository | Source-specific processors; `code/geo.py`; source-specific coordinate utilities. For MERIT-Basins reach assignment and upstream-basin tracing, refer to the basin-matching scripts in [`sed_data_integration`](https://github.com/jellywzx/sed_data_integration) |
| **Sect. 3.3** QC procedures and flagging system | Physical screening, statistical outlier detection, SSC-Q consistency checking, and flag propagation to derived variables | `code/qc.py`; `code/constants.py`; source-specific processors |

> **Basin-matching note:** `Sed_data` prepares the standardized spatial information required for hydrological matching, but does not implement the basin-matching workflow itself. The basin-matching procedures described in manuscript Sect. 3.2 are implemented in the companion [`sed_data_integration`](https://github.com/jellywzx/sed_data_integration) repository. Refer there to scripts such as `s4_basin_trace_watch.py`, `basin_tracer.py`, and `basin_policy.py` for MERIT-Basins reach assignment, upstream-basin tracing, and resolved/unresolved matching rules.

---

## Repository structure

```text
Sed_data/
├── run_pipeline.py               # Top-level dataset runner
├── tool.py                       # Backward-compatible shared-function interface
├── code/                         # Shared source-processing implementation
│   ├── constants.py              # QC flags and physical constants
│   ├── daily_aggregation.py      # Sub-daily -> daily aggregation
│   ├── geo.py                    # Coordinate helpers
│   ├── global_attrs.py           # Global metadata support
│   ├── metadata.py               # CF/ACDD metadata utilities
│   ├── qc.py                     # QC1 / QC2 / QC3 and flag propagation
│   ├── time_utils.py             # Temporal-period and climatology helpers
│   ├── units.py                  # Unit and sediment-variable conversions
│   ├── output.py                 # Summary/output helpers
│   ├── validation.py             # Input/output validation utilities
│   └── runtime.py                # Source/output path resolution
│
├── GloRiSe/
├── GFQA_v2/
├── USGS/
├── Hydat/
├── bayern/
├── HYBAM/
├── Rhine/
├── Eurasian_River/
├── Yajiang/
├── Mekong_Delta/
├── Myanmar/
├── Chao_Phraya_River/
├── Robotham/
├── NERC/
├── Fukushima/
├── Shashi_Jianli/
├── Huanghe/
├── Milliman/
├── HMA/
├── ALi_De_Boer/
├── Vanmaercke/
├── GSED/
├── Dethier/
└── RiverSed/
```

The repository may also contain legacy, auxiliary, validation, or development scripts. The canonical processing order is defined by `run_pipeline.py`.

---

## Running the source-data pipelines

The top-level runner records the canonical execution order for each source dataset.

```bash
# List available dataset pipelines
python run_pipeline.py --list

# Preview one dataset without executing it
python run_pipeline.py GloRiSe --dry-run

# Run one dataset
python run_pipeline.py Milliman

# Run several datasets
python run_pipeline.py USGS GFQA_v2 HYBAM

# Run all registered datasets
python run_pipeline.py --all
```

Source and output roots can be overridden when needed:

```bash
python run_pipeline.py GloRiSe \
  --source-root /path/to/Source \
  --output-root /path/to/Output_r
```

or through environment variables:

```bash
export SEDIMENT_SOURCE_ROOT=/path/to/Source
export SEDIMENT_OUTPUT_ROOT=/path/to/Output_r
```

---

## Source datasets used in the manuscript release

### Main station-reference sources

| Dataset | Canonical processing script(s) |
| --- | --- |
| GloRiSe | `GloRiSe/1_generate_netcdf_SS.py` -> `GloRiSe/2_qc_and_standardize_glorise.py` |
| GFQA_v2 | `GFQA_v2/gfqa_to_netcdf_daily_dualqc.py` |
| USGS NWIS | `USGS/process_usgs.py` |
| HYDAT | `Hydat/2_extract_sediment_data_prallel.py` -> `Hydat/3_update_sediment_nc_fixed.py` -> `Hydat/4_process_hydat_cf18.py` |
| Bayern | `bayern/convert_bayern_to_netcdf.py` -> `bayern/qc_and_standardize.py` |
| HYBAM | `HYBAM/hybam_comprehensive_processor.py` |
| Rhine | `Rhine/process_rhine.py` |
| Eurasian River | `Eurasian_River/process_eurasian_river.py` |
| Yajiang | `Yajiang/convert_to_nc.py` -> `Yajiang/process_yajiang.py` |
| Mekong Delta | `Mekong_Delta/process_mekong_delta.py` |
| Myanmar Rivers | `Myanmar/convert_to_netcdf.py` |
| Chao Phraya River | `Chao_Phraya_River/process_chao_phraya.py` |
| Robotham | `Robotham/convert_to_netcdf_v2.py` |
| NERC Avon | `NERC/convert_NERC_to_netcdf.py` |
| Fukushima | `Fukushima/fukushima_qc_and_cf_enhancement.py` |
| Shashi-Jianli | `Shashi_Jianli/process_shashi_jianli.py` |
| Huanghe | `Huanghe/convert_to_netcdf.py` -> `Huanghe/qc_and_standardize.py` |

### Climatology sources

| Dataset | Canonical processing script(s) |
| --- | --- |
| Milliman | `Milliman/1_convert_to_netcdf.py` -> `Milliman/2_fix_netcdf_units.py` -> `Milliman/3_add_variables_to_netcdf.py` -> `Milliman/4_convert_units_to_daily.py` -> `Milliman/5_qc_and_standardize.py` |
| HMA | `HMA/convert_to_netcdf_cf18_qc.py` |
| Ali and De Boer | `ALi_De_Boer/process_data_tool.py` |
| Vanmaercke | `Vanmaercke/convert_to_netcdf.py` -> `Vanmaercke/qc_and_standardize.py` |
| Huanghe | `Huanghe/convert_to_netcdf.py` -> `Huanghe/qc_and_standardize.py` |

### Satellite-derived sources

| Dataset | Canonical processing script(s) |
| --- | --- |
| GSED | `GSED/1_process_gsed_cf18.py` |
| Dethier | `Dethier/process_dethier_tool.py` |
| RivSed | `RiverSed/convert_to_netcdf.py` |

`RiverSed` is the repository directory corresponding to the product referred to as **RivSed** in the manuscript.

---

## Reproducibility notes

- `run_pipeline.py` should be treated as the canonical source-processing entry point.
- New source processors should reuse functions from `code/` rather than duplicate shared QC or conversion logic.
- Source-specific processing is retained where the original data structure requires it.
- Raw source files are not necessarily distributed with this repository and may need to be obtained from the original data providers.
- EUSEDcollab processing code may be retained for reproducibility, but the corresponding source observations are not redistributed when redistribution permission is unavailable.
- Exact reproduction of a published dataset version requires the source files and repository revision associated with that release.
