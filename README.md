# Sediment Source-Data Harmonization and Quality-Control Workflow

This repository contains the **source-level processing workflow** used to prepare river discharge (Q), suspended sediment concentration (SSC), and suspended sediment load (SSL) observations for the dataset described in:

> *A harmonized global station-reference dataset of river discharge, suspended sediment concentration, and suspended sediment load*

This README follows the methodology structure of **Sect. 3.1–3.4** of the manuscript, but documents **only the processing implemented in the `Sed_data` repository**.

The repository converts heterogeneous source files into standardized, quality-controlled, source-traceable NetCDF products. Its main responsibilities are:

- metadata harmonization;
- temporal standardization;
- variable mapping and unit conversion;
- derivation of missing sediment variables where appropriate;
- coordinate standardization;
- source-level quality control;
- CF-1.8 / ACDD-1.3 compatible NetCDF output.

The later release-level operations described in manuscript Sect. 3.4 are outside the scope of this repository and are therefore not documented here.

---

## Manuscript-to-code map

| Manuscript section | Scope covered by `Sed_data` | Main scripts/modules |
| --- | --- | --- |
| **Sect. 3.1** Metadata, temporal, variable, and unit harmonization | Source-specific parsing, metadata standardization, temporal harmonization, variable mapping, unit conversion, and derivation of missing Q/SSC/SSL variables | `run_pipeline.py`; source-specific processors; `code/time_utils.py`; `code/units.py`; `code/daily_aggregation.py`; `code/metadata.py`; `code/global_attrs.py`; `code/constants.py` |
| **Sect. 3.2** Georeferencing and basin matching | Coordinate parsing, conversion, and standardization required before later hydrological matching | Source-specific processors; `code/geo.py`; source-specific coordinate utilities |
| **Sect. 3.3** QC procedures and flagging system | Physical screening, statistical outlier detection, SSC-Q consistency checking, and flag propagation to derived variables | `code/qc.py`; `code/constants.py`; source-specific processors |
| **Sect. 3.4** Temporal screening, station consolidation, and time-series integration | Not implemented in `Sed_data`; the source-level outputs produced here provide the standardized input products required by later integration | No `Sed_data` script corresponds directly to this section |

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

# 3.1 Metadata, temporal, variable, and unit harmonization

Each source dataset is processed independently because the original files differ in format, variable definitions, units, metadata structure, and temporal support.

The goal of this stage is to convert these heterogeneous inputs into a common station-oriented NetCDF representation while preserving source provenance.

## Metadata harmonization

Where available, source processors preserve and standardize:

- source station identifier;
- station name;
- river name;
- longitude and latitude;
- source-reported upstream area;
- source dataset name;
- source URL and references;
- creator or contributing institution;
- temporal coverage;
- source-reported temporal resolution;
- variables provided by the source;
- processing history and data limitations.

Shared metadata utilities are implemented mainly in:

```text
code/metadata.py
code/global_attrs.py
code/output.py
```

The standardized NetCDF files follow CF-1.8 / ACDD-1.3 conventions as far as supported by the source metadata.

## Temporal harmonization

Source-specific time encodings are converted to a common NetCDF-compatible time representation.

Shared temporal helpers are implemented in:

```text
code/time_utils.py
```

For climatological or multi-year mean observations, a representative time is assigned using:

```text
1 July of the middle year of the source-reported period
```

while the original temporal span is preserved in metadata.

Higher-frequency observations can be aggregated to daily values when required by the source workflow. Shared daily aggregation utilities are implemented in:

```text
code/daily_aggregation.py
```

The daily aggregation workflow:

1. collapses duplicate timestamps;
2. groups sub-daily observations by calendar day;
3. calculates daily Q and SSC means from valid observations;
4. preserves source SSL where available;
5. derives daily SSL from daily Q and SSC only when source SSL is unavailable;
6. propagates QC flags to the aggregated values.

## Variable harmonization

The three core variables are standardized as:

| Variable | Meaning | Standard unit |
| --- | --- | --- |
| `Q` | river discharge | `m3 s-1` |
| `SSC` | suspended sediment concentration | `mg L-1` |
| `SSL` | suspended sediment load | `ton day-1` |

Variable mapping follows the physical meaning of the source variable rather than only its original name.

Examples include:

- compatible source TSS concentration measurements mapped to SSC;
- source sediment-flux variables mapped to SSL;
- source runoff or annual flux variables converted to the corresponding standardized Q or SSL representation where required.

## Derived variables

When one sediment variable is missing but the other required variables are available at the same station and time step, the missing variable may be derived.

The shared Q-SSC-SSL relation is:

```text
SSL (ton day-1) = Q (m3 s-1) * SSC (mg L-1) * 0.0864
```

where:

```text
0.0864 = 86400 s day-1 * 1000 L m-3 / 1e9 mg ton-1
```

The inverse relation can be used to derive SSC:

```text
SSC = SSL / (Q * 0.0864)
```

or Q when SSC and SSL are available:

```text
Q = SSL / (SSC * 0.0864)
```

Shared constants and conversion utilities are implemented in:

```text
code/constants.py
code/units.py
```

Directly reported values are preferred. Derived values are explicitly distinguished from directly reported observations through the quality flags described in Sect. 3.3.

For source data expressed as sediment totals over longer periods, the scripts convert them to daily-equivalent load rates where required by the harmonized representation.

---

# 3.2 Georeferencing and spatial standardization

Within `Sed_data`, Sect. 3.2 is represented by **source-level coordinate preparation**.

The repository standardizes source coordinates so that every usable station or remotely sensed location can be represented consistently by geographic longitude and latitude.

Shared coordinate utilities are implemented in:

```text
code/geo.py
```

Source-specific processors may additionally perform dataset-specific coordinate conversion before NetCDF output.

Examples include:

- conversion of degrees-minutes-seconds coordinates to decimal degrees;
- conversion from projected coordinate systems to geographic longitude/latitude;
- extraction of station coordinates from source metadata or associated spatial files;
- preservation of source-reported upstream area where available;
- validation of missing or invalid coordinate values.

For example, the Bayern workflow converts projected UTM coordinates to WGS84 longitude/latitude before writing the standardized station files.

Satellite-derived products retain source-specific spatial metadata needed to describe their river-reach or observation locations. Their source-level spatial information is preserved without forcing them to be treated as conventional gauge outlets.

This repository does **not** perform the final basin-assignment or station-consolidation operations described later in the manuscript.

---

# 3.3 Quality-control procedures and flagging system

Quality control is performed independently for Q, SSC, and SSL.

The shared QC implementation is:

```text
code/qc.py
```

and the shared flag definitions are in:

```text
code/constants.py
```

## Final QC flags

| Flag | Meaning |
| ---: | --- |
| `0` | good data |
| `1` | derived / estimated data |
| `2` | suspect data |
| `3` | bad data |
| `9` | missing data |

The internal stepwise QC flag `8` means that a particular QC test was not applied, usually because too few valid observations were available. It is not one of the five final release-quality categories.

## QC1 - physical plausibility

QC1 identifies physically invalid and missing values.

```text
missing / non-finite / fill value -> flag 9
negative Q, SSC, or SSL           -> flag 3
otherwise                          -> eligible for later QC
```

Zero values are retained as physically valid but are excluded from logarithmic tests.

## QC2 - log-IQR statistical screening

QC2 identifies station-specific statistical outliers.

For each independently reported variable:

1. retain positive finite observations passing QC1;
2. require at least five eligible observations;
3. transform the values using `log10`;
4. calculate the 25th percentile (Q1), 75th percentile (Q3), and IQR;
5. flag observations outside the `1.5 * IQR` bounds in log space as suspect.

The corresponding helper functions include:

```text
compute_log_iqr_bounds()
apply_log_iqr_screening()
apply_qc2_log_iqr_if_independent()
```

Derived variables are not treated as fully independent observations for QC2; their final flags instead reflect both their derived status and the quality of their input variables.

## QC3 - SSC-Q hydrological consistency

QC3 checks whether SSC is consistent with the station-specific relation between discharge and suspended sediment concentration.

For stations with at least five valid paired Q-SSC observations:

1. fit a relation between `log10(Q)` and `log10(SSC)`;
2. calculate residuals from the fitted relation;
3. construct a residual IQR envelope;
4. flag SSC values outside the envelope as suspect.

The relevant implementation includes:

```text
build_ssc_q_envelope()
check_ssc_q_consistency()
```

If SSL was derived from Q and SSC, an SSC-Q inconsistency can also propagate to the derived SSL flag.

## Flag propagation for derived values

Derived variables inherit quality information from their required inputs.

The shared logic follows the effective priority:

```text
bad input       -> derived value = bad
missing input   -> derived value = missing
suspect input   -> derived value = suspect
otherwise       -> derived value = derived / estimated
```

This behavior is implemented through functions such as:

```text
propagate_derived_flag_from_inputs()
propagate_input_flags_to_derived_ssl()
apply_hydro_qc_with_provenance()
```

The QC system therefore distinguishes between:

- directly reported observations that pass QC;
- values derived from other usable variables;
- statistically or hydrologically suspect observations;
- physically invalid values;
- missing values.

Original numerical values are retained so that users can decide which QC classes are appropriate for their application.

---

# 3.4 Interface to the later integration stage

No release-level station consolidation or time-series arbitration is implemented in `Sed_data`.

The role of this repository is to produce standardized source-station files containing the information needed for later integration, including:

- standardized Q, SSC, and SSL variables;
- standardized units;
- source-level temporal information;
- longitude and latitude;
- upstream area where available;
- final and stepwise QC flags;
- source station identity;
- dataset provenance;
- CF/ACDD metadata.

Accordingly, Sect. 3.4 is included here only to define the **output boundary** of the `Sed_data` workflow. No downstream integration scripts are documented in this README.

---

## Standard source-level output

The source processors generally write one standardized NetCDF representation per source station or source location.

Typical variables include:

```text
time
lat
lon
altitude
upstream_area

Q
SSC
SSL

Q_flag
SSC_flag
SSL_flag
```

When available, stepwise QC variables are also retained, for example:

```text
Q_flag_qc1_physical
Q_flag_qc2_log_iqr

SSC_flag_qc1_physical
SSC_flag_qc2_log_iqr
SSC_flag_qc3_ssc_q

SSL_flag_qc1_physical
SSL_flag_qc2_log_iqr
SSL_flag_qc3_from_ssc_q
```

Typical standardized units are:

```text
Q   : m3 s-1
SSC : mg L-1
SSL : ton day-1
```

Typical flag meanings are:

```text
0 = good
1 = derived / estimated
2 = suspect
3 = bad
9 = missing
```

---

## Reproducibility notes

- `run_pipeline.py` should be treated as the canonical source-processing entry point.
- New source processors should reuse functions from `code/` rather than duplicate shared QC or conversion logic.
- Source-specific processing is retained where the original data structure requires it.
- Raw source files are not necessarily distributed with this repository and may need to be obtained from the original data providers.
- EUSEDcollab processing code may be retained for reproducibility, but the corresponding source observations are not redistributed when redistribution permission is unavailable.
- Exact reproduction of a published dataset version requires the source files and repository revision associated with that release.
