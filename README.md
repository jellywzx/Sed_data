# Sediment Reference Dataset Source Processing Scripts

This repository contains source-specific processing scripts and shared Python
utilities for converting river sediment observations into CF-1.8 / ACDD-1.3
NetCDF products.

## Environment

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate sed-reference-scripts
```

Or install the Python dependencies into an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Usage

List available dataset processors:

```bash
python run_pipeline.py --list
```

Preview or run a dataset:

```bash
python run_pipeline.py GloRiSe --dry-run
python run_pipeline.py Milliman
```

Override local data roots with command-line options or environment variables:

```bash
python run_pipeline.py GloRiSe \
  --source-root /path/to/Source \
  --output-root /path/to/Output_r
```

Detailed workflow notes are in `docs/readme.md` and `docs/PIPELINES.md`.

## Publication Notes

This public branch tracks code, lightweight documentation, configuration, and
small static metadata only. Large source data, generated NetCDF files, reports,
logs, plots, and local maintenance outputs should be distributed through the
associated data repository or regenerated locally.

Before public release, check:

```bash
git status --short
python -m py_compile *.py code/*.py */*.py
python run_pipeline.py --list
```

Citation metadata is in `CITATION.cff`. Source code is licensed under MIT;
data products and manuscript figures should use the license stated in their
own release metadata.
