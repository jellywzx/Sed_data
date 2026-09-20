# Processed-output record audit

The repository-wide audit reports how many source-level records remain eligible
for the sediment integration workflow after each dataset has been processed.

Run:

```bash
python audit_processed_outputs.py
```

or use a custom output root:

```bash
python audit_processed_outputs.py --output-root /path/to/Output_r
```

The command writes:

```text
Output_r/audit/dataset_output_audit.csv
Output_r/audit/dataset_output_audit_by_resolution.csv
```

## Record definitions

- `nc_time_steps`: all time steps written to final QC NetCDF files.
- `sediment_eligible_records`: records with non-missing SSC or SSL.
- `q_only_records`: Q exists while both SSC and SSL are missing.
- `fully_missing_records`: Q, SSC, and SSL are all missing.
- `screened_for_integration_records`: Q-only plus fully-missing records.
- `analysis_ready_sediment_records`: SSC or SSL exists with final flag 0 or 1.
- `sediment_suspect_bad_only_records`: sediment-bearing records for which no
  SSC/SSL value has final flag 0 or 1.

Final QC flag counts for Q, SSC, and SSL are also reported separately.

Suspect/bad values (flags 2/3) are retained in source NetCDF files and are not
counted as records removed by source processing.

## Scope

This is a post-processing audit. It intentionally does not infer raw-input
attrition from:

```text
raw source rows - final NetCDF time steps
```

because source processors may aggregate sub-daily data, collapse duplicate
timestamps, perform outer joins, or otherwise change row counts without
screening observations. If raw-source attrition is required, it should be
tracked by reason-specific counters inside the corresponding source processor.

For a correctly formed final output:

```text
nc_time_steps
= sediment_eligible_records
+ q_only_records
+ fully_missing_records
```

and:

```text
screened_for_integration_records
= q_only_records + fully_missing_records
```

The `flag_value_mismatch_count`, `files_missing_final_flags`, and
`unreadable_files` columns are diagnostics that should be zero before using
the summary as a publication statistic.
