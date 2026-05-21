# Project Change Document

## Pipeline Storage Format

- Changed the three normalization scripts to write Parquet instead of CSV:
  - `normalize_transit.py`
  - `normalize_eia_gas.py`
  - `normalize_fred_inflation.py`
- Changed `build_integrated_monthly_base.py` to read normalized Parquet inputs and write the integrated monthly base as Parquet.
- Added `pyarrow` to `requirements.txt` for pandas Parquet read/write support.
- Updated S3 output file names from `*.csv` to `*.parquet`.

## Environment Variable Cleanup

- Replaced overloaded env vars with source-specific names to avoid one script accidentally using another source's path or metadata.
- Raw input discovery now uses stable prefixes by default:
  - `TRANSIT_RAW_PREFIX`
  - `GAS_RAW_PREFIX`
  - `FRED_RAW_PREFIX`
- Exact raw keys remain available as optional overrides for backfills/debugging:
  - `TRANSIT_RAW_KEY`
  - `GAS_RAW_KEY`
  - `FRED_RAW_KEY`
- Normalized integration inputs now use stable prefixes by default:
  - `TRANSIT_NORMALIZED_PREFIX`
  - `GAS_NORMALIZED_PREFIX`
  - `INFLATION_NORMALIZED_PREFIX`
- Exact normalized keys remain available as optional overrides:
  - `TRANSIT_KEY`
  - `GAS_KEY`
  - `INFLATION_KEY`
- Output prefixes are now source-specific:
  - `TRANSIT_OUTPUT_PREFIX`
  - `GAS_OUTPUT_PREFIX`
  - `INFLATION_OUTPUT_PREFIX`
  - `INTEGRATED_OUTPUT_PREFIX`
- Source and series metadata variables are source-specific:
  - `TRANSIT_SOURCE_NAME`, `TRANSIT_SERIES_ID`
  - `GAS_SOURCE_NAME`, `GAS_SERIES_ID`
  - `FRED_SOURCE_NAME`, `FRED_SERIES_ID`

## Latest-Object Discovery

- Added S3 latest-key discovery helpers so scripts can find the latest `load_date=...`, `run_date=...`, or `run_id=...` object automatically.
- Normalizers no longer require manually editing dated raw S3 keys each month.
- The integrated builder no longer requires manually editing dated normalized Parquet keys each run.
- Fixed FRED latest-key selection so it picks `fred_inflation.json`, not `metadata.json`.

## ECS Run Metadata

- Changed ECS-oriented pipeline outputs from day-level `run_date=<YYYY-MM-DD>` folders to timestamped `run_id=<YYYY-MM-DDTHH-MM-SSZ>` folders.
- Added optional `PIPELINE_RUN_ID` support so an orchestrator, such as Step Functions, can pass the same run ID through every task in one pipeline execution.
- Added optional `IMAGE_URI` metadata capture so outputs can be traced back to the exact ECR image tag used for the run.
- Updated normalizers, the integrated monthly builder, and the feature table script to write `run_id` and runtime metadata into their metadata JSON outputs.
- Added `write_pipeline_manifest.py` to write a top-level manifest at `pipeline_runs/run_id=<run_id>/manifest.json` after feature table creation.

## Lambda Ingestion Copies

- Updated local copies of ingestion Lambda scripts to use source-specific env vars:
  - `lambda_gas.py`: `GAS_SOURCE_NAME`, `GAS_SOURCE_URL`
  - `lambda_inflation.py`: `FRED_SOURCE_NAME`, `FRED_API_KEY`
- Kept Lambda output contract aligned with downstream normalizers:
  - raw files under `raw/<source_name>/load_date=<date>/...`
  - latest ingestion state under `state/<source_name>/latest_ingested.json`
- Replaced Python 3.10 union type hints with Python 3.9-compatible `Optional[...]`.

## Local Development Setup

- Created `.venv` and installed project dependencies.
- Added/updated `requirements.txt` with:
  - `boto3`
  - `pandas`
  - `openpyxl`
  - `xlrd`
  - `pyarrow`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `python-dotenv`
- Added optional `.env` loading to local scripts using `python-dotenv`.
- Created `.gitignore` to exclude local secrets, virtualenv files, caches, and notebook checkpoints.

## Notebook Cleanup

- Cleaned `Copy_of_Integrated_transit_feature_eng.ipynb`:
  - consolidated repeated imports
  - cleared stale Colab outputs
  - added `.env` loading
  - changed local CSV loading to S3 Parquet loading
  - added latest integrated Parquet discovery under `INTEGRATED_OUTPUT_PREFIX`
  - added missing `feature_set_library` construction
  - replaced fixed row trimming with `dropna` on required model inputs
  - added guards for missing feature columns and empty train/validation splits
  - fixed Python 3.9-incompatible type annotation
- Notebook now reads:
  - explicit `INTEGRATED_KEY` if set
  - otherwise latest `integrated_monthly_base.parquet` under `INTEGRATED_OUTPUT_PREFIX`

## Notebook Modeling Alignment Fix

- Updated `Copy_of_Integrated_transit_feature_eng.ipynb` to avoid dropping months solely because exogenous monthly inputs are missing.
- Added exogenous imputation for gas and CPI fields:
  - interior gaps are filled with linear interpolation between observed months
  - trailing gaps are filled with a simple projection from the most recent five observed values
  - imputed values are flagged with numeric audit columns
- Changed H3 target construction from positional `shift(-3)` to a calendar-date merge on `date + 3 months`.
- This preserves true monthly horizon alignment when an intermediate month has missing exogenous values, such as the missing CPI values for `2025-10-01`.

## Feature Table Script

- Added `create_feature_table.py` as the pipeline step after `build_integrated_monthly_base.py`.
- The script reads integrated monthly Parquet, validates monthly continuity, trims leading unavailable gas history, imputes exogenous gas/CPI gaps, builds H3 time/regime/lag/rolling/target features, and writes a feature-store folder.
- Local output when `--output-dir feature_store/integrated_monthly_h3/run_id=<run_id>` is provided:
  - `feature_store/integrated_monthly_h3/run_id=<run_id>/feature_table.parquet`
  - `feature_store/integrated_monthly_h3/run_id=<run_id>/feature_families.json`
  - `feature_store/integrated_monthly_h3/run_id=<run_id>/feature_metadata.json`
  - `feature_store/integrated_monthly_h3/run_id=<run_id>/imputation_log.parquet`
  - `feature_store/integrated_monthly_h3/run_id=<run_id>/feature_family_audit.parquet`
- Current default S3 output:
  - `features/integrated_monthly_h3/run_id=<run_id>/feature_table.parquet`
  - `features/integrated_monthly_h3/run_id=<run_id>/feature_families.json`
  - `features/integrated_monthly_h3/run_id=<run_id>/feature_metadata.json`
  - `features/integrated_monthly_h3/run_id=<run_id>/imputation_log.parquet`
  - `features/integrated_monthly_h3/run_id=<run_id>/feature_family_audit.parquet`

## Streamlined Modeling Script

- Added `run_aws_streamlined_models.py` as the starting point for the AWS-friendly modeling comparison layer.
- The script currently resolves and loads the feature table plus `feature_families.json` from local paths, an explicit S3 run ID, or the latest S3 feature-table run.
- It mirrors the existing feature family definitions by reading the generated feature-family artifact instead of redefining them separately.
- It builds a 2021-present H3 UPT rolling evaluation frame.
- It now carries forward modeling patterns from the notebook:
  - seasonal naive benchmark
  - raw/direct and residual modes
  - MAE, RMSE, R2, directional accuracy, and improvement-vs-naive metrics
  - notebook-aligned XGBoost parameter style
- The AWS-streamlined grid currently compares seasonal naive, Ridge, Lasso, and XGBoost across the existing feature families.
- XGBoost produces monthly predictions but refits annually by default to keep the AWS demo lightweight.
- Local validation loaded `275` feature rows, `118` columns, `14` feature families, and `60` evaluable monthly forecast origins from `2021-01-01` through `2025-12-01`.
- Local smoke validation produced `9,240` predictions and selected `ridge__raw__history_regime_service__alpha-10.0` as champion on the weighted score.

## Pipeline Test Results

- Successfully tested the local pipeline against S3:
  - `normalize_transit.py`
  - `normalize_eia_gas.py`
  - `normalize_fred_inflation.py`
  - `build_integrated_monthly_base.py`
- Latest successful outputs:
  - `normalized/transit/run_date=2026-05-15/transit_normalized.parquet`
  - `normalized/gas/run_date=2026-05-15/gas_monthly_normalized.parquet`
  - `normalized/inflation/run_date=2026-05-15/inflation_normalized.parquet`
  - `integrated/monthly_base/run_date=2026-05-15/integrated_monthly_base.parquet`
- ECS tests later confirmed these scripts can run from the shared container image:
  - `normalize_transit.py`
  - `normalize_eia_gas.py`
  - `normalize_fred_inflation.py`
  - `build_integrated_monthly_base.py`
  - `create_feature_table.py`
- Integrated output shape:
  - `291` rows
  - `21` columns
  - date range `2002-01-01` to `2026-03-01`

## Known Follow-Ups

- Upgrade local Python from 3.9 to 3.10+ because boto3 has announced Python 3.9 support deprecation.
- Consider adding a committed `.env.example` with safe placeholder values.
- Consider packaging shared S3 latest-key discovery logic into a utility module if more scripts are added.
