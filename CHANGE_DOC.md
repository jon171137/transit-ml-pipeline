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
- Added explicit local output base URI arguments for smoke tests:
  - `--results-base-uri`
  - `--dashboard-base-uri`
- Confirmed the script runs successfully as an ECS/Step Functions state after `write_pipeline_manifest.py`.
- The AWS workflow now writes streamlined modeling artifacts under `model_results/aws_streamlined/run_id=<run_id>/` and dashboard-ready artifacts under `dashboard/aws_streamlined/run_id=<run_id>/`.

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
  - `write_pipeline_manifest.py`
  - `run_aws_streamlined_models.py`
- Integrated output shape:
  - `291` rows
  - `21` columns
  - date range `2002-01-01` to `2026-03-01`

## Known Follow-Ups

- Upgrade local Python from 3.9 to 3.10+ because boto3 has announced Python 3.9 support deprecation.
- Consider adding a committed `.env.example` with safe placeholder values.
- Consider packaging shared S3 latest-key discovery logic into a utility module if more scripts are added.

## Experiment Metadata and MLflow Tracking

- Added `docs/experiment_metadata_contract.md` to define the broader experiment artifact contract.
- The contract separates raw experiment facts from dashboard exports:
  - `experiment_manifest.json`
  - `model_runs.parquet`
  - `predictions.parquet`
  - `metrics.parquet`
  - `feature_sets.parquet`
  - `feature_importance.parquet`
  - dashboard-ready derived artifacts
- Updated `run_aws_streamlined_models.py` so the existing AWS-streamlined outputs now include durable fields for future dashboard and DuckDB use:
  - `experiment_id`
  - `pipeline_run_id`
  - `model_config_id`
  - `model_run_id`
  - `feature_set_id`
  - `model_family`
  - `model_build`
  - `hyperparameters_json`
  - `metric_extras_json`
- Added `feature_sets.parquet` and `experiment_manifest.json` to the model result outputs.
- Added optional MLflow logging to `run_aws_streamlined_models.py`.
  - Enable with `--enable-mlflow` or `ENABLE_MLFLOW=true`.
  - Optional controls:
    - `--mlflow-tracking-uri`
    - `--mlflow-experiment-name`
    - `--mlflow-run-name`
  - The script logs experiment-level params, champion metrics, summary counts, and compact artifacts.
- Added `mlflow` to `requirements.txt`.
- Local schema smoke test passed against `feature_store/test_run_id`:
  - `3,696` predictions from a 2024-present smoke window
  - `154` metric rows
  - `28` feature set rows
- Local MLflow smoke test passed with tracking URI `/private/tmp/transit_ml_mlflow_smoke/mlruns`.

## Dashboard Overview Export

- Added dashboard-ready overview artifacts from `run_aws_streamlined_models.py`:
  - `overview_top_models.parquet`
  - `overview_prediction_paths.parquet`
- Updated `dashboard/app.py` so the first Overview tab now follows the planned top-model comparison layout:
  - model family filter
  - model build filter
  - feature family filter
  - ranking metric selector
  - top-five model prediction paths against actual ridership and seasonal naive
  - model detail table with hyperparameters and core metrics
- The dashboard remains backward-compatible with older artifact folders by deriving the overview view from `model_leaderboard.parquet` and `forecast_paths.parquet` when the new overview files are absent.
- Local dashboard smoke test passed using `/private/tmp/transit_ml_overview_smoke/dashboard`.
- Standardized dashboard filter naming around `model_family`, `model_build`, `mode`, and `feature_family_name`.
- Added matching filters to the Model Performance tab so larger experiment runs can be sliced before viewing the leaderboard, rolling error, and model-build comparison.

## Wider Historical Simulation Controls

- Added configurable historical evaluation controls to `run_aws_streamlined_models.py`:
  - `--as-of-end`
  - `--as-of-frequency-months`
  - `--refit-frequency-months`
- The runner already trained each forecast from rows strictly before `as_of_date`; the new controls make that simulation window and cadence explicit.
- Changed the streamlined modeling default XGBoost/refit cadence to monthly.
- Added `evaluation_period` and `shock_period_flag` to prediction/performance outputs for future shock/recovery views.
- Added `experiment_manifest.json` to dashboard outputs so the app can display horizon, forecast cadence, and refit cadence.
- Dashboard updates:
  - Header now explains as-of dates, target dates, forecast horizon, forecast cadence, and refit cadence.
  - Overview now includes a target-date window.
  - Forecast Explorer now includes as-of and target-date windows.
  - Model Performance now includes an as-of-date window for rolling error.
- Local wide smoke test passed:
  - command used `--as-of-start 2016-01-01 --as-of-frequency-months 3 --refit-frequency-months 3`
  - `40` quarterly as-of origins
  - target dates from `2016-04-01` through `2026-01-01`
  - `6,160` predictions
  - dashboard rendered successfully from `/private/tmp/transit_ml_wide_smoke/dashboard`
- Local monthly-refit simulation passed:
  - command used `--as-of-start 2016-01-01 --as-of-frequency-months 1 --refit-frequency-months 1 --xgb-refresh-months 1`
  - `120` monthly as-of origins
  - target dates from `2016-04-01` through `2026-03-01`
  - `18,480` predictions
  - copied dashboard artifacts to `dashboard_artifacts/aws_streamlined/latest`

## Period-Specific Metrics

- Updated `metrics.parquet` to use long-form evaluation-scope rows:
  - `overall`
  - `pre_covid`
  - `covid_shock`
  - `recovery`
  - `recent`
- Added dashboard-wide leaderboard metrics derived from those long rows:
  - `pre_covid_mae`
  - `covid_shock_mae`
  - `recovery_mae`
  - `recent_mae`
  - `shock_penalty`
  - `recovery_ratio`
  - `recent_recovery_ratio`
  - `shock_abs_increase`
- Added dashboard explanation panels describing the period definitions and ratio interpretation.
- Local period-metrics simulation passed:
  - `metrics.parquet` shape: `770` rows
  - `154` rows per evaluation scope
  - dashboard leaderboard shape: `154` rows with period-specific columns
  - copied latest dashboard artifacts to `dashboard_artifacts/aws_streamlined/latest`

## Dashboard Period Ranking and Interaction Features

- Updated `dashboard/app.py` so Model Performance can rank by:
  - overall selection score, MAE, RMSE, R2, directional accuracy
  - pre-COVID MAE
  - COVID shock MAE
  - recovery MAE
  - recent MAE
  - shock/recovery ratio metrics
- Added a dedicated Period Metrics table to Model Performance so shock,
  recovery, and recent metrics are visible without relying on horizontal table
  scrolling.
- Made dashboard artifact caching sensitive to file modification time so
  replacing files under `dashboard_artifacts/aws_streamlined/latest` refreshes
  the schema and values correctly.
- Added targeted regime interaction features to `create_feature_table.py`:
  - history x COVID/post-COVID flags
  - time/seasonality x COVID/post-COVID flags
  - gas/CPI x COVID/post-COVID flags
  - service levels x COVID/post-COVID flags
- Added interaction-expanded feature families:
  - `history_regime_linear_interactions`
  - `history_regime_time_linear_interactions`
  - `history_regime_exog_linear_interactions`
  - `history_regime_all_exog_linear_interactions`
- Local feature-table smoke test passed:
  - feature table shape: `275` rows x `148` columns
  - `30` generated interaction columns
  - `18` feature families
  - `0` missing family features
- Added scoped experiment controls to `run_aws_streamlined_models.py`:
  - repeatable `--include-feature-family`
  - repeatable `--include-model-type`
- Local interaction experiments completed:
  - full interaction run: `18` feature families, `16,632` predictions, `198` leaderboard configurations
  - scoped interaction run: `5` feature families, `4,620` predictions, `55` leaderboard configurations
  - full and scoped dashboard artifacts copied to:
    - `dashboard_artifacts/aws_streamlined/interaction_full`
    - `dashboard_artifacts/aws_streamlined/interaction_scoped`
- Initial result:
  - best overall score in the full run was `xgboost`, raw mode, `history_regime_time_linear_interactions`
  - comparable non-interaction `history_regime_time` was second and within the champion equivalence band

## King County Income Source

- Added `lambda_income.py` for FRED ingestion of King County median household
  income:
  - FRED series: `MHIWA53033A052NCEN`
  - source: U.S. Census Bureau SAIPE via FRED
  - annual observations from `1989` through latest available `2024`
- Added `normalize_fred_income.py`.
  - Converts annual income to monthly prior-year context.
  - Emits:
    - `king_county_median_household_income_prior_year`
    - `king_county_monthly_household_income_prior_year`
    - `king_county_income_yoy_pct_prior_year`
    - `king_county_income_2yr_pct_prior_year`
    - `income_reference_method`
  - Projects missing future reference years using a five-year dollar trend and
    labels those rows as `projected_5yr_dollar_trend`.
- Updated `build_integrated_monthly_base.py`:
  - income normalized input is optional but included when available
  - supports local path, S3 key, or `s3://` URI inputs
  - supports local `--output-dir` for integration smoke tests
- Updated `create_feature_table.py`:
  - carries optional income columns into the feature table
  - adds income feature families:
    - `history_regime_income`
    - `history_regime_income_pressure`
    - `history_regime_income_linear_interactions`
  - adds income pressure features:
    - `income_yoy_pct_x_gas_price_yoy_diff`
    - `income_yoy_pct_x_cpi_all_items_yoy_diff`
    - `income_2yr_pct_x_cpi_core_yoy_diff`
    - `gas_price_to_monthly_income`
- Local validation:
  - FRED income fetch returned `36` annual observations, `1989` through `2024`
  - normalized income table shape: `291` rows x `11` columns
  - integration with income shape: `291` rows x `29` columns
  - income-aware feature table shape: `275` rows x `161` columns
  - `21` feature families with `0` missing family features
  - income modeling smoke test produced `2,016` predictions and `120` metric rows

## Feature Policies and Local Parallelism

- Updated `run_aws_streamlined_models.py` with repeatable feature policy
  controls:
  - `--feature-policy none`
  - `--feature-policy corr_pruned`
- `corr_pruned` currently applies to linear models only.
  - It computes correlations inside each as-of training window.
  - It drops columns above the configured correlation threshold before fitting.
  - It records the reduced `selected_feature_names_json` per model run.
  - Non-linear/baseline model types fall back to `none`.
- Added `feature_policy` to:
  - model configuration IDs
  - feature set IDs
  - predictions
  - model runs
  - metrics
  - leaderboard/dashboard artifacts
- Added `--n-jobs` process-level parallelism for independent model
  configuration tasks.
  - XGBoost now uses one internal thread per configuration to avoid nested
    thread oversubscription during parallel runs.
- Updated `dashboard/app.py`:
  - Overview, Forecast Explorer, and Model Performance now expose
    `feature_policy` filters.
  - Leaderboard/detail tables include `feature_policy` when available.
- Local smoke tests passed:
  - sequential feature-policy run produced `48` predictions
  - parallel `--n-jobs 2` feature-policy run produced `48` predictions
  - `corr_pruned` reduced the income-pressure smoke family from `66` features
    to `46` selected features

## Medium Experiment V1

- Ran the first medium local dashboard-shaping experiment.
- Scope:
  - monthly as-of dates from `2016-01-01` through `2025-12-01`
  - 3-month UPT forecast horizon
  - raw and residual modes
  - models: seasonal naive, Ridge, Lasso, XGBoost
  - feature policies: `none`, `corr_pruned`
  - 8 representative feature families spanning history, regime/time,
    exogenous, income, and interaction variants
- Outputs:
  - `18,240` prediction rows
  - `18,240` model-run rows
  - `760` metric rows
  - `152` leaderboard configurations
- Dashboard artifacts were written to:
  - `dashboard_artifacts/aws_streamlined/medium_v1`
  - copied to `dashboard_artifacts/aws_streamlined/latest`
- The selected champion was XGBoost, raw mode,
  `history_regime_time`, `feature_policy=none`.
  - MAE: `428,050`
  - RMSE: `841,868`
  - selection score: `531,504`
- The raw best score was slightly lower for
  `history_regime_time_linear_interactions`, but the simplicity-aware champion
  rule selected the simpler non-interaction feature family because it was within
  the 2 percent equivalence band.
- The run definition is documented in `docs/medium_experiment_v1.md`.

## Dashboard Overview Ranking Fix

- Investigated Overview chart behavior when ranking by period-specific metrics.
- Findings:
  - Pre-COVID ranking often selected multiple seasonal-naive configurations
    with identical predictions because seasonal naive does not actually use
    the attached feature-family labels.
  - Recent-MAE ranking could select a model with strong recent performance but
    extreme earlier-period predictions, which distorted the full-window chart.
- Updated `dashboard/app.py`:
  - period-specific ranking metrics now default the chart date window to the
    matching target period
  - duplicate prediction paths are skipped so the Top Five chart displays
    distinct lines
  - Overview tables now include period-specific MAE fields alongside overall
    metrics and derived ratios

## Seasonal Naive Baseline Cleanup

- Updated `run_aws_streamlined_models.py` so seasonal naive is no longer crossed
  with every feature family and feature policy.
- Seasonal naive is now emitted once as:
  - `model_family=baseline`
  - `model_build=seasonal_naive`
  - `feature_family_name=baseline_naive`
  - `feature_policy=none`
  - `mode=raw`
- This keeps the baseline available for charts and improvement-vs-naive metrics
  without polluting the leaderboard with identical duplicate prediction paths.
- Smoke test passed:
  - requested 3 feature families and 2 feature policies
  - leaderboard contained exactly 1 seasonal-naive configuration
  - forecast paths contained exactly 1 seasonal-naive config

## Project Overview Page Draft

- Added `dashboard/content.py` as a centralized place for reusable dashboard
  narrative copy.
- Moved period-metric explanation text out of inline Streamlit rendering blocks.
- Added a new `Project Overview` tab describing:
  - the forecasting-under-disruption motivation
  - the AWS/local system split
  - the rolling as-of-date experiment structure
  - the dashboard's purpose for technical review
- Renamed the original `Overview` tab to `Modeling Overview` to separate the
  project narrative from the model-comparison page.

## Dashboard Navigation Refactor

- Moved the dashboard to a two-level navigation structure.
- Sidebar sections now define the main project pages:
  - `Project Overview`
  - `System`
  - `Data`
  - `Experiment`
  - `Results Explorer`
- The existing analytical tabs now live under `Results Explorer`:
  - `Modeling Overview`
  - `Forecast Explorer`
  - `Model Performance`
  - `Feature Strategy`
  - `Operational Footprint`
- Added first-pass placeholder copy for `System`, `Data`, and `Experiment` in
  `dashboard/content.py` so those pages can be iterated independently from the
  results explorer.

## Dashboard Visual Identity Pass

- Added a light civic-style visual treatment inspired by the King County
  reference palette without using official labels, icons, or branding.
- Added a persistent page banner:
  - `Personal Forecasting Project by Jon Sellers`
- Added Streamlit theme overrides for:
  - teal top accent
  - blue link accents
  - quieter sidebar surface
  - tab and metric color polish

## Dashboard Image Assets

- Added `dashboard/assets/images/` for Streamlit-displayed project screenshots.
- Added a reusable image gallery renderer in `dashboard/app.py`.
- The `System` page now automatically displays supported images from that
  folder.
- Supported image types:
  - `.png`
  - `.jpg`
  - `.jpeg`
  - `.webp`

## Data Page Draft And Compact Header

- Reworked the persistent experiment summary header into a compact KPI strip so
  static pages have more vertical room.
- Expanded the `Data` page with:
  - source and processing map
  - feature family examples from the loaded artifact summary
  - feature-type examples for lags, rolling features, time features, regime
    indicators, exogenous context, and targeted interactions
  - a single rolling forecast step example showing as-of date, target date,
    evaluation period, model prediction, seasonal naive prediction, and error

## Project Overview Copy Pass

- Revised `Project Overview` language to read more like a polished portfolio
  introduction.
- Removed overly direct motivation language and reframed the project around why
  monthly transit ridership is a useful forecasting case study:
  - seasonality
  - long-run trend
  - operational context
  - economic context
  - COVID-era structural break
- Expanded the page to connect modeling goals, system design, rolling as-of
  evaluation, and the intended reviewer reading path.

## Experiment Page Draft

- Expanded the `Experiment` page from placeholder copy into a fuller first
  draft covering:
  - rolling historical as-of forecasting setup
  - comparison dimensions for model family, model build, mode, feature family,
    and feature policy
  - leaderboard selection score and simplicity-aware champion rule
  - period-specific metrics for pre-COVID, COVID shock, recovery, and recent
    windows
  - current medium-run scope versus the planned broader experiment
  - reviewer-oriented questions to inspect beyond the top leaderboard row

## DuckDB Experiment Mart

- Added `build_experiment_mart.py`.
- Added `duckdb` to `requirements.txt`.
- The mart builder loads model-result artifacts into a local DuckDB file:
  - `predictions`
  - `model_runs`
  - `metrics`
  - `feature_importance`
  - `feature_sets`
  - `feature_family_summary`
  - JSON manifest tables
  - `experiment_runs`
- The builder creates dashboard-shaped views and can export dashboard-ready
  Parquet/JSON files.
- Smoke test passed against `medium_v1`:
  - DuckDB file: `experiments_output/medium_v1/experiments.duckdb`
  - dashboard export: `dashboard_artifacts/aws_streamlined/medium_v1_from_duckdb`
  - exported dashboard files matched original row counts and column order
  - `model_leaderboard` preserved the wide period-metric columns needed by the
    Streamlit dashboard
- Updated `README.md` and `docs/experiment_metadata_contract.md` with the mart
  workflow and deployment interpretation.

## Documentation Status Pass

- Updated `README.md` to reflect the current project shape:
  - AWS-validated core pipeline
  - local-ready income expansion
  - Streamlit dashboard structure
  - medium local experiment status
  - DuckDB experiment mart role
  - near-term `medium_v2` rerun plan
- Added a caveat to `docs/medium_experiment_v1.md` that the run predates the
  seasonal-naive baseline cleanup.
- Added a new ignored local planning snapshot:
  - `local_notes/project_snapshot_2026-05-27.md`

## Medium Experiment V2

- Reran the representative medium local experiment after the seasonal-naive
  baseline cleanup.
- Scope matched `medium_v1`:
  - monthly as-of dates from `2016-01-01` through `2025-12-01`
  - target dates from `2016-04-01` through `2026-03-01`
  - models: seasonal naive, Ridge, Lasso, XGBoost
  - feature policies: `none`, `corr_pruned`
  - 8 representative feature families
- Outputs:
  - `145` model configurations
  - `17,400` prediction rows
  - `17,400` model-run rows
  - `725` metric rows
  - exactly `1` seasonal-naive leaderboard row
- Built a DuckDB mart from `medium_v2`:
  - `experiments_output/medium_v2/experiments.duckdb`
- Exported dashboard-compatible artifacts from DuckDB:
  - `dashboard_artifacts/aws_streamlined/medium_v2_from_duckdb`
- Updated local dashboard `latest` artifacts to the DuckDB export:
  - `dashboard_artifacts/aws_streamlined/latest`
- Added `docs/medium_experiment_v2.md`.

## Large Experiment Phase A Planning

- Added draft Phase A config:
  - `experiment_configs/large_phase_a_v1.yaml`
- Added config-driven smoke config:
  - `experiment_configs/phase_a_smoke.yaml`
- Added a dry-run planner:
  - `plan_large_experiment.py`
- Added planning documentation:
  - `docs/large_experiment_phase_a_plan.md`
- Added `PyYAML` to `requirements.txt` for experiment config parsing.
- Phase A draft scope:
  - seasonal naive
  - Ridge
  - Lasso
  - ElasticNet
  - Random Forest
  - Extra Trees
  - XGBoost
  - 21 income-aware feature families
  - raw and residual modes where applicable
  - `none` and `corr_pruned` feature policies for linear models
- Planner output for the draft config:
  - `180` monthly as-of origins
  - `2,227` total model configurations after adding the notebook-aligned
    XGBoost range
  - `400,860` estimated model/as-of rows
  - after runner updates, all listed Phase A model builds are implemented
- Expanded the Phase A XGBoost grid to include the stronger parameter ranges
  used in `transit_integrated_modeling.ipynb`: `n_estimators` 200/500,
  `max_depth` 3/4, `learning_rate` .05/.10, and `min_child_weight` 1/3,
  while retaining lower-learning-rate conservative settings.

## Phase A Runner Preparation

- Updated `run_aws_streamlined_models.py` to support:
  - YAML experiment configs via `--experiment-config`
  - ElasticNet
  - RandomForestRegressor
  - ExtraTreesRegressor
  - per-model-configuration chunk outputs
  - `--resume` reuse of completed chunks
  - failed-configuration logging in the chunk directory
  - `ensemble_method` metadata for bagging, randomized bagging, and boosting
- Tree model internal parallelism is kept at `n_jobs=1` so outer
  process-level parallelism remains the main concurrency control.
- Config-driven smoke test passed:
  - `37` model configurations
  - `222` prediction rows
  - `74` metric rows
  - `37` leaderboard rows
  - exactly `1` seasonal-naive leaderboard row
- Resume behavior passed by rerunning the smoke config and reusing all `37`
  completed chunks.
- Smoke outputs loaded successfully through `build_experiment_mart.py` and
  exported dashboard-compatible artifacts.

## Large Experiment Phase A Completed

- Ran the full `large_phase_a_v1` local experiment with:
  - `2,227` model configurations
  - `180` monthly as-of origins
  - `400,860` predictions/model-as-of rows
  - `11,135` metric rows
  - `17,684,904` feature-importance rows
  - no failed checkpoint files
- MLflow created/logged experiment:
  - `transit-forecasting-large-phase-a`
- Built the DuckDB experiment mart:
  - `experiments_output/large_phase_a_v1/experiments.duckdb`
- Exported dashboard-compatible artifacts:
  - `dashboard_artifacts/aws_streamlined/large_phase_a_v1_from_duckdb`
- Updated local dashboard `latest` artifacts to the Phase A export bundle.
- Phase A champion selected by the configured rule:
  - model: `xgboost`
  - mode: `raw`
  - feature family: `history_regime_time`
  - feature policy: `none`
  - parameters: `n_estimators=500`, `max_depth=4`, `learning_rate=0.05`,
    `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`
  - overall MAE: `340,381`
  - overall RMSE: `670,718`
  - overall R2: `0.874`
  - selection score: `422,965`

## Large Experiment Phase B Preparation

- Added Phase B autoregressive experiment support:
  - `run_autoregressive_models.py`
  - `experiment_configs/phase_b_autoregressive_v1.yaml`
  - `experiment_configs/phase_b_smoke.yaml`
  - `docs/large_experiment_phase_b_plan.md`
- Added `statsmodels` to `requirements.txt` for ARIMA/SARIMA/SARIMAX.
- Updated `run_aws_streamlined_models.py` metadata helpers so
  autoregressive model builds classify as `model_family=autoregressive`.
- Updated `plan_large_experiment.py` so it can summarize Phase B config grids.
- Added `combine_experiment_results.py` to merge Phase A and Phase B result
  folders into one dashboard-compatible artifact set.
- Phase B v1 planned scope:
  - `99` autoregressive model configurations
  - `180` monthly as-of origins
  - `17,820` estimated model/as-of rows
  - ARIMA, SARIMA, and SARIMAX builds
  - SARIMAX service, economic, income-pressure, and service-economic exogenous
    sets
- Phase B smoke test passed:
  - `5` autoregressive model configurations
  - `30` prediction rows
  - dashboard-compatible artifacts written under `/private/tmp/transit_phase_b_smoke`
- Tested merge plumbing by combining Phase A results with the Phase B smoke
  output and rebuilding a DuckDB/dashboard export under `/private/tmp`.

## Large Experiment Phase B Completed

- Ran the full `phase_b_autoregressive_v1` local experiment with:
  - `99` autoregressive model configurations
  - `180` monthly as-of origins
  - `17,820` predictions/model-as-of rows
  - `495` metric rows
  - no failed checkpoint files
- MLflow created/logged experiment:
  - `transit-forecasting-phase-b-autoregressive`
- Phase B champion selected by the configured rule:
  - model: `sarimax`
  - mode: `raw`
  - exogenous family: `service`
  - parameters: `order=(3,1,1)`, `seasonal_order=(1,0,0,12)`, `trend=n`
  - overall MAE: `450,468`
  - overall RMSE: `1,070,158`
  - overall R2: `0.679`
  - selection score: `605,391`
- Combined Phase A and Phase B artifacts:
  - `experiments_output/combined_phase_ab_v1/results`
  - `dashboard_artifacts/aws_streamlined/combined_phase_ab_v1`
- Built the combined DuckDB experiment mart:
  - `experiments_output/combined_phase_ab_v1/experiments.duckdb`
- Exported combined dashboard-compatible artifacts:
  - `dashboard_artifacts/aws_streamlined/combined_phase_ab_v1_from_duckdb`
- Updated local dashboard `latest` artifacts to the combined Phase A+B export:
  - `dashboard_artifacts/aws_streamlined/latest`
- Combined export summary:
  - `418,680` prediction rows
  - `11,630` metric rows
  - `2,326` leaderboard rows
  - model families present: baseline, linear, tree, autoregressive

## Complexity and Representation Metadata Preparation

- Added model-aware feature-policy support for the next Phase A rerun:
  - `none`
  - `corr_pruned`
  - `variance_pruned`
  - `mutual_info_top_20` / `mutual_info_top_30`
  - `lasso_selected`
  - `tree_top_20` / `tree_top_30`
- Added `complexity_profile.parquet` to experiment results, DuckDB mart loads,
  and dashboard exports.
- Added complexity fields for model comparison:
  - selected/input feature counts
  - feature reduction ratio
  - model size proxy
  - complexity score
  - interpretability score
  - compute score
- Added forward-compatible representation/runtime fields for future neural-net
  and GPU-backed experiments:
  - `representation_policy`
  - `representation_params_json`
  - `n_representation_features`
  - `sequence_length`
  - `sequence_stride`
  - `prediction_head`
  - `training_window_months`
  - `validation_strategy`
  - `early_stopping_used`
  - `epochs_trained`
  - `best_epoch`
  - `framework`
  - `framework_version`
  - `hardware_type`
  - `device`
  - `gpu_name`
  - `cuda_version`
- Updated the dashboard definition copy with explanations for feature policies,
  representation policies, and complexity scores.
- Updated `docs/experiment_metadata_contract.md` for the expanded artifact
  contract.
- Added `experiment_configs/large_phase_a_v2_complexity.yaml` for the next
  large Phase A rerun candidate.
- Validated only the small smoke config, not the full large rerun:
  - config: `experiment_configs/phase_a_policy_smoke.yaml`
  - output: `/private/tmp/transit_phase_a_policy_smoke`
  - `109` model configurations
  - `654` prediction/model-as-of rows
  - `109` complexity profile rows
  - dashboard export includes representation and complexity columns

## Phase A/B v2 Complexity Reruns

- Ran the full Phase A v2 complexity experiment:
  - config: `experiment_configs/large_phase_a_v2_complexity.yaml`
  - output: `experiments_output/large_phase_a_v2_complexity/results`
  - dashboard export: `dashboard_artifacts/aws_streamlined/large_phase_a_v2_complexity_from_duckdb`
  - `7,141` model configurations
  - `1,285,380` prediction/model-as-of rows
  - no failed checkpoint files
- Added and ran the Phase B v2 autoregressive complexity experiment:
  - config: `experiment_configs/phase_b_autoregressive_v2_complexity.yaml`
  - output: `experiments_output/phase_b_autoregressive_v2_complexity/results`
  - dashboard export: `dashboard_artifacts/aws_streamlined/phase_b_autoregressive_v2_complexity_from_duckdb`
  - `99` autoregressive model configurations
  - `17,820` prediction/model-as-of rows
- Updated `build_experiment_mart.py` so an empty autoregressive
  `feature_importance.parquet` is loaded as an empty typed table instead of
  failing DuckDB ingestion.
- Combined Phase A v2 and Phase B v2 into one dashboard bundle:
  - combined results: `experiments_output/combined_phase_ab_v2_complexity/results`
  - combined DuckDB mart: `experiments_output/combined_phase_ab_v2_complexity/experiments.duckdb`
  - combined dashboard export: `dashboard_artifacts/aws_streamlined/combined_phase_ab_v2_complexity_from_duckdb`
  - local dashboard `latest` refreshed from the combined export
  - `7,240` leaderboard/complexity rows
  - `1,303,200` prediction/model-as-of rows
  - model families present: baseline, linear, tree, autoregressive

## Public Dashboard Deployment Prep

- Added `build_public_dashboard_bundle.py` to create a smaller static dashboard
  artifact bundle for Streamlit Community Cloud and similar lightweight hosts.
- The public bundle defaults to keeping configurations that are in the best
  5 percent for at least one core performance metric, plus baseline and
  champion configurations.
- Generated `dashboard/public_artifacts/latest` from the full combined Phase
  A/B v2 dashboard export:
  - full dashboard export: `83M`
  - public dashboard bundle: `23M`
  - retained model configurations: `1,902` of `7,240`
  - retained forecast rows: `342,360`
  - retained performance rows: `342,360`
- Updated the dashboard default artifact path to use
  `dashboard/public_artifacts/latest`, while preserving
  `DASHBOARD_ARTIFACT_DIR` and `FEATURE_FAMILIES_PATH` overrides for local full
  artifact exploration.
- Added `requirements-dashboard.txt` and `dashboard/requirements.txt` with the
  minimal dashboard runtime dependencies.
- Added LinkedIn and GitHub links to the top dashboard banner.
