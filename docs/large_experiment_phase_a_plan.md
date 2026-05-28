# Large Experiment Phase A Plan

Date drafted: 2026-05-27

Run completed: 2026-05-28

Phase A is the first broad local experiment sweep. It should prove the project
can scale from a dashboard-shaping medium run into a much larger, resumable
historical forecasting experiment without changing the dashboard artifact
contract.

## Completed Run Summary

The full Phase A run completed locally with checkpoints enabled:

```text
completed model configurations: 2,227
predictions/model-as-of rows: 400,860
metric rows: 11,135
feature-importance rows: 17,684,904
failed checkpoint files: none
```

The DuckDB experiment mart was rebuilt from the run outputs and exported a
dashboard-compatible bundle:

```text
experiments_output/large_phase_a_v1/experiments.duckdb
dashboard_artifacts/aws_streamlined/large_phase_a_v1_from_duckdb
```

Local dashboard `latest` artifacts were updated to the Phase A export bundle.

The champion selected by the configured simplicity-aware rule was:

```text
model: XGBoost
mode: raw
feature family: history_regime_time
feature policy: none
parameters: n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=1
overall MAE: 340,381
overall RMSE: 670,718
overall R2: 0.874
selection score: 422,965
evaluation window: target months 2011-04-01 through 2026-03-01
```

## Scope

Phase A stays with tabular models that can use the current monthly feature
table:

- seasonal naive baseline
- Ridge
- Lasso
- ElasticNet
- Random Forest
- Extra Trees
- XGBoost

It intentionally defers ARIMA/SARIMAX and neural networks. Those model families
need different data shapes and training controls, so they should be added in
later phases instead of forcing them into the current tabular loop.

## Experiment Config

The draft config is:

```text
experiment_configs/large_phase_a_v1.yaml
```

The config defines:

- input feature artifacts
- output folders
- rolling as-of forecast window
- model builds
- hyperparameter grids
- feature families
- feature policies
- MLflow settings
- checkpoint/chunk locations

Use the planner before launching a run:

```bash
.venv/bin/python plan_large_experiment.py \
  --config experiment_configs/large_phase_a_v1.yaml
```

The planner expands the config into estimated model configuration counts and
model-run rows. It does not train models.

Latest planner output after adding the notebook-aligned XGBoost settings:

```text
as_of_count: 180
total_model_configurations: 2,227
estimated_model_run_rows: 400,860
feature families validated: 21 of 21
```

## Current Draft Window

```text
as_of_start=2011-01-01
as_of_end=2025-12-01
as_of_frequency_months=1
horizon=3
target=upt
```

This produces one forecast origin per month. Each origin trains only on rows
before the as-of date and forecasts the month three months ahead.

## Feature Families

Phase A includes all current income-aware feature families:

```text
history_only
history_recent
history_rolls
history_full
history_regime
history_regime_time
history_regime_gas
history_regime_cpi
history_regime_service
history_regime_gas_cpi
history_regime_gas_service
history_regime_cpi_service
history_regime_all_exog
history_regime_time_all_exog
history_regime_linear_interactions
history_regime_time_linear_interactions
history_regime_exog_linear_interactions
history_regime_all_exog_linear_interactions
history_regime_income
history_regime_income_pressure
history_regime_income_linear_interactions
```

The interaction families are primarily intended to help linear models test
explicit regime and economic-pressure effects. Tree models can already capture
many interactions implicitly, but including these families still lets the
dashboard compare whether the engineered interactions help or add noise.

## Feature Policies

Initial policies:

```text
baseline: none
linear: none, corr_pruned
tree: none
```

`corr_pruned` should stay training-window-safe: correlations are computed only
inside each as-of training window. Future feature-selection policies can be
added as separate branches:

- `lasso_selected`
- `pca`
- `mutual_info_top_k`

Those should remain policies, not permanent feature-table transformations,
because they must be fit separately for each as-of date to avoid leakage.

## Hyperparameter Grid Philosophy

The Phase A grid should be broad enough to compare model behavior without
turning the first large run into an unbounded search.

Current draft:

- Ridge: 4 alpha values
- Lasso: 4 alpha values
- ElasticNet: 6 alpha/l1-ratio values
- Random Forest: 3 conservative tree settings
- Extra Trees: 3 conservative tree settings
- XGBoost: 19 settings, combining conservative boosted-tree settings with the
  stronger notebook-anchor range from `transit_integrated_modeling.ipynb`
  (`n_estimators` 200/500, depth 3/4, learning rate .05/.10, min child weight
  1/3, plus a few lower-learning-rate stability settings)

This is not meant to find a globally optimal model. It is meant to create a
large, interpretable comparison surface across model class, feature family,
feature policy, mode, and shock/recovery period.

## Runner Support

`run_aws_streamlined_models.py` now supports the Phase A model builds:

```text
seasonal_naive
ridge
lasso
elastic_net
random_forest
extra_trees
xgboost
```

It also supports:

- YAML config loading through `--experiment-config`
- per-model-configuration chunk outputs
- `--resume` behavior for completed chunks
- failed-configuration logging in the chunk directory
- `ensemble_method` metadata for tree models:
  - `bagging` for Random Forest
  - `randomized_bagging` for Extra Trees
  - `boosting` for XGBoost

Remaining pre-large-run improvements:

1. Add additional metrics such as median absolute error, max absolute error,
   bias, sMAPE, and artifact size.
2. Consider whether child MLflow runs per chunk/model family are worth adding
   after the first full Phase A pass. The current runner logs a compact MLflow
   summary at the end of the run and uses Parquet checkpoints for durability.

The preferred direction is a new top-level historical runner with internal
modules, not an endless expansion of the streamlined AWS script:

```text
run_historical_experiment.py
modeling/
  registry.py
  baseline.py
  linear.py
  tree.py
  feature_policies.py
  metrics.py
  checkpoints.py
```

The streamlined AWS runner can remain useful for small cloud-validated runs.
The historical runner can be optimized for broad local sweeps.

For now, the existing runner has enough Phase A support to run a controlled
large local pass after the final grid choices are reviewed.

## Chunking And Resume

The large run should not rely on one all-or-nothing process. It should write
chunk outputs, then merge them into the existing Parquet/DuckDB/dashboard flow.

Proposed local shape:

```text
experiments_output/large_phase_a_v1/
  config.yaml
  run_manifest.json
  chunks/
    chunk_0001/
      predictions.parquet
      model_runs.parquet
      metrics.parquet
      feature_importance.parquet
    chunk_0002/
      ...
  checkpoints/
    completed_configs.parquet
    failed_configs.parquet
  results/
    predictions.parquet
    model_runs.parquet
    metrics.parquet
    feature_importance.parquet
    feature_sets.parquet
  experiments.duckdb
```

Resume behavior should skip already-completed model configuration IDs unless
the user explicitly asks to replace them.

## MLflow Role

MLflow should be used as the experiment tracker, not as the dashboard database.

Use MLflow for:

- experiment run lineage
- config snapshots
- high-level parameters
- champion metrics
- summary artifacts
- possibly one parent run per phase and child runs per chunk/model family

Do not make Streamlit depend on a live MLflow server. The dashboard should read
curated Parquet/JSON exports from the experiment mart, because that is simpler,
portable, and cheaper for a portfolio demo.

Recommended Phase A MLflow structure:

```text
experiment: transit-forecasting-large-phase-a
parent run: large_phase_a_v1
child runs:
  baseline
  linear_ridge
  linear_lasso
  linear_elastic_net
  tree_random_forest
  tree_extra_trees
  tree_xgboost
```

The final dashboard artifacts should still come from:

```text
Parquet/JSON results → DuckDB mart → dashboard export bundle
```

## Later Phases

Phase B: autoregressive models

- ARIMA
- SARIMA
- SARIMAX
- model-specific AIC/BIC
- exogenous regressors for SARIMAX
- likely a separate runner module

Phase C: neural nets

- MLP
- RNN
- GRU
- LSTM
- sequence-window datasets
- train/validation split per as-of date
- early stopping
- learning-rate scheduling
- optional GPU execution on Colab/Linux PC

Learning-rate scheduling belongs in Phase C, not Phase A, because none of the
Phase A sklearn/tree models use neural-net style optimizer schedules.

## Review Questions Before Running

- Is 2011 the right start date, or should Phase A start later because some
  feature families require longer lag history?
- Should `corr_pruned` use `0.95`, or should the threshold be a config value
  swept across a small set?
- After the first full pass, should the next pass trim weak feature/model
  combinations or expand around promising regions?

## Smoke Test

The config-driven smoke test is:

```text
experiment_configs/phase_a_smoke.yaml
```

It exercises every Phase A model build across two feature families and a short
2024 window. The smoke run completed successfully:

```text
model configurations: 37
predictions: 222 rows
metrics: 74 rows
leaderboard: 37 rows
seasonal_naive leaderboard rows: 1
```

Resume behavior was verified by rerunning the same config and reusing all 37
completed chunks. The smoke outputs also loaded through `build_experiment_mart.py`
and exported dashboard-compatible artifacts.
