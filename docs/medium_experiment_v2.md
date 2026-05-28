# Medium Experiment V2

Date run: 2026-05-27

Purpose: rerun the representative local dashboard-shaping experiment after the
seasonal-naive baseline cleanup, then validate the DuckDB mart export path.

This run supersedes `medium_v1` for dashboard iteration. `medium_v1` remains
useful historically, but it crossed the seasonal naive baseline with every
feature family and feature policy. `medium_v2` emits seasonal naive once as a
true baseline configuration.

## Inputs

```text
feature_table_uri=/private/tmp/feature_income_test/feature_table.parquet
feature_families_uri=/private/tmp/feature_income_test/feature_families.json
as_of_start=2016-01-01
as_of_end=2025-12-01
as_of_frequency_months=1
refit_frequency_months=1
horizon=3
target=upt
n_jobs=4
```

The resulting target window is 2016-04-01 through 2026-03-01.

## Model Scope

```text
seasonal_naive
ridge
lasso
xgboost
```

Raw/direct and residual modes are included for trainable models. Seasonal naive
is emitted once as:

```text
model_family=baseline
model_build=seasonal_naive
feature_family_name=baseline_naive
feature_policy=none
mode=raw
```

## Feature Families

```text
history_only
history_regime_time
history_regime_all_exog
history_regime_time_all_exog
history_regime_time_linear_interactions
history_regime_all_exog_linear_interactions
history_regime_income
history_regime_income_pressure
```

## Feature Policies

```text
none
corr_pruned
```

`corr_pruned` applies to linear models only. It is fit within each as-of
training window and records the selected columns per model run. Baseline and
tree models fall back to `none`.

## Local Command

```bash
.venv/bin/python run_aws_streamlined_models.py \
  --feature-table-uri /private/tmp/feature_income_test/feature_table.parquet \
  --feature-families-uri /private/tmp/feature_income_test/feature_families.json \
  --results-base-uri experiments_output/medium_v2/results \
  --dashboard-base-uri dashboard_artifacts/aws_streamlined/medium_v2 \
  --as-of-start 2016-01-01 \
  --as-of-end 2025-12-01 \
  --as-of-frequency-months 1 \
  --refit-frequency-months 1 \
  --include-feature-family history_only \
  --include-feature-family history_regime_time \
  --include-feature-family history_regime_all_exog \
  --include-feature-family history_regime_time_all_exog \
  --include-feature-family history_regime_time_linear_interactions \
  --include-feature-family history_regime_all_exog_linear_interactions \
  --include-feature-family history_regime_income \
  --include-feature-family history_regime_income_pressure \
  --include-model-type naive \
  --include-model-type ridge \
  --include-model-type lasso \
  --include-model-type xgboost \
  --feature-policy none \
  --feature-policy corr_pruned \
  --n-jobs 4
```

## Output Summary

```text
prepared model configuration tasks: 145
predictions: 17,400 rows
model_runs: 17,400 rows
metrics: 725 rows
leaderboard: 145 rows
seasonal_naive leaderboard rows: 1
model result folder size: 54 MB
dashboard export folder size: 1.4 MB
```

Raw model outputs were written to:

```text
experiments_output/medium_v2/results
dashboard_artifacts/aws_streamlined/medium_v2
```

The DuckDB mart and dashboard export were written to:

```text
experiments_output/medium_v2/experiments.duckdb
dashboard_artifacts/aws_streamlined/medium_v2_from_duckdb
```

The dashboard `latest` folder was updated to the DuckDB export:

```text
dashboard_artifacts/aws_streamlined/latest
```

## DuckDB Validation

`build_experiment_mart.py` loaded the result artifacts and exported
dashboard-compatible files.

```text
forecast_paths: 17,400 rows
performance_over_time: 17,400 rows
model_leaderboard: 145 rows
feature_family_summary: 17 rows
champion_predictions: 120 rows
overview_top_models: 5 rows
overview_prediction_paths: 600 rows
```

## Champion

The champion selected by the simplicity-aware rule:

```text
model_build=xgboost
mode=raw
feature_family_name=history_regime_time
feature_policy=none
mae=428,050
rmse=841,868
selection_score=531,504
```

The raw best score remained slightly lower for an interaction-expanded
configuration, but the champion rule selected the simpler non-interaction
feature family because it was within the 2 percent equivalence band.

