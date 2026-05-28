# Medium Experiment V1

Date run: 2026-05-27

Purpose: create a representative local experiment artifact for shaping the
dashboard and validating the metadata contract before the larger local research
sweep.

Note: this run predates the seasonal-naive baseline cleanup added later on
2026-05-27. It is still useful as a dashboard-shaping artifact, but the next
medium run should be treated as `medium_v2` so the seasonal naive baseline is
emitted once instead of being crossed with every feature family and feature
policy.

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
naive
ridge
lasso
xgboost
```

Both raw/direct and residual modes are included.

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
  --results-base-uri experiments_output/medium_v1/results \
  --dashboard-base-uri dashboard_artifacts/aws_streamlined/medium_v1 \
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
predictions: 18,240 rows
model_runs: 18,240 rows
metrics: 760 rows
leaderboard: 152 rows
model result size: 19 MB
dashboard artifact size: 1.7 MB
```

Dashboard artifacts were also copied to:

```text
dashboard_artifacts/aws_streamlined/latest
```

## Initial Champion

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

The raw best score was slightly lower for
`history_regime_time_linear_interactions`, but the champion rule selected the
simpler `history_regime_time` configuration because it was within the 2 percent
equivalence band.
