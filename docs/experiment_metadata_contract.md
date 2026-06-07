# Experiment Metadata Contract

This document defines the durable model experiment outputs used by the broader
historical forecasting lab and the public dashboard. The goal is to keep model
training, experiment tracking, SQL analysis, and Streamlit presentation aligned.

## Design Principles

- Store raw experiment facts in portable Parquet/JSON artifacts.
- Treat rankings as contextual to an evaluation scope, not permanent model facts.
- Use JSON fields for model-specific details that do not fit every model family.
- Keep dashboard exports derived from the raw artifacts so the app stays fast.
- Use MLflow as an experiment tracker, while Parquet/DuckDB remain the dashboard
  and analysis source of truth.

## Identity Fields

These fields should appear across artifacts whenever relevant.

| Field | Meaning |
|---|---|
| `experiment_id` | Logical experiment/run collection, usually the pipeline run ID or local sweep ID |
| `pipeline_run_id` | Upstream pipeline run that produced the feature table |
| `model_config_id` | Stable ID for model build + mode + feature family + hyperparameters |
| `model_run_id` | Stable ID for one fitted/evaluated model instance, usually config + as-of date |
| `feature_set_id` | Stable ID for a named feature family and exact feature list |
| `as_of_date` | Month from which the forecast is cast |
| `target_date` | Month being forecasted |
| `target` | Forecast target, e.g. `upt` |
| `horizon` | Forecast horizon in months |

## Core Artifacts

### `experiment_manifest.json`

One JSON document per experiment.

Required fields:

```text
experiment_id
created_at_utc
pipeline_run_id
feature_artifacts
results_base_uri
dashboard_base_uri
target
horizon
as_of_start
models
modes
feature_family_count
prediction_count
model_run_count
metric_count
champion_config_id
selection_rule
runtime
```

`runtime` should include:

```text
compute_context
image_uri
code_version
step_function_execution_arn
mlflow_tracking_uri
mlflow_experiment_name
```

### `model_runs.parquet`

One row per model configuration evaluated at one as-of date.

Required fields:

```text
experiment_id
pipeline_run_id
model_run_id
model_config_id
as_of_date
target
horizon
model_family
model_build
model_type
mode
feature_family_name
feature_policy
feature_set_id
regime_policy
regime_event_id
regime_observed_start_date
regime_features_available_as_of
uses_target_period_features
hyperparameters_json
n_features
n_features_before_policy
n_features_after_policy
selected_feature_names_json
dropped_feature_names_json
feature_policy_params_json
representation_policy
representation_params_json
n_representation_features
sequence_length
sequence_stride
prediction_head
training_window_months
validation_strategy
early_stopping_used
epochs_trained
best_epoch
framework
framework_version
hardware_type
device
gpu_name
cuda_version
n_train
model_refit
train_seconds
predict_seconds
status
artifact_uri
```

Notes:

- `model_family` is the broad group: `baseline`, `linear`, `tree`,
  `autoregressive`, or `neural_net`.
- `model_build` is the specific build: `seasonal_naive`, `ridge`, `lasso`,
  `xgboost`, `sarimax`, `gru`, etc.
- `hyperparameters_json` is an object serialized as JSON. Empty JSON `{}` is
  valid for models without hyperparameters.
- Interaction-expanded feature families should be represented as distinct
  `feature_family_name` values, not hidden inside the same baseline family.
- Feature selection and pruning choices should be captured in `feature_policy`
  because those transforms must be fit within each as-of training window to
  avoid look-ahead leakage.
- `selected_feature_names_json` records the post-policy feature list actually
  used by that model run. For `feature_policy = none`, it should match the
  feature family. For policies such as `corr_pruned`, it records the reduced
  training-window-safe set.
- `regime_policy` describes whether the selected feature set includes
  event-aware regime signals. The current generalized value is
  `pandemic_observed`; models without those columns use `none`.
- `regime_features_available_as_of` is true only when the as-of month is on or
  after the observed disruption start. Before that point, generalized pandemic
  features must be neutral values, never a countdown to a known future event.
- `uses_target_period_features` should be false for model inputs. Target-period
  labels such as `covid_shock`, `recovery`, and `recent` are allowed in
  evaluation artifacts, but should not be used as training features.
- `representation_policy` captures transforms that change the input
  representation rather than merely selecting columns. Examples include
  `tabular_raw`, `pca_20`, `pca_95`, `sequence_raw`, `sequence_pca_20`, or a
  future neural embedding. This keeps neural-net and dimensionality-reduction
  experiments compatible with tabular model artifacts.
- Runtime fields such as `hardware_type`, `device`, `gpu_name`, and
  `cuda_version` are optional for CPU tabular runs, but should be populated for
  GPU-backed neural-net runs so training cost and portability can be analyzed.

### `predictions.parquet`

One row per prediction.

Required fields:

```text
experiment_id
pipeline_run_id
model_run_id
model_config_id
prediction_id
as_of_date
target_date
target
horizon
model_family
model_build
model_type
mode
feature_family_name
feature_set_id
regime_policy
regime_event_id
regime_observed_start_date
regime_features_available_as_of
uses_target_period_features
actual
prediction
baseline_prediction
error
abs_error
squared_error
ape
evaluation_period
shock_period_flag
model_refit
```

Notes:

- `baseline_prediction` should hold the primary naive comparison value.
- `ape` should be nullable when the actual value is zero or missing.
- `evaluation_period` supports dashboard views such as `overall`,
  `covid_shock`, `recovery`, `recent`, or custom period labels.
- `regime_policy` is model-facing metadata. `evaluation_period` is
  analysis-facing metadata. This separation is what lets the project study
  pandemic-era performance without letting models train on target-period labels.

### `metrics.parquet`

One row per model configuration and evaluation scope.

Required fields:

```text
experiment_id
pipeline_run_id
model_config_id
evaluation_scope
evaluation_start_date
evaluation_end_date
target
horizon
model_family
model_build
model_type
mode
feature_family_name
feature_set_id
n_predictions
n_features
mae
rmse
r2
r2_adjusted
diracc
mae_naive
rmse_naive
mae_improvement_vs_naive
rmse_improvement_vs_naive
selection_score
rank
metric_extras_json
avg_train_seconds
total_train_seconds
```

Notes:

- `rank` is calculated within each `evaluation_scope`.
- `r2` is ordinary coefficient of determination. `r2_adjusted` is nullable and
  should be populated only when the adjustment is meaningful and the evaluation
  group has enough observations relative to selected predictors. In the current
  tabular runners, it is used as a linear-model diagnostic and is not part of
  champion selection.
- `metric_extras_json` can store model-family-specific values such as AIC/BIC,
  validation loss, coverage, or other diagnostics.

### `feature_sets.parquet`

One row per exact feature set used in modeling.

Required fields:

```text
experiment_id
feature_set_id
feature_family_name
mode
feature_policy
feature_count
feature_hash
feature_names_json
description
includes_lags
includes_rolling
includes_exogenous
includes_service
includes_interactions
includes_regime
regime_policy
regime_event_id
regime_observed_start_date
uses_target_period_features
regime_feature_names_json
```

Interaction feature names use an `_x_` infix, for example
`upt_lag12_x_pandemic_disruption_active`. The interaction design is
deliberately targeted around generalized, as-of-safe regime flags rather than a
full polynomial expansion, so the dashboard can compare whether explicit
observed-shock/recovery interactions help linear models without exploding the
feature space. COVID-19 is the current event instance; model-facing feature
names should use generalized `pandemic_*` or `known_disruption_*` language
rather than a hindsight-specific event name.

The current feature-table builder uses `pandemic_observed`,
`pandemic_disruption_active`, `post_pandemic_observed`, and
`months_since_pandemic_observed`. These are neutral before the disruption is
observed. Legacy COVID-specific aliases may exist for compatibility, but new
families and documentation should use the generalized names.

Income features use annual King County median household income as monthly
prior-year context. Feature names should make that time basis explicit, for
example `king_county_income_yoy_pct_prior_year`. When projected income values
are used beyond the latest observed FRED year, the normalized income table must
carry `income_reference_method` so the downstream narrative can distinguish
observed and projected socioeconomic context.

### `feature_importance.parquet`

One row per feature importance/coefficient value.

Required fields:

```text
experiment_id
pipeline_run_id
model_run_id
model_config_id
as_of_date
model_family
model_build
model_type
mode
feature_family_name
feature_set_id
feature_name
importance_type
importance
importance_abs
rank
```

Notes:

- Linear models can use `coefficient` or `standardized_coefficient`.
- Tree models can use `gain`, `split`, `weight`, or package-specific
  importance values.
- Models without meaningful feature importance may omit rows.

### `complexity_profile.parquet`

One row per model configuration with the derived complexity, interpretability,
and compute fields used by the dashboard.

Required fields:

```text
experiment_id
pipeline_run_id
model_config_id
config_id
model_family
model_build
model_type
ensemble_method
mode
feature_family_name
feature_policy
feature_set_id
hyperparameters_json
feature_policy_params_json
representation_policy
representation_params_json
n_input_features
n_selected_features
feature_reduction_ratio
n_representation_features
sequence_length
sequence_stride
prediction_head
training_window_months
validation_strategy
early_stopping_used
epochs_trained
best_epoch
model_size_proxy
complexity_score
interpretability_score
compute_score
avg_train_seconds
total_train_seconds
model_run_count
refit_count
framework
framework_version
hardware_type
device
gpu_name
cuda_version
overall_mae
overall_rmse
overall_r2
overall_r2_adjusted
overall_selection_score
```

Notes:

- `complexity_score` is a normalized within-experiment comparison based on
  selected features, model-size proxy, and measured training time. It should not
  be compared as an absolute value across unrelated experiments.
- `interpretability_score` is a heuristic readability score, not a statistical
  diagnostic. It is intended to support portfolio discussion about parsimony,
  explainability, and operational simplicity.
- AIC/BIC should be stored where the model family makes that meaningful, most
  commonly autoregressive likelihood-based models. For tree and neural-net
  models, size/runtime/validation diagnostics are more appropriate complexity
  signals.

## Dashboard Export Artifacts

The dashboard should read derived, presentation-shaped files rather than doing
heavy joins on every page load.

Recommended exports:

```text
overview_top_models.parquet
overview_prediction_paths.parquet
leaderboard.parquet
performance_over_time.parquet
champion_timeline.parquet
feature_family_summary.parquet
model_family_summary.parquet
shock_recovery_summary.parquet
feature_importance_summary.parquet
complexity_profile.parquet
```

The current streamlined AWS runner may continue writing its existing dashboard
files while we migrate toward these names.

## DuckDB Experiment Mart

The local experiment mart is a derived analytical layer built from the portable
Parquet/JSON artifacts. It is not the source of truth; it is a query-optimized
copy for local analysis and dashboard export validation.

Recommended core tables:

```text
experiment_runs
predictions
model_runs
metrics
feature_sets
feature_importance
feature_family_summary
complexity_profile
```

Recommended dashboard views:

```text
forecast_paths
performance_over_time
model_leaderboard
feature_family_summary_dashboard
champion_predictions
overview_top_models
overview_prediction_paths
complexity_profile_dashboard
```

For the portfolio deployment, S3 should remain the durable artifact store. A
DuckDB file can be produced as a read-optimized derived artifact, then used to
export the smaller Parquet/JSON files consumed by Streamlit.

## MLflow Usage

MLflow should be optional and additive.

Recommended behavior:

- Always write Parquet/JSON artifacts.
- If MLflow is enabled, create one parent run for the experiment.
- Log experiment-level params: target, horizon, as-of start, model list, feature
  count, and output URIs.
- Log experiment-level metrics: prediction count, model run count, champion MAE,
  champion RMSE, champion selection score.
- Log compact artifacts: manifest JSON, champion JSON, top leaderboard CSV, and
  feature family summary CSV.
- For larger local sweeps, optionally create nested runs per model
  configuration or model family.

Recommended environment variables:

```text
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=mlruns
MLFLOW_EXPERIMENT_NAME=transit-forecasting
MLFLOW_RUN_NAME=<optional readable run name>
```

Do not make the public dashboard depend on a live MLflow server. MLflow is the
lab notebook and registry; Parquet/DuckDB/dashboard exports are the public
analysis layer.
