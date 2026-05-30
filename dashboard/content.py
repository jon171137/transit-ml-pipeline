PROJECT_OVERVIEW = """
### What This Project Is

This project is a forecasting lab built around monthly transit ridership and
service data. The central question is practical: if a forecasting system had
been running through a major disruption like COVID, which modeling choices would
have held up, which would have failed, and how would we know?

Transit ridership is a useful case study because it combines several forecasting
problems in one place. The series has strong seasonality, long-run trend,
operational context, economic context, and a large structural break. That makes
it a compact but realistic setting for studying model behavior under changing
conditions rather than only optimizing a static train/test split.

The project is designed as both a modeling study and a systems-design artifact.
It asks not just which model performs best, but how the data would be refreshed,
how features would be regenerated, how experiments would be tracked, and how a
reviewer could inspect performance over time.

### What The System Does

The pipeline ingests and normalizes several data sources, builds a monthly
integrated feature table, creates feature families, trains forecasting models
across repeated historical as-of dates, and publishes curated artifacts for this
dashboard. The current dashboard bundle combines Phase A tabular models with
Phase B autoregressive models under the same rolling H3 UPT evaluation contract.

The AWS version demonstrates a production-shaped workflow: containerized Python
jobs, ECS tasks, Step Functions orchestration, S3 artifact storage, CloudWatch
logs, and dashboard-ready outputs. The local version is designed for broader
experimentation where running many model and feature combinations would be
cheaper and easier on a personal machine.

That split is intentional. AWS proves the workflow can be orchestrated and
monitored in a cloud environment. Local runs provide the freedom to scale up the
research grid without treating every exploratory model as a cloud bill.

### What The Experiment Simulates

Each as-of month acts like a historical checkpoint. The model can train only on
data available before that month, then it forecasts three months ahead. Repeating
that process over time creates a synthetic deployment history: not just one
train/test split, but a rolling view of how each strategy would have performed
as the world changed.

The dashboard compares model families, feature families, feature policies, and
raw-vs-residual approaches across that rolling history. The period-specific
metrics separate ordinary pre-COVID behavior from the shock, recovery, and more
recent operating regime.

### What The Dashboard Is For

This dashboard is meant to make the project inspectable. It shows model rankings,
forecast paths, period-specific performance, feature-family comparisons, and the
operational footprint of the pipeline. The final version should let a technical
reviewer quickly understand the system design, the modeling strategy, and the
tradeoffs between performance, stability, and simplicity.

The intended reading path is: understand the project goal, inspect how the
pipeline is structured, review the data and feature engineering choices, then
use the results explorer to compare what different modeling strategies did over
time.

### Current Scope

The current artifacts come from a larger local A/B experiment. Phase A covers
baseline, regularized linear, bagging-style tree, randomized tree, and boosted
tree models across income-aware feature families and model-specific feature
policies. Phase B adds ARIMA, SARIMA, and SARIMAX configurations using compact
service, economic, income-pressure, and service-economic exogenous sets.

The dashboard now reads a combined DuckDB-derived export that includes baseline,
linear, tree, and autoregressive model families. The next major modeling stage
is Phase C: neural-net and sequence-style models, likely run on a GPU-capable
machine while preserving the same artifact contract.
"""


SYSTEM_OVERVIEW = """
### Architecture

The system is split into a production-shaped AWS path and a larger local
research path. AWS handles the repeatable pipeline demonstration: containerized
Python jobs run through ECS, Step Functions controls ordering and parallel
normalization, S3 stores versioned artifacts, and CloudWatch captures operational
logs.

The local path handles broader experiments. The runners write portable
Parquet/JSON artifacts, MLflow receives experiment tracking metadata, DuckDB
builds a queryable analytical mart, and the dashboard reads a curated export
rather than querying raw training outputs directly.

### Reasoning

The goal is not to make AWS do every expensive experiment. The AWS pipeline
shows that the workflow can be orchestrated and monitored in a realistic cloud
environment, while local runs handle broader model sweeps where cost and
iteration speed matter more.

### Alternatives In Context

Future versions can compare ECS jobs with Batch, Lambda for lighter ingestion,
managed training jobs, or a fuller MLflow-backed registry. The current design is
kept intentionally legible for a portfolio reviewer: the cloud side proves
orchestration, while the local side proves experiment breadth and artifact
discipline.
"""


DATA_OVERVIEW = """
### Primary EDA

The core series is monthly transit ridership and service context. The target in
the current experiment is H3 UPT, forecast three months ahead from each rolling
as-of month. The data has seasonality, trend, and a visible COVID-era structural
break, which makes it a useful forecasting case.

### Secondary EDA

External context includes gas prices, CPI/inflation, and King County median
household income. Income is annual FRED data converted into prior-year monthly
context, so each forecast month uses information that would have been known
without borrowing from the future. These features are not assumed to be magic
predictors; they create testable hypotheses about economic pressure and changing
travel behavior.

### Calculated Features

The feature table includes lags, rolling summaries, regime flags, exogenous
signals, income pressure indicators, and targeted interaction terms. The current
stable feature snapshot contains income-aware families such as
`history_regime_income`, `history_regime_income_pressure`, and
`history_regime_income_linear_interactions`.

### Time-Based Features

Time features represent month-of-year seasonality, long-run trend, target-month
context, and COVID/recovery regimes. They support direct raw forecasts, residual
models built around a seasonal naive baseline, and time-series models that
receive compact exogenous sets.
"""


EXPERIMENT_OVERVIEW = """
### Forecasting Setup

The experiment is built around a rolling historical simulation. Each as-of month
is treated like a point where a forecasting system could have been deployed. At
that point, the model can train only on data before the as-of date, then it
forecasts three months ahead.

This produces a synthetic deployment history rather than a single train/test
split. The result is a month-by-month record of what each model family, feature
family, and feature policy would have predicted as the ridership system moved
through ordinary seasonal variation, COVID disruption, recovery, and the more
recent operating period.

### Current Experiment Blocks

The current dashboard bundle combines two completed local experiment blocks:

- Phase A: baseline, linear, and tree-based models over the monthly feature
  table. This includes seasonal naive, Ridge, Lasso, Elastic Net, random forest,
  Extra Trees, and XGBoost.
- Phase B: autoregressive models using the same rolling as-of evaluation frame.
  This includes ARIMA, SARIMA, and SARIMAX, with SARIMAX using compact service,
  economic, income-pressure, and service-economic exogenous sets.

Both phases are evaluated from the same as-of timeline and exported through the
same artifact contract, so their forecasts can be compared in one dashboard
without special-case logic.

### Comparison Dimensions

The experiment compares several layers of modeling decisions:

- `model_family`: broad modeling group such as baseline, linear, tree, and
  autoregressive. Neural-net models are reserved for a later Phase C.
- `model_build`: specific implementation such as seasonal naive, Ridge, Lasso,
  XGBoost, ARIMA, SARIMA, or SARIMAX.
- `mode`: raw/direct forecasts versus residual forecasts built around a
  seasonal naive baseline.
- `feature_family_name`: different subsets of engineered features, from compact
  history-only sets to wider exogenous and interaction sets.
- `feature_policy`: model-aware feature treatment such as no policy,
  correlation pruning, variance pruning, mutual-information selection,
  Lasso-based selection, or tree-importance selection.

This structure lets the project ask more than "which model won?" It can compare
whether simpler models are close enough, whether exogenous features help, whether
interactions are worth the complexity, and whether residual modeling behaves
better through structural change.

### Leaderboard

The dashboard now keeps several weighted error scores instead of treating one
score as the only answer:

- `typical_error_score = 0.90 * MAE + 0.10 * RMSE`
- `balanced_score = 0.75 * MAE + 0.25 * RMSE`
- `large_error_score = 0.50 * MAE + 0.50 * RMSE`

MAE keeps the score focused on typical absolute error. RMSE adds pressure
against occasional large misses. The typical-error score is useful when the
main question is ordinary month-to-month accuracy. The large-error score is
useful when avoiding occasional bad misses matters more. The balanced score
preserves the original project default between those two priorities.

The current champion is still selected with the balanced score plus a 2 percent
equivalence band, then the simpler model is preferred when performance is close
enough. The point is not only to chase the lowest possible error; it is also to
show the tradeoff between performance, stability, and parsimony. If a simpler
feature set performs nearly as well as a larger one, that is an important
result.

### Period Metrics

Overall metrics can hide when a model performs well. The experiment therefore
also calculates performance by target-month period:

- `pre_covid`: ordinary pre-disruption behavior
- `covid_shock`: the initial structural break
- `recovery`: early recovery and adjustment
- `recent`: the newer operating regime

The same score recipes are also calculated inside each period. That makes it
possible to ask whether a model is best under ordinary conditions, best during
the COVID shock, best during recovery, or best in the recent operating regime.
The derived ratios compare each disruption/recovery period against pre-COVID
error. A model with a low overall score but a high shock penalty may be accurate
on average while still being fragile at the moment when robustness matters most.

### Primary Insights

The current run is large enough to support real comparison across model family,
model build, feature family, feature policy, raw/residual mode, and
period-specific shock/recovery behavior. It is still not the final research
universe: neural-net models, deeper local sweeps, and any future target variants
can be added later without changing the dashboard contract.

The early pattern is already useful. Strong tree models lead the combined
leaderboard under the current selection rule, while SARIMAX provides a useful
autoregressive comparison point with explicit time-series structure and AIC/BIC
diagnostics stored in the raw run artifacts and complexity profile.

### Broader Experiment Scope

The larger local run will continue expanding breadth and depth while preserving
the same output contract. Planned directions include revisiting Phase A feature
policies if needed, adding Phase C neural-net time-series models, and moving GPU
work to a desktop/Linux environment for sequence models.

The key requirement for the broader run is checkpointed, queryable experiment
storage. DuckDB now sits between raw experiment artifacts and the dashboard,
letting the project preserve detailed Parquet outputs while publishing a curated
static bundle for Streamlit.

### What To Look For

The most interesting results are not necessarily the top row of the leaderboard.
Useful questions include:

- Does residual modeling outperform direct raw forecasting, and when?
- Do exogenous features help enough to justify their complexity?
- Do interaction features help linear models handle the disruption?
- Which models recover fastest after the shock period?
- Are the best recent models also good historically, or are they specialized to
  the newer regime?
- When two models are close, is the simpler one easier to explain and operate?
"""


PERIOD_METRIC_EXPLANATION = """
Metrics are calculated from target months, not training months. Each row asks:
if the model was trained at each historical as-of date, how accurate was its
3-month-ahead forecast for the target month?

`pre_covid` covers target months through February 2020.
`covid_shock` covers March 2020 through June 2021.
`recovery` covers July 2021 through December 2022.
`recent` covers January 2023 onward.

The dashboard provides three weighted score families:

`typical_error_score = 0.90 * MAE + 0.10 * RMSE`.
This emphasizes ordinary month-to-month absolute error, so it is useful when
typical accuracy matters more than rare misses.

`balanced_score = 0.75 * MAE + 0.25 * RMSE`.
This is the original project scoring logic and balances typical accuracy with a
moderate penalty for larger mistakes.

`large_error_score = 0.50 * MAE + 0.50 * RMSE`.
This gives RMSE enough weight to surface models that avoid large misses, even if
their ordinary absolute error is not the lowest.

Each score is also computed within the pre-COVID, shock, recovery, and recent
periods. Period-specific score ratios use the same structure as the MAE and RMSE
ratios: period score divided by the pre-COVID score.

`shock_penalty = covid_shock_mae / pre_covid_mae`.
This asks how much worse the model became during the abrupt COVID disruption
compared with its normal pre-COVID error, emphasizing shock sensitivity.

`recovery_ratio = recovery_mae / pre_covid_mae`.
This asks whether the model regained its earlier accuracy during the first
recovery period, emphasizing adaptation after the structural break.

`recent_recovery_ratio = recent_mae / pre_covid_mae`.
This asks whether the model's recent error has returned near its original
baseline, emphasizing long-run stabilization rather than only crisis response.

Lower values are better for MAE, RMSE, weighted error scores, and the ratio metrics.
Higher values are better for R2 and directional accuracy.

Adjusted R2 is tracked where it is statistically meaningful, currently as a
linear-model diagnostic. It penalizes added predictors, so it helps distinguish
genuine explanatory improvement from feature count inflation. It is not used as
the champion selection metric because the project is primarily judging forecast
accuracy.
"""


PERIOD_METRIC_SHORT_EXPLANATION = """
The leaderboard includes overall and period-specific MAE, RMSE, and weighted
error scores. These are useful for finding models that were not just accurate
overall, but also resilient through disruption.

`typical_error_score = 0.90 * MAE + 0.10 * RMSE`.
This favors ordinary month-to-month accuracy.

`balanced_score = 0.75 * MAE + 0.25 * RMSE`.
This preserves the original project default and lightly penalizes large misses.

`large_error_score = 0.50 * MAE + 0.50 * RMSE`.
This emphasizes consistency by making large misses more expensive.

`shock_penalty = covid_shock_mae / pre_covid_mae`.
This emphasizes how fragile or resilient a model was when ridership behavior
abruptly changed.

`recovery_ratio = recovery_mae / pre_covid_mae`.
This emphasizes whether the model adapted as the system moved out of the
initial shock period.

`recent_recovery_ratio = recent_mae / pre_covid_mae`.
This emphasizes whether the model has stabilized in the newer operating regime.

Values near 1.0 mean the model's error was similar to its pre-COVID error.
Values above 1.0 mean error increased relative to pre-COVID.
"""


REPRESENTATION_AND_COMPLEXITY_EXPLANATION = """
`feature_family_name` describes the eligible columns before any model-specific
selection step. This is the human-facing feature strategy: history-only,
history plus regime flags, exogenous context, income pressure, targeted
interactions, and so on.

`feature_policy` describes a training-window-safe selection or pruning step
applied after the feature family is chosen. Examples include correlation
pruning, variance pruning, mutual-information selection, Lasso-based selection,
and tree-importance selection. These policies are refit at each as-of month so
they do not use future information.

`representation_policy` is reserved for transforms that change the model input
representation rather than merely selecting columns. Examples include
`tabular_raw`, `pca_20`, `pca_95`, `sequence_raw`, `sequence_pca_20`, or a
future learned latent representation. This is especially important for neural
net and RNN-style experiments, where sequence length and compressed feature
spaces become part of the model design.

`complexity_score` is a normalized comparison score using selected feature
count, a model-size proxy, and training time. It is useful for comparing models
inside the same experiment, but it is not an absolute measure of complexity.

`interpretability_score` is a heuristic readability score. Simpler baselines
and linear models start higher; high feature counts, ensemble size, compressed
representations, and neural architectures reduce the score because they are
harder to explain cleanly.

`compute_score` summarizes relative training burden from measured runtime. It
helps separate models that are only slightly more accurate from models that are
meaningfully more expensive to retrain.

For future neural-net runs, `sequence_length`, `prediction_head`,
`training_window_months`, `early_stopping_used`, `epochs_trained`, `best_epoch`,
`framework`, `hardware_type`, `device`, `gpu_name`, and `cuda_version` make the
same artifact shape work across CPU tabular models and GPU sequence models.
"""
