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
dashboard.

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

The current artifacts come from a medium-sized local experiment. They are useful
for validating the dashboard and the metadata contract before the larger
experiment run. The broader run will expand the model families, feature policies,
and historical comparisons while keeping the same artifact shape.

In other words, this version is already functional, but it is still the scaffold
for the larger experiment. The next stage is to harden the local experiment
store, run a broader model grid, and let the dashboard evolve from a prototype
into a polished project narrative.
"""


SYSTEM_OVERVIEW = """
### Architecture

The system is split into a production-shaped AWS path and a larger local
research path. AWS handles the repeatable pipeline demonstration: containerized
Python jobs run through ECS, Step Functions controls ordering and parallel
normalization, S3 stores versioned artifacts, and CloudWatch captures operational
logs.

### Reasoning

The goal is not to make AWS do every expensive experiment. The AWS pipeline
shows that the workflow can be orchestrated and monitored in a realistic cloud
environment, while local runs handle broader model sweeps where cost and
iteration speed matter more.

### Alternatives In Context

Future versions can compare ECS jobs with Batch, Lambda for lighter ingestion,
managed training jobs, or a fuller MLflow-backed registry. The current design is
kept intentionally legible for a portfolio reviewer.
"""


DATA_OVERVIEW = """
### Primary EDA

The core series is monthly transit ridership and service context. The data has
seasonality, trend, and a visible COVID-era structural break, which makes it a
useful forecasting case.

### Secondary EDA

External context includes gas prices, CPI/inflation, and King County median
household income. These features are not assumed to be magic predictors; they
create testable hypotheses about economic pressure and changing travel behavior.

### Calculated Features

The feature table includes lags, rolling summaries, regime flags, exogenous
signals, income pressure indicators, and targeted interaction terms.

### Time-Based Features

Time features represent month-of-year seasonality, long-run trend, target-month
context, and COVID/recovery regimes. They support both direct models and
residual models built around a seasonal naive baseline.
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

### Comparison Dimensions

The experiment compares several layers of modeling decisions:

- `model_family`: broad modeling group such as baseline, linear, tree, and later
  autoregressive or neural-net approaches.
- `model_build`: specific implementation such as seasonal naive, Ridge, Lasso,
  XGBoost, or future ARIMA/LSTM-style variants.
- `mode`: raw/direct forecasts versus residual forecasts built around a
  seasonal naive baseline.
- `feature_family_name`: different subsets of engineered features, from compact
  history-only sets to wider exogenous and interaction sets.
- `feature_policy`: model-aware feature treatment such as no policy or
  correlation pruning for linear models.

This structure lets the project ask more than "which model won?" It can compare
whether simpler models are close enough, whether exogenous features help, whether
interactions are worth the complexity, and whether residual modeling behaves
better through structural change.

### Leaderboard

The main leaderboard ranks configurations by a selection score:

`selection_score = 0.75 * MAE + 0.25 * RMSE`

MAE keeps the score focused on typical absolute error. RMSE adds pressure
against occasional large misses. The champion rule then applies a 2 percent
equivalence band and prefers the simpler model when performance is close enough.

That rule is intentionally opinionated. The project is not only chasing the
lowest possible error; it is also showing the tradeoff between performance,
stability, and parsimony. If a simpler feature set performs nearly as well as a
larger one, that is an important result.

### Period Metrics

Overall metrics can hide when a model performs well. The experiment therefore
also calculates performance by target-month period:

- `pre_covid`: ordinary pre-disruption behavior
- `covid_shock`: the initial structural break
- `recovery`: early recovery and adjustment
- `recent`: the newer operating regime

The derived ratios compare each disruption/recovery period against pre-COVID
error. A model with a low overall score but a high shock penalty may be accurate
on average while still being fragile at the moment when robustness matters most.

### Primary Insights

The current medium run is a dashboard-shaping artifact, not the final research
sweep. It already supports comparisons across raw vs residual modeling, linear
vs tree models, feature-family breadth, and period-specific shock/recovery
metrics.

Early results are useful for testing the shape of the system: the dashboard,
metadata contract, feature-policy logic, and artifact format. They should be
read as a working experimental slice, not the final claim about the best model.

### Broader Experiment Scope

The larger local run will expand model breadth and depth while preserving the
same output contract. Planned directions include broader regularized linear
specifications, tree ensembles, XGBoost grids, autoregressive variants, and
neural-net time-series models.

The key requirement for the broader run is checkpointed, queryable experiment
storage. Once results are large enough, DuckDB or a similar local analytical
store should sit between raw experiment artifacts and the dashboard so the UI can
compare many runs without reading every Parquet file directly.

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

`shock_penalty = covid_shock_mae / pre_covid_mae`.
This asks how much worse the model became during the abrupt COVID disruption
compared with its normal pre-COVID error, emphasizing shock sensitivity.

`recovery_ratio = recovery_mae / pre_covid_mae`.
This asks whether the model regained its earlier accuracy during the first
recovery period, emphasizing adaptation after the structural break.

`recent_recovery_ratio = recent_mae / pre_covid_mae`.
This asks whether the model's recent error has returned near its original
baseline, emphasizing long-run stabilization rather than only crisis response.

Lower values are better for MAE, RMSE, selection score, and the ratio metrics.
Higher values are better for R2 and directional accuracy.
"""


PERIOD_METRIC_SHORT_EXPLANATION = """
The leaderboard uses overall performance for the main rank, but it also includes
period-specific MAE columns. These are useful for finding models that were not
just accurate overall, but also resilient through disruption.

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
