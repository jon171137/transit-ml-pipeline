PROJECT_OVERVIEW_CASE_STUDY = """
### What This Project Is

- Uses transit ridership as a compact case study with seasonality, trend,
  operational context, economic context, and a major structural break.
- Studies how model behavior changes across ordinary periods, COVID shock,
  recovery, and the more recent operating regime.
- Designed as both a modeling study and a systems-design artifact.
"""


PROJECT_OVERVIEW_SYSTEM = """
### What The System Does

- Ingests and normalizes transit service, fuel price, general inflation, and income context.
- Builds a monthly integrated feature table and feature-family definitions.
- Trains models across repeated historical as-of dates.
- Publishes curated Parquet/JSON artifacts for dashboard exploration.
"""


PROJECT_OVERVIEW = """

### What The Experiment Simulates

Each as-of month acts like a historical checkpoint. The model can train only on
data available before that month, then it forecasts three months ahead.

Repeating that process over time creates a synthetic deployment history: not
just one train/test split, but a rolling view of how each strategy would have
performed.

### Current Scope

The experimentation has currently covered a range of modeling approaches.

The dashboard now reads a combined DuckDB-derived export that includes baseline,
linear, tree, and autoregressive model families under one artifact contract.
Neural-net artifacts are being handled as the next experiment block and will fit
the same export structure once rerun against the pandemic-safe feature table.
"""


SYSTEM_ARCHITECTURE = """
### Architecture

- AWS path demonstrates a production-shaped workflow with containerized Python
  jobs, ECS tasks, Step Functions orchestration, S3 artifacts, and CloudWatch
  logs.
- Local experiment runners handle the broader research sweeps where cost,
  iteration speed, and long-running model grids matter more.
- Training runners write portable Parquet/JSON artifacts under a shared
  metadata contract.
- DuckDB builds a queryable analytical mart, then exports dashboard-ready files.
- The public Streamlit bundle keeps full lightweight model metadata, curated
  compatibility path files, and partitioned full path-level rows. The app
  loads small metadata first, then reads larger path rows only for views that
  need them.
"""


SYSTEM_REASONING = """
### Reasoning

- AWS proves orchestration, repeatability, logging, and artifact handoff.
- Local/GPU runs make large model grids practical without turning cloud cost
  into the main constraint.
- DuckDB is the bridge between raw experiment outputs and presentation-shaped
  dashboard files.
- The dashboard consumes the same artifact shape regardless of whether training
  happened through AWS, local CPU runs, or future GPU-backed sequence runs.
- This split keeps the project readable as a portfolio system while still
  supporting deep experimentation.
"""


SYSTEM_OVERVIEW = """
### Alternatives In Context

- AWS Batch could replace ECS for larger queued training workloads.
- Lambda remains useful for lighter ingestion or source-specific fetch steps.
- Managed training jobs would make sense if the project moved toward scheduled
  retraining at scale.
- The current runners can optionally log compact MLflow experiment summaries;
  a fuller MLflow registry with promoted model stages would be a later
  extension, not a dependency of this dashboard.
- The current design favors clarity: cloud orchestration plus local experiment
  breadth under one artifact contract, then a static dashboard bundle for
  public inspection.
"""


SYSTEM_ARTIFACT_FLOW = """
### Dashboard Artifact Flow

The dashboard does not train models or query a live experiment server. It reads
static Parquet/JSON artifacts produced after the experiment runners finish.

The current flow is:

1. **Pipeline artifacts:** source data is normalized, joined to a monthly base,
   and converted into an as-of-safe feature table.
2. **Experiment artifacts:** model runners write predictions, model-run rows,
   metrics, feature-set metadata, and manifests under a common contract.
   They also produce MLflow-compatible metadata and can optionally log compact
   experiment summaries, but the dashboard does not depend on a live MLflow
   server.
3. **DuckDB mart:** completed experiment folders are loaded into a local DuckDB
   mart so results can be validated, combined, reshaped, and exported.
4. **Public bundle:** `build_public_dashboard_bundle.py` creates a Streamlit
   bundle under `dashboard/public_artifacts/latest`.
5. **Streamlit app:** the dashboard reads the bundle as static files. It uses
   full model metadata for filtering and loads larger forecast/performance path
   rows only after the user narrows the model selection.

This keeps the public app lightweight enough to host while preserving a much
larger result universe for inspection.
"""


DATA_PRIMARY_DATA = """
### Primary Data | Monthly transit series combining ridership with the service levels used to operate that ridership.

- **Unlinked Passenger Trip (UPT):** one person boarding a transit vehicle,
  counted every time they board, regardless of transfers.
- **Vehicle Revenue Miles (VRM):** all miles a transit vehicle travels while it
  is in revenue service and available to carry passengers.
- **Vehicle Revenue Hours (VRH):** all hours a transit vehicle operates in
  revenue service and is available to carry passengers.
- **Vehicles Operated in Maximum Service (VOMS):** often labeled "Peak
  Vehicles," the largest number of revenue vehicles actually in service
  simultaneously for a given mode on the busiest day or reporting period,
  typically during the peak demand window.

*For VRM and VRH, revenue service includes layover and recovery time, but
excludes deadhead (non-revenue) mileage such as pull-outs, pull-ins, and
training or maintenance runs.*

Source: [FTA NTD Complete Monthly Ridership](https://www.transit.dot.gov/ntd/data-product/monthly-module-adjusted-data-release).
"""


DATA_SECONDARY_DATA = """
### Secondary Data | External context includes gas prices, CPI/inflation, and King County median household income.

- **[EIA gasoline prices](https://www.eia.gov/petroleum/gasdiesel/)** contribute transportation cost context and are
  normalized monthly, then used directly plus year-over-year and change
  features.
- **FRED CPI series** ([All Items](https://fred.stlouisfed.org/series/CPIAUCSL),
  [Core](https://fred.stlouisfed.org/series/CPILFESL)) contribute general and core price-pressure context and are
  normalized monthly, imputed when needed, then used as exogenous and
  interaction features.
- **[FRED King County median household income](https://fred.stlouisfed.org/series/MHIWA53033A052NCEN)** contributes annual socioeconomic
  context and is converted to prior-year monthly context with income-growth and
  affordability-pressure features.
"""


DATA_CALCULATED_FEATURES = """
### Calculated Features

- Feature table includes lags, rolling summaries, adaptive regime flags, exogenous
  signals, income pressure indicators, and targeted interaction terms.
- Income-aware families include `history_regime_income`,
  `history_regime_income_pressure`, and
  `history_regime_income_linear_interactions`.
- Feature families define human-readable modeling strategies before
  model-specific feature policies are applied.
"""


DATA_TIME_FEATURES = """
### Time-Based Features

- Time features represent month-of-year seasonality, long-run trend,
  target-month context, and observed disruption/recovery regimes.
- They support raw/direct forecasts and residual models built around a seasonal
  naive baseline.
- Autoregressive and neural sequence models use the same rolling as-of frame,
  with compact exogenous or sequence representations where appropriate.
"""


DATA_AS_OF_REGIME_FEATURES = """
### As-Of Safe Regime Features

- In a feature family name, `regime` means the model can use deployable
  context about an observed operating regime: seasonality/trend context,
  pandemic/disruption flags once the disruption is known, and time-since-known
  disruption signals.
- It does **not** mean the model receives future-aware countdown features.
  Pre-disruption rows do not get a "months until pandemic" signal, because a
  real model in 2018 would not have known that shock was coming.
- The model-facing disruption fields use general names such as
  `pandemic_observed`, `pandemic_disruption_active`, `post_pandemic_observed`,
  and `months_since_pandemic_observed`. COVID-19 is the event instance in this
  dataset; the modeling idea is broader public-health or safety disruption
  awareness.
- Before the disruption is observable, these fields are neutral. After the
  disruption is observable, they let the experiment ask whether explicitly
  marking the changed operating environment helps forecasting during shock,
  recovery, and the later regime.
- Evaluation-period labels such as pre-COVID, shock, recovery, and recent are
  still used after the fact for scoring and interpretation. Those labels are
  analytical metadata; the model inputs are limited to information that would
  have been known at each as-of date.
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

The current dashboard bundle is centered on the pandemic-safe v3 rerun:

- Phase A v3: baseline, linear, and tree-based models over the rebuilt monthly
  feature table. This includes seasonal naive, Ridge, Lasso, Elastic Net,
  random forest, Extra Trees, and XGBoost.
- Phase B v3: autoregressive models using the same rolling as-of evaluation frame
  and the same pandemic-safe feature table.
  This includes ARIMA, SARIMA, and SARIMAX, with SARIMAX using compact service,
  economic, income-pressure, and service-economic exogenous sets.
- Phase C: GRU and LSTM neural sequence models have been explored separately.
  They are treated as follow-up artifacts until rerun and reconciled against the
  pandemic-safe feature table.

Phase A and Phase B now share the same monthly as-of timeline, pandemic-safe
feature logic, and dashboard artifact contract. Phase C is designed to fit that
same contract once the neural rerun is complete.

### Comparison Dimensions

The experiment compares several layers of modeling decisions:

- `model_family`: broad modeling group such as baseline, linear, tree,
  autoregressive, and neural-net when included.
- `model_build`: specific implementation such as seasonal naive, Ridge, Lasso,
  XGBoost, ARIMA, SARIMA, SARIMAX, GRU, or LSTM.
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

### Model Family Review

#### Seasonal Baseline

**Basics:** A seasonal-naive model predicts from the matching month one year
earlier. Monthly transit ridership has pronounced seasonality, so this is a
meaningful minimum bar rather than a ceremonial baseline.

**Variations:** The experiment keeps one canonical `seasonal_naive`
configuration. It is reused as the reference forecast for error comparisons and
for models trained in residual mode.

**Parameter Search:** No hyperparameter grid is needed. The forecast uses a
12-month seasonal lag and the shared 3-month forecast horizon.

**Observations:** The baseline captures ordinary seasonal shape but struggles
when the system shifts sharply. Its overall MAE is about 753,000 riders. That
makes it useful for interpreting whether more complex models are earning their
additional operational and explanatory cost.

#### Regularized Linear Models

**Basics:** Linear models are a strong fit for testing whether lagged ridership,
service, time, and economic context have stable additive relationships with the
target. Regularization matters because wider feature families contain correlated
lags, rolling summaries, and interaction terms.

**Variations:** Phase A v3 evaluates Ridge, Lasso, and Elastic Net in raw and
residual modes. It also varies input treatment across no pruning,
variance pruning, correlation pruning, mutual-information selection, and
Lasso-based selection where applicable.

**Parameter Search:** Ridge searches `alpha` values from `0.1` to `100`. Lasso
searches `alpha` values from `1` to `1000`. Elastic Net searches `alpha` values
from `0.1` to `10` and `l1_ratio` values of `0.25` and `0.50`.

**Observations:** Elastic Net leads the linear family with an overall MAE near
422,000 and RMSE near 888,000. Ridge is close behind. The result is useful:
regularized linear models remain competitive and comparatively legible even
after the experiment expands into larger ensemble grids.

#### Autoregressive Models

**Basics:** Autoregressive models explicitly represent time-series structure.
They are natural comparators for monthly forecasting because they model lagged
dependence and, for seasonal variants, recurring annual patterns directly.

**Variations:** Phase B v3 evaluates ARIMA, SARIMA, and SARIMAX. SARIMA adds
12-month seasonal terms. SARIMAX adds compact exogenous sets for service,
economic lagged context, income pressure, and combined service-economic
signals.

**Parameter Search:** ARIMA evaluates 12 `(p, d, q)` orders with trend options
`n` and `c`. SARIMA evaluates 6 nonseasonal orders against 4 seasonal orders
with period `12`. SARIMAX evaluates 5 nonseasonal orders, 3 seasonal orders with
period `12`, and 4 compact exogenous sets.

**Observations:** SARIMAX is the strongest autoregressive build with overall MAE
near 450,000 and RMSE near 1.07 million. It does not lead the full leaderboard,
but it adds an important structural comparison and produces AIC/BIC diagnostics
that help discuss fit-versus-complexity tradeoffs.

#### Tree Ensembles

**Basics:** Tree ensembles can learn nonlinear effects and interactions without
requiring each relationship to be manually specified. That is useful in a
ridership series where service, seasonality, economic pressure, and regime
changes may not combine additively.

**Variations:** Phase A v3 evaluates random forest, Extra Trees, and XGBoost across
raw and residual modes, engineered feature families, and tree-appropriate
policies including variance pruning, mutual-information selection, and
tree-importance top-30 selection.

**Parameter Search:** Random forest and Extra Trees vary tree count from `300`
to `500`, depth from `4` or `6` through unrestricted growth, minimum leaf size
from `3` to `5`, and feature sampling between `sqrt` and `0.7`. XGBoost explores
tree counts from `100` to `500`, depths from `2` to `4`, learning rates from
`0.02` to `0.10`, subsampling from `0.8` to `0.9`, column sampling from `0.7`
to `0.8`, and minimum child weights from `1` to `5`.

**Observations:** XGBoost produces the strongest overall results in the current
bundle. Leading tree configurations land around 300,000 to 305,000 MAE, 525,000
to 541,000 RMSE, and R-squared near `0.92`, depending on the ranking recipe and
parsimony tie-break. Random forest and Extra Trees also perform well, showing
that nonlinear ensemble structure contributes beyond a single boosted
implementation.

#### Neural Sequence Models

**Basics:** Recurrent neural networks are designed to learn patterns from
ordered sequences. For this project, the interesting question is whether
sequence models can recover useful temporal representations across a major
structural break, not whether additional network capacity automatically wins.

**Variations:** Earlier Phase C screening compared MLP, simple RNN, GRU, and
LSTM models. Capacity experiments then focused the neural work on GRU and LSTM.
Those runs compared raw and residual modes, compact and wider feature families,
dynamic feature policies, raw sequence representations, and selected PCA
sequence representations. The next neural pass should rerun the finalists
against the pandemic-safe feature table so they line up cleanly with Phase A v3
and Phase B v3.

**Parameter Search:** The latest neural finalist plan uses sequence lengths of
`36` months, batch size `24`, up to `300` epochs, early-stopping patience `25`,
and learning-rate reduction patience `8`. GRU variants compare recurrent layer
sizes such as `[512, 100]` and `[256, 100]`; LSTM variants include a larger
`[1000, 100]` style architecture inspired by earlier notebook experiments.
Dense heads use `[200, 10]`, with dropout, weight decay, and learning rate
varied through the finalist definitions.

**Observations:** The neural work so far is best read as exploratory. It showed
that larger sequence models can become useful once capacity and training
patience are increased, but those artifacts should be rerun against the same
pandemic-safe feature table before being treated as directly comparable to the
current A/B dashboard bundle.

### Leaderboard

The dashboard keeps one weighted selection score alongside the direct error
metrics:

`balanced_score = 0.75 * MAE + 0.25 * RMSE`

MAE keeps the score focused on typical absolute error. RMSE adds pressure
against occasional large misses. The balanced score is the project default
because it preserves that tradeoff without turning the leaderboard into a wall
of nearly redundant metrics.

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

MAE and RMSE are calculated inside each period. That makes it possible to ask
whether a model is best under ordinary conditions, best during the COVID shock,
best during recovery, or best in the recent operating regime. The derived ratios
compare each disruption/recovery period against pre-COVID error. A model with a
low overall score but a high shock penalty may be accurate on average while
still being fragile at the moment when robustness matters most.

### Primary Insights

The current v3 A/B run is large enough to support real comparison across model
family, model build, feature family, feature policy, raw/residual mode, and
period-specific shock/recovery behavior. Additional local refinements, neural
reruns, and any future target variants can be added without changing the
dashboard contract.

The current pattern is already useful. Strong boosted-tree models lead the
combined leaderboard under the current selection rule. Regularized linear
models remain surprisingly competitive and interpretable. SARIMAX provides a
useful time-series comparison point with explicit structure and AIC/BIC
diagnostics. Neural finalists remain the next block to reconcile under the same
pandemic-safe feature logic.

### Broader Experiment Scope

The larger local research path can continue expanding breadth and depth while
preserving the same output contract. Useful follow-up directions include
targeted Phase A refinements, a pandemic-safe Phase C rerun around the most
promising GRU/LSTM sequence models, and additional target variants.

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

The dashboard provides one weighted selection score alongside direct MAE and
RMSE metrics:

`balanced_score = 0.75 * MAE + 0.25 * RMSE`.
This is the project scoring logic and balances typical accuracy with a moderate
penalty for larger mistakes.

MAE and RMSE are computed overall and within the pre-COVID, shock, recovery,
and recent periods. Period-specific ratios compare a period's error to the
pre-COVID error.

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
The leaderboard includes the overall balanced score plus overall and
period-specific MAE/RMSE. These are useful for finding models that were not just
accurate overall, but also resilient through disruption.

`balanced_score = 0.75 * MAE + 0.25 * RMSE`.
This preserves the original project default and lightly penalizes large misses.

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
`tabular_raw`, `pca_20`, `pca_95`, `sequence_raw`, `sequence_pca_20`,
`sequence_pca_95`, or a future learned latent representation. For Phase C,
`sequence_pca_20` retains up to 20 principal components and `sequence_pca_95`
retains enough components to explain 95 percent of the fit-window variance.
PCA is refit inside each historical training window after any feature policy is
applied. This is especially important for neural-net and RNN-style experiments,
where sequence length and compressed feature spaces become part of the model
design.

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
