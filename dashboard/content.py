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

- Ingests and normalizes transit, service, fuel, inflation, and income context.
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
linear, tree, autoregressive, and neural-net model families under one artifact
contract.
"""


SYSTEM_ARCHITECTURE = """
### Architecture

- AWS path demonstrates a production-shaped workflow with containerized Python
  jobs, ECS tasks, Step Functions orchestration, S3 artifacts, and CloudWatch
  logs.
- Local path handles broader experiment sweeps where cost and iteration speed
  matter more.
- Training runners write portable Parquet/JSON artifacts under a shared
  metadata contract.
- DuckDB builds a queryable analytical mart, then exports dashboard-ready files.
- Streamlit reads curated static artifacts so the public app stays fast and
  inexpensive to host.
"""


SYSTEM_REASONING = """
### Reasoning

- AWS proves orchestration, repeatability, logging, and artifact handoff.
- Local/GPU runs make large model grids practical without turning cloud cost
  into the main constraint.
- The dashboard consumes the same artifact shape regardless of where training
  happened.
- This split keeps the project readable as a portfolio system while still
  supporting deep experimentation.
"""


SYSTEM_OVERVIEW = """
### Alternatives In Context

- AWS Batch could replace ECS for larger queued training workloads.
- Lambda remains useful for lighter ingestion or source-specific fetch steps.
- Managed training jobs would make sense if the project moved toward scheduled
  retraining at scale.
- A fuller MLflow registry could promote candidates into explicit model stages.
- The current design favors clarity: cloud orchestration plus local experiment
  breadth under one artifact contract.
"""


DATA_PRIMARY_EDA = """
### Primary EDA

- Core series is monthly transit ridership and service context.
- Current target is H3 UPT, forecast three months ahead from each rolling
  as-of month.
- Series includes seasonality, trend, and a visible COVID-era structural break.
- Useful for testing whether models handle both ordinary seasonality and
  disrupted operating regimes.
"""


DATA_SECONDARY_EDA = """
### Secondary EDA

- External context includes gas prices, CPI/inflation, and King County median
  household income.
- Income is annual FRED data converted into prior-year monthly context.
- Forecast months only use context that would have been known at the as-of date.
- These features create testable hypotheses about economic pressure and
  changing travel behavior.
"""


DATA_CALCULATED_FEATURES = """
### Calculated Features

- Feature table includes lags, rolling summaries, regime flags, exogenous
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
  target-month context, and COVID/recovery regimes.
- They support raw/direct forecasts and residual models built around a seasonal
  naive baseline.
- Autoregressive and neural sequence models use the same rolling as-of frame,
  with compact exogenous or sequence representations where appropriate.
"""


DATA_AS_OF_REGIME_FEATURES = """
### As-Of Safe Regime Features

- Regime labels are useful, but they must not give the model information from
  the future.
- Pre-COVID forecasts should not receive a countdown-style feature such as
  "months until COVID"; that would leak the future shock into the training
  process.
- The cleaner design is adaptive: before a disruption is observable, regime
  features are neutral. After the disruption is known, parallel experiments can
  compare models with and without explicit shock/recovery features.
- Evaluation-period labels can still be used after the fact to compare
  pre-COVID, shock, recovery, and recent performance. Those labels are
  analytical metadata, not necessarily deployable model inputs.
- The next rerun will separate these ideas more carefully so regime-aware
  models measure whether known disruption context helps recovery forecasting,
  not whether a model benefited from hindsight.
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

The current dashboard bundle combines three completed local experiment blocks:

- Phase A: baseline, linear, and tree-based models over the monthly feature
  table. This includes seasonal naive, Ridge, Lasso, Elastic Net, random forest,
  Extra Trees, and XGBoost.
- Phase B: autoregressive models using the same rolling as-of evaluation frame.
  This includes ARIMA, SARIMA, and SARIMAX, with SARIMAX using compact service,
  economic, income-pressure, and service-economic exogenous sets.
- Phase C: GPU-trained neural sequence finalists. This includes GRU and LSTM
  models run monthly across the full evaluation window after broader screening
  and capacity experiments narrowed the grid.

All three phases are evaluated from the same monthly as-of timeline and exported through the
same artifact contract, so their forecasts can be compared in one dashboard
without special-case logic.

### Comparison Dimensions

The experiment compares several layers of modeling decisions:

- `model_family`: broad modeling group such as baseline, linear, tree,
  autoregressive, and neural-net.
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

**Variations:** Phase A evaluates Ridge, Lasso, and Elastic Net in raw and
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

**Variations:** Phase B evaluates ARIMA, SARIMA, and SARIMAX. SARIMA adds
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

**Variations:** Phase A evaluates random forest, Extra Trees, and XGBoost across
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
LSTM models. Capacity experiments then focused the monthly finalist run on GRU
and LSTM. The finalist bundle compares raw and residual modes, compact and wider
feature families, dynamic feature policies, raw sequence representations, and
selected PCA sequence representations.

**Parameter Search:** The monthly finalists use sequence lengths of `36`
months, batch size `24`, up to `300` epochs, early-stopping patience `25`, and
learning-rate reduction patience `8`. GRU variants compare recurrent layer
sizes `[512, 100]` and `[256, 100]` with learning rates `0.001` and `0.0003`.
LSTM variants compare `[256, 100]` and a larger `[1000, 100]` network with
learning rates `0.0003` and `0.0025`. Dense heads use `[200, 10]`, with dropout
and weight decay varied through the finalist definitions.

**Observations:** The strongest neural finalist is a residual GRU using a
compact recent-history family, with overall MAE near 587,000. LSTM is close
behind. Neural models do not displace the best tree or regularized linear
models in the full-window leaderboard, but the result is still informative:
larger sequence models are not automatically better, and residual modeling plus
compact histories deserves closer inspection in the recent regime.

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
period-specific shock/recovery behavior. Additional local refinements and any
future target variants can be added without changing the dashboard contract.

The current pattern is already useful. Strong boosted-tree models lead the
combined leaderboard under the current selection rule. Regularized linear
models remain surprisingly competitive and interpretable. SARIMAX provides a
useful time-series comparison point with explicit structure and AIC/BIC
diagnostics. Neural finalists add a GPU-trained sequence-model comparison under
the same monthly historical evaluation frame.

### Broader Experiment Scope

The larger local research path can continue expanding breadth and depth while
preserving the same output contract. Useful follow-up directions include
targeted Phase A refinements, narrower neural follow-ups around the most
promising residual sequence models, and additional target variants.

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
