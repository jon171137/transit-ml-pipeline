# Large Experiment Phase B: Autoregressive Models

Phase B adds time-series model classes to the same rolling evaluation contract used by Phase A. The goal is to compare whether explicitly autoregressive model structure competes with the feature-table-driven linear and tree models when all are judged on the same H3 UPT forecast task.

## Scope

- Target: monthly `upt`
- Horizon: 3 months ahead
- Rolling as-of dates: monthly from 2011-01-01 through 2025-12-01
- Evaluation artifact shape: identical to Phase A `predictions`, `model_runs`, `metrics`, `feature_sets`, dashboard exports, and DuckDB mart inputs
- Model family: `autoregressive`
- Model builds: `arima`, `sarima`, `sarimax`

## Configs

Primary run:

```bash
.venv/bin/python run_autoregressive_models.py \
  --experiment-config experiment_configs/phase_b_autoregressive_v1.yaml
```

Smoke run:

```bash
.venv/bin/python run_autoregressive_models.py \
  --experiment-config experiment_configs/phase_b_smoke.yaml
```

Planning summary:

```bash
.venv/bin/python plan_large_experiment.py \
  --config experiment_configs/phase_b_autoregressive_v1.yaml
```

## Grid

The Phase B v1 grid is intentionally broad enough to be useful:

- ARIMA: 12 order/trend candidates after invalid integrated-trend pairs are removed
- SARIMA: 24 seasonal candidates
- SARIMAX: 60 seasonal/exogenous candidates

SARIMAX exogenous sets are kept interpretable:

- `service`: contemporaneous service context (`vrm`, `vrh`, `voms`)
- `economic_lagged`: gas, CPI, and prior-year income movement
- `income_pressure`: income, inflation, and gas/income pressure terms
- `service_economic`: service plus compact economic context

## Integration With Phase A

Phase B results can be combined with Phase A results using:

```bash
.venv/bin/python combine_experiment_results.py \
  --results-dir experiments_output/large_phase_a_v1/results \
  --results-dir experiments_output/phase_b_autoregressive_v1/results \
  --output-results-dir experiments_output/combined_phase_ab_v1/results \
  --output-dashboard-dir dashboard_artifacts/aws_streamlined/combined_phase_ab_v1 \
  --experiment-id combined_phase_ab_v1
```

Then build the DuckDB mart/dashboard export from that combined folder:

```bash
.venv/bin/python build_experiment_mart.py \
  --results-dir experiments_output/combined_phase_ab_v1/results \
  --dashboard-dir dashboard_artifacts/aws_streamlined/combined_phase_ab_v1 \
  --duckdb-path experiments_output/combined_phase_ab_v1/experiments.duckdb \
  --dashboard-export-dir dashboard_artifacts/aws_streamlined/combined_phase_ab_v1_from_duckdb \
  --replace
```

The dashboard can then point at the combined export:

```bash
DASHBOARD_ARTIFACT_DIR=dashboard_artifacts/aws_streamlined/combined_phase_ab_v1_from_duckdb \
  .venv/bin/python -m streamlit run dashboard/app.py --server.headless true --server.port 8503
```

## Notes And Caveats

- The SARIMAX exogenous variables use values available in the feature table. For this portfolio experiment, data-release timing is treated as a known simplification rather than modeled source-by-source.
- Phase B currently logs feature sets but not coefficient-style feature importance. AIC/BIC are stored on `model_runs` for autoregressive diagnostics.
- ARIMA/SARIMA/SARIMAX are refit at every as-of month. This is intentionally heavier than the streamlined AWS demo because the broad local experiment is meant to stress the modeling space.
