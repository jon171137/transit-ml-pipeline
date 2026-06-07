# AWS Catch-Up TODO

Date noted: 2026-06-06

This file tracks local project changes that still need to be carried into the AWS version of the pipeline. The local workflow is currently ahead of the cloud smoke-test workflow.

## Income Ingestion And Integration

- Add the FRED income ingestion path to AWS using `lambda_income.py` / `normalize_fred_income.py`.
- Store normalized income artifacts under the intended S3 prefix.
- Update the Step Functions state machine so income normalization runs with the other source normalizers.
- Update `build_integrated_monthly_base.py` in the ECS image and task flow so the integrated monthly base includes:
  - `income_reference_year`
  - `king_county_median_household_income_prior_year`
  - `king_county_monthly_household_income_prior_year`
  - `king_county_income_yoy_pct_prior_year`
  - `king_county_income_2yr_pct_prior_year`

## Pandemic-Safe Regime Features

- Rebuild the AWS feature-table artifact using the updated `create_feature_table.py`.
- Confirm the S3 feature table contains model-facing pandemic regime columns:
  - `pandemic_observed`
  - `pandemic_disruption_active`
  - `post_pandemic_observed`
  - `months_since_pandemic_observed`
- Confirm feature family definitions use the `pandemic_*` columns, not the legacy `covid_*` aliases.
- Keep legacy aliases only for compatibility while older dashboard/run artifacts are being phased out.

## Container And Task Definitions

- Rebuild and push a new ECS image after the income and pandemic-safe feature updates are complete.
- Update the ECS task definition image URI.
- Confirm task environment variables still cover the full pipeline:
  - bucket and S3 prefixes
  - `PIPELINE_RUN_ID`
  - `IMAGE_URI`
  - FRED secret mapping

## Step Functions Pipeline

- Add income normalization into the orchestration before integration.
- Keep source normalizers parallel where possible:
  - transit
  - gas
  - inflation
  - income
- Run integration only after all source normalizers complete.
- Run feature-table creation after integration.
- Add a small modeling/training smoke step after feature-table creation.
- Keep retry, timeout, and CloudWatch log policies aligned with the current state machine.

## Modeling Smoke Test

- Add a small cloud-side model run that proves the AWS pipeline can train and publish a dashboard-compatible artifact without running the full local research grid.
- Recommended smoke scope:
  - seasonal naive
  - one Ridge configuration
  - one XGBoost configuration
  - one compact feature family
  - one short as-of window
- Output should land in S3 under a distinct run ID and be usable by the dashboard artifact contract.

## Dashboard Artifact Flow

- Decide whether the public dashboard continues to use curated static artifacts or later reads an AWS-published dashboard bundle.
- If using AWS-published artifacts, add a promotion step that copies the selected S3 run output into the static public bundle path.
- Keep Streamlit Community Cloud deployment free of AWS credentials unless there is a clear reason to add signed/cloud access later.
