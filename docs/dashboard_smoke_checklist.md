# Dashboard Smoke Checklist

Use this checklist before pushing dashboard code, dashboard content, or public bundle changes.

## Automated Checks

Validate the committed public artifact bundle and recursive dashboard runtime imports:

```bash
.venv/bin/python scripts/validate_dashboard_bundle.py
```

Compile the Streamlit app, page modules, shared helpers, and validator:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/transit_pycache \
  .venv/bin/python -m py_compile \
  dashboard/app.py \
  dashboard/*.py \
  dashboard/sections/*.py \
  scripts/validate_dashboard_bundle.py
```

Check for whitespace errors before committing:

```bash
git diff --check
```

## Local Runtime Check

Run the dashboard against the committed public bundle:

```bash
DASHBOARD_ARTIFACT_DIR=dashboard/public_artifacts/latest \
  .venv/bin/python -m streamlit run dashboard/app.py --server.port 8507
```

Then inspect:

- Project Overview loads without errors.
- Data page loads source-series tabs, EDA callouts, and availability tables.
- System page reflects the current artifact structure.
- Experiment page shows feature-family counts and policy/transform explanations.
- Model Explorer filters stick after selection and update charts/tables.
- Insights page loads COVID shock charts and result tables without hidden table scrolling surprises.

## Public Deployment Check

After pushing to `main`:

- Confirm GitHub shows the expected commit.
- In Streamlit Cloud, use **Relaunch to update** if the live app shows stale code.
- Check logs for dependency installation, import errors, or artifact loading errors.
- Hard refresh the browser after Streamlit reports that the app updated.

## Common Failure Clues

- Missing dependency: update `dashboard/requirements.txt`, then rerun the validator.
- Missing artifact: rebuild or recommit `dashboard/public_artifacts/latest/`.
- Old navigation on live site: confirm Streamlit Cloud pulled latest `main`, then relaunch.
- Slow interaction: first check Model Explorer path loading and selected result slice size.
