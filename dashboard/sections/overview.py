from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from typing import Optional

from constants import PROJECT_OVERVIEW_VIDEO_PATH
from content import PROJECT_OVERVIEW, PROJECT_OVERVIEW_CASE_STUDY, PROJECT_OVERVIEW_SYSTEM
from formatting import date_range_label, format_int, manifest_value


REFRESH_COMPARISON = Path("dashboard/public_artifacts/latest/refresh_2026_comparison/model_leaderboard.parquet")


def render_refresh_comparison() -> None:
    """Keep the corrected comparison separate from the older experiment archive."""
    if not REFRESH_COMPARISON.exists():
        return
    results = pd.read_parquet(REFRESH_COMPARISON)
    required = {"model_build", "feature_family_name", "mode", "mae", "rmse", "selection_score_balanced"}
    if not required.issubset(results.columns):
        st.warning("The updated model comparison is unavailable because its artifact is incomplete.")
        return

    st.subheader("Updated Model Comparison: Prophet And Challengers")
    st.write(
        "A focused follow-up compares selected models on 185 monthly rolling forecasts "
        "with target months April 2011 through August 2026. It includes four Prophet "
        "configurations and selected XGBoost, linear, SARIMAX, and seasonal-naive "
        "references, using corrected three-month-ahead training-label availability."
    )
    st.caption(
        "This is a separate nine-configuration comparison, not a re-run of the full "
        "historical archive shown elsewhere in the dashboard. Prophet finalists were "
        "chosen using a screen spanning this period, including 2026; their scores are "
        "retrospective, not an untouched holdout. Source publication lags and historical "
        "data vintages remain limitations."
    )
    labels = {"xgboost": "XGBoost", "prophet": "Prophet", "sarimax": "SARIMAX",
              "elastic_net": "Elastic Net", "ridge": "Ridge", "seasonal_naive": "Seasonal naive"}
    display = results.copy().sort_values("selection_score_balanced")
    display["Model"] = display["model_build"].map(labels).fillna(display["model_build"])
    display["Configuration"] = display["feature_family_name"].str.replace("_", " ", regex=False) + " / " + display["mode"].astype(str)
    display["Model and configuration"] = display["Model"] + " | " + display["Configuration"]
    fig = px.bar(
        display.sort_values("mae", ascending=True), x="mae", y="Model and configuration",
        color="Model", orientation="h", hover_data={"rmse": ":,.0f", "mae": ":,.0f"},
        labels={"mae": "Mean absolute error (monthly UPT)", "Model and configuration": ""},
    )
    fig.update_layout(template="plotly_white", showlegend=False, height=430, margin={"l": 30, "r": 25, "t": 20, "b": 45})
    st.plotly_chart(fig, width="stretch")
    st.caption("Each bar is one configuration; lower MAE is better. The table also shows RMSE and the balanced ranking score.")
    table = display[["Model", "Configuration", "mae", "rmse", "selection_score_balanced"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE", "selection_score_balanced": "Balanced score"}
    )
    for column in ["MAE", "RMSE", "Balanced score"]:
        table[column] = table[column].map(lambda value: f"{value:,.0f}")
    st.table(table.reset_index(drop=True))
    st.write(
        "The best Prophet configuration uses ridership history and calendar features. "
        "It is competitive overall and in stable periods, but XGBoost's lower errors "
        "during the COVID shock and recovery give it the full-range lead. Residual "
        "Elastic Net performed best on the five latest 2026 target months among the "
        "previously locked challengers; that small sample is not enough to establish "
        "a durable champion."
    )


def render_project_overview_video() -> None:
    st.subheader("Video Overview")
    st.write(
        "A narrated walkthrough of the project motivation, approach, system design, "
        "and main modeling takeaways."
    )
    if PROJECT_OVERVIEW_VIDEO_PATH.exists():
        st.video(str(PROJECT_OVERVIEW_VIDEO_PATH), format="video/mp4", width="stretch")
    else:
        st.info(f"Expected overview video not found: `{PROJECT_OVERVIEW_VIDEO_PATH}`")


def render_public_bundle_note(experiment_manifest: dict) -> None:
    bundle = experiment_manifest.get("public_dashboard_bundle")
    if not bundle:
        return
    retained = format_int(bundle.get("selected_configurations"))
    source = format_int(bundle.get("source_configurations"))
    full_path_rows = bundle.get("full_path_rows", {})
    flat_path_rows = bundle.get("flat_path_rows", {})
    full_forecast_rows = format_int(full_path_rows.get("forecast_paths")) if isinstance(full_path_rows, dict) else "-"
    flat_forecast_rows = format_int(flat_path_rows.get("forecast_paths")) if isinstance(flat_path_rows, dict) else "-"
    keep_fraction = bundle.get("keep_fraction")
    keep_pct = f"{float(keep_fraction) * 100:.0f}%" if keep_fraction is not None else "configured"
    st.info(
        "This live dashboard uses a performance-aware public artifact bundle. "
        f"The model index includes {source} source configurations for filtering and comparison. "
        f"Flat compatibility path files retain {retained} configurations "
        f"({flat_forecast_rows} forecast rows) from the best {keep_pct} of core metrics "
        "plus baseline/champion models. "
        f"The full path-level forecast universe ({full_forecast_rows} rows) is partitioned "
        "and loaded on demand after filters are applied."
    )


def render_overview_page(
    experiment_manifest: dict,
    forecast_paths: pd.DataFrame,
    champion: dict,
    displayed_prediction_count: Optional[int] = None,
    indexed_config_count: Optional[int] = None,
) -> None:
    overview_intro_cols = st.columns(2)
    overview_intro_cols[0].markdown(PROJECT_OVERVIEW_CASE_STUDY)
    overview_intro_cols[1].markdown(PROJECT_OVERVIEW_SYSTEM)
    st.markdown(PROJECT_OVERVIEW)
    render_refresh_comparison()
    render_project_overview_video()
    render_public_bundle_note(experiment_manifest)

    bundle = experiment_manifest.get("public_dashboard_bundle", {})
    full_config_count = bundle.get("source_configurations") or experiment_manifest.get("model_config_count") or indexed_config_count
    indexed_config_count = indexed_config_count or full_config_count
    full_prediction_count = experiment_manifest.get("prediction_count") or len(forecast_paths)
    displayed_prediction_count = displayed_prediction_count or len(forecast_paths)

    overview_cols = st.columns(6)
    overview_cols[0].metric(
        "Forecast Horizon",
        f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', '-'))} months",
    )
    overview_cols[1].metric("Full Model Configs", format_int(full_config_count))
    overview_cols[2].metric("Indexed Configs", format_int(indexed_config_count))
    overview_cols[3].metric("Full Rolling Predictions", format_int(full_prediction_count))
    overview_cols[4].metric("Flat Path Predictions", format_int(displayed_prediction_count))
    overview_cols[5].metric("Target Window", date_range_label(forecast_paths, "target_date"))
