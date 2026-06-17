import pandas as pd
import streamlit as st
from typing import Optional

from content import PROJECT_OVERVIEW, PROJECT_OVERVIEW_CASE_STUDY, PROJECT_OVERVIEW_SYSTEM
from formatting import date_range_label, format_int, manifest_value


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
