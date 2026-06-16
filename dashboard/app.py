import base64
import json
import os
import re
import sys
from html import escape
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import polars as pl
except ImportError:  # Polars is an optimization, not a hard local-dev requirement.
    pl = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from content import (
    DATA_CALCULATED_FEATURES,
    DATA_AS_OF_REGIME_FEATURES,
    DATA_PRIMARY_DATA,
    DATA_SECONDARY_DATA,
    DATA_TIME_FEATURES,
    EXPERIMENT_OVERVIEW,
    PERIOD_METRIC_EXPLANATION,
    PERIOD_METRIC_SHORT_EXPLANATION,
    PROJECT_OVERVIEW,
    PROJECT_OVERVIEW_CASE_STUDY,
    PROJECT_OVERVIEW_SYSTEM,
    REPRESENTATION_AND_COMPLEXITY_EXPLANATION,
    SYSTEM_ARCHITECTURE,
    SYSTEM_ARTIFACT_FLOW,
    SYSTEM_OVERVIEW,
    SYSTEM_REASONING,
)


DEFAULT_ARTIFACT_DIR = Path("dashboard/public_artifacts/latest")
IMAGE_ASSET_DIR = Path("dashboard/assets/images")
DEFAULT_FEATURE_FAMILIES_PATH = Path("dashboard/public_artifacts/latest/feature_families.json")
DEFAULT_INTEGRATED_BASE_PATH = Path("raw_files/integrated_monthly_base.parquet")
DEFAULT_FEATURE_TABLE_PATH = Path("feature_store/income_interactions_h3_v1/feature_table.parquet")
DEFAULT_IMPUTATION_LOG_PATH = Path("feature_store/income_interactions_h3_v1/imputation_log.parquet")
RESULTS_INSIGHTS_NOTEBOOK_PATH = Path("experiment_results_insights.ipynb")
PHASE_A_V3_CONFIG_PATH = Path("experiment_configs/large_phase_a_v3_pandemic_safe.yaml")
PHASE_B_V3_CONFIG_PATH = Path("experiment_configs/phase_b_autoregressive_v3_pandemic_safe.yaml")
PHASE_C_MONTHLY_CONFIG_PATH = Path("experiment_configs/phase_c_neural_monthly_finalists.yaml")
# Keep the MOV as the editable capture; use MP4 for browser and Streamlit Cloud playback.
SYSTEM_ARCH_VIDEO_PATH = IMAGE_ASSET_DIR / "Transit_System_Build.mp4"
STEP_FUNCTION_SCREENSHOT_PATH = IMAGE_ASSET_DIR / "Step_Function_Screenshot.png"
VIDEO_MIME_TYPES = {".mov": "video/quicktime", ".mp4": "video/mp4", ".webm": "video/webm"}
MODEL_FAMILY_ORDER = ["baseline", "linear", "autoregressive", "tree", "neural_net", "neural"]
MODEL_BUILD_ORDER = [
    "seasonal_naive",
    "ridge",
    "lasso",
    "elastic_net",
    "arima",
    "sarima",
    "sarimax",
    "random_forest",
    "extra_trees",
    "xgboost",
    "mlp",
    "cnn",
    "rnn",
    "gru",
    "lstm",
]
BASELINE_MODEL_FAMILIES = {"baseline"}
BASELINE_MODEL_BUILDS = {"seasonal_naive", "naive"}
REQUIRED_FILES = {
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
    "champion_predictions": "champion_predictions.parquet",
    "champion_selection": "champion_selection.json",
}
OPTIONAL_FILES = {
    "model_leaderboard_full": "model_leaderboard_full.parquet",
    "feature_family_summary_full": "feature_family_summary_full.parquet",
    "complexity_profile_full": "complexity_profile_full.parquet",
    "path_partition_manifest": "path_partition_manifest.json",
    "overview_top_models": "overview_top_models.parquet",
    "overview_prediction_paths": "overview_prediction_paths.parquet",
    "experiment_manifest": "experiment_manifest.json",
}
PATH_DATASET_DIRS = {
    "forecast_paths": "forecast_paths_by_build",
    "performance_over_time": "performance_over_time_by_build",
}

SCORE_RECIPES = {
    "balanced": {"label": "Balanced score", "mae_weight": 0.75, "rmse_weight": 0.25},
}
PER_BUILD_LIMIT_OPTIONS = ["Top 1", "Top 3", "Top 5", "Top 10", "Top 25", "All"]
TOTAL_LIMIT_OPTIONS = ["All", "Top 5", "Top 10", "Top 15", "Top 25", "Top 50", "Top 100"]
EVALUATION_PERIODS = {
    "pre_covid": "Pre-COVID",
    "covid_shock": "COVID shock",
    "recovery": "Recovery",
    "recent": "Recent",
}

RANK_METRIC_OPTIONS = {
    "Balanced score": ("selection_score_balanced", True),
    "MAE": ("mae", True),
    "RMSE": ("rmse", True),
    "R-squared": ("r2", False),
    "Adjusted R-squared": ("r2_adjusted", False),
    "Directional accuracy": ("diracc", False),
    "Pre-COVID MAE": ("pre_covid_mae", True),
    "Pre-COVID RMSE": ("pre_covid_rmse", True),
    "COVID shock MAE": ("covid_shock_mae", True),
    "COVID shock RMSE": ("covid_shock_rmse", True),
    "Recovery MAE": ("recovery_mae", True),
    "Recovery RMSE": ("recovery_rmse", True),
    "Recent MAE": ("recent_mae", True),
    "Recent RMSE": ("recent_rmse", True),
    "Shock penalty": ("shock_penalty", True),
    "Recovery ratio": ("recovery_ratio", True),
    "Recent recovery ratio": ("recent_recovery_ratio", True),
    "RMSE shock penalty": ("rmse_shock_penalty", True),
    "RMSE recovery ratio": ("rmse_recovery_ratio", True),
    "RMSE recent recovery ratio": ("rmse_recent_recovery_ratio", True),
}

FEATURE_POLICY_DESCRIPTIONS = {
    "none": "Use every column in the selected feature family.",
    "corr_pruned": "Within each as-of training window, drop features that are highly correlated with earlier columns.",
    "variance_pruned": "Within each as-of training window, drop near-constant features with almost no variance.",
    "mutual_info_top_20": "Rank features by mutual information with the target inside the training window and keep the top 20.",
    "mutual_info_top_30": "Rank features by mutual information with the target inside the training window and keep the top 30.",
    "lasso_selected": "Fit a Lasso selector inside the training window and keep features with nonzero coefficients.",
    "tree_top_20": "Fit a shallow Extra Trees selector inside the training window and keep the 20 most important features.",
    "tree_top_30": "Fit a shallow Extra Trees selector inside the training window and keep the 30 most important features.",
}

FEATURE_TRANSFORM_DESCRIPTIONS = {
    "identity": "Use the selected feature columns as-is.",
    "log_signed": "Add signed log1p versions of selected features, preserving direction for negative values.",
    "quadratic": "Add squared terms for selected features so linear models can fit curved relationships.",
    "cubic": "Add squared and cubed terms for selected features so linear models can fit stronger nonlinear curvature.",
    "log_signed_quadratic_cubic": "Add signed-log, squared, and cubed versions of selected features in one expanded representation.",
}

FEATURE_TRANSFORM_LABELS = {
    "identity": "No transform",
    "log_signed": "Signed log",
    "quadratic": "Quadratic",
    "cubic": "Cubic",
    "log_signed_quadratic_cubic": "Signed log + quadratic + cubic",
}
FEATURE_TRANSFORM_ORDER = [
    "identity",
    "log_signed",
    "quadratic",
    "cubic",
    "log_signed_quadratic_cubic",
]

PERIOD_RANK_WINDOWS = {
    "Pre-COVID MAE": (None, "2020-02-01"),
    "Pre-COVID RMSE": (None, "2020-02-01"),
    "COVID shock MAE": ("2020-03-01", "2021-06-01"),
    "COVID shock RMSE": ("2020-03-01", "2021-06-01"),
    "Shock penalty": ("2020-03-01", "2021-06-01"),
    "RMSE shock penalty": ("2020-03-01", "2021-06-01"),
    "Recovery MAE": ("2021-07-01", "2022-12-01"),
    "Recovery RMSE": ("2021-07-01", "2022-12-01"),
    "Recovery ratio": ("2021-07-01", "2022-12-01"),
    "RMSE recovery ratio": ("2021-07-01", "2022-12-01"),
    "Recent MAE": ("2023-01-01", None),
    "Recent RMSE": ("2023-01-01", None),
    "Recent recovery ratio": ("2023-01-01", None),
    "RMSE recent recovery ratio": ("2023-01-01", None),
}


st.set_page_config(
    page_title="Transit Forecasting Lab",
    page_icon="",
    layout="wide",
)


def inject_site_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --portfolio-teal: #007f68;
            --portfolio-teal-dark: #00614f;
            --portfolio-blue: #075fed;
            --portfolio-red: #e33f3f;
            --portfolio-ink: #2f323a;
            --portfolio-muted: #6b7280;
            --portfolio-surface: #f7faf9;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #ffffff;
            color: var(--portfolio-ink);
        }

        .block-container {
            padding-top: 2.1rem;
        }

        .portfolio-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border-top: 4px solid var(--portfolio-teal);
            background: linear-gradient(90deg, rgba(0, 127, 104, 0.11), rgba(7, 95, 237, 0.04));
            border-bottom: 1px solid rgba(0, 127, 104, 0.16);
            padding: 0.55rem 0.9rem;
            margin: 0 0 0.7rem;
            color: var(--portfolio-ink);
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0;
        }

        .element-container:has(.portfolio-banner),
        .element-container:has(.dashboard-hero) {
            margin-bottom: 0 !important;
        }

        .portfolio-banner span {
            color: var(--portfolio-teal-dark);
        }

        .portfolio-title {
            color: var(--portfolio-ink);
            font-size: 1.45rem;
            line-height: 1.05;
            font-weight: 800;
            white-space: nowrap;
        }

        .portfolio-links {
            color: var(--portfolio-ink);
            text-align: right;
            white-space: nowrap;
        }

        .portfolio-banner a {
            color: var(--portfolio-ink);
            text-decoration: none;
            font-weight: 700;
        }

        .portfolio-banner a:hover {
            color: var(--portfolio-blue);
            text-decoration: underline;
        }

        .portfolio-banner .banner-link-divider {
            color: var(--portfolio-muted);
            margin: 0 0.35rem;
            font-weight: 500;
        }

        h1, h2, h3 {
            color: var(--portfolio-ink);
            letter-spacing: 0;
        }

        p, li, label, span {
            color: inherit;
        }

        [data-testid="stSidebar"] {
            background: var(--portfolio-surface);
            border-right: 1px solid rgba(0, 127, 104, 0.12);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--portfolio-teal-dark);
        }

        div[data-testid="stMetricValue"] {
            color: var(--portfolio-ink);
            font-size: 1.45rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--portfolio-muted);
            font-size: 0.78rem;
        }

        .compact-kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 0.65rem;
            margin: 0.85rem 0 0.75rem;
        }

        .compact-kpi {
            border-top: 2px solid rgba(0, 127, 104, 0.35);
            background: rgba(247, 250, 249, 0.7);
            padding: 0.55rem 0.65rem;
            min-height: 4rem;
        }

        .compact-kpi .label {
            color: var(--portfolio-muted);
            font-size: 0.73rem;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }

        .compact-kpi .value {
            color: var(--portfolio-ink);
            font-size: 1.08rem;
            line-height: 1.2;
            font-weight: 650;
            overflow-wrap: anywhere;
        }

        .compact-context {
            color: var(--portfolio-muted);
            font-size: 0.82rem;
            margin-bottom: 1.25rem;
        }

        .dashboard-hero {
            display: block;
            margin: 0.15rem 0 0.7rem;
        }

        .dashboard-title h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.03;
        }

        .dashboard-title p {
            color: var(--portfolio-muted);
            margin: 0.3rem 0 0;
            font-size: 0.93rem;
        }

        .champion-summary {
            border-top: 3px solid rgba(0, 127, 104, 0.45);
            background: rgba(247, 250, 249, 0.75);
            padding: 0.75rem 0.85rem;
        }

        .champion-summary .summary-title {
            color: var(--portfolio-teal-dark);
            font-size: 0.78rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .champion-summary .summary-title-inline {
            color: var(--portfolio-ink);
            font-size: 1.05rem;
            line-height: 1.22;
            margin-bottom: 0.75rem;
        }

        .champion-summary .summary-title-inline .summary-title-label {
            color: var(--portfolio-teal-dark);
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .champion-summary .summary-title-inline .summary-title-divider {
            color: rgba(47, 50, 58, 0.48);
            margin: 0 0.25rem;
        }

        .champion-summary .summary-title-inline .summary-title-context {
            color: var(--portfolio-ink);
            font-weight: 450;
            letter-spacing: 0;
        }

        .champion-summary-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            column-gap: 0.9rem;
            row-gap: 0.45rem;
        }

        .champion-summary .label {
            color: var(--portfolio-muted);
            font-size: 0.68rem;
            line-height: 1.1;
        }

        .champion-summary .value {
            color: var(--portfolio-ink);
            font-size: 0.9rem;
            line-height: 1.2;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        .champion-summary p {
            font-size: 0.9rem;
            line-height: 1.35;
            margin: 0 0 0.45rem;
        }

        .champion-summary ul {
            margin: 0.5rem 0 0;
            padding-left: 1.15rem;
        }

        .champion-summary li {
            margin-bottom: 0.4rem;
            line-height: 1.35;
        }

        .champion-summary .summary-note {
            color: var(--portfolio-muted);
            font-size: 0.78rem;
            line-height: 1.3;
            margin: 0.55rem 0 0;
        }

        .eda-section {
            margin: 1.05rem 0 1.15rem;
            padding-top: 0.9rem;
            border-top: 1px solid rgba(47, 50, 58, 0.12);
        }

        .eda-section h3 {
            margin: 0 0 0.15rem;
            color: var(--portfolio-ink);
            font-size: 1.05rem;
            letter-spacing: 0;
        }

        .eda-section .eda-kicker {
            color: var(--portfolio-teal-dark);
            font-size: 0.82rem;
            font-weight: 750;
            letter-spacing: 0.03em;
            margin: 0 0 0.3rem;
            text-transform: uppercase;
        }

        .eda-section .eda-context {
            color: var(--portfolio-muted);
            font-size: 0.86rem;
            line-height: 1.35;
            margin: 0 0 0.8rem;
            max-width: 60rem;
        }

        .mom-callout-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(112px, 1fr));
            gap: 0.42rem;
        }

        .mom-month-card {
            background: rgba(247, 250, 249, 0.78);
            border-top: 3px solid rgba(0, 127, 104, 0.38);
            padding: 0.45rem 0.48rem 0.5rem;
            min-height: 5.45rem;
        }

        .mom-month-title {
            color: var(--portfolio-ink);
            font-size: 0.78rem;
            font-weight: 760;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }

        .mom-badge-row {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
        }

        .mom-badge {
            align-items: center;
            border: 2px solid rgba(47, 50, 58, 0.82);
            border-radius: 999px;
            display: inline-flex;
            flex-direction: column;
            height: 2.75rem;
            justify-content: center;
            min-width: 2.75rem;
            padding: 0.2rem;
            text-align: center;
        }

        .mom-badge.transit-positive {
            background: #1f6f8b;
            color: #ffffff;
        }

        .mom-badge.transit-negative {
            background: #b7791f;
            color: #ffffff;
        }

        .mom-badge.gas-positive {
            background: #c85214;
            color: #ffffff;
        }

        .mom-badge.gas-negative {
            background: #1f6f8b;
            color: #ffffff;
        }

        .mom-badge .series {
            color: inherit;
            font-size: 0.53rem;
            font-weight: 760;
            line-height: 1.05;
        }

        .mom-badge .value {
            color: inherit;
            font-size: 0.62rem;
            font-weight: 720;
            line-height: 1.15;
            margin-top: 0.08rem;
        }

        .eda-chart-panel {
            background: rgba(247, 250, 249, 0.48);
            border-top: 3px solid rgba(0, 127, 104, 0.24);
            margin: 0.75rem 0 1.15rem;
            padding: 0.7rem 0.8rem 0.85rem;
        }

        .eda-chart-panel h4 {
            color: var(--portfolio-ink);
            font-size: 0.96rem;
            line-height: 1.2;
            margin: 0 0 0.4rem;
        }

        .eda-chart-panel p {
            color: var(--portfolio-muted);
            font-size: 0.84rem;
            line-height: 1.38;
            margin: 0 0 0.45rem;
        }

        .artifact-flow-table {
            border: 1px solid rgba(47, 50, 58, 0.14);
            border-radius: 0.55rem;
            margin: 0.75rem 0 0.55rem;
            overflow: hidden;
            width: 100%;
        }

        .artifact-flow-row {
            display: grid;
            grid-template-columns: minmax(9rem, 0.85fr) minmax(13rem, 1.45fr) minmax(16rem, 2fr);
        }

        .artifact-flow-row + .artifact-flow-row {
            border-top: 1px solid rgba(47, 50, 58, 0.11);
        }

        .artifact-flow-cell {
            border-right: 1px solid rgba(47, 50, 58, 0.11);
            color: var(--portfolio-ink);
            font-size: 0.86rem;
            line-height: 1.35;
            overflow-wrap: anywhere;
            padding: 0.58rem 0.7rem;
            white-space: normal;
            word-break: normal;
        }

        .artifact-flow-cell:last-child {
            border-right: 0;
        }

        .artifact-flow-head .artifact-flow-cell {
            background: rgba(247, 250, 249, 0.95);
            color: var(--portfolio-muted);
            font-weight: 700;
        }

        .artifact-flow-cell code {
            background: rgba(47, 50, 58, 0.055);
            border-radius: 0.22rem;
            color: var(--portfolio-ink);
            padding: 0.05rem 0.16rem;
            white-space: normal;
        }

        @media (max-width: 760px) {
            .artifact-flow-row {
                grid-template-columns: 1fr;
            }

            .artifact-flow-cell {
                border-right: 0;
            }

            .artifact-flow-cell + .artifact-flow-cell {
                border-top: 1px solid rgba(47, 50, 58, 0.08);
            }
        }

        .system-asset-block {
            margin-top: 1.1rem;
        }

        .system-asset-block h3 {
            margin-bottom: 0.25rem;
        }

        .system-asset-block p {
            color: var(--portfolio-muted);
            margin-top: 0;
            max-width: 62rem;
        }

        .system-arch-video-frame {
            width: 100%;
            aspect-ratio: 1692 / 1852;
            overflow: hidden;
            border: 1px solid rgba(0, 127, 104, 0.18);
            border-radius: 6px;
            background: #0f172a;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }

        .system-arch-video {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        @media (max-width: 1200px) {
            .compact-kpi-grid {
                grid-template-columns: repeat(2, minmax(130px, 1fr));
            }

            .dashboard-hero {
                grid-template-columns: 1fr;
            }

            .portfolio-banner {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.35rem;
            }

            .portfolio-links {
                text-align: left;
                white-space: normal;
            }
        }

        button[role="tab"][aria-selected="true"] {
            color: var(--portfolio-teal-dark);
            border-bottom-color: var(--portfolio-teal) !important;
        }

        div[data-testid="stTabs"] {
            margin-top: 0.25rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.75rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            min-height: 2.35rem;
            padding-top: 0.15rem;
            padding-bottom: 0.35rem;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            padding-top: 0.9rem;
        }

        button[kind="primary"],
        div.stButton > button:first-child {
            border-color: var(--portfolio-teal);
        }

        a {
            color: var(--portfolio-blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_project_banner() -> None:
    st.markdown(
        """
        <div class="portfolio-banner">
            <div class="portfolio-title">Transit Forecasting Lab</div>
            <div class="portfolio-links">
                <span>Personal Forecasting Project</span> by
                <a href="https://www.linkedin.com/in/sellersjon" target="_blank" rel="noopener noreferrer">Jon Sellers</a>
                <span class="banner-link-divider">|</span>
                <a href="https://github.com/jon171137/transit-ml-pipeline" target="_blank" rel="noopener noreferrer">GitHub Repo</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_looping_system_video(path: Path, title: str, description: str) -> None:
    st.markdown('<div class="system-asset-block">', unsafe_allow_html=True)
    st.subheader(title)
    st.write(description)
    if not path.exists():
        st.info(f"Expected video asset not found: `{path}`")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    mime_type = VIDEO_MIME_TYPES.get(path.suffix.lower(), "video/mp4")
    encoded_video = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div class="system-arch-video-frame">
            <video class="system-arch-video" width="1692" height="1852" autoplay muted loop playsinline preload="auto">
                <source src="data:{mime_type};base64,{encoded_video}" type="{mime_type}">
                Your browser does not support embedded video playback.
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_system_screenshot(path: Path, title: str, description: str) -> None:
    st.markdown('<div class="system-asset-block">', unsafe_allow_html=True)
    text_col, image_col = st.columns([1, 1])
    with text_col:
        st.subheader(title)
        st.write(description)
    with image_col:
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"Expected screenshot asset not found: `{path}`")
    st.markdown("</div>", unsafe_allow_html=True)


def compact_kpi(label: str, value) -> str:
    return (
        '<div class="compact-kpi">'
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        "</div>"
    )


def champion_summary_item(label: str, value) -> str:
    return (
        "<div>"
        f'<div class="label">{escape(str(label))}</div>'
        f'<div class="value">{escape(str(value))}</div>'
        "</div>"
    )


def summary_panel_from_markdown(markdown_text: str) -> str:
    title = "Summary"
    blocks = []
    current_item = None
    current_note = None

    def render_inline_markdown(text: str) -> str:
        html = escape(text)
        html = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
            html,
        )
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", html)
        return html

    def flush_item() -> None:
        nonlocal current_item
        if current_item:
            blocks.append(("item", current_item))
            current_item = None

    def flush_note() -> None:
        nonlocal current_note
        if current_note:
            blocks.append(("note", current_note))
            current_note = None

    for raw_line in markdown_text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            flush_item()
            flush_note()
            continue
        if line.startswith("### "):
            flush_item()
            flush_note()
            title = line.replace("### ", "", 1).strip()
        elif line.startswith("- "):
            flush_item()
            flush_note()
            current_item = line.replace("- ", "", 1).strip()
        elif current_item:
            current_item += " " + line
        else:
            current_note = f"{current_note} {line}" if current_note else line
    flush_item()
    flush_note()

    body_parts = []
    in_list = False
    for block_type, text in blocks:
        if block_type == "item":
            if not in_list:
                body_parts.append("<ul>")
                in_list = True
            body_parts.append(f"<li>{render_inline_markdown(text)}</li>")
        else:
            if in_list:
                body_parts.append("</ul>")
                in_list = False
            body_parts.append(f'<p class="summary-note">{render_inline_markdown(text)}</p>')
    if in_list:
        body_parts.append("</ul>")

    if " | " in title:
        title_label, title_context = title.split(" | ", 1)
        title_html = (
            '<div class="summary-title-inline">'
            f'<span class="summary-title-label">{escape(title_label)}</span>'
            '<span class="summary-title-divider">|</span>'
            f'<span class="summary-title-context">{escape(title_context)}</span>'
            "</div>"
        )
    else:
        title_html = f'<div class="summary-title">{escape(title)}</div>'

    return (
        '<div class="champion-summary">'
        f"{title_html}"
        f"{''.join(body_parts)}"
        "</div>"
    )


def render_dashboard_header(
    champion: dict,
    forecast_paths: pd.DataFrame,
    experiment_manifest: dict,
) -> None:
    return None


def render_champion_snapshot(
    champion: dict,
    forecast_paths: pd.DataFrame,
    experiment_manifest: dict,
) -> None:
    horizon = manifest_value(experiment_manifest, "horizon", champion.get("horizon", "-"))
    cadence = months_label(
        manifest_value(
            experiment_manifest,
            "as_of_frequency_months",
            month_interval_label(forecast_paths["as_of_date"]),
        )
    )
    summary_items = [
        ("Champion", champion.get("model_type", "-")),
        ("Feature family", champion.get("feature_family_name", "-")),
        (
            "Feature transform",
            FEATURE_TRANSFORM_LABELS.get(
                str(champion.get("feature_transform", "identity")),
                str(champion.get("feature_transform", "identity")).replace("_", " ").title(),
            ),
        ),
        ("Balanced score", format_int(champion.get("selection_score_balanced", champion.get("selection_score")))),
        ("MAE / RMSE", f"{format_int(champion.get('mae'))} / {format_int(champion.get('rmse'))}"),
        ("Target dates", date_range_label(forecast_paths, "target_date")),
        ("Horizon / cadence", f"{horizon} months / {cadence}"),
    ]
    st.markdown(
        """
        <div class="champion-summary">
            <div class="summary-title">Current Champion Snapshot</div>
            <div class="champion-summary-grid">
        """
        + "".join(champion_summary_item(label, value) for label, value in summary_items)
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experiment_summary(
    champion: dict,
    forecast_paths: pd.DataFrame,
    experiment_manifest: dict,
) -> None:
    items = [
        ("Champion", champion.get("model_type", "-")),
        ("Feature family", champion.get("feature_family_name", "-")),
        (
            "Feature transform",
            FEATURE_TRANSFORM_LABELS.get(
                str(champion.get("feature_transform", "identity")),
                str(champion.get("feature_transform", "identity")).replace("_", " ").title(),
            ),
        ),
        ("Mode", champion.get("mode", "-")),
        ("Balanced score", format_int(champion.get("selection_score_balanced", champion.get("selection_score")))),
        ("Champion MAE", format_int(champion.get("mae"))),
        ("Champion RMSE", format_int(champion.get("rmse"))),
        ("Predictions", format_int(len(forecast_paths))),
        ("Target dates", date_range_label(forecast_paths, "target_date")),
        ("As-of dates", date_range_label(forecast_paths, "as_of_date")),
        (
            "Horizon / cadence",
            (
                f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', '-'))} months"
                f" / {months_label(manifest_value(experiment_manifest, 'as_of_frequency_months', month_interval_label(forecast_paths['as_of_date'])))}"
            ),
        ),
    ]
    st.markdown(
        '<div class="compact-kpi-grid">' + "".join(compact_kpi(label, value) for label, value in items) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="compact-context">Each as-of month trains only on data before that month, '
        "then forecasts the target horizon ahead. Aggregated metrics compare those rolling "
        "historical forecasts with actual outcomes.</div>",
        unsafe_allow_html=True,
    )


def display_model_family_name(value: str) -> str:
    labels = {
        "baseline": "Baseline",
        "linear": "Linear Models",
        "autoregressive": "Autoregressive Models",
        "tree": "Tree-Based Models",
        "neural_net": "Neural Nets",
        "neural": "Neural Nets",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def display_model_family_prefix(value: str) -> str:
    labels = {
        "baseline": "Baseline",
        "linear": "Linear",
        "autoregressive": "Autoregressive",
        "tree": "Tree",
        "neural_net": "Neural Net",
        "neural": "Neural Net",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def display_model_build_name(value: str) -> str:
    labels = {
        "seasonal_naive": "Seasonal naive",
        "elastic_net": "Elastic net",
        "random_forest": "Random forest",
        "extra_trees": "Extra trees",
        "xgboost": "XGBoost",
        "arima": "ARIMA",
        "sarima": "SARIMA",
        "sarimax": "SARIMAX",
        "gru": "GRU",
        "lstm": "LSTM",
        "mlp": "MLP",
        "cnn": "CNN",
        "rnn": "RNN",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def model_build_display_label(model_family, model_build) -> str:
    return f"{display_model_family_prefix(model_family)}: {display_model_build_name(model_build)}"


def configurations_label(value) -> str:
    count = format_int(value)
    noun = "configuration" if pd.notna(value) and int(value) == 1 else "configurations"
    return f"{count} {noun}"


def render_model_scope_summary(model_scope: pd.DataFrame) -> None:
    st.markdown(model_scope_summary_html(model_scope), unsafe_allow_html=True)


def model_scope_summary_html(model_scope: pd.DataFrame) -> str:
    rows = []
    for family, group in model_scope.groupby("model_family", sort=False):
        build_parts = [
            f"{display_model_build_name(row.model_build)} ({configurations_label(row.configurations)})"
            for row in group.itertuples(index=False)
        ]
        rows.append(
            "<p>"
            f"<strong>{escape(display_model_family_name(family))}:</strong> "
            f"{escape(', '.join(build_parts))}"
            "</p>"
        )
    return "".join(rows)


def experiment_overview_with_regime_note() -> str:
    marker = "### Current Experiment Blocks"
    if marker not in EXPERIMENT_OVERVIEW:
        return EXPERIMENT_OVERVIEW + "\n\n" + DATA_AS_OF_REGIME_FEATURES
    return EXPERIMENT_OVERVIEW.replace(marker, DATA_AS_OF_REGIME_FEATURES + "\n\n" + marker, 1)


def configured_integrated_base_path() -> Path:
    return Path(os.environ.get("INTEGRATED_BASE_PATH", DEFAULT_INTEGRATED_BASE_PATH))


def configured_feature_table_path() -> Path:
    return Path(os.environ.get("FEATURE_TABLE_PATH", DEFAULT_FEATURE_TABLE_PATH))


def configured_imputation_log_path() -> Path:
    return Path(os.environ.get("IMPUTATION_LOG_PATH", DEFAULT_IMPUTATION_LOG_PATH))


@st.cache_data(show_spinner=False)
def load_integrated_base(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_feature_table(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_imputation_log(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


def integrated_source_series_options() -> list[dict[str, str]]:
    return [
        {
            "column": "upt",
            "label": "UPT",
            "description": "Unlinked passenger trips",
            "unit": "Passenger boardings",
            "source": "FTA NTD",
        },
        {
            "column": "vrm",
            "label": "VRM",
            "description": "Vehicle revenue miles",
            "unit": "Miles",
            "source": "FTA NTD",
        },
        {
            "column": "vrh",
            "label": "VRH",
            "description": "Vehicle revenue hours",
            "unit": "Hours",
            "source": "FTA NTD",
        },
        {
            "column": "voms",
            "label": "VOMS",
            "description": "Vehicles operated in maximum service",
            "unit": "Vehicles",
            "source": "FTA NTD",
        },
        {
            "column": "seattle_gas_price_avg",
            "label": "Gas Avg",
            "description": "Seattle gasoline price average",
            "unit": "Dollars per gallon",
            "source": "EIA",
        },
        {
            "column": "seattle_gas_price_std",
            "label": "Gas Std",
            "description": "Within-month Seattle gasoline price standard deviation",
            "unit": "Dollars per gallon",
            "source": "EIA",
        },
        {
            "column": "cpi_all_items_sa",
            "label": "CPI All",
            "description": "CPI all items, seasonally adjusted",
            "unit": "Index",
            "source": "FRED",
        },
        {
            "column": "cpi_core_sa",
            "label": "CPI Core",
            "description": "CPI all items less food and energy, seasonally adjusted",
            "unit": "Index",
            "source": "FRED",
        },
        {
            "column": "king_county_median_household_income_prior_year",
            "label": "Income",
            "description": "King County median household income, prior-year context",
            "unit": "Dollars",
            "source": "FRED",
        },
    ]


def integrated_source_series_data() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame(), []

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    if "date" not in df.columns:
        return pd.DataFrame(), []

    options = [option for option in integrated_source_series_options() if option["column"] in df.columns]
    if not options:
        return pd.DataFrame(), []

    cols = ["date", *[option["column"] for option in options]]
    data = (
        df[cols]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .dropna(subset=["date"])
    )
    return data, options


def source_series_figure(data: pd.DataFrame, option: dict[str, str]) -> go.Figure:
    column = option["column"]
    plot_df = data[["date", column]].dropna().copy()
    fig = go.Figure()
    if plot_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df[column],
            mode="lines",
            name=option["label"],
            line={"color": "#007f68", "width": 2.5},
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "%{x|%b %Y}<br>"
                "%{y:,.2f}<extra></extra>"
            ),
        )
    )
    covid_marker = pd.Timestamp("2020-03-01")
    fig.add_shape(
        type="line",
        x0=covid_marker,
        x1=covid_marker,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "rgba(47,50,58,0.45)", "dash": "dash", "width": 1.2},
    )
    fig.add_annotation(
        x=covid_marker,
        y=1,
        xref="x",
        yref="paper",
        text="COVID",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "rgba(47,50,58,0.75)"},
    )
    fig.update_layout(
        title=f"{option['label']}: {option['description']}",
        height=360,
        margin={"l": 70, "r": 25, "t": 56, "b": 48},
        xaxis_title="Month",
        yaxis_title=option["unit"],
        template="plotly_white",
    )
    return fig


def format_month(value) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%b %Y")


def date_spine_missing_count(df: pd.DataFrame) -> int:
    if df.empty or "date" not in df:
        return 0
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.to_period("M").dt.to_timestamp()
    if dates.empty:
        return 0
    expected = pd.date_range(dates.min(), dates.max(), freq="MS")
    return int(len(expected.difference(pd.DatetimeIndex(dates))))


def availability_row(
    df: pd.DataFrame,
    column: str,
    label: str,
    source: str,
    stage: str,
    note: str = "",
) -> dict:
    if df.empty or "date" not in df or column not in df:
        return {
            "Series": label,
            "Source": source,
            "Stage": stage,
            "First available": "-",
            "Last available": "-",
            "Observed months": 0,
            "Missing months": "-",
            "Missing %": "-",
            "Note": note or "Column not present in this artifact.",
        }

    dates = pd.to_datetime(df["date"], errors="coerce")
    values = df[column]
    observed_dates = dates[values.notna()]
    observed = int(values.notna().sum())
    missing = int(values.isna().sum())
    total = int(len(values))
    return {
        "Series": label,
        "Source": source,
        "Stage": stage,
        "First available": format_month(observed_dates.min()) if observed else "-",
        "Last available": format_month(observed_dates.max()) if observed else "-",
        "Observed months": observed,
        "Missing months": missing,
        "Missing %": f"{(missing / total * 100):.1f}%" if total else "-",
        "Note": note,
    }


def data_availability_report_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    integrated_path = configured_integrated_base_path()
    feature_path = configured_feature_table_path()
    imputation_log_path = configured_imputation_log_path()

    integrated = pd.DataFrame()
    feature_table = pd.DataFrame()
    imputation_log = pd.DataFrame()
    if integrated_path.exists():
        integrated = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
        if "date" in integrated:
            integrated["date"] = pd.to_datetime(integrated["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if feature_path.exists():
        feature_table = load_feature_table(str(feature_path), file_modified_ns(feature_path)).copy()
        if "date" in feature_table:
            feature_table["date"] = pd.to_datetime(feature_table["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if imputation_log_path.exists():
        imputation_log = load_imputation_log(str(imputation_log_path), file_modified_ns(imputation_log_path)).copy()

    availability_specs = [
        ("upt", "UPT", "FTA NTD", "Integrated base", "Transit ridership is complete across the integrated monthly spine."),
        ("vrm", "VRM", "FTA NTD", "Integrated base", "Service miles are complete across the integrated monthly spine."),
        ("vrh", "VRH", "FTA NTD", "Integrated base", "Service hours are complete across the integrated monthly spine."),
        ("voms", "VOMS", "FTA NTD", "Integrated base", "Peak vehicles are complete across the integrated monthly spine."),
        ("seattle_gas_price_avg", "Seattle gas price avg", "EIA", "Integrated base", "Gas data begins in May 2003; earlier months remain unavailable in the raw integrated base."),
        ("seattle_gas_price_std", "Seattle gas price std", "EIA", "Integrated base", "Within-month gas price variability begins with the gas source series in May 2003."),
        ("cpi_all_items_sa", "CPI all items", "FRED", "Integrated base", "The local integrated-base copy has one CPI gap, while the model-ready feature table is complete."),
        ("cpi_core_sa", "CPI core", "FRED", "Integrated base", "The local integrated-base copy has one CPI gap, while the model-ready feature table is complete."),
        ("king_county_median_household_income_prior_year", "King County income", "FRED", "Feature table", "Annual income is converted to prior-year monthly context for modeling."),
    ]
    availability_rows = []
    for column, label, source, stage, note in availability_specs:
        artifact = feature_table if stage == "Feature table" else integrated
        availability_rows.append(availability_row(artifact, column, label, source, stage, note))
    availability = pd.DataFrame(availability_rows)

    readiness_rows = []
    if not integrated.empty and "date" in integrated:
        readiness_rows.append(
            {
                "Check": "Integrated monthly date spine",
                "Result": f"{format_month(integrated['date'].min())} to {format_month(integrated['date'].max())}",
                "Detail": f"{len(integrated):,} rows; {date_spine_missing_count(integrated):,} missing calendar months.",
            }
        )
    if not feature_table.empty and "date" in feature_table:
        target_missing = int(feature_table["upt_target_h3"].isna().sum()) if "upt_target_h3" in feature_table else 0
        readiness_rows.append(
            {
                "Check": "Model-ready feature table",
                "Result": f"{format_month(feature_table['date'].min())} to {format_month(feature_table['date'].max())}",
                "Detail": (
                    f"{len(feature_table):,} rows; starts later because gas availability and lag/rolling features "
                    "require historical lookback."
                ),
            }
        )
        readiness_rows.append(
            {
                "Check": "H3 target availability",
                "Result": f"{target_missing:,} missing target rows",
                "Detail": "The final three as-of rows naturally lack observed future UPT targets.",
            }
        )
        base_cols = [
            "upt",
            "vrm",
            "vrh",
            "voms",
            "seattle_gas_price_avg",
            "seattle_gas_price_std",
            "cpi_all_items_sa",
            "cpi_core_sa",
            "king_county_median_household_income_prior_year",
        ]
        available_base_cols = [col for col in base_cols if col in feature_table]
        base_missing = int(feature_table[available_base_cols].isna().sum().sum()) if available_base_cols else 0
        readiness_rows.append(
            {
                "Check": "Base source values in feature table",
                "Result": f"{base_missing:,} missing values",
                "Detail": "The modeling input rows have complete base source values after trimming/preparation.",
            }
        )
    readiness = pd.DataFrame(readiness_rows)

    imputation_cols = [
        col
        for col in feature_table.columns
        if "imputed" in str(col) or str(col).endswith("_was_imputed")
    ] if not feature_table.empty else []
    imputation_rows = []
    for col in imputation_cols:
        active = int(pd.to_numeric(feature_table[col], errors="coerce").fillna(0).sum())
        imputation_rows.append(
            {
                "Imputation flag": col,
                "Active rows": active,
                "Active %": f"{(active / len(feature_table) * 100):.1f}%" if len(feature_table) else "-",
            }
        )
    imputation_summary = pd.DataFrame(imputation_rows)
    if not imputation_summary.empty:
        imputation_summary = imputation_summary.sort_values(["Active rows", "Imputation flag"], ascending=[False, True])

    metadata = {
        "integrated_exists": integrated_path.exists(),
        "feature_table_exists": feature_path.exists(),
        "imputation_log_exists": imputation_log_path.exists(),
        "imputation_log_rows": int(len(imputation_log)),
    }
    return availability, readiness, imputation_summary, metadata


def render_data_availability_report() -> None:
    availability, readiness, imputation_summary, metadata = data_availability_report_tables()
    st.markdown("### Data Availability, Missingness, And Imputation")
    st.write(
        "This report summarizes the joined source data before modeling and the "
        "model-ready feature table after trimming, lag construction, and source "
        "preparation. The distinction matters: early source gaps can exist in the "
        "integrated base even when the final modeling rows are complete."
    )
    if availability.empty and readiness.empty:
        st.info("Source availability artifacts were not found in this environment.")
        return

    if not readiness.empty:
        st.markdown("**Pipeline readiness checks**")
        st.dataframe(readiness, use_container_width=True, hide_index=True)

    if not availability.empty:
        st.markdown("**Source series availability**")
        st.dataframe(availability, use_container_width=True, hide_index=True)

    st.markdown("**Imputation activity**")
    if imputation_summary.empty:
        st.write(
            "No imputation indicator columns were found in the feature table. "
            "That usually means this artifact was produced before imputation flags were added."
        )
    else:
        active_total = int(imputation_summary["Active rows"].sum())
        st.write(
            f"The feature table includes imputation indicators, but this run has "
            f"{active_total:,} active imputation-flag rows. The imputation log contains "
            f"{metadata.get('imputation_log_rows', 0):,} row(s)."
        )
        st.dataframe(imputation_summary, use_container_width=True, hide_index=True)
    st.caption(
        "Imputation is designed for inside-window interpolation and trailing trend fills on selected "
        "monthly exogenous series. In the current modeling artifact, the selected date window and "
        "available source files leave those flags inactive."
    )


def feature_family_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def feature_family_count_frame(feature_families: dict) -> pd.DataFrame:
    if not feature_families:
        return pd.DataFrame()

    feature_path = configured_feature_table_path()
    available_columns = None
    if feature_path.exists():
        feature_table = load_feature_table(str(feature_path), file_modified_ns(feature_path))
        available_columns = set(feature_table.columns)

    rows = []
    for family_name, features in feature_families.items():
        feature_list = [str(feature) for feature in features]
        requested = len(feature_list)
        if available_columns is None:
            available = requested
        else:
            available = sum(feature in available_columns for feature in feature_list)
        rows.append(
            {
                "feature_family_name": family_name,
                "Feature family": feature_family_label(family_name),
                "Requested features": requested,
                "Available features": available,
                "Missing features": max(requested - available, 0),
            }
        )
    return pd.DataFrame(rows).sort_values("Available features", ascending=False)


def feature_family_count_figure(counts: pd.DataFrame) -> go.Figure:
    plot_df = counts.sort_values("Available features", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=plot_df["Feature family"],
            x=plot_df["Available features"],
            orientation="h",
            name="Available in feature table",
            marker={"color": "#007f68"},
            text=plot_df["Available features"],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Available features: %{x:,}<br>"
                "Requested features: %{customdata[0]:,}<br>"
                "Missing features: %{customdata[1]:,}<extra></extra>"
            ),
            customdata=plot_df[["Requested features", "Missing features"]],
        )
    )
    if plot_df["Missing features"].sum() > 0:
        fig.add_trace(
            go.Bar(
                y=plot_df["Feature family"],
                x=plot_df["Missing features"],
                orientation="h",
                name="Requested but unavailable",
                marker={"color": "rgba(47, 50, 58, 0.25)"},
                hovertemplate="<b>%{y}</b><br>Missing features: %{x:,}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Feature Count By Family",
        barmode="stack",
        height=max(460, 24 * len(plot_df) + 120),
        margin={"l": 220, "r": 40, "t": 70, "b": 50},
        xaxis_title="Feature count",
        yaxis_title="",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#2f323a"},
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    fig.update_xaxes(gridcolor="rgba(47, 50, 58, 0.12)")
    fig.update_yaxes(showgrid=False)
    return fig


def render_feature_family_sections(family_summary: pd.DataFrame, feature_families: dict) -> None:
    st.markdown("### Feature Family Examples")
    if {"feature_family_name", "best_selection_score"}.issubset(family_summary.columns):
        family_display = family_summary.copy()
        score_column = (
            "best_selection_score_balanced"
            if "best_selection_score_balanced" in family_display.columns
            else "best_selection_score"
        )
        family_display = family_display.sort_values(score_column).head(20)
        family_columns = [
            "feature_family_name",
            "mode",
            "best_selection_score_balanced",
            "best_rmse",
            "best_mae",
            "best_r2",
            "best_diracc",
        ]
        family_columns = [column for column in family_columns if column in family_display.columns]
        curated_examples = [
            ("history_regime_time", "raw"),
            ("history_regime_time_linear_interactions", "residual"),
            ("history_regime_income", None),
            ("history_regime_cpi", None),
        ]
        curated_rows = []
        for family_name, mode in curated_examples:
            row_candidates = family_display[family_display["feature_family_name"] == family_name]
            if mode is not None and "mode" in row_candidates:
                row_candidates = row_candidates[row_candidates["mode"].astype(str) == mode]
            if not row_candidates.empty:
                curated_rows.append(row_candidates.iloc[[0]])
        if curated_rows:
            family_display = pd.concat(curated_rows, ignore_index=True)
        family_display = family_display[family_columns]
        st.dataframe(family_display, use_container_width=True, hide_index=True)
    else:
        family_cols = [col for col in ["feature_family_name", "mode"] if col in family_summary]
        st.dataframe(family_summary[family_cols].drop_duplicates(), use_container_width=True, hide_index=True)

    st.markdown("### Feature Count By Family")
    st.write(
        "Feature families are intentionally uneven: some are compact baselines, "
        "while others include rolling history, regime context, external economic "
        "signals, or targeted interaction terms. The count below shows how much "
        "candidate signal each named family brings into the model-selection stage."
    )
    count_frame = feature_family_count_frame(feature_families)
    if count_frame.empty:
        st.info("Feature family definitions were not found, so the count chart cannot be rendered.")
    else:
        st.plotly_chart(feature_family_count_figure(count_frame), use_container_width=True)

    st.markdown("### Feature Family Definitions")
    st.write(
        "These are the named feature families available to the Phase A tabular "
        "models. A feature family is the human-readable modeling strategy before "
        "any model-specific feature policy such as correlation pruning, mutual "
        "information selection, or tree-importance selection is applied. Use the "
        "inspector below to see what a selected family includes."
    )
    if feature_families:
        selected_family = st.selectbox(
            "Inspect one feature family",
            sorted(feature_families),
            index=0,
            key="experiment_feature_family_definition_select",
        )
        selected_features = [str(feature) for feature in feature_families[selected_family]]
        feature_count = len(selected_features)
        feature_word = "feature" if feature_count == 1 else "features"
        st.caption(f"{feature_count} {feature_word} included")
        feature_cols = st.columns(4)
        for index, feature_name in enumerate(selected_features):
            feature_cols[index % len(feature_cols)].markdown(f"- `{feature_name}`")

        with st.expander("Show the feature family definition JSON"):
            st.code(json.dumps(feature_families, indent=2), language="json")
    else:
        st.info(
            "Feature family definitions were not found in this environment. "
            "For local runs, the dashboard looks for "
            f"`{DEFAULT_FEATURE_FAMILIES_PATH}` or the `FEATURE_FAMILIES_PATH` environment variable."
        )

    st.markdown("### Feature Types")
    feature_type_rows = [
        {
            "Feature type": "Lagged ridership",
            "Examples": "upt_lag1, upt_lag3, upt_lag12",
            "Purpose": "Give models recent momentum and same-month-last-year context.",
        },
        {
            "Feature type": "Rolling history",
            "Examples": "rolling means, rolling changes, recent trend summaries",
            "Purpose": "Smooth noisy month-to-month movement and expose trajectory.",
        },
        {
            "Feature type": "Time and seasonality",
            "Examples": "time_index_months, month_sin, month_cos, target_month_sin",
            "Purpose": "Represent long-run trend and recurring calendar pattern.",
        },
        {
            "Feature type": "Regime indicators",
            "Examples": "pandemic_observed, pandemic_disruption_active, months_since_pandemic_observed",
            "Purpose": "Let models distinguish ordinary history from known disruption and recovery periods.",
        },
        {
            "Feature type": "Exogenous context",
            "Examples": "gas price, CPI, core CPI, income growth",
            "Purpose": "Test whether external economic pressure improves forecasts.",
        },
        {
            "Feature type": "Targeted interactions",
            "Examples": "income_yoy_pct_x_gas_price_yoy_diff, lag_x_pandemic flags",
            "Purpose": "Let linear models express selected non-additive relationships without a full polynomial explosion.",
        },
    ]
    st.dataframe(pd.DataFrame(feature_type_rows), use_container_width=True, hide_index=True)


def pre_covid_mom_callouts(threshold: float = 2.0) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    callout_cols = [c for c in ["upt", "vrm", "vrh", "seattle_gas_price_avg"] if c in df.columns]
    if not callout_cols:
        return pd.DataFrame()

    pre_covid = (
        df.loc[df["date"] <= pd.Timestamp("2019-12-01"), ["date", *callout_cols]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    mom_pct = pre_covid.set_index("date")[callout_cols].pct_change(fill_method=None) * 100
    mom_pct = mom_pct.replace([float("inf"), float("-inf")], pd.NA)
    mom_pct["month_num"] = mom_pct.index.month
    avg_mom_pct_by_month = mom_pct.groupby("month_num")[callout_cols].mean().reindex(range(1, 13))

    label_map = {
        "upt": "UPT",
        "vrm": "VRM",
        "vrh": "VRH",
        "seattle_gas_price_avg": "Gas",
    }
    series_order = {col: idx for idx, col in enumerate(callout_cols)}
    rows = []
    for month_num, row in avg_mom_pct_by_month.iterrows():
        for col in callout_cols:
            value = row[col]
            if pd.notna(value) and abs(value) > threshold:
                rows.append(
                    {
                        "month_num": month_num,
                        "month": pd.Timestamp(2000, month_num, 1).strftime("%B"),
                        "series": label_map.get(col, col),
                        "series_order": series_order[col],
                        "avg_mom_pct": float(value),
                    }
                )
    return pd.DataFrame(rows)


def pre_covid_mom_callouts_html(callouts: pd.DataFrame, threshold: float = 2.0) -> str:
    if callouts.empty:
        return ""

    month_cards = []
    for month_num, group in callouts.sort_values(["month_num", "series_order"]).groupby("month_num", sort=True):
        month_name = str(group["month"].iloc[0])
        badges = []
        for row in group.itertuples(index=False):
            direction = "positive" if row.avg_mom_pct > 0 else "negative"
            series_kind = "gas" if str(row.series).lower() == "gas" else "transit"
            badge_class = f"{series_kind}-{direction}"
            value = f"{row.avg_mom_pct:+.1f}%"
            badges.append(
                f'<div class="mom-badge {badge_class}">'
                f'<div class="series">{escape(str(row.series))}</div>'
                f'<div class="value">{escape(value)}</div>'
                "</div>"
            )
        month_cards.append(
            '<div class="mom-month-card">'
            f'<div class="mom-month-title">{escape(month_name)}</div>'
            f'<div class="mom-badge-row">{"".join(badges)}</div>'
            "</div>"
        )

    return (
        '<section class="eda-section">'
        "<h3>EDA</h3>"
        '<div class="eda-kicker">Summary of month-over-month data trends up until COVID</div>'
        '<p class="eda-context">'
        "Average pre-2020 month-over-month changes are shown when the magnitude is "
        f"greater than {threshold:.0f}%. Transit metrics use teal for increases and amber for decreases; "
        "gas uses orange for rising cost pressure and teal for declines."
        "</p>"
        f'<div class="mom-callout-grid">{"".join(month_cards)}</div>'
        "</section>"
    )


def lagged_upt_yoy_correlations(max_lag: int = 12) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    needed_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in needed_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    yoy = (
        df[["date", *available_cols]]
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
    )

    label_map = {
        "cpi_all_items_sa": "CPI all items",
        "cpi_core_sa": "CPI core",
        "seattle_gas_price_avg": "Seattle gas price",
        "voms": "VOMS",
        "vrh": "VRH",
        "vrm": "VRM",
    }
    rows = []
    for predictor in [col for col in available_cols if col != "upt"]:
        for lag in range(max_lag + 1):
            aligned = pd.concat(
                {
                    "upt_yoy": yoy["upt"],
                    "predictor_yoy_lagged": yoy[predictor].shift(lag),
                },
                axis=1,
            ).dropna()
            if len(aligned) > 20:
                rows.append(
                    {
                        "predictor": predictor,
                        "series": label_map.get(predictor, predictor),
                        "lag_months": lag,
                        "correlation": float(aligned["upt_yoy"].corr(aligned["predictor_yoy_lagged"])),
                        "n": len(aligned),
                    }
                )
    return pd.DataFrame(rows)


def lagged_correlation_figure(lagged_corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if lagged_corr.empty:
        return fig

    color_map = {
        "CPI all items": "#075fed",
        "CPI core": "#4f8ad9",
        "Seattle gas price": "#c85214",
        "VOMS": "#7c3aed",
        "VRH": "#007f68",
        "VRM": "#6b7280",
    }
    for series, group in lagged_corr.groupby("series", sort=False):
        group = group.sort_values("lag_months")
        fig.add_trace(
            go.Scatter(
                x=group["lag_months"],
                y=group["correlation"],
                mode="lines+markers",
                name=series,
                line={"color": color_map.get(series), "width": 2.2},
                marker={"size": 7},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Lag: %{x} months<br>"
                    "Correlation: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_color="rgba(47,50,58,0.45)", line_width=1)
    fig.update_layout(
        title="Lagged Correlation With UPT YoY Change",
        height=430,
        margin={"l": 55, "r": 20, "t": 54, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        xaxis_title="Predictor lag in months",
        yaxis_title="Correlation with UPT YoY",
        yaxis={"range": [-0.25, 0.65], "zeroline": False},
        template="plotly_white",
    )
    return fig


def granger_predictive_screening(max_lag: int = 6) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    try:
        import warnings

        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    yoy = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
    )
    label_map = {
        "vrm": "VRM",
        "vrh": "VRH",
        "voms": "VOMS",
        "seattle_gas_price_avg": "Gas price",
        "cpi_all_items_sa": "CPI all",
        "cpi_core_sa": "CPI core",
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for predictor in [col for col in available_cols if col != "upt"]:
            test_df = yoy[["upt", predictor]].dropna()
            if len(test_df) < 40:
                continue
            try:
                result = grangercausalitytests(test_df[["upt", predictor]], maxlag=max_lag, verbose=False)
                pvals = [float(result[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)]
                best_lag = int(np.argmin(pvals) + 1)
                rows.append(
                    {
                        "predictor": predictor,
                        "series": label_map.get(predictor, predictor),
                        "best_lag": best_lag,
                        "min_p_value": min(pvals),
                        "pvals_by_lag": ", ".join(f"{p:.3f}" for p in pvals),
                        "n": len(test_df),
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("min_p_value")


def granger_predictive_figure(screening: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if screening.empty:
        return fig

    plot_df = screening.copy()
    plot_df["p_for_plot"] = plot_df["min_p_value"].clip(lower=1e-6)
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_for_plot"])
    plot_df = plot_df.sort_values("neg_log10_p", ascending=True)
    threshold = -np.log10(0.05)
    colors = np.where(plot_df["min_p_value"] < 0.05, "#007f68", "rgba(107, 114, 128, 0.55)")

    fig.add_trace(
        go.Bar(
            x=plot_df["neg_log10_p"],
            y=plot_df["series"],
            orientation="h",
            marker={"color": colors},
            text=[f"p={p:.3f}, lag={lag}" for p, lag in zip(plot_df["min_p_value"], plot_df["best_lag"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "-log10(p): %{x:.2f}<br>"
                "%{text}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="rgba(47,50,58,0.55)",
        annotation_text="p = 0.05",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Granger-Style UPT YoY Predictive Screening",
        height=340,
        margin={"l": 70, "r": 95, "t": 56, "b": 52},
        xaxis_title="-log10(minimum p-value across 1-6 month lags)",
        yaxis_title="",
        template="plotly_white",
    )
    return fig


def covid_break_diagnostics() -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    try:
        import statsmodels.api as sm
        from scipy import stats
    except ImportError:
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    yoy = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
        .rename(columns={col: f"{col}_yoy" for col in available_cols})
    )
    model_df = yoy.copy()
    model_df["target_date"] = model_df.index + pd.DateOffset(months=3)
    model_df["upt_yoy_target_h3"] = model_df["upt_yoy"].shift(-3)
    model_df["target_post_covid"] = (model_df["target_date"] >= pd.Timestamp("2020-03-01")).astype(int)
    month_dummies = pd.get_dummies(
        model_df["target_date"].dt.month,
        prefix="target_month",
        drop_first=True,
        dtype=float,
    )
    month_dummies.index = model_df.index
    regression_df = pd.concat([model_df, month_dummies], axis=1).dropna()

    regression_terms = [
        "upt_yoy",
        "vrm_yoy",
        "vrh_yoy",
        "voms_yoy",
        "seattle_gas_price_avg_yoy",
        "cpi_all_items_sa_yoy",
        "cpi_core_sa_yoy",
        "target_post_covid",
        *list(month_dummies.columns),
    ]
    regression_terms = [term for term in regression_terms if term in regression_df.columns]
    if len(regression_df) < len(regression_terms) * 3:
        return pd.DataFrame()

    break_date = pd.Timestamp("2020-03-01")

    def regression_ssr(input_df: pd.DataFrame) -> tuple[float, int, int]:
        local_y = input_df["upt_yoy_target_h3"]
        local_x = sm.add_constant(input_df[regression_terms])
        fit = sm.OLS(local_y, local_x).fit()
        return float((fit.resid**2).sum()), len(input_df), local_x.shape[1]

    pooled_ssr, _, pooled_k = regression_ssr(regression_df)
    pre_break_df = regression_df.loc[regression_df["target_date"] < break_date]
    post_break_df = regression_df.loc[regression_df["target_date"] >= break_date]
    if len(pre_break_df) <= pooled_k or len(post_break_df) <= pooled_k:
        return pd.DataFrame()

    pre_ssr, pre_n, _ = regression_ssr(pre_break_df)
    post_ssr, post_n, _ = regression_ssr(post_break_df)
    denominator_df = pre_n + post_n - 2 * pooled_k
    if denominator_df <= 0:
        return pd.DataFrame()

    chow_f = ((pooled_ssr - (pre_ssr + post_ssr)) / pooled_k) / (
        (pre_ssr + post_ssr) / denominator_df
    )
    chow_p = 1 - stats.f.cdf(chow_f, pooled_k, denominator_df)
    mean_test = stats.ttest_ind(
        pre_break_df["upt_yoy_target_h3"],
        post_break_df["upt_yoy_target_h3"],
        equal_var=False,
    )
    return pd.DataFrame(
        [
            {
                "test": "Coefficient stability break",
                "statistic": float(chow_f),
                "p_value": float(chow_p),
                "pre_n": int(pre_n),
                "post_n": int(post_n),
                "interpretation": "Tests whether the H3 regression relationship is stable before and after COVID.",
            },
            {
                "test": "Mean UPT YoY difference",
                "statistic": float(mean_test.statistic),
                "p_value": float(mean_test.pvalue),
                "pre_n": int(pre_n),
                "post_n": int(post_n),
                "interpretation": "Tests whether average target UPT YoY differs across periods.",
            },
        ]
    )


def covid_break_figure(break_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if break_summary.empty:
        return fig

    plot_df = break_summary.copy()
    plot_df["p_for_plot"] = plot_df["p_value"].clip(lower=1e-16)
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_for_plot"])
    colors = np.where(plot_df["p_value"] < 0.05, "#c85214", "rgba(107, 114, 128, 0.55)")
    fig.add_trace(
        go.Bar(
            x=plot_df["test"],
            y=plot_df["neg_log10_p"],
            marker={"color": colors},
            text=[f"p={p:.3g}" for p in plot_df["p_value"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>-log10(p): %{y:.2f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_hline(
        y=-np.log10(0.05),
        line_dash="dash",
        line_color="rgba(47,50,58,0.55)",
        annotation_text="p = 0.05",
        annotation_position="top right",
    )
    fig.update_layout(
        title="COVID Break Diagnostics For H3 UPT YoY Regression",
        height=330,
        margin={"l": 55, "r": 30, "t": 56, "b": 78},
        xaxis_title="",
        yaxis_title="-log10(p-value)",
        template="plotly_white",
    )
    return fig


def trend_month_residualize_dashboard(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    valid = y.notna()
    if valid.sum() < 24:
        return pd.Series(index=series.index, data=pd.NA, name=series.name)

    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "time_index": range(len(series)),
        },
        index=series.index,
        dtype=float,
    )
    month_dummies = pd.get_dummies(series.index.month, prefix="month", drop_first=True, dtype=float)
    month_dummies.index = series.index
    design = pd.concat([design, month_dummies], axis=1)

    x = design.loc[valid].to_numpy(dtype=float)
    target = y.loc[valid].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    fitted = design.to_numpy(dtype=float) @ beta
    return pd.Series(y.to_numpy(dtype=float) - fitted, index=series.index, name=series.name)


def dashboard_correlation_matrices() -> dict[str, pd.DataFrame]:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return {}

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or len(available_cols) < 2:
        return {}

    label_map = {
        "upt": "UPT",
        "vrm": "VRM",
        "vrh": "VRH",
        "voms": "VOMS",
        "seattle_gas_price_avg": "Gas price",
        "cpi_all_items_sa": "CPI all",
        "cpi_core_sa": "CPI core",
    }
    data = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .rename(columns=label_map)
    )
    residuals = data.apply(trend_month_residualize_dashboard)
    transformed = {
        "First Differences": data.diff(),
        "YoY Percent Changes": data.pct_change(periods=12, fill_method=None).replace([float("inf"), float("-inf")], pd.NA),
        "Trend + Month Residuals": residuals,
    }
    return {name: matrix.corr(method="pearson") for name, matrix in transformed.items()}


def correlation_heatmap_figure(corr_matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": "Pearson r"},
            text=corr_matrix.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=460,
        margin={"l": 70, "r": 20, "t": 56, "b": 70},
        xaxis={"side": "bottom", "tickangle": -35},
        yaxis={"autorange": "reversed"},
        template="plotly_white",
    )
    return fig


def render_public_bundle_note(experiment_manifest: dict) -> None:
    bundle = experiment_manifest.get("public_dashboard_bundle")
    if not bundle:
        return
    retained = format_int(bundle.get("selected_configurations"))
    source = format_int(bundle.get("source_configurations"))
    full_path_rows = bundle.get("full_path_rows", {})
    full_forecast_rows = format_int(full_path_rows.get("forecast_paths")) if isinstance(full_path_rows, dict) else "-"
    keep_fraction = bundle.get("keep_fraction")
    keep_pct = f"{float(keep_fraction) * 100:.0f}%" if keep_fraction is not None else "configured"
    st.info(
        "This live dashboard uses a performance-aware public artifact bundle. "
        f"The lightweight model index can cover up to {source} source configurations, "
        f"while the flat compatibility path files retain {retained} configurations "
        f"from the best {keep_pct} of core metrics plus baseline/champion models. "
        f"When available, full path-level rows ({full_forecast_rows} forecasts) are "
        "loaded on demand from partitioned Parquet files after filters are applied."
    )


def render_system_artifact_flow(experiment_manifest: dict, flat_forecast_rows: int) -> None:
    st.markdown(SYSTEM_ARTIFACT_FLOW)

    bundle = experiment_manifest.get("public_dashboard_bundle", {})
    full_path_rows = bundle.get("full_path_rows", {}) if isinstance(bundle.get("full_path_rows"), dict) else {}
    artifact_cols = st.columns(4)
    artifact_cols[0].metric(
        "Full model index",
        format_int(bundle.get("source_configurations") or bundle.get("full_metadata_configurations")),
    )
    artifact_cols[1].metric("Curated flat configs", format_int(bundle.get("selected_configurations")))
    artifact_cols[2].metric("Full forecast rows", format_int(full_path_rows.get("forecast_paths")))
    artifact_cols[3].metric("Flat forecast rows", format_int(flat_forecast_rows))

    file_rows = [
        {
            "Layer": "Full model metadata",
            "Path / files": ["model_leaderboard_full.parquet", "complexity_profile_full.parquet"],
            "Dashboard role": "Filter and compare the full public experiment index without loading every prediction row.",
        },
        {
            "Layer": "Curated compatibility paths",
            "Path / files": ["forecast_paths.parquet", "performance_over_time.parquet"],
            "Dashboard role": "Fast initial overview/champion context and backward-compatible dashboard loading.",
        },
        {
            "Layer": "Partitioned full paths",
            "Path / files": ["forecast_paths_by_build/", "performance_over_time_by_build/"],
            "Dashboard role": "On-demand forecast and rolling-error rows after Model Explorer filters are applied.",
        },
        {
            "Layer": "Bundle manifests",
            "Path / files": [
                "experiment_manifest.json",
                "public_bundle_manifest.json",
                "path_partition_manifest.json",
            ],
            "Dashboard role": "Document source experiment IDs, curation rules, counts, and partition layout.",
        },
    ]
    rows_html = [
        """
        <div class="artifact-flow-row artifact-flow-head">
            <div class="artifact-flow-cell">Layer</div>
            <div class="artifact-flow-cell">Path / files</div>
            <div class="artifact-flow-cell">Dashboard role</div>
        </div>
        """
    ]
    for row in file_rows:
        path_html = ", ".join(f"<code>{escape(path)}</code>" for path in row["Path / files"])
        rows_html.append(
            f"""
            <div class="artifact-flow-row">
                <div class="artifact-flow-cell">{escape(row["Layer"])}</div>
                <div class="artifact-flow-cell">{path_html}</div>
                <div class="artifact-flow-cell">{escape(row["Dashboard role"])}</div>
            </div>
            """
        )
    st.markdown(
        f'<div class="artifact-flow-table">{"".join(rows_html)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The public dashboard bundle is static, but it is not limited to a tiny leaderboard. "
        "Small metadata files describe the broader model universe; larger path-level data is "
        "split by model build so the app can read only the rows needed for the current view."
    )


def render_data_page(
    family_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    champion: dict,
    feature_families: dict,
) -> None:
    data_intro_cols = st.columns(2)
    data_intro_cols[0].markdown(summary_panel_from_markdown(DATA_PRIMARY_DATA), unsafe_allow_html=True)
    data_intro_cols[1].markdown(summary_panel_from_markdown(DATA_SECONDARY_DATA), unsafe_allow_html=True)

    source_series, source_options = integrated_source_series_data()
    if source_options:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Integrated monthly source series</h4>
                <p>
                    Before feature engineering, the pipeline joins each normalized source
                    to a common monthly grain. Use these tabs to inspect one raw integrated
                    signal at a time across the full available history.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source_tabs = st.tabs([option["label"] for option in source_options])
        for tab_panel, option in zip(source_tabs, source_options):
            with tab_panel:
                st.plotly_chart(source_series_figure(source_series, option), use_container_width=True)
                start_date = source_series.loc[source_series[option["column"]].notna(), "date"].min()
                end_date = source_series.loc[source_series[option["column"]].notna(), "date"].max()
                if pd.notna(start_date) and pd.notna(end_date):
                    st.caption(
                        f"{option['source']} source signal, "
                        f"{start_date:%b %Y} through {end_date:%b %Y}."
                    )

    mom_callouts = pre_covid_mom_callouts()
    mom_callout_html = pre_covid_mom_callouts_html(mom_callouts)
    if mom_callout_html:
        st.markdown(mom_callout_html, unsafe_allow_html=True)

    lagged_corr = lagged_upt_yoy_correlations()
    if not lagged_corr.empty:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Lagged relationships after reducing shared trend</h4>
                <p>
                    Because UPT and price indexes both tend to rise over long periods,
                    raw level correlations can overstate the relationship. This view
                    compares year-over-year changes instead, then tests whether each
                    predictor's YoY movement leads UPT YoY movement by 0 to 12 months.
                    CPI shows the strongest short-lag association, but that should be
                    interpreted as a broad macro or regime signal rather than causal
                    evidence that inflation mechanically increases ridership.
                </p>
                <p>
                    Service measures are more mixed: they move with UPT at shorter lags,
                    then fade or turn negative at longer lags. That pattern is useful for
                    forecasting exploration, but still observational and potentially shaped
                    by service planning, recovery timing, and COVID-era structural change.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(lagged_correlation_figure(lagged_corr), use_container_width=True)

    corr_matrices = dashboard_correlation_matrices()
    if corr_matrices:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Correlation matrices after reducing time effects</h4>
                <p>
                    These Pearson correlation matrices compare transformed versions of
                    the integrated data rather than raw levels. First differences show
                    short-run movement, year-over-year percent changes compare growth
                    against the same month one year earlier, and trend + month residuals
                    show relationships after a simple time-trend and calendar-month
                    adjustment. This reduces the chance that shared upward trends in
                    series such as ridership and price indexes dominate the interpretation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        matrix_tabs = st.tabs(list(corr_matrices.keys()))
        for tab_panel, (title, matrix) in zip(matrix_tabs, corr_matrices.items()):
            with tab_panel:
                st.plotly_chart(correlation_heatmap_figure(matrix, title), use_container_width=True)

    granger_screening = granger_predictive_screening()
    if not granger_screening.empty:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Granger-style predictive screening</h4>
                <p>
                    This screen asks whether past values of each YoY predictor improve
                    prediction of UPT YoY beyond past UPT alone. The chart shows the
                    strongest p-value found across one- to six-month lags for each
                    predictor, so it should be read as a ranking of candidate signals,
                    not as a formal causal result.
                </p>
                <p>
                    In this pass, VRH, VOMS, and CPI measures surface as the clearest
                    predictive candidates. Gas prices and VRM are weaker in this specific
                    YoY lag test, even though they can still matter in other
                    transformations or in the full rolling forecast models.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(granger_predictive_figure(granger_screening), use_container_width=True)

    break_summary = covid_break_diagnostics()
    if not break_summary.empty:
        break_rows = break_summary.set_index("test")
        stability_p = break_rows.loc["Coefficient stability break", "p_value"]
        mean_p = break_rows.loc["Mean UPT YoY difference", "p_value"]
        st.markdown(
            f"""
            <div class="eda-chart-panel">
                <h4>COVID-era structural break diagnostics</h4>
                <p>
                    The H3 diagnostic regression predicts UPT YoY three months ahead
                    from as-of-month YoY signals, target-month calendar effects, and a
                    post-COVID indicator. A coefficient-stability test then compares
                    whether that relationship looks the same before and after March 2020.
                </p>
                <p>
                    The coefficient-stability result is extremely strong
                    (<strong>p = {stability_p:.2g}</strong>), which supports treating
                    COVID as a structural break in the modeling pipeline. The simpler
                    pre/post mean comparison is not significant
                    (<strong>p = {mean_p:.3f}</strong>), so the useful takeaway is not
                    just that ridership visibly dropped, but that the relationship
                    between ridership, service, prices, and seasonality changed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    data_feature_cols = st.columns(2)
    data_feature_cols[0].markdown(summary_panel_from_markdown(DATA_CALCULATED_FEATURES), unsafe_allow_html=True)
    data_feature_cols[1].markdown(summary_panel_from_markdown(DATA_TIME_FEATURES), unsafe_allow_html=True)

    st.markdown("### Single Forecast Step Example")
    config_id = champion.get("model_config_id") or champion.get("config_id")
    sample = forecast_paths[forecast_paths["model_config_id"] == config_id].copy()
    if sample.empty:
        sample = forecast_paths.copy()
    if not sample.empty:
        sample = sample.sort_values("target_date").iloc[len(sample) // 2]
        target_date = pd.Timestamp(sample["target_date"])
        as_of_date = pd.Timestamp(sample["as_of_date"])
        pandemic_start = pd.Timestamp("2020-03-01")
        months_since_pandemic = max(
            0,
            (as_of_date.year - pandemic_start.year) * 12 + (as_of_date.month - pandemic_start.month),
        )
        example_rows = [
            ("as_of_date", as_of_date.date().isoformat(), "Training data is limited to rows before this month."),
            ("target_date", target_date.date().isoformat(), "This is the month being forecast three months ahead."),
            ("target_month", target_date.strftime("%B"), "Seasonality features encode this month cyclically."),
            ("evaluation_period", sample.get("evaluation_period", "-"), "Used for pre-COVID, shock, recovery, and recent metrics."),
            ("months_since_pandemic_observed", str(months_since_pandemic), "A time-since-observed-disruption signal available only from the as-of month."),
            ("actual_upt", format_int(sample.get("actual")), "Observed ridership for the target month."),
            ("prediction", format_int(sample.get("prediction")), "The selected model's forecast for that target month."),
            ("seasonal_naive_prediction", format_int(sample.get("seasonal_naive_prediction")), "Same-month-last-year baseline used for comparison and residual mode."),
            ("absolute_error", format_int(sample.get("abs_error")), "Distance between prediction and observed ridership."),
        ]
        st.dataframe(
            pd.DataFrame(example_rows, columns=["Field", "Example value", "Interpretation"]),
            use_container_width=True,
            hide_index=True,
        )

    render_data_availability_report()


def render_experiment_page(
    family_summary: pd.DataFrame,
    feature_families: dict,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
    champion: dict,
    experiment_manifest: dict,
    phase_a_config_text: str,
    phase_b_config_text: str,
    phase_c_config_text: str,
) -> None:
    st.subheader("Experiment")
    render_public_bundle_note(experiment_manifest)

    summary_items = [
        ("Experiment bundle", experiment_manifest.get("experiment_id") or experiment_manifest.get("run_id") or "-"),
        (
            "Target / horizon",
            (
                f"{str(champion.get('target', experiment_manifest.get('target', 'upt'))).upper()}"
                f" / {manifest_value(experiment_manifest, 'horizon', champion.get('horizon', 3))} months"
            ),
        ),
        ("As-of window", date_range_label(forecast_paths, "as_of_date")),
        ("Target window", date_range_label(forecast_paths, "target_date")),
        ("Model configurations", format_int(len(leaderboard))),
        ("Rolling predictions", format_int(len(forecast_paths))),
        ("Metric rows", format_int(len(performance))),
    ]
    model_scope = (
        leaderboard.groupby(["model_family", "model_build"], dropna=False)
        .size()
        .reset_index(name="configurations")
    )
    model_scope = model_taxonomy_sort(model_scope)

    summary_cols = st.columns(2)
    summary_cols[0].markdown(
        """
        <div class="champion-summary">
            <div class="summary-title">Loaded Experiment Summary</div>
            <div class="champion-summary-grid">
        """
        + "".join(champion_summary_item(label, value) for label, value in summary_items)
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_cols[1].markdown(
        """
        <div class="champion-summary">
            <div class="summary-title">Model Scope In This Bundle</div>
        """
        + model_scope_summary_html(model_scope)
        + """
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_feature_family_sections(family_summary, feature_families)

    st.markdown("### Feature Policies")
    st.write(
        "Feature families define the candidate inputs for a model. Feature "
        "policies define what happens after that candidate set is chosen, such "
        "as pruning correlated columns or selecting the strongest variables "
        "inside each rolling as-of training window. Keeping these separate lets "
        "the experiment compare feature ideas and model-specific input control "
        "without mixing the two concepts."
    )
    if "feature_policy" in leaderboard.columns:
        policy_scope = (
            leaderboard.groupby("feature_policy", dropna=False)
            .size()
            .reset_index(name="configurations")
            .sort_values(["feature_policy"])
        )
        policy_scope["meaning"] = policy_scope["feature_policy"].map(
            lambda policy: FEATURE_POLICY_DESCRIPTIONS.get(
                str(policy),
                "Model-specific feature selection or pruning policy from the experiment config.",
            )
        )
        policy_scope = policy_scope.rename(
            columns={
                "feature_policy": "Feature policy",
                "configurations": "Configurations",
                "meaning": "Meaning",
            }
        )
        st.table(policy_scope)
    if "feature_transform" in leaderboard.columns:
        st.markdown("### Feature Transforms")
        st.write(
            "Feature transforms describe the mathematical representation used after "
            "a feature family and feature policy are chosen. They are kept separate "
            "from feature families: the family answers what source signals are used, "
            "while the transform answers how those selected signals are represented "
            "for the model."
        )
        transform_scope = (
            leaderboard.groupby(["feature_transform", "feature_transform_label"], dropna=False)
            .size()
            .reset_index(name="configurations")
            .sort_values(["feature_transform_label"])
        )
        transform_scope["meaning"] = transform_scope["feature_transform"].map(
            lambda transform: FEATURE_TRANSFORM_DESCRIPTIONS.get(
                str(transform),
                "Mathematical representation applied to the selected feature columns.",
            )
        )
        transform_scope = transform_scope.rename(
            columns={
                "feature_transform_label": "Feature transform",
                "configurations": "Configurations",
                "meaning": "Meaning",
            }
        )[["Feature transform", "Configurations", "Meaning"]]
        st.table(transform_scope)
        st.markdown("#### How The Transform Families Are Implemented")
        st.write(
            "In this experiment bundle, transforms are broad representation tests. "
            "A feature family first defines candidate source signals, the feature "
            "policy selects or prunes columns inside each rolling training window, "
            "and the transform family then expands those selected numeric columns. "
            "Regularized models are scaled before fitting, so ridge, lasso, and "
            "elastic net can shrink weak transformed terms, but the transform choice "
            "itself is still applied consistently across the selected variables."
        )
        transform_detail_rows = [
            {
                "Transform family": "No transform",
                "Terms created for each selected feature x": "x",
                "Interpretation": "Baseline linear representation.",
            },
            {
                "Transform family": "Signed log",
                "Terms created for each selected feature x": "x; sign(x) * log1p(abs(x))",
                "Interpretation": "Tests compressed scale effects while preserving negative values.",
            },
            {
                "Transform family": "Quadratic",
                "Terms created for each selected feature x": "x; x^2",
                "Interpretation": "Tests curvature such as diminishing or accelerating effects.",
            },
            {
                "Transform family": "Cubic",
                "Terms created for each selected feature x": "x; x^2; x^3",
                "Interpretation": "Tests asymmetric curvature; the quadratic term is included with the cubic term.",
            },
            {
                "Transform family": "Signed log + quadratic + cubic",
                "Terms created for each selected feature x": "x; sign(x) * log1p(abs(x)); x^2; x^3",
                "Interpretation": "Stress-test representation that gives regularization several nonlinear shapes to choose from.",
            },
        ]
        st.dataframe(pd.DataFrame(transform_detail_rows), use_container_width=True, hide_index=True)
        st.info(
            "The cubic transform is hierarchical in this run: a selected feature gets "
            "the original value, the quadratic term, and the cubic term. It is not a "
            "cubic-only model."
        )
        st.markdown("#### Modeling Implications")
        st.write(
            "These runs are useful for asking whether nonlinear representations are "
            "worth exploring, but they are not yet variable-specific transform recipes. "
            "For example, they do not test one model that uses a log transform for gas "
            "prices, a quadratic term for service hours, and only the original scale "
            "for CPI unless that combination happens to appear through a broader "
            "all-selected-variable transform family and regularization."
        )
        st.write(
            "A stronger next iteration would inspect transformed-variable correlations "
            "and collinearity before modeling, then create source-aware transform "
            "recipes. That would let the experiment compare targeted combinations "
            "such as log gas-price movement, quadratic service capacity, and linear "
            "inflation context against the broader all-variable transform grids."
        )
        strategy_rows = [
            {
                "Next-step idea": "Screen transformed candidates",
                "Why it helps": "Compare correlations, VIF/collinearity, and rolling stability before expanding the model grid.",
            },
            {
                "Next-step idea": "Use variable-specific transform recipes",
                "Why it helps": "Allows log for one source signal and quadratic or cubic terms for another instead of applying every transform everywhere.",
            },
            {
                "Next-step idea": "Keep hierarchy constraints",
                "Why it helps": "When x^2 or x^3 is used, keep the original x term so nonlinear coefficients remain interpretable.",
            },
            {
                "Next-step idea": "Compare targeted recipes to broad regularized grids",
                "Why it helps": "Shows whether domain-guided transforms improve forecasts beyond relying on regularization to clean up a large expansion.",
            },
        ]
        st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)
    with st.expander("How feature, representation, and complexity policies are interpreted"):
        st.markdown(REPRESENTATION_AND_COMPLEXITY_EXPLANATION)

    st.markdown("### Experiment Configs Used")
    st.write(
        "These YAML files define the model grids, feature families, feature "
        "policies, rolling forecast window, parallelization settings, MLflow "
        "tracking names, checkpoint locations, and output artifact folders used "
        "for the current dashboard bundle and the next neural follow-up."
    )
    config_rows = [
        {
            "Phase": "A v3 pandemic-safe",
            "Config file": str(PHASE_A_V3_CONFIG_PATH),
            "Role": "Baseline, regularized linear, tree, and XGBoost grids over the rebuilt pandemic-safe feature table.",
            "Loaded": "yes" if phase_a_config_text else "missing",
        },
        {
            "Phase": "B v3 pandemic-safe",
            "Config file": str(PHASE_B_V3_CONFIG_PATH),
            "Role": "ARIMA, SARIMA, and SARIMAX rerun on the same as-of-safe feature table with compact exogenous sets.",
            "Loaded": "yes" if phase_b_config_text else "missing",
        },
        {
            "Phase": "C monthly finalists",
            "Config file": str(PHASE_C_MONTHLY_CONFIG_PATH),
            "Role": "Prior GRU/LSTM finalist config; neural runs are the next block to reconcile with the v3 pandemic-safe feature table.",
            "Loaded": "yes" if phase_c_config_text else "missing",
        },
    ]
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

    if phase_a_config_text:
        with st.expander("Show Phase A v3 pandemic-safe config YAML"):
            st.code(phase_a_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_A_V3_CONFIG_PATH}`.")

    if phase_b_config_text:
        with st.expander("Show Phase B v3 pandemic-safe config YAML"):
            st.code(phase_b_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_B_V3_CONFIG_PATH}`.")

    if phase_c_config_text:
        with st.expander("Show Phase C monthly finalists config YAML"):
            st.code(phase_c_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_C_MONTHLY_CONFIG_PATH}`.")

    st.markdown(experiment_overview_with_regime_note())


def render_insights_page(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
    champion: dict,
    experiment_manifest: dict,
) -> None:
    st.subheader("Insights")
    st.write(
        "This page is the companion to Model Explorer. Model Explorer is built for "
        "interactive filtering; Insights is built for directed interpretation of "
        "what the experiment results suggest across model classes, feature choices, "
        "time periods, and forecast failure modes."
    )
    notebook_status = "available" if RESULTS_INSIGHTS_NOTEBOOK_PATH.exists() else "planned"
    st.caption(
        f"Working notebook: `{RESULTS_INSIGHTS_NOTEBOOK_PATH}` ({notebook_status}). "
        "The notebook is where exploratory result analysis can stay deeper and messier "
        "before selected findings are promoted into this dashboard page."
    )

    bundle = experiment_manifest.get("public_dashboard_bundle", {})
    full_config_count = bundle.get("source_configurations") or experiment_manifest.get("model_config_count") or len(leaderboard)
    full_path_rows = bundle.get("full_path_rows", {})
    full_forecast_rows = (
        full_path_rows.get("forecast_paths")
        if isinstance(full_path_rows, dict)
        else experiment_manifest.get("prediction_count")
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Indexed configs", format_int(len(leaderboard)))
    summary_cols[1].metric("Full source configs", format_int(full_config_count))
    summary_cols[2].metric("On-demand forecast rows", format_int(full_forecast_rows or len(forecast_paths)))
    summary_cols[3].metric("Champion", str(champion.get("model_build_label", champion.get("model_type", "-"))))

    inquiry_rows = [
        {
            "Question": "Which models are robust across periods?",
            "How this page/notebook investigates it": "Compare overall rank against pre-COVID, shock, recovery, and recent-period error.",
        },
        {
            "Question": "Do nonlinear feature transforms help?",
            "How this page/notebook investigates it": "Compare no-transform, signed-log, quadratic, cubic, and combined transforms within linear model builds.",
        },
        {
            "Question": "Which feature policies are doing useful work?",
            "How this page/notebook investigates it": "Compare none, correlation pruning, mutual information, and tree selectors across comparable model families.",
        },
        {
            "Question": "Where do forecasts break down?",
            "How this page/notebook investigates it": "Inspect residuals, large-error months, and error concentration by evaluation period.",
        },
    ]
    st.markdown("### Directed Result Questions")
    st.dataframe(pd.DataFrame(inquiry_rows), use_container_width=True, hide_index=True)

    candidate_models = exclude_baseline_candidates(leaderboard)
    score_col = "selection_score_balanced" if "selection_score_balanced" in candidate_models else "selection_score"
    if not candidate_models.empty and score_col in candidate_models:
        st.markdown("### COVID Shock Forecast Paths")
        st.write(
            "This chart fixes the Model Explorer view to one intentionally narrow "
            "question: taking the best balanced-score configuration from each model "
            "build, how did the forecast paths behave from immediately before the "
            "COVID shock through the point where several models begin moving back "
            "toward observed ridership?"
        )
        best_by_build = limit_configs_per_build(candidate_models, "Balanced score", "Top 1")
        best_by_build = limit_total_configs(best_by_build, "Balanced score", "All")
        best_configs = best_by_build["model_config_id"].astype(str).tolist()
        shock_paths = load_forecast_rows_for_configs(run_dir, best_by_build, best_configs)
        shock_paths = apply_date_window(
            shock_paths,
            "target_date",
            pd.Timestamp("2020-01-01").date(),
            pd.Timestamp("2021-05-01").date(),
        )
        selected_shock_models, shock_chart_paths, duplicate_count = select_distinct_model_paths(
            best_by_build,
            shock_paths,
            max_models=25,
        )
        if shock_chart_paths.empty:
            st.info("No forecast paths were found for the fixed COVID shock window.")
        else:
            st.plotly_chart(
                top_model_chart(shock_chart_paths, "Best Model-Build Paths During COVID Shock"),
                use_container_width=True,
            )
            st.caption(
                "Selection: non-baseline model configurations only; top one per model build by balanced score. "
                "The dashed orange line is the seasonal-naive reference, while the thick black line is actual UPT. "
                "The spread in 2020 shows how differently model families reacted to an abrupt structural break; "
                "the convergence by early 2021 is a useful clue for comparing shock handling against recovery behavior."
            )
            if duplicate_count:
                st.caption(
                    f"Skipped {duplicate_count} duplicate prediction path(s) so the chart emphasizes distinct lines."
                )
            render_metric_dataframe(
                overview_table(selected_shock_models.head(25)),
                max_rows=25,
                max_visible_rows=25,
            )

        st.markdown("#### Top 10 Overall Models In The Same Window")
        st.write(
            "The companion view removes the per-build diversity rule and asks a more "
            "leaderboard-like question: among the top 10 non-baseline configurations "
            "overall, how similar are their shock-period paths?"
        )
        top10_overall = limit_total_configs(candidate_models, "Balanced score", "Top 10")
        top10_configs = top10_overall["model_config_id"].astype(str).tolist()
        top10_paths = load_forecast_rows_for_configs(run_dir, top10_overall, top10_configs)
        top10_paths = apply_date_window(
            top10_paths,
            "target_date",
            pd.Timestamp("2020-01-01").date(),
            pd.Timestamp("2021-05-01").date(),
        )
        selected_top10_models, top10_chart_paths, top10_duplicate_count = select_distinct_model_paths(
            top10_overall,
            top10_paths,
            max_models=10,
        )
        if top10_chart_paths.empty:
            st.info("No top-10 forecast paths were found for the fixed COVID shock window.")
        else:
            st.plotly_chart(
                top_model_chart(top10_chart_paths, "Top 10 Overall Paths During COVID Shock"),
                use_container_width=True,
            )
            st.caption(
                "Selection: top 10 non-baseline configurations by balanced score, regardless of model build. "
                "If these paths cluster tightly, the leaderboard is pointing toward a shared modeling strategy; "
                "if they diverge, similar aggregate scores may be hiding different shock-period behavior."
            )
            if top10_duplicate_count:
                st.caption(
                    f"Skipped {top10_duplicate_count} duplicate prediction path(s) so the chart emphasizes distinct lines."
                )
            render_metric_dataframe(
                overview_table(selected_top10_models.head(10)),
                max_rows=10,
                max_visible_rows=10,
            )

        xgboost_count = int(top10_overall["model_build"].astype(str).eq("xgboost").sum()) if "model_build" in top10_overall else 0
        st.markdown("### Why XGBoost Is A Plausible Front-Runner")
        st.write(
            "The current leaderboard suggests that XGBoost is not merely winning because it is a tree model. "
            "It is likely benefiting from the combination of boosted shallow trees, regularization, and wide "
            "feature families that include lagged history, regime indicators, time context, and targeted interactions."
        )
        if not top10_overall.empty:
            st.caption(
                f"In the current top-10 balanced-score slice, {xgboost_count} of "
                f"{len(top10_overall)} configurations are XGBoost."
            )
        xgb_rows = [
            {
                "Likely contributor": "Boosting learns sequential corrections",
                "Why it matters here": "A random forest averages independent trees; boosting can fit residual structure left by earlier trees, which is useful when ordinary seasonality breaks.",
                "Follow-up test": "Compare XGBoost to sklearn gradient boosting / histogram gradient boosting under matched feature sets.",
            },
            {
                "Likely contributor": "Nonlinear interactions without manual recipes",
                "Why it matters here": "The strongest feature families contain regime, service, time, and interaction signals. Boosted trees can choose thresholds and combinations without requiring explicit linear terms.",
                "Follow-up test": "Run XGBoost with interaction-heavy families removed, then with only compact history/regime families.",
            },
            {
                "Likely contributor": "Regularized incremental fit",
                "Why it matters here": "Learning rate, tree depth, subsampling, and child-weight constraints can make XGBoost flexible without letting every noisy feature dominate.",
                "Follow-up test": "Ablate learning rate, depth, subsample, colsample, and min_child_weight while holding the feature family constant.",
            },
            {
                "Likely contributor": "Feature-policy alignment",
                "Why it matters here": "Tree-top feature policies may pair especially well with XGBoost because the selector and final model both favor split-based nonlinear signal.",
                "Follow-up test": "Compare `none`, `tree_top_30`, mutual information, and correlation pruning within XGBoost and Random Forest on the same families.",
            },
        ]
        st.dataframe(pd.DataFrame(xgb_rows), use_container_width=True, hide_index=True)
        st.info(
            "A follow-up experiment is worth planning, but it should be framed as an ablation. "
            "Adding more random-forest variants can test whether the gap is just tree capacity, "
            "but it will not reproduce the central XGBoost ingredient: sequential gradient boosting. "
            "The cleaner comparison is a matched tree-family study: tuned Random Forest and Extra Trees, "
            "gradient-boosted trees, and XGBoost ablations across the same feature families and scoring windows."
        )

    if not candidate_models.empty and score_col in candidate_models:
        build_summary = (
            candidate_models.groupby(["model_family", "model_build", "model_build_label"], dropna=False)
            .agg(
                configurations=("config_id", "nunique"),
                best_balanced_score=(score_col, "min"),
                median_balanced_score=(score_col, "median"),
                best_mae=("mae", "min"),
                best_rmse=("rmse", "min"),
                best_r2=("r2", "max"),
            )
            .reset_index()
        )
        build_summary = model_taxonomy_sort(build_summary).sort_values("best_balanced_score")
        st.markdown("### First-Pass Model Build Summary")
        st.write(
            "This is a compact starting point for the deeper notebook analysis: one row "
            "per model build, ranked by the best balanced score found in the current "
            "dashboard artifact bundle."
        )
        fig = px.bar(
            build_summary.sort_values("best_balanced_score", ascending=True),
            x="best_balanced_score",
            y="model_build_label",
            color="model_family",
            orientation="h",
            labels={
                "best_balanced_score": "Best balanced score",
                "model_build_label": "Model build",
                "model_family": "Model family",
            },
            hover_data=["configurations", "best_mae", "best_rmse", "best_r2"],
        )
        fig.update_layout(
            height=max(380, 34 * len(build_summary) + 90),
            margin={"l": 170, "r": 30, "t": 35, "b": 45},
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#2f323a"},
        )
        st.plotly_chart(fig, use_container_width=True)
        render_metric_dataframe(
            build_summary[
                [
                    "model_build_label",
                    "model_family",
                    "configurations",
                    "best_balanced_score",
                    "median_balanced_score",
                    "best_mae",
                    "best_rmse",
                    "best_r2",
                ]
            ],
            max_rows=50,
        )

    if not candidate_models.empty and {"model_family", "feature_transform_label", score_col}.issubset(candidate_models.columns):
        linear_models = candidate_models[candidate_models["model_family"] == "linear"].copy()
        if not linear_models.empty:
            transform_summary = (
                linear_models.groupby(["model_build_label", "feature_transform", "feature_transform_label"], dropna=False)
                .agg(
                    configurations=("config_id", "nunique"),
                    best_balanced_score=(score_col, "min"),
                    median_balanced_score=(score_col, "median"),
                    best_mae=("mae", "min"),
                    best_rmse=("rmse", "min"),
                )
                .reset_index()
            )
            transform_summary["_transform_order"] = transform_summary["feature_transform"].map(
                lambda value: order_index(str(value), FEATURE_TRANSFORM_ORDER)[0]
            )
            transform_summary = transform_summary.sort_values(
                ["model_build_label", "_transform_order", "best_balanced_score"]
            ).drop(columns=["_transform_order", "feature_transform"])
            st.markdown("### Linear Transform Screening")
            st.write(
                "This first-pass table compares transform families inside the regularized "
                "linear models. It is a screening view, not yet a conclusion about which "
                "variables should receive which transform. No transform is the untransformed "
                "baseline and is shown first for each model build. Non-identity transform "
                "families keep the original selected features and append transformed terms."
            )
            lasso_rows = linear_models[linear_models["model_build"].astype(str).eq("lasso")]
            if not lasso_rows.empty and set(lasso_rows["feature_transform"].dropna().astype(str)) == {"identity"}:
                st.info(
                    "Lasso appears with No transform only in the current dashboard bundle. "
                    "The transform code supports lasso, but the merged nonlinear follow-up "
                    "did not add transformed lasso configurations; those would need a targeted "
                    "rerun if we want a direct lasso transform comparison."
                )
            render_metric_dataframe(transform_summary, max_rows=75, max_visible_rows=15)

    period_cols = [
        ("Pre-COVID", "pre_covid_mae"),
        ("COVID shock", "covid_shock_mae"),
        ("Recovery", "recovery_mae"),
        ("Recent", "recent_mae"),
    ]
    if not candidate_models.empty and all(col in candidate_models for _, col in period_cols):
        period_rows = []
        for label, column in period_cols:
            best_mae = candidate_models[column].min()
            median_mae = candidate_models[column].median()
            improvement = safe_ratio(median_mae - best_mae, median_mae)
            period_rows.append(
                {
                    "Period": label,
                    "Best MAE": best_mae,
                    "Median MAE": median_mae,
                    "Best vs median improvement": f"{improvement:.1%}" if pd.notna(improvement) else "-",
                }
            )
        st.markdown("### Period Difficulty Snapshot")
        st.write(
            "A useful result-inspection habit is to separate the global leaderboard "
            "from period-specific behavior. This snapshot starts that comparison by "
            "showing the best and median MAE by evaluation period, plus how much the "
            "best configuration improves on the typical configuration in that period."
        )
        st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def load_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


def safe_partition_value(value) -> str:
    text = "unknown" if pd.isna(value) else str(value)
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    return safe.strip("_") or "unknown"


def model_id_column(df: pd.DataFrame) -> str:
    if "model_config_id" in df.columns:
        return "model_config_id"
    if "config_id" in df.columns:
        return "config_id"
    raise KeyError("Expected either model_config_id or config_id.")


def path_collection_modified_ns(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_mtime_ns
    return max((file.stat().st_mtime_ns for file in path.rglob("*.parquet")), default=path.stat().st_mtime_ns)


def path_partition_files(dataset_dir: Path, model_builds: tuple[str, ...]) -> list[Path]:
    if not dataset_dir.exists():
        return []
    if model_builds:
        files = []
        for build in model_builds:
            files.extend((dataset_dir / f"model_build={safe_partition_value(build)}").glob("*.parquet"))
        return sorted(path for path in files if path.exists())
    return sorted(dataset_dir.glob("**/*.parquet"))


@st.cache_data(show_spinner=False)
def load_path_rows_for_configs(
    run_dir: str,
    dataset_name: str,
    config_ids: tuple[str, ...],
    model_builds: tuple[str, ...],
    modified_ns: int,
) -> pd.DataFrame:
    _ = modified_ns
    if not config_ids:
        return pd.DataFrame()

    run_path = Path(run_dir)
    dataset_dir = run_path / PATH_DATASET_DIRS[dataset_name]
    files = path_partition_files(dataset_dir, model_builds)
    config_values = [str(config_id) for config_id in config_ids]

    if files and pl is not None:
        lazy_frames = [pl.scan_parquet(str(path)) for path in files]
        lazy = lazy_frames[0] if len(lazy_frames) == 1 else pl.concat(lazy_frames, how="diagonal_relaxed")
        columns = set(lazy.collect_schema().names())
        id_col = "model_config_id" if "model_config_id" in columns else "config_id"
        filtered = lazy.filter(pl.col(id_col).cast(pl.Utf8).is_in(config_values))
        return filtered.collect().to_pandas()

    if files:
        frames = []
        for path in files:
            frame = pd.read_parquet(path)
            id_col = model_id_column(frame)
            frames.append(frame[frame[id_col].astype(str).isin(config_values)])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fallback_path = run_path / REQUIRED_FILES[dataset_name]
    if fallback_path.exists():
        frame = load_parquet(str(fallback_path), file_modified_ns(fallback_path))
        id_col = model_id_column(frame)
        return frame[frame[id_col].astype(str).isin(config_values)].copy()
    return pd.DataFrame()


def load_forecast_rows_for_configs(
    run_dir: Path,
    ranked_models: pd.DataFrame,
    config_ids: list[str],
) -> pd.DataFrame:
    builds = tuple(sorted(ranked_models["model_build"].dropna().astype(str).unique())) if "model_build" in ranked_models else ()
    dataset_path = run_dir / PATH_DATASET_DIRS["forecast_paths"]
    loaded = load_path_rows_for_configs(
        str(run_dir),
        "forecast_paths",
        tuple(str(config_id) for config_id in config_ids),
        builds,
        path_collection_modified_ns(dataset_path if dataset_path.exists() else run_dir / REQUIRED_FILES["forecast_paths"]),
    )
    return ensure_model_taxonomy(normalize_dates(loaded, ["as_of_date", "target_date"]))


def load_performance_rows_for_configs(
    run_dir: Path,
    ranked_models: pd.DataFrame,
    config_ids: list[str],
) -> pd.DataFrame:
    builds = tuple(sorted(ranked_models["model_build"].dropna().astype(str).unique())) if "model_build" in ranked_models else ()
    dataset_path = run_dir / PATH_DATASET_DIRS["performance_over_time"]
    loaded = load_path_rows_for_configs(
        str(run_dir),
        "performance_over_time",
        tuple(str(config_id) for config_id in config_ids),
        builds,
        path_collection_modified_ns(dataset_path if dataset_path.exists() else run_dir / REQUIRED_FILES["performance_over_time"]),
    )
    return ensure_model_taxonomy(normalize_dates(loaded, ["as_of_date", "target_date"]))


@st.cache_data(show_spinner=False)
def load_json(path: str, modified_ns: int) -> dict:
    _ = modified_ns
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_text(path: str, modified_ns: int) -> str:
    _ = modified_ns
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def file_modified_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def configured_artifact_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))


def configured_feature_families_path() -> Path:
    return Path(os.environ.get("FEATURE_FAMILIES_PATH", DEFAULT_FEATURE_FAMILIES_PATH))


def discover_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    if all((base_dir / filename).exists() for filename in REQUIRED_FILES.values()):
        return [base_dir]

    run_dirs = [
        path
        for path in base_dir.glob("**/*")
        if path.is_dir() and all((path / filename).exists() for filename in REQUIRED_FILES.values())
    ]
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def load_artifacts(run_dir: Path) -> dict:
    def parquet_artifact(filename: str) -> pd.DataFrame:
        path = run_dir / filename
        return load_parquet(str(path), file_modified_ns(path))

    def json_artifact(filename: str) -> dict:
        path = run_dir / filename
        return load_json(str(path), file_modified_ns(path))

    artifacts = {
        "forecast_paths": parquet_artifact(REQUIRED_FILES["forecast_paths"]),
        "performance_over_time": parquet_artifact(REQUIRED_FILES["performance_over_time"]),
        "model_leaderboard": parquet_artifact(REQUIRED_FILES["model_leaderboard"]),
        "feature_family_summary": parquet_artifact(REQUIRED_FILES["feature_family_summary"]),
        "champion_predictions": parquet_artifact(REQUIRED_FILES["champion_predictions"]),
        "champion_selection": json_artifact(REQUIRED_FILES["champion_selection"]),
    }
    for artifact_name, filename in OPTIONAL_FILES.items():
        path = run_dir / filename
        if path.exists():
            if filename.endswith(".json"):
                artifacts[artifact_name] = load_json(str(path), file_modified_ns(path))
            else:
                artifacts[artifact_name] = load_parquet(str(path), file_modified_ns(path))
    return artifacts


def load_feature_family_definitions() -> dict:
    path = configured_feature_families_path()
    if not path.exists():
        return {}
    return load_json(str(path), file_modified_ns(path))


def load_config_text(path: Path) -> str:
    if not path.exists():
        return ""
    return load_text(str(path), file_modified_ns(path))


def format_int(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def format_float(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


METRIC_TABLE_COLUMN_GROUPS = [
    {
        "columns": {
            "rank",
            "model_build_label",
            "mode",
            "feature_family_name",
            "feature_policy",
            "hyperparameters",
            "configurations",
        },
        "background": "#f7faf9",
        "border": "#8fc9bd",
    },
    {
        "columns": {
            "mae",
            "rmse",
            "r2",
            "r2_adjusted",
            "diracc",
            "selection_score_balanced",
        },
        "background": "#eff6ff",
        "border": "#93c5fd",
    },
    {
        "columns": {"pre_covid_mae", "pre_covid_rmse"},
        "background": "#f0fdf4",
        "border": "#86efac",
    },
    {
        "columns": {"covid_shock_mae", "covid_shock_rmse"},
        "background": "#fef2f2",
        "border": "#fca5a5",
    },
    {
        "columns": {"recovery_mae", "recovery_rmse"},
        "background": "#fff7ed",
        "border": "#fdba74",
    },
    {
        "columns": {"recent_mae", "recent_rmse"},
        "background": "#f5f3ff",
        "border": "#c4b5fd",
    },
    {
        "columns": {
            "shock_penalty",
            "rmse_shock_penalty",
            "recovery_ratio",
            "rmse_recovery_ratio",
            "recent_recovery_ratio",
            "rmse_recent_recovery_ratio",
            "complexity_score",
            "interpretability_score",
            "compute_score",
        },
        "background": "#fffbeb",
        "border": "#fcd34d",
    },
]


def styled_metric_table(frame: pd.DataFrame):
    if frame.empty:
        return frame

    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    table_styles = []
    for group in METRIC_TABLE_COLUMN_GROUPS:
        matched_columns = [column for column in frame.columns if column in group["columns"]]
        if not matched_columns:
            continue
        styles.loc[:, matched_columns] = (
            f"background-color: {group['background']}; "
            f"border-left: 1px solid {group['border']};"
        )
        for column in matched_columns:
            column_index = frame.columns.get_loc(column)
            table_styles.extend(
                [
                    {
                        "selector": f"th.col{column_index}",
                        "props": [
                            ("background-color", group["background"]),
                            ("border-left", f"1px solid {group['border']}"),
                        ],
                    },
                    {
                        "selector": f"td.col{column_index}",
                        "props": [
                            ("background-color", group["background"]),
                            ("border-left", f"1px solid {group['border']}"),
                        ],
                    },
                ]
            )

    def apply_styles(_):
        return styles

    return frame.style.apply(apply_styles, axis=None).set_table_styles(table_styles)


def dataframe_height_for_rows(row_count: int, max_visible_rows: int = 12) -> int:
    visible_rows = min(max(row_count, 1), max_visible_rows)
    return 39 + visible_rows * 35


def render_metric_dataframe(
    frame: pd.DataFrame,
    max_rows: int = None,
    max_visible_rows: int = None,
) -> None:
    display = frame.head(max_rows) if max_rows is not None else frame
    dataframe_kwargs = {}
    if max_visible_rows is not None:
        dataframe_kwargs["height"] = dataframe_height_for_rows(len(display), max_visible_rows)
    st.dataframe(
        styled_metric_table(display),
        hide_index=True,
        use_container_width=True,
        **dataframe_kwargs,
    )


def date_range_label(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return "-"
    dates = pd.to_datetime(df[column])
    return f"{dates.min().date()} to {dates.max().date()}"


def normalize_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_datetime(out[column])
    return out


def model_family_for(model_type: str) -> str:
    if model_type == "naive":
        return "baseline"
    if model_type in {"ridge", "lasso", "elastic_net"}:
        return "linear"
    if model_type in {"arima", "sarima", "sarimax"}:
        return "autoregressive"
    if model_type in {"random_forest", "extra_trees", "xgboost"}:
        return "tree"
    if model_type in {"mlp", "rnn", "gru", "lstm"}:
        return "neural_net"
    return "other"


def model_build_for(model_type: str) -> str:
    return "seasonal_naive" if model_type == "naive" else model_type


def ensure_model_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model_type" in out:
        if "model_family" not in out:
            out["model_family"] = out["model_type"].map(model_family_for)
        if "model_build" not in out:
            out["model_build"] = out["model_type"].map(model_build_for)
    if "model_config_id" not in out and "config_id" in out:
        out["model_config_id"] = out["config_id"]
    if "feature_transform" not in out:
        out["feature_transform"] = "identity"
    else:
        out["feature_transform"] = out["feature_transform"].fillna("identity")
    out["feature_transform_label"] = out["feature_transform"].map(
        lambda value: FEATURE_TRANSFORM_LABELS.get(str(value), str(value).replace("_", " ").title())
    )
    if {"model_family", "model_build"}.issubset(out.columns):
        out["model_build_label"] = [
            model_build_display_label(family, build)
            for family, build in zip(out["model_family"], out["model_build"])
        ]
    return out


def safe_ratio(numerator, denominator) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def weighted_error_score(mae: pd.Series, rmse: pd.Series, recipe: dict) -> pd.Series:
    return recipe["mae_weight"] * pd.to_numeric(mae, errors="coerce") + recipe["rmse_weight"] * pd.to_numeric(
        rmse,
        errors="coerce",
    )


def enrich_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add score variants and period ratios when older artifacts only have base metrics."""
    out = df.copy()
    if {"mae", "rmse"}.issubset(out.columns):
        for recipe_name, recipe in SCORE_RECIPES.items():
            out[f"selection_score_{recipe_name}"] = weighted_error_score(out["mae"], out["rmse"], recipe)
        if "selection_score" not in out:
            out["selection_score"] = out["selection_score_balanced"]

    for period in EVALUATION_PERIODS:
        mae_col = f"{period}_mae"
        rmse_col = f"{period}_rmse"
        if {mae_col, rmse_col}.issubset(out.columns):
            for recipe_name, recipe in SCORE_RECIPES.items():
                out[f"{period}_selection_score_{recipe_name}"] = weighted_error_score(
                    out[mae_col],
                    out[rmse_col],
                    recipe,
                )
            legacy_col = f"{period}_selection_score"
            if legacy_col not in out:
                out[legacy_col] = out[f"{period}_selection_score_balanced"]

    ratio_specs = {
        "shock_penalty": ("covid_shock_mae", "pre_covid_mae"),
        "recovery_ratio": ("recovery_mae", "pre_covid_mae"),
        "recent_recovery_ratio": ("recent_mae", "pre_covid_mae"),
        "rmse_shock_penalty": ("covid_shock_rmse", "pre_covid_rmse"),
        "rmse_recovery_ratio": ("recovery_rmse", "pre_covid_rmse"),
        "rmse_recent_recovery_ratio": ("recent_rmse", "pre_covid_rmse"),
    }
    for recipe_name in SCORE_RECIPES:
        prefix = f"{recipe_name}_score"
        ratio_specs[f"{prefix}_shock_penalty"] = (
            f"covid_shock_selection_score_{recipe_name}",
            f"pre_covid_selection_score_{recipe_name}",
        )
        ratio_specs[f"{prefix}_recovery_ratio"] = (
            f"recovery_selection_score_{recipe_name}",
            f"pre_covid_selection_score_{recipe_name}",
        )
        ratio_specs[f"{prefix}_recent_recovery_ratio"] = (
            f"recent_selection_score_{recipe_name}",
            f"pre_covid_selection_score_{recipe_name}",
        )

    for output_col, (numerator_col, denominator_col) in ratio_specs.items():
        if {numerator_col, denominator_col}.issubset(out.columns):
            out[output_col] = [
                safe_ratio(numerator, denominator)
                for numerator, denominator in zip(out[numerator_col], out[denominator_col])
            ]
    return out


def filtered_frame(
    df: pd.DataFrame,
    model_family="All",
    model_build="All",
    model_build_label="All",
    mode="All",
    feature_family="All",
    feature_policy="All",
    feature_transform="All",
) -> pd.DataFrame:
    out = apply_optional_value_filter(df, "model_build_label", model_build_label)
    out = apply_optional_value_filter(out, "model_family", model_family)
    out = apply_optional_value_filter(out, "model_build", model_build)
    out = apply_optional_value_filter(out, "mode", mode)
    out = apply_optional_value_filter(out, "feature_family_name", feature_family)
    out = apply_optional_value_filter(out, "feature_policy", feature_policy)
    out = apply_optional_value_filter(out, "feature_transform_label", feature_transform)
    return out


def exclude_baseline_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep seasonal naive as a chart reference, not as an interactive candidate model."""
    out = df.copy()
    if "model_family" in out:
        out = out[~out["model_family"].astype(str).isin(BASELINE_MODEL_FAMILIES)]
    if "model_build" in out:
        out = out[~out["model_build"].astype(str).isin(BASELINE_MODEL_BUILDS)]
    return out


def date_bounds(df: pd.DataFrame, column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(df[column]).dropna()
    if dates.empty:
        today = pd.Timestamp.today().normalize()
        return today, today
    return dates.min(), dates.max()


def apply_date_window(
    df: pd.DataFrame,
    column: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    if df.empty or column not in df:
        return df
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    dates = pd.to_datetime(df[column])
    return df[(dates >= start_ts) & (dates <= end_ts)].copy()


def month_interval_label(dates: pd.Series) -> str:
    clean_dates = pd.to_datetime(dates).dropna().sort_values().drop_duplicates()
    if len(clean_dates) < 2:
        return "-"
    month_diffs = (
        (clean_dates.dt.year.diff() * 12 + clean_dates.dt.month.diff())
        .dropna()
        .astype(int)
    )
    if month_diffs.empty:
        return "-"
    interval = int(month_diffs.mode().iloc[0])
    return f"{interval} month" if interval == 1 else f"{interval} months"


def order_index(value, ordered_values: list[str]) -> tuple[int, str]:
    value_text = str(value)
    try:
        return ordered_values.index(value_text), value_text
    except ValueError:
        return len(ordered_values), value_text


def ordered_unique(values: pd.Series, ordered_values=None) -> list[str]:
    clean_values = [str(value) for value in values.dropna().unique()]
    if not ordered_values:
        return sorted(clean_values)
    return sorted(clean_values, key=lambda value: order_index(value, ordered_values))


def ordered_model_build_labels(df: pd.DataFrame) -> list[str]:
    if df.empty or "model_build_label" not in df:
        return []
    label_rows = model_taxonomy_sort(
        df[["model_family", "model_build", "model_build_label"]].drop_duplicates()
    )
    return label_rows["model_build_label"].tolist()


def model_taxonomy_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_model_family_order"] = out["model_family"].map(
        lambda value: order_index(value, MODEL_FAMILY_ORDER)[0]
    ) if "model_family" in out else len(MODEL_FAMILY_ORDER)
    out["_model_build_order"] = out["model_build"].map(
        lambda value: order_index(value, MODEL_BUILD_ORDER)[0]
    ) if "model_build" in out else len(MODEL_BUILD_ORDER)
    sort_columns = [
        column
        for column in ["_model_family_order", "_model_build_order", "model_family", "model_build"]
        if column in out.columns
    ]
    return out.sort_values(sort_columns).drop(columns=["_model_family_order", "_model_build_order"], errors="ignore")


def manifest_value(manifest: dict, key: str, fallback="-") -> str:
    value = manifest.get(key)
    if value is None or value == "":
        return fallback
    return str(value)


def months_label(value) -> str:
    if value is None or value == "" or value == "-":
        return "-"
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric_value} month" if numeric_value == 1 else f"{numeric_value} months"


def line_forecast_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["target_date"],
            y=df["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(width=3),
            opacity=1,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["target_date"],
            y=df["prediction"],
            mode="lines+markers",
            name="Prediction",
            line=dict(width=2),
            opacity=0.62,
        )
    )
    if "seasonal_naive_prediction" in df:
        fig.add_trace(
            go.Scatter(
                x=df["target_date"],
                y=df["seasonal_naive_prediction"],
                mode="lines",
                name="Seasonal naive",
                line=dict(dash="dash"),
                opacity=1,
            )
        )
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Target month",
        yaxis_title="UPT",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="left",
            x=0,
            tracegroupgap=4,
        ),
        margin=dict(l=10, r=10, t=50, b=145),
        height=560,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#2f323a"),
    )
    fig.update_xaxes(title_standoff=28)
    return fig


def rolling_error_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="as_of_date",
        y="rolling_6mo_mae",
        color="config_id",
        labels={
            "as_of_date": "As-of date",
            "rolling_6mo_mae": "Rolling 6-month MAE",
            "config_id": "Configuration",
        },
    )
    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def all_selectbox(label: str, values: pd.Series, key: str, container=st) -> str:
    if label == "Model family":
        clean_values = ordered_unique(values, MODEL_FAMILY_ORDER)
    elif label == "Model build":
        clean_values = ordered_unique(values, MODEL_BUILD_ORDER)
    else:
        clean_values = ordered_unique(values)
    return container.selectbox(label, ["All"] + clean_values, key=key)


def optional_multiselect(
    label: str,
    values: pd.Series,
    key: str,
    container=st,
    order: Optional[list[str]] = None,
) -> list[str]:
    clean_values = ordered_unique(values, order)
    return container.multiselect(
        label,
        clean_values,
        default=[],
        key=key,
        placeholder="All",
        help="Leave empty to include all options.",
    )


def apply_optional_filter(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if value == "All" or column not in df:
        return df
    return df[df[column].astype(str) == value]


def apply_optional_multi_filter(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    if not values or column not in df:
        return df
    return df[df[column].astype(str).isin([str(value) for value in values])]


def apply_optional_value_filter(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    if isinstance(value, list):
        return apply_optional_multi_filter(df, column, value)
    return apply_optional_filter(df, column, value)


def parse_json_display(value) -> str:
    if pd.isna(value) or value in ("", "{}"):
        return "N/A"
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return str(value)
    if not payload:
        return "N/A"
    return ", ".join(f"{key}={payload[key]}" for key in sorted(payload))


def parse_json_payload(value) -> dict:
    if pd.isna(value) or value in ("", "{}"):
        return {}
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def shorthand_hyperparameters(value, max_items: int = 4) -> str:
    aliases = {
        "n_estimators": "n",
        "max_depth": "depth",
        "learning_rate": "lr",
        "min_child_weight": "mcw",
        "alpha": "alpha",
        "l1_ratio": "l1",
        "max_features": "max_feat",
        "min_samples_leaf": "leaf",
    }
    payload = parse_json_payload(value)
    if not payload:
        return "params: none"
    preferred_keys = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "alpha",
        "l1_ratio",
        "max_features",
        "min_samples_leaf",
    ]
    ordered_keys = [key for key in preferred_keys if key in payload]
    ordered_keys.extend(sorted(key for key in payload if key not in ordered_keys))
    parts = [
        f"{aliases.get(key, key)}={payload[key]}"
        for key in ordered_keys[:max_items]
    ]
    if len(ordered_keys) > max_items:
        parts.append("...")
    return ", ".join(parts)


def selectbox_index(options: list[str], default_value) -> int:
    if default_value is None:
        return 0
    default_text = str(default_value)
    return options.index(default_text) if default_text in options else 0


def format_rank_metric_value(metric_label: str, value) -> str:
    if pd.isna(value):
        return "N/A"
    metric_column, _ = RANK_METRIC_OPTIONS[metric_label]
    if metric_column in {"r2", "r2_adjusted", "diracc"}:
        return format_float(value, 4)
    if metric_column.endswith("_ratio") or metric_column.endswith("_penalty"):
        return format_float(value, 3)
    return format_int(value)


def ranked_model_label(row: pd.Series, metric_label: str) -> str:
    metric_column, _ = RANK_METRIC_OPTIONS[metric_label]
    metric_value = row.get(metric_column)
    metric_fragment = (
        f"{metric_label.lower()} {format_rank_metric_value(metric_label, metric_value)}"
        if pd.notna(metric_value)
        else f"{metric_label.lower()} N/A"
    )
    params = shorthand_hyperparameters(row.get("hyperparameters_json", "{}"))
    build_label = row.get(
        "model_build_label",
        model_build_display_label(row.get("model_family", ""), row.get("model_build", row.get("model_type", "model"))),
    )
    transform_label = row.get("feature_transform_label", FEATURE_TRANSFORM_LABELS.get("identity", "No transform"))
    return (
        f"#{int(row['rank'])} | {build_label} "
        f"| {row.get('mode', '-')} | {row.get('feature_family_name', '-')} "
        f"| {transform_label} "
        f"| {metric_fragment} | MAE {format_int(row.get('mae'))} "
        f"| RMSE {format_int(row.get('rmse'))} | {params}"
    )


def top_model_chart(paths: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if paths.empty:
        fig.update_layout(title=title)
        return fig

    actual = paths[["target_date", "actual"]].drop_duplicates().sort_values("target_date")
    fig.add_trace(
        go.Scatter(
            x=actual["target_date"],
            y=actual["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(width=4, color="#111827"),
            opacity=1,
        )
    )

    if "baseline_prediction" in paths:
        baseline = (
            paths[["target_date", "baseline_prediction"]]
            .drop_duplicates()
            .sort_values("target_date")
        )
        fig.add_trace(
            go.Scatter(
                x=baseline["target_date"],
                y=baseline["baseline_prediction"],
                mode="lines",
                name="Seasonal naive",
                line=dict(dash="dash", color="#f97316"),
                opacity=1,
            )
        )

    for _, group in paths.sort_values(["rank", "target_date"]).groupby("model_config_id", sort=False):
        first = group.iloc[0]
        label = (
            f"#{int(first['rank'])} {first.get('model_build_label', first.get('model_build', first.get('model_type', 'model')))} | "
            f"{first.get('feature_family_name', '-')} | "
            f"{first.get('feature_transform_label', 'No transform')}"
        )
        fig.add_trace(
            go.Scatter(
                x=group["target_date"],
                y=group["prediction"],
                mode="lines+markers",
                name=label,
                line=dict(width=2),
                opacity=0.58,
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Target month",
        yaxis_title="UPT",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.34,
            xanchor="left",
            x=0,
            tracegroupgap=4,
        ),
        margin=dict(l=10, r=10, t=50, b=190),
        height=650,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#2f323a"),
    )
    fig.update_xaxes(title_standoff=32)
    return fig


def default_target_window_for_rank(
    metric_label: str,
    target_min: pd.Timestamp,
    target_max: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_raw, end_raw = PERIOD_RANK_WINDOWS.get(metric_label, (None, None))
    start = max(target_min, pd.Timestamp(start_raw)) if start_raw else target_min
    end = min(target_max, pd.Timestamp(end_raw)) if end_raw else target_max
    if start > end:
        return target_min, target_max
    return start, end


def select_distinct_model_paths(
    ranked_models: pd.DataFrame,
    paths: pd.DataFrame,
    max_models: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Choose top models while avoiding visually identical prediction paths."""
    selected_rows = []
    selected_ids = []
    signatures = set()
    duplicate_count = 0

    for _, row in ranked_models.iterrows():
        config_id = row["model_config_id"]
        path = paths[paths["model_config_id"] == config_id].sort_values("target_date")
        if path.empty:
            continue
        signature = tuple(path["prediction"].round(3).tolist())
        if signature in signatures:
            duplicate_count += 1
            continue
        signatures.add(signature)
        selected_rows.append(row)
        selected_ids.append(config_id)
        if len(selected_rows) >= max_models:
            break

    selected_models = pd.DataFrame(selected_rows)
    selected_paths = paths[paths["model_config_id"].isin(selected_ids)].copy()
    rank_lookup = {config_id: rank for rank, config_id in enumerate(selected_ids, start=1)}
    if not selected_models.empty:
        selected_models = selected_models.copy()
        selected_models["rank"] = selected_models["model_config_id"].map(rank_lookup)
    if not selected_paths.empty:
        selected_paths["rank"] = selected_paths["model_config_id"].map(rank_lookup)
    return selected_models, selected_paths, duplicate_count


def overview_table(top_models: pd.DataFrame) -> pd.DataFrame:
    display = top_models.copy()
    if "hyperparameters_json" in display:
        display["hyperparameters"] = display["hyperparameters_json"].apply(parse_json_display)
    else:
        display["hyperparameters"] = "N/A"
    columns = [
        "rank",
        "model_build_label",
        "feature_family_name",
        "feature_transform_label",
        "mode",
        "feature_policy",
        "hyperparameters",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score_balanced",
        "pre_covid_mae",
        "pre_covid_rmse",
        "covid_shock_mae",
        "covid_shock_rmse",
        "recovery_mae",
        "recovery_rmse",
        "recent_mae",
        "recent_rmse",
        "shock_penalty",
        "rmse_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
    ]
    available = [column for column in columns if column in display]
    return display[available]


def forecast_ranked_table(ranked_models: pd.DataFrame) -> pd.DataFrame:
    display = ranked_models.copy()
    if "hyperparameters_json" in display:
        display["hyperparameters"] = display["hyperparameters_json"].apply(parse_json_display)
    else:
        display["hyperparameters"] = "N/A"
    columns = [
        "rank",
        "model_build_label",
        "mode",
        "feature_family_name",
        "feature_transform_label",
        "feature_policy",
        "hyperparameters",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score_balanced",
        "pre_covid_mae",
        "pre_covid_rmse",
        "covid_shock_mae",
        "covid_shock_rmse",
        "recovery_mae",
        "recovery_rmse",
        "recent_mae",
        "recent_rmse",
    ]
    available = [column for column in columns if column in display]
    return display[available]


def available_rank_options(df: pd.DataFrame) -> list[str]:
    return [label for label, (column, _) in RANK_METRIC_OPTIONS.items() if column in df.columns]


def sort_by_rank_metric(df: pd.DataFrame, label: str) -> pd.DataFrame:
    column, ascending = RANK_METRIC_OPTIONS[label]
    ranked = df.dropna(subset=[column]).sort_values(column, ascending=ascending).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def metric_mapping_frame(
    leaderboard: pd.DataFrame,
    model_build_labels: list[str],
    rank_label: str,
    per_build_limit: str,
    total_limit: str = "All",
    feature_families: Optional[list[str]] = None,
    feature_policies: Optional[list[str]] = None,
    feature_transforms: Optional[list[str]] = None,
) -> pd.DataFrame:
    if not model_build_labels:
        return leaderboard.iloc[0:0].copy()
    frame = leaderboard[leaderboard["model_build_label"].astype(str).isin(model_build_labels)].copy()
    frame = apply_optional_multi_filter(frame, "feature_family_name", feature_families or [])
    frame = apply_optional_multi_filter(frame, "feature_policy", feature_policies or [])
    frame = apply_optional_multi_filter(frame, "feature_transform_label", feature_transforms or [])
    if per_build_limit == "All":
        return limit_total_configs(frame, rank_label, total_limit)

    limit = int(per_build_limit.replace("Top ", ""))
    ranked_slices = []
    for _, group in frame.groupby("model_build_label", sort=False):
        ranked_slices.append(sort_by_rank_metric(group, rank_label).head(limit))
    if not ranked_slices:
        return frame.iloc[0:0].copy()
    limited = pd.concat(ranked_slices, ignore_index=True)
    return limit_total_configs(limited, rank_label, total_limit)


def limit_configs_per_build(frame: pd.DataFrame, rank_label: str, per_build_limit: str) -> pd.DataFrame:
    if frame.empty or per_build_limit == "All":
        return frame
    limit = int(per_build_limit.replace("Top ", ""))
    ranked_slices = []
    group_col = "model_build_label" if "model_build_label" in frame.columns else "model_build"
    for _, group in frame.groupby(group_col, sort=False):
        ranked_slices.append(sort_by_rank_metric(group, rank_label).head(limit))
    if not ranked_slices:
        return frame.iloc[0:0].copy()
    limited = pd.concat(ranked_slices, ignore_index=True)
    return sort_by_rank_metric(limited, rank_label)


def limit_total_configs(frame: pd.DataFrame, rank_label: str, total_limit: str) -> pd.DataFrame:
    if frame.empty or total_limit == "All":
        return frame
    limit = int(total_limit.replace("Top ", ""))
    return sort_by_rank_metric(frame, rank_label).head(limit)


def average_forecast_paths_by_build(paths: pd.DataFrame, ranked_models: pd.DataFrame) -> pd.DataFrame:
    if paths.empty or ranked_models.empty:
        return paths.iloc[0:0].copy()

    model_lookup = ranked_models[
        ["model_config_id", "model_family", "model_build", "model_build_label"]
    ].drop_duplicates()
    paths_with_build = paths.merge(model_lookup, on="model_config_id", how="inner", suffixes=("", "_model"))
    if paths_with_build.empty:
        return paths.iloc[0:0].copy()

    aggregation = {
        "actual": "first",
        "prediction": "mean",
    }
    for column in ["baseline_prediction", "seasonal_naive_prediction"]:
        if column in paths_with_build.columns:
            aggregation[column] = "mean"

    averaged = (
        paths_with_build.groupby(["model_family", "model_build", "model_build_label", "target_date"], as_index=False)
        .agg(aggregation)
        .sort_values(["model_family", "model_build", "target_date"])
    )
    build_order = {build: index for index, build in enumerate(MODEL_BUILD_ORDER)}
    build_rank = {
        build: index + 1
        for index, build in enumerate(
            sorted(averaged["model_build"].dropna().unique(), key=lambda value: build_order.get(str(value), 999))
        )
    }
    averaged["rank"] = averaged["model_build"].map(build_rank)
    averaged["model_config_id"] = "avg::" + averaged["model_build"].astype(str)
    averaged["feature_family_name"] = "Average of selected configs"
    averaged["mode"] = "average"
    return averaged


def average_performance_by_build(performance: pd.DataFrame, ranked_models: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or ranked_models.empty:
        return performance.iloc[0:0].copy()

    model_lookup = ranked_models[
        ["config_id", "model_family", "model_build", "model_build_label"]
    ].drop_duplicates()
    perf_with_build = performance.merge(model_lookup, on="config_id", how="inner", suffixes=("", "_model"))
    if perf_with_build.empty:
        return performance.iloc[0:0].copy()

    averaged = (
        perf_with_build.groupby(["model_family", "model_build", "model_build_label", "as_of_date"], as_index=False)
        .agg(rolling_6mo_mae=("rolling_6mo_mae", "mean"))
        .sort_values(["model_family", "model_build", "as_of_date"])
    )
    averaged["config_id"] = "avg::" + averaged["model_build"].astype(str)
    return averaged


def aggregate_metric_mapping(frame: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    x_col, _ = RANK_METRIC_OPTIONS[x_metric]
    y_col, _ = RANK_METRIC_OPTIONS[y_metric]
    group_cols = ["model_family", "model_build", "model_build_label"]
    optional_cols = [
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score_balanced",
        "shock_penalty",
        "rmse_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
        "complexity_score",
        "interpretability_score",
        "compute_score",
    ]
    agg_cols = sorted({x_col, y_col, *[column for column in optional_cols if column in frame.columns]})
    agg_spec = {column: (column, "mean") for column in agg_cols}
    summary = frame.groupby(group_cols, as_index=False).agg(**agg_spec)
    counts = frame.groupby(group_cols, as_index=False).size().rename(columns={"size": "configurations"})
    summary = summary.merge(counts, on=group_cols, how="left")
    summary["feature_family_name"] = "average of selected slice"
    summary["feature_policy"] = "mixed"
    summary["feature_transform_label"] = "mixed"
    return summary


def metric_mapping_hover_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "model_family",
        "model_build_label",
        "model_build",
        "feature_family_name",
        "feature_transform_label",
        "feature_policy",
        "mode",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score_balanced",
        "shock_penalty",
        "rmse_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
        "complexity_score",
        "interpretability_score",
        "compute_score",
        "configurations",
    ]
    return [column for column in columns if column in frame.columns]


def metric_mapping_chart(
    frame: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_by: str,
    aggregate_points: bool,
) -> go.Figure:
    x_col, _ = RANK_METRIC_OPTIONS[x_metric]
    y_col, _ = RANK_METRIC_OPTIONS[y_metric]
    plot_frame = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_frame.empty:
        fig = go.Figure()
        fig.update_layout(title="No matching metric points")
        return fig

    size_col = "configurations" if aggregate_points and "configurations" in plot_frame.columns else None
    fig = px.scatter(
        plot_frame,
        x=x_col,
        y=y_col,
        color=color_by if color_by in plot_frame.columns else "model_build",
        size=size_col,
        hover_data=metric_mapping_hover_columns(plot_frame),
        category_orders={
            "model_family": MODEL_FAMILY_ORDER,
            "model_build": MODEL_BUILD_ORDER,
        },
        labels={
            x_col: x_metric,
            y_col: y_metric,
            "model_family": "Model family",
            "model_build_label": "Model build",
            "model_build": "Model build",
            "feature_policy": "Feature policy",
            "feature_transform_label": "Feature transform",
        },
    )
    fig.update_traces(marker=dict(opacity=0.78, line=dict(width=0.5, color="white")))
    fig.update_layout(
        title=f"{y_metric} vs {x_metric}",
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def render_missing_artifacts(base_dir: Path) -> None:
    inject_site_theme()
    render_project_banner()
    st.title("Transit Forecasting Lab")
    st.info("Dashboard artifacts were not found yet.")
    st.write("Expected local artifact folder:")
    st.code(str(base_dir), language="text")
    st.write("Place a completed dashboard export there, or set:")
    st.code("export DASHBOARD_ARTIFACT_DIR=/path/to/dashboard/aws_streamlined/run_id=<run_id>", language="bash")
    st.write("Expected files:")
    st.code("\n".join(REQUIRED_FILES.values()), language="text")


def main() -> None:
    inject_site_theme()
    artifact_base = configured_artifact_dir()
    run_dirs = discover_run_dirs(artifact_base)
    if not run_dirs:
        render_missing_artifacts(artifact_base)
        return

    st.sidebar.title("Run")
    selected_dir = st.sidebar.selectbox(
        "Dashboard artifact folder",
        run_dirs,
        format_func=lambda path: path.name if path.name != "latest" else str(path),
    )

    artifacts = load_artifacts(selected_dir)
    forecast_paths = ensure_model_taxonomy(
        normalize_dates(artifacts["forecast_paths"], ["as_of_date", "target_date"])
    )
    performance = ensure_model_taxonomy(
        normalize_dates(artifacts["performance_over_time"], ["as_of_date", "target_date"])
    )
    curated_leaderboard = enrich_score_columns(ensure_model_taxonomy(artifacts["model_leaderboard"]))
    leaderboard_source = artifacts.get("model_leaderboard_full", artifacts["model_leaderboard"])
    leaderboard = enrich_score_columns(ensure_model_taxonomy(leaderboard_source))
    family_summary = artifacts.get("feature_family_summary_full", artifacts["feature_family_summary"]).copy()
    champion_predictions = normalize_dates(artifacts["champion_predictions"], ["as_of_date", "target_date"])
    champion = artifacts["champion_selection"]
    experiment_manifest = artifacts.get("experiment_manifest", {})
    feature_families = load_feature_family_definitions()
    phase_a_config_text = load_config_text(PHASE_A_V3_CONFIG_PATH)
    phase_b_config_text = load_config_text(PHASE_B_V3_CONFIG_PATH)
    phase_c_config_text = load_config_text(PHASE_C_MONTHLY_CONFIG_PATH)
    overview_top_models = enrich_score_columns(
        ensure_model_taxonomy(artifacts.get("overview_top_models", leaderboard.head(5).copy()))
    )
    overview_prediction_paths = artifacts.get("overview_prediction_paths")
    if overview_prediction_paths is None:
        top_configs = overview_top_models["config_id"].head(5).tolist()
        overview_prediction_paths = forecast_paths[forecast_paths["config_id"].isin(top_configs)].copy()
        rank_lookup = dict(zip(top_configs, range(1, len(top_configs) + 1)))
        overview_prediction_paths["rank"] = overview_prediction_paths["config_id"].map(rank_lookup)
    overview_prediction_paths = ensure_model_taxonomy(
        normalize_dates(overview_prediction_paths, ["as_of_date", "target_date"])
    )
    for frame in [
        forecast_paths,
        performance,
        leaderboard,
        overview_top_models,
        overview_prediction_paths,
    ]:
        if "feature_policy" not in frame:
            frame["feature_policy"] = "none"
    candidate_leaderboard = exclude_baseline_candidates(leaderboard)

    render_project_banner()
    render_dashboard_header(champion, forecast_paths, experiment_manifest)

    section_options = [
        "Project Overview",
        "Data",
        "System",
        "Experiment",
        "Model Explorer",
        "Insights",
    ]
    active_section = st.segmented_control(
        "Dashboard section",
        section_options,
        default=section_options[0],
        key="dashboard_section",
        label_visibility="collapsed",
        width="stretch",
    )
    if active_section is None:
        active_section = section_options[0]

    if active_section == "Project Overview":
        overview_intro_cols = st.columns(2)
        overview_intro_cols[0].markdown(PROJECT_OVERVIEW_CASE_STUDY)
        overview_intro_cols[1].markdown(PROJECT_OVERVIEW_SYSTEM)
        st.markdown(PROJECT_OVERVIEW)
        render_public_bundle_note(experiment_manifest)

        bundle = experiment_manifest.get("public_dashboard_bundle", {})
        full_config_count = bundle.get("source_configurations") or experiment_manifest.get("model_config_count") or len(leaderboard)
        full_prediction_count = experiment_manifest.get("prediction_count") or len(forecast_paths)
        displayed_prediction_count = len(forecast_paths)

        overview_cols = st.columns(6)
        overview_cols[0].metric(
            "Forecast Horizon",
            f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', '-'))} months",
        )
        overview_cols[1].metric("Full Model Configs", format_int(full_config_count))
        overview_cols[2].metric("Indexed Configs", format_int(len(leaderboard)))
        overview_cols[3].metric("Full Rolling Predictions", format_int(full_prediction_count))
        overview_cols[4].metric("Flat Path Predictions", format_int(displayed_prediction_count))
        overview_cols[5].metric("Target Window", date_range_label(forecast_paths, "target_date"))

    elif active_section == "Data":
        render_data_page(family_summary, leaderboard, forecast_paths, champion, feature_families)

    elif active_section == "System":
        system_cols = st.columns(2)
        system_cols[0].markdown(SYSTEM_ARCHITECTURE)
        system_cols[1].markdown(SYSTEM_REASONING)
        st.markdown(SYSTEM_OVERVIEW)
        render_looping_system_video(
            SYSTEM_ARCH_VIDEO_PATH,
            "End-to-End System Architecture",
            (
                "A short looping walkthrough of the pipeline shape: source ingestion, S3 storage layers, "
                "ECS and Step Functions processing, local extended experiments, the DuckDB mart, and the "
                "Streamlit dashboard artifact bundle."
            ),
        )
        render_system_screenshot(
            STEP_FUNCTION_SCREENSHOT_PATH,
            "AWS Step Functions Run",
            (
                "The AWS orchestration view for the streamlined pipeline, showing parallel normalization, "
                "integration, feature-table creation, manifest writing, and smoke-scale model training "
                "running through ECS Fargate tasks."
            ),
        )
        render_system_artifact_flow(experiment_manifest, len(forecast_paths))

    elif active_section == "Experiment":
        render_experiment_page(
            family_summary,
            feature_families,
            leaderboard,
            forecast_paths,
            performance,
            champion,
            experiment_manifest,
            phase_a_config_text,
            phase_b_config_text,
            phase_c_config_text,
        )

    elif active_section == "Model Explorer":
        render_champion_snapshot(champion, forecast_paths, experiment_manifest)
        st.subheader("Top Models Against Actual Ridership")
        filter_cols = st.columns([2.1, 1.0, 1.2])
        selected_model_build_labels = optional_multiselect(
            "Model builds",
            pd.Series(ordered_model_build_labels(candidate_leaderboard)),
            "overview_model_builds",
            filter_cols[0],
        )
        build_scope = apply_optional_multi_filter(candidate_leaderboard, "model_build_label", selected_model_build_labels)
        selected_mode = all_selectbox(
            "Mode",
            build_scope["mode"],
            "overview_mode",
            filter_cols[1],
        )
        metric_label = filter_cols[2].selectbox(
            "Rank by",
            available_rank_options(leaderboard),
            key="overview_metric",
        )
        mode_scope = apply_optional_filter(build_scope, "mode", selected_mode)
        filter_cols_secondary = st.columns([1.7, 1.15, 1.25, 0.9, 0.9, 1.1])
        selected_feature_families = optional_multiselect(
            "Feature families",
            mode_scope["feature_family_name"],
            "overview_feature_families",
            filter_cols_secondary[0],
        )
        feature_scope = apply_optional_multi_filter(mode_scope, "feature_family_name", selected_feature_families)
        selected_feature_policies = optional_multiselect(
            "Feature policies",
            feature_scope["feature_policy"],
            "overview_feature_policies",
            filter_cols_secondary[1],
        )
        policy_scope = apply_optional_multi_filter(feature_scope, "feature_policy", selected_feature_policies)
        selected_feature_transforms = optional_multiselect(
            "Feature transforms",
            policy_scope["feature_transform_label"],
            "overview_feature_transforms",
            filter_cols_secondary[2],
        )
        overview_per_build_limit = filter_cols_secondary[3].selectbox(
            "Configs per build",
            PER_BUILD_LIMIT_OPTIONS,
            index=0,
            key="overview_per_build_limit",
        )
        overview_total_limit = filter_cols_secondary[4].selectbox(
            "Top total",
            TOTAL_LIMIT_OPTIONS,
            index=0,
            key="overview_total_limit",
            help="Applies after the per-build cap. For example, use Top 3 per build and Top 15 total.",
        )
        overview_path_mode = filter_cols_secondary[5].selectbox(
            "Path mode",
            ["Each configuration", "Average by model build"],
            key="overview_path_mode",
        )
        filtered_top = filtered_frame(
            candidate_leaderboard,
            model_build_label=selected_model_build_labels,
            mode=selected_mode,
            feature_family=selected_feature_families,
            feature_policy=selected_feature_policies,
            feature_transform=selected_feature_transforms,
        )
        metric = RANK_METRIC_OPTIONS[metric_label][0]
        if metric in filtered_top:
            filtered_top = sort_by_rank_metric(filtered_top, metric_label).copy()
        filtered_top = limit_configs_per_build(filtered_top, metric_label, overview_per_build_limit)
        filtered_top = limit_total_configs(filtered_top, metric_label, overview_total_limit)

        candidate_configs = filtered_top["model_config_id"].tolist()
        chart_paths = load_forecast_rows_for_configs(selected_dir, filtered_top, candidate_configs)
        target_min, target_max = date_bounds(chart_paths if not chart_paths.empty else forecast_paths, "target_date")
        default_target_start, default_target_end = default_target_window_for_rank(
            metric_label,
            target_min,
            target_max,
        )
        date_cols = st.columns([1, 1, 3])
        overview_target_start = date_cols[0].date_input(
            "Target start",
            default_target_start.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key=f"overview_target_start_{metric_label}",
        )
        overview_target_end = date_cols[1].date_input(
            "Target end",
            default_target_end.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key=f"overview_target_end_{metric_label}",
        )
        windowed_paths = apply_date_window(
            chart_paths,
            "target_date",
            overview_target_start,
            overview_target_end,
        )
        if overview_path_mode == "Average by model build":
            selected_models = filtered_top.copy()
            chart_paths = average_forecast_paths_by_build(windowed_paths, filtered_top)
            duplicate_count = 0
        else:
            if overview_total_limit == "All":
                max_paths = 25 if overview_per_build_limit != "All" else 10
            else:
                max_paths = int(overview_total_limit.replace("Top ", ""))
            selected_models, chart_paths, duplicate_count = select_distinct_model_paths(
                filtered_top,
                windowed_paths,
                max_models=max_paths,
            )
            chart_paths = chart_paths.sort_values(["rank", "target_date"])

        if metric_label in PERIOD_RANK_WINDOWS:
            st.caption(
                f"`{metric_label}` ranks models using a period-specific metric, so the chart "
                "defaults to that target-date period. You can widen the dates manually."
            )
        if duplicate_count:
            st.caption(
                f"Skipped {duplicate_count} duplicate prediction path(s) so the chart shows distinct lines. "
                "This commonly happens when seasonal-naive rows are repeated across feature-family labels."
            )

        st.plotly_chart(
            top_model_chart(chart_paths, "Selected Model Paths"),
            use_container_width=True,
        )
        render_metric_dataframe(overview_table(selected_models.head(50)))

        with st.expander("Champion selection rule"):
            st.write(champion.get("selection_rule", "No selection rule recorded."))

        with st.expander("How period-specific metrics are interpreted"):
            st.markdown(PERIOD_METRIC_EXPLANATION)

        with st.expander("Period metrics and derived ratios"):
            st.markdown(PERIOD_METRIC_SHORT_EXPLANATION)

        with st.expander("Feature policies, representations, and complexity scores"):
            st.markdown(REPRESENTATION_AND_COMPLEXITY_EXPLANATION)

        st.subheader("Rolling Error Over Time")
        rolling_date_cols = st.columns([1, 1, 2])
        rolling_configs = filtered_top["config_id"].tolist()
        rolling_subset = load_performance_rows_for_configs(selected_dir, filtered_top, rolling_configs)
        overview_as_of_min, overview_as_of_max = date_bounds(rolling_subset if not rolling_subset.empty else performance, "as_of_date")
        overview_as_of_start = rolling_date_cols[0].date_input(
            "As-of start",
            overview_as_of_min.date(),
            min_value=overview_as_of_min.date(),
            max_value=overview_as_of_max.date(),
            key="overview_rolling_as_of_start",
        )
        overview_as_of_end = rolling_date_cols[1].date_input(
            "As-of end",
            overview_as_of_max.date(),
            min_value=overview_as_of_min.date(),
            max_value=overview_as_of_max.date(),
            key="overview_rolling_as_of_end",
        )
        rolling_subset = apply_date_window(
            rolling_subset,
            "as_of_date",
            overview_as_of_start,
            overview_as_of_end,
        )
        if overview_path_mode == "Average by model build":
            rolling_subset = average_performance_by_build(rolling_subset, filtered_top)
        else:
            rolling_subset = rolling_subset[
                rolling_subset["config_id"].isin(filtered_top.head(12)["config_id"])
            ].copy()
        st.plotly_chart(rolling_error_chart(rolling_subset), use_container_width=True)

        st.subheader("Model Build Comparison")
        comparison_metric = RANK_METRIC_OPTIONS[metric_label][0]
        comparison_ascending = RANK_METRIC_OPTIONS[metric_label][1]
        comparison_agg = "min" if comparison_ascending else "max"
        if comparison_metric in filtered_top.columns and not filtered_top.empty:
            model_summary = (
                filtered_top.groupby(["model_family", "model_build", "model_build_label"], as_index=False)
                .agg(
                    best_mae=("mae", "min"),
                    best_rmse=("rmse", "min"),
                    best_metric=(comparison_metric, comparison_agg),
                )
                .sort_values("best_metric", ascending=comparison_ascending)
            )
            fig = px.bar(
                model_summary,
                x="model_build_label",
                y="best_metric",
                color="model_family",
                category_orders={
                    "model_family": MODEL_FAMILY_ORDER,
                    "model_build_label": ordered_model_build_labels(model_summary),
                },
                labels={
                    "model_build_label": "Model build",
                    "model_family": "Model family",
                    "best_metric": f"Best {metric_label}",
                },
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#2f323a"),
            )
            if comparison_metric in {"r2", "r2_adjusted"}:
                fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Selected Model Forecast Rows")
        detail_candidates = filtered_top.copy().reset_index(drop=True)
        if detail_candidates.empty:
            st.info("No model configurations are available for the selected Model Explorer filters.")
        else:
            detail_candidates["rank"] = range(1, len(detail_candidates) + 1)
            detail_labels = [
                ranked_model_label(row, metric_label)
                for _, row in detail_candidates.iterrows()
            ]
            detail_label_by_config = dict(zip(detail_candidates["config_id"], detail_labels))
            default_detail_config = (
                selected_models.iloc[0]["config_id"]
                if not selected_models.empty and selected_models.iloc[0].get("config_id") in detail_label_by_config
                else detail_candidates.iloc[0]["config_id"]
            )
            selected_detail_label = st.selectbox(
                "Model for forecast-row table",
                detail_labels,
                index=detail_labels.index(detail_label_by_config[default_detail_config]),
                key="overview_detail_model",
            )
            detail_config_id = dict(zip(detail_labels, detail_candidates["config_id"]))[selected_detail_label]
            detail_model = detail_candidates[detail_candidates["config_id"] == detail_config_id].iloc[0]
            st.caption(
                f"Feature transform: {detail_model.get('feature_transform_label', 'No transform')}. "
                "Hyperparameters: "
                f"{parse_json_display(detail_model.get('hyperparameters_json', '{}'))}. "
                f"Table follows the selected model over the active target-date window."
            )
            detail_forecast = load_forecast_rows_for_configs(
                selected_dir,
                detail_candidates[detail_candidates["config_id"] == detail_config_id],
                [detail_config_id],
            )
            detail_forecast = detail_forecast.sort_values("target_date")
            detail_forecast = apply_date_window(
                detail_forecast,
                "target_date",
                overview_target_start,
                overview_target_end,
            )
            detail_columns = [
                "as_of_date",
                "target_date",
                "actual",
                "prediction",
                "seasonal_naive_prediction",
                "error",
                "abs_error",
                "evaluation_period",
            ]
            detail_columns = [column for column in detail_columns if column in detail_forecast.columns]
            st.dataframe(
                detail_forecast[detail_columns],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Metric Mapping")
        st.write(
            "Map model configurations across two evaluation metrics to inspect tradeoffs, "
            "clusters, and cases where a model family performs well on one dimension but "
            "not another."
        )
        metric_options = available_rank_options(leaderboard)
        build_options = ordered_model_build_labels(candidate_leaderboard)
        control_cols = st.columns([1.25, 1.25, 1.15, 0.9, 0.9, 1.1])
        x_metric = control_cols[0].selectbox(
            "X axis",
            metric_options,
            index=selectbox_index(metric_options, "MAE"),
            key="mapping_x_metric",
        )
        y_metric = control_cols[1].selectbox(
            "Y axis",
            metric_options,
            index=selectbox_index(metric_options, "R-squared"),
            key="mapping_y_metric",
        )
        rank_metric = control_cols[2].selectbox(
            "Rank slice by",
            metric_options,
            index=selectbox_index(metric_options, "Balanced score"),
            key="mapping_rank_metric",
        )
        per_build_limit = control_cols[3].selectbox(
            "Configs per build",
            PER_BUILD_LIMIT_OPTIONS,
            index=1,
            key="mapping_per_build_limit",
        )
        total_limit = control_cols[4].selectbox(
            "Top total",
            TOTAL_LIMIT_OPTIONS,
            index=0,
            key="mapping_total_limit",
            help="Applies after the per-build cap.",
        )
        point_mode = control_cols[5].selectbox(
            "Point mode",
            ["Each configuration", "Average by model build"],
            key="mapping_point_mode",
        )
        selected_builds = st.multiselect(
            "Model builds",
            build_options,
            default=build_options,
            key="mapping_model_builds",
        )
        mapping_filter_cols = st.columns([1.35, 1.1, 1.1, 0.9])
        selected_mapping_families = optional_multiselect(
            "Feature families",
            candidate_leaderboard["feature_family_name"],
            "mapping_feature_families",
            mapping_filter_cols[0],
        )
        mapping_family_scope = apply_optional_multi_filter(
            candidate_leaderboard,
            "feature_family_name",
            selected_mapping_families,
        )
        selected_mapping_policies = optional_multiselect(
            "Feature policies",
            mapping_family_scope["feature_policy"],
            "mapping_feature_policies",
            mapping_filter_cols[1],
        )
        mapping_policy_scope = apply_optional_multi_filter(
            mapping_family_scope,
            "feature_policy",
            selected_mapping_policies,
        )
        selected_mapping_transforms = mapping_filter_cols[2].multiselect(
            "Feature transforms",
            ordered_unique(mapping_policy_scope["feature_transform_label"]),
            default=[],
            key="mapping_feature_transforms",
            placeholder="All",
            help="Leave empty to include all transforms.",
        )
        color_by = mapping_filter_cols[3].selectbox(
            "Color by",
            ["model_family", "model_build", "feature_transform_label", "feature_policy", "mode"],
            key="mapping_color_by",
        )

        mapping_frame = metric_mapping_frame(
            candidate_leaderboard,
            selected_builds,
            rank_metric,
            per_build_limit,
            total_limit,
            selected_mapping_families,
            selected_mapping_policies,
            selected_mapping_transforms,
        )
        aggregate_points = point_mode == "Average by model build"
        if aggregate_points and not mapping_frame.empty:
            mapping_frame = aggregate_metric_mapping(mapping_frame, x_metric, y_metric)

        if mapping_frame.empty:
            st.info("No model configurations match the selected metric mapping controls.")
        else:
            st.caption(
                f"Showing {format_int(len(mapping_frame))} point(s). "
                f"Slice is selected by `{rank_metric}` within each chosen model build."
            )
            st.plotly_chart(
                metric_mapping_chart(
                    mapping_frame,
                    x_metric,
                    y_metric,
                    color_by,
                    aggregate_points,
                ),
                use_container_width=True,
            )
            table_cols = [
                "model_build_label",
                "mode",
                "feature_transform_label",
                "feature_policy",
                "feature_family_name",
                "configurations",
                "mae",
                "rmse",
                "r2",
                "r2_adjusted",
                "diracc",
                "selection_score_balanced",
                "pre_covid_mae",
                "pre_covid_rmse",
                "covid_shock_mae",
                "covid_shock_rmse",
                "recovery_mae",
                "recovery_rmse",
                "recent_mae",
                "recent_rmse",
                "shock_penalty",
                "rmse_shock_penalty",
                "recovery_ratio",
                "rmse_recovery_ratio",
                "recent_recovery_ratio",
                "rmse_recent_recovery_ratio",
                "complexity_score",
                "interpretability_score",
                "compute_score",
            ]
            available_mapping_cols = [column for column in table_cols if column in mapping_frame.columns]
            render_metric_dataframe(mapping_frame[available_mapping_cols], max_rows=200)

        st.subheader("Operational Footprint")
        if {"model_type", "total_train_seconds", "avg_train_seconds"}.issubset(leaderboard.columns):
            runtime = (
                leaderboard.groupby("model_type", as_index=False)
                .agg(
                    total_train_seconds=("total_train_seconds", "sum"),
                    avg_train_seconds=("avg_train_seconds", "mean"),
                    configs=("config_id", "nunique"),
                )
                .sort_values("total_train_seconds", ascending=False)
            )
            st.dataframe(runtime, use_container_width=True, hide_index=True)
            fig = px.bar(
                runtime,
                x="model_type",
                y="total_train_seconds",
                color="model_type",
                labels={"model_type": "Model type", "total_train_seconds": "Total train seconds"},
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#2f323a"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Loaded Artifacts**")
        st.code(str(selected_dir), language="text")
        st.write("This app reads curated dashboard artifacts only. It does not trigger training jobs.")

    elif active_section == "Insights":
        render_insights_page(
            selected_dir,
            leaderboard,
            forecast_paths,
            performance,
            champion,
            experiment_manifest,
        )

if __name__ == "__main__":
    main()
