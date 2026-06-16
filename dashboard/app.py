import sys
from pathlib import Path

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from constants import (
    PHASE_A_V3_CONFIG_PATH,
    PHASE_B_V3_CONFIG_PATH,
    PHASE_C_MONTHLY_CONFIG_PATH,
    REQUIRED_FILES,
)
from data_access import (
    configured_artifact_dir,
    discover_run_dirs,
    load_artifacts,
    load_config_text,
    load_feature_family_definitions,
)
from model_helpers import (
    enrich_score_columns,
    ensure_model_taxonomy,
    normalize_dates,
)
from pages.experiment import render_experiment_page
from pages.data import render_data_page
from pages.insights import render_insights_page
from pages.model_explorer import render_model_explorer_page
from pages.overview import render_overview_page
from pages.system import render_system_page


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
    render_project_banner()

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
        render_overview_page(experiment_manifest, leaderboard, forecast_paths, champion)

    elif active_section == "Data":
        render_data_page(family_summary, leaderboard, forecast_paths, champion, feature_families)

    elif active_section == "System":
        render_system_page(experiment_manifest, len(forecast_paths))

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
        render_model_explorer_page(
            selected_dir,
            leaderboard,
            forecast_paths,
            performance,
            champion,
            experiment_manifest,
        )

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
