import sys
from pathlib import Path
from typing import Optional

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from constants import (
    OPTIONAL_FILES,
    PHASE_A_V3_CONFIG_PATH,
    PHASE_B_V3_CONFIG_PATH,
    PHASE_C_MONTHLY_CONFIG_PATH,
    REQUIRED_FILES,
)
from data_access import (
    configured_artifact_dir,
    discover_run_dirs,
    load_config_text,
    load_feature_family_definitions,
    load_json,
    load_parquet,
    file_modified_ns,
)
from model_helpers import (
    enrich_score_columns,
    ensure_model_taxonomy,
    normalize_dates,
)

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


def _required_artifact_path(run_dir: Path, key: str) -> Path:
    return run_dir / REQUIRED_FILES[key]


def _optional_artifact_path(run_dir: Path, key: str) -> Path:
    return run_dir / OPTIONAL_FILES[key]


def load_required_parquet(run_dir: Path, key: str):
    path = _required_artifact_path(run_dir, key)
    return load_parquet(str(path), file_modified_ns(path))


def load_optional_parquet(run_dir: Path, key: str):
    path = _optional_artifact_path(run_dir, key)
    if not path.exists():
        return None
    return load_parquet(str(path), file_modified_ns(path))


def load_required_json(run_dir: Path, key: str) -> dict:
    path = _required_artifact_path(run_dir, key)
    return load_json(str(path), file_modified_ns(path))


def load_optional_json(run_dir: Path, key: str) -> dict:
    path = _optional_artifact_path(run_dir, key)
    if not path.exists():
        return {}
    return load_json(str(path), file_modified_ns(path))


def load_public_bundle_manifest(run_dir: Path) -> dict:
    path = run_dir / "public_bundle_manifest.json"
    if not path.exists():
        return {}
    return load_json(str(path), file_modified_ns(path))


def merge_public_bundle_counts(experiment_manifest: dict, public_manifest: dict) -> dict:
    if not public_manifest:
        return experiment_manifest

    merged = dict(experiment_manifest)
    bundle = dict(merged.get("public_dashboard_bundle", {}))
    bundle.setdefault("source_configurations", public_manifest.get("source_configurations"))
    bundle.setdefault("full_metadata_configurations", public_manifest.get("full_leaderboard_rows"))
    bundle.setdefault("selected_configurations", public_manifest.get("selected_configurations"))
    bundle.setdefault("keep_fraction", public_manifest.get("keep_fraction"))
    bundle.setdefault(
        "full_path_rows",
        {
            "forecast_paths": public_manifest.get("full_forecast_rows"),
            "performance_over_time": public_manifest.get("full_performance_rows"),
        },
    )
    bundle.setdefault(
        "flat_path_rows",
        {
            "forecast_paths": public_manifest.get("forecast_rows"),
            "performance_over_time": public_manifest.get("performance_rows"),
        },
    )
    merged["public_dashboard_bundle"] = bundle
    return merged


def ensure_feature_policy_column(*frames) -> None:
    for frame in frames:
        if frame is not None and "feature_policy" not in frame:
            frame["feature_policy"] = "none"


def load_leaderboard(run_dir: Path, *, full: bool = True):
    source = load_optional_parquet(run_dir, "model_leaderboard_full") if full else None
    if source is None:
        source = load_required_parquet(run_dir, "model_leaderboard")
    leaderboard = enrich_score_columns(ensure_model_taxonomy(source))
    ensure_feature_policy_column(leaderboard)
    return leaderboard


def load_family_summary(run_dir: Path):
    summary = load_optional_parquet(run_dir, "feature_family_summary_full")
    if summary is None:
        summary = load_required_parquet(run_dir, "feature_family_summary")
    return summary.copy()


def load_complexity_profile(run_dir: Path):
    profile = load_optional_parquet(run_dir, "complexity_profile_full")
    if profile is None:
        profile = load_optional_parquet(run_dir, "complexity_profile")
    if profile is None:
        return None
    profile = ensure_model_taxonomy(profile)
    ensure_feature_policy_column(profile)
    return profile


def load_champion_predictions(run_dir: Path):
    predictions = normalize_dates(
        load_required_parquet(run_dir, "champion_predictions"),
        ["as_of_date", "target_date"],
    )
    predictions = ensure_model_taxonomy(predictions)
    ensure_feature_policy_column(predictions)
    return predictions


def load_overview_predictions(run_dir: Path):
    predictions = load_optional_parquet(run_dir, "overview_prediction_paths")
    if predictions is None:
        predictions = load_required_parquet(run_dir, "champion_predictions")
    predictions = ensure_model_taxonomy(normalize_dates(predictions, ["as_of_date", "target_date"]))
    ensure_feature_policy_column(predictions)
    return predictions


def load_forecast_paths(run_dir: Path):
    paths = ensure_model_taxonomy(
        normalize_dates(load_required_parquet(run_dir, "forecast_paths"), ["as_of_date", "target_date"])
    )
    ensure_feature_policy_column(paths)
    return paths


def load_performance_rows(run_dir: Path):
    performance = ensure_model_taxonomy(
        normalize_dates(load_required_parquet(run_dir, "performance_over_time"), ["as_of_date", "target_date"])
    )
    ensure_feature_policy_column(performance)
    return performance


def flat_forecast_row_count(experiment_manifest: dict, fallback: Optional[int] = None) -> Optional[int]:
    bundle = experiment_manifest.get("public_dashboard_bundle", {})
    flat_path_rows = bundle.get("flat_path_rows", {})
    if isinstance(flat_path_rows, dict) and flat_path_rows.get("forecast_paths") is not None:
        return flat_path_rows.get("forecast_paths")
    return fallback


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

    champion = load_required_json(selected_dir, "champion_selection")
    experiment_manifest = merge_public_bundle_counts(
        load_optional_json(selected_dir, "experiment_manifest"),
        load_public_bundle_manifest(selected_dir),
    )
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
        from sections.overview import render_overview_page

        overview_prediction_paths = load_overview_predictions(selected_dir)
        render_overview_page(
            experiment_manifest,
            overview_prediction_paths,
            champion,
            flat_forecast_row_count(experiment_manifest, len(overview_prediction_paths)),
            experiment_manifest.get("public_dashboard_bundle", {}).get("source_configurations"),
        )

    elif active_section == "Data":
        from sections.data import render_data_page

        champion_predictions = load_champion_predictions(selected_dir)
        render_data_page(champion_predictions, champion)

    elif active_section == "System":
        from sections.system import render_system_page

        render_system_page(experiment_manifest, flat_forecast_row_count(experiment_manifest))

    elif active_section == "Experiment":
        from sections.experiment import render_experiment_page

        family_summary = load_family_summary(selected_dir)
        feature_families = load_feature_family_definitions()
        leaderboard = load_leaderboard(selected_dir)
        forecast_paths = load_forecast_paths(selected_dir)
        performance = load_performance_rows(selected_dir)
        phase_a_config_text = load_config_text(PHASE_A_V3_CONFIG_PATH)
        phase_b_config_text = load_config_text(PHASE_B_V3_CONFIG_PATH)
        phase_c_config_text = load_config_text(PHASE_C_MONTHLY_CONFIG_PATH)
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
        from sections.model_explorer import render_model_explorer_page

        leaderboard = load_leaderboard(selected_dir)
        forecast_paths = load_forecast_paths(selected_dir)
        performance = load_performance_rows(selected_dir)
        complexity_profile = load_complexity_profile(selected_dir)
        render_model_explorer_page(
            selected_dir,
            leaderboard,
            forecast_paths,
            performance,
            complexity_profile,
            champion,
            experiment_manifest,
        )

    elif active_section == "Insights":
        from sections.insights import render_insights_page

        leaderboard = load_leaderboard(selected_dir)
        forecast_paths = load_forecast_paths(selected_dir)
        performance = load_performance_rows(selected_dir)
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
