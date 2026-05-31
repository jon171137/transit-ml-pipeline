import json
import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from content import (
        DATA_OVERVIEW,
        EXPERIMENT_OVERVIEW,
        PERIOD_METRIC_EXPLANATION,
        PERIOD_METRIC_SHORT_EXPLANATION,
        PROJECT_OVERVIEW,
        REPRESENTATION_AND_COMPLEXITY_EXPLANATION,
        SYSTEM_OVERVIEW,
    )
except ImportError:
    from dashboard.content import (
        DATA_OVERVIEW,
        EXPERIMENT_OVERVIEW,
        PERIOD_METRIC_EXPLANATION,
        PERIOD_METRIC_SHORT_EXPLANATION,
        PROJECT_OVERVIEW,
        REPRESENTATION_AND_COMPLEXITY_EXPLANATION,
        SYSTEM_OVERVIEW,
    )


DEFAULT_ARTIFACT_DIR = Path("dashboard/public_artifacts/latest")
IMAGE_ASSET_DIR = Path("dashboard/assets/images")
DEFAULT_FEATURE_FAMILIES_PATH = Path("dashboard/public_artifacts/latest/feature_families.json")
PHASE_A_V2_CONFIG_PATH = Path("experiment_configs/large_phase_a_v2_complexity.yaml")
PHASE_B_V2_CONFIG_PATH = Path("experiment_configs/phase_b_autoregressive_v2_complexity.yaml")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
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
REQUIRED_FILES = {
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
    "champion_predictions": "champion_predictions.parquet",
    "champion_selection": "champion_selection.json",
}
OPTIONAL_FILES = {
    "overview_top_models": "overview_top_models.parquet",
    "overview_prediction_paths": "overview_prediction_paths.parquet",
    "experiment_manifest": "experiment_manifest.json",
}

SCORE_RECIPES = {
    "typical": {"label": "Typical-error score", "mae_weight": 0.90, "rmse_weight": 0.10},
    "balanced": {"label": "Balanced score", "mae_weight": 0.75, "rmse_weight": 0.25},
    "large_error": {"label": "Large-error score", "mae_weight": 0.50, "rmse_weight": 0.50},
}
EVALUATION_PERIODS = {
    "pre_covid": "Pre-COVID",
    "covid_shock": "COVID shock",
    "recovery": "Recovery",
    "recent": "Recent",
}

RANK_METRIC_OPTIONS = {
    "Balanced score": ("selection_score_balanced", True),
    "Typical-error score": ("selection_score_typical", True),
    "Large-error score": ("selection_score_large_error", True),
    "MAE": ("mae", True),
    "RMSE": ("rmse", True),
    "R-squared": ("r2", False),
    "Adjusted R-squared": ("r2_adjusted", False),
    "Directional accuracy": ("diracc", False),
    "Pre-COVID MAE": ("pre_covid_mae", True),
    "Pre-COVID RMSE": ("pre_covid_rmse", True),
    "Pre-COVID typical-error score": ("pre_covid_selection_score_typical", True),
    "Pre-COVID balanced score": ("pre_covid_selection_score_balanced", True),
    "Pre-COVID large-error score": ("pre_covid_selection_score_large_error", True),
    "COVID shock MAE": ("covid_shock_mae", True),
    "COVID shock RMSE": ("covid_shock_rmse", True),
    "COVID shock typical-error score": ("covid_shock_selection_score_typical", True),
    "COVID shock balanced score": ("covid_shock_selection_score_balanced", True),
    "COVID shock large-error score": ("covid_shock_selection_score_large_error", True),
    "Recovery MAE": ("recovery_mae", True),
    "Recovery RMSE": ("recovery_rmse", True),
    "Recovery typical-error score": ("recovery_selection_score_typical", True),
    "Recovery balanced score": ("recovery_selection_score_balanced", True),
    "Recovery large-error score": ("recovery_selection_score_large_error", True),
    "Recent MAE": ("recent_mae", True),
    "Recent RMSE": ("recent_rmse", True),
    "Recent typical-error score": ("recent_selection_score_typical", True),
    "Recent balanced score": ("recent_selection_score_balanced", True),
    "Recent large-error score": ("recent_selection_score_large_error", True),
    "Shock penalty": ("shock_penalty", True),
    "Recovery ratio": ("recovery_ratio", True),
    "Recent recovery ratio": ("recent_recovery_ratio", True),
    "RMSE shock penalty": ("rmse_shock_penalty", True),
    "RMSE recovery ratio": ("rmse_recovery_ratio", True),
    "RMSE recent recovery ratio": ("rmse_recent_recovery_ratio", True),
    "Typical score shock penalty": ("typical_score_shock_penalty", True),
    "Typical score recovery ratio": ("typical_score_recovery_ratio", True),
    "Typical score recent recovery ratio": ("typical_score_recent_recovery_ratio", True),
    "Balanced score shock penalty": ("balanced_score_shock_penalty", True),
    "Balanced score recovery ratio": ("balanced_score_recovery_ratio", True),
    "Balanced score recent recovery ratio": ("balanced_score_recent_recovery_ratio", True),
    "Large-error score shock penalty": ("large_error_score_shock_penalty", True),
    "Large-error score recovery ratio": ("large_error_score_recovery_ratio", True),
    "Large-error score recent recovery ratio": ("large_error_score_recent_recovery_ratio", True),
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

PERIOD_RANK_WINDOWS = {
    "Pre-COVID MAE": (None, "2020-02-01"),
    "Pre-COVID RMSE": (None, "2020-02-01"),
    "Pre-COVID typical-error score": (None, "2020-02-01"),
    "Pre-COVID balanced score": (None, "2020-02-01"),
    "Pre-COVID large-error score": (None, "2020-02-01"),
    "COVID shock MAE": ("2020-03-01", "2021-06-01"),
    "COVID shock RMSE": ("2020-03-01", "2021-06-01"),
    "COVID shock typical-error score": ("2020-03-01", "2021-06-01"),
    "COVID shock balanced score": ("2020-03-01", "2021-06-01"),
    "COVID shock large-error score": ("2020-03-01", "2021-06-01"),
    "Shock penalty": ("2020-03-01", "2021-06-01"),
    "RMSE shock penalty": ("2020-03-01", "2021-06-01"),
    "Typical score shock penalty": ("2020-03-01", "2021-06-01"),
    "Balanced score shock penalty": ("2020-03-01", "2021-06-01"),
    "Large-error score shock penalty": ("2020-03-01", "2021-06-01"),
    "Recovery MAE": ("2021-07-01", "2022-12-01"),
    "Recovery RMSE": ("2021-07-01", "2022-12-01"),
    "Recovery typical-error score": ("2021-07-01", "2022-12-01"),
    "Recovery balanced score": ("2021-07-01", "2022-12-01"),
    "Recovery large-error score": ("2021-07-01", "2022-12-01"),
    "Recovery ratio": ("2021-07-01", "2022-12-01"),
    "RMSE recovery ratio": ("2021-07-01", "2022-12-01"),
    "Typical score recovery ratio": ("2021-07-01", "2022-12-01"),
    "Balanced score recovery ratio": ("2021-07-01", "2022-12-01"),
    "Large-error score recovery ratio": ("2021-07-01", "2022-12-01"),
    "Recent MAE": ("2023-01-01", None),
    "Recent RMSE": ("2023-01-01", None),
    "Recent typical-error score": ("2023-01-01", None),
    "Recent balanced score": ("2023-01-01", None),
    "Recent large-error score": ("2023-01-01", None),
    "Recent recovery ratio": ("2023-01-01", None),
    "RMSE recent recovery ratio": ("2023-01-01", None),
    "Typical score recent recovery ratio": ("2023-01-01", None),
    "Balanced score recent recovery ratio": ("2023-01-01", None),
    "Large-error score recent recovery ratio": ("2023-01-01", None),
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

        .block-container {
            padding-top: 1.6rem;
        }

        .portfolio-banner {
            border-top: 7px solid var(--portfolio-teal);
            background: linear-gradient(90deg, rgba(0, 127, 104, 0.11), rgba(7, 95, 237, 0.04));
            border-bottom: 1px solid rgba(0, 127, 104, 0.16);
            padding: 0.7rem 1rem;
            margin: -0.2rem 0 1.4rem;
            color: var(--portfolio-ink);
            font-size: 0.92rem;
            font-weight: 600;
            letter-spacing: 0;
        }

        .portfolio-banner span {
            color: var(--portfolio-teal-dark);
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
            display: grid;
            grid-template-columns: minmax(300px, 1.1fr) minmax(360px, 0.9fr);
            gap: 1.5rem;
            align-items: start;
            margin: 0.6rem 0 1.2rem;
        }

        .dashboard-title h1 {
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.08;
        }

        .dashboard-title p {
            color: var(--portfolio-muted);
            margin: 0.7rem 0 0;
            font-size: 0.96rem;
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

        @media (max-width: 1200px) {
            .compact-kpi-grid {
                grid-template-columns: repeat(2, minmax(130px, 1fr));
            }

            .dashboard-hero {
                grid-template-columns: 1fr;
            }
        }

        button[role="tab"][aria-selected="true"] {
            color: var(--portfolio-teal-dark);
            border-bottom-color: var(--portfolio-teal) !important;
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
            <span>Personal Forecasting Project</span> by
            <a href="https://www.linkedin.com/in/sellersjon" target="_blank" rel="noopener noreferrer">Jon Sellers</a>
            <span class="banner-link-divider">|</span>
            <a href="https://github.com/jon171137/transit-ml-pipeline" target="_blank" rel="noopener noreferrer">GitHub Repo</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_name_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def discover_dashboard_images(image_dir: Path = IMAGE_ASSET_DIR) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def render_image_gallery(title: str, intro: str = None) -> None:
    images = discover_dashboard_images()
    st.subheader(title)
    if intro:
        st.write(intro)

    if not images:
        st.info(
            "Add PNG, JPG, JPEG, or WebP files to "
            f"`{IMAGE_ASSET_DIR}` and they will appear here."
        )
        return

    for image_path in images:
        st.image(
            str(image_path),
            caption=display_name_from_path(image_path),
            use_container_width=True,
        )


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


def render_dashboard_header(
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
        ("Balanced score", format_int(champion.get("selection_score_balanced", champion.get("selection_score")))),
        ("MAE / RMSE", f"{format_int(champion.get('mae'))} / {format_int(champion.get('rmse'))}"),
        ("Target dates", date_range_label(forecast_paths, "target_date")),
        ("Horizon / cadence", f"{horizon} months / {cadence}"),
    ]
    st.markdown(
        """
        <div class="dashboard-hero">
            <div class="dashboard-title">
                <h1>Transit Forecasting Lab</h1>
                <p>Rolling H3 UPT forecast comparison across local and AWS-shaped pipeline artifacts</p>
            </div>
            <div class="champion-summary">
                <div class="summary-title">Current Champion Snapshot</div>
                <div class="champion-summary-grid">
        """
        + "".join(champion_summary_item(label, value) for label, value in summary_items)
        + """
                </div>
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


def render_public_bundle_note(experiment_manifest: dict) -> None:
    bundle = experiment_manifest.get("public_dashboard_bundle")
    if not bundle:
        return
    retained = format_int(bundle.get("selected_configurations"))
    source = format_int(bundle.get("source_configurations"))
    keep_fraction = bundle.get("keep_fraction")
    keep_pct = f"{float(keep_fraction) * 100:.0f}%" if keep_fraction is not None else "configured"
    st.info(
        "This live dashboard uses a curated public artifact bundle for speed. "
        f"It retains {retained} of {source} model configurations: models in the best "
        f"{keep_pct} for at least one core performance metric, plus the baseline and "
        "overall champion. The full local experiment output is preserved separately "
        "for deeper analysis."
    )


def render_data_page(
    family_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    champion: dict,
    feature_families: dict,
) -> None:
    st.subheader("Data")
    st.markdown(DATA_OVERVIEW)

    st.markdown("### Source And Processing Map")
    source_rows = [
        {
            "Source": "Transit monthly ridership and service",
            "What it contributes": "Target ridership plus service/supply context.",
            "Processing role": "Normalized to monthly grain, joined into the integrated base, then transformed into lags and rolling features.",
        },
        {
            "Source": "EIA gasoline price series",
            "What it contributes": "Transportation cost context.",
            "Processing role": "Normalized monthly and used directly plus year-over-year/change features.",
        },
        {
            "Source": "FRED CPI / inflation series",
            "What it contributes": "General and core price-pressure context.",
            "Processing role": "Normalized monthly, imputed when needed, then used as exogenous and interaction features.",
        },
        {
            "Source": "FRED King County median household income",
            "What it contributes": "Annual socioeconomic context.",
            "Processing role": "Converted to prior-year monthly context with income-growth and affordability-pressure features.",
        },
    ]
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

    st.markdown("### Feature Family Examples")
    if {"feature_family_name", "best_selection_score"}.issubset(family_summary.columns):
        family_display = family_summary.copy()
        family_display = family_display.sort_values("best_selection_score").head(20)
        st.dataframe(family_display, use_container_width=True, hide_index=True)
    else:
        family_cols = [col for col in ["feature_family_name", "mode"] if col in family_summary]
        st.dataframe(family_summary[family_cols].drop_duplicates(), use_container_width=True, hide_index=True)

    st.markdown("### Feature Family Definitions")
    st.write(
        "These are the named feature families available to the Phase A tabular "
        "models. A feature family is the human-readable modeling strategy before "
        "any model-specific feature policy such as correlation pruning, mutual "
        "information selection, or tree-importance selection is applied."
    )
    if feature_families:
        family_rows = [
            {
                "feature_family_name": family_name,
                "n_defined_features": len(features),
                "included_features": ", ".join(features),
            }
            for family_name, features in sorted(feature_families.items())
        ]
        st.dataframe(pd.DataFrame(family_rows), use_container_width=True, hide_index=True)

        selected_family = st.selectbox(
            "Inspect one feature family",
            sorted(feature_families),
            index=0,
            key="data_feature_family_definition_select",
        )
        selected_features = pd.DataFrame(
            {
                "position": range(1, len(feature_families[selected_family]) + 1),
                "feature_name": feature_families[selected_family],
            }
        )
        st.dataframe(selected_features, use_container_width=True, hide_index=True)

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
            "Examples": "is_covid_disruption, is_post_covid, months_since_covid_impact",
            "Purpose": "Let models distinguish ordinary history from disruption and recovery periods.",
        },
        {
            "Feature type": "Exogenous context",
            "Examples": "gas price, CPI, core CPI, income growth",
            "Purpose": "Test whether external economic pressure improves forecasts.",
        },
        {
            "Feature type": "Targeted interactions",
            "Examples": "income_yoy_pct_x_gas_price_yoy_diff, lag_x_regime flags",
            "Purpose": "Let linear models express selected non-additive relationships without a full polynomial explosion.",
        },
    ]
    st.dataframe(pd.DataFrame(feature_type_rows), use_container_width=True, hide_index=True)

    st.markdown("### Single Forecast Step Example")
    config_id = champion.get("model_config_id") or champion.get("config_id")
    sample = forecast_paths[forecast_paths["model_config_id"] == config_id].copy()
    if sample.empty:
        sample = forecast_paths.copy()
    if not sample.empty:
        sample = sample.sort_values("target_date").iloc[len(sample) // 2]
        target_date = pd.Timestamp(sample["target_date"])
        as_of_date = pd.Timestamp(sample["as_of_date"])
        months_since_covid = (target_date.year - 2020) * 12 + (target_date.month - 3)
        example_rows = [
            ("as_of_date", as_of_date.date().isoformat(), "Training data is limited to rows before this month."),
            ("target_date", target_date.date().isoformat(), "This is the month being forecast three months ahead."),
            ("target_month", target_date.strftime("%B"), "Seasonality features encode this month cyclically."),
            ("evaluation_period", sample.get("evaluation_period", "-"), "Used for pre-COVID, shock, recovery, and recent metrics."),
            ("months_since_covid_impact", months_since_covid, "A time-since-disruption signal for regime-aware features."),
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


def render_experiment_page(
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
    champion: dict,
    experiment_manifest: dict,
    phase_a_config_text: str,
    phase_b_config_text: str,
) -> None:
    st.subheader("Experiment")
    render_public_bundle_note(experiment_manifest)

    summary_rows = [
        {
            "Item": "Experiment bundle",
            "Current value": experiment_manifest.get("experiment_id") or experiment_manifest.get("run_id") or "-",
        },
        {
            "Item": "Target and horizon",
            "Current value": (
                f"{str(champion.get('target', experiment_manifest.get('target', 'upt'))).upper()}, "
                f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', 3))} months ahead"
            ),
        },
        {
            "Item": "As-of window",
            "Current value": date_range_label(forecast_paths, "as_of_date"),
        },
        {
            "Item": "Target window",
            "Current value": date_range_label(forecast_paths, "target_date"),
        },
        {
            "Item": "Model configurations",
            "Current value": format_int(len(leaderboard)),
        },
        {
            "Item": "Rolling predictions",
            "Current value": format_int(len(forecast_paths)),
        },
        {
            "Item": "Metric rows",
            "Current value": format_int(len(performance)),
        },
    ]
    st.markdown("### Loaded Experiment Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    model_scope = (
        leaderboard.groupby(["model_family", "model_build"], dropna=False)
        .size()
        .reset_index(name="configurations")
    )
    model_scope = model_taxonomy_sort(model_scope)
    st.markdown("### Model Scope In This Bundle")
    st.dataframe(model_scope, use_container_width=True, hide_index=True)

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
    with st.expander("How feature, representation, and complexity policies are interpreted"):
        st.markdown(REPRESENTATION_AND_COMPLEXITY_EXPLANATION)

    st.markdown("### Experiment Configs Used")
    st.write(
        "These YAML files define the model grids, feature families, feature "
        "policies, rolling forecast window, parallelization settings, MLflow "
        "tracking names, checkpoint locations, and output artifact folders used "
        "for the current combined A/B v2 dashboard bundle."
    )
    config_rows = [
        {
            "Phase": "A v2",
            "Config file": str(PHASE_A_V2_CONFIG_PATH),
            "Role": "Baseline, linear, tree, and XGBoost grids over tabular feature families.",
            "Loaded": "yes" if phase_a_config_text else "missing",
        },
        {
            "Phase": "B v2",
            "Config file": str(PHASE_B_V2_CONFIG_PATH),
            "Role": "ARIMA, SARIMA, and SARIMAX grids with compact exogenous sets.",
            "Loaded": "yes" if phase_b_config_text else "missing",
        },
    ]
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

    if phase_a_config_text:
        with st.expander("Show Phase A v2 config YAML"):
            st.code(phase_a_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_A_V2_CONFIG_PATH}`.")

    if phase_b_config_text:
        with st.expander("Show Phase B v2 config YAML"):
            st.code(phase_b_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_B_V2_CONFIG_PATH}`.")

    st.markdown(EXPERIMENT_OVERVIEW)


@st.cache_data(show_spinner=False)
def load_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


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
    model_family: str = "All",
    model_build: str = "All",
    mode: str = "All",
    feature_family: str = "All",
    feature_policy: str = "All",
) -> pd.DataFrame:
    out = apply_optional_filter(df, "model_family", model_family)
    out = apply_optional_filter(out, "model_build", model_build)
    out = apply_optional_filter(out, "mode", mode)
    out = apply_optional_filter(out, "feature_family_name", feature_family)
    out = apply_optional_filter(out, "feature_policy", feature_policy)
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
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["target_date"],
            y=df["prediction"],
            mode="lines+markers",
            name="Prediction",
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
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Target month",
        yaxis_title="UPT",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=95),
    )
    fig.update_xaxes(title_standoff=18)
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


def apply_optional_filter(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if value == "All" or column not in df:
        return df
    return df[df[column].astype(str) == value]


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
    return (
        f"#{int(row['rank'])} | {row.get('model_build', row.get('model_type', 'model'))} "
        f"| {row.get('mode', '-')} | {row.get('feature_family_name', '-')} "
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
            )
        )

    for _, group in paths.sort_values(["rank", "target_date"]).groupby("model_config_id", sort=False):
        first = group.iloc[0]
        label = (
            f"#{int(first['rank'])} {first.get('model_build', first.get('model_type', 'model'))} | "
            f"{first.get('feature_family_name', '-')}"
        )
        fig.add_trace(
            go.Scatter(
                x=group["target_date"],
                y=group["prediction"],
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Target month",
        yaxis_title="UPT",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=95),
    )
    fig.update_xaxes(title_standoff=18)
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
        "model_family",
        "model_build",
        "feature_family_name",
        "mode",
        "feature_policy",
        "hyperparameters",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score",
        "selection_score_typical",
        "selection_score_balanced",
        "selection_score_large_error",
        "pre_covid_mae",
        "pre_covid_rmse",
        "pre_covid_selection_score_typical",
        "pre_covid_selection_score_balanced",
        "pre_covid_selection_score_large_error",
        "covid_shock_mae",
        "covid_shock_rmse",
        "covid_shock_selection_score_typical",
        "covid_shock_selection_score_balanced",
        "covid_shock_selection_score_large_error",
        "recovery_mae",
        "recovery_rmse",
        "recovery_selection_score_typical",
        "recovery_selection_score_balanced",
        "recovery_selection_score_large_error",
        "recent_mae",
        "recent_rmse",
        "recent_selection_score_typical",
        "recent_selection_score_balanced",
        "recent_selection_score_large_error",
        "shock_penalty",
        "rmse_shock_penalty",
        "typical_score_shock_penalty",
        "balanced_score_shock_penalty",
        "large_error_score_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "typical_score_recovery_ratio",
        "balanced_score_recovery_ratio",
        "large_error_score_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
        "typical_score_recent_recovery_ratio",
        "balanced_score_recent_recovery_ratio",
        "large_error_score_recent_recovery_ratio",
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
        "model_build",
        "mode",
        "feature_family_name",
        "feature_policy",
        "hyperparameters",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score",
        "selection_score_typical",
        "selection_score_balanced",
        "selection_score_large_error",
        "pre_covid_mae",
        "pre_covid_rmse",
        "pre_covid_selection_score_typical",
        "pre_covid_selection_score_balanced",
        "pre_covid_selection_score_large_error",
        "covid_shock_mae",
        "covid_shock_rmse",
        "covid_shock_selection_score_typical",
        "covid_shock_selection_score_balanced",
        "covid_shock_selection_score_large_error",
        "recovery_mae",
        "recovery_rmse",
        "recovery_selection_score_typical",
        "recovery_selection_score_balanced",
        "recovery_selection_score_large_error",
        "recent_mae",
        "recent_rmse",
        "recent_selection_score_typical",
        "recent_selection_score_balanced",
        "recent_selection_score_large_error",
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
    model_builds: list[str],
    rank_label: str,
    per_build_limit: str,
) -> pd.DataFrame:
    if not model_builds:
        return leaderboard.iloc[0:0].copy()
    frame = leaderboard[leaderboard["model_build"].astype(str).isin(model_builds)].copy()
    if per_build_limit == "All":
        return frame

    limit = int(per_build_limit.replace("Top ", ""))
    ranked_slices = []
    for _, group in frame.groupby("model_build", sort=False):
        ranked_slices.append(sort_by_rank_metric(group, rank_label).head(limit))
    if not ranked_slices:
        return frame.iloc[0:0].copy()
    return pd.concat(ranked_slices, ignore_index=True)


def aggregate_metric_mapping(frame: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    x_col, _ = RANK_METRIC_OPTIONS[x_metric]
    y_col, _ = RANK_METRIC_OPTIONS[y_metric]
    group_cols = ["model_family", "model_build"]
    optional_cols = [
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score",
        "selection_score_typical",
        "selection_score_balanced",
        "selection_score_large_error",
        "shock_penalty",
        "rmse_shock_penalty",
        "typical_score_shock_penalty",
        "balanced_score_shock_penalty",
        "large_error_score_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "typical_score_recovery_ratio",
        "balanced_score_recovery_ratio",
        "large_error_score_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
        "typical_score_recent_recovery_ratio",
        "balanced_score_recent_recovery_ratio",
        "large_error_score_recent_recovery_ratio",
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
    return summary


def metric_mapping_hover_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        "model_family",
        "model_build",
        "feature_family_name",
        "feature_policy",
        "mode",
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score",
        "selection_score_typical",
        "selection_score_balanced",
        "selection_score_large_error",
        "shock_penalty",
        "rmse_shock_penalty",
        "typical_score_shock_penalty",
        "balanced_score_shock_penalty",
        "large_error_score_shock_penalty",
        "recovery_ratio",
        "rmse_recovery_ratio",
        "typical_score_recovery_ratio",
        "balanced_score_recovery_ratio",
        "large_error_score_recovery_ratio",
        "recent_recovery_ratio",
        "rmse_recent_recovery_ratio",
        "typical_score_recent_recovery_ratio",
        "balanced_score_recent_recovery_ratio",
        "large_error_score_recent_recovery_ratio",
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
            "model_build": "Model build",
            "feature_policy": "Feature policy",
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
    page = st.sidebar.radio(
        "Section",
        [
            "Project Overview",
            "System",
            "Data",
            "Experiment",
            "Results Explorer",
        ],
    )

    artifacts = load_artifacts(selected_dir)
    forecast_paths = ensure_model_taxonomy(
        normalize_dates(artifacts["forecast_paths"], ["as_of_date", "target_date"])
    )
    performance = ensure_model_taxonomy(
        normalize_dates(artifacts["performance_over_time"], ["as_of_date", "target_date"])
    )
    leaderboard = enrich_score_columns(ensure_model_taxonomy(artifacts["model_leaderboard"]))
    family_summary = artifacts["feature_family_summary"].copy()
    champion_predictions = normalize_dates(artifacts["champion_predictions"], ["as_of_date", "target_date"])
    champion = artifacts["champion_selection"]
    experiment_manifest = artifacts.get("experiment_manifest", {})
    feature_families = load_feature_family_definitions()
    phase_a_config_text = load_config_text(PHASE_A_V2_CONFIG_PATH)
    phase_b_config_text = load_config_text(PHASE_B_V2_CONFIG_PATH)
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
    render_dashboard_header(champion, forecast_paths, experiment_manifest)

    if page == "Project Overview":
        st.subheader("Project Overview")
        render_public_bundle_note(experiment_manifest)
        st.markdown(PROJECT_OVERVIEW)

        overview_cols = st.columns(4)
        overview_cols[0].metric(
            "Forecast Horizon",
            f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', '-'))} months",
        )
        overview_cols[1].metric("Model Configs", format_int(len(leaderboard)))
        overview_cols[2].metric("Rolling Predictions", format_int(len(forecast_paths)))
        overview_cols[3].metric("Target Window", date_range_label(forecast_paths, "target_date"))
        return

    if page == "System":
        st.subheader("System")
        st.markdown(SYSTEM_OVERVIEW)
        render_image_gallery(
            "System Screenshots",
            "Drop architecture sketches, AWS Step Functions captures, or other system screenshots here as the cloud side evolves.",
        )
        return

    if page == "Data":
        render_data_page(family_summary, leaderboard, forecast_paths, champion, feature_families)
        return

    if page == "Experiment":
        render_experiment_page(
            leaderboard,
            forecast_paths,
            performance,
            champion,
            experiment_manifest,
            phase_a_config_text,
            phase_b_config_text,
        )
        return

    (
        tab_modeling_overview,
        tab_forecast,
        tab_performance,
        tab_metric_mapping,
        tab_features,
        tab_ops,
    ) = st.tabs(
        [
            "Modeling Overview",
            "Forecast Explorer",
            "Model Performance",
            "Metric Mapping",
            "Feature Strategy",
            "Operational Footprint",
        ]
    )

    with tab_modeling_overview:
        st.subheader("Top Models Against Actual Ridership")
        filter_cols = st.columns(6)
        selected_model_family = all_selectbox(
            "Model family",
            leaderboard["model_family"],
            "overview_model_family",
            filter_cols[0],
        )
        family_scope = apply_optional_filter(leaderboard, "model_family", selected_model_family)
        selected_model_build = all_selectbox(
            "Model build",
            family_scope["model_build"],
            "overview_model_build",
            filter_cols[1],
        )
        build_scope = apply_optional_filter(family_scope, "model_build", selected_model_build)
        selected_mode = all_selectbox(
            "Mode",
            build_scope["mode"],
            "overview_mode",
            filter_cols[2],
        )
        mode_scope = apply_optional_filter(build_scope, "mode", selected_mode)
        selected_feature_family = all_selectbox(
            "Feature family",
            mode_scope["feature_family_name"],
            "overview_feature_family",
            filter_cols[3],
        )
        feature_scope = apply_optional_filter(mode_scope, "feature_family_name", selected_feature_family)
        selected_feature_policy = all_selectbox(
            "Feature policy",
            feature_scope["feature_policy"],
            "overview_feature_policy",
            filter_cols[4],
        )
        metric_label = filter_cols[5].selectbox(
            "Rank by",
            available_rank_options(leaderboard),
            key="overview_metric",
        )
        filtered_top = filtered_frame(
            leaderboard,
            model_family=selected_model_family,
            model_build=selected_model_build,
            mode=selected_mode,
            feature_family=selected_feature_family,
            feature_policy=selected_feature_policy,
        )
        metric = RANK_METRIC_OPTIONS[metric_label][0]
        if metric in filtered_top:
            filtered_top = sort_by_rank_metric(filtered_top, metric_label).copy()

        candidate_configs = filtered_top["model_config_id"].tolist()
        chart_paths = forecast_paths[forecast_paths["model_config_id"].isin(candidate_configs)].copy()
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
        selected_models, chart_paths, duplicate_count = select_distinct_model_paths(
            filtered_top,
            windowed_paths,
            max_models=5,
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
            top_model_chart(chart_paths, "Top Five Model Paths"),
            use_container_width=True,
        )
        st.dataframe(overview_table(selected_models), hide_index=True, use_container_width=True)

        with st.expander("Champion selection rule"):
            st.write(champion.get("selection_rule", "No selection rule recorded."))

        with st.expander("How period-specific metrics are interpreted"):
            st.markdown(PERIOD_METRIC_EXPLANATION)

    with tab_forecast:
        st.subheader("Forecast Explorer")
        filter_cols = st.columns(6)
        model_family_options = ordered_unique(leaderboard["model_family"], MODEL_FAMILY_ORDER)
        model_family = filter_cols[0].selectbox(
            "Model family",
            model_family_options,
            index=selectbox_index(model_family_options, champion.get("model_family")),
            key="forecast_model_family",
        )
        family_scope = leaderboard[leaderboard["model_family"].astype(str) == model_family]
        default_build = champion.get("model_build") if champion.get("model_family") == model_family else None
        build_options = ordered_unique(family_scope["model_build"], MODEL_BUILD_ORDER)
        model_build = filter_cols[1].selectbox(
            "Model build",
            build_options,
            index=selectbox_index(build_options, default_build),
            key="forecast_model_build",
        )
        build_scope = family_scope[family_scope["model_build"].astype(str) == model_build]
        default_mode = (
            champion.get("mode")
            if champion.get("model_family") == model_family and champion.get("model_build") == model_build
            else None
        )
        mode_options = sorted(str(value) for value in build_scope["mode"].dropna().unique())
        mode = filter_cols[2].selectbox(
            "Mode",
            mode_options,
            index=selectbox_index(mode_options, default_mode),
            key="forecast_mode",
        )
        mode_scope = build_scope[build_scope["mode"].astype(str) == mode]
        default_family = (
            champion.get("feature_family_name")
            if champion.get("model_family") == model_family
            and champion.get("model_build") == model_build
            and champion.get("mode") == mode
            else None
        )
        family_options = sorted(str(value) for value in mode_scope["feature_family_name"].dropna().unique())
        family = filter_cols[3].selectbox(
            "Feature family",
            family_options,
            index=selectbox_index(family_options, default_family),
            key="forecast_feature_family",
        )
        policy_scope = build_scope[
            (build_scope["mode"].astype(str) == mode)
            & (build_scope["feature_family_name"].astype(str) == family)
        ]
        default_policy = (
            champion.get("feature_policy", "none")
            if champion.get("model_family") == model_family
            and champion.get("model_build") == model_build
            and champion.get("mode") == mode
            and champion.get("feature_family_name") == family
            else None
        )
        policy_options = sorted(str(value) for value in policy_scope["feature_policy"].dropna().unique())
        feature_policy = filter_cols[4].selectbox(
            "Feature policy",
            policy_options,
            index=selectbox_index(policy_options, default_policy),
            key="forecast_feature_policy",
        )
        forecast_rank_label = filter_cols[5].selectbox(
            "Rank by",
            available_rank_options(leaderboard),
            key="forecast_metric",
        )
        ranked_candidates = filtered_frame(
            leaderboard,
            model_family=model_family,
            model_build=model_build,
            mode=mode,
            feature_family=family,
            feature_policy=feature_policy,
        )
        ranked_candidates = sort_by_rank_metric(ranked_candidates, forecast_rank_label)
        if ranked_candidates.empty:
            st.info("No model configurations match the selected filters.")
            return

        selected_labels = [
            ranked_model_label(row, forecast_rank_label)
            for _, row in ranked_candidates.iterrows()
        ]
        label_by_config = dict(zip(ranked_candidates["config_id"], selected_labels))
        default_config = (
            champion.get("config_id")
            if champion.get("config_id") in label_by_config
            else ranked_candidates.iloc[0]["config_id"]
        )
        selected_label = st.selectbox(
            "Selected ranked model",
            selected_labels,
            index=selected_labels.index(label_by_config[default_config]),
            key="forecast_ranked_model",
        )
        label_to_config = dict(zip(selected_labels, ranked_candidates["config_id"]))
        config_id = label_to_config[selected_label]
        selected_model = ranked_candidates[ranked_candidates["config_id"] == config_id].iloc[0]
        st.caption(
            "Hyperparameters: "
            f"{parse_json_display(selected_model.get('hyperparameters_json', '{}'))}. "
            f"Ranked by {forecast_rank_label.lower()} within the active filters."
        )

        st.markdown("**Top matching ranked models**")
        st.dataframe(
            forecast_ranked_table(ranked_candidates.head(10)),
            hide_index=True,
            use_container_width=True,
        )

        candidates = forecast_paths[forecast_paths["config_id"] == config_id].copy()
        date_cols = st.columns([1, 1, 1, 1, 1])
        as_of_min, as_of_max = date_bounds(candidates, "as_of_date")
        target_min, target_max = date_bounds(candidates, "target_date")
        forecast_as_of_start = date_cols[0].date_input(
            "As-of start",
            as_of_min.date(),
            min_value=as_of_min.date(),
            max_value=as_of_max.date(),
            key="forecast_as_of_start",
        )
        forecast_as_of_end = date_cols[1].date_input(
            "As-of end",
            as_of_max.date(),
            min_value=as_of_min.date(),
            max_value=as_of_max.date(),
            key="forecast_as_of_end",
        )
        forecast_target_start = date_cols[2].date_input(
            "Target start",
            target_min.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key="forecast_target_start",
        )
        forecast_target_end = date_cols[3].date_input(
            "Target end",
            target_max.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key="forecast_target_end",
        )
        selected_forecast = candidates.sort_values("target_date")
        selected_forecast = apply_date_window(
            selected_forecast,
            "as_of_date",
            forecast_as_of_start,
            forecast_as_of_end,
        )
        selected_forecast = apply_date_window(
            selected_forecast,
            "target_date",
            forecast_target_start,
            forecast_target_end,
        )
        st.plotly_chart(
            line_forecast_chart(selected_forecast, "Selected Forecast vs Actual"),
            use_container_width=True,
        )
        st.dataframe(
            selected_forecast[
                [
                    "as_of_date",
                    "target_date",
                    "actual",
                    "prediction",
                    "seasonal_naive_prediction",
                    "error",
                    "abs_error",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_performance:
        st.subheader("Model Leaderboard")
        perf_filter_cols = st.columns(6)
        perf_model_family = all_selectbox(
            "Model family",
            leaderboard["model_family"],
            "performance_model_family",
            perf_filter_cols[0],
        )
        perf_family_scope = apply_optional_filter(leaderboard, "model_family", perf_model_family)
        perf_model_build = all_selectbox(
            "Model build",
            perf_family_scope["model_build"],
            "performance_model_build",
            perf_filter_cols[1],
        )
        perf_build_scope = apply_optional_filter(perf_family_scope, "model_build", perf_model_build)
        perf_mode = all_selectbox("Mode", perf_build_scope["mode"], "performance_mode", perf_filter_cols[2])
        perf_mode_scope = apply_optional_filter(perf_build_scope, "mode", perf_mode)
        perf_feature_family = all_selectbox(
            "Feature family",
            perf_mode_scope["feature_family_name"],
            "performance_feature_family",
            perf_filter_cols[3],
        )
        perf_feature_scope = apply_optional_filter(
            perf_mode_scope,
            "feature_family_name",
            perf_feature_family,
        )
        perf_feature_policy = all_selectbox(
            "Feature policy",
            perf_feature_scope["feature_policy"],
            "performance_feature_policy",
            perf_filter_cols[4],
        )
        perf_rank_label = perf_filter_cols[5].selectbox(
            "Rank by",
            available_rank_options(leaderboard),
            key="performance_metric",
        )
        filtered_leaderboard = filtered_frame(
            leaderboard,
            model_family=perf_model_family,
            model_build=perf_model_build,
            mode=perf_mode,
            feature_family=perf_feature_family,
            feature_policy=perf_feature_policy,
        )
        filtered_leaderboard = sort_by_rank_metric(filtered_leaderboard, perf_rank_label)

        display_cols = [
            "rank",
            "model_family",
            "model_build",
            "mode",
            "feature_policy",
            "feature_family_name",
            "n_features",
            "mae",
            "rmse",
            "r2",
            "r2_adjusted",
            "diracc",
            "selection_score",
            "selection_score_typical",
            "selection_score_balanced",
            "selection_score_large_error",
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
        available_cols = [column for column in display_cols if column in leaderboard.columns]
        st.dataframe(filtered_leaderboard[available_cols].head(50), use_container_width=True, hide_index=True)

        period_cols = [
            "rank",
            "model_family",
            "model_build",
            "mode",
            "feature_policy",
            "feature_family_name",
            "selection_score",
            "selection_score_typical",
            "selection_score_balanced",
            "selection_score_large_error",
            "pre_covid_mae",
            "pre_covid_rmse",
            "pre_covid_selection_score_typical",
            "pre_covid_selection_score_balanced",
            "pre_covid_selection_score_large_error",
            "covid_shock_mae",
            "covid_shock_rmse",
            "covid_shock_selection_score_typical",
            "covid_shock_selection_score_balanced",
            "covid_shock_selection_score_large_error",
            "recovery_mae",
            "recovery_rmse",
            "recovery_selection_score_typical",
            "recovery_selection_score_balanced",
            "recovery_selection_score_large_error",
            "recent_mae",
            "recent_rmse",
            "recent_selection_score_typical",
            "recent_selection_score_balanced",
            "recent_selection_score_large_error",
            "shock_penalty",
            "rmse_shock_penalty",
            "typical_score_shock_penalty",
            "balanced_score_shock_penalty",
            "large_error_score_shock_penalty",
            "recovery_ratio",
            "rmse_recovery_ratio",
            "typical_score_recovery_ratio",
            "balanced_score_recovery_ratio",
            "large_error_score_recovery_ratio",
            "recent_recovery_ratio",
            "rmse_recent_recovery_ratio",
            "typical_score_recent_recovery_ratio",
            "balanced_score_recent_recovery_ratio",
            "large_error_score_recent_recovery_ratio",
        ]
        available_period_cols = [column for column in period_cols if column in filtered_leaderboard.columns]
        if len(available_period_cols) > 6:
            st.markdown("**Period Metrics**")
            st.dataframe(
                filtered_leaderboard[available_period_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Period metrics and derived ratios"):
            st.markdown(PERIOD_METRIC_SHORT_EXPLANATION)

        with st.expander("Feature policies, representations, and complexity scores"):
            st.markdown(REPRESENTATION_AND_COMPLEXITY_EXPLANATION)

        st.subheader("Rolling Error Over Time")
        perf_date_cols = st.columns([1, 1, 2])
        perf_as_of_min, perf_as_of_max = date_bounds(performance, "as_of_date")
        perf_as_of_start = perf_date_cols[0].date_input(
            "As-of start",
            perf_as_of_min.date(),
            min_value=perf_as_of_min.date(),
            max_value=perf_as_of_max.date(),
            key="performance_as_of_start",
        )
        perf_as_of_end = perf_date_cols[1].date_input(
            "As-of end",
            perf_as_of_max.date(),
            min_value=perf_as_of_min.date(),
            max_value=perf_as_of_max.date(),
            key="performance_as_of_end",
        )
        top_configs = filtered_leaderboard.head(8)["config_id"].tolist()
        rolling_subset = performance[performance["config_id"].isin(top_configs)].copy()
        rolling_subset = apply_date_window(
            rolling_subset,
            "as_of_date",
            perf_as_of_start,
            perf_as_of_end,
        )
        st.plotly_chart(rolling_error_chart(rolling_subset), use_container_width=True)

        st.subheader("Model Build Comparison")
        comparison_metric = RANK_METRIC_OPTIONS[perf_rank_label][0]
        comparison_ascending = RANK_METRIC_OPTIONS[perf_rank_label][1]
        comparison_agg = "min" if comparison_ascending else "max"
        model_summary = (
            filtered_leaderboard.groupby(["model_family", "model_build"], as_index=False)
            .agg(
                best_mae=("mae", "min"),
                best_rmse=("rmse", "min"),
                best_metric=(comparison_metric, comparison_agg),
            )
            .sort_values("best_metric", ascending=comparison_ascending)
        )
        fig = px.bar(
            model_summary,
            x="model_build",
            y="best_metric",
            color="model_family",
            category_orders={
                "model_family": MODEL_FAMILY_ORDER,
                "model_build": MODEL_BUILD_ORDER,
            },
            labels={
                "model_build": "Model build",
                "model_family": "Model family",
                "best_metric": f"Best {perf_rank_label}",
            },
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_metric_mapping:
        st.subheader("Metric Mapping")
        st.write(
            "Map model configurations across two evaluation metrics to inspect tradeoffs, "
            "clusters, and cases where a model family performs well on one dimension but "
            "not another."
        )
        metric_options = available_rank_options(leaderboard)
        build_options = ordered_unique(leaderboard["model_build"], MODEL_BUILD_ORDER)
        control_cols = st.columns([1, 1, 1, 1, 1])
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
            ["Top 1", "Top 5", "Top 10", "Top 25", "All"],
            index=1,
            key="mapping_per_build_limit",
        )
        point_mode = control_cols[4].selectbox(
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
        color_by = st.selectbox(
            "Color by",
            ["model_family", "model_build", "feature_policy", "mode"],
            key="mapping_color_by",
        )

        mapping_frame = metric_mapping_frame(
            leaderboard,
            selected_builds,
            rank_metric,
            per_build_limit,
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
                "model_family",
                "model_build",
                "mode",
                "feature_policy",
                "feature_family_name",
                "configurations",
                "mae",
                "rmse",
                "r2",
                "r2_adjusted",
                "diracc",
                "selection_score",
                "selection_score_typical",
                "selection_score_balanced",
                "selection_score_large_error",
                "pre_covid_mae",
                "pre_covid_rmse",
                "pre_covid_selection_score_typical",
                "pre_covid_selection_score_balanced",
                "pre_covid_selection_score_large_error",
                "covid_shock_mae",
                "covid_shock_rmse",
                "covid_shock_selection_score_typical",
                "covid_shock_selection_score_balanced",
                "covid_shock_selection_score_large_error",
                "recovery_mae",
                "recovery_rmse",
                "recovery_selection_score_typical",
                "recovery_selection_score_balanced",
                "recovery_selection_score_large_error",
                "recent_mae",
                "recent_rmse",
                "recent_selection_score_typical",
                "recent_selection_score_balanced",
                "recent_selection_score_large_error",
                "shock_penalty",
                "rmse_shock_penalty",
                "typical_score_shock_penalty",
                "balanced_score_shock_penalty",
                "large_error_score_shock_penalty",
                "recovery_ratio",
                "rmse_recovery_ratio",
                "typical_score_recovery_ratio",
                "balanced_score_recovery_ratio",
                "large_error_score_recovery_ratio",
                "recent_recovery_ratio",
                "rmse_recent_recovery_ratio",
                "typical_score_recent_recovery_ratio",
                "balanced_score_recent_recovery_ratio",
                "large_error_score_recent_recovery_ratio",
                "complexity_score",
                "interpretability_score",
                "compute_score",
            ]
            available_mapping_cols = [column for column in table_cols if column in mapping_frame.columns]
            st.dataframe(
                mapping_frame[available_mapping_cols].head(200),
                use_container_width=True,
                hide_index=True,
            )

    with tab_features:
        st.subheader("Feature Family Ranking")
        family_display = family_summary.sort_values("best_selection_score").copy()
        st.dataframe(family_display.head(20), use_container_width=True, hide_index=True)

        fig = px.bar(
            family_display.head(14),
            x="best_selection_score",
            y="feature_family_name",
            color="mode",
            orientation="h",
            labels={
                "best_selection_score": "Best balanced score (lower is better)",
                "feature_family_name": "Feature family",
                "mode": "Mode",
            },
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_ops:
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
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Loaded Artifacts")
        st.code(str(selected_dir), language="text")
        st.write("This app reads curated dashboard artifacts only. It does not trigger training jobs.")


if __name__ == "__main__":
    main()
