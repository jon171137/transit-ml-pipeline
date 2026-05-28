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
        SYSTEM_OVERVIEW,
    )
except ImportError:
    from dashboard.content import (
        DATA_OVERVIEW,
        EXPERIMENT_OVERVIEW,
        PERIOD_METRIC_EXPLANATION,
        PERIOD_METRIC_SHORT_EXPLANATION,
        PROJECT_OVERVIEW,
        SYSTEM_OVERVIEW,
    )


DEFAULT_ARTIFACT_DIR = Path("dashboard_artifacts/aws_streamlined/latest")
IMAGE_ASSET_DIR = Path("dashboard/assets/images")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
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

RANK_METRIC_OPTIONS = {
    "Selection score": ("selection_score", True),
    "MAE": ("mae", True),
    "RMSE": ("rmse", True),
    "R-squared": ("r2", False),
    "Directional accuracy": ("diracc", False),
    "Pre-COVID MAE": ("pre_covid_mae", True),
    "COVID shock MAE": ("covid_shock_mae", True),
    "Recovery MAE": ("recovery_mae", True),
    "Recent MAE": ("recent_mae", True),
    "Shock penalty": ("shock_penalty", True),
    "Recovery ratio": ("recovery_ratio", True),
    "Recent recovery ratio": ("recent_recovery_ratio", True),
}

PERIOD_RANK_WINDOWS = {
    "Pre-COVID MAE": (None, "2020-02-01"),
    "COVID shock MAE": ("2020-03-01", "2021-06-01"),
    "Shock penalty": ("2020-03-01", "2021-06-01"),
    "Recovery MAE": ("2021-07-01", "2022-12-01"),
    "Recovery ratio": ("2021-07-01", "2022-12-01"),
    "Recent MAE": ("2023-01-01", None),
    "Recent recovery ratio": ("2023-01-01", None),
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

        @media (max-width: 1200px) {
            .compact-kpi-grid {
                grid-template-columns: repeat(2, minmax(130px, 1fr));
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
            <span>Personal Forecasting Project</span> by Jon Sellers
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


def render_experiment_summary(
    champion: dict,
    forecast_paths: pd.DataFrame,
    experiment_manifest: dict,
) -> None:
    items = [
        ("Champion", champion.get("model_type", "-")),
        ("Feature family", champion.get("feature_family_name", "-")),
        ("Mode", champion.get("mode", "-")),
        ("Selection score", format_int(champion.get("selection_score"))),
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


def render_data_page(
    family_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    champion: dict,
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


@st.cache_data(show_spinner=False)
def load_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(path: str, modified_ns: int) -> dict:
    _ = modified_ns
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def file_modified_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def configured_artifact_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))


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
    if model_type in {"ridge", "lasso"}:
        return "linear"
    if model_type == "xgboost":
        return "tree"
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
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
    )
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
    clean_values = sorted(str(value) for value in values.dropna().unique())
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
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
    )
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
        "diracc",
        "selection_score",
        "pre_covid_mae",
        "covid_shock_mae",
        "recovery_mae",
        "recent_mae",
        "shock_penalty",
        "recovery_ratio",
        "recent_recovery_ratio",
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
    leaderboard = ensure_model_taxonomy(artifacts["model_leaderboard"])
    family_summary = artifacts["feature_family_summary"].copy()
    champion_predictions = normalize_dates(artifacts["champion_predictions"], ["as_of_date", "target_date"])
    champion = artifacts["champion_selection"]
    experiment_manifest = artifacts.get("experiment_manifest", {})
    overview_top_models = ensure_model_taxonomy(artifacts.get("overview_top_models", leaderboard.head(5).copy()))
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
    st.title("Transit Forecasting Lab")
    st.caption("AWS streamlined model comparison for H3 UPT forecasts")
    render_experiment_summary(champion, forecast_paths, experiment_manifest)

    if page == "Project Overview":
        st.subheader("Project Overview")
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
        render_data_page(family_summary, leaderboard, forecast_paths, champion)
        return

    if page == "Experiment":
        st.subheader("Experiment")
        st.markdown(EXPERIMENT_OVERVIEW)
        return

    (
        tab_modeling_overview,
        tab_forecast,
        tab_performance,
        tab_features,
        tab_ops,
    ) = st.tabs(
        [
            "Modeling Overview",
            "Forecast Explorer",
            "Model Performance",
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
        model_family = filter_cols[0].selectbox(
            "Model family",
            sorted(forecast_paths["model_family"].unique()),
            key="forecast_model_family",
        )
        family_scope = forecast_paths[forecast_paths["model_family"] == model_family]
        model_build = filter_cols[1].selectbox(
            "Model build",
            sorted(family_scope["model_build"].unique()),
            key="forecast_model_build",
        )
        build_scope = family_scope[family_scope["model_build"] == model_build]
        mode_options = sorted(build_scope["mode"].unique())
        mode = filter_cols[2].selectbox("Mode", mode_options, key="forecast_mode")
        family_options = sorted(
            build_scope[build_scope["mode"] == mode]["feature_family_name"].unique()
        )
        family = filter_cols[3].selectbox("Feature family", family_options, key="forecast_feature_family")
        policy_scope = build_scope[
            (build_scope["mode"] == mode)
            & (build_scope["feature_family_name"] == family)
        ]
        feature_policy = filter_cols[4].selectbox(
            "Feature policy",
            sorted(policy_scope["feature_policy"].unique()),
            key="forecast_feature_policy",
        )
        candidates = forecast_paths[
            (forecast_paths["model_family"] == model_family)
            & (forecast_paths["model_build"] == model_build)
            & (forecast_paths["mode"] == mode)
            & (forecast_paths["feature_family_name"] == family)
            & (forecast_paths["feature_policy"] == feature_policy)
        ].copy()
        config_options = sorted(candidates["config_id"].unique())
        default_config = champion.get("config_id") if champion.get("config_id") in config_options else config_options[0]
        config_id = filter_cols[5].selectbox(
            "Config",
            config_options,
            index=config_options.index(default_config),
        )
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
        selected_forecast = candidates[candidates["config_id"] == config_id].sort_values("target_date")
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
            "diracc",
            "selection_score",
            "pre_covid_mae",
            "covid_shock_mae",
            "recovery_mae",
            "recent_mae",
            "shock_penalty",
            "recovery_ratio",
            "recent_recovery_ratio",
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
            "pre_covid_mae",
            "covid_shock_mae",
            "recovery_mae",
            "recent_mae",
            "shock_penalty",
            "recovery_ratio",
            "recent_recovery_ratio",
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
            labels={
                "model_build": "Model build",
                "model_family": "Model family",
                "best_metric": f"Best {perf_rank_label}",
            },
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

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
                "best_selection_score": "Best selection score (lower is better)",
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
