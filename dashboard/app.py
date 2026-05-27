import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_ARTIFACT_DIR = Path("dashboard_artifacts/aws_streamlined/latest")
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


st.set_page_config(
    page_title="Transit Forecasting Lab",
    page_icon="",
    layout="wide",
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
    st.title("Transit Forecasting Lab")
    st.info("Dashboard artifacts were not found yet.")
    st.write("Expected local artifact folder:")
    st.code(str(base_dir), language="text")
    st.write("Place a completed dashboard export there, or set:")
    st.code("export DASHBOARD_ARTIFACT_DIR=/path/to/dashboard/aws_streamlined/run_id=<run_id>", language="bash")
    st.write("Expected files:")
    st.code("\n".join(REQUIRED_FILES.values()), language="text")


def main() -> None:
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

    st.title("Transit Forecasting Lab")
    st.caption("AWS streamlined model comparison for H3 UPT forecasts")

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Champion", champion.get("model_type", "-"))
        col2.metric("Feature family", champion.get("feature_family_name", "-"))
        col3.metric("Mode", champion.get("mode", "-"))
        col4.metric("Selection score", format_int(champion.get("selection_score")))

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Champion MAE", format_int(champion.get("mae")))
        col6.metric("Champion RMSE", format_int(champion.get("rmse")))
        col7.metric("Predictions", format_int(len(forecast_paths)))
        col8.metric("Target dates", date_range_label(forecast_paths, "target_date"))

        context_cols = st.columns(5)
        context_cols[0].metric("As-of dates", date_range_label(forecast_paths, "as_of_date"))
        context_cols[1].metric("Forecast horizon", f"{manifest_value(experiment_manifest, 'horizon', champion.get('horizon', '-'))} months")
        context_cols[2].metric(
            "Forecast cadence",
            months_label(
                manifest_value(
                    experiment_manifest,
                    "as_of_frequency_months",
                    month_interval_label(forecast_paths["as_of_date"]),
                )
            ),
        )
        context_cols[3].metric(
            "Refit cadence",
            months_label(manifest_value(experiment_manifest, "refit_frequency_months", "-")),
        )
        context_cols[4].metric("Evaluation", manifest_value(experiment_manifest, "target", champion.get("target", "-")).upper())
        st.caption(
            "Each as-of month trains only on data before that month, then forecasts the target "
            "horizon ahead. Aggregated metrics compare those rolling historical forecasts with actual outcomes."
        )

    tab_overview, tab_forecast, tab_performance, tab_features, tab_ops = st.tabs(
        [
            "Overview",
            "Forecast Explorer",
            "Model Performance",
            "Feature Strategy",
            "Operational Footprint",
        ]
    )

    with tab_overview:
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
        date_cols = st.columns([1, 1, 3])
        target_min, target_max = date_bounds(forecast_paths, "target_date")
        overview_target_start = date_cols[0].date_input(
            "Target start",
            target_min.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key="overview_target_start",
        )
        overview_target_end = date_cols[1].date_input(
            "Target end",
            target_max.date(),
            min_value=target_min.date(),
            max_value=target_max.date(),
            key="overview_target_end",
        )
        metric = RANK_METRIC_OPTIONS[metric_label][0]
        if metric in filtered_top:
            filtered_top = sort_by_rank_metric(filtered_top, metric_label).head(5).copy()

        selected_configs = filtered_top["model_config_id"].tolist()
        chart_paths = forecast_paths[forecast_paths["model_config_id"].isin(selected_configs)].copy()
        chart_paths = apply_date_window(
            chart_paths,
            "target_date",
            overview_target_start,
            overview_target_end,
        )
        chart_paths = chart_paths.drop(columns=["rank"], errors="ignore").merge(
            filtered_top[["model_config_id", "rank"]],
            on="model_config_id",
            how="left",
        )
        chart_paths = chart_paths.sort_values(["rank", "target_date"])

        st.plotly_chart(
            top_model_chart(chart_paths, "Top Five Model Paths"),
            use_container_width=True,
        )
        st.dataframe(overview_table(filtered_top), hide_index=True, use_container_width=True)

        with st.expander("Champion selection rule"):
            st.write(champion.get("selection_rule", "No selection rule recorded."))

        with st.expander("How period-specific metrics are interpreted"):
            st.markdown(
                """
                Metrics are calculated from target months, not training months. Each row asks:
                if the model was trained at each historical as-of date, how accurate was its
                3-month-ahead forecast for the target month?

                `pre_covid` covers target months through February 2020.
                `covid_shock` covers March 2020 through June 2021.
                `recovery` covers July 2021 through December 2022.
                `recent` covers January 2023 onward.

                `shock_penalty = covid_shock_mae / pre_covid_mae`.
                `recovery_ratio = recovery_mae / pre_covid_mae`.
                `recent_recovery_ratio = recent_mae / pre_covid_mae`.

                Lower values are better for MAE, RMSE, selection score, and the ratio metrics.
                Higher values are better for R2 and directional accuracy.
                """
            )

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
            st.markdown(
                """
                The leaderboard uses overall performance for the main rank, but it also includes
                period-specific MAE columns. These are useful for finding models that were not
                just accurate overall, but also resilient through disruption.

                `shock_penalty` compares COVID shock MAE with pre-COVID MAE.
                `recovery_ratio` compares recovery MAE with pre-COVID MAE.
                `recent_recovery_ratio` compares recent MAE with pre-COVID MAE.

                Values near 1.0 mean the model's error was similar to its pre-COVID error.
                Values above 1.0 mean error increased relative to pre-COVID.
                """
            )

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
