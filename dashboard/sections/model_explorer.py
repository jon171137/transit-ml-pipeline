"""Model Explorer page for interactive result inspection."""

import json
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from charts import metric_mapping_chart, rolling_error_chart, top_model_chart
from constants import (
    FEATURE_TRANSFORM_LABELS,
    MODEL_BUILD_ORDER,
    MODEL_FAMILY_ORDER,
    PER_BUILD_LIMIT_OPTIONS,
    PERIOD_RANK_WINDOWS,
    RANK_METRIC_OPTIONS,
    TOTAL_LIMIT_OPTIONS,
)
from content import (
    PERIOD_METRIC_EXPLANATION,
    PERIOD_METRIC_SHORT_EXPLANATION,
    REPRESENTATION_AND_COMPLEXITY_EXPLANATION,
)
from formatting import date_range_label, format_float, format_int, manifest_value, months_label
from model_helpers import (
    aggregate_metric_mapping,
    apply_date_window,
    apply_optional_filter,
    apply_optional_multi_filter,
    available_rank_options,
    average_forecast_paths_by_build,
    average_performance_by_build,
    date_bounds,
    default_target_window_for_rank,
    exclude_baseline_candidates,
    filtered_frame,
    limit_configs_per_build,
    limit_total_configs,
    load_forecast_rows_for_configs,
    load_performance_rows_for_configs,
    metric_mapping_frame,
    month_interval_label,
    ordered_model_build_labels,
    ordered_unique,
    overview_table,
    parse_json_display,
    ranked_model_label,
    render_metric_dataframe,
    select_distinct_model_paths,
    selectbox_index,
    sort_by_rank_metric,
)
from ui_components import champion_summary_item


SENSITIVITY_FEATURE_VARIABLES = {
    "Input feature count": "n_input_features",
    "Selected feature count": "n_selected_features",
    "Representation feature count": "n_representation_features",
    "Feature reduction ratio": "feature_reduction_ratio",
    "Sequence length": "sequence_length",
    "Model size proxy": "model_size_proxy",
    "Complexity score": "complexity_score",
    "Compute score": "compute_score",
    "Interpretability score": "interpretability_score",
    "Average train seconds": "avg_train_seconds",
    "Total train seconds": "total_train_seconds",
    "Model run count": "model_run_count",
    "Refit count": "refit_count",
}

SENSITIVITY_COLOR_VARIABLES = {
    "Feature family": "feature_family_name",
    "Feature policy": "feature_policy",
    "Feature transform": "feature_transform_label",
    "Mode": "mode",
}

SENSITIVITY_Y_SLICE_OPTIONS = {
    "All scores": 1.0,
    "Best 99%": 0.99,
    "Best 95%": 0.95,
    "Best 90%": 0.90,
    "Best 75%": 0.75,
    "Best 50%": 0.50,
    "Best 25%": 0.25,
    "Best 10%": 0.10,
}

HYPERPARAMETER_DISPLAY_OVERRIDES = {
    "alpha": "alpha",
    "l1_ratio": "L1 ratio",
    "max_iter": "max iterations",
    "n_estimators": "number of estimators",
    "max_depth": "max depth",
    "max_features": "max features",
    "min_samples_leaf": "min samples leaf",
    "min_child_weight": "min child weight",
    "learning_rate": "learning rate",
    "subsample": "subsample",
    "colsample_bytree": "column sample by tree",
    "batch_size": "batch size",
    "dropout": "dropout",
    "weight_decay": "weight decay",
    "sequence_length": "sequence length",
    "recurrent_hidden_sizes": "recurrent hidden sizes",
    "dense_head_sizes": "dense head sizes",
    "dense_head_dropouts": "dense head dropouts",
    "pre_head_dropout": "pre-head dropout",
    "inter_recurrent_dropouts": "inter-recurrent dropouts",
    "early_stopping_patience": "early stopping patience",
    "validation_rows": "validation rows",
    "max_epochs": "max epochs",
    "lr_factor": "LR factor",
    "lr_patience": "LR patience",
    "min_lr": "minimum LR",
}


def safe_hyperparameter_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_").lower()


def compact_hyperparameter_value(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(compact_hyperparameter_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def numeric_list(value) -> list[float]:
    if not isinstance(value, list):
        return []
    numeric_values = []
    for item in value:
        if isinstance(item, bool):
            numeric_values.append(float(item))
        elif isinstance(item, (int, float)):
            numeric_values.append(float(item))
    return numeric_values


def parse_hyperparameter_record(raw_value) -> dict:
    if raw_value is None or raw_value == "":
        return {}
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return {}
    if isinstance(raw_value, dict):
        params = raw_value
    else:
        try:
            params = json.loads(str(raw_value))
        except (TypeError, json.JSONDecodeError):
            return {}
    if not isinstance(params, dict):
        return {}

    record = {}
    for raw_key, value in params.items():
        key = safe_hyperparameter_name(raw_key)
        if not key:
            continue
        column = f"hp_{key}"
        values = numeric_list(value)
        if values:
            record[column] = compact_hyperparameter_value(value)
            record[f"{column}_first"] = values[0]
            record[f"{column}_total"] = sum(values)
            record[f"{column}_max"] = max(values)
            record[f"{column}_count"] = len(values)
        elif isinstance(value, (int, float, bool)) and not isinstance(value, bool):
            record[column] = value
        elif isinstance(value, bool):
            record[column] = str(value)
        else:
            record[column] = compact_hyperparameter_value(value)
    return record


def format_sensitivity_value(value) -> str:
    if pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return format_int(numeric)
    if abs(numeric) < 0.01:
        return f"{numeric:.4g}"
    if abs(numeric) < 1:
        return f"{numeric:.3f}".rstrip("0").rstrip(".")
    return f"{numeric:,.2f}".rstrip("0").rstrip(".")


def add_hyperparameter_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "hyperparameters_json" not in frame.columns:
        return frame.copy()
    records = [parse_hyperparameter_record(value) for value in frame["hyperparameters_json"]]
    param_frame = pd.DataFrame(records, index=frame.index)
    if param_frame.empty:
        return frame.copy()
    return pd.concat([frame.copy(), param_frame], axis=1)


def join_complexity_columns(leaderboard: pd.DataFrame, complexity_profile: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = leaderboard.copy()
    if complexity_profile is None or complexity_profile.empty:
        return out
    key = "model_config_id" if "model_config_id" in out and "model_config_id" in complexity_profile else "config_id"
    if key not in out or key not in complexity_profile:
        return out
    extra_columns = [
        "avg_train_seconds",
        "total_train_seconds",
        "model_run_count",
        "refit_count",
        "avg_n_train",
        "min_selected_features",
        "max_selected_features",
    ]
    extra_columns = [column for column in extra_columns if column in complexity_profile and column not in out]
    if not extra_columns:
        return out
    extras = complexity_profile[[key, *extra_columns]].drop_duplicates(subset=[key])
    return out.merge(extras, on=key, how="left")


def is_varied(series: pd.Series) -> bool:
    return series.dropna().astype(str).nunique() > 1


def is_mostly_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return converted.notna().mean() >= 0.9


def hyperparameter_label(column: str) -> str:
    raw_name = column.removeprefix("hp_")
    suffix_label = ""
    for suffix, label in [
        ("_first", "first"),
        ("_total", "total"),
        ("_max", "max"),
        ("_count", "count"),
    ]:
        if raw_name.endswith(suffix):
            raw_name = raw_name[: -len(suffix)]
            suffix_label = f" ({label})"
            break
    display_name = HYPERPARAMETER_DISPLAY_OVERRIDES.get(raw_name, raw_name.replace("_", " "))
    return f"Hyperparameter: {display_name}{suffix_label}"


def sensitivity_axis_options(frame: pd.DataFrame) -> dict[str, str]:
    options = {}
    hyperparameter_columns = [
        column
        for column in frame.columns
        if column.startswith("hp_") and is_varied(frame[column])
    ]
    hyperparameter_columns = sorted(
        hyperparameter_columns,
        key=lambda column: (
            0 if is_mostly_numeric(frame[column]) else 1,
            hyperparameter_label(column),
        ),
    )
    for column in hyperparameter_columns:
        options[hyperparameter_label(column)] = column
    for label, column in SENSITIVITY_FEATURE_VARIABLES.items():
        if column in frame and is_varied(frame[column]):
            options[label] = column
    return options


def sensitivity_color_options(frame: pd.DataFrame) -> dict[str, str]:
    options = {"None": ""}
    for label, column in SENSITIVITY_COLOR_VARIABLES.items():
        if column in frame and is_varied(frame[column]):
            options[label] = column
    for column in sorted([col for col in frame.columns if col.startswith("hp_")]):
        if column in frame and is_varied(frame[column]) and frame[column].dropna().astype(str).nunique() <= 12:
            options[hyperparameter_label(column)] = column
    return options


def sensitivity_size_options(frame: pd.DataFrame) -> dict[str, str]:
    options = {"None": ""}
    for label, column in SENSITIVITY_FEATURE_VARIABLES.items():
        if column not in frame or not is_varied(frame[column]):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any() and numeric.max() > 0:
            options[label] = column
    return options


def default_sensitivity_x_label(options: dict[str, str]) -> str:
    preferred_fragments = [
        "alpha",
        "learning rate",
        "max depth",
        "number of estimators",
        "recurrent hidden sizes (total)",
        "dense head sizes (total)",
        "Selected feature count",
    ]
    for fragment in preferred_fragments:
        for label in options:
            if fragment.lower() in label.lower():
                return label
    return next(iter(options))


def sensitivity_hover_columns(frame: pd.DataFrame, selected_columns: list[str]) -> list[str]:
    columns = [
        "model_config_id",
        "model_build_label",
        "feature_family_name",
        "feature_policy",
        "feature_transform_label",
        "mode",
        "mae",
        "rmse",
        "r2",
        "selection_score_balanced",
        "n_selected_features",
        "n_representation_features",
        "complexity_score",
        "compute_score",
        *selected_columns,
    ]
    seen = set()
    available = []
    for column in columns:
        if column in frame and column not in seen:
            available.append(column)
            seen.add(column)
    return available


def sensitivity_relationship_chart(
    frame: pd.DataFrame,
    x_label: str,
    x_col: str,
    y_label: str,
    y_col: str,
    color_label: str,
    color_col: str,
    size_label: str,
    size_col: str,
):
    plot_frame = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_frame.empty:
        fig = px.scatter(pd.DataFrame({"x": [], "y": []}), x="x", y="y")
        fig.update_layout(title="No matching build-sensitivity points")
        return fig

    x_numeric = is_mostly_numeric(plot_frame[x_col])
    x_unique = plot_frame[x_col].dropna().astype(str).nunique()
    color_arg = color_col if color_col and color_col in plot_frame.columns else None
    size_arg = None
    if size_col and size_col in plot_frame.columns and x_numeric and x_unique > 12:
        plot_frame[size_col] = pd.to_numeric(plot_frame[size_col], errors="coerce")
        if plot_frame[size_col].notna().any() and plot_frame[size_col].max() > 0:
            size_arg = size_col

    selected_hover_columns = [col for col in [x_col, color_col, size_col] if col]
    hover_columns = sensitivity_hover_columns(plot_frame, selected_hover_columns)

    if x_numeric and x_unique > 12:
        plot_frame[x_col] = pd.to_numeric(plot_frame[x_col], errors="coerce")
        fig = px.scatter(
            plot_frame,
            x=x_col,
            y=y_col,
            color=color_arg,
            size=size_arg,
            hover_data=hover_columns,
            labels={
                x_col: x_label,
                y_col: y_label,
                color_col: color_label,
                size_col: size_label,
            },
        )
    else:
        x_display_col = "_sensitivity_x_display"
        if x_numeric:
            numeric_values = pd.to_numeric(plot_frame[x_col], errors="coerce")
            plot_frame[x_display_col] = numeric_values.map(format_sensitivity_value)
            category_order = [
                format_sensitivity_value(value)
                for value in sorted(numeric_values.dropna().unique())
            ]
        else:
            plot_frame[x_display_col] = plot_frame[x_col].astype(str)
            category_order = sorted(plot_frame[x_display_col].dropna().unique())
        fig = px.strip(
            plot_frame,
            x=x_display_col,
            y=y_col,
            color=color_arg,
            hover_data=hover_columns,
            category_orders={x_display_col: category_order},
            labels={
                x_display_col: x_label,
                y_col: y_label,
                color_col: color_label,
            },
            stripmode="overlay",
        )
    fig.update_traces(marker=dict(opacity=0.74, line=dict(width=0.5, color="white")))
    fig.update_layout(
        title=f"{y_label} by {x_label}",
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=60),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#2f323a"),
    )
    return fig


def sensitivity_summary_frame(
    frame: pd.DataFrame,
    x_label: str,
    x_col: str,
    y_label: str,
    y_col: str,
    metric_ascending: bool,
) -> pd.DataFrame:
    if frame.empty or x_col not in frame or y_col not in frame:
        return pd.DataFrame()
    summary_source = frame.dropna(subset=[x_col, y_col]).copy()
    if summary_source.empty:
        return pd.DataFrame()

    x_numeric = is_mostly_numeric(summary_source[x_col])
    if x_numeric:
        numeric_x = pd.to_numeric(summary_source[x_col], errors="coerce")
        unique_count = numeric_x.dropna().nunique()
        if unique_count > 12:
            bin_count = min(6, unique_count)
            summary_source["_x_group"] = pd.qcut(numeric_x, q=bin_count, duplicates="drop").astype(str)
            group_label = f"{x_label} range"
        else:
            summary_source["_x_group"] = numeric_x.map(format_sensitivity_value)
            group_label = x_label
    else:
        summary_source["_x_group"] = summary_source[x_col].astype(str)
        group_label = x_label

    agg_kwargs = {
        "configurations": (y_col, "size"),
        f"best_{y_col}": (y_col, "min" if metric_ascending else "max"),
        f"median_{y_col}": (y_col, "median"),
    }
    if "mae" in summary_source:
        agg_kwargs["best_mae"] = ("mae", "min")
        agg_kwargs["median_mae"] = ("mae", "median")
    if "n_selected_features" in summary_source:
        agg_kwargs["median_selected_features"] = ("n_selected_features", "median")
    if "complexity_score" in summary_source:
        agg_kwargs["median_complexity_score"] = ("complexity_score", "median")

    summary = summary_source.groupby("_x_group", dropna=False).agg(**agg_kwargs).reset_index()
    summary = summary.rename(
        columns={
            "_x_group": group_label,
            f"best_{y_col}": f"Best {y_label}",
            f"median_{y_col}": f"Median {y_label}",
            "best_mae": "Best MAE",
            "median_mae": "Median MAE",
            "median_selected_features": "Median selected features",
            "median_complexity_score": "Median complexity score",
            "configurations": "Configurations",
        }
    )
    sort_col = f"Best {y_label}"
    return summary.sort_values(sort_col, ascending=metric_ascending)


def apply_sensitivity_y_slice(
    frame: pd.DataFrame,
    y_col: str,
    metric_ascending: bool,
    slice_label: str,
) -> pd.DataFrame:
    if frame.empty or slice_label == "All scores":
        return frame
    keep_share = SENSITIVITY_Y_SLICE_OPTIONS.get(slice_label, 1.0)
    if keep_share >= 1.0:
        return frame
    ranked = frame.dropna(subset=[y_col]).sort_values(y_col, ascending=metric_ascending).copy()
    keep_count = max(1, math.ceil(len(ranked) * keep_share))
    return ranked.head(keep_count)


def render_build_sensitivity_inspector(
    leaderboard: pd.DataFrame,
    complexity_profile: Optional[pd.DataFrame],
    metric_options: list[str],
) -> None:
    st.subheader("Build Sensitivity Inspector")
    st.write(
        "Inspect one model build at a time to see how configuration choices line up "
        "with scoring metrics. This view uses config-level leaderboard metadata and "
        "does not load path-level forecast rows."
    )
    if leaderboard.empty or "model_build_label" not in leaderboard:
        st.info("No model-build metadata is available for sensitivity inspection.")
        return

    sensitivity_base = add_hyperparameter_columns(join_complexity_columns(leaderboard, complexity_profile))
    build_options = ordered_model_build_labels(sensitivity_base)
    if not build_options:
        st.info("No model builds are available for sensitivity inspection.")
        return

    build_cols = st.columns([1.35, 0.9, 0.9, 0.95, 0.8])
    selected_build = build_cols[0].selectbox(
        "Model build",
        build_options,
        key="sensitivity_model_build",
    )
    build_frame = sensitivity_base[sensitivity_base["model_build_label"].astype(str).eq(selected_build)].copy()
    y_metric = build_cols[1].selectbox(
        "Y metric",
        metric_options,
        index=selectbox_index(metric_options, "MAE"),
        key="sensitivity_y_metric",
    )
    y_score_slice = build_cols[2].selectbox(
        "Y score slice",
        list(SENSITIVITY_Y_SLICE_OPTIONS),
        index=0,
        key="sensitivity_y_score_slice",
        help=(
            "Keeps the best share of configurations by the selected Y metric. "
            "For error metrics this keeps the lowest values; for R-squared and "
            "directional accuracy it keeps the highest values."
        ),
    )
    rank_metric = build_cols[3].selectbox(
        "Rank slice by",
        metric_options,
        index=selectbox_index(metric_options, "Balanced score"),
        key="sensitivity_rank_metric",
    )
    top_total = build_cols[4].selectbox(
        "Configs shown",
        TOTAL_LIMIT_OPTIONS,
        index=selectbox_index(TOTAL_LIMIT_OPTIONS, "Top 100"),
        key="sensitivity_total_limit",
    )

    ranked_frame = limit_total_configs(build_frame, rank_metric, top_total)
    axis_options = sensitivity_axis_options(ranked_frame)
    if not axis_options:
        st.info("The selected model build does not have varying hyperparameter or feature-size fields to plot.")
        return

    x_default = default_sensitivity_x_label(axis_options)
    relationship_cols = st.columns([1.45, 1.1, 1.0])
    x_label = relationship_cols[0].selectbox(
        "X variable",
        list(axis_options),
        index=selectbox_index(list(axis_options), x_default),
        key="sensitivity_x_variable",
    )
    color_options = sensitivity_color_options(ranked_frame)
    color_label = relationship_cols[1].selectbox(
        "Color by",
        list(color_options),
        index=0,
        key="sensitivity_color_by",
    )
    size_options = sensitivity_size_options(ranked_frame)
    size_default = "Selected feature count" if "Selected feature count" in size_options else "None"
    size_label = relationship_cols[2].selectbox(
        "Size by",
        list(size_options),
        index=selectbox_index(list(size_options), size_default),
        key="sensitivity_size_by",
    )

    y_col, metric_ascending = RANK_METRIC_OPTIONS[y_metric]
    x_col = axis_options[x_label]
    color_col = color_options[color_label]
    size_col = size_options[size_label]
    if y_col not in ranked_frame:
        st.info(f"`{y_metric}` is not available for this model build.")
        return
    chart_frame_full = ranked_frame.dropna(subset=[x_col, y_col]).copy()
    chart_frame = apply_sensitivity_y_slice(
        chart_frame_full,
        y_col,
        metric_ascending,
        y_score_slice,
    )
    if chart_frame.empty:
        st.info("No model configurations match the selected sensitivity controls.")
        return

    st.caption(
        f"Showing {format_int(len(chart_frame))} of {format_int(len(chart_frame_full))} "
        f"`{selected_build}` configuration(s). "
        f"Y score slice: `{y_score_slice}`. "
        "Lower is better for error metrics; higher is better for R-squared and directional accuracy."
    )
    st.plotly_chart(
        sensitivity_relationship_chart(
            chart_frame,
            x_label,
            x_col,
            y_metric,
            y_col,
            color_label,
            color_col,
            size_label,
            size_col,
        ),
        width="stretch",
    )
    summary = sensitivity_summary_frame(chart_frame, x_label, x_col, y_metric, y_col, metric_ascending)
    if not summary.empty:
        st.markdown("**Grouped sensitivity summary**")
        render_metric_dataframe(summary, max_rows=30, max_visible_rows=10)


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


def format_duration(seconds) -> str:
    if pd.isna(seconds):
        return "-"
    value = float(seconds)
    if value < 0:
        return "-"
    if value < 0.01:
        return f"{value * 1000:.2f} ms"
    if value < 1:
        return f"{value * 1000:.1f} ms"
    if value < 60:
        return f"{value:.2f} sec"
    if value < 3600:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.1f} hr"


def selected_complexity_profile(complexity_profile: Optional[pd.DataFrame], selected_models: pd.DataFrame) -> pd.DataFrame:
    if complexity_profile is None or complexity_profile.empty or selected_models.empty:
        return pd.DataFrame()
    if "config_id" not in complexity_profile or "config_id" not in selected_models:
        return pd.DataFrame()

    config_ids = set(selected_models["config_id"].dropna().astype(str))
    profile = complexity_profile.copy()
    profile["config_id"] = profile["config_id"].astype(str)
    return profile[profile["config_id"].isin(config_ids)].copy()


def operational_footprint_summary(profile: pd.DataFrame) -> pd.DataFrame:
    required = {"model_build_label", "config_id", "total_train_seconds", "avg_train_seconds"}
    if profile.empty or not required.issubset(profile.columns):
        return pd.DataFrame()

    aggregations = {
        "configurations": ("config_id", "nunique"),
        "total_train_seconds": ("total_train_seconds", "sum"),
        "median_config_train_seconds": ("total_train_seconds", "median"),
        "median_refit_train_seconds": ("avg_train_seconds", "median"),
    }
    optional_aggs = {
        "refit_count": ("refit_count", "median"),
        "avg_n_train": ("avg_n_train", "median"),
        "compute_score": ("compute_score", "median"),
        "complexity_score": ("complexity_score", "median"),
    }
    for column, spec in optional_aggs.items():
        if column in profile.columns:
            aggregations[column] = spec

    group_cols = ["model_family", "model_build", "model_build_label"]
    summary = (
        profile.groupby(group_cols, dropna=False, as_index=False)
        .agg(**aggregations)
        .sort_values("total_train_seconds", ascending=False)
    )
    return summary


def render_operational_footprint(complexity_profile: Optional[pd.DataFrame], selected_models: pd.DataFrame) -> None:
    st.subheader("Operational Footprint")
    active_profile = selected_complexity_profile(complexity_profile, selected_models)
    summary = operational_footprint_summary(active_profile)
    if summary.empty:
        st.info(
            "No measured training-footprint artifact is available for the active model selection. "
            "The dashboard can compare accuracy, but it cannot yet report training or inference cost "
            "for this slice."
        )
        return

    total_configs = int(active_profile["config_id"].nunique())
    total_train_seconds = pd.to_numeric(active_profile["total_train_seconds"], errors="coerce").sum()
    median_config_seconds = pd.to_numeric(active_profile["total_train_seconds"], errors="coerce").median()
    median_refit_seconds = pd.to_numeric(active_profile["avg_train_seconds"], errors="coerce").median()
    most_expensive = summary.iloc[0]["model_build_label"]

    st.caption(
        "Training footprint is calculated from `complexity_profile_full.parquet` for the active "
        "Top Models selection above. These are recorded experiment-run fit times across rolling "
        "historical refits, not live infrastructure costs. Inference latency and memory use are not "
        "currently logged in the artifact bundle."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected configs", format_int(total_configs))
    metric_cols[1].metric("Total recorded train time", format_duration(total_train_seconds))
    metric_cols[2].metric("Median train time / config", format_duration(median_config_seconds))
    metric_cols[3].metric("Median fit time / refit", format_duration(median_refit_seconds))
    st.caption(f"Largest selected training share: {most_expensive}.")

    fig = px.bar(
        summary,
        x="model_build_label",
        y="total_train_seconds",
        color="model_family",
        category_orders={
            "model_family": MODEL_FAMILY_ORDER,
            "model_build_label": ordered_model_build_labels(summary),
        },
        labels={
            "model_build_label": "Model build",
            "model_family": "Model family",
            "total_train_seconds": "Total recorded training seconds",
        },
        hover_data={
            "configurations": ":,.0f",
            "total_train_seconds": ":,.2f",
            "median_config_train_seconds": ":,.2f",
            "median_refit_train_seconds": ":,.4f",
        },
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#2f323a"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    display = summary.copy()
    display["Total train time"] = display["total_train_seconds"].map(format_duration)
    display["Median train time / config"] = display["median_config_train_seconds"].map(format_duration)
    display["Median fit time / refit"] = display["median_refit_train_seconds"].map(format_duration)
    if "refit_count" in display:
        display["Median refits / config"] = display["refit_count"].map(format_int)
    if "avg_n_train" in display:
        display["Median training rows / refit"] = display["avg_n_train"].map(format_int)
    if "compute_score" in display:
        display["Median compute score"] = display["compute_score"].map(lambda value: format_float(value, 1))
    if "complexity_score" in display:
        display["Median complexity score"] = display["complexity_score"].map(lambda value: format_float(value, 1))

    display_cols = [
        "model_build_label",
        "configurations",
        "Total train time",
        "Median train time / config",
        "Median fit time / refit",
        "Median refits / config",
        "Median training rows / refit",
        "Median compute score",
        "Median complexity score",
    ]
    display_cols = [column for column in display_cols if column in display.columns]
    display = display[display_cols].rename(
        columns={
            "model_build_label": "Model build",
            "configurations": "Configurations",
        }
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=min(460, 38 * (len(display) + 1)),
    )
    st.markdown(
        """
        **How the complexity score is calculated.** The score is a normalized
        within-experiment comparison, not an absolute engineering cost. Each
        model configuration gets three normalized inputs: selected feature
        count, a model-size proxy, and recorded training time. The final score
        is scaled from 0 to 100:

        `complexity_score = 100 * (0.40 * feature_count + 0.35 * model_size + 0.25 * training_time)`

        The model-size proxy is tailored by model type. Baseline models are set
        near 1. Linear models use selected feature count. Random Forest and Extra
        Trees use roughly `n_estimators * max_depth`. XGBoost uses
        `n_estimators * max_depth * colsample_bytree`. ARIMA/SARIMA/SARIMAX use
        order terms plus feature count. Neural models approximate parameter
        count from sequence length, feature count, hidden sizes, recurrent gate
        count, and dense head layers.

        `compute_score` is narrower: it is based only on log-normalized total
        training seconds. A higher complexity score therefore means the selected
        configuration is heavier relative to this experiment's model set, but it
        does not directly report memory use, infrastructure cost, or inference
        latency.
        """
    )

    context_cols = [col for col in ["framework", "hardware_type", "device", "gpu_name", "cuda_version"] if col in active_profile]
    if context_cols:
        context = active_profile[context_cols].replace("", pd.NA).drop_duplicates().dropna(how="all")
        if not context.empty:
            with st.expander("Recorded runtime context"):
                st.dataframe(context.head(12), width="stretch", hide_index=True)


def render_model_explorer_page(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
    complexity_profile: Optional[pd.DataFrame],
    champion: dict,
    experiment_manifest: dict,
) -> None:
    candidate_leaderboard = exclude_baseline_candidates(leaderboard)
    render_champion_snapshot(champion, forecast_paths, experiment_manifest)
    st.subheader("Top Models Against Actual Ridership")
    st.info(
        "How to read this page: use Top 1 per build to compare model families, "
        "then use Top total to see whether the leaderboard concentrates around "
        "one strategy. The date window is useful for stress-testing specific "
        "periods such as COVID shock or recovery, and Metric Mapping below helps "
        "separate accuracy from tradeoffs like shock behavior and stability."
    )
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
    chart_paths = load_forecast_rows_for_configs(run_dir, filtered_top, candidate_configs)
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
        width="stretch",
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
    rolling_subset = load_performance_rows_for_configs(run_dir, filtered_top, rolling_configs)
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
    st.plotly_chart(rolling_error_chart(rolling_subset), width="stretch")
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
        st.plotly_chart(fig, width="stretch")
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
            run_dir,
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
            width="stretch",
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
            width="stretch",
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
    render_build_sensitivity_inspector(candidate_leaderboard, complexity_profile, metric_options)
    render_operational_footprint(complexity_profile, filtered_top)
    st.markdown("**Loaded Artifacts**")
    st.code(str(run_dir), language="text")
    st.write("This app reads curated dashboard artifacts only. It does not trigger training jobs.")
