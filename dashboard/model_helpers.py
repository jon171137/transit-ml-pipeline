"""Shared model-result helpers for dashboard pages."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from constants import (
    BASELINE_MODEL_BUILDS,
    BASELINE_MODEL_FAMILIES,
    EVALUATION_PERIODS,
    FEATURE_TRANSFORM_LABELS,
    MODEL_BUILD_ORDER,
    MODEL_FAMILY_ORDER,
    PATH_DATASET_DIRS,
    PERIOD_RANK_WINDOWS,
    RANK_METRIC_OPTIONS,
    REQUIRED_FILES,
    SCORE_RECIPES,
)
from data_access import load_path_rows_for_configs, path_collection_modified_ns
from formatting import feature_transform_label, format_float, format_int, model_build_display_label


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
        width="stretch",
        **dataframe_kwargs,
    )


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
    out["feature_transform_label"] = out["feature_transform"].map(feature_transform_label)
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
