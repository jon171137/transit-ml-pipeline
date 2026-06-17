"""Model Explorer page for interactive result inspection."""

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
from formatting import date_range_label, format_int, manifest_value, months_label
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


def render_model_explorer_page(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
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
        st.dataframe(runtime, width="stretch", hide_index=True)
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
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Loaded Artifacts**")
    st.code(str(run_dir), language="text")
    st.write("This app reads curated dashboard artifacts only. It does not trigger training jobs.")
