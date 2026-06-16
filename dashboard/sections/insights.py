"""Insights page for dashboard result interpretation."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from charts import top_model_chart
from constants import FEATURE_TRANSFORM_ORDER, RESULTS_INSIGHTS_NOTEBOOK_PATH
from formatting import format_int
from model_helpers import (
    apply_date_window,
    exclude_baseline_candidates,
    limit_configs_per_build,
    limit_total_configs,
    load_forecast_rows_for_configs,
    model_taxonomy_sort,
    order_index,
    overview_table,
    render_metric_dataframe,
    safe_ratio,
    select_distinct_model_paths,
)


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
        "This page distills selected findings from the experiment results. "
        "Model Explorer is built for interactive filtering; Insights is built "
        "for a guided read of what the runs suggest about model families, "
        "feature choices, evaluation periods, and recurring forecast failure modes."
    )
    notebook_status = "available" if RESULTS_INSIGHTS_NOTEBOOK_PATH.exists() else "planned"
    st.caption(
        f"Companion analysis notebook: `{RESULTS_INSIGHTS_NOTEBOOK_PATH}` ({notebook_status}). "
        "The notebook retains deeper exploratory checks; this page promotes the "
        "findings most useful for reviewing the public dashboard results."
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
            "Question": "Which model families remain competitive across periods?",
            "How this page investigates it": "Compare overall rank against pre-COVID, COVID shock, recovery, and recent-period error.",
        },
        {
            "Question": "Do nonlinear linear-model transforms help?",
            "How this page investigates it": "Compare untransformed, signed-log, quadratic, cubic, and combined transform screens within regularized linear models.",
        },
        {
            "Question": "Which feature policies appear useful?",
            "How this page investigates it": "Compare unfiltered, pruning, mutual information, and tree-based selectors across comparable model families.",
        },
        {
            "Question": "Where do forecasts fail in similar ways?",
            "How this page investigates it": "Inspect shock-window paths, residual behavior, and error concentration by evaluation period.",
        },
    ]
    st.markdown("### Directed Result Questions")
    st.write(
        "The sections below are organized around review questions rather than raw "
        "output tables. Each question separates leaderboard rank from model behavior: "
        "which strategies are accurate overall, which survive hard periods, and where "
        "similar aggregate scores hide different forecast paths."
    )
    st.dataframe(pd.DataFrame(inquiry_rows), use_container_width=True, hide_index=True)

    candidate_models = exclude_baseline_candidates(leaderboard)
    score_col = "selection_score_balanced" if "selection_score_balanced" in candidate_models else "selection_score"
    if not candidate_models.empty and score_col in candidate_models:
        st.markdown("### COVID Shock Forecast Paths")
        st.write(
            "The COVID shock is the clearest stress test in the forecast history. "
            "In the first shock months, the best representative from each model "
            "build generally overpredicts ridership because the models are extrapolating "
            "from pre-shock structure into a sudden break. The value of this view is "
            "not that any model anticipated the break; it shows how quickly different "
            "model families moved back toward the new observed level."
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
                "Selection: top one non-baseline configuration per model build by balanced score, "
                "target months January 2020 through May 2021. The dashed orange line is the "
                "seasonal-naive reference, and the thick black line is actual UPT. In this view, "
                "XGBoost is less wrong during the initial shock than most other model-build "
                "representatives, but all selected families struggle with the abrupt level shift."
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
            "This companion view removes the model-family diversity rule and shows "
            "the ten strongest non-baseline configurations overall. In the current "
            "bundle, all ten are XGBoost, which means the pure leaderboard view is "
            "more concentrated than the per-build comparison above."
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
                "These paths cluster around a shared modeling strategy: boosted trees with history, regime, "
                "and time-context features. They still overpredict the first shock months, but they do so "
                "less severely and more consistently than the broader per-build set."
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
            "The current evidence supports a hypothesis rather than a final explanation: "
            "XGBoost appears to benefit from the interaction between boosted trees and "
            "feature families that already encode recent history, regime context, calendar "
            "timing, and targeted interactions. The top-10 balanced-score slice is entirely "
            "XGBoost, so the next question is not simply whether trees work, but which "
            "boosted-tree ingredients matter most."
        )
        if not top10_overall.empty:
            st.caption(
                f"In the current top-10 balanced-score slice, {xgboost_count} of "
                f"{len(top10_overall)} configurations are XGBoost."
            )
        xgb_rows = [
            {
                "Evidence": "Top 10 overall configurations are all XGBoost",
                "Interpretation": "The strongest leaderboard slice is concentrated in one boosted-tree strategy, not spread evenly across model classes.",
                "Follow-up test": "Compare XGBoost against tuned Random Forest, Extra Trees, and sklearn gradient-boosted trees under matched feature families.",
            },
            {
                "Evidence": "Top XGBoost paths are less wrong in the initial COVID shock than the per-build-diverse set",
                "Interpretation": "Boosted trees may be using regime, time, and history features more effectively under sudden level shifts.",
                "Follow-up test": "Hold the XGBoost model fixed and ablate regime, time, interaction, and service feature groups.",
            },
            {
                "Evidence": "Most top XGBoost configurations use tree_top_30",
                "Interpretation": "Split-based feature selection may align well with split-based final models.",
                "Follow-up test": "Compare none, tree_top_30, mutual information, and correlation pruning inside the same XGBoost grid.",
            },
            {
                "Evidence": "Top models still overpredict early COVID months",
                "Interpretation": "Performance leadership does not mean the structural break was solved.",
                "Follow-up test": "Add residual diagnostics, conformal intervals, and period-specific calibration checks.",
            },
        ]
        st.dataframe(pd.DataFrame(xgb_rows), use_container_width=True, hide_index=True)
        st.info(
            "A useful next experiment is a matched tree-family ablation. More random-forest "
            "variants could test whether the gap is just tree capacity, but random forests do "
            "not reproduce XGBoost's sequential residual-correction mechanism. The cleaner "
            "test is to compare tuned Random Forest, Extra Trees, gradient-boosted trees, and "
            "XGBoost ablations across the same feature families, periods, and scoring rules."
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
            "This summary ranks one row per model build by the best balanced score "
            "found in the current artifact bundle. It is a map of where the search "
            "found strong candidates, not proof that every model family received "
            "identical search depth."
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
                "The transform screen asks whether regularized linear models benefit "
                "from adding broad nonlinear terms. These runs should be read as "
                "screening experiments, not variable-specific transform recommendations: "
                "transformed families append terms such as signed logs or powers to "
                "the selected features, then regularization decides which terms survive."
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
            "Overall rank can hide period-specific behavior. This snapshot compares "
            "the best and median MAE by evaluation period so the dashboard can "
            "separate ordinary forecasting difficulty from pandemic shock and "
            "recovery dynamics."
        )
        st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)
