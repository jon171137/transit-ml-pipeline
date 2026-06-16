import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from constants import MODEL_BUILD_ORDER, MODEL_FAMILY_ORDER, RANK_METRIC_OPTIONS

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

