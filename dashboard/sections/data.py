"""Data page and source-data EDA helpers."""

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from charts import (
    correlation_heatmap_figure,
    granger_predictive_figure,
    lagged_correlation_figure,
    source_series_figure,
)
from content import (
    DATA_CALCULATED_FEATURES,
    DATA_PRIMARY_DATA,
    DATA_SECONDARY_DATA,
    DATA_TIME_FEATURES,
)
from data_access import (
    configured_feature_table_path,
    configured_imputation_log_path,
    configured_integrated_base_path,
    file_modified_ns,
    load_feature_table,
    load_imputation_log,
    load_integrated_base,
)
from formatting import format_int
from ui_components import summary_panel_from_markdown


def integrated_source_series_options() -> list[dict[str, str]]:
    return [
        {
            "column": "upt",
            "label": "UPT",
            "description": "Unlinked passenger trips",
            "unit": "Passenger boardings",
            "source": "FTA NTD",
        },
        {
            "column": "vrm",
            "label": "VRM",
            "description": "Vehicle revenue miles",
            "unit": "Miles",
            "source": "FTA NTD",
        },
        {
            "column": "vrh",
            "label": "VRH",
            "description": "Vehicle revenue hours",
            "unit": "Hours",
            "source": "FTA NTD",
        },
        {
            "column": "voms",
            "label": "VOMS",
            "description": "Vehicles operated in maximum service",
            "unit": "Vehicles",
            "source": "FTA NTD",
        },
        {
            "column": "seattle_gas_price_avg",
            "label": "Gas Avg",
            "description": "Seattle gasoline price average",
            "unit": "Dollars per gallon",
            "source": "EIA",
        },
        {
            "column": "seattle_gas_price_std",
            "label": "Gas Std",
            "description": "Within-month Seattle gasoline price standard deviation",
            "unit": "Dollars per gallon",
            "source": "EIA",
        },
        {
            "column": "cpi_all_items_sa",
            "label": "CPI All",
            "description": "CPI all items, seasonally adjusted",
            "unit": "Index",
            "source": "FRED",
        },
        {
            "column": "cpi_core_sa",
            "label": "CPI Core",
            "description": "CPI all items less food and energy, seasonally adjusted",
            "unit": "Index",
            "source": "FRED",
        },
        {
            "column": "king_county_median_household_income_prior_year",
            "label": "Income",
            "description": "King County median household income, prior-year context",
            "unit": "Dollars",
            "source": "FRED",
        },
    ]


def integrated_source_series_data() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame(), []

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    if "date" not in df.columns:
        return pd.DataFrame(), []

    options = [option for option in integrated_source_series_options() if option["column"] in df.columns]
    if not options:
        return pd.DataFrame(), []

    cols = ["date", *[option["column"] for option in options]]
    data = (
        df[cols]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .dropna(subset=["date"])
    )
    return data, options


def format_month(value) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%b %Y")


def date_spine_missing_count(df: pd.DataFrame) -> int:
    if df.empty or "date" not in df:
        return 0
    dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.to_period("M").dt.to_timestamp()
    if dates.empty:
        return 0
    expected = pd.date_range(dates.min(), dates.max(), freq="MS")
    return int(len(expected.difference(pd.DatetimeIndex(dates))))


def availability_row(
    df: pd.DataFrame,
    column: str,
    label: str,
    source: str,
    stage: str,
    note: str = "",
) -> dict:
    if df.empty or "date" not in df or column not in df:
        return {
            "Series": label,
            "Source": source,
            "Stage": stage,
            "First available": "-",
            "Last available": "-",
            "Observed months": 0,
            "Missing months": "-",
            "Missing %": "-",
            "Note": note or "Column not present in this artifact.",
        }

    dates = pd.to_datetime(df["date"], errors="coerce")
    values = df[column]
    observed_dates = dates[values.notna()]
    observed = int(values.notna().sum())
    missing = int(values.isna().sum())
    total = int(len(values))
    return {
        "Series": label,
        "Source": source,
        "Stage": stage,
        "First available": format_month(observed_dates.min()) if observed else "-",
        "Last available": format_month(observed_dates.max()) if observed else "-",
        "Observed months": observed,
        "Missing months": missing,
        "Missing %": f"{(missing / total * 100):.1f}%" if total else "-",
        "Note": note,
    }


def data_availability_report_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    integrated_path = configured_integrated_base_path()
    feature_path = configured_feature_table_path()
    imputation_log_path = configured_imputation_log_path()

    integrated = pd.DataFrame()
    feature_table = pd.DataFrame()
    imputation_log = pd.DataFrame()
    if integrated_path.exists():
        integrated = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
        if "date" in integrated:
            integrated["date"] = pd.to_datetime(integrated["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if feature_path.exists():
        feature_table = load_feature_table(str(feature_path), file_modified_ns(feature_path)).copy()
        if "date" in feature_table:
            feature_table["date"] = pd.to_datetime(feature_table["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if imputation_log_path.exists():
        imputation_log = load_imputation_log(str(imputation_log_path), file_modified_ns(imputation_log_path)).copy()

    availability_specs = [
        ("upt", "UPT", "FTA NTD", "Integrated base", "Transit ridership is complete across the integrated monthly spine."),
        ("vrm", "VRM", "FTA NTD", "Integrated base", "Service miles are complete across the integrated monthly spine."),
        ("vrh", "VRH", "FTA NTD", "Integrated base", "Service hours are complete across the integrated monthly spine."),
        ("voms", "VOMS", "FTA NTD", "Integrated base", "Peak vehicles are complete across the integrated monthly spine."),
        ("seattle_gas_price_avg", "Seattle gas price avg", "EIA", "Integrated base", "Gas data begins in May 2003; earlier months remain unavailable in the raw integrated base."),
        ("seattle_gas_price_std", "Seattle gas price std", "EIA", "Integrated base", "Within-month gas price variability begins with the gas source series in May 2003."),
        ("cpi_all_items_sa", "CPI all items", "FRED", "Integrated base", "The local integrated-base copy has one CPI gap, while the model-ready feature table is complete."),
        ("cpi_core_sa", "CPI core", "FRED", "Integrated base", "The local integrated-base copy has one CPI gap, while the model-ready feature table is complete."),
        ("king_county_median_household_income_prior_year", "King County income", "FRED", "Feature table", "Annual income is converted to prior-year monthly context for modeling."),
    ]
    availability_rows = []
    for column, label, source, stage, note in availability_specs:
        artifact = feature_table if stage == "Feature table" else integrated
        availability_rows.append(availability_row(artifact, column, label, source, stage, note))
    availability = pd.DataFrame(availability_rows)

    readiness_rows = []
    if not integrated.empty and "date" in integrated:
        readiness_rows.append(
            {
                "Check": "Integrated monthly date spine",
                "Result": f"{format_month(integrated['date'].min())} to {format_month(integrated['date'].max())}",
                "Detail": f"{len(integrated):,} rows; {date_spine_missing_count(integrated):,} missing calendar months.",
            }
        )
    if not feature_table.empty and "date" in feature_table:
        target_missing = int(feature_table["upt_target_h3"].isna().sum()) if "upt_target_h3" in feature_table else 0
        readiness_rows.append(
            {
                "Check": "Model-ready feature table",
                "Result": f"{format_month(feature_table['date'].min())} to {format_month(feature_table['date'].max())}",
                "Detail": (
                    f"{len(feature_table):,} rows; starts later because gas availability and lag/rolling features "
                    "require historical lookback."
                ),
            }
        )
        readiness_rows.append(
            {
                "Check": "H3 target availability",
                "Result": f"{target_missing:,} missing target rows",
                "Detail": "The final three as-of rows naturally lack observed future UPT targets.",
            }
        )
        base_cols = [
            "upt",
            "vrm",
            "vrh",
            "voms",
            "seattle_gas_price_avg",
            "seattle_gas_price_std",
            "cpi_all_items_sa",
            "cpi_core_sa",
            "king_county_median_household_income_prior_year",
        ]
        available_base_cols = [col for col in base_cols if col in feature_table]
        base_missing = int(feature_table[available_base_cols].isna().sum().sum()) if available_base_cols else 0
        readiness_rows.append(
            {
                "Check": "Base source values in feature table",
                "Result": f"{base_missing:,} missing values",
                "Detail": "The modeling input rows have complete base source values after trimming/preparation.",
            }
        )
    readiness = pd.DataFrame(readiness_rows)

    imputation_cols = [
        col
        for col in feature_table.columns
        if "imputed" in str(col) or str(col).endswith("_was_imputed")
    ] if not feature_table.empty else []
    imputation_rows = []
    for col in imputation_cols:
        active = int(pd.to_numeric(feature_table[col], errors="coerce").fillna(0).sum())
        imputation_rows.append(
            {
                "Imputation flag": col,
                "Active rows": active,
                "Active %": f"{(active / len(feature_table) * 100):.1f}%" if len(feature_table) else "-",
            }
        )
    imputation_summary = pd.DataFrame(imputation_rows)
    if not imputation_summary.empty:
        imputation_summary = imputation_summary.sort_values(["Active rows", "Imputation flag"], ascending=[False, True])

    metadata = {
        "integrated_exists": integrated_path.exists(),
        "feature_table_exists": feature_path.exists(),
        "imputation_log_exists": imputation_log_path.exists(),
        "imputation_log_rows": int(len(imputation_log)),
    }
    return availability, readiness, imputation_summary, metadata


def render_data_availability_report() -> None:
    availability, readiness, imputation_summary, metadata = data_availability_report_tables()
    st.markdown("### Data Availability, Missingness, And Imputation")
    st.write(
        "This report summarizes the joined source data before modeling and the "
        "model-ready feature table after trimming, lag construction, and source "
        "preparation. The distinction matters: early source gaps can exist in the "
        "integrated base even when the final modeling rows are complete."
    )
    if availability.empty and readiness.empty:
        st.info("Source availability artifacts were not found in this environment.")
        return

    if not readiness.empty:
        st.markdown("**Pipeline readiness checks**")
        st.dataframe(readiness, width="stretch", hide_index=True)

    if not availability.empty:
        st.markdown("**Source series availability**")
        st.dataframe(availability, width="stretch", hide_index=True)

    st.markdown("**Imputation activity**")
    if imputation_summary.empty:
        st.write(
            "No imputation indicator columns were found in the feature table. "
            "That usually means this artifact was produced before imputation flags were added."
        )
    else:
        active_total = int(imputation_summary["Active rows"].sum())
        st.write(
            f"The feature table includes imputation indicators, but this run has "
            f"{active_total:,} active imputation-flag rows. The imputation log contains "
            f"{metadata.get('imputation_log_rows', 0):,} row(s)."
        )
        st.dataframe(imputation_summary, width="stretch", hide_index=True)
    st.caption(
        "Imputation is designed for inside-window interpolation and trailing trend fills on selected "
        "monthly exogenous series. In the current modeling artifact, the selected date window and "
        "available source files leave those flags inactive."
    )


def pre_covid_mom_callouts(threshold: float = 2.0) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    callout_cols = [c for c in ["upt", "vrm", "vrh", "seattle_gas_price_avg"] if c in df.columns]
    if not callout_cols:
        return pd.DataFrame()

    pre_covid = (
        df.loc[df["date"] <= pd.Timestamp("2019-12-01"), ["date", *callout_cols]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    mom_pct = pre_covid.set_index("date")[callout_cols].pct_change(fill_method=None) * 100
    mom_pct = mom_pct.replace([float("inf"), float("-inf")], pd.NA)
    mom_pct["month_num"] = mom_pct.index.month
    avg_mom_pct_by_month = mom_pct.groupby("month_num")[callout_cols].mean().reindex(range(1, 13))

    label_map = {
        "upt": "UPT",
        "vrm": "VRM",
        "vrh": "VRH",
        "seattle_gas_price_avg": "Gas",
    }
    series_order = {col: idx for idx, col in enumerate(callout_cols)}
    rows = []
    for month_num, row in avg_mom_pct_by_month.iterrows():
        for col in callout_cols:
            value = row[col]
            if pd.notna(value) and abs(value) > threshold:
                rows.append(
                    {
                        "month_num": month_num,
                        "month": pd.Timestamp(2000, month_num, 1).strftime("%B"),
                        "series": label_map.get(col, col),
                        "series_order": series_order[col],
                        "avg_mom_pct": float(value),
                    }
                )
    return pd.DataFrame(rows)


def pre_covid_mom_callouts_html(callouts: pd.DataFrame, threshold: float = 2.0) -> str:
    if callouts.empty:
        return ""

    month_cards = []
    for month_num, group in callouts.sort_values(["month_num", "series_order"]).groupby("month_num", sort=True):
        month_name = str(group["month"].iloc[0])
        badges = []
        for row in group.itertuples(index=False):
            direction = "positive" if row.avg_mom_pct > 0 else "negative"
            series_kind = "gas" if str(row.series).lower() == "gas" else "transit"
            badge_class = f"{series_kind}-{direction}"
            value = f"{row.avg_mom_pct:+.1f}%"
            badges.append(
                f'<div class="mom-badge {badge_class}">'
                f'<div class="series">{escape(str(row.series))}</div>'
                f'<div class="value">{escape(value)}</div>'
                "</div>"
            )
        month_cards.append(
            '<div class="mom-month-card">'
            f'<div class="mom-month-title">{escape(month_name)}</div>'
            f'<div class="mom-badge-row">{"".join(badges)}</div>'
            "</div>"
        )

    return (
        '<section class="eda-section">'
        "<h3>EDA</h3>"
        '<div class="eda-kicker">Summary of month-over-month data trends up until COVID</div>'
        '<p class="eda-context">'
        "Average pre-2020 month-over-month changes are shown when the magnitude is "
        f"greater than {threshold:.0f}%. Transit metrics use teal for increases and amber for decreases; "
        "gas uses orange for rising cost pressure and teal for declines."
        "</p>"
        f'<div class="mom-callout-grid">{"".join(month_cards)}</div>'
        "</section>"
    )


def lagged_upt_yoy_correlations(max_lag: int = 12) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    needed_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in needed_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    yoy = (
        df[["date", *available_cols]]
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
    )

    label_map = {
        "cpi_all_items_sa": "CPI all items",
        "cpi_core_sa": "CPI core",
        "seattle_gas_price_avg": "Seattle gas price",
        "voms": "VOMS",
        "vrh": "VRH",
        "vrm": "VRM",
    }
    rows = []
    for predictor in [col for col in available_cols if col != "upt"]:
        for lag in range(max_lag + 1):
            aligned = pd.concat(
                {
                    "upt_yoy": yoy["upt"],
                    "predictor_yoy_lagged": yoy[predictor].shift(lag),
                },
                axis=1,
            ).dropna()
            if len(aligned) > 20:
                rows.append(
                    {
                        "predictor": predictor,
                        "series": label_map.get(predictor, predictor),
                        "lag_months": lag,
                        "correlation": float(aligned["upt_yoy"].corr(aligned["predictor_yoy_lagged"])),
                        "n": len(aligned),
                    }
                )
    return pd.DataFrame(rows)


def granger_predictive_screening(max_lag: int = 6) -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    try:
        import warnings

        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    yoy = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
    )
    label_map = {
        "vrm": "VRM",
        "vrh": "VRH",
        "voms": "VOMS",
        "seattle_gas_price_avg": "Gas price",
        "cpi_all_items_sa": "CPI all",
        "cpi_core_sa": "CPI core",
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for predictor in [col for col in available_cols if col != "upt"]:
            test_df = yoy[["upt", predictor]].dropna()
            if len(test_df) < 40:
                continue
            try:
                result = grangercausalitytests(test_df[["upt", predictor]], maxlag=max_lag, verbose=False)
                pvals = [float(result[lag][0]["ssr_ftest"][1]) for lag in range(1, max_lag + 1)]
                best_lag = int(np.argmin(pvals) + 1)
                rows.append(
                    {
                        "predictor": predictor,
                        "series": label_map.get(predictor, predictor),
                        "best_lag": best_lag,
                        "min_p_value": min(pvals),
                        "pvals_by_lag": ", ".join(f"{p:.3f}" for p in pvals),
                        "n": len(test_df),
                    }
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("min_p_value")


def covid_break_diagnostics() -> pd.DataFrame:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return pd.DataFrame()

    try:
        import statsmodels.api as sm
        from scipy import stats
    except ImportError:
        return pd.DataFrame()

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or "upt" not in available_cols:
        return pd.DataFrame()

    yoy = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .pct_change(periods=12, fill_method=None)
        .replace([float("inf"), float("-inf")], pd.NA)
        .rename(columns={col: f"{col}_yoy" for col in available_cols})
    )
    model_df = yoy.copy()
    model_df["target_date"] = model_df.index + pd.DateOffset(months=3)
    model_df["upt_yoy_target_h3"] = model_df["upt_yoy"].shift(-3)
    model_df["target_post_covid"] = (model_df["target_date"] >= pd.Timestamp("2020-03-01")).astype(int)
    month_dummies = pd.get_dummies(
        model_df["target_date"].dt.month,
        prefix="target_month",
        drop_first=True,
        dtype=float,
    )
    month_dummies.index = model_df.index
    regression_df = pd.concat([model_df, month_dummies], axis=1).dropna()

    regression_terms = [
        "upt_yoy",
        "vrm_yoy",
        "vrh_yoy",
        "voms_yoy",
        "seattle_gas_price_avg_yoy",
        "cpi_all_items_sa_yoy",
        "cpi_core_sa_yoy",
        "target_post_covid",
        *list(month_dummies.columns),
    ]
    regression_terms = [term for term in regression_terms if term in regression_df.columns]
    if len(regression_df) < len(regression_terms) * 3:
        return pd.DataFrame()

    break_date = pd.Timestamp("2020-03-01")

    def regression_ssr(input_df: pd.DataFrame) -> tuple[float, int, int]:
        local_y = input_df["upt_yoy_target_h3"]
        local_x = sm.add_constant(input_df[regression_terms])
        fit = sm.OLS(local_y, local_x).fit()
        return float((fit.resid**2).sum()), len(input_df), local_x.shape[1]

    pooled_ssr, _, pooled_k = regression_ssr(regression_df)
    pre_break_df = regression_df.loc[regression_df["target_date"] < break_date]
    post_break_df = regression_df.loc[regression_df["target_date"] >= break_date]
    if len(pre_break_df) <= pooled_k or len(post_break_df) <= pooled_k:
        return pd.DataFrame()

    pre_ssr, pre_n, _ = regression_ssr(pre_break_df)
    post_ssr, post_n, _ = regression_ssr(post_break_df)
    denominator_df = pre_n + post_n - 2 * pooled_k
    if denominator_df <= 0:
        return pd.DataFrame()

    chow_f = ((pooled_ssr - (pre_ssr + post_ssr)) / pooled_k) / (
        (pre_ssr + post_ssr) / denominator_df
    )
    chow_p = 1 - stats.f.cdf(chow_f, pooled_k, denominator_df)
    mean_test = stats.ttest_ind(
        pre_break_df["upt_yoy_target_h3"],
        post_break_df["upt_yoy_target_h3"],
        equal_var=False,
    )
    return pd.DataFrame(
        [
            {
                "test": "Coefficient stability break",
                "statistic": float(chow_f),
                "p_value": float(chow_p),
                "pre_n": int(pre_n),
                "post_n": int(post_n),
                "interpretation": "Tests whether the H3 regression relationship is stable before and after COVID.",
            },
            {
                "test": "Mean UPT YoY difference",
                "statistic": float(mean_test.statistic),
                "p_value": float(mean_test.pvalue),
                "pre_n": int(pre_n),
                "post_n": int(post_n),
                "interpretation": "Tests whether average target UPT YoY differs across periods.",
            },
        ]
    )


def trend_month_residualize_dashboard(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    valid = y.notna()
    if valid.sum() < 24:
        return pd.Series(index=series.index, data=pd.NA, name=series.name)

    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "time_index": range(len(series)),
        },
        index=series.index,
        dtype=float,
    )
    month_dummies = pd.get_dummies(series.index.month, prefix="month", drop_first=True, dtype=float)
    month_dummies.index = series.index
    design = pd.concat([design, month_dummies], axis=1)

    x = design.loc[valid].to_numpy(dtype=float)
    target = y.loc[valid].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    fitted = design.to_numpy(dtype=float) @ beta
    return pd.Series(y.to_numpy(dtype=float) - fitted, index=series.index, name=series.name)


def dashboard_correlation_matrices() -> dict[str, pd.DataFrame]:
    integrated_path = configured_integrated_base_path()
    if not integrated_path.exists():
        return {}

    df = load_integrated_base(str(integrated_path), file_modified_ns(integrated_path)).copy()
    signal_cols = [
        "upt",
        "vrm",
        "vrh",
        "voms",
        "seattle_gas_price_avg",
        "cpi_all_items_sa",
        "cpi_core_sa",
    ]
    available_cols = [col for col in signal_cols if col in df.columns]
    if "date" not in df.columns or len(available_cols) < 2:
        return {}

    label_map = {
        "upt": "UPT",
        "vrm": "VRM",
        "vrh": "VRH",
        "voms": "VOMS",
        "seattle_gas_price_avg": "Gas price",
        "cpi_all_items_sa": "CPI all",
        "cpi_core_sa": "CPI core",
    }
    data = (
        df[["date", *available_cols]]
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce").dt.to_period("M").dt.to_timestamp())
        .sort_values("date")
        .set_index("date")[available_cols]
        .rename(columns=label_map)
    )
    residuals = data.apply(trend_month_residualize_dashboard)
    transformed = {
        "First Differences": data.diff(),
        "YoY Percent Changes": data.pct_change(periods=12, fill_method=None).replace([float("inf"), float("-inf")], pd.NA),
        "Trend + Month Residuals": residuals,
    }
    return {name: matrix.corr(method="pearson") for name, matrix in transformed.items()}


def render_data_page(
    forecast_examples: pd.DataFrame,
    champion: dict,
) -> None:
    data_intro_cols = st.columns(2)
    data_intro_cols[0].markdown(summary_panel_from_markdown(DATA_PRIMARY_DATA), unsafe_allow_html=True)
    data_intro_cols[1].markdown(summary_panel_from_markdown(DATA_SECONDARY_DATA), unsafe_allow_html=True)

    source_series, source_options = integrated_source_series_data()
    if source_options:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Integrated monthly source series</h4>
                <p>
                    Before feature engineering, the pipeline joins each normalized source
                    to a common monthly grain. Use these tabs to inspect one raw integrated
                    signal at a time across the full available history.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        source_tabs = st.tabs([option["label"] for option in source_options])
        for tab_panel, option in zip(source_tabs, source_options):
            with tab_panel:
                st.plotly_chart(source_series_figure(source_series, option), width="stretch")
                start_date = source_series.loc[source_series[option["column"]].notna(), "date"].min()
                end_date = source_series.loc[source_series[option["column"]].notna(), "date"].max()
                if pd.notna(start_date) and pd.notna(end_date):
                    st.caption(
                        f"{option['source']} source signal, "
                        f"{start_date:%b %Y} through {end_date:%b %Y}."
                    )

    mom_callouts = pre_covid_mom_callouts()
    mom_callout_html = pre_covid_mom_callouts_html(mom_callouts)
    if mom_callout_html:
        st.markdown(mom_callout_html, unsafe_allow_html=True)

    lagged_corr = lagged_upt_yoy_correlations()
    if not lagged_corr.empty:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Lagged relationships after reducing shared trend</h4>
                <p>
                    Because UPT and price indexes both tend to rise over long periods,
                    raw level correlations can overstate the relationship. This view
                    compares year-over-year changes instead, then tests whether each
                    predictor's YoY movement leads UPT YoY movement by 0 to 12 months.
                    CPI shows the strongest short-lag association, but that should be
                    interpreted as a broad macro or regime signal rather than causal
                    evidence that inflation mechanically increases ridership.
                </p>
                <p>
                    Service measures are more mixed: they move with UPT at shorter lags,
                    then fade or turn negative at longer lags. That pattern is useful for
                    forecasting exploration, but still observational and potentially shaped
                    by service planning, recovery timing, and COVID-era structural change.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(lagged_correlation_figure(lagged_corr), width="stretch")

    corr_matrices = dashboard_correlation_matrices()
    if corr_matrices:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Correlation matrices after reducing time effects</h4>
                <p>
                    These Pearson correlation matrices compare transformed versions of
                    the integrated data rather than raw levels. First differences show
                    short-run movement, year-over-year percent changes compare growth
                    against the same month one year earlier, and trend + month residuals
                    show relationships after a simple time-trend and calendar-month
                    adjustment. This reduces the chance that shared upward trends in
                    series such as ridership and price indexes dominate the interpretation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        matrix_tabs = st.tabs(list(corr_matrices.keys()))
        for tab_panel, (title, matrix) in zip(matrix_tabs, corr_matrices.items()):
            with tab_panel:
                st.plotly_chart(correlation_heatmap_figure(matrix, title), width="stretch")

    granger_screening = granger_predictive_screening()
    if not granger_screening.empty:
        st.markdown(
            """
            <div class="eda-chart-panel">
                <h4>Granger-style predictive screening</h4>
                <p>
                    This screen asks whether past values of each YoY predictor improve
                    prediction of UPT YoY beyond past UPT alone. The chart shows the
                    strongest p-value found across one- to six-month lags for each
                    predictor, so it should be read as a ranking of candidate signals,
                    not as a formal causal result.
                </p>
                <p>
                    In this pass, VRH, VOMS, and CPI measures surface as the clearest
                    predictive candidates. Gas prices and VRM are weaker in this specific
                    YoY lag test, even though they can still matter in other
                    transformations or in the full rolling forecast models.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(granger_predictive_figure(granger_screening), width="stretch")

    break_summary = covid_break_diagnostics()
    if not break_summary.empty:
        break_rows = break_summary.set_index("test")
        stability_p = break_rows.loc["Coefficient stability break", "p_value"]
        mean_p = break_rows.loc["Mean UPT YoY difference", "p_value"]
        st.markdown(
            f"""
            <div class="eda-chart-panel">
                <h4>COVID-era structural break diagnostics</h4>
                <p>
                    The H3 diagnostic regression predicts UPT YoY three months ahead
                    from as-of-month YoY signals, target-month calendar effects, and a
                    post-COVID indicator. A coefficient-stability test then compares
                    whether that relationship looks the same before and after March 2020.
                </p>
                <p>
                    The coefficient-stability result is extremely strong
                    (<strong>p = {stability_p:.2g}</strong>), which supports treating
                    COVID as a structural break in the modeling pipeline. The simpler
                    pre/post mean comparison is not significant
                    (<strong>p = {mean_p:.3f}</strong>), so the useful takeaway is not
                    just that ridership visibly dropped, but that the relationship
                    between ridership, service, prices, and seasonality changed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    data_feature_cols = st.columns(2)
    data_feature_cols[0].markdown(summary_panel_from_markdown(DATA_CALCULATED_FEATURES), unsafe_allow_html=True)
    data_feature_cols[1].markdown(summary_panel_from_markdown(DATA_TIME_FEATURES), unsafe_allow_html=True)

    st.markdown("### Single Forecast Step Example")
    config_id = champion.get("model_config_id") or champion.get("config_id")
    id_col = "model_config_id" if "model_config_id" in forecast_examples.columns else "config_id"
    sample = forecast_examples[forecast_examples[id_col] == config_id].copy()
    if sample.empty:
        sample = forecast_examples.copy()
    if not sample.empty:
        sample = sample.sort_values("target_date").iloc[len(sample) // 2]
        target_date = pd.Timestamp(sample["target_date"])
        as_of_date = pd.Timestamp(sample["as_of_date"])
        pandemic_start = pd.Timestamp("2020-03-01")
        months_since_pandemic = max(
            0,
            (as_of_date.year - pandemic_start.year) * 12 + (as_of_date.month - pandemic_start.month),
        )
        example_rows = [
            ("as_of_date", as_of_date.date().isoformat(), "Training data is limited to rows before this month."),
            ("target_date", target_date.date().isoformat(), "This is the month being forecast three months ahead."),
            ("target_month", target_date.strftime("%B"), "Seasonality features encode this month cyclically."),
            ("evaluation_period", sample.get("evaluation_period", "-"), "Used for pre-COVID, shock, recovery, and recent metrics."),
            ("months_since_pandemic_observed", str(months_since_pandemic), "A time-since-observed-disruption signal available only from the as-of month."),
            ("actual_upt", format_int(sample.get("actual")), "Observed ridership for the target month."),
            ("prediction", format_int(sample.get("prediction")), "The selected model's forecast for that target month."),
            ("seasonal_naive_prediction", format_int(sample.get("seasonal_naive_prediction")), "Same-month-last-year baseline used for comparison and residual mode."),
            ("absolute_error", format_int(sample.get("abs_error")), "Distance between prediction and observed ridership."),
        ]
        st.dataframe(
            pd.DataFrame(example_rows, columns=["Field", "Example value", "Interpretation"]),
            width="stretch",
            hide_index=True,
        )

    render_data_availability_report()
