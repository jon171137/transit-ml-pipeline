import json
from html import escape
from textwrap import dedent

import pandas as pd
import streamlit as st

from charts import feature_family_count_figure
from constants import (
    DEFAULT_FEATURE_FAMILIES_PATH,
    FEATURE_POLICY_DESCRIPTIONS,
    FEATURE_TRANSFORM_DESCRIPTIONS,
    MODEL_BUILD_ORDER,
    MODEL_FAMILY_ORDER,
    PHASE_A_V3_CONFIG_PATH,
    PHASE_B_V3_CONFIG_PATH,
    PHASE_C_MONTHLY_CONFIG_PATH,
)
from content import (
    DATA_AS_OF_REGIME_FEATURES,
    EXPERIMENT_OVERVIEW,
    REPRESENTATION_AND_COMPLEXITY_EXPLANATION,
)
from data_access import configured_feature_table_path, file_modified_ns, load_feature_table
from formatting import date_range_label, format_int, manifest_value
from sections.overview import render_public_bundle_note
from ui_components import champion_summary_item, model_scope_summary_html


def order_index(value, ordered_values: list[str]) -> tuple[int, str]:
    text = str(value)
    try:
        return ordered_values.index(text), text
    except ValueError:
        return len(ordered_values), text


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

def experiment_overview_with_regime_note() -> str:
    marker = "### Current Experiment Blocks"
    if marker not in EXPERIMENT_OVERVIEW:
        return EXPERIMENT_OVERVIEW + "\n\n" + DATA_AS_OF_REGIME_FEATURES
    return EXPERIMENT_OVERVIEW.replace(marker, DATA_AS_OF_REGIME_FEATURES + "\n\n" + marker, 1)


def feature_family_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def feature_family_count_frame(feature_families: dict) -> pd.DataFrame:
    if not feature_families:
        return pd.DataFrame()

    feature_path = configured_feature_table_path()
    available_columns = None
    if feature_path.exists():
        feature_table = load_feature_table(str(feature_path), file_modified_ns(feature_path))
        available_columns = set(feature_table.columns)

    rows = []
    for family_name, features in feature_families.items():
        feature_list = [str(feature) for feature in features]
        requested = len(feature_list)
        if available_columns is None:
            available = requested
        else:
            available = sum(feature in available_columns for feature in feature_list)
        rows.append(
            {
                "feature_family_name": family_name,
                "Feature family": feature_family_label(family_name),
                "Requested features": requested,
                "Available features": available,
                "Missing features": max(requested - available, 0),
            }
        )
    return pd.DataFrame(rows).sort_values("Available features", ascending=False)


FEATURE_GROUP_ORDER = [
    "Ridership History",
    "Service History",
    "Seasonality And Time",
    "Regime Signals",
    "Economic Context",
    "Interactions",
    "Other",
]


def feature_group_label(feature_name: str) -> str:
    name = str(feature_name)
    lower_name = name.lower()
    if "_x_" in lower_name:
        return "Interactions"
    if lower_name.startswith("upt_"):
        return "Ridership History"
    if lower_name.startswith(("vrm", "vrh", "voms")):
        return "Service History"
    if (
        lower_name.startswith(("month_", "target_month_", "time_index"))
        or "months_since" in lower_name
    ):
        return "Seasonality And Time"
    if "pandemic" in lower_name or "covid" in lower_name or "regime" in lower_name:
        return "Regime Signals"
    if any(token in lower_name for token in ["gas", "cpi", "income", "inflation"]):
        return "Economic Context"
    return "Other"


def grouped_feature_names(feature_names: list[str]) -> dict[str, list[str]]:
    groups = {group: [] for group in FEATURE_GROUP_ORDER}
    for feature_name in feature_names:
        groups.setdefault(feature_group_label(feature_name), []).append(feature_name)
    return {group: features for group, features in groups.items() if features}


def feature_chip_grid_html(feature_names: list[str]) -> str:
    groups = grouped_feature_names(feature_names)
    group_blocks = []
    for group_name, group_features in groups.items():
        chips = "\n".join(
            f'<span class="feature-definition-chip" title="{escape(feature_name)}">'
            f"{escape(feature_name)}</span>"
            for feature_name in group_features
        )
        feature_word = "feature" if len(group_features) == 1 else "features"
        group_blocks.append(
            dedent(f"""
            <section class="feature-definition-group">
                <div class="feature-definition-group-title">
                    <span>{escape(group_name)}</span>
                    <span>{len(group_features)} {feature_word}</span>
                </div>
                <div class="feature-definition-chip-grid">
                    {chips}
                </div>
            </section>
            """).strip()
        )
    return (
        dedent("""
        <style>
        .feature-definition-panel {
            border-top: 3px solid rgba(0, 127, 104, 0.28);
            background: #f8fbfa;
            padding: 0.95rem 1rem 1.05rem;
            margin: 0.45rem 0 1rem;
        }

        .feature-definition-group {
            margin: 0 0 0.95rem;
        }

        .feature-definition-group:last-child {
            margin-bottom: 0;
        }

        .feature-definition-group-title {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.75rem;
            color: #2f323a;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.42rem;
        }

        .feature-definition-group-title span:last-child {
            color: #6b7280;
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0;
            text-transform: none;
            white-space: nowrap;
        }

        .feature-definition-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem 0.48rem;
            align-items: flex-start;
        }

        .feature-definition-chip {
            display: inline-flex;
            max-width: min(100%, 32rem);
            border: 1px solid rgba(0, 127, 104, 0.13);
            border-radius: 0.42rem;
            background: #f1f5f4;
            color: #04783e;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            font-size: 0.84rem;
            line-height: 1.28;
            padding: 0.22rem 0.48rem;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        @media (max-width: 760px) {
            .feature-definition-panel {
                padding: 0.85rem 0.75rem;
            }

            .feature-definition-chip {
                font-size: 0.78rem;
                max-width: 100%;
            }
        }
        </style>
        <div class="feature-definition-panel">
        """).strip()
        + "\n".join(group_blocks)
        + "\n</div>"
    )


def render_feature_types_section() -> None:
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
            "Examples": "pandemic_observed, pandemic_disruption_active, months_since_pandemic_observed",
            "Purpose": "Let models distinguish ordinary history from known disruption and recovery periods.",
        },
        {
            "Feature type": "Exogenous context",
            "Examples": "gas price, CPI, core CPI, income growth",
            "Purpose": "Test whether external economic pressure improves forecasts.",
        },
        {
            "Feature type": "Targeted interactions",
            "Examples": "income_yoy_pct_x_gas_price_yoy_diff, lag_x_pandemic flags",
            "Purpose": "Let linear models express selected non-additive relationships without a full polynomial explosion.",
        },
    ]
    st.dataframe(pd.DataFrame(feature_type_rows), width="stretch", hide_index=True)


def render_feature_family_sections(family_summary: pd.DataFrame, feature_families: dict) -> None:
    st.markdown("### Feature Family Examples")
    if {"feature_family_name", "best_selection_score"}.issubset(family_summary.columns):
        family_display = family_summary.copy()
        score_column = (
            "best_selection_score_balanced"
            if "best_selection_score_balanced" in family_display.columns
            else "best_selection_score"
        )
        family_display = family_display.sort_values(score_column).head(20)
        family_columns = [
            "feature_family_name",
            "mode",
            "best_selection_score_balanced",
            "best_rmse",
            "best_mae",
            "best_r2",
            "best_diracc",
        ]
        family_columns = [column for column in family_columns if column in family_display.columns]
        curated_examples = [
            ("history_regime_time", "raw"),
            ("history_regime_time_linear_interactions", "residual"),
            ("history_regime_income", None),
            ("history_regime_cpi", None),
        ]
        curated_rows = []
        for family_name, mode in curated_examples:
            row_candidates = family_display[family_display["feature_family_name"] == family_name]
            if mode is not None and "mode" in row_candidates:
                row_candidates = row_candidates[row_candidates["mode"].astype(str) == mode]
            if not row_candidates.empty:
                curated_rows.append(row_candidates.iloc[[0]])
        if curated_rows:
            family_display = pd.concat(curated_rows, ignore_index=True)
        family_display = family_display[family_columns]
        st.dataframe(family_display, width="stretch", hide_index=True)
    else:
        family_cols = [col for col in ["feature_family_name", "mode"] if col in family_summary]
        st.dataframe(family_summary[family_cols].drop_duplicates(), width="stretch", hide_index=True)

    st.markdown("### Feature Count By Family")
    st.write(
        "Feature families are intentionally uneven: some are compact baselines, "
        "while others include rolling history, regime context, external economic "
        "signals, or targeted interaction terms. The count below shows how much "
        "candidate signal each named family brings into the model-selection stage."
    )
    count_frame = feature_family_count_frame(feature_families)
    if count_frame.empty:
        st.info("Feature family definitions were not found, so the count chart cannot be rendered.")
    else:
        st.plotly_chart(feature_family_count_figure(count_frame), width="stretch")

    st.markdown("### Feature Family Definitions")
    st.write(
        "These are the named feature families available to the Phase A tabular "
        "models. A feature family is the human-readable modeling strategy before "
        "any model-specific feature policy such as correlation pruning, mutual "
        "information selection, or tree-importance selection is applied. Use the "
        "inspector below to see what a selected family includes."
    )
    if feature_families:
        selected_family = st.selectbox(
            "Inspect one feature family",
            sorted(feature_families),
            index=0,
            key="experiment_feature_family_definition_select",
        )
        selected_features = [str(feature) for feature in feature_families[selected_family]]
        feature_count = len(selected_features)
        feature_word = "feature" if feature_count == 1 else "features"
        st.caption(f"{feature_count} {feature_word} included")
        st.markdown(feature_chip_grid_html(selected_features), unsafe_allow_html=True)

        with st.expander("Show the feature family definition JSON"):
            st.code(json.dumps(feature_families, indent=2), language="json")
    else:
        st.info(
            "Feature family definitions were not found in this environment. "
            "For local runs, the dashboard looks for "
            f"`{DEFAULT_FEATURE_FAMILIES_PATH}` or the `FEATURE_FAMILIES_PATH` environment variable."
        )

def render_experiment_page(
    family_summary: pd.DataFrame,
    feature_families: dict,
    leaderboard: pd.DataFrame,
    forecast_paths: pd.DataFrame,
    performance: pd.DataFrame,
    champion: dict,
    experiment_manifest: dict,
    phase_a_config_text: str,
    phase_b_config_text: str,
    phase_c_config_text: str,
) -> None:
    st.subheader("Experiment")
    render_public_bundle_note(experiment_manifest)

    summary_items = [
        ("Experiment bundle", experiment_manifest.get("experiment_id") or experiment_manifest.get("run_id") or "-"),
        (
            "Target / horizon",
            (
                f"{str(champion.get('target', experiment_manifest.get('target', 'upt'))).upper()}"
                f" / {manifest_value(experiment_manifest, 'horizon', champion.get('horizon', 3))} months"
            ),
        ),
        ("As-of window", date_range_label(forecast_paths, "as_of_date")),
        ("Target window", date_range_label(forecast_paths, "target_date")),
        ("Model configurations", format_int(len(leaderboard))),
        ("Rolling predictions", format_int(len(forecast_paths))),
        ("Metric rows", format_int(len(performance))),
    ]
    model_scope = (
        leaderboard.groupby(["model_family", "model_build"], dropna=False)
        .size()
        .reset_index(name="configurations")
    )
    model_scope = model_taxonomy_sort(model_scope)

    summary_cols = st.columns(2)
    summary_cols[0].markdown(
        """
        <div class="champion-summary">
            <div class="summary-title">Loaded Experiment Summary</div>
            <div class="champion-summary-grid">
        """
        + "".join(champion_summary_item(label, value) for label, value in summary_items)
        + """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_cols[1].markdown(
        """
        <div class="champion-summary">
            <div class="summary-title">Model Scope In This Bundle</div>
        """
        + model_scope_summary_html(model_scope)
        + """
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_feature_types_section()
    render_feature_family_sections(family_summary, feature_families)

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
    if "feature_transform" in leaderboard.columns:
        st.markdown("### Feature Transforms")
        st.write(
            "Feature transforms describe the mathematical representation used after "
            "a feature family and feature policy are chosen. They are kept separate "
            "from feature families: the family answers what source signals are used, "
            "while the transform answers how those selected signals are represented "
            "for the model."
        )
        transform_scope = (
            leaderboard.groupby(["feature_transform", "feature_transform_label"], dropna=False)
            .size()
            .reset_index(name="configurations")
            .sort_values(["feature_transform_label"])
        )
        transform_scope["meaning"] = transform_scope["feature_transform"].map(
            lambda transform: FEATURE_TRANSFORM_DESCRIPTIONS.get(
                str(transform),
                "Mathematical representation applied to the selected feature columns.",
            )
        )
        transform_scope = transform_scope.rename(
            columns={
                "feature_transform_label": "Feature transform",
                "configurations": "Configurations",
                "meaning": "Meaning",
            }
        )[["Feature transform", "Configurations", "Meaning"]]
        st.table(transform_scope)
        st.markdown("#### How The Transform Families Are Implemented")
        st.write(
            "In this experiment bundle, transforms are broad representation tests. "
            "A feature family first defines candidate source signals, the feature "
            "policy selects or prunes columns inside each rolling training window, "
            "and the transform family then expands those selected numeric columns. "
            "Regularized models are scaled before fitting, so ridge, lasso, and "
            "elastic net can shrink weak transformed terms, but the transform choice "
            "itself is still applied consistently across the selected variables."
        )
        transform_detail_rows = [
            {
                "Transform family": "No transform",
                "Terms created for each selected feature x": "x",
                "Interpretation": "Baseline linear representation.",
            },
            {
                "Transform family": "Signed log",
                "Terms created for each selected feature x": "x; sign(x) * log1p(abs(x))",
                "Interpretation": "Tests compressed scale effects while preserving negative values.",
            },
            {
                "Transform family": "Quadratic",
                "Terms created for each selected feature x": "x; x^2",
                "Interpretation": "Tests curvature such as diminishing or accelerating effects.",
            },
            {
                "Transform family": "Cubic",
                "Terms created for each selected feature x": "x; x^2; x^3",
                "Interpretation": "Tests asymmetric curvature; the quadratic term is included with the cubic term.",
            },
            {
                "Transform family": "Signed log + quadratic + cubic",
                "Terms created for each selected feature x": "x; sign(x) * log1p(abs(x)); x^2; x^3",
                "Interpretation": "Stress-test representation that gives regularization several nonlinear shapes to choose from.",
            },
        ]
        st.dataframe(pd.DataFrame(transform_detail_rows), width="stretch", hide_index=True)
        st.info(
            "The cubic transform is hierarchical in this run: a selected feature gets "
            "the original value, the quadratic term, and the cubic term. It is not a "
            "cubic-only model."
        )
        st.markdown("#### Modeling Implications")
        st.write(
            "These runs are useful for asking whether nonlinear representations are "
            "worth exploring, but they are not yet variable-specific transform recipes. "
            "For example, they do not test one model that uses a log transform for gas "
            "prices, a quadratic term for service hours, and only the original scale "
            "for CPI unless that combination happens to appear through a broader "
            "all-selected-variable transform family and regularization."
        )
        st.write(
            "A stronger next iteration would inspect transformed-variable correlations "
            "and collinearity before modeling, then create source-aware transform "
            "recipes. That would let the experiment compare targeted combinations "
            "such as log gas-price movement, quadratic service capacity, and linear "
            "inflation context against the broader all-variable transform grids."
        )
        strategy_rows = [
            {
                "Next-step idea": "Screen transformed candidates",
                "Why it helps": "Compare correlations, VIF/collinearity, and rolling stability before expanding the model grid.",
            },
            {
                "Next-step idea": "Use variable-specific transform recipes",
                "Why it helps": "Allows log for one source signal and quadratic or cubic terms for another instead of applying every transform everywhere.",
            },
            {
                "Next-step idea": "Keep hierarchy constraints",
                "Why it helps": "When x^2 or x^3 is used, keep the original x term so nonlinear coefficients remain interpretable.",
            },
            {
                "Next-step idea": "Compare targeted recipes to broad regularized grids",
                "Why it helps": "Shows whether domain-guided transforms improve forecasts beyond relying on regularization to clean up a large expansion.",
            },
        ]
        st.dataframe(pd.DataFrame(strategy_rows), width="stretch", hide_index=True)
    with st.expander("How feature, representation, and complexity policies are interpreted"):
        st.markdown(REPRESENTATION_AND_COMPLEXITY_EXPLANATION)

    st.markdown("### Experiment Configs Used")
    st.write(
        "These YAML files define the model grids, feature families, feature "
        "policies, rolling forecast window, parallelization settings, optional "
        "MLflow summary-logging settings, checkpoint locations, and output "
        "artifact folders used for the current dashboard bundle and the next "
        "neural follow-up."
    )
    config_rows = [
        {
            "Phase": "A v3 pandemic-safe",
            "Config file": str(PHASE_A_V3_CONFIG_PATH),
            "Role": "Baseline, regularized linear, tree, and XGBoost grids over the rebuilt pandemic-safe feature table.",
            "Loaded": "yes" if phase_a_config_text else "missing",
        },
        {
            "Phase": "B v3 pandemic-safe",
            "Config file": str(PHASE_B_V3_CONFIG_PATH),
            "Role": "ARIMA, SARIMA, and SARIMAX rerun on the same as-of-safe feature table with compact exogenous sets.",
            "Loaded": "yes" if phase_b_config_text else "missing",
        },
        {
            "Phase": "C monthly finalists",
            "Config file": str(PHASE_C_MONTHLY_CONFIG_PATH),
            "Role": "Prior GRU/LSTM finalist config; neural runs are the next block to reconcile with the v3 pandemic-safe feature table.",
            "Loaded": "yes" if phase_c_config_text else "missing",
        },
    ]
    st.dataframe(pd.DataFrame(config_rows), width="stretch", hide_index=True)

    if phase_a_config_text:
        with st.expander("Show Phase A v3 pandemic-safe config YAML"):
            st.code(phase_a_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_A_V3_CONFIG_PATH}`.")

    if phase_b_config_text:
        with st.expander("Show Phase B v3 pandemic-safe config YAML"):
            st.code(phase_b_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_B_V3_CONFIG_PATH}`.")

    if phase_c_config_text:
        with st.expander("Show Phase C monthly finalists config YAML"):
            st.code(phase_c_config_text, language="yaml")
    else:
        st.warning(f"Could not find `{PHASE_C_MONTHLY_CONFIG_PATH}`.")

    st.markdown(experiment_overview_with_regime_note())
