"""Create a lightweight dashboard artifact bundle for public deployment.

The full experiment export is useful locally, but it can be too large for a
simple Streamlit Community Cloud deployment. This script keeps configurations
that are strong on at least one important metric, then exports all associated
forecast/performance rows for those configurations.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("dashboard_artifacts/aws_streamlined/latest")
DEFAULT_OUTPUT_DIR = Path("dashboard/public_artifacts/latest")
DEFAULT_FEATURE_FAMILIES = Path("feature_store/income_interactions_h3_v1/feature_families.json")

REQUIRED_PARQUET = {
    "champion_predictions": "champion_predictions.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
}

OPTIONAL_PARQUET = {
    "complexity_profile": "complexity_profile.parquet",
}

JSON_FILES = [
    "batch_manifest.json",
    "champion_selection.json",
    "experiment_manifest.json",
    "mart_validation.json",
]

CORE_LOWER_IS_BETTER = [
    "selection_score",
    "selection_score_typical",
    "selection_score_balanced",
    "selection_score_large_error",
    "mae",
    "rmse",
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

CORE_HIGHER_IS_BETTER = [
    "r2",
    "r2_adjusted",
]

SECONDARY_LOWER_IS_BETTER = [
    "complexity_score",
    "compute_score",
]

SECONDARY_HIGHER_IS_BETTER = [
    "diracc",
    "pre_covid_diracc",
    "covid_shock_diracc",
    "recovery_diracc",
    "recent_diracc",
    "interpretability_score",
]

SCORE_RECIPES = {
    "typical": {"mae_weight": 0.90, "rmse_weight": 0.10},
    "balanced": {"mae_weight": 0.75, "rmse_weight": 0.25},
    "large_error": {"mae_weight": 0.50, "rmse_weight": 0.50},
}
EVALUATION_PERIODS = ["pre_covid", "covid_shock", "recovery", "recent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated public Streamlit artifact bundle.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-families", type=Path, default=DEFAULT_FEATURE_FAMILIES)
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.05,
        help="Keep configurations in the best fraction for any tracked metric.",
    )
    parser.add_argument(
        "--include-secondary-retention-metrics",
        action="store_true",
        help=(
            "Also use directional accuracy, complexity, compute, and interpretability "
            "metrics as retention gates. These columns are always kept when present, "
            "but are not part of the default public pruning rule."
        ),
    )
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always keep baseline models for comparison context.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_parquets(input_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name, filename in REQUIRED_PARQUET.items():
        path = input_dir / filename
        require_file(path)
        frames[name] = pd.read_parquet(path)
    for name, filename in OPTIONAL_PARQUET.items():
        path = input_dir / filename
        if path.exists():
            frames[name] = pd.read_parquet(path)
    return frames


def model_id_column(df: pd.DataFrame) -> str:
    if "model_config_id" in df.columns:
        return "model_config_id"
    if "config_id" in df.columns:
        return "config_id"
    raise KeyError("Expected either model_config_id or config_id.")


def weighted_error_score(mae: pd.Series, rmse: pd.Series, recipe: dict) -> pd.Series:
    return recipe["mae_weight"] * pd.to_numeric(mae, errors="coerce") + recipe["rmse_weight"] * pd.to_numeric(
        rmse,
        errors="coerce",
    )


def safe_ratio(numerator, denominator) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def enrich_score_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def configs_for_metric(df: pd.DataFrame, metric: str, keep_fraction: float, lower_is_better: bool) -> set[str]:
    if metric not in df.columns:
        return set()
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df[values.notna()].copy()
    if valid.empty:
        return set()
    values = pd.to_numeric(valid[metric], errors="coerce")
    quantile = keep_fraction if lower_is_better else 1 - keep_fraction
    threshold = values.quantile(quantile)
    if lower_is_better:
        keep = valid[values <= threshold]
    else:
        keep = valid[values >= threshold]
    return set(keep[model_id_column(keep)].astype(str))


def select_public_configs(
    leaderboard: pd.DataFrame,
    champion: dict,
    keep_fraction: float,
    include_baseline: bool,
    include_secondary_retention_metrics: bool,
) -> tuple[set[str], dict]:
    selected: set[str] = set()
    by_metric: dict[str, int] = {}

    lower_metrics = list(CORE_LOWER_IS_BETTER)
    higher_metrics = list(CORE_HIGHER_IS_BETTER)
    if include_secondary_retention_metrics:
        lower_metrics.extend(SECONDARY_LOWER_IS_BETTER)
        higher_metrics.extend(SECONDARY_HIGHER_IS_BETTER)

    for metric in lower_metrics:
        configs = configs_for_metric(leaderboard, metric, keep_fraction, lower_is_better=True)
        selected |= configs
        if configs:
            by_metric[metric] = len(configs)

    for metric in higher_metrics:
        configs = configs_for_metric(leaderboard, metric, keep_fraction, lower_is_better=False)
        selected |= configs
        if configs:
            by_metric[metric] = len(configs)

    if include_baseline and "model_family" in leaderboard.columns:
        baseline_ids = set(
            leaderboard.loc[leaderboard["model_family"].astype(str) == "baseline", model_id_column(leaderboard)]
            .astype(str)
            .tolist()
        )
        selected |= baseline_ids
        by_metric["always_baseline"] = len(baseline_ids)

    champion_id = champion.get("model_config_id") or champion.get("config_id")
    if champion_id:
        selected.add(str(champion_id))
        by_metric["always_champion"] = 1

    summary = {
        "keep_fraction": keep_fraction,
        "secondary_retention_metrics": include_secondary_retention_metrics,
        "selected_by_metric": by_metric,
        "selected_configurations": len(selected),
        "source_configurations": int(leaderboard[model_id_column(leaderboard)].nunique()),
    }
    return selected, summary


def filter_by_configs(df: pd.DataFrame, selected: set[str]) -> pd.DataFrame:
    id_col = model_id_column(df)
    return df[df[id_col].astype(str).isin(selected)].copy()


def recompute_feature_family_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty or not {"feature_family_name", "mode"}.issubset(leaderboard.columns):
        return pd.DataFrame()

    agg_spec = {}
    source_to_output = {
        "selection_score": ("best_selection_score", "min"),
        "selection_score_typical": ("best_selection_score_typical", "min"),
        "selection_score_balanced": ("best_selection_score_balanced", "min"),
        "selection_score_large_error": ("best_selection_score_large_error", "min"),
        "rmse": ("best_rmse", "min"),
        "mae": ("best_mae", "min"),
        "r2": ("best_r2", "max"),
        "r2_adjusted": ("best_r2_adjusted", "max"),
        "diracc": ("best_diracc", "max"),
        "rmse_improvement_vs_naive": ("best_rmse_improvement_vs_naive", "max"),
        "mae_improvement_vs_naive": ("best_mae_improvement_vs_naive", "max"),
    }
    for source, (output, func) in source_to_output.items():
        if source in leaderboard.columns:
            agg_spec[output] = (source, func)
    if "rmse" in leaderboard.columns:
        agg_spec["avg_rmse"] = ("rmse", "mean")
    if "mae" in leaderboard.columns:
        agg_spec["avg_mae"] = ("mae", "mean")

    summary = (
        leaderboard.groupby(["feature_family_name", "mode"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["best_selection_score", "feature_family_name"], na_position="last")
    )
    return summary


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.keep_fraction <= 1:
        raise ValueError("--keep-fraction must be greater than 0 and less than or equal to 1.")

    frames = read_parquets(args.input_dir)
    champion_path = args.input_dir / "champion_selection.json"
    require_file(champion_path)
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    frames["model_leaderboard"] = enrich_score_columns(frames["model_leaderboard"])

    selected, summary = select_public_configs(
        frames["model_leaderboard"],
        champion,
        keep_fraction=args.keep_fraction,
        include_baseline=args.include_baseline,
        include_secondary_retention_metrics=args.include_secondary_retention_metrics,
    )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = filter_by_configs(frames["model_leaderboard"], selected)
    forecast_paths = filter_by_configs(frames["forecast_paths"], selected)
    performance = filter_by_configs(frames["performance_over_time"], selected)
    champion_predictions = filter_by_configs(frames["champion_predictions"], selected)

    leaderboard.to_parquet(args.output_dir / "model_leaderboard.parquet", index=False)
    forecast_paths.to_parquet(args.output_dir / "forecast_paths.parquet", index=False)
    performance.to_parquet(args.output_dir / "performance_over_time.parquet", index=False)
    champion_predictions.to_parquet(args.output_dir / "champion_predictions.parquet", index=False)
    recompute_feature_family_summary(leaderboard).to_parquet(
        args.output_dir / "feature_family_summary.parquet",
        index=False,
    )

    if "complexity_profile" in frames:
        filter_by_configs(frames["complexity_profile"], selected).to_parquet(
            args.output_dir / "complexity_profile.parquet",
            index=False,
        )

    overview_top = leaderboard.sort_values("selection_score", ascending=True).head(5).copy()
    overview_top["rank"] = range(1, len(overview_top) + 1)
    overview_top.to_parquet(args.output_dir / "overview_top_models.parquet", index=False)

    overview_ids = set(overview_top[model_id_column(overview_top)].astype(str))
    overview_paths = forecast_paths[forecast_paths[model_id_column(forecast_paths)].astype(str).isin(overview_ids)].copy()
    rank_lookup = dict(zip(overview_top[model_id_column(overview_top)].astype(str), overview_top["rank"]))
    overview_paths["rank"] = overview_paths[model_id_column(overview_paths)].astype(str).map(rank_lookup)
    overview_paths.to_parquet(args.output_dir / "overview_prediction_paths.parquet", index=False)

    for filename in JSON_FILES:
        source = args.input_dir / filename
        if source.exists():
            payload = json.loads(source.read_text(encoding="utf-8"))
            if filename == "experiment_manifest.json":
                payload["public_dashboard_bundle"] = {
                    **summary,
                    "source_dir": str(args.input_dir),
                    "curation_rule": (
                        "Keep configurations in the best keep_fraction for core performance "
                        "metrics, plus baseline and champion configurations."
                    ),
                }
            shutil.copy2(source, args.output_dir / filename)
            if filename == "experiment_manifest.json":
                write_json(args.output_dir / filename, payload)

    if args.feature_families.exists():
        shutil.copy2(args.feature_families, args.output_dir / "feature_families.json")

    summary.update(
        {
            "output_dir": str(args.output_dir),
            "leaderboard_rows": int(len(leaderboard)),
            "forecast_rows": int(len(forecast_paths)),
            "performance_rows": int(len(performance)),
        }
    )
    write_json(args.output_dir / "public_bundle_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
