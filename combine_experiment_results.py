"""Combine compatible experiment result folders into one dashboard-ready run.

Use this after separate experiment blocks finish, for example Phase A linear/tree
results plus Phase B autoregressive results. The script concatenates the common
Parquet artifacts, recomputes the combined champion/dashboard views, and writes a
single result folder that can be loaded into DuckDB or pointed at by Streamlit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_aws_streamlined_models import (
    build_dashboard_outputs,
    build_complexity_profile,
    build_family_summary,
    calculate_metrics,
    select_champion,
)


CORE_TABLES = ["predictions", "model_runs", "feature_importance", "feature_sets"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge experiment result artifacts.")
    parser.add_argument(
        "--results-dir",
        action="append",
        required=True,
        help="Input result directory. Repeat once per experiment block.",
    )
    parser.add_argument("--output-results-dir", required=True, help="Combined result output directory.")
    parser.add_argument("--output-dashboard-dir", required=True, help="Combined dashboard artifact output directory.")
    parser.add_argument("--experiment-id", default="combined_experiment", help="Combined manifest experiment ID.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def read_table(results_dirs: list[Path], table_name: str) -> pd.DataFrame:
    frames = []
    filename = f"{table_name}.parquet"
    for folder in results_dirs:
        path = folder / filename
        if path.exists():
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_manifest(experiment_id: str, results_dirs: list[Path], predictions: pd.DataFrame, model_runs: pd.DataFrame, metrics: pd.DataFrame, champion: dict) -> dict:
    source_manifests = [read_json(folder / "experiment_manifest.json") for folder in results_dirs]
    return {
        "run_id": experiment_id,
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_dirs": [str(folder) for folder in results_dirs],
        "source_experiments": [
            manifest.get("experiment_id") or manifest.get("run_id")
            for manifest in source_manifests
            if manifest
        ],
        "target": source_manifests[0].get("target") if source_manifests else None,
        "horizon": source_manifests[0].get("horizon") if source_manifests else None,
        "as_of_start": min(
            [manifest.get("as_of_start") for manifest in source_manifests if manifest.get("as_of_start")],
            default=None,
        ),
        "as_of_end": max(
            [manifest.get("as_of_end") for manifest in source_manifests if manifest.get("as_of_end")],
            default=None,
        ),
        "as_of_frequency_months": source_manifests[0].get("as_of_frequency_months") if source_manifests else None,
        "models": sorted(metrics["model_type"].dropna().unique().tolist()) if not metrics.empty else [],
        "modes": sorted(metrics["mode"].dropna().unique().tolist()) if not metrics.empty else [],
        "feature_policies": sorted(metrics["feature_policy"].dropna().unique().tolist()) if "feature_policy" in metrics else [],
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "metric_count": int(len(metrics)),
        "champion_config_id": champion.get("config_id"),
        "selection_rule": champion.get("selection_rule"),
        "runtime": {"compute_context": "local_combined"},
    }


def main() -> None:
    args = parse_args()
    results_dirs = [Path(path) for path in args.results_dir]
    output_results_dir = Path(args.output_results_dir)
    output_dashboard_dir = Path(args.output_dashboard_dir)

    predictions = read_table(results_dirs, "predictions")
    model_runs = read_table(results_dirs, "model_runs")
    feature_importance = read_table(results_dirs, "feature_importance")
    feature_sets = read_table(results_dirs, "feature_sets")

    if predictions.empty:
        raise ValueError("No predictions were found in the supplied result directories.")
    if model_runs.empty:
        raise ValueError("No model_runs were found in the supplied result directories.")

    metrics = calculate_metrics(predictions)
    family_summary = build_family_summary(metrics)
    champion = select_champion(metrics)
    complexity_profile = build_complexity_profile(model_runs, metrics)
    dashboard_outputs = build_dashboard_outputs(
        predictions,
        model_runs,
        metrics,
        family_summary,
        champion,
        complexity_profile=complexity_profile,
    )
    manifest = build_manifest(args.experiment_id, results_dirs, predictions, model_runs, metrics, champion)

    write_table(output_results_dir / "predictions.parquet", predictions)
    write_table(output_results_dir / "model_runs.parquet", model_runs)
    write_table(output_results_dir / "metrics.parquet", metrics)
    write_table(output_results_dir / "feature_importance.parquet", feature_importance)
    write_table(output_results_dir / "feature_sets.parquet", feature_sets)
    write_table(output_results_dir / "feature_family_summary.parquet", family_summary)
    write_table(output_results_dir / "complexity_profile.parquet", complexity_profile)
    write_json(output_results_dir / "champion_selection.json", champion)
    write_json(output_results_dir / "batch_manifest.json", manifest)
    write_json(output_results_dir / "experiment_manifest.json", manifest)

    for filename, df in dashboard_outputs.items():
        write_table(output_dashboard_dir / filename, df)
    write_json(output_dashboard_dir / "champion_selection.json", champion)
    write_json(output_dashboard_dir / "experiment_manifest.json", manifest)

    print(
        json.dumps(
            {
                "output_results_dir": str(output_results_dir),
                "output_dashboard_dir": str(output_dashboard_dir),
                "prediction_count": len(predictions),
                "model_run_count": len(model_runs),
                "metric_count": len(metrics),
                "champion_config_id": champion.get("config_id"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
