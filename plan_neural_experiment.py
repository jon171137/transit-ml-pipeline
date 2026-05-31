"""Summarize a Phase C neural experiment grid before spending GPU time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Phase C neural experiment YAML config.")
    parser.add_argument("--config", default="experiment_configs/phase_c_neural_screening.yaml")
    parser.add_argument("--feature-families-uri", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def month_count(start: str, end: str, frequency_months: int) -> int:
    dates = pd.date_range(start=start, end=end, freq="MS")
    return int(len(dates[::frequency_months]))


def build_summary(config: dict, feature_families_path: Path | None) -> dict:
    forecast = config.get("forecast") or {}
    families = (config.get("feature_families") or {}).get("include") or []
    modes = config.get("modes") or ["raw"]
    feature_policies = config.get("feature_policies") or ["none"]
    representation_policies = config.get("representation_policies") or ["sequence_raw"]
    as_of_count = month_count(
        forecast["as_of_start"],
        forecast["as_of_end"],
        int(forecast.get("as_of_frequency_months", 1)),
    )
    model_rows = []
    for model_build, details in (config.get("models") or {}).items():
        if not details.get("enabled", False):
            continue
        param_count = len(details.get("param_grid") or [{}])
        model_rows.append(
            {
                "model_family": "neural_net",
                "model_build": model_build,
                "param_count": param_count,
                "feature_family_count": len(families),
                "mode_count": len(modes),
                "feature_policy_count": len(feature_policies),
                "representation_policy_count": len(representation_policies),
                "model_config_count": (
                    param_count
                    * len(families)
                    * len(modes)
                    * len(feature_policies)
                    * len(representation_policies)
                ),
            }
        )
    available = set(json.loads(feature_families_path.read_text())) if feature_families_path else set()
    total_configs = sum(row["model_config_count"] for row in model_rows)
    return {
        "experiment_id": config.get("experiment_id"),
        "status": config.get("status"),
        "forecast": forecast,
        "as_of_count": as_of_count,
        "model_builds": model_rows,
        "feature_families": families,
        "modes": modes,
        "feature_policies": feature_policies,
        "representation_policies": representation_policies,
        "feature_family_validation": {
            "validated": bool(feature_families_path),
            "missing": sorted(set(families) - available),
        },
        "total_model_configurations": total_configs,
        "estimated_model_fits": total_configs * as_of_count,
        "checkpointing": (config.get("execution") or {}).get("checkpointing", {}),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text()) or {}
    families_uri = args.feature_families_uri or (config.get("inputs") or {}).get("feature_families_uri")
    families_path = Path(families_uri) if families_uri and not str(families_uri).startswith("s3://") else None
    summary = build_summary(config, families_path)
    rendered = json.dumps(summary, indent=2, default=str)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
