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


def policy_representation_variants(config: dict) -> list[dict]:
    configured = config.get("policy_representation_variants") or []
    if configured:
        return configured
    feature_policies = config.get("feature_policies") or ["none"]
    representation_policies = config.get("representation_policies") or ["sequence_raw"]
    return [
        {
            "feature_policy": feature_policy,
            "representation_policy": representation_policy,
        }
        for feature_policy in feature_policies
        for representation_policy in representation_policies
    ]


def variant_applies(variant: dict, n_family_features: int) -> bool:
    if n_family_features < int(variant.get("min_family_features", 0)):
        return False
    max_family_features = variant.get("max_family_features")
    return max_family_features is None or n_family_features <= int(max_family_features)


def build_summary(config: dict, feature_families_path: Path | None) -> dict:
    forecast = config.get("forecast") or {}
    families = (config.get("feature_families") or {}).get("include") or []
    modes = config.get("modes") or ["raw"]
    variants = policy_representation_variants(config)
    as_of_count = month_count(
        forecast["as_of_start"],
        forecast["as_of_end"],
        int(forecast.get("as_of_frequency_months", 1)),
    )
    available_families = json.loads(feature_families_path.read_text()) if feature_families_path else {}
    applicable_variant_total = sum(
        sum(variant_applies(variant, len(available_families.get(family, []))) for variant in variants)
        for family in families
    )
    if not feature_families_path:
        applicable_variant_total = len(families) * len(variants)
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
                "policy_representation_variant_count": len(variants),
                "applicable_family_variant_count": applicable_variant_total,
                "model_config_count": (
                    param_count
                    * len(modes)
                    * applicable_variant_total
                ),
            }
        )
    available = set(available_families)
    total_configs = sum(row["model_config_count"] for row in model_rows)
    return {
        "experiment_id": config.get("experiment_id"),
        "status": config.get("status"),
        "forecast": forecast,
        "as_of_count": as_of_count,
        "model_builds": model_rows,
        "feature_families": families,
        "modes": modes,
        "policy_representation_variants": variants,
        "feature_family_validation": {
            "validated": bool(feature_families_path),
            "missing": sorted(set(families) - available),
        },
        "total_model_configurations": total_configs,
        "estimated_model_fits": total_configs * as_of_count,
        "estimate_note": (
            "Upper bound after configured family-width applicability rules. "
            "Runtime dynamic policy deduplication may skip equivalent rolling selected-feature branches."
        ),
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
