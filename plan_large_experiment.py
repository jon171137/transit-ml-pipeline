"""Inspect a large experiment config before launching expensive model runs.

The goal of this script is intentionally modest: expand a YAML experiment
definition into counts and a reviewable task estimate. It does not train models.
Use it before a broad local sweep to catch accidental grid explosions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a large experiment YAML config.")
    parser.add_argument(
        "--config",
        default="experiment_configs/large_phase_a_v1.yaml",
        help="Path to a large experiment YAML config.",
    )
    parser.add_argument(
        "--feature-families-uri",
        default=None,
        help="Optional feature_families.json override used to validate configured families.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the planning summary JSON.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def month_count(start: str, end: str, frequency_months: int) -> int:
    if frequency_months < 1:
        raise ValueError("frequency_months must be at least 1")
    dates = pd.date_range(start=start, end=end, freq="MS")
    return int(len(dates[::frequency_months]))


def enabled_models(config: dict) -> list[dict]:
    rows = []
    autoregressive_names = {"arima", "sarima", "sarimax"}

    def valid_trends_for_order(order: list[int], trends: list[str]) -> int:
        d = int(order[1]) if len(order) > 1 else 0
        return sum(1 for trend in trends if not (trend in {"c", "t", "ct"} and d > 0))

    for model_family, builds in (config.get("models") or {}).items():
        if model_family in autoregressive_names:
            details = builds or {}
            if not details.get("enabled", False):
                continue
            orders = details.get("orders") or [[]]
            trends = details.get("trends") or ["n"]
            order_trend_count = sum(valid_trends_for_order(order, trends) for order in orders)
            seasonal_count = len(details.get("seasonal_orders") or [[0, 0, 0, 0]])
            exog_count = len(details.get("exog_sets") or {"univariate": []})
            rows.append(
                {
                    "model_family": "autoregressive",
                    "model_build": model_family,
                    "implementation_status": details.get("implementation_status", "implemented"),
                    "param_count": order_trend_count * seasonal_count * exog_count,
                }
            )
            continue
        for model_build, details in (builds or {}).items():
            if not details.get("enabled", False):
                continue
            param_grid = details.get("param_grid") or [{}]
            rows.append(
                {
                    "model_family": model_family,
                    "model_build": model_build,
                    "implementation_status": details.get("implementation_status", "unknown"),
                    "param_count": len(param_grid),
                }
            )
    return rows


def count_model_configurations(config: dict) -> list[dict]:
    feature_families = (config.get("feature_families") or {}).get("include") or []
    modes_by_family = config.get("modes") or {}
    policies_by_family = config.get("feature_policies") or {}
    rows = []

    for model in enabled_models(config):
        model_family = model["model_family"]
        model_build = model["model_build"]
        param_count = model["param_count"]
        status = model["implementation_status"]

        if model_family == "baseline":
            # Seasonal naive is intentionally emitted once.
            rows.append(
                {
                    **model,
                    "feature_family_count": 1,
                    "mode_count": 1,
                    "feature_policy_count": 1,
                    "model_config_count": param_count,
                }
            )
            continue

        if model_family == "autoregressive":
            rows.append(
                {
                    **model,
                    "feature_family_count": 1,
                    "mode_count": 1,
                    "feature_policy_count": 1,
                    "model_config_count": param_count,
                }
            )
            continue

        mode_count = len(modes_by_family.get(model_family, ["raw"]))
        policy_count = len(policies_by_family.get(model_family, ["none"]))
        config_count = len(feature_families) * param_count * mode_count * policy_count
        rows.append(
            {
                **model,
                "feature_family_count": len(feature_families),
                "mode_count": mode_count,
                "feature_policy_count": policy_count,
                "model_config_count": config_count,
            }
        )
    return rows


def validate_feature_families(config: dict, feature_families_path: Path | None) -> dict:
    requested = set((config.get("feature_families") or {}).get("include") or [])
    if not feature_families_path:
        return {
            "validated": False,
            "requested_count": len(requested),
            "missing": [],
            "extra_available": [],
        }
    available = set(load_json(feature_families_path))
    return {
        "validated": True,
        "requested_count": len(requested),
        "available_count": len(available),
        "missing": sorted(requested - available),
        "extra_available": sorted(available - requested),
    }


def build_summary(config: dict, feature_families_path: Path | None) -> dict:
    forecast = config.get("forecast") or {}
    as_of_count = month_count(
        forecast["as_of_start"],
        forecast["as_of_end"],
        int(forecast.get("as_of_frequency_months", 1)),
    )
    model_rows = count_model_configurations(config)
    total_model_configs = sum(row["model_config_count"] for row in model_rows)
    implemented_model_configs = sum(
        row["model_config_count"]
        for row in model_rows
        if row["implementation_status"] == "implemented"
    )
    planned_model_configs = total_model_configs - implemented_model_configs

    return {
        "experiment_id": config.get("experiment_id"),
        "status": config.get("status"),
        "forecast": forecast,
        "as_of_count": as_of_count,
        "model_builds": model_rows,
        "total_model_configurations": total_model_configs,
        "implemented_model_configurations": implemented_model_configs,
        "planned_model_configurations": planned_model_configs,
        "estimated_model_run_rows": total_model_configs * as_of_count,
        "estimated_implemented_model_run_rows": implemented_model_configs * as_of_count,
        "feature_family_validation": validate_feature_families(config, feature_families_path),
        "mlflow": (config.get("tracking") or {}).get("mlflow", {}),
        "checkpointing": (config.get("execution") or {}).get("checkpointing", {}),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    feature_families_uri = args.feature_families_uri or (config.get("inputs") or {}).get("feature_families_uri")
    feature_families_path = Path(feature_families_uri) if feature_families_uri and not str(feature_families_uri).startswith("s3://") else None
    summary = build_summary(config, feature_families_path)

    rendered = json.dumps(summary, indent=2, default=str)
    print(rendered)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
