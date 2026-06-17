#!/usr/bin/env python3
"""Validate the committed Streamlit dashboard artifact bundle.

This is a lightweight pre-deploy smoke check. It intentionally validates the
static files that Streamlit Cloud reads rather than the larger local experiment
archive.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_BUNDLE_DIR = Path("dashboard/public_artifacts/latest")
DASHBOARD_APP_PATH = Path("dashboard/app.py")
DASHBOARD_REQUIREMENTS_PATH = Path("dashboard/requirements.txt")

REQUIRED_FILES = {
    "forecast_paths": "forecast_paths.parquet",
    "performance_over_time": "performance_over_time.parquet",
    "model_leaderboard": "model_leaderboard.parquet",
    "feature_family_summary": "feature_family_summary.parquet",
    "champion_predictions": "champion_predictions.parquet",
    "champion_selection": "champion_selection.json",
}

REQUIRED_OPTIONAL_FOR_FULL_BUNDLE = {
    "model_leaderboard_full": "model_leaderboard_full.parquet",
    "feature_family_summary_full": "feature_family_summary_full.parquet",
    "complexity_profile_full": "complexity_profile_full.parquet",
    "path_partition_manifest": "path_partition_manifest.json",
    "public_bundle_manifest": "public_bundle_manifest.json",
    "experiment_manifest": "experiment_manifest.json",
}

REQUIRED_EDA_INPUTS = {
    "integrated monthly base": "integrated_monthly_base.parquet",
    "feature table": "feature_table.parquet",
    "imputation log": "imputation_log.parquet",
}

REQUIRED_COLUMNS = {
    "model_leaderboard.parquet": {
        "model_config_id",
        "config_id",
        "model_family",
        "model_build",
        "mode",
        "feature_family_name",
        "feature_policy",
        "feature_transform",
        "mae",
        "rmse",
        "selection_score_balanced",
    },
    "forecast_paths.parquet": {
        "config_id",
        "as_of_date",
        "target_date",
        "model_family",
        "model_build",
        "actual",
        "prediction",
        "error",
        "abs_error",
    },
    "performance_over_time.parquet": {
        "config_id",
        "as_of_date",
        "target_date",
        "model_family",
        "model_build",
        "abs_error",
    },
    "feature_family_summary.parquet": {
        "feature_family_name",
    },
    "champion_predictions.parquet": {
        "as_of_date",
        "target_date",
        "actual",
        "prediction",
    },
}

# Imports that dashboard/app.py and its local dashboard modules need from the
# deploy-time dashboard requirements.
DASHBOARD_IMPORT_PACKAGE_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "polars": "polars",
    "streamlit": "streamlit",
}


class ValidationError(RuntimeError):
    """Raised for bundle validation failures."""


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def check_required_files(bundle_dir: Path, failures: list[str]) -> None:
    for label, filename in REQUIRED_FILES.items():
        path = bundle_dir / filename
        if not path.exists():
            fail(f"Missing required {label}: {path}", failures)
        else:
            ok(f"Found required {label}: {filename}")

    for label, filename in REQUIRED_OPTIONAL_FOR_FULL_BUNDLE.items():
        path = bundle_dir / filename
        if not path.exists():
            fail(f"Missing current public-bundle {label}: {path}", failures)
        else:
            ok(f"Found current public-bundle {label}: {filename}")

    for label, filename in REQUIRED_EDA_INPUTS.items():
        path = bundle_dir / "eda_inputs" / filename
        if not path.exists():
            fail(f"Missing Data-page EDA input {label}: {path}", failures)
        else:
            ok(f"Found Data-page EDA input {label}: eda_inputs/{filename}")


def check_eda_inputs(bundle_dir: Path, failures: list[str]) -> None:
    for label, filename in REQUIRED_EDA_INPUTS.items():
        path = bundle_dir / "eda_inputs" / filename
        if not path.exists():
            continue
        try:
            rows = parquet_row_count(path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            fail(f"Could not read Data-page EDA input {label}: {exc}", failures)
            continue
        if rows <= 0 and label != "imputation log":
            fail(f"Data-page EDA input {label} is empty", failures)
        elif rows <= 0:
            ok(f"Data-page EDA input {label} is readable and empty")
        else:
            ok(f"Data-page EDA input {label} has {rows:,} rows")


def check_columns(bundle_dir: Path, failures: list[str]) -> None:
    for filename, required_columns in REQUIRED_COLUMNS.items():
        path = bundle_dir / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            fail(f"Could not read {filename}: {exc}", failures)
            continue
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            fail(f"{filename} missing required columns: {', '.join(missing)}", failures)
        else:
            ok(f"{filename} contains required columns")
        if frame.empty:
            fail(f"{filename} is empty", failures)
        else:
            ok(f"{filename} has {len(frame):,} rows")


def check_manifest_counts(bundle_dir: Path, failures: list[str]) -> None:
    manifest_path = bundle_dir / "public_bundle_manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)

    count_checks = {
        "leaderboard_rows": "model_leaderboard.parquet",
        "full_leaderboard_rows": "model_leaderboard_full.parquet",
        "forecast_rows": "forecast_paths.parquet",
        "performance_rows": "performance_over_time.parquet",
    }
    for manifest_key, filename in count_checks.items():
        path = bundle_dir / filename
        if not path.exists() or manifest_key not in manifest:
            continue
        actual = parquet_row_count(path)
        expected = int(manifest[manifest_key])
        if actual != expected:
            fail(f"{filename} row count {actual:,} does not match {manifest_key}={expected:,}", failures)
        else:
            ok(f"{filename} row count matches manifest ({actual:,})")

    if "source_configurations" in manifest and "full_leaderboard_rows" in manifest:
        if int(manifest["source_configurations"]) != int(manifest["full_leaderboard_rows"]):
            fail("source_configurations does not match full_leaderboard_rows", failures)
        else:
            ok("source_configurations matches full_leaderboard_rows")

    if "selected_configurations" in manifest and "leaderboard_rows" in manifest:
        if int(manifest["selected_configurations"]) != int(manifest["leaderboard_rows"]):
            fail("selected_configurations does not match leaderboard_rows", failures)
        else:
            ok("selected_configurations matches leaderboard_rows")


def check_partition_manifest(bundle_dir: Path, failures: list[str]) -> None:
    manifest_path = bundle_dir / "path_partition_manifest.json"
    public_manifest_path = bundle_dir / "public_bundle_manifest.json"
    if not manifest_path.exists():
        return
    partition_manifest = read_json(manifest_path)
    public_manifest = read_json(public_manifest_path) if public_manifest_path.exists() else {}
    datasets = partition_manifest.get("datasets", {})

    expected_public_counts = {
        "forecast_paths": public_manifest.get("full_forecast_rows"),
        "performance_over_time": public_manifest.get("full_performance_rows"),
    }

    for dataset_name, dataset_info in datasets.items():
        dataset_path = bundle_dir / str(dataset_info.get("path", ""))
        if not dataset_path.exists():
            fail(f"Partition dataset path missing for {dataset_name}: {dataset_path}", failures)
            continue
        partitions = dataset_info.get("partitions", [])
        if not partitions:
            fail(f"Partition dataset {dataset_name} has no partitions", failures)
            continue
        summed_rows = 0
        for part in partitions:
            part_path = bundle_dir / part["path"]
            if not part_path.exists():
                fail(f"Missing partition file for {dataset_name}: {part_path}", failures)
                continue
            actual = parquet_row_count(part_path)
            expected = int(part.get("rows", -1))
            summed_rows += expected if expected >= 0 else actual
            if actual != expected:
                fail(f"{part['path']} row count {actual:,} does not match manifest rows={expected:,}", failures)
        expected_dataset_rows = int(dataset_info.get("rows", -1))
        if expected_dataset_rows >= 0 and summed_rows != expected_dataset_rows:
            fail(f"{dataset_name} partition row sum {summed_rows:,} does not match dataset rows={expected_dataset_rows:,}", failures)
        elif expected_dataset_rows >= 0:
            ok(f"{dataset_name} partition rows sum to {expected_dataset_rows:,}")
        expected_public = expected_public_counts.get(dataset_name)
        if expected_public is not None and expected_dataset_rows != int(expected_public):
            fail(f"{dataset_name} rows {expected_dataset_rows:,} do not match public manifest full rows={int(expected_public):,}", failures)
        elif expected_public is not None:
            ok(f"{dataset_name} full row count matches public manifest")


def local_module_path(module_name: str, base_dir: Path) -> Optional[Path]:
    module_parts = module_name.split(".")
    file_candidate = base_dir.joinpath(*module_parts).with_suffix(".py")
    if file_candidate.exists():
        return file_candidate
    package_candidate = base_dir.joinpath(*module_parts) / "__init__.py"
    if package_candidate.exists():
        return package_candidate
    sibling_candidate = base_dir / f"{module_parts[0]}.py"
    return sibling_candidate if sibling_candidate.exists() else None


def imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def import_names_from_python_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def dashboard_import_names(entry_path: Path) -> set[str]:
    """Collect imports from app.py and local dashboard modules it imports."""
    base_dir = entry_path.parent
    imports: set[str] = set()
    visited: set[Path] = set()
    pending = [entry_path]

    while pending:
        path = pending.pop()
        resolved = path.resolve()
        if resolved in visited or not path.exists():
            continue
        visited.add(resolved)
        names = import_names_from_python_file(path)
        imports.update(names)
        for module_name in imported_module_names(path):
            local_path = local_module_path(module_name, base_dir)
            if local_path is not None:
                pending.append(local_path)

    return imports


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        token = line.split("#", 1)[0].strip()
        for sep in ("==", ">=", "<=", "~=", "[", ";"):
            token = token.split(sep, 1)[0].strip()
        names.add(token.lower().replace("_", "-"))
    return names


def check_dashboard_requirements(failures: list[str]) -> None:
    if not DASHBOARD_APP_PATH.exists() or not DASHBOARD_REQUIREMENTS_PATH.exists():
        return
    app_imports = dashboard_import_names(DASHBOARD_APP_PATH)
    needed = {
        requirement
        for import_name, requirement in DASHBOARD_IMPORT_PACKAGE_NAMES.items()
        if import_name in app_imports
    }
    declared = requirement_names(DASHBOARD_REQUIREMENTS_PATH)
    missing = sorted(req for req in needed if req.lower().replace("_", "-") not in declared)
    if missing:
        fail(f"dashboard/requirements.txt missing dashboard module imports: {', '.join(missing)}", failures)
    else:
        ok("dashboard/requirements.txt covers dashboard module runtime imports")


def validate(bundle_dir: Path) -> int:
    failures: list[str] = []
    if not bundle_dir.exists():
        fail(f"Bundle directory does not exist: {bundle_dir}", failures)
    else:
        ok(f"Validating dashboard bundle: {bundle_dir}")
        check_required_files(bundle_dir, failures)
        check_eda_inputs(bundle_dir, failures)
        check_columns(bundle_dir, failures)
        check_manifest_counts(bundle_dir, failures)
        check_partition_manifest(bundle_dir, failures)
    check_dashboard_requirements(failures)

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).", file=sys.stderr)
        return 1
    print("\nDashboard bundle validation passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=f"Dashboard bundle directory to validate. Default: {DEFAULT_BUNDLE_DIR}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return validate(args.bundle_dir)


if __name__ == "__main__":
    raise SystemExit(main())
