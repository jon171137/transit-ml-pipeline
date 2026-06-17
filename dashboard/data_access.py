import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_FEATURE_FAMILIES_PATH,
    DEFAULT_FEATURE_TABLE_PATH,
    DEFAULT_IMPUTATION_LOG_PATH,
    DEFAULT_INTEGRATED_BASE_PATH,
    EDA_FEATURE_TABLE_FILENAME,
    EDA_IMPUTATION_LOG_FILENAME,
    EDA_INPUT_DIRNAME,
    EDA_INTEGRATED_BASE_FILENAME,
    OPTIONAL_FILES,
    PATH_DATASET_DIRS,
    REQUIRED_FILES,
)

try:
    import polars as pl
except ImportError:  # Polars is an optimization, not a hard local-dev requirement.
    pl = None


def configured_integrated_base_path() -> Path:
    if "INTEGRATED_BASE_PATH" in os.environ:
        return Path(os.environ["INTEGRATED_BASE_PATH"])
    artifact_path = configured_artifact_dir() / EDA_INPUT_DIRNAME / EDA_INTEGRATED_BASE_FILENAME
    return artifact_path if artifact_path.exists() else DEFAULT_INTEGRATED_BASE_PATH


def configured_feature_table_path() -> Path:
    if "FEATURE_TABLE_PATH" in os.environ:
        return Path(os.environ["FEATURE_TABLE_PATH"])
    artifact_path = configured_artifact_dir() / EDA_INPUT_DIRNAME / EDA_FEATURE_TABLE_FILENAME
    return artifact_path if artifact_path.exists() else DEFAULT_FEATURE_TABLE_PATH


def configured_imputation_log_path() -> Path:
    if "IMPUTATION_LOG_PATH" in os.environ:
        return Path(os.environ["IMPUTATION_LOG_PATH"])
    artifact_path = configured_artifact_dir() / EDA_INPUT_DIRNAME / EDA_IMPUTATION_LOG_FILENAME
    return artifact_path if artifact_path.exists() else DEFAULT_IMPUTATION_LOG_PATH


@st.cache_data(show_spinner=False)
def load_integrated_base(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_feature_table(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_imputation_log(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    _ = modified_ns
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(path: str, modified_ns: int) -> dict:
    _ = modified_ns
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_text(path: str, modified_ns: int) -> str:
    _ = modified_ns
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def file_modified_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


def configured_artifact_dir() -> Path:
    return Path(os.environ.get("DASHBOARD_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR))


def configured_feature_families_path() -> Path:
    return Path(os.environ.get("FEATURE_FAMILIES_PATH", DEFAULT_FEATURE_FAMILIES_PATH))


def safe_partition_value(value) -> str:
    text = "unknown" if pd.isna(value) else str(value)
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text)
    return safe.strip("_") or "unknown"


def model_id_column(df: pd.DataFrame) -> str:
    if "model_config_id" in df.columns:
        return "model_config_id"
    if "config_id" in df.columns:
        return "config_id"
    raise KeyError("Expected either model_config_id or config_id.")


def path_collection_modified_ns(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_mtime_ns
    return max((file.stat().st_mtime_ns for file in path.rglob("*.parquet")), default=path.stat().st_mtime_ns)


def path_partition_files(dataset_dir: Path, model_builds: tuple[str, ...]) -> list[Path]:
    if not dataset_dir.exists():
        return []
    if model_builds:
        files = []
        for build in model_builds:
            files.extend((dataset_dir / f"model_build={safe_partition_value(build)}").glob("*.parquet"))
        return sorted(path for path in files if path.exists())
    return sorted(dataset_dir.glob("**/*.parquet"))


@st.cache_data(show_spinner=False)
def load_path_rows_for_configs(
    run_dir: str,
    dataset_name: str,
    config_ids: tuple[str, ...],
    model_builds: tuple[str, ...],
    modified_ns: int,
) -> pd.DataFrame:
    _ = modified_ns
    if not config_ids:
        return pd.DataFrame()

    run_path = Path(run_dir)
    dataset_dir = run_path / PATH_DATASET_DIRS[dataset_name]
    files = path_partition_files(dataset_dir, model_builds)
    config_values = [str(config_id) for config_id in config_ids]

    if files and pl is not None:
        lazy_frames = [pl.scan_parquet(str(path)) for path in files]
        lazy = lazy_frames[0] if len(lazy_frames) == 1 else pl.concat(lazy_frames, how="diagonal_relaxed")
        columns = set(lazy.collect_schema().names())
        id_col = "model_config_id" if "model_config_id" in columns else "config_id"
        filtered = lazy.filter(pl.col(id_col).cast(pl.Utf8).is_in(config_values))
        return filtered.collect().to_pandas()

    if files:
        frames = []
        for path in files:
            frame = pd.read_parquet(path)
            id_col = model_id_column(frame)
            frames.append(frame[frame[id_col].astype(str).isin(config_values)])
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fallback_path = run_path / REQUIRED_FILES[dataset_name]
    if fallback_path.exists():
        frame = load_parquet(str(fallback_path), file_modified_ns(fallback_path))
        id_col = model_id_column(frame)
        return frame[frame[id_col].astype(str).isin(config_values)].copy()
    return pd.DataFrame()


def discover_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    if all((base_dir / filename).exists() for filename in REQUIRED_FILES.values()):
        return [base_dir]

    run_dirs = [
        path
        for path in base_dir.glob("**/*")
        if path.is_dir() and all((path / filename).exists() for filename in REQUIRED_FILES.values())
    ]
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def load_artifacts(run_dir: Path) -> dict:
    def parquet_artifact(filename: str) -> pd.DataFrame:
        path = run_dir / filename
        return load_parquet(str(path), file_modified_ns(path))

    def json_artifact(filename: str) -> dict:
        path = run_dir / filename
        return load_json(str(path), file_modified_ns(path))

    artifacts = {
        "forecast_paths": parquet_artifact(REQUIRED_FILES["forecast_paths"]),
        "performance_over_time": parquet_artifact(REQUIRED_FILES["performance_over_time"]),
        "model_leaderboard": parquet_artifact(REQUIRED_FILES["model_leaderboard"]),
        "feature_family_summary": parquet_artifact(REQUIRED_FILES["feature_family_summary"]),
        "champion_predictions": parquet_artifact(REQUIRED_FILES["champion_predictions"]),
        "champion_selection": json_artifact(REQUIRED_FILES["champion_selection"]),
    }
    for artifact_name, filename in OPTIONAL_FILES.items():
        path = run_dir / filename
        if path.exists():
            if filename.endswith(".json"):
                artifacts[artifact_name] = load_json(str(path), file_modified_ns(path))
            else:
                artifacts[artifact_name] = load_parquet(str(path), file_modified_ns(path))
    return artifacts


def load_feature_family_definitions() -> dict:
    path = configured_feature_families_path()
    if not path.exists():
        return {}
    return load_json(str(path), file_modified_ns(path))


def load_config_text(path: Path) -> str:
    if not path.exists():
        return ""
    return load_text(str(path), file_modified_ns(path))
