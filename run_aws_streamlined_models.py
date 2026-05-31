from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import yaml
except ImportError:
    yaml = None

# Usage notes:
#
# 1. AWS/ECS default path
#    Step Functions/ECS should normally run this with only PIPELINE_RUN_ID set:
#
#      python run_aws_streamlined_models.py
#
#    The script will read:
#      s3://<bucket>/features/integrated_monthly_h3/run_id=<PIPELINE_RUN_ID>/
#
#    and write:
#      s3://<bucket>/model_results/aws_streamlined/run_id=<PIPELINE_RUN_ID>/
#      s3://<bucket>/dashboard/aws_streamlined/run_id=<PIPELINE_RUN_ID>/
#
# 2. Local smoke test with local feature artifacts and local outputs
#    Use explicit base URIs for local outputs. Do not use --results-prefix for
#    local tests unless you intentionally want to write to S3 under that prefix.
#
#      python run_aws_streamlined_models.py \
#        --feature-table-uri feature_store/test_run_id/feature_table.parquet \
#        --feature-families-uri feature_store/test_run_id/feature_families.json \
#        --results-base-uri local_model_results/aws_streamlined/test_run \
#        --dashboard-base-uri local_dashboard/aws_streamlined/test_run
#
# 3. Explicit S3 feature run
#    Useful when testing one known feature-table run without relying on the
#    latest S3 object:
#
#      python run_aws_streamlined_models.py \
#        --feature-run-id manual-manifest-20260519T193000Z
#
# 4. Important modeling knobs
#    --as-of-start controls the first monthly forecast origin. Default: 2021-01-01.
#    --min-train-rows controls the minimum training rows required before fitting.
#    --xgb-refresh-months controls how often XGBoost refits. Default: 12, so it
#    still produces monthly predictions but only refreshes the model annually to
#    keep the AWS demo reasonably light.
#
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import mlflow
except ImportError:
    mlflow = None

if load_dotenv:
    load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
FEATURE_OUTPUT_PREFIX = os.environ.get("FEATURE_OUTPUT_PREFIX", "features/integrated_monthly_h3")
MODEL_RESULTS_PREFIX = os.environ.get("MODEL_RESULTS_PREFIX", "model_results/aws_streamlined")
DASHBOARD_OUTPUT_PREFIX = os.environ.get("DASHBOARD_OUTPUT_PREFIX", "dashboard/aws_streamlined")
DEFAULT_MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "transit-forecasting")

DEFAULT_TARGET = "upt"
DEFAULT_HORIZON = 3
DEFAULT_AS_OF_START = "2021-01-01"
FEATURE_TABLE_FILENAME = "feature_table.parquet"
FEATURE_FAMILIES_FILENAME = "feature_families.json"
MODEL_MODES = ["raw", "residual"]
MODEL_TYPES = ["naive", "ridge", "lasso", "elastic_net", "random_forest", "extra_trees", "xgboost"]
MODEL_ORDER = {
    "naive": 0,
    "ridge": 1,
    "lasso": 2,
    "elastic_net": 3,
    "random_forest": 4,
    "extra_trees": 5,
    "xgboost": 6,
    "arima": 7,
    "sarima": 8,
    "sarimax": 9,
}
SIMPLICITY_THRESHOLD = 0.02
SCORE_RECIPES = {
    "typical": {"mae_weight": 0.90, "rmse_weight": 0.10},
    "balanced": {"mae_weight": 0.75, "rmse_weight": 0.25},
    "large_error": {"mae_weight": 0.50, "rmse_weight": 0.50},
}
FEATURE_POLICIES = [
    "none",
    "corr_pruned",
    "variance_pruned",
    "mutual_info_top_20",
    "mutual_info_top_30",
    "lasso_selected",
    "tree_top_20",
    "tree_top_30",
]
CORR_PRUNE_THRESHOLD = 0.95
VARIANCE_PRUNE_THRESHOLD = 1e-8
MUTUAL_INFO_RANDOM_STATE = 42
LASSO_SELECTOR_ALPHA = 10.0
TREE_SELECTOR_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the streamlined AWS modeling comparison across notebook-aligned feature "
            "families, raw/residual modes, and a compact naive/ridge/lasso/XGBoost grid."
        )
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help=(
            "Optional YAML experiment config. When provided, it can supply feature inputs, "
            "outputs, model grids, feature families, MLflow settings, and checkpoint folders."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=BUCKET_NAME,
        help="S3 bucket for default feature inputs and modeling outputs.",
    )
    parser.add_argument(
        "--feature-prefix",
        default=FEATURE_OUTPUT_PREFIX,
        help="S3 prefix containing feature-table run partitions.",
    )
    parser.add_argument(
        "--feature-run-id",
        default=os.environ.get("PIPELINE_RUN_ID"),
        help=(
            "Feature-table run ID to load. Defaults to PIPELINE_RUN_ID. "
            "If omitted, the latest feature table under --feature-prefix is used."
        ),
    )
    parser.add_argument(
        "--feature-table-uri",
        default=None,
        help="Explicit local path or s3:// URI for feature_table.parquet.",
    )
    parser.add_argument(
        "--feature-families-uri",
        default=None,
        help="Explicit local path or s3:// URI for feature_families.json.",
    )
    parser.add_argument(
        "--include-feature-family",
        action="append",
        default=None,
        help=(
            "Feature family to include. Repeat for multiple families. "
            "Defaults to all feature families."
        ),
    )
    parser.add_argument(
        "--include-model-type",
        action="append",
        choices=MODEL_TYPES,
        default=None,
        help=(
            "Model type to include. Repeat for multiple model types. "
            "Defaults to all streamlined model types."
        ),
    )
    parser.add_argument(
        "--feature-policy",
        action="append",
        choices=FEATURE_POLICIES,
        default=None,
        help=(
            "Feature policy to apply. Repeat for multiple policies. "
            "'corr_pruned' currently applies to linear models only; other model "
            "families fall back to 'none'. Defaults to 'none'."
        ),
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["upt"],
        help="Target series to model. The streamlined AWS version currently supports UPT.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon in months.",
    )
    parser.add_argument(
        "--as-of-start",
        default=DEFAULT_AS_OF_START,
        help="First monthly as-of date to evaluate, formatted YYYY-MM-DD.",
    )
    parser.add_argument(
        "--as-of-end",
        default=None,
        help="Optional last monthly as-of date to evaluate, formatted YYYY-MM-DD.",
    )
    parser.add_argument(
        "--as-of-frequency-months",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth monthly forecast origin. Use 3 for quarterly "
            "dashboard-grain simulations while preserving 3-month-ahead targets."
        ),
    )
    parser.add_argument(
        "--results-prefix",
        default=MODEL_RESULTS_PREFIX,
        help="S3 prefix for model result artifacts.",
    )
    parser.add_argument(
        "--dashboard-prefix",
        default=DASHBOARD_OUTPUT_PREFIX,
        help="S3 prefix for dashboard-ready artifacts.",
    )
    parser.add_argument(
        "--results-base-uri",
        default=None,
        help=(
            "Explicit local path or s3:// URI for model results. "
            "If omitted, defaults to s3://<bucket>/<results-prefix>/run_id=<run_id>."
        ),
    )
    parser.add_argument(
        "--dashboard-base-uri",
        default=None,
        help=(
            "Explicit local path or s3:// URI for dashboard outputs. "
            "If omitted, defaults to s3://<bucket>/<dashboard-prefix>/run_id=<run_id>."
        ),
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=60,
        help="Minimum non-null training rows required before fitting a model.",
    )
    parser.add_argument(
        "--xgb-refresh-months",
        type=int,
        default=1,
        help=(
            "How often XGBoost models are refit during rolling evaluation. "
            "Default is monthly."
        ),
    )
    parser.add_argument(
        "--refit-frequency-months",
        type=int,
        default=1,
        help=(
            "Generalized refit cadence for all non-naive models. Default is monthly."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for model-config evaluation. Default: 1.",
    )
    parser.add_argument(
        "--chunk-dir",
        default=None,
        help="Optional local directory for per-model-configuration chunk outputs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional local directory for completed_configs.parquet and failed_configs.parquet.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip model-configuration chunks already present in --chunk-dir.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=os.environ.get("ENABLE_MLFLOW", "").lower() in {"1", "true", "yes"},
        help="Log experiment-level metadata, metrics, and compact artifacts to MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI. Defaults to MLflow's local ./mlruns behavior when enabled.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help="MLflow experiment name to use when MLflow logging is enabled.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=os.environ.get("MLFLOW_RUN_NAME"),
        help="Optional MLflow run name. Defaults to the experiment/run ID.",
    )
    return parser.parse_args()


def read_yaml_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("YAML config support requires PyYAML. Install project requirements first.")
    return yaml.safe_load(Path(path).read_text()) or {}


def model_type_from_config_name(model_build: str) -> str:
    if model_build == "seasonal_naive":
        return "naive"
    return model_build


def model_grid_from_config(config: dict) -> list[dict]:
    rows = []
    models = config.get("models") or {}
    for _model_family, builds in models.items():
        for model_build, details in (builds or {}).items():
            if not details.get("enabled", False):
                continue
            model_type = model_type_from_config_name(model_build)
            for params in details.get("param_grid") or [{}]:
                rows.append({"model_type": model_type, "params": params or {}})
    return rows


def feature_policy_list_from_config(config: dict) -> list[str]:
    policies = []
    for values in (config.get("feature_policies") or {}).values():
        for policy in values or []:
            if policy not in policies:
                policies.append(policy)
    return policies or ["none"]


def apply_experiment_config(args: argparse.Namespace) -> argparse.Namespace:
    args.experiment_config_payload = None
    args.model_grid = None
    if not args.experiment_config:
        return args

    config = read_yaml_config(args.experiment_config)
    args.experiment_config_payload = config
    inputs = config.get("inputs") or {}
    outputs = config.get("outputs") or {}
    forecast = config.get("forecast") or {}
    execution = config.get("execution") or {}
    tracking = (config.get("tracking") or {}).get("mlflow") or {}
    checkpointing = execution.get("checkpointing") or {}

    args.feature_table_uri = inputs.get("feature_table_uri", args.feature_table_uri)
    args.feature_families_uri = inputs.get("feature_families_uri", args.feature_families_uri)
    args.results_base_uri = outputs.get("results_base_uri", args.results_base_uri)
    args.dashboard_base_uri = outputs.get("dashboard_base_uri", args.dashboard_base_uri)

    args.target = forecast.get("target", args.target)
    args.horizon = int(forecast.get("horizon", args.horizon))
    args.as_of_start = forecast.get("as_of_start", args.as_of_start)
    args.as_of_end = forecast.get("as_of_end", args.as_of_end)
    args.as_of_frequency_months = int(forecast.get("as_of_frequency_months", args.as_of_frequency_months))
    args.refit_frequency_months = int(forecast.get("refit_frequency_months", args.refit_frequency_months))
    args.min_train_rows = int(forecast.get("min_train_rows", args.min_train_rows))

    args.n_jobs = int(execution.get("n_jobs", args.n_jobs))
    args.chunk_dir = checkpointing.get("chunk_dir", args.chunk_dir)
    args.checkpoint_dir = checkpointing.get("checkpoint_dir", args.checkpoint_dir)
    args.resume = bool(checkpointing.get("resume", args.resume))

    included_families = (config.get("feature_families") or {}).get("include")
    if included_families:
        args.include_feature_family = list(included_families)

    model_grid = model_grid_from_config(config)
    if model_grid:
        args.model_grid = model_grid
        args.include_model_type = sorted({row["model_type"] for row in model_grid})

    configured_policies = feature_policy_list_from_config(config)
    if configured_policies:
        args.feature_policy = configured_policies

    if tracking:
        args.enable_mlflow = bool(tracking.get("enabled", args.enable_mlflow))
        args.mlflow_tracking_uri = tracking.get("tracking_uri", args.mlflow_tracking_uri)
        args.mlflow_experiment_name = tracking.get("experiment_name", args.mlflow_experiment_name)
        args.mlflow_run_name = tracking.get("run_name", args.mlflow_run_name)

    return args


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    bucket_and_key = uri[len("s3://") :]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def join_uri(base_uri: str, filename: str) -> str:
    return f"{base_uri.rstrip('/')}/{filename}"


def read_json_uri(uri: str) -> dict:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        logger.info("Reading JSON from s3://%s/%s", bucket, key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    path = Path(uri)
    logger.info("Reading JSON from %s", path)
    return json.loads(path.read_text())


def read_parquet_uri(uri: str) -> pd.DataFrame:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        logger.info("Reading Parquet from s3://%s/%s", bucket, key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    path = Path(uri)
    logger.info("Reading Parquet from %s", path)
    return pd.read_parquet(path)


def write_json_uri(uri: str, payload: dict) -> None:
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def write_parquet_uri(uri: str, df: pd.DataFrame) -> None:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def find_latest_s3_key(bucket: str, prefix: str, filename: str) -> str:
    normalized_prefix = prefix.strip("/")
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized_prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(f"/{filename}"):
                candidates.append(key)

    if not candidates:
        raise FileNotFoundError(f"No {filename!r} objects found under s3://{bucket}/{normalized_prefix}/")

    latest_key = max(candidates)
    logger.info("Resolved latest %s under s3://%s/%s/ to %s", filename, bucket, normalized_prefix, latest_key)
    return latest_key


def resolve_feature_base_uri(args: argparse.Namespace) -> str:
    if args.feature_table_uri and args.feature_families_uri:
        table_base = args.feature_table_uri.rsplit("/", 1)[0]
        families_base = args.feature_families_uri.rsplit("/", 1)[0]
        if table_base != families_base:
            logger.warning("Feature table and family files are not in the same folder.")
        return table_base

    if args.feature_run_id:
        prefix = args.feature_prefix.strip("/")
        return s3_uri(args.bucket, f"{prefix}/run_id={args.feature_run_id}")

    latest_table_key = find_latest_s3_key(args.bucket, args.feature_prefix, FEATURE_TABLE_FILENAME)
    return s3_uri(args.bucket, latest_table_key.rsplit("/", 1)[0])


def resolve_feature_artifacts(args: argparse.Namespace) -> dict:
    base_uri = resolve_feature_base_uri(args)
    feature_table_uri = args.feature_table_uri or join_uri(base_uri, FEATURE_TABLE_FILENAME)
    feature_families_uri = args.feature_families_uri or join_uri(base_uri, FEATURE_FAMILIES_FILENAME)

    return {
        "feature_base_uri": base_uri,
        "feature_table_uri": feature_table_uri,
        "feature_families_uri": feature_families_uri,
    }


def current_model_run_id(feature_base_uri: str) -> str:
    env_run_id = os.environ.get("PIPELINE_RUN_ID")
    if env_run_id:
        return env_run_id

    marker = "/run_id="
    if marker in feature_base_uri:
        return feature_base_uri.split(marker, 1)[1].split("/", 1)[0]

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def resolve_output_base_uri(explicit_uri: str | None, bucket: str, prefix: str, run_id: str) -> str:
    if explicit_uri:
        return explicit_uri.rstrip("/")
    return s3_uri(bucket, f"{prefix.strip('/')}/run_id={run_id}")


def validate_feature_table(feature_table: pd.DataFrame, target: str, horizon: int) -> str:
    if "date" not in feature_table.columns:
        raise ValueError("Feature table must include a 'date' column.")

    target_col = f"{target}_target_h{horizon}"
    if target_col not in feature_table.columns:
        raise ValueError(f"Feature table must include target column {target_col!r}.")

    if feature_table["date"].duplicated().any():
        raise ValueError("Feature table has duplicate dates.")

    return target_col


def add_seasonal_naive_proxy(
    df: pd.DataFrame,
    target_col: str,
    seasonal_periods: int = 12,
) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work = work.sort_values("date").reset_index(drop=True)
    work["seasonal_naive_proxy"] = work[target_col].shift(seasonal_periods)
    return work


def build_evaluation_frame(
    feature_table: pd.DataFrame,
    target_col: str,
    as_of_start: str,
    as_of_end: str | None,
    as_of_frequency_months: int,
    horizon: int,
) -> pd.DataFrame:
    if as_of_frequency_months < 1:
        raise ValueError("--as-of-frequency-months must be at least 1.")

    eval_df = feature_table.copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    eval_df = eval_df.sort_values("date").reset_index(drop=True)
    eval_df["target_date"] = eval_df["date"] + pd.DateOffset(months=horizon)
    eval_df = add_seasonal_naive_proxy(eval_df, target_col=target_col, seasonal_periods=12)

    as_of_start_ts = pd.Timestamp(as_of_start)
    as_of_end_ts = pd.Timestamp(as_of_end) if as_of_end else None
    evaluable = eval_df[
        (eval_df["date"] >= as_of_start_ts)
        & eval_df[target_col].notna()
        & eval_df["seasonal_naive_proxy"].notna()
    ].copy()
    if as_of_end_ts is not None:
        evaluable = evaluable[evaluable["date"] <= as_of_end_ts].copy()

    if as_of_frequency_months > 1:
        months_since_start = (
            (evaluable["date"].dt.year - as_of_start_ts.year) * 12
            + (evaluable["date"].dt.month - as_of_start_ts.month)
        )
        evaluable = evaluable[months_since_start % as_of_frequency_months == 0].copy()

    if evaluable.empty:
        raise ValueError(
            f"No evaluable rows found from {as_of_start!r} to {as_of_end!r} "
            "with non-null target and naive baseline."
        )

    return evaluable


EVALUATION_SCOPES = ["overall", "pre_covid", "covid_shock", "recovery", "recent"]


def evaluation_period_for(target_date) -> str:
    target_ts = pd.Timestamp(target_date)
    if pd.Timestamp("2020-03-01") <= target_ts <= pd.Timestamp("2021-06-01"):
        return "covid_shock"
    if pd.Timestamp("2021-07-01") <= target_ts <= pd.Timestamp("2022-12-01"):
        return "recovery"
    if target_ts >= pd.Timestamp("2023-01-01"):
        return "recent"
    return "pre_covid"


def is_shock_period(target_date) -> bool:
    return evaluation_period_for(target_date) == "covid_shock"


def months_between(start_date, end_date) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month)


def summarize_feature_families(feature_families: dict, feature_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, columns in feature_families.items():
        available = [col for col in columns if col in feature_table.columns]
        missing = [col for col in columns if col not in feature_table.columns]
        rows.append(
            {
                "feature_family_name": name,
                "requested_features": len(columns),
                "available_features": len(available),
                "missing_features": missing,
            }
        )
    return pd.DataFrame(rows)


def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    actual_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(pred))
    return float((actual_diff == pred_diff).mean()) if len(actual_diff) > 0 else np.nan


def adjusted_r2_score_value(r2: float, n_observations: int, n_predictors: int, model_family: str) -> float:
    if model_family != "linear":
        return np.nan
    if not np.isfinite(r2):
        return np.nan
    if n_observations <= n_predictors + 1:
        return np.nan
    return float(1 - ((1 - r2) * (n_observations - 1) / (n_observations - n_predictors - 1)))


def model_param_grid(model_grid: list[dict] | None = None) -> list[dict]:
    if model_grid is not None:
        return model_grid

    configs = [{"model_type": "naive", "params": {}}]

    for alpha in [1.0, 10.0]:
        configs.append({"model_type": "ridge", "params": {"alpha": alpha}})
    for alpha in [100.0, 1000.0]:
        configs.append({"model_type": "lasso", "params": {"alpha": alpha, "max_iter": 5000}})

    configs.append(
        {
            "model_type": "xgboost",
            "params": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
            },
        }
    )

    return configs


def build_model(model_type: str, params: dict):
    if model_type == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(**params)),
            ]
        )
    if model_type == "lasso":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(**params)),
            ]
        )
    if model_type == "elastic_net":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**params)),
            ]
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=1,
            **params,
        )
    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            random_state=42,
            n_jobs=1,
            **params,
        )
    if model_type == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            **params,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def effective_feature_policy(model_type: str, requested_policy: str) -> str:
    linear_models = {"ridge", "lasso", "elastic_net"}
    tree_models = {"random_forest", "extra_trees", "xgboost"}
    if requested_policy == "corr_pruned" and model_type not in linear_models:
        return "none"
    if requested_policy == "lasso_selected" and model_type not in linear_models:
        return "none"
    if requested_policy.startswith("tree_top_") and model_type not in tree_models:
        return "none"
    if requested_policy.startswith("mutual_info_top_") and model_type not in linear_models | tree_models:
        return "none"
    if requested_policy == "variance_pruned" and model_type not in linear_models | tree_models:
        return "none"
    return requested_policy


def parse_top_k_policy(policy: str, prefix: str, default: int) -> int:
    if not policy.startswith(prefix):
        return default
    try:
        return max(1, int(policy.replace(prefix, "", 1)))
    except ValueError:
        return default


def top_k_columns(scores: pd.Series, k: int) -> list[str]:
    clean_scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_scores.empty:
        return []
    return clean_scores.sort_values(ascending=False).head(k).index.tolist()


def apply_feature_policy(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    y_train: pd.Series | None,
    policy: str,
    threshold: float = CORR_PRUNE_THRESHOLD,
) -> dict:
    result = {
        "selected_features": list(feature_cols),
        "dropped_features": [],
        "policy_params": {},
        "n_features_before_policy": len(feature_cols),
        "n_features_after_policy": len(feature_cols),
    }
    if policy == "none" or len(feature_cols) <= 1:
        return result

    X = train_df[feature_cols].astype(float)
    selected = list(feature_cols)
    params = {}

    if policy == "variance_pruned":
        variances = X.var(axis=0)
        params = {"threshold": VARIANCE_PRUNE_THRESHOLD}
        selected = variances[variances > VARIANCE_PRUNE_THRESHOLD].index.tolist()
    elif policy == "corr_pruned":
        params = {"threshold": threshold}
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [
            column
            for column in upper.columns
            if any(upper[column] > threshold)
        ]
        selected = [column for column in feature_cols if column not in set(drop_cols)]
    elif policy.startswith("mutual_info_top_"):
        if y_train is None:
            raise ValueError("mutual_info_top_k requires y_train.")
        k = parse_top_k_policy(policy, "mutual_info_top_", default=30)
        params = {"k": k, "random_state": MUTUAL_INFO_RANDOM_STATE}
        scores = mutual_info_regression(
            X,
            y_train.astype(float),
            random_state=MUTUAL_INFO_RANDOM_STATE,
        )
        selected = top_k_columns(pd.Series(scores, index=feature_cols), min(k, len(feature_cols)))
    elif policy == "lasso_selected":
        if y_train is None:
            raise ValueError("lasso_selected requires y_train.")
        params = {"alpha": LASSO_SELECTOR_ALPHA, "max_iter": 10000}
        selector = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=LASSO_SELECTOR_ALPHA, max_iter=10000)),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            selector.fit(X, y_train.astype(float))
        coefs = pd.Series(np.abs(selector.named_steps["model"].coef_), index=feature_cols)
        selected = coefs[coefs > 0].sort_values(ascending=False).index.tolist()
    elif policy.startswith("tree_top_"):
        if y_train is None:
            raise ValueError("tree_top_k requires y_train.")
        k = parse_top_k_policy(policy, "tree_top_", default=30)
        params = {
            "k": k,
            "selector_model": "extra_trees",
            "n_estimators": 100,
            "max_depth": 4,
            "random_state": TREE_SELECTOR_RANDOM_STATE,
        }
        selector = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=TREE_SELECTOR_RANDOM_STATE,
            n_jobs=1,
        )
        selector.fit(X, y_train.astype(float))
        selected = top_k_columns(
            pd.Series(selector.feature_importances_, index=feature_cols),
            min(k, len(feature_cols)),
        )
    else:
        raise ValueError(f"Unsupported feature policy: {policy}")

    if not selected:
        selected = feature_cols[:1]
    dropped = [column for column in feature_cols if column not in set(selected)]
    result.update(
        {
            "selected_features": selected,
            "dropped_features": dropped,
            "policy_params": params,
            "n_features_after_policy": len(selected),
        }
    )
    return result


def eligible_feature_columns(feature_families: dict, family_name: str, feature_table: pd.DataFrame) -> list[str]:
    if family_name not in feature_families:
        raise KeyError(f"Unknown feature family: {family_name}")
    return [col for col in feature_families[family_name] if col in feature_table.columns]


def config_id(model_type: str, mode: str, feature_family_name: str, params: dict, feature_policy: str = "none") -> str:
    policy_text = f"__policy-{feature_policy}" if feature_policy != "none" else ""
    if params:
        params_text = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
        return f"{model_type}__{mode}__{feature_family_name}{policy_text}__{params_text}"
    return f"{model_type}__{mode}__{feature_family_name}{policy_text}"


def model_family_for(model_type: str) -> str:
    if model_type == "naive":
        return "baseline"
    if model_type in {"ridge", "lasso", "elastic_net"}:
        return "linear"
    if model_type in {"random_forest", "extra_trees", "xgboost"}:
        return "tree"
    if model_type in {"arima", "sarima", "sarimax"}:
        return "autoregressive"
    if model_type in {"mlp", "rnn", "gru", "lstm"}:
        return "neural_net"
    return "other"


def ensemble_method_for(model_type: str) -> str:
    if model_type == "random_forest":
        return "bagging"
    if model_type == "extra_trees":
        return "randomized_bagging"
    if model_type == "xgboost":
        return "boosting"
    return ""


def model_build_for(model_type: str) -> str:
    return "seasonal_naive" if model_type == "naive" else model_type


def framework_for(model_type: str) -> str:
    if model_type == "xgboost":
        return "xgboost"
    if model_type in {"arima", "sarima", "sarimax"}:
        return "statsmodels"
    if model_type in {"mlp", "rnn", "gru", "lstm"}:
        return "pytorch"
    if model_type in {"naive", "ridge", "lasso", "elastic_net", "random_forest", "extra_trees"}:
        return "sklearn"
    return ""


def default_representation_metadata(model_family: str = "tabular") -> dict:
    if model_family == "neural_net":
        policy = "sequence_raw"
    else:
        policy = "tabular_raw"
    return {
        "representation_policy": policy,
        "representation_params_json": "{}",
        "n_representation_features": np.nan,
        "sequence_length": np.nan,
        "sequence_stride": np.nan,
        "prediction_head": "direct_horizon",
        "training_window_months": np.nan,
        "validation_strategy": "rolling_as_of",
        "early_stopping_used": False,
        "epochs_trained": np.nan,
        "best_epoch": np.nan,
        "hardware_type": os.environ.get("HARDWARE_TYPE", "cpu"),
        "device": os.environ.get("DEVICE", "cpu"),
        "gpu_name": os.environ.get("GPU_NAME", ""),
        "cuda_version": os.environ.get("CUDA_VERSION", ""),
    }


def feature_set_id(feature_family_name: str, mode: str, feature_cols: list[str], feature_policy: str = "none") -> str:
    payload = json.dumps(
        {
            "feature_family_name": feature_family_name,
            "mode": mode,
            "feature_policy": feature_policy,
            "feature_names": sorted(feature_cols),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{feature_family_name}__{mode}__{feature_policy}__{digest}"


def feature_set_hash(feature_cols: list[str]) -> str:
    payload = json.dumps(sorted(feature_cols), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def safe_ape(actual: float, prediction: float) -> float:
    if pd.isna(actual) or actual == 0:
        return np.nan
    return float(abs(prediction - actual) / abs(actual))


def prediction_for_row(
    model,
    model_type: str,
    mode: str,
    row: pd.Series,
    feature_cols: list[str],
) -> float:
    if model_type == "naive":
        return float(row["seasonal_naive_proxy"])

    x_row = pd.DataFrame([row[feature_cols].to_dict()]).astype(float)
    pred_value = float(model.predict(x_row)[0])
    if mode == "residual":
        return float(row["seasonal_naive_proxy"] + pred_value)
    return pred_value


def extract_feature_importance(
    model,
    model_type: str,
    feature_cols: list[str],
) -> list[dict]:
    rows = []
    if model_type in {"ridge", "lasso", "elastic_net"}:
        coefs = model.named_steps["model"].coef_
        for feature_name, importance in zip(feature_cols, coefs):
            rows.append(
                {
                    "feature_name": feature_name,
                    "importance_type": "coefficient",
                    "importance": float(importance),
                    "importance_abs": float(abs(importance)),
                }
            )
    elif model_type in {"random_forest", "extra_trees", "xgboost"}:
        for feature_name, importance in zip(feature_cols, model.feature_importances_):
            rows.append(
                {
                    "feature_name": feature_name,
                    "importance_type": "feature_importance",
                    "importance": float(importance),
                    "importance_abs": float(abs(importance)),
                }
            )
    return sorted(rows, key=lambda row: row["importance_abs"], reverse=True)


def feature_set_row(
    experiment_id: str,
    feature_set_id_value: str,
    feature_family_name: str,
    mode: str,
    feature_policy: str,
    feature_cols: list[str],
) -> dict:
    return {
        "experiment_id": experiment_id,
        "feature_set_id": feature_set_id_value,
        "feature_family_name": feature_family_name,
        "mode": mode,
        "feature_policy": feature_policy,
        "feature_count": len(feature_cols),
        "feature_hash": feature_set_hash(feature_cols),
        "feature_names_json": json.dumps(feature_cols, sort_keys=True),
        "description": "",
        "includes_lags": any("lag" in col.lower() for col in feature_cols),
        "includes_rolling": any(
            token in col.lower() for col in feature_cols for token in ["roll", "rolling"]
        ),
        "includes_exogenous": any(
            token in col.lower()
            for col in feature_cols
            for token in ["gas", "cpi", "inflation", "income"]
        ),
        "includes_service": any("service" in col.lower() for col in feature_cols),
        "includes_interactions": any(
            token in col.lower() for col in feature_cols for token in ["interact", "_x_"]
        ),
    }


def run_config_evaluation(task: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_table = task["full_table"]
    evaluation_frame = task["evaluation_frame"]
    feature_cols = task["feature_cols"]
    config = task["config"]
    model_type = config["model_type"]
    params = config["params"]
    mode = task["mode"]
    feature_family_name = task["feature_family_name"]
    feature_policy = task["feature_policy"]
    experiment_id = task["experiment_id"]
    pipeline_run_id = task["pipeline_run_id"]
    target_col = task["target_col"]
    target = task["target"]
    horizon = task["horizon"]
    min_train_rows = task["min_train_rows"]
    xgb_refresh_months = task["xgb_refresh_months"]
    refit_frequency_months = task["refit_frequency_months"]

    model_family = model_family_for(model_type)
    model_build = model_build_for(model_type)
    ensemble_method = ensemble_method_for(model_type)
    representation_metadata = default_representation_metadata(model_family)
    hyperparameters_json = json.dumps(params, sort_keys=True)
    run_config_id = config_id(model_type, mode, feature_family_name, params, feature_policy)
    current_feature_set_id = feature_set_id(feature_family_name, mode, feature_cols, feature_policy)

    predictions = []
    model_runs = []
    feature_importance_rows = []
    feature_set_rows = [
        feature_set_row(
            experiment_id,
            current_feature_set_id,
            feature_family_name,
            mode,
            feature_policy,
            feature_cols,
        )
    ]
    cached_model = None
    cached_train_date = None
    cached_feature_cols = None
    cached_policy_result = None

    for _, eval_row in evaluation_frame.iterrows():
        as_of_date = eval_row["date"]
        train_df = full_table[full_table["date"] < as_of_date].copy()
        required_cols = [target_col, "seasonal_naive_proxy"] + feature_cols
        train_df = train_df.dropna(subset=required_cols)

        if len(train_df) < min_train_rows:
            continue

        started = time.perf_counter()
        model = None
        selected_feature_cols = feature_cols
        policy_result = {
            "selected_features": list(feature_cols),
            "dropped_features": [],
            "policy_params": {},
            "n_features_before_policy": len(feature_cols),
            "n_features_after_policy": len(feature_cols),
        }
        if model_type != "naive":
            should_fit = True
            if model_type == "xgboost" and cached_model is not None:
                refresh_months = refit_frequency_months or xgb_refresh_months
                months_since_fit = months_between(cached_train_date, as_of_date)
                should_fit = months_since_fit >= refresh_months
            elif cached_model is not None and refit_frequency_months is not None:
                months_since_fit = months_between(cached_train_date, as_of_date)
                should_fit = months_since_fit >= refit_frequency_months

            if should_fit:
                if mode == "residual":
                    y_train = train_df[target_col] - train_df["seasonal_naive_proxy"]
                else:
                    y_train = train_df[target_col]
                policy_result = apply_feature_policy(train_df, feature_cols, y_train, feature_policy)
                selected_feature_cols = policy_result["selected_features"]
                X_train = train_df[selected_feature_cols].astype(float)

                model = build_model(model_type, params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(X_train, y_train)
                cached_model = model
                cached_train_date = as_of_date
                cached_feature_cols = selected_feature_cols
                cached_policy_result = policy_result
            else:
                model = cached_model
                selected_feature_cols = cached_feature_cols or feature_cols
                policy_result = cached_policy_result or policy_result

        train_seconds = time.perf_counter() - started
        pred = prediction_for_row(model, model_type, mode, eval_row, selected_feature_cols)
        actual = float(eval_row[target_col])
        naive = float(eval_row["seasonal_naive_proxy"])
        error = pred - actual
        target_date = eval_row["target_date"]
        evaluation_period = evaluation_period_for(target_date)
        shock_period_flag = is_shock_period(target_date)

        model_run_id = f"{run_config_id}__as_of_{as_of_date.date().isoformat()}"
        prediction_id = model_run_id
        model_refit = bool(model_type == "naive" or cached_train_date is None or cached_train_date == as_of_date)
        row_common = {
            "experiment_id": experiment_id,
            "pipeline_run_id": pipeline_run_id,
            "model_run_id": model_run_id,
            "model_config_id": run_config_id,
            "prediction_id": prediction_id,
            "config_id": run_config_id,
            "as_of_date": as_of_date.date().isoformat(),
            "target": target,
            "horizon": horizon,
            "model_family": model_family,
            "model_build": model_build,
            "model_type": model_type,
            "ensemble_method": ensemble_method,
            "mode": mode,
            "feature_family_name": feature_family_name,
            "feature_policy": feature_policy,
            "feature_set_id": current_feature_set_id,
            "n_features": len(selected_feature_cols),
            "n_features_before_policy": int(policy_result["n_features_before_policy"]),
            "n_features_after_policy": int(policy_result["n_features_after_policy"]),
            "representation_policy": representation_metadata["representation_policy"],
            "n_representation_features": len(selected_feature_cols),
            "sequence_length": representation_metadata["sequence_length"],
            "sequence_stride": representation_metadata["sequence_stride"],
            "prediction_head": representation_metadata["prediction_head"],
            "n_train": int(len(train_df)),
        }
        predictions.append(
            {
                **row_common,
                "target_date": target_date.date().isoformat(),
                "actual": actual,
                "prediction": pred,
                "baseline_prediction": naive,
                "seasonal_naive_prediction": naive,
                "model_refit": model_refit,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error**2,
                "ape": safe_ape(actual, pred),
                "naive_error": naive - actual,
                "naive_abs_error": abs(naive - actual),
                "evaluation_period": evaluation_period,
                "shock_period_flag": shock_period_flag,
                "train_seconds": train_seconds,
            }
        )

        model_runs.append(
            {
                **row_common,
                "params": hyperparameters_json,
                "hyperparameters_json": hyperparameters_json,
                "selected_feature_names_json": json.dumps(selected_feature_cols, sort_keys=True),
                "dropped_feature_names_json": json.dumps(policy_result["dropped_features"], sort_keys=True),
                "feature_policy_params_json": json.dumps(policy_result["policy_params"], sort_keys=True),
                "representation_policy": representation_metadata["representation_policy"],
                "representation_params_json": representation_metadata["representation_params_json"],
                "n_representation_features": len(selected_feature_cols),
                "sequence_length": representation_metadata["sequence_length"],
                "sequence_stride": representation_metadata["sequence_stride"],
                "prediction_head": representation_metadata["prediction_head"],
                "training_window_months": representation_metadata["training_window_months"],
                "validation_strategy": representation_metadata["validation_strategy"],
                "early_stopping_used": representation_metadata["early_stopping_used"],
                "epochs_trained": representation_metadata["epochs_trained"],
                "best_epoch": representation_metadata["best_epoch"],
                "framework": framework_for(model_type),
                "framework_version": "",
                "hardware_type": representation_metadata["hardware_type"],
                "device": representation_metadata["device"],
                "gpu_name": representation_metadata["gpu_name"],
                "cuda_version": representation_metadata["cuda_version"],
                "refit_frequency_months": (
                    refit_frequency_months
                    if refit_frequency_months is not None
                    else (xgb_refresh_months if model_type == "xgboost" else 1)
                ),
                "model_refit": model_refit,
                "train_seconds": train_seconds,
                "predict_seconds": 0.0,
                "status": "succeeded",
                "artifact_uri": "",
            }
        )

        if model is not None:
            for importance_rank, importance_row in enumerate(
                extract_feature_importance(model, model_type, selected_feature_cols),
                start=1,
            ):
                importance_row.update(
                    {
                        "experiment_id": experiment_id,
                        "pipeline_run_id": pipeline_run_id,
                        "model_run_id": model_run_id,
                        "model_config_id": run_config_id,
                        "prediction_id": prediction_id,
                        "config_id": run_config_id,
                        "as_of_date": as_of_date.date().isoformat(),
                        "model_family": model_family,
                        "model_build": model_build,
                        "model_type": model_type,
                        "mode": mode,
                        "feature_family_name": feature_family_name,
                        "feature_policy": feature_policy,
                        "feature_set_id": current_feature_set_id,
                        "rank": importance_rank,
                    }
                )
                feature_importance_rows.append(importance_row)

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(model_runs),
        pd.DataFrame(feature_importance_rows),
        pd.DataFrame(feature_set_rows),
    )


def task_identifier(task: dict) -> str:
    return config_id(
        task["config"]["model_type"],
        task["mode"],
        task["feature_family_name"],
        task["config"]["params"],
        task["feature_policy"],
    )


def chunk_folder(chunk_dir: str | Path, task_id: str) -> Path:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:12]
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in task_id[:80])
    return Path(chunk_dir) / f"{safe_prefix}__{digest}"


def chunk_is_complete(chunk_dir: str | Path, task_id: str) -> bool:
    folder = chunk_folder(chunk_dir, task_id)
    return (folder / "predictions.parquet").exists() and (folder / "model_runs.parquet").exists()


def write_chunk_result(chunk_dir: str | Path, task_id: str, chunk: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    folder = chunk_folder(chunk_dir, task_id)
    folder.mkdir(parents=True, exist_ok=True)
    names = ["predictions", "model_runs", "feature_importance", "feature_sets"]
    for name, df in zip(names, chunk):
        if not df.empty:
            df.to_parquet(folder / f"{name}.parquet", index=False)
    write_json_uri(str(folder / "chunk_manifest.json"), {"task_id": task_id, "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def read_chunk_result(chunk_dir: str | Path, task_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = chunk_folder(chunk_dir, task_id)
    frames = []
    for name in ["predictions", "model_runs", "feature_importance", "feature_sets"]:
        path = folder / f"{name}.parquet"
        frames.append(pd.read_parquet(path) if path.exists() else pd.DataFrame())
    return tuple(frames)


def append_checkpoint_rows(checkpoint_dir: str | Path, filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    folder = Path(checkpoint_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    new_rows = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        new_rows = pd.concat([existing, new_rows], ignore_index=True)
    if "task_id" in new_rows.columns:
        new_rows = new_rows.drop_duplicates(subset=["task_id"], keep="last")
    new_rows.to_parquet(path, index=False)


def run_model_comparison(
    feature_table: pd.DataFrame,
    feature_families: dict,
    evaluation_frame: pd.DataFrame,
    experiment_id: str,
    pipeline_run_id: str | None,
    target_col: str,
    target: str,
    horizon: int,
    min_train_rows: int,
    xgb_refresh_months: int,
    refit_frequency_months: int | None,
    include_model_types: set[str] | None = None,
    feature_policies: list[str] | None = None,
    model_grid: list[dict] | None = None,
    n_jobs: int = 1,
    chunk_dir: str | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_table = add_seasonal_naive_proxy(feature_table, target_col=target_col, seasonal_periods=12)
    full_table["date"] = pd.to_datetime(full_table["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    full_table = full_table.sort_values("date").reset_index(drop=True)

    configs = [
        config
        for config in model_param_grid(model_grid)
        if include_model_types is None or config["model_type"] in include_model_types
    ]
    if not configs:
        raise ValueError("No model configurations remain after applying --include-model-type filters.")
    requested_policies = feature_policies or ["none"]

    tasks = []
    naive_configs = [config for config in configs if config["model_type"] == "naive"]
    for config in naive_configs:
        tasks.append(
            {
                "full_table": full_table,
                "evaluation_frame": evaluation_frame,
                "feature_cols": [],
                "config": config,
                "mode": "raw",
                "feature_family_name": "baseline_naive",
                "feature_policy": "none",
                "experiment_id": experiment_id,
                "pipeline_run_id": pipeline_run_id,
                "target_col": target_col,
                "target": target,
                "horizon": horizon,
                "min_train_rows": min_train_rows,
                "xgb_refresh_months": xgb_refresh_months,
                "refit_frequency_months": refit_frequency_months,
            }
        )

    model_configs = [config for config in configs if config["model_type"] != "naive"]
    for feature_family_name in feature_families:
        feature_cols = eligible_feature_columns(feature_families, feature_family_name, full_table)
        if not feature_cols:
            logger.warning("Skipping feature family with no available columns: %s", feature_family_name)
            continue

        for config in model_configs:
            model_type = config["model_type"]
            for mode in MODEL_MODES:
                for requested_policy in requested_policies:
                    policy = effective_feature_policy(model_type, requested_policy)
                    if any(
                        task["feature_family_name"] == feature_family_name
                        and task["config"] == config
                        and task["mode"] == mode
                        and task["feature_policy"] == policy
                        for task in tasks
                    ):
                        continue
                    tasks.append(
                        {
                            "full_table": full_table,
                            "evaluation_frame": evaluation_frame,
                            "feature_cols": feature_cols,
                            "config": config,
                            "mode": mode,
                            "feature_family_name": feature_family_name,
                            "feature_policy": policy,
                            "experiment_id": experiment_id,
                            "pipeline_run_id": pipeline_run_id,
                            "target_col": target_col,
                            "target": target,
                            "horizon": horizon,
                            "min_train_rows": min_train_rows,
                            "xgb_refresh_months": xgb_refresh_months,
                            "refit_frequency_months": refit_frequency_months,
                        }
                    )

    logger.info("Prepared %s model configuration tasks.", len(tasks))
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    chunks = []
    tasks_to_run = tasks
    failed_rows = []
    completed_rows = []
    if chunk_dir:
        for task in tasks:
            task["task_id"] = task_identifier(task)
        if resume:
            existing_tasks = [task for task in tasks if chunk_is_complete(chunk_dir, task["task_id"])]
            tasks_to_run = [task for task in tasks if not chunk_is_complete(chunk_dir, task["task_id"])]
            if existing_tasks:
                logger.info("Resuming from %s completed model configuration chunks.", len(existing_tasks))
                chunks.extend(read_chunk_result(chunk_dir, task["task_id"]) for task in existing_tasks)
                completed_rows.extend(
                    {
                        "task_id": task["task_id"],
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "source": "existing_chunk",
                    }
                    for task in existing_tasks
                )
        logger.info("Chunk checkpoint directory: %s", chunk_dir)
    if checkpoint_dir:
        logger.info("Run checkpoint directory: %s", checkpoint_dir)
        if completed_rows:
            append_checkpoint_rows(checkpoint_dir, "completed_configs.parquet", completed_rows)

    if n_jobs and n_jobs > 1 and len(tasks_to_run) > 1:
        logger.info("Running model configuration tasks with %s parallel workers.", n_jobs)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {executor.submit(run_config_evaluation, task): task for task in tasks_to_run}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get("task_id") or task_identifier(task)
                try:
                    chunk = future.result()
                    chunks.append(chunk)
                    if chunk_dir:
                        write_chunk_result(chunk_dir, task_id, chunk)
                    completed_rows.append({"task_id": task_id, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
                    if checkpoint_dir:
                        append_checkpoint_rows(checkpoint_dir, "completed_configs.parquet", completed_rows)
                except Exception as exc:
                    failed_rows.append({"task_id": task_id, "error": repr(exc), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
                    if checkpoint_dir:
                        append_checkpoint_rows(checkpoint_dir, "failed_configs.parquet", failed_rows)
                    raise RuntimeError(f"Model configuration failed: {task_id}") from exc
    else:
        for task in tasks_to_run:
            task_id = task.get("task_id") or task_identifier(task)
            try:
                chunk = run_config_evaluation(task)
                chunks.append(chunk)
                if chunk_dir:
                    write_chunk_result(chunk_dir, task_id, chunk)
                completed_rows.append({"task_id": task_id, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
                if checkpoint_dir:
                    append_checkpoint_rows(checkpoint_dir, "completed_configs.parquet", completed_rows)
            except Exception as exc:
                failed_rows.append({"task_id": task_id, "error": repr(exc), "failed_at_utc": datetime.now(timezone.utc).isoformat()})
                if checkpoint_dir:
                    append_checkpoint_rows(checkpoint_dir, "failed_configs.parquet", failed_rows)
                raise RuntimeError(f"Model configuration failed: {task_id}") from exc

    predictions_frames = [chunk[0] for chunk in chunks if not chunk[0].empty]
    model_run_frames = [chunk[1] for chunk in chunks if not chunk[1].empty]
    importance_frames = [chunk[2] for chunk in chunks if not chunk[2].empty]
    feature_set_frames = [chunk[3] for chunk in chunks if not chunk[3].empty]

    predictions = pd.concat(predictions_frames, ignore_index=True) if predictions_frames else pd.DataFrame()
    model_runs = pd.concat(model_run_frames, ignore_index=True) if model_run_frames else pd.DataFrame()
    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    feature_sets = (
        pd.concat(feature_set_frames, ignore_index=True).drop_duplicates(subset=["feature_set_id"])
        if feature_set_frames
        else pd.DataFrame()
    )

    return predictions, model_runs, feature_importance, feature_sets


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "config_id",
        "model_config_id",
        "model_family",
        "model_build",
        "model_type",
        "ensemble_method",
        "mode",
        "feature_family_name",
        "feature_policy",
        "feature_set_id",
    ]

    def append_metric_row(keys, group: pd.DataFrame, evaluation_scope: str) -> None:
        (
            config_id_value,
            model_config_id,
            model_family,
            model_build,
            model_type,
            ensemble_method,
            mode,
            feature_family_name,
            feature_policy,
            feature_set_id_value,
        ) = keys
        if group.empty:
            return
        actual = group["actual"].to_numpy()
        pred = group["prediction"].to_numpy()
        naive = group["seasonal_naive_prediction"].to_numpy()
        mae = mean_absolute_error(actual, pred)
        rmse = mean_squared_error(actual, pred) ** 0.5
        mae_naive = mean_absolute_error(actual, naive)
        rmse_naive = mean_squared_error(actual, naive) ** 0.5
        r2 = float(r2_score(actual, pred)) if len(group) > 1 else np.nan
        n_features = int(group["n_features"].max())
        r2_adjusted = adjusted_r2_score_value(r2, len(group), n_features, model_family)
        metric_row = {
            "experiment_id": group["experiment_id"].iloc[0],
            "pipeline_run_id": group["pipeline_run_id"].iloc[0],
            "model_config_id": model_config_id,
            "config_id": config_id_value,
            "evaluation_scope": evaluation_scope,
            "evaluation_start_date": group["target_date"].min(),
            "evaluation_end_date": group["target_date"].max(),
            "target": group["target"].iloc[0],
            "horizon": int(group["horizon"].iloc[0]),
            "model_family": model_family,
            "model_build": model_build,
            "model_type": model_type,
            "ensemble_method": ensemble_method,
            "mode": mode,
            "feature_family_name": feature_family_name,
            "feature_policy": feature_policy,
            "feature_set_id": feature_set_id_value,
            "n_predictions": int(len(group)),
            "n_features": n_features,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": r2,
            "r2_adjusted": r2_adjusted,
            "diracc": directional_accuracy(actual, pred),
            "mae_naive": float(mae_naive),
            "rmse_naive": float(rmse_naive),
            "mae_improvement_vs_naive": float(mae_naive - mae),
            "rmse_improvement_vs_naive": float(rmse_naive - rmse),
            "rank": 0,
            "metric_extras_json": "{}",
            "avg_train_seconds": float(group["train_seconds"].mean()),
            "total_train_seconds": float(group["train_seconds"].sum()),
        }
        for recipe_name, recipe in SCORE_RECIPES.items():
            metric_row[f"selection_score_{recipe_name}"] = float(
                recipe["mae_weight"] * mae + recipe["rmse_weight"] * rmse
            )
        metric_row["selection_score"] = metric_row["selection_score_balanced"]
        rows.append(metric_row)

    for keys, group in predictions.groupby(group_cols):
        append_metric_row(keys, group, "overall")
        for evaluation_scope in EVALUATION_SCOPES:
            if evaluation_scope == "overall":
                continue
            period_group = group[group["evaluation_period"] == evaluation_scope]
            append_metric_row(keys, period_group, evaluation_scope)

    metrics = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)
    if not metrics.empty:
        metrics["rank"] = (
            metrics.groupby("evaluation_scope")["selection_score"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
    return metrics


def build_family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics[metrics["evaluation_scope"] == "overall"].copy()
    return (
        metrics.groupby(["feature_family_name", "mode"], dropna=False)
        .agg(
            best_selection_score=("selection_score", "min"),
            best_selection_score_typical=("selection_score_typical", "min"),
            best_selection_score_balanced=("selection_score_balanced", "min"),
            best_selection_score_large_error=("selection_score_large_error", "min"),
            best_rmse=("rmse", "min"),
            best_mae=("mae", "min"),
            best_r2=("r2", "max"),
            best_r2_adjusted=("r2_adjusted", "max"),
            best_diracc=("diracc", "max"),
            best_rmse_improvement_vs_naive=("rmse_improvement_vs_naive", "max"),
            best_mae_improvement_vs_naive=("mae_improvement_vs_naive", "max"),
            avg_rmse=("rmse", "mean"),
            avg_mae=("mae", "mean"),
        )
        .sort_values("best_selection_score")
        .reset_index()
    )


def select_champion(metrics: pd.DataFrame) -> dict:
    if metrics.empty:
        raise ValueError("Cannot select champion from empty metrics.")

    metrics = metrics[metrics["evaluation_scope"] == "overall"].copy()
    if metrics.empty:
        raise ValueError("Cannot select champion without overall metric rows.")

    best_score = metrics["selection_score"].min()
    threshold = best_score * (1 + SIMPLICITY_THRESHOLD)
    candidates = metrics[metrics["selection_score"] <= threshold].copy()
    candidates["model_order"] = candidates["model_type"].map(MODEL_ORDER).fillna(99)
    candidates = candidates.sort_values(
        ["model_order", "n_features", "selection_score", "mae", "rmse"]
    ).reset_index(drop=True)

    champion = candidates.iloc[0].to_dict()
    champion["selection_rule"] = (
        "Choose the simplest model within 2% of the best balanced weighted score, "
        "where balanced score = 0.75 * MAE + 0.25 * RMSE. The dashboard also "
        "stores typical-error and large-error score variants for alternate rankings."
    )
    champion["best_raw_selection_score"] = float(best_score)
    champion["equivalence_threshold"] = float(threshold)
    return champion


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def safe_json_loads(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def numeric_param(params: dict, name: str, default: float = 0.0) -> float:
    value = params.get(name, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def order_sum(params: dict, name: str) -> float:
    values = params.get(name) or []
    if not isinstance(values, (list, tuple)):
        return 0.0
    return float(sum(abs(int(value)) for value in values if value is not None))


def model_size_proxy(row: pd.Series) -> float:
    params = safe_json_loads(row.get("hyperparameters_json"))
    model_type = row.get("model_type")
    n_features = numeric_param(row, "n_selected_features", numeric_param(row, "n_features", 0.0))
    if model_type == "naive":
        return 1.0
    if model_type in {"ridge", "lasso", "elastic_net"}:
        return max(1.0, n_features)
    if model_type in {"random_forest", "extra_trees"}:
        depth = numeric_param(params, "max_depth", 12.0)
        estimators = numeric_param(params, "n_estimators", 1.0)
        return max(1.0, estimators * max(1.0, depth))
    if model_type == "xgboost":
        depth = numeric_param(params, "max_depth", 1.0)
        estimators = numeric_param(params, "n_estimators", 1.0)
        colsample = numeric_param(params, "colsample_bytree", 1.0)
        return max(1.0, estimators * max(1.0, depth) * max(0.1, colsample))
    if model_type in {"arima", "sarima", "sarimax"}:
        return max(1.0, order_sum(params, "order") + order_sum(params, "seasonal_order") + n_features)
    if model_type == "mlp":
        sequence_length = numeric_param(params, "sequence_length", 1.0)
        hidden_size = numeric_param(params, "hidden_size", 1.0)
        return max(1.0, sequence_length * n_features * hidden_size + hidden_size)
    if model_type in {"rnn", "gru", "lstm"}:
        hidden_size = numeric_param(params, "hidden_size", 1.0)
        num_layers = numeric_param(params, "num_layers", 1.0)
        gate_count = {"rnn": 1.0, "gru": 3.0, "lstm": 4.0}[model_type]
        first_layer = gate_count * (n_features * hidden_size + hidden_size * hidden_size + hidden_size)
        later_layers = max(0.0, num_layers - 1.0) * gate_count * (2.0 * hidden_size**2 + hidden_size)
        return max(1.0, first_layer + later_layers + hidden_size)
    return max(1.0, n_features)


def normalize_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric.empty:
        return numeric
    low = numeric.min()
    high = numeric.max()
    if high == low:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - low) / (high - low)


def base_interpretability(row: pd.Series) -> float:
    family = row.get("model_family")
    if family == "baseline":
        return 100.0
    if family == "linear":
        return 86.0
    if family == "autoregressive":
        return 76.0
    if family == "tree":
        return 58.0
    return 50.0


def build_complexity_profile(model_runs: pd.DataFrame, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    if model_runs.empty:
        return pd.DataFrame()

    group_cols = [
        "model_config_id",
        "config_id",
        "model_family",
        "model_build",
        "model_type",
        "ensemble_method",
        "mode",
        "feature_family_name",
        "feature_policy",
        "feature_set_id",
    ]
    work = model_runs.copy()
    if "n_features_before_policy" not in work:
        work["n_features_before_policy"] = work["n_features"]
    if "n_features_after_policy" not in work:
        work["n_features_after_policy"] = work["n_features"]
    if "feature_policy_params_json" not in work:
        work["feature_policy_params_json"] = "{}"
    if "dropped_feature_names_json" not in work:
        work["dropped_feature_names_json"] = "[]"
    if "aic" not in work:
        work["aic"] = np.nan
    if "bic" not in work:
        work["bic"] = np.nan
    default_columns = {
        "representation_policy": "tabular_raw",
        "representation_params_json": "{}",
        "n_representation_features": np.nan,
        "sequence_length": np.nan,
        "sequence_stride": np.nan,
        "prediction_head": "direct_horizon",
        "training_window_months": np.nan,
        "validation_strategy": "rolling_as_of",
        "early_stopping_used": False,
        "epochs_trained": np.nan,
        "best_epoch": np.nan,
        "framework": "",
        "framework_version": "",
        "hardware_type": "cpu",
        "device": "cpu",
        "gpu_name": "",
        "cuda_version": "",
    }
    for column, default_value in default_columns.items():
        if column not in work:
            work[column] = default_value

    profile = (
        work.groupby(group_cols, dropna=False)
        .agg(
            experiment_id=("experiment_id", "first"),
            pipeline_run_id=("pipeline_run_id", "first"),
            hyperparameters_json=("hyperparameters_json", "first"),
            feature_policy_params_json=("feature_policy_params_json", "first"),
            representation_policy=("representation_policy", "first"),
            representation_params_json=("representation_params_json", "first"),
            n_representation_features=("n_representation_features", "mean"),
            sequence_length=("sequence_length", "first"),
            sequence_stride=("sequence_stride", "first"),
            prediction_head=("prediction_head", "first"),
            training_window_months=("training_window_months", "first"),
            validation_strategy=("validation_strategy", "first"),
            early_stopping_used=("early_stopping_used", "first"),
            epochs_trained=("epochs_trained", "max"),
            best_epoch=("best_epoch", "min"),
            framework=("framework", "first"),
            framework_version=("framework_version", "first"),
            hardware_type=("hardware_type", "first"),
            device=("device", "first"),
            gpu_name=("gpu_name", "first"),
            cuda_version=("cuda_version", "first"),
            n_input_features=("n_features_before_policy", "max"),
            n_selected_features=("n_features_after_policy", "mean"),
            min_selected_features=("n_features_after_policy", "min"),
            max_selected_features=("n_features_after_policy", "max"),
            avg_train_seconds=("train_seconds", "mean"),
            total_train_seconds=("train_seconds", "sum"),
            model_run_count=("model_run_id", "nunique"),
            refit_count=("model_refit", "sum"),
            avg_n_train=("n_train", "mean"),
            aic_mean=("aic", "mean"),
            bic_mean=("bic", "mean"),
            selected_feature_names_json=("selected_feature_names_json", "first"),
            dropped_feature_names_json=("dropped_feature_names_json", "first"),
        )
        .reset_index()
    )
    profile["n_selected_features"] = profile["n_selected_features"].round(3)
    profile["feature_reduction_ratio"] = profile.apply(
        lambda row: safe_ratio(
            row["n_input_features"] - row["n_selected_features"],
            row["n_input_features"],
        ),
        axis=1,
    ).fillna(0.0)
    profile["model_size_proxy"] = profile.apply(model_size_proxy, axis=1)
    profile["compute_proxy"] = profile["total_train_seconds"].clip(lower=0)
    feature_norm = normalize_series(profile["n_selected_features"])
    size_norm = normalize_series(np.log1p(profile["model_size_proxy"]))
    compute_norm = normalize_series(np.log1p(profile["compute_proxy"]))
    profile["complexity_score"] = (100 * (0.40 * feature_norm + 0.35 * size_norm + 0.25 * compute_norm)).round(3)
    profile["compute_score"] = (100 * compute_norm).round(3)
    profile["interpretability_score"] = profile.apply(base_interpretability, axis=1)
    profile["interpretability_score"] = (
        profile["interpretability_score"]
        - 0.25 * normalize_series(profile["n_selected_features"]) * 100
        - 0.15 * normalize_series(np.log1p(profile["model_size_proxy"])) * 100
    ).clip(lower=0, upper=100).round(3)

    if metrics is not None and not metrics.empty:
        overall_cols = [
            "model_config_id",
            "mae",
            "rmse",
            "r2",
            "r2_adjusted",
            "selection_score",
            "selection_score_typical",
            "selection_score_balanced",
            "selection_score_large_error",
        ]
        overall = metrics[metrics["evaluation_scope"] == "overall"][
            [col for col in overall_cols if col in metrics.columns]
        ].copy()
        overall = overall.rename(
            columns={
                "mae": "overall_mae",
                "rmse": "overall_rmse",
                "r2": "overall_r2",
                "r2_adjusted": "overall_r2_adjusted",
                "selection_score": "overall_selection_score",
                "selection_score_typical": "overall_selection_score_typical",
                "selection_score_balanced": "overall_selection_score_balanced",
                "selection_score_large_error": "overall_selection_score_large_error",
            }
        )
        profile = profile.merge(overall, on="model_config_id", how="left")
    return profile.sort_values(["complexity_score", "model_config_id"]).reset_index(drop=True)


def build_wide_leaderboard(metrics: pd.DataFrame, config_details: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "model_config_id",
        "config_id",
        "model_family",
        "model_build",
        "model_type",
        "ensemble_method",
        "mode",
        "feature_family_name",
        "feature_policy",
        "feature_set_id",
    ]
    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "r2_adjusted",
        "diracc",
        "selection_score",
        "selection_score_typical",
        "selection_score_balanced",
        "selection_score_large_error",
        "mae_improvement_vs_naive",
        "rmse_improvement_vs_naive",
        "n_predictions",
    ]
    available_metric_cols = [col for col in metric_cols if col in metrics.columns]
    wide = metrics.pivot_table(
        index=index_cols,
        columns="evaluation_scope",
        values=available_metric_cols,
        aggfunc="first",
    )
    wide.columns = [f"{scope}_{metric}" for metric, scope in wide.columns]
    wide = wide.reset_index()

    wide = wide.merge(
        config_details[["model_config_id", "hyperparameters_json"]],
        on="model_config_id",
        how="left",
    )

    if "overall_selection_score" in wide:
        wide["selection_score"] = wide["overall_selection_score"]
    if "overall_selection_score_typical" in wide:
        wide["selection_score_typical"] = wide["overall_selection_score_typical"]
    if "overall_selection_score_balanced" in wide:
        wide["selection_score_balanced"] = wide["overall_selection_score_balanced"]
    elif "selection_score" in wide:
        wide["selection_score_balanced"] = wide["selection_score"]
    if "overall_selection_score_large_error" in wide:
        wide["selection_score_large_error"] = wide["overall_selection_score_large_error"]
    if "overall_mae" in wide:
        wide["mae"] = wide["overall_mae"]
    if "overall_rmse" in wide:
        wide["rmse"] = wide["overall_rmse"]
    if "overall_r2" in wide:
        wide["r2"] = wide["overall_r2"]
    if "overall_r2_adjusted" in wide:
        wide["r2_adjusted"] = wide["overall_r2_adjusted"]
    elif "r2_adjusted" in metrics.columns:
        wide["r2_adjusted"] = np.nan
    if "overall_diracc" in wide:
        wide["diracc"] = wide["overall_diracc"]
    if "overall_n_predictions" in wide:
        wide["n_predictions"] = wide["overall_n_predictions"]

    wide["shock_penalty"] = wide.apply(
        lambda row: safe_ratio(row.get("covid_shock_mae"), row.get("pre_covid_mae")),
        axis=1,
    )
    wide["recovery_ratio"] = wide.apply(
        lambda row: safe_ratio(row.get("recovery_mae"), row.get("pre_covid_mae")),
        axis=1,
    )
    wide["recent_recovery_ratio"] = wide.apply(
        lambda row: safe_ratio(row.get("recent_mae"), row.get("pre_covid_mae")),
        axis=1,
    )
    wide["rmse_shock_penalty"] = wide.apply(
        lambda row: safe_ratio(row.get("covid_shock_rmse"), row.get("pre_covid_rmse")),
        axis=1,
    )
    wide["rmse_recovery_ratio"] = wide.apply(
        lambda row: safe_ratio(row.get("recovery_rmse"), row.get("pre_covid_rmse")),
        axis=1,
    )
    wide["rmse_recent_recovery_ratio"] = wide.apply(
        lambda row: safe_ratio(row.get("recent_rmse"), row.get("pre_covid_rmse")),
        axis=1,
    )
    for recipe_name in SCORE_RECIPES:
        prefix = f"{recipe_name}_score"
        wide[f"{prefix}_shock_penalty"] = wide.apply(
            lambda row, name=recipe_name: safe_ratio(
                row.get(f"covid_shock_selection_score_{name}"),
                row.get(f"pre_covid_selection_score_{name}"),
            ),
            axis=1,
        )
        wide[f"{prefix}_recovery_ratio"] = wide.apply(
            lambda row, name=recipe_name: safe_ratio(
                row.get(f"recovery_selection_score_{name}"),
                row.get(f"pre_covid_selection_score_{name}"),
            ),
            axis=1,
        )
        wide[f"{prefix}_recent_recovery_ratio"] = wide.apply(
            lambda row, name=recipe_name: safe_ratio(
                row.get(f"recent_selection_score_{name}"),
                row.get(f"pre_covid_selection_score_{name}"),
            ),
            axis=1,
        )
    wide["shock_abs_increase"] = wide.get("covid_shock_mae", np.nan) - wide.get("pre_covid_mae", np.nan)
    wide = wide.sort_values("selection_score").reset_index(drop=True)
    wide["rank"] = np.arange(1, len(wide) + 1)
    return wide


def build_dashboard_outputs(
    predictions: pd.DataFrame,
    model_runs: pd.DataFrame,
    metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    champion: dict,
    complexity_profile: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    champion_predictions = predictions[predictions["config_id"] == champion["config_id"]].copy()
    forecast_paths = predictions[
        [
            "config_id",
            "as_of_date",
            "target_date",
            "model_family",
            "model_build",
            "model_type",
            "ensemble_method",
            "mode",
            "feature_family_name",
            "feature_policy",
            "feature_set_id",
            "actual",
            "prediction",
            "baseline_prediction",
            "seasonal_naive_prediction",
            "model_refit",
            "error",
            "abs_error",
            "evaluation_period",
            "shock_period_flag",
        ]
    ].copy()

    performance_over_time = forecast_paths.copy()
    performance_over_time["as_of_date"] = pd.to_datetime(performance_over_time["as_of_date"])
    performance_over_time = performance_over_time.sort_values(["config_id", "as_of_date"])
    performance_over_time["rolling_3mo_mae"] = (
        performance_over_time.groupby("config_id")["abs_error"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    performance_over_time["rolling_6mo_mae"] = (
        performance_over_time.groupby("config_id")["abs_error"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    performance_over_time["rolling_12mo_mae"] = (
        performance_over_time.groupby("config_id")["abs_error"]
        .rolling(12, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    config_details = (
        model_runs[
            [
                "model_config_id",
                "config_id",
                "model_family",
                "model_build",
                "model_type",
                "ensemble_method",
                "mode",
                "feature_family_name",
                "feature_policy",
                "feature_set_id",
                "hyperparameters_json",
            ]
        ]
        .drop_duplicates(subset=["model_config_id"])
        .copy()
    )

    leaderboard = build_wide_leaderboard(metrics, config_details)
    if complexity_profile is None:
        complexity_profile = build_complexity_profile(model_runs, metrics)
    if not complexity_profile.empty:
        complexity_cols = [
            "model_config_id",
            "n_input_features",
            "n_selected_features",
            "feature_reduction_ratio",
            "representation_policy",
            "n_representation_features",
            "sequence_length",
            "prediction_head",
            "validation_strategy",
            "framework",
            "hardware_type",
            "device",
            "model_size_proxy",
            "complexity_score",
            "interpretability_score",
            "compute_score",
            "aic_mean",
            "bic_mean",
        ]
        available_complexity_cols = [col for col in complexity_cols if col in complexity_profile.columns]
        leaderboard = leaderboard.merge(
            complexity_profile[available_complexity_cols].drop_duplicates(subset=["model_config_id"]),
            on="model_config_id",
            how="left",
        )

    overview_top_models = leaderboard.sort_values("selection_score").head(5).copy()
    overview_top_models["rank"] = np.arange(1, len(overview_top_models) + 1)
    overview_prediction_paths = forecast_paths[
        forecast_paths["config_id"].isin(overview_top_models["config_id"])
    ].copy()
    overview_prediction_paths["rank"] = overview_prediction_paths["config_id"].map(
        overview_top_models.set_index("config_id")["rank"]
    )
    overview_prediction_paths["model_config_id"] = overview_prediction_paths["config_id"]
    overview_prediction_paths = overview_prediction_paths.sort_values(["rank", "target_date"])

    return {
        "forecast_paths.parquet": forecast_paths,
        "performance_over_time.parquet": performance_over_time,
        "model_leaderboard.parquet": leaderboard,
        "feature_family_summary.parquet": family_summary,
        "champion_predictions.parquet": champion_predictions,
        "overview_top_models.parquet": overview_top_models,
        "overview_prediction_paths.parquet": overview_prediction_paths,
        "complexity_profile.parquet": complexity_profile,
    }


def log_to_mlflow(
    args: argparse.Namespace,
    manifest: dict,
    champion: dict,
    metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
) -> None:
    if not args.enable_mlflow:
        return
    if mlflow is None:
        raise ImportError("MLflow logging was enabled, but the 'mlflow' package is not installed.")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    run_name = args.mlflow_run_name or manifest["run_id"]
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "experiment_id": manifest["run_id"],
                "pipeline_run_id": manifest["runtime"].get("pipeline_run_id") or "",
                "compute_context": manifest["runtime"].get("compute_context") or "",
                "image_uri": manifest["runtime"].get("image_uri") or "",
            }
        )
        mlflow.log_params(
            {
                "target": manifest["target"],
                "horizon": manifest["horizon"],
                "as_of_start": manifest["as_of_start"],
                "feature_family_count": manifest["feature_family_count"],
                "models": ",".join(manifest["models"]),
                "modes": ",".join(manifest["modes"]),
                "as_of_end": manifest.get("as_of_end") or "",
                "as_of_frequency_months": manifest["as_of_frequency_months"],
                "refit_frequency_months": manifest.get("refit_frequency_months") or "",
                "xgb_refresh_months": manifest["xgb_refresh_months"],
                "results_base_uri": manifest["results_base_uri"],
                "dashboard_base_uri": manifest["dashboard_base_uri"],
            }
        )
        mlflow.log_metrics(
            {
                "prediction_count": manifest["prediction_count"],
                "model_run_count": manifest["model_run_count"],
                "metric_count": manifest["metric_count"],
                "champion_mae": float(champion.get("mae", np.nan)),
                "champion_rmse": float(champion.get("rmse", np.nan)),
                "champion_selection_score": float(champion.get("selection_score", np.nan)),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "experiment_manifest.json"
            champion_path = temp_path / "champion_selection.json"
            leaderboard_path = temp_path / "leaderboard_top_25.csv"
            family_summary_path = temp_path / "feature_family_summary.csv"

            manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
            champion_path.write_text(json.dumps(champion, indent=2, default=str), encoding="utf-8")
            metrics.sort_values("selection_score").head(25).to_csv(leaderboard_path, index=False)
            family_summary.to_csv(family_summary_path, index=False)

            mlflow.log_artifact(str(manifest_path), artifact_path="metadata")
            mlflow.log_artifact(str(champion_path), artifact_path="metadata")
            mlflow.log_artifact(str(leaderboard_path), artifact_path="tables")
            mlflow.log_artifact(str(family_summary_path), artifact_path="tables")


def main() -> int:
    args = apply_experiment_config(parse_args())
    artifacts = resolve_feature_artifacts(args)

    feature_table = read_parquet_uri(artifacts["feature_table_uri"])
    feature_families = read_json_uri(artifacts["feature_families_uri"])
    if args.include_feature_family:
        requested_families = set(args.include_feature_family)
        missing_families = sorted(requested_families - set(feature_families))
        if missing_families:
            raise ValueError(f"Requested feature families not found: {missing_families}")
        feature_families = {
            name: columns
            for name, columns in feature_families.items()
            if name in requested_families
        }
    model_run_id = (
        args.experiment_config_payload.get("experiment_id")
        if args.experiment_config_payload
        else current_model_run_id(artifacts["feature_base_uri"])
    )
    target_col = validate_feature_table(feature_table, args.target, args.horizon)
    evaluation_frame = build_evaluation_frame(
        feature_table,
        target_col,
        args.as_of_start,
        args.as_of_end,
        args.as_of_frequency_months,
        args.horizon,
    )
    family_audit = summarize_feature_families(feature_families, feature_table)

    results_base_uri = resolve_output_base_uri(
        args.results_base_uri,
        args.bucket,
        args.results_prefix,
        model_run_id,
    )
    dashboard_base_uri = resolve_output_base_uri(
        args.dashboard_base_uri,
        args.bucket,
        args.dashboard_prefix,
        model_run_id,
    )

    logger.info("Loaded feature table: %s rows, %s columns", len(feature_table), len(feature_table.columns))
    logger.info("Loaded feature families: %s", len(feature_families))
    logger.info(
        "Evaluation frame: %s rows, as-of %s to %s, target dates %s to %s",
        len(evaluation_frame),
        evaluation_frame["date"].min().date(),
        evaluation_frame["date"].max().date(),
        evaluation_frame["target_date"].min().date(),
        evaluation_frame["target_date"].max().date(),
    )
    logger.info("Feature artifact base: %s", artifacts["feature_base_uri"])
    logger.info("Model results will write under: %s", results_base_uri)
    logger.info("Dashboard artifacts will write under: %s", dashboard_base_uri)
    logger.info("Feature family audit:\n%s", family_audit.to_string(index=False))

    pipeline_run_id = os.environ.get("PIPELINE_RUN_ID")
    predictions, model_runs, feature_importance, feature_sets = run_model_comparison(
        feature_table=feature_table,
        feature_families=feature_families,
        evaluation_frame=evaluation_frame,
        experiment_id=model_run_id,
        pipeline_run_id=pipeline_run_id,
        target_col=target_col,
        target=args.target,
        horizon=args.horizon,
        min_train_rows=args.min_train_rows,
        xgb_refresh_months=args.xgb_refresh_months,
        refit_frequency_months=args.refit_frequency_months,
        include_model_types=set(args.include_model_type) if args.include_model_type else None,
        feature_policies=args.feature_policy or ["none"],
        model_grid=args.model_grid,
        n_jobs=args.n_jobs,
        chunk_dir=args.chunk_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    if predictions.empty:
        raise ValueError("No predictions were produced. Check feature availability and min_train_rows.")

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

    write_parquet_uri(join_uri(results_base_uri, "predictions.parquet"), predictions)
    write_parquet_uri(join_uri(results_base_uri, "model_runs.parquet"), model_runs)
    write_parquet_uri(join_uri(results_base_uri, "metrics.parquet"), metrics)
    write_parquet_uri(join_uri(results_base_uri, "feature_importance.parquet"), feature_importance)
    write_parquet_uri(join_uri(results_base_uri, "feature_sets.parquet"), feature_sets)
    write_parquet_uri(join_uri(results_base_uri, "feature_family_summary.parquet"), family_summary)
    write_parquet_uri(join_uri(results_base_uri, "complexity_profile.parquet"), complexity_profile)
    write_json_uri(join_uri(results_base_uri, "champion_selection.json"), champion)

    manifest = {
        "run_id": model_run_id,
        "experiment_id": model_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_artifacts": artifacts,
        "results_base_uri": results_base_uri,
        "dashboard_base_uri": dashboard_base_uri,
        "target": args.target,
        "horizon": args.horizon,
        "as_of_start": args.as_of_start,
        "as_of_end": args.as_of_end,
        "as_of_frequency_months": int(args.as_of_frequency_months),
        "models": sorted(set(metrics["model_type"])),
        "modes": sorted(set(metrics["mode"])),
        "feature_policies": sorted(set(metrics["feature_policy"])) if "feature_policy" in metrics else ["none"],
        "experiment_config": args.experiment_config,
        "requested_feature_families": args.include_feature_family or "all",
        "requested_model_types": args.include_model_type or "all",
        "requested_feature_policies": args.feature_policy or ["none"],
        "n_jobs": int(args.n_jobs),
        "chunk_dir": args.chunk_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "resume": bool(args.resume),
        "feature_family_count": int(len(feature_families)),
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "metric_count": int(len(metrics)),
        "complexity_profile_count": int(len(complexity_profile)),
        "xgb_refresh_months": int(args.xgb_refresh_months),
        "refit_frequency_months": args.refit_frequency_months,
        "champion_config_id": champion["config_id"],
        "selection_rule": champion["selection_rule"],
        "runtime": {
            "compute_context": os.environ.get("COMPUTE_CONTEXT", "ecs" if os.environ.get("AWS_EXECUTION_ENV") else "local"),
            "pipeline_run_id": pipeline_run_id,
            "image_uri": os.environ.get("IMAGE_URI"),
            "code_version": os.environ.get("CODE_VERSION"),
            "step_function_execution_arn": os.environ.get("STEP_FUNCTION_EXECUTION_ARN"),
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
            "mlflow_experiment_name": args.mlflow_experiment_name if args.enable_mlflow else None,
        },
    }
    write_json_uri(join_uri(results_base_uri, "batch_manifest.json"), manifest)
    write_json_uri(join_uri(results_base_uri, "experiment_manifest.json"), manifest)

    log_to_mlflow(args, manifest, champion, metrics, family_summary)

    for filename, df in dashboard_outputs.items():
        write_parquet_uri(join_uri(dashboard_base_uri, filename), df)
    write_json_uri(join_uri(dashboard_base_uri, "champion_selection.json"), champion)
    write_json_uri(join_uri(dashboard_base_uri, "experiment_manifest.json"), manifest)

    logger.info("Produced %s predictions across %s model/as-of records.", len(predictions), len(model_runs))
    logger.info("Wrote model results to %s", results_base_uri)
    logger.info("Wrote dashboard artifacts to %s", dashboard_base_uri)
    logger.info("Champion: %s", json.dumps(champion, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
