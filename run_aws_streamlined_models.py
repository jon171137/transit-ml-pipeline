import argparse
import io
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

if load_dotenv:
    load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
FEATURE_OUTPUT_PREFIX = os.environ.get("FEATURE_OUTPUT_PREFIX", "features/integrated_monthly_h3")
MODEL_RESULTS_PREFIX = os.environ.get("MODEL_RESULTS_PREFIX", "model_results/aws_streamlined")
DASHBOARD_OUTPUT_PREFIX = os.environ.get("DASHBOARD_OUTPUT_PREFIX", "dashboard/aws_streamlined")

DEFAULT_TARGET = "upt"
DEFAULT_HORIZON = 3
DEFAULT_AS_OF_START = "2021-01-01"
FEATURE_TABLE_FILENAME = "feature_table.parquet"
FEATURE_FAMILIES_FILENAME = "feature_families.json"
MODEL_MODES = ["raw", "residual"]
MODEL_ORDER = {"naive": 0, "ridge": 1, "lasso": 2, "xgboost": 3}
SIMPLICITY_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the streamlined AWS modeling comparison across notebook-aligned feature "
            "families, raw/residual modes, and a compact naive/ridge/lasso/XGBoost grid."
        )
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
        default=12,
        help=(
            "How often XGBoost models are refit during rolling evaluation. "
            "Monthly predictions are still produced between refits."
        ),
    )
    return parser.parse_args()


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
    horizon: int,
) -> pd.DataFrame:
    eval_df = feature_table.copy()
    eval_df["date"] = pd.to_datetime(eval_df["date"])
    eval_df = eval_df.sort_values("date").reset_index(drop=True)
    eval_df["target_date"] = eval_df["date"] + pd.DateOffset(months=horizon)
    eval_df = add_seasonal_naive_proxy(eval_df, target_col=target_col, seasonal_periods=12)

    as_of_start_ts = pd.Timestamp(as_of_start)
    evaluable = eval_df[
        (eval_df["date"] >= as_of_start_ts)
        & eval_df[target_col].notna()
        & eval_df["seasonal_naive_proxy"].notna()
    ].copy()

    if evaluable.empty:
        raise ValueError(
            f"No evaluable rows found on or after {as_of_start!r} with non-null target and naive baseline."
        )

    return evaluable


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


def model_param_grid() -> list[dict]:
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
    if model_type == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            **params,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def eligible_feature_columns(feature_families: dict, family_name: str, feature_table: pd.DataFrame) -> list[str]:
    if family_name not in feature_families:
        raise KeyError(f"Unknown feature family: {family_name}")
    return [col for col in feature_families[family_name] if col in feature_table.columns]


def config_id(model_type: str, mode: str, feature_family_name: str, params: dict) -> str:
    if params:
        params_text = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
        return f"{model_type}__{mode}__{feature_family_name}__{params_text}"
    return f"{model_type}__{mode}__{feature_family_name}"


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
    if model_type in {"ridge", "lasso"}:
        coefs = model.named_steps["model"].coef_
        for feature_name, importance in zip(feature_cols, coefs):
            rows.append(
                {
                    "feature_name": feature_name,
                    "importance": float(importance),
                    "importance_abs": float(abs(importance)),
                }
            )
    elif model_type == "xgboost":
        for feature_name, importance in zip(feature_cols, model.feature_importances_):
            rows.append(
                {
                    "feature_name": feature_name,
                    "importance": float(importance),
                    "importance_abs": float(abs(importance)),
                }
            )
    return rows


def run_model_comparison(
    feature_table: pd.DataFrame,
    feature_families: dict,
    evaluation_frame: pd.DataFrame,
    target_col: str,
    target: str,
    horizon: int,
    min_train_rows: int,
    xgb_refresh_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_table = add_seasonal_naive_proxy(feature_table, target_col=target_col, seasonal_periods=12)
    full_table["date"] = pd.to_datetime(full_table["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    full_table = full_table.sort_values("date").reset_index(drop=True)

    predictions = []
    model_runs = []
    feature_importance_rows = []
    configs = model_param_grid()

    for feature_family_name in feature_families:
        feature_cols = eligible_feature_columns(feature_families, feature_family_name, full_table)
        if not feature_cols:
            logger.warning("Skipping feature family with no available columns: %s", feature_family_name)
            continue

        for config in configs:
            model_type = config["model_type"]
            params = config["params"]
            modes = ["raw"] if model_type == "naive" else MODEL_MODES

            for mode in modes:
                run_config_id = config_id(model_type, mode, feature_family_name, params)
                cached_model = None
                cached_train_date = None

                for _, eval_row in evaluation_frame.iterrows():
                    as_of_date = eval_row["date"]
                    train_df = full_table[full_table["date"] < as_of_date].copy()
                    required_cols = [target_col, "seasonal_naive_proxy"] + feature_cols
                    train_df = train_df.dropna(subset=required_cols)

                    if len(train_df) < min_train_rows:
                        continue

                    started = time.perf_counter()
                    model = None
                    if model_type != "naive":
                        should_fit = True
                        if model_type == "xgboost" and cached_model is not None:
                            months_since_fit = (
                                (as_of_date.year - cached_train_date.year) * 12
                                + (as_of_date.month - cached_train_date.month)
                            )
                            should_fit = months_since_fit >= xgb_refresh_months

                        if should_fit:
                            X_train = train_df[feature_cols].astype(float)
                            if mode == "residual":
                                y_train = train_df[target_col] - train_df["seasonal_naive_proxy"]
                            else:
                                y_train = train_df[target_col]

                            model = build_model(model_type, params)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", ConvergenceWarning)
                                model.fit(X_train, y_train)
                            cached_model = model
                            cached_train_date = as_of_date
                        else:
                            model = cached_model

                    train_seconds = time.perf_counter() - started
                    pred = prediction_for_row(model, model_type, mode, eval_row, feature_cols)
                    actual = float(eval_row[target_col])
                    naive = float(eval_row["seasonal_naive_proxy"])
                    error = pred - actual

                    prediction_id = f"{run_config_id}__as_of_{as_of_date.date().isoformat()}"
                    predictions.append(
                        {
                            "prediction_id": prediction_id,
                            "config_id": run_config_id,
                            "as_of_date": as_of_date.date().isoformat(),
                            "target_date": eval_row["target_date"].date().isoformat(),
                            "target": target,
                            "horizon": horizon,
                            "model_type": model_type,
                            "mode": mode,
                            "feature_family_name": feature_family_name,
                            "n_features": len(feature_cols),
                            "n_train": int(len(train_df)),
                            "actual": actual,
                            "prediction": pred,
                            "seasonal_naive_prediction": naive,
                            "model_refit": bool(
                                model_type == "naive"
                                or cached_train_date is None
                                or cached_train_date == as_of_date
                            ),
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error**2,
                            "naive_error": naive - actual,
                            "naive_abs_error": abs(naive - actual),
                            "train_seconds": train_seconds,
                        }
                    )

                    model_runs.append(
                        {
                            "prediction_id": prediction_id,
                            "config_id": run_config_id,
                            "as_of_date": as_of_date.date().isoformat(),
                            "model_type": model_type,
                            "mode": mode,
                            "feature_family_name": feature_family_name,
                            "params": json.dumps(params, sort_keys=True),
                            "n_features": len(feature_cols),
                            "n_train": int(len(train_df)),
                            "model_refit": bool(
                                model_type == "naive"
                                or cached_train_date is None
                                or cached_train_date == as_of_date
                            ),
                            "train_seconds": train_seconds,
                        }
                    )

                    if model is not None:
                        for importance_row in extract_feature_importance(model, model_type, feature_cols):
                            importance_row.update(
                                {
                                    "prediction_id": prediction_id,
                                    "config_id": run_config_id,
                                    "as_of_date": as_of_date.date().isoformat(),
                                    "model_type": model_type,
                                    "mode": mode,
                                    "feature_family_name": feature_family_name,
                                }
                            )
                            feature_importance_rows.append(importance_row)

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(model_runs),
        pd.DataFrame(feature_importance_rows),
    )


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["config_id", "model_type", "mode", "feature_family_name"]
    for keys, group in predictions.groupby(group_cols):
        config_id_value, model_type, mode, feature_family_name = keys
        actual = group["actual"].to_numpy()
        pred = group["prediction"].to_numpy()
        naive = group["seasonal_naive_prediction"].to_numpy()
        mae = mean_absolute_error(actual, pred)
        rmse = mean_squared_error(actual, pred) ** 0.5
        mae_naive = mean_absolute_error(actual, naive)
        rmse_naive = mean_squared_error(actual, naive) ** 0.5
        rows.append(
            {
                "config_id": config_id_value,
                "model_type": model_type,
                "mode": mode,
                "feature_family_name": feature_family_name,
                "n_predictions": int(len(group)),
                "n_features": int(group["n_features"].max()),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2_score(actual, pred)) if len(group) > 1 else np.nan,
                "diracc": directional_accuracy(actual, pred),
                "mae_naive": float(mae_naive),
                "rmse_naive": float(rmse_naive),
                "mae_improvement_vs_naive": float(mae_naive - mae),
                "rmse_improvement_vs_naive": float(rmse_naive - rmse),
                "selection_score": float(0.75 * mae + 0.25 * rmse),
                "avg_train_seconds": float(group["train_seconds"].mean()),
                "total_train_seconds": float(group["train_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)


def build_family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["feature_family_name", "mode"], dropna=False)
        .agg(
            best_selection_score=("selection_score", "min"),
            best_rmse=("rmse", "min"),
            best_mae=("mae", "min"),
            best_r2=("r2", "max"),
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

    best_score = metrics["selection_score"].min()
    threshold = best_score * (1 + SIMPLICITY_THRESHOLD)
    candidates = metrics[metrics["selection_score"] <= threshold].copy()
    candidates["model_order"] = candidates["model_type"].map(MODEL_ORDER).fillna(99)
    candidates = candidates.sort_values(
        ["model_order", "n_features", "selection_score", "mae", "rmse"]
    ).reset_index(drop=True)

    champion = candidates.iloc[0].to_dict()
    champion["selection_rule"] = (
        "Choose the simplest model within 2% of the best weighted score, "
        "where score = 0.75 * MAE + 0.25 * RMSE."
    )
    champion["best_raw_selection_score"] = float(best_score)
    champion["equivalence_threshold"] = float(threshold)
    return champion


def build_dashboard_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    family_summary: pd.DataFrame,
    champion: dict,
) -> dict[str, pd.DataFrame]:
    champion_predictions = predictions[predictions["config_id"] == champion["config_id"]].copy()
    forecast_paths = predictions[
        [
            "config_id",
            "as_of_date",
            "target_date",
            "model_type",
            "mode",
            "feature_family_name",
            "actual",
            "prediction",
            "seasonal_naive_prediction",
            "error",
            "abs_error",
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

    leaderboard = metrics.sort_values("selection_score").reset_index(drop=True)
    leaderboard["rank"] = np.arange(1, len(leaderboard) + 1)

    return {
        "forecast_paths.parquet": forecast_paths,
        "performance_over_time.parquet": performance_over_time,
        "model_leaderboard.parquet": leaderboard,
        "feature_family_summary.parquet": family_summary,
        "champion_predictions.parquet": champion_predictions,
    }


def main() -> int:
    args = parse_args()
    artifacts = resolve_feature_artifacts(args)

    feature_table = read_parquet_uri(artifacts["feature_table_uri"])
    feature_families = read_json_uri(artifacts["feature_families_uri"])
    model_run_id = current_model_run_id(artifacts["feature_base_uri"])
    target_col = validate_feature_table(feature_table, args.target, args.horizon)
    evaluation_frame = build_evaluation_frame(feature_table, target_col, args.as_of_start, args.horizon)
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

    predictions, model_runs, feature_importance = run_model_comparison(
        feature_table=feature_table,
        feature_families=feature_families,
        evaluation_frame=evaluation_frame,
        target_col=target_col,
        target=args.target,
        horizon=args.horizon,
        min_train_rows=args.min_train_rows,
        xgb_refresh_months=args.xgb_refresh_months,
    )
    if predictions.empty:
        raise ValueError("No predictions were produced. Check feature availability and min_train_rows.")

    metrics = calculate_metrics(predictions)
    family_summary = build_family_summary(metrics)
    champion = select_champion(metrics)
    dashboard_outputs = build_dashboard_outputs(predictions, metrics, family_summary, champion)

    write_parquet_uri(join_uri(results_base_uri, "predictions.parquet"), predictions)
    write_parquet_uri(join_uri(results_base_uri, "model_runs.parquet"), model_runs)
    write_parquet_uri(join_uri(results_base_uri, "metrics.parquet"), metrics)
    write_parquet_uri(join_uri(results_base_uri, "feature_importance.parquet"), feature_importance)
    write_parquet_uri(join_uri(results_base_uri, "feature_family_summary.parquet"), family_summary)
    write_json_uri(join_uri(results_base_uri, "champion_selection.json"), champion)

    manifest = {
        "run_id": model_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_artifacts": artifacts,
        "results_base_uri": results_base_uri,
        "dashboard_base_uri": dashboard_base_uri,
        "target": args.target,
        "horizon": args.horizon,
        "as_of_start": args.as_of_start,
        "models": sorted(set(metrics["model_type"])),
        "modes": sorted(set(metrics["mode"])),
        "feature_family_count": int(len(feature_families)),
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "xgb_refresh_months": int(args.xgb_refresh_months),
        "champion_config_id": champion["config_id"],
        "runtime": {
            "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
            "image_uri": os.environ.get("IMAGE_URI"),
        },
    }
    write_json_uri(join_uri(results_base_uri, "batch_manifest.json"), manifest)

    for filename, df in dashboard_outputs.items():
        write_parquet_uri(join_uri(dashboard_base_uri, filename), df)
    write_json_uri(join_uri(dashboard_base_uri, "champion_selection.json"), champion)

    logger.info("Produced %s predictions across %s model/as-of records.", len(predictions), len(model_runs))
    logger.info("Wrote model results to %s", results_base_uri)
    logger.info("Wrote dashboard artifacts to %s", dashboard_base_uri)
    logger.info("Champion: %s", json.dumps(champion, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
