"""Run Phase B autoregressive transit forecasting experiments.

This runner complements ``run_aws_streamlined_models.py``. It evaluates
ARIMA/SARIMA/SARIMAX-style models on the same rolling as-of frame and writes the
same Parquet/JSON artifact contract so the dashboard and DuckDB mart can read
the results beside Phase A linear/tree experiments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None

from run_aws_streamlined_models import (
    add_seasonal_naive_proxy,
    append_checkpoint_rows,
    build_dashboard_outputs,
    build_complexity_profile,
    build_evaluation_frame,
    build_family_summary,
    chunk_is_complete,
    config_id,
    current_model_run_id,
    default_representation_metadata,
    evaluation_period_for,
    feature_set_id,
    feature_set_row,
    is_shock_period,
    join_uri,
    log_to_mlflow,
    months_between,
    read_chunk_result,
    read_json_uri,
    read_parquet_uri,
    resolve_output_base_uri,
    safe_ape,
    select_champion,
    validate_feature_table,
    write_chunk_result,
    write_json_uri,
    write_parquet_uri,
    calculate_metrics,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
MODEL_RESULTS_PREFIX = os.environ.get("MODEL_RESULTS_PREFIX", "model_results/autoregressive")
DASHBOARD_OUTPUT_PREFIX = os.environ.get("DASHBOARD_OUTPUT_PREFIX", "dashboard/autoregressive")


FEATURE_IMPORTANCE_COLUMNS = [
    "experiment_id",
    "pipeline_run_id",
    "model_run_id",
    "model_config_id",
    "prediction_id",
    "config_id",
    "as_of_date",
    "model_family",
    "model_build",
    "model_type",
    "mode",
    "feature_family_name",
    "feature_policy",
    "feature_set_id",
    "feature_name",
    "importance_type",
    "importance",
    "importance_abs",
    "rank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ARIMA/SARIMA/SARIMAX Phase B experiments.")
    parser.add_argument("--experiment-config", required=True, help="YAML experiment config path.")
    parser.add_argument("--bucket", default=BUCKET_NAME)
    parser.add_argument("--results-prefix", default=MODEL_RESULTS_PREFIX)
    parser.add_argument("--dashboard-prefix", default=DASHBOARD_OUTPUT_PREFIX)
    parser.add_argument("--enable-mlflow", action="store_true", default=False)
    parser.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "transit-forecasting-phase-b"))
    parser.add_argument("--mlflow-run-name", default=os.environ.get("MLFLOW_RUN_NAME"))
    return parser.parse_args()


def read_yaml_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("YAML config support requires PyYAML. Install project requirements first.")
    return yaml.safe_load(Path(path).read_text()) or {}


def tuple_order(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def valid_trend_for_order(trend: str, order: tuple[int, int, int]) -> bool:
    # Statsmodels does not allow a lower-order trend than the integrated order.
    if trend in {"c", "t", "ct"} and order[1] > 0:
        return False
    return True


def params_from_config(config: dict) -> list[dict]:
    rows: list[dict] = []
    models = config.get("models") or {}

    arima = models.get("arima") or {}
    if arima.get("enabled", False):
        for order in arima.get("orders") or []:
            order_tuple = tuple_order(order)
            for trend in arima.get("trends") or ["n"]:
                if valid_trend_for_order(trend, order_tuple):
                    rows.append(
                        {
                            "model_type": "arima",
                            "feature_family_name": "univariate",
                            "feature_cols": [],
                            "params": {
                                "order": order_tuple,
                                "seasonal_order": (0, 0, 0, 0),
                                "trend": trend,
                            },
                        }
                    )

    sarima = models.get("sarima") or {}
    if sarima.get("enabled", False):
        for order in sarima.get("orders") or []:
            order_tuple = tuple_order(order)
            for seasonal_order in sarima.get("seasonal_orders") or []:
                seasonal_tuple = tuple_order(seasonal_order)
                for trend in sarima.get("trends") or ["n"]:
                    if valid_trend_for_order(trend, order_tuple):
                        rows.append(
                            {
                                "model_type": "sarima",
                                "feature_family_name": "seasonal_univariate",
                                "feature_cols": [],
                                "params": {
                                    "order": order_tuple,
                                    "seasonal_order": seasonal_tuple,
                                    "trend": trend,
                                },
                            }
                        )

    sarimax = models.get("sarimax") or {}
    if sarimax.get("enabled", False):
        exog_sets = sarimax.get("exog_sets") or {}
        for set_name, exog_cols in exog_sets.items():
            for order in sarimax.get("orders") or []:
                order_tuple = tuple_order(order)
                for seasonal_order in sarimax.get("seasonal_orders") or []:
                    seasonal_tuple = tuple_order(seasonal_order)
                    for trend in sarimax.get("trends") or ["n"]:
                        if valid_trend_for_order(trend, order_tuple):
                            rows.append(
                                {
                                    "model_type": "sarimax",
                                    "feature_family_name": set_name,
                                    "feature_cols": list(exog_cols or []),
                                    "params": {
                                        "order": order_tuple,
                                        "seasonal_order": seasonal_tuple,
                                        "trend": trend,
                                    },
                                }
                            )
    return rows


def jsonable_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            clean[key] = list(value)
        else:
            clean[key] = value
    return clean


def task_identifier(task: dict) -> str:
    config = task["config"]
    return config_id(
        config["model_type"],
        "raw",
        config["feature_family_name"],
        jsonable_params(config["params"]),
        "none",
    )


def fit_forecast(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target: str,
    horizon: int,
    params: dict,
    feature_cols: list[str],
) -> tuple[float, float, float]:
    if SARIMAX is None:
        raise ImportError("Phase B requires statsmodels. Install project requirements first.")

    train_indexed = train_df.sort_values("date").set_index("date")
    future_indexed = future_df.sort_values("date").set_index("date")
    endog = train_indexed[target].astype(float)
    exog_train = train_indexed[feature_cols].astype(float) if feature_cols else None
    exog_future = future_indexed[feature_cols].astype(float) if feature_cols else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=endog,
            exog=exog_train,
            order=params["order"],
            seasonal_order=params["seasonal_order"],
            trend=params.get("trend", "n"),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)

    forecast = result.forecast(steps=len(future_df), exog=exog_future)
    return float(forecast.iloc[-1]), float(result.aic), float(result.bic)


def run_autoregressive_config(task: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_table = task["full_table"]
    evaluation_frame = task["evaluation_frame"]
    config = task["config"]
    experiment_id = task["experiment_id"]
    pipeline_run_id = task["pipeline_run_id"]
    target = task["target"]
    horizon = task["horizon"]
    min_train_rows = task["min_train_rows"]
    feature_cols = [col for col in config["feature_cols"] if col in full_table.columns]
    model_type = config["model_type"]
    feature_family_name = config["feature_family_name"]
    params = config["params"]
    representation_metadata = default_representation_metadata("autoregressive")

    hyperparameters_json = json.dumps(jsonable_params(params), sort_keys=True)
    run_config_id = task_identifier(task)
    current_feature_set_id = feature_set_id(feature_family_name, "raw", feature_cols, "none")

    full_table = full_table.sort_values("date").reset_index(drop=True)
    predictions = []
    model_runs = []
    feature_set_rows = [
        feature_set_row(
            experiment_id,
            current_feature_set_id,
            feature_family_name,
            "raw",
            "none",
            feature_cols,
        )
    ]

    for _, eval_row in evaluation_frame.iterrows():
        as_of_date = pd.Timestamp(eval_row["date"])
        target_date = pd.Timestamp(eval_row["target_date"])
        train_df = full_table[full_table["date"] < as_of_date].copy()
        train_required = [target] + feature_cols
        train_df = train_df.dropna(subset=train_required)
        future_df = full_table[(full_table["date"] > train_df["date"].max()) & (full_table["date"] <= target_date)].copy()
        future_df = future_df.dropna(subset=feature_cols) if feature_cols else future_df

        steps_needed = months_between(train_df["date"].max(), target_date) if not train_df.empty else 0
        if len(train_df) < min_train_rows or len(future_df) < steps_needed or steps_needed < horizon:
            continue

        started = time.perf_counter()
        try:
            pred, aic, bic = fit_forecast(
                train_df=train_df,
                future_df=future_df,
                target=target,
                horizon=horizon,
                params=params,
                feature_cols=feature_cols,
            )
            status = "succeeded"
            failure_reason = ""
        except Exception as exc:
            pred, aic, bic = np.nan, np.nan, np.nan
            status = "failed"
            failure_reason = repr(exc)
        train_seconds = time.perf_counter() - started
        if not np.isfinite(pred):
            continue

        actual = float(eval_row[task["target_col"]])
        naive = float(eval_row["seasonal_naive_proxy"])
        error = pred - actual
        evaluation_period = evaluation_period_for(target_date)
        shock_period_flag = is_shock_period(target_date)
        model_run_id = f"{run_config_id}__as_of_{as_of_date.date().isoformat()}"

        row_common = {
            "experiment_id": experiment_id,
            "pipeline_run_id": pipeline_run_id,
            "model_run_id": model_run_id,
            "model_config_id": run_config_id,
            "prediction_id": model_run_id,
            "config_id": run_config_id,
            "as_of_date": as_of_date.date().isoformat(),
            "target": target,
            "horizon": horizon,
            "model_family": "autoregressive",
            "model_build": model_type,
            "model_type": model_type,
            "ensemble_method": "",
            "mode": "raw",
            "feature_family_name": feature_family_name,
            "feature_policy": "none",
            "feature_set_id": current_feature_set_id,
            "n_features": len(feature_cols),
            "n_features_before_policy": len(feature_cols),
            "n_features_after_policy": len(feature_cols),
            "representation_policy": representation_metadata["representation_policy"],
            "n_representation_features": len(feature_cols),
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
                "model_refit": True,
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
                "selected_feature_names_json": json.dumps(feature_cols, sort_keys=True),
                "dropped_feature_names_json": "[]",
                "feature_policy_params_json": "{}",
                "representation_policy": representation_metadata["representation_policy"],
                "representation_params_json": representation_metadata["representation_params_json"],
                "n_representation_features": len(feature_cols),
                "sequence_length": representation_metadata["sequence_length"],
                "sequence_stride": representation_metadata["sequence_stride"],
                "prediction_head": representation_metadata["prediction_head"],
                "training_window_months": representation_metadata["training_window_months"],
                "validation_strategy": representation_metadata["validation_strategy"],
                "early_stopping_used": representation_metadata["early_stopping_used"],
                "epochs_trained": representation_metadata["epochs_trained"],
                "best_epoch": representation_metadata["best_epoch"],
                "framework": "statsmodels",
                "framework_version": "",
                "hardware_type": representation_metadata["hardware_type"],
                "device": representation_metadata["device"],
                "gpu_name": representation_metadata["gpu_name"],
                "cuda_version": representation_metadata["cuda_version"],
                "refit_frequency_months": 1,
                "model_refit": True,
                "train_seconds": train_seconds,
                "predict_seconds": 0.0,
                "status": status,
                "failure_reason": failure_reason,
                "artifact_uri": "",
                "aic": aic,
                "bic": bic,
            }
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(model_runs),
        pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS),
        pd.DataFrame(feature_set_rows),
    )


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


def run_all_configs(
    full_table: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    configs: list[dict],
    experiment_id: str,
    pipeline_run_id: str | None,
    target_col: str,
    target: str,
    horizon: int,
    min_train_rows: int,
    n_jobs: int,
    chunk_dir: str | None,
    checkpoint_dir: str | None,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [
        {
            "full_table": full_table,
            "evaluation_frame": evaluation_frame,
            "config": config,
            "experiment_id": experiment_id,
            "pipeline_run_id": pipeline_run_id,
            "target_col": target_col,
            "target": target,
            "horizon": horizon,
            "min_train_rows": min_train_rows,
        }
        for config in configs
    ]
    logger.info("Prepared %s autoregressive model configurations.", len(tasks))

    predictions_parts = []
    model_run_parts = []
    importance_parts = []
    feature_set_parts = []
    pending_tasks = []

    for task in tasks:
        task_id = task_identifier(task)
        if resume and chunk_dir and chunk_is_complete(chunk_dir, task_id):
            chunk = read_chunk_result(chunk_dir, task_id)
            predictions_parts.append(chunk[0])
            model_run_parts.append(chunk[1])
            importance_parts.append(chunk[2])
            feature_set_parts.append(chunk[3])
            continue
        pending_tasks.append(task)

    def record_chunk(task_id: str, chunk: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        predictions_parts.append(chunk[0])
        model_run_parts.append(chunk[1])
        importance_parts.append(chunk[2])
        feature_set_parts.append(chunk[3])
        if chunk_dir:
            write_chunk_result(chunk_dir, task_id, chunk)
        if checkpoint_dir:
            append_checkpoint_rows(
                checkpoint_dir,
                "completed_configs.parquet",
                [{"task_id": task_id, "completed_at_utc": datetime.now(timezone.utc).isoformat()}],
            )

    if n_jobs <= 1:
        for index, task in enumerate(pending_tasks, start=1):
            task_id = task_identifier(task)
            logger.info("Running autoregressive config %s/%s: %s", index, len(pending_tasks), task_id)
            try:
                record_chunk(task_id, run_autoregressive_config(task))
            except Exception as exc:
                logger.exception("Autoregressive config failed: %s", task_id)
                if checkpoint_dir:
                    append_checkpoint_rows(
                        checkpoint_dir,
                        "failed_configs.parquet",
                        [{"task_id": task_id, "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}],
                    )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {executor.submit(run_autoregressive_config, task): task for task in pending_tasks}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                task_id = task_identifier(task)
                logger.info("Completed autoregressive config %s/%s: %s", index, len(pending_tasks), task_id)
                try:
                    record_chunk(task_id, future.result())
                except Exception as exc:
                    logger.exception("Autoregressive config failed: %s", task_id)
                    if checkpoint_dir:
                        append_checkpoint_rows(
                            checkpoint_dir,
                            "failed_configs.parquet",
                            [{"task_id": task_id, "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)}],
                        )

    return (
        concat_frames(predictions_parts),
        concat_frames(model_run_parts),
        concat_frames(importance_parts),
        concat_frames(feature_set_parts),
    )


def namespace_for_mlflow(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    tracking = (config.get("tracking") or {}).get("mlflow") or {}
    args.enable_mlflow = bool(tracking.get("enabled", args.enable_mlflow))
    args.mlflow_tracking_uri = tracking.get("tracking_uri", args.mlflow_tracking_uri)
    args.mlflow_experiment_name = tracking.get("experiment_name", args.mlflow_experiment_name)
    args.mlflow_run_name = tracking.get("run_name", args.mlflow_run_name)
    return args


def main() -> int:
    args = parse_args()
    config = read_yaml_config(args.experiment_config)
    args = namespace_for_mlflow(args, config)

    inputs = config.get("inputs") or {}
    outputs = config.get("outputs") or {}
    forecast = config.get("forecast") or {}
    execution = config.get("execution") or {}
    checkpointing = execution.get("checkpointing") or {}

    feature_table_uri = inputs["feature_table_uri"]
    feature_families_uri = inputs.get("feature_families_uri")
    feature_table = read_parquet_uri(feature_table_uri)
    feature_families = read_json_uri(feature_families_uri) if feature_families_uri else {}
    feature_table["date"] = pd.to_datetime(feature_table["date"]).dt.to_period("M").dt.to_timestamp()

    target = forecast.get("target", "upt")
    horizon = int(forecast.get("horizon", 3))
    target_col = validate_feature_table(feature_table, target, horizon)
    full_table = add_seasonal_naive_proxy(feature_table, target_col=target_col, seasonal_periods=12)
    evaluation_frame = build_evaluation_frame(
        feature_table,
        target_col,
        forecast.get("as_of_start", "2011-01-01"),
        forecast.get("as_of_end"),
        int(forecast.get("as_of_frequency_months", 1)),
        horizon,
    )
    model_run_id = config.get("experiment_id") or current_model_run_id(str(Path(feature_table_uri).parent))
    results_base_uri = resolve_output_base_uri(
        outputs.get("results_base_uri"),
        args.bucket,
        args.results_prefix,
        model_run_id,
    )
    dashboard_base_uri = resolve_output_base_uri(
        outputs.get("dashboard_base_uri"),
        args.bucket,
        args.dashboard_prefix,
        model_run_id,
    )
    model_configs = params_from_config(config)

    missing_exog = sorted(
        {
            col
            for row in model_configs
            for col in row.get("feature_cols", [])
            if col not in full_table.columns
        }
    )
    if missing_exog:
        raise ValueError(f"Configured SARIMAX exogenous columns are missing from feature table: {missing_exog}")

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
    logger.info("Autoregressive results will write under: %s", results_base_uri)
    logger.info("Dashboard artifacts will write under: %s", dashboard_base_uri)

    predictions, model_runs, feature_importance, feature_sets = run_all_configs(
        full_table=full_table,
        evaluation_frame=evaluation_frame,
        configs=model_configs,
        experiment_id=model_run_id,
        pipeline_run_id=os.environ.get("PIPELINE_RUN_ID"),
        target_col=target_col,
        target=target,
        horizon=horizon,
        min_train_rows=int(forecast.get("min_train_rows", 72)),
        n_jobs=int(execution.get("n_jobs", 1)),
        chunk_dir=checkpointing.get("chunk_dir"),
        checkpoint_dir=checkpointing.get("checkpoint_dir"),
        resume=bool(checkpointing.get("resume", False)),
    )
    if predictions.empty:
        raise ValueError("No autoregressive predictions were produced.")

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
        "feature_artifacts": {
            "feature_table_uri": feature_table_uri,
            "feature_families_uri": feature_families_uri,
        },
        "results_base_uri": results_base_uri,
        "dashboard_base_uri": dashboard_base_uri,
        "target": target,
        "horizon": horizon,
        "as_of_start": forecast.get("as_of_start", "2011-01-01"),
        "as_of_end": forecast.get("as_of_end"),
        "as_of_frequency_months": int(forecast.get("as_of_frequency_months", 1)),
        "models": sorted(set(metrics["model_type"])),
        "modes": sorted(set(metrics["mode"])),
        "feature_policies": sorted(set(metrics["feature_policy"])),
        "experiment_config": args.experiment_config,
        "requested_model_types": sorted(set(row["model_type"] for row in model_configs)),
        "n_jobs": int(execution.get("n_jobs", 1)),
        "chunk_dir": checkpointing.get("chunk_dir"),
        "checkpoint_dir": checkpointing.get("checkpoint_dir"),
        "resume": bool(checkpointing.get("resume", False)),
        "feature_family_count": int(len(feature_families)),
        "model_config_count": int(len(model_configs)),
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "metric_count": int(len(metrics)),
        "complexity_profile_count": int(len(complexity_profile)),
        "xgb_refresh_months": 0,
        "refit_frequency_months": 1,
        "champion_config_id": champion["config_id"],
        "selection_rule": champion["selection_rule"],
        "runtime": {
            "compute_context": os.environ.get("COMPUTE_CONTEXT", "local"),
            "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
            "image_uri": os.environ.get("IMAGE_URI"),
            "code_version": os.environ.get("CODE_VERSION"),
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
    logger.info("Wrote autoregressive results to %s", results_base_uri)
    logger.info("Wrote autoregressive dashboard artifacts to %s", dashboard_base_uri)
    logger.info("Champion: %s", json.dumps(champion, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
