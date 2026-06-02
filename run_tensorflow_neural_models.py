"""Run Phase C TensorFlow/Keras sequence-model transit experiments.

This sibling runner exists to test Keras-native recurrent behavior, especially
`recurrent_dropout`, while keeping the same rolling historical simulation and
artifact contract as the PyTorch neural runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled at runtime for optional dependency
    tf = None

try:
    import yaml
except ImportError:
    yaml = None

from run_aws_streamlined_models import (
    add_seasonal_naive_proxy,
    append_checkpoint_rows,
    apply_feature_policy,
    build_complexity_profile,
    build_dashboard_outputs,
    build_evaluation_frame,
    build_family_summary,
    calculate_metrics,
    chunk_is_complete,
    config_id,
    current_model_run_id,
    evaluation_period_for,
    feature_set_id,
    feature_set_row,
    is_shock_period,
    join_uri,
    log_to_mlflow,
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
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
MODEL_RESULTS_PREFIX = os.environ.get("MODEL_RESULTS_PREFIX", "model_results/neural_tensorflow")
DASHBOARD_OUTPUT_PREFIX = os.environ.get("DASHBOARD_OUTPUT_PREFIX", "dashboard/neural_tensorflow")
TENSORFLOW_RUNNER_CONTRACT_VERSION = "v1_keras_sequence"

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
    parser = argparse.ArgumentParser(description="Run TensorFlow/Keras Phase C neural experiments.")
    parser.add_argument("--experiment-config", required=True, help="YAML experiment config path.")
    parser.add_argument("--bucket", default=BUCKET_NAME)
    parser.add_argument("--results-prefix", default=MODEL_RESULTS_PREFIX)
    parser.add_argument("--dashboard-prefix", default=DASHBOARD_OUTPUT_PREFIX)
    parser.add_argument("--enable-mlflow", action="store_true", default=False)
    parser.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--mlflow-experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "transit-forecasting-phase-c-tensorflow"),
    )
    parser.add_argument("--mlflow-run-name", default=os.environ.get("MLFLOW_RUN_NAME"))
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("EXPERIMENT_SHARD_INDEX", "0")))
    parser.add_argument("--shard-count", type=int, default=int(os.environ.get("EXPERIMENT_SHARD_COUNT", "1")))
    return parser.parse_args()


def read_yaml_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("YAML config support requires PyYAML. Install project requirements first.")
    return yaml.safe_load(Path(path).read_text()) or {}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                logger.debug("TensorFlow op determinism was not available.", exc_info=True)


def configure_tensorflow_device(requested: str) -> tuple[str, str, str]:
    if tf is None:
        raise ImportError("TensorFlow is required. Install requirements-tensorflow.txt or tensorflow manually.")
    requested = requested.lower()
    gpus = tf.config.list_physical_devices("GPU")
    if requested in {"cuda", "gpu"} and not gpus:
        raise RuntimeError("GPU/CUDA was requested but TensorFlow does not see a GPU.")
    if requested == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            logger.warning("TensorFlow devices were already initialized; CPU-only visibility could not be forced.")
        return "cpu", "", ""
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return "gpu", gpus[0].name, ""
    return "cpu", "", ""


def sequence_samples(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    sequence_length: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = ["date", target_col, "seasonal_naive_proxy", *feature_cols]
    ordered = frame[required].dropna().sort_values("date").reset_index(drop=True)
    x_rows = ordered[feature_cols].astype(float).to_numpy()
    target = ordered[target_col].astype(float).to_numpy()
    if mode == "residual":
        target = target - ordered["seasonal_naive_proxy"].astype(float).to_numpy()
    sequences = []
    targets = []
    for index in range(sequence_length - 1, len(ordered)):
        start = index - sequence_length + 1
        if not is_contiguous_month_window(ordered["date"].iloc[start : index + 1]):
            continue
        sequences.append(x_rows[start : index + 1])
        targets.append(target[index])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def is_contiguous_month_window(dates: pd.Series) -> bool:
    month_ordinals = pd.DatetimeIndex(pd.to_datetime(dates)).to_period("M").astype(int)
    return len(month_ordinals) < 2 or bool(np.all(np.diff(month_ordinals) == 1))


def prediction_sequence(frame: pd.DataFrame, as_of_date, feature_cols: list[str], sequence_length: int) -> np.ndarray | None:
    required = ["date", *feature_cols]
    ordered = frame.loc[frame["date"] <= as_of_date, required].dropna().sort_values("date")
    if len(ordered) < sequence_length or pd.Timestamp(ordered.iloc[-1]["date"]) != pd.Timestamp(as_of_date):
        return None
    latest = ordered.tail(sequence_length)
    if not is_contiguous_month_window(latest["date"]):
        return None
    return latest[feature_cols].astype(float).to_numpy(dtype=np.float32)


def build_representation_transformer(policy: str, scaled_fit_rows: np.ndarray) -> PCA | None:
    if policy == "sequence_raw":
        return None
    if policy == "sequence_pca_95":
        transformer = PCA(n_components=0.95, svd_solver="full")
    elif policy == "sequence_pca_20":
        max_components = min(20, scaled_fit_rows.shape[0], scaled_fit_rows.shape[1])
        transformer = PCA(n_components=max(1, max_components))
    else:
        raise ValueError(f"Unsupported neural representation policy: {policy}")
    transformer.fit(scaled_fit_rows)
    return transformer


def model_configs_from_config(config: dict) -> list[dict]:
    rows = []
    for model_type, details in (config.get("models") or {}).items():
        if details.get("enabled", False):
            for params in details.get("param_grid") or [{}]:
                rows.append({"model_type": model_type, "params": params or {}})
    return rows


def policy_representation_variants_from_config(config: dict) -> list[dict]:
    configured = config.get("policy_representation_variants") or []
    if configured:
        return [
            {
                "feature_policy": variant.get("feature_policy", "none"),
                "representation_policy": variant.get("representation_policy", "sequence_raw"),
                "min_family_features": int(variant.get("min_family_features", 0)),
                "max_family_features": (
                    int(variant["max_family_features"])
                    if variant.get("max_family_features") is not None
                    else None
                ),
            }
            for variant in configured
        ]
    feature_policies = config.get("feature_policies") or ["none"]
    representation_policies = config.get("representation_policies") or ["sequence_raw"]
    return [
        {"feature_policy": feature_policy, "representation_policy": representation_policy}
        for feature_policy in feature_policies
        for representation_policy in representation_policies
    ]


def policy_variant_applies(variant: dict, n_family_features: int) -> bool:
    if n_family_features < int(variant.get("min_family_features", 0)):
        return False
    max_family_features = variant.get("max_family_features")
    return max_family_features is None or n_family_features <= int(max_family_features)


def rolling_policy_signature(
    full_table: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    mode: str,
    feature_policy: str,
) -> tuple[tuple[str, ...], ...]:
    signatures = []
    for _, eval_row in evaluation_frame.iterrows():
        train_df = full_table[full_table["date"] < eval_row["date"]].copy()
        required_cols = [target_col, "seasonal_naive_proxy", *feature_cols]
        policy_train_df = train_df.dropna(subset=required_cols)
        if mode == "residual":
            policy_target = policy_train_df[target_col] - policy_train_df["seasonal_naive_proxy"]
        else:
            policy_target = policy_train_df[target_col]
        policy_result = apply_feature_policy(policy_train_df, feature_cols, policy_target, feature_policy)
        signatures.append(tuple(policy_result["selected_features"]))
    return tuple(signatures)


def namespace_for_mlflow(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    tracking = (config.get("tracking") or {}).get("mlflow") or {}
    args.enable_mlflow = bool(tracking.get("enabled", args.enable_mlflow))
    args.mlflow_tracking_uri = tracking.get("tracking_uri", args.mlflow_tracking_uri)
    args.mlflow_experiment_name = tracking.get("experiment_name", args.mlflow_experiment_name)
    args.mlflow_run_name = tracking.get("run_name", args.mlflow_run_name)
    return args


def neural_task_id(
    feature_family_name: str,
    mode: str,
    model_config: dict,
    feature_policy: str,
    representation_policy: str,
) -> str:
    params = dict(model_config["params"])
    params["sequence_length"] = int(params.get("sequence_length", 12))
    params["runner_contract_version"] = TENSORFLOW_RUNNER_CONTRACT_VERSION
    params["representation_policy"] = representation_policy
    return config_id(model_config["model_type"], mode, feature_family_name, params, feature_policy)


def shard_uri(uri: str | None, shard_index: int, shard_count: int) -> str | None:
    if not uri or shard_count <= 1:
        return uri
    return join_uri(uri, f"shard_{shard_index:03d}_of_{shard_count:03d}")


def build_keras_model(model_type: str, n_features: int, params: dict):
    if tf is None:
        raise ImportError("TensorFlow is required for build_keras_model.")
    sequence_length = int(params["sequence_length"])
    recurrent_hidden_sizes = [int(value) for value in params.get("recurrent_hidden_sizes", [100])]
    recurrent_dropout = float(params.get("recurrent_dropout", 0.0))
    layer_dropout = float(params.get("dropout", 0.0))
    inter_recurrent_dropouts = params.get("inter_recurrent_dropouts")
    if inter_recurrent_dropouts is None:
        inter_recurrent_dropouts = []
    elif not isinstance(inter_recurrent_dropouts, list):
        inter_recurrent_dropouts = [inter_recurrent_dropouts]
    dense_head_sizes = [int(value) for value in params.get("dense_head_sizes", [])]
    dense_head_dropouts = params.get("dense_head_dropouts", [])
    if not isinstance(dense_head_dropouts, list):
        dense_head_dropouts = [dense_head_dropouts] * len(dense_head_sizes)
    dense_l2 = float(params.get("dense_l2", params.get("weight_decay", 0.0)))

    recurrent_cls = {"rnn": tf.keras.layers.SimpleRNN, "gru": tf.keras.layers.GRU, "lstm": tf.keras.layers.LSTM}[
        model_type
    ]
    model = tf.keras.Sequential()
    for layer_index, hidden_size in enumerate(recurrent_hidden_sizes):
        recurrent_kwargs = {
            "activation": params.get("activation", "tanh"),
            "dropout": layer_dropout,
            "recurrent_dropout": recurrent_dropout,
            "return_sequences": layer_index < len(recurrent_hidden_sizes) - 1,
        }
        if layer_index == 0:
            recurrent_kwargs["input_shape"] = (sequence_length, n_features)
        model.add(
            recurrent_cls(
                hidden_size,
                **recurrent_kwargs,
            )
        )
        if layer_index < len(recurrent_hidden_sizes) - 1:
            drop = float(inter_recurrent_dropouts[layer_index]) if layer_index < len(inter_recurrent_dropouts) else 0.0
            if drop:
                model.add(tf.keras.layers.Dropout(drop))
    pre_head_dropout = float(params.get("pre_head_dropout", 0.0))
    if pre_head_dropout:
        model.add(tf.keras.layers.Dropout(pre_head_dropout))
    regularizer = tf.keras.regularizers.l2(dense_l2) if dense_l2 else None
    for index, dense_size in enumerate(dense_head_sizes):
        model.add(tf.keras.layers.Dense(dense_size, activation="relu", kernel_regularizer=regularizer))
        dense_dropout = float(dense_head_dropouts[index]) if index < len(dense_head_dropouts) else 0.0
        if dense_dropout:
            model.add(tf.keras.layers.Dropout(dense_dropout))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(params.get("learning_rate", 1e-3))),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    params: dict,
    representation_policy: str,
) -> tuple[object, StandardScaler, PCA | None, StandardScaler, dict]:
    validation_rows = int(params.get("validation_rows", 12))
    if len(x_train) <= validation_rows + 1:
        raise ValueError("Not enough sequence samples for the configured validation holdout.")
    x_fit, x_val = x_train[:-validation_rows], x_train[-validation_rows:]
    y_fit, y_val = y_train[:-validation_rows], y_train[-validation_rows:]
    scaler = StandardScaler()
    flat_fit = x_fit.reshape(-1, x_fit.shape[-1])
    scaler.fit(flat_fit)
    scaled_fit = scaler.transform(flat_fit).reshape(x_fit.shape)
    scaled_val = scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)

    representation_transformer = build_representation_transformer(
        representation_policy,
        scaled_fit.reshape(-1, scaled_fit.shape[-1]),
    )

    def represented(values: np.ndarray) -> np.ndarray:
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        if representation_transformer is not None:
            flat = representation_transformer.transform(flat)
        return flat.reshape(shape[0], shape[1], -1).astype(np.float32)

    represented_fit = represented(scaled_fit)
    represented_val = represented(scaled_val)
    target_scaler = StandardScaler()
    y_fit_scaled = target_scaler.fit_transform(y_fit.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    tf.keras.backend.clear_session()
    model = build_keras_model(model_type, represented_fit.shape[-1], params)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(params.get("early_stopping_patience", 10)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(params.get("lr_factor", 0.5)),
            patience=int(params.get("lr_patience", 2)),
            min_lr=float(params.get("min_lr", 0.0)),
            verbose=0,
        ),
    ]
    history = model.fit(
        represented_fit,
        y_fit_scaled,
        validation_data=(represented_val, y_val_scaled),
        epochs=int(params.get("max_epochs", 30)),
        batch_size=int(params.get("batch_size", 16)),
        callbacks=callbacks,
        verbose=int(params.get("verbose", 0)),
        shuffle=False,
    )
    val_losses = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.epoch)
    best_loss = float(np.min(val_losses)) if val_losses else np.nan
    return model, scaler, representation_transformer, target_scaler, {
        "epochs_trained": int(len(history.epoch)),
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "early_stopping_used": len(history.epoch) < int(params.get("max_epochs", 30)),
        "n_representation_features": int(represented_fit.shape[-1]),
    }


def predict(
    model,
    scaler: StandardScaler,
    representation_transformer: PCA | None,
    target_scaler: StandardScaler,
    sequence: np.ndarray,
) -> float:
    shape = sequence.shape
    represented = scaler.transform(sequence.reshape(-1, shape[-1]))
    if representation_transformer is not None:
        represented = representation_transformer.transform(represented)
    tensor = represented.reshape(1, shape[0], -1).astype(np.float32)
    scaled_prediction = float(model.predict(tensor, verbose=0)[0, 0])
    return float(target_scaler.inverse_transform([[scaled_prediction]])[0, 0])


def run_config(
    full_table: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    feature_cols: list[str],
    feature_family_name: str,
    mode: str,
    config: dict,
    experiment_id: str,
    pipeline_run_id: str | None,
    target_col: str,
    target: str,
    horizon: int,
    min_train_rows: int,
    seed: int,
    deterministic: bool,
    feature_policy: str,
    representation_policy: str,
    runtime_context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_type = config["model_type"]
    params = dict(config["params"])
    sequence_length = int(params.get("sequence_length", 12))
    params["sequence_length"] = sequence_length
    hyperparameters_json = json.dumps(params, sort_keys=True)
    id_params = {**params, "representation_policy": representation_policy}
    run_config_id = config_id(model_type, mode, feature_family_name, id_params, feature_policy)
    current_feature_set_id = feature_set_id(feature_family_name, mode, feature_cols, feature_policy)
    feature_sets = [
        feature_set_row(
            experiment_id,
            current_feature_set_id,
            feature_family_name,
            mode,
            feature_policy,
            feature_cols,
        )
    ]
    predictions = []
    model_runs = []

    for _, eval_row in evaluation_frame.iterrows():
        as_of_date = eval_row["date"]
        train_df = full_table[full_table["date"] < as_of_date].copy()
        required_cols = [target_col, "seasonal_naive_proxy", *feature_cols]
        policy_train_df = train_df.dropna(subset=required_cols)
        if mode == "residual":
            policy_target = policy_train_df[target_col] - policy_train_df["seasonal_naive_proxy"]
        else:
            policy_target = policy_train_df[target_col]
        policy_result = apply_feature_policy(policy_train_df, feature_cols, policy_target, feature_policy)
        selected_feature_cols = policy_result["selected_features"]
        x_train, y_train = sequence_samples(train_df, selected_feature_cols, target_col, sequence_length, mode)
        sequence = prediction_sequence(full_table, as_of_date, selected_feature_cols, sequence_length)
        if len(x_train) < min_train_rows or sequence is None:
            continue

        set_seed(seed, deterministic=deterministic)
        started = time.perf_counter()
        model, scaler, representation_transformer, target_scaler, fit_metadata = fit_model(
            x_train,
            y_train,
            model_type,
            params,
            representation_policy,
        )
        train_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        prediction = predict(model, scaler, representation_transformer, target_scaler, sequence)
        predict_seconds = time.perf_counter() - predict_started

        naive = float(eval_row["seasonal_naive_proxy"])
        if mode == "residual":
            prediction += naive
        actual = float(eval_row[target_col])
        error = prediction - actual
        target_date = eval_row["target_date"]
        model_run_id = f"{run_config_id}__as_of_{as_of_date.date().isoformat()}"
        common = {
            "experiment_id": experiment_id,
            "pipeline_run_id": pipeline_run_id,
            "model_run_id": model_run_id,
            "model_config_id": run_config_id,
            "prediction_id": model_run_id,
            "config_id": run_config_id,
            "as_of_date": as_of_date.date().isoformat(),
            "target": target,
            "horizon": horizon,
            "model_family": "neural_net",
            "model_build": model_type,
            "model_type": model_type,
            "ensemble_method": "",
            "mode": mode,
            "feature_family_name": feature_family_name,
            "feature_policy": feature_policy,
            "feature_set_id": current_feature_set_id,
            "n_features": len(selected_feature_cols),
            "n_features_before_policy": int(policy_result["n_features_before_policy"]),
            "n_features_after_policy": int(policy_result["n_features_after_policy"]),
            "representation_policy": representation_policy,
            "n_representation_features": fit_metadata["n_representation_features"],
            "sequence_length": sequence_length,
            "sequence_stride": 1,
            "prediction_head": "direct_horizon",
            "n_train": int(len(x_train)),
        }
        predictions.append(
            {
                **common,
                "target_date": target_date.date().isoformat(),
                "actual": actual,
                "prediction": prediction,
                "baseline_prediction": naive,
                "seasonal_naive_prediction": naive,
                "model_refit": True,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error**2,
                "ape": safe_ape(actual, prediction),
                "naive_error": naive - actual,
                "naive_abs_error": abs(naive - actual),
                "evaluation_period": evaluation_period_for(target_date),
                "shock_period_flag": is_shock_period(target_date),
                "train_seconds": train_seconds,
            }
        )
        model_runs.append(
            {
                **common,
                "params": hyperparameters_json,
                "hyperparameters_json": hyperparameters_json,
                "selected_feature_names_json": json.dumps(selected_feature_cols, sort_keys=True),
                "dropped_feature_names_json": json.dumps(policy_result["dropped_features"], sort_keys=True),
                "feature_policy_params_json": json.dumps(policy_result["policy_params"], sort_keys=True),
                "representation_params_json": json.dumps(
                    {
                        "sequence_length": sequence_length,
                        "target_scaling": "standard",
                        "n_representation_features": fit_metadata["n_representation_features"],
                    },
                    sort_keys=True,
                ),
                "training_window_months": np.nan,
                "validation_strategy": "rolling_as_of_ordered_holdout",
                "early_stopping_used": fit_metadata["early_stopping_used"],
                "epochs_trained": fit_metadata["epochs_trained"],
                "best_epoch": fit_metadata["best_epoch"],
                "framework": "tensorflow",
                "framework_version": tf.__version__,
                "hardware_type": runtime_context["hardware_type"],
                "device": runtime_context["device"],
                "gpu_name": runtime_context["gpu_name"],
                "cuda_version": "",
                "refit_frequency_months": 1,
                "model_refit": True,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "status": "succeeded",
                "artifact_uri": "",
                "metric_extras_json": json.dumps({"validation_loss": fit_metadata["validation_loss"]}),
            }
        )
        tf.keras.backend.clear_session()

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(model_runs),
        pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS),
        pd.DataFrame(feature_sets),
    )


def main() -> int:
    args = parse_args()
    config = read_yaml_config(args.experiment_config)
    args = namespace_for_mlflow(args, config)
    inputs = config.get("inputs") or {}
    outputs = config.get("outputs") or {}
    forecast = config.get("forecast") or {}
    execution = config.get("execution") or {}

    device_label, gpu_name, cuda_version = configure_tensorflow_device(str(execution.get("device", "auto")))
    seed = int(execution.get("random_seed", 42))
    deterministic = bool(execution.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be between 0 and --shard-count - 1.")

    feature_table_uri = inputs["feature_table_uri"]
    feature_families_uri = inputs["feature_families_uri"]
    feature_table = read_parquet_uri(feature_table_uri)
    feature_families = read_json_uri(feature_families_uri)
    feature_table["date"] = pd.to_datetime(feature_table["date"]).dt.to_period("M").dt.to_timestamp()
    target = forecast.get("target", "upt")
    horizon = int(forecast.get("horizon", 3))
    target_col = validate_feature_table(feature_table, target, horizon)
    full_table = add_seasonal_naive_proxy(feature_table, target_col=target_col, seasonal_periods=12)
    evaluation_frame = build_evaluation_frame(
        feature_table,
        target_col,
        forecast.get("as_of_start", "2024-01-01"),
        forecast.get("as_of_end"),
        int(forecast.get("as_of_frequency_months", 1)),
        horizon,
    )
    experiment_id = config.get("experiment_id") or current_model_run_id(str(Path(feature_table_uri).parent))
    results_base_uri = resolve_output_base_uri(outputs.get("results_base_uri"), args.bucket, args.results_prefix, experiment_id)
    dashboard_base_uri = resolve_output_base_uri(outputs.get("dashboard_base_uri"), args.bucket, args.dashboard_prefix, experiment_id)
    checkpointing = execution.get("checkpointing") or {}
    results_base_uri = shard_uri(results_base_uri, args.shard_index, args.shard_count)
    dashboard_base_uri = shard_uri(dashboard_base_uri, args.shard_index, args.shard_count)
    chunk_dir = shard_uri(checkpointing.get("chunk_dir"), args.shard_index, args.shard_count)
    checkpoint_dir = shard_uri(checkpointing.get("checkpoint_dir"), args.shard_index, args.shard_count)
    resume = bool(checkpointing.get("resume", False))
    configs = model_configs_from_config(config)
    included_families = (config.get("feature_families") or {}).get("include") or []
    modes = config.get("modes") or ["raw"]
    policy_representation_variants = policy_representation_variants_from_config(config)
    dynamic_policy_dedup = bool(execution.get("dynamic_policy_dedup", False))
    runtime_context = {
        "hardware_type": "gpu" if device_label == "gpu" else "cpu",
        "device": device_label,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
    }

    logger.info("Using TensorFlow device: %s", device_label)
    logger.info("Feature table: %s rows, %s columns", len(feature_table), len(feature_table.columns))
    logger.info("TensorFlow neural configs: %s", len(configs))
    logger.info("Policy/representation variants: %s", policy_representation_variants)
    logger.info("Dynamic policy deduplication: %s", dynamic_policy_dedup)
    logger.info("Shard: %s of %s", args.shard_index, args.shard_count)
    logger.info("Deterministic execution: %s", deterministic)

    chunks = []
    completed_rows = []
    failed_rows = []
    skipped_policy_variants = []
    skipped_shard_task_count = 0
    scheduled_task_index = 0
    resumed_count = 0
    for family_name in included_families:
        feature_cols = feature_families[family_name]
        missing = [column for column in feature_cols if column not in full_table]
        if missing:
            raise ValueError(f"Missing configured features for {family_name}: {missing}")
        for mode in modes:
            scheduled_policy_signatures: dict[str, dict[tuple[tuple[str, ...], ...], str]] = {}
            for variant in policy_representation_variants:
                feature_policy = variant["feature_policy"]
                representation_policy = variant["representation_policy"]
                if not policy_variant_applies(variant, len(feature_cols)):
                    logger.info(
                        "Skipping inapplicable variant %s | %s | %s | %s features",
                        family_name,
                        feature_policy,
                        representation_policy,
                        len(feature_cols),
                    )
                    skipped_policy_variants.append(
                        {
                            "feature_family_name": family_name,
                            "mode": mode,
                            "feature_policy": feature_policy,
                            "representation_policy": representation_policy,
                            "reason": "family_feature_count_outside_variant_bounds",
                            "n_family_features": len(feature_cols),
                        }
                    )
                    continue
                if dynamic_policy_dedup:
                    signature = rolling_policy_signature(
                        full_table,
                        evaluation_frame,
                        feature_cols,
                        target_col,
                        mode,
                        feature_policy,
                    )
                    signatures_for_representation = scheduled_policy_signatures.setdefault(representation_policy, {})
                    equivalent_policy = signatures_for_representation.get(signature)
                    if equivalent_policy is not None:
                        logger.info(
                            "Skipping equivalent policy %s | %s | %s; matches %s",
                            family_name,
                            feature_policy,
                            representation_policy,
                            equivalent_policy,
                        )
                        skipped_policy_variants.append(
                            {
                                "feature_family_name": family_name,
                                "mode": mode,
                                "feature_policy": feature_policy,
                                "representation_policy": representation_policy,
                                "reason": "equivalent_rolling_selected_features",
                                "equivalent_policy": equivalent_policy,
                                "n_family_features": len(feature_cols),
                            }
                        )
                        continue
                    signatures_for_representation[signature] = feature_policy
                for model_config in configs:
                    current_task_index = scheduled_task_index
                    scheduled_task_index += 1
                    if current_task_index % args.shard_count != args.shard_index:
                        skipped_shard_task_count += 1
                        continue
                    task_id = neural_task_id(
                        family_name,
                        mode,
                        model_config,
                        feature_policy,
                        representation_policy,
                    )
                    if resume and chunk_dir and chunk_is_complete(chunk_dir, task_id):
                        logger.info("Resuming completed chunk %s", task_id)
                        chunks.append(read_chunk_result(chunk_dir, task_id))
                        resumed_count += 1
                        continue
                    logger.info(
                        "Running %s | %s | %s | %s | %s",
                        model_config["model_type"],
                        mode,
                        family_name,
                        feature_policy,
                        representation_policy,
                    )
                    try:
                        chunk = run_config(
                            full_table,
                            evaluation_frame,
                            feature_cols,
                            family_name,
                            mode,
                            model_config,
                            experiment_id,
                            os.environ.get("PIPELINE_RUN_ID"),
                            target_col,
                            target,
                            horizon,
                            int(forecast.get("min_train_rows", 24)),
                            seed,
                            deterministic,
                            feature_policy,
                            representation_policy,
                            runtime_context,
                        )
                        if chunk[0].empty or chunk[1].empty:
                            raise ValueError("Configuration produced no prediction or model-run rows.")
                        if chunk_dir:
                            write_chunk_result(chunk_dir, task_id, chunk)
                        chunks.append(chunk)
                        completed_rows.append(
                            {
                                "task_id": task_id,
                                "model_type": model_config["model_type"],
                                "mode": mode,
                                "feature_family_name": family_name,
                                "feature_policy": feature_policy,
                                "representation_policy": representation_policy,
                                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        if checkpoint_dir:
                            append_checkpoint_rows(checkpoint_dir, "completed_configs.parquet", completed_rows[-1:])
                    except Exception as exc:
                        logger.exception("Failed configuration %s", task_id)
                        failed_rows.append(
                            {
                                "task_id": task_id,
                                "model_type": model_config["model_type"],
                                "mode": mode,
                                "feature_family_name": family_name,
                                "feature_policy": feature_policy,
                                "representation_policy": representation_policy,
                                "error": str(exc),
                                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        if checkpoint_dir:
                            append_checkpoint_rows(checkpoint_dir, "failed_configs.parquet", failed_rows[-1:])

    predictions = pd.concat([chunk[0] for chunk in chunks if not chunk[0].empty], ignore_index=True)
    model_runs = pd.concat([chunk[1] for chunk in chunks if not chunk[1].empty], ignore_index=True)
    feature_importance_chunks = [chunk[2] for chunk in chunks if not chunk[2].empty]
    feature_importance = (
        pd.concat(feature_importance_chunks, ignore_index=True)
        if feature_importance_chunks
        else pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)
    )
    feature_sets = pd.concat([chunk[3] for chunk in chunks if not chunk[3].empty], ignore_index=True).drop_duplicates(
        "feature_set_id"
    )
    if predictions.empty:
        raise ValueError("No TensorFlow neural predictions were produced.")

    metrics = calculate_metrics(predictions)
    family_summary = build_family_summary(metrics)
    champion = select_champion(metrics)
    complexity_profile = build_complexity_profile(model_runs, metrics)
    dashboard_outputs = build_dashboard_outputs(predictions, model_runs, metrics, family_summary, champion, complexity_profile)

    artifacts = {
        "predictions.parquet": predictions,
        "model_runs.parquet": model_runs,
        "metrics.parquet": metrics,
        "feature_importance.parquet": feature_importance,
        "feature_sets.parquet": feature_sets,
        "feature_family_summary.parquet": family_summary,
        "complexity_profile.parquet": complexity_profile,
    }
    for filename, frame in artifacts.items():
        write_parquet_uri(join_uri(results_base_uri, filename), frame)
    write_json_uri(join_uri(results_base_uri, "champion_selection.json"), champion)

    manifest = {
        "run_id": experiment_id,
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_artifacts": {"feature_table_uri": feature_table_uri, "feature_families_uri": feature_families_uri},
        "results_base_uri": results_base_uri,
        "dashboard_base_uri": dashboard_base_uri,
        "target": target,
        "horizon": horizon,
        "as_of_start": forecast.get("as_of_start"),
        "as_of_end": forecast.get("as_of_end"),
        "as_of_frequency_months": int(forecast.get("as_of_frequency_months", 1)),
        "models": sorted(set(metrics["model_type"])),
        "modes": sorted(set(metrics["mode"])),
        "feature_policies": sorted(set(metrics["feature_policy"])),
        "representation_policies": sorted(set(model_runs["representation_policy"])),
        "requested_feature_families": included_families,
        "model_config_count": int(len(metrics[metrics["evaluation_scope"] == "overall"])),
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "metric_count": int(len(metrics)),
        "complexity_profile_count": int(len(complexity_profile)),
        "resumed_config_count": resumed_count,
        "completed_config_count": len(completed_rows),
        "failed_config_count": len(failed_rows),
        "dynamic_policy_dedup": dynamic_policy_dedup,
        "skipped_policy_variant_count": len(skipped_policy_variants),
        "skipped_policy_variants": skipped_policy_variants,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "scheduled_task_count_before_sharding": scheduled_task_index,
        "skipped_shard_task_count": skipped_shard_task_count,
        "champion_config_id": champion["config_id"],
        "selection_rule": champion["selection_rule"],
        "runtime": {
            "compute_context": os.environ.get("COMPUTE_CONTEXT", "local"),
            "framework": "tensorflow",
            "framework_version": tf.__version__,
            "hardware_type": runtime_context["hardware_type"],
            "device": runtime_context["device"],
            "gpu_name": runtime_context["gpu_name"],
            "cuda_version": runtime_context["cuda_version"],
            "deterministic": deterministic,
            "chunk_dir": chunk_dir,
            "checkpoint_dir": checkpoint_dir,
            "resume": resume,
            "runner_contract_version": TENSORFLOW_RUNNER_CONTRACT_VERSION,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
            "mlflow_experiment_name": args.mlflow_experiment_name if args.enable_mlflow else None,
        },
    }
    write_json_uri(join_uri(results_base_uri, "batch_manifest.json"), manifest)
    write_json_uri(join_uri(results_base_uri, "experiment_manifest.json"), manifest)
    log_to_mlflow(args, manifest, champion, metrics, family_summary)

    for filename, frame in dashboard_outputs.items():
        write_parquet_uri(join_uri(dashboard_base_uri, filename), frame)
    write_json_uri(join_uri(dashboard_base_uri, "champion_selection.json"), champion)
    write_json_uri(join_uri(dashboard_base_uri, "experiment_manifest.json"), manifest)
    logger.info("Produced %s predictions across %s TensorFlow model/as-of records.", len(predictions), len(model_runs))
    logger.info("Wrote TensorFlow neural results to %s", results_base_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
