"""Run Phase C PyTorch sequence-model transit forecasting experiments.

The Phase C runner keeps the same rolling historical simulation and portable
artifact contract as the tabular and autoregressive runners. Each as-of month
fits only on sequence samples whose feature rows occur before that month.
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
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import yaml
except ImportError:
    yaml = None

from run_aws_streamlined_models import (
    add_seasonal_naive_proxy,
    build_complexity_profile,
    build_dashboard_outputs,
    build_evaluation_frame,
    build_family_summary,
    calculate_metrics,
    config_id,
    current_model_run_id,
    evaluation_period_for,
    feature_set_id,
    feature_set_row,
    is_shock_period,
    join_uri,
    log_to_mlflow,
    read_json_uri,
    read_parquet_uri,
    resolve_output_base_uri,
    safe_ape,
    select_champion,
    validate_feature_table,
    write_json_uri,
    write_parquet_uri,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
MODEL_RESULTS_PREFIX = os.environ.get("MODEL_RESULTS_PREFIX", "model_results/neural")
DASHBOARD_OUTPUT_PREFIX = os.environ.get("DASHBOARD_OUTPUT_PREFIX", "dashboard/neural")

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
    parser = argparse.ArgumentParser(description="Run PyTorch Phase C neural experiments.")
    parser.add_argument("--experiment-config", required=True, help="YAML experiment config path.")
    parser.add_argument("--bucket", default=BUCKET_NAME)
    parser.add_argument("--results-prefix", default=MODEL_RESULTS_PREFIX)
    parser.add_argument("--dashboard-prefix", default=DASHBOARD_OUTPUT_PREFIX)
    parser.add_argument("--enable-mlflow", action="store_true", default=False)
    parser.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "transit-forecasting-phase-c"))
    parser.add_argument("--mlflow-run-name", default=os.environ.get("MLFLOW_RUN_NAME"))
    return parser.parse_args()


def read_yaml_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("YAML config support requires PyYAML. Install project requirements first.")
    return yaml.safe_load(Path(path).read_text()) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


class SequenceRegressor(nn.Module):
    def __init__(self, model_type: str, n_features: int, params: dict):
        super().__init__()
        self.model_type = model_type
        hidden_size = int(params.get("hidden_size", 32))
        num_layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.0)) if num_layers > 1 else 0.0
        if model_type == "mlp":
            sequence_length = int(params["sequence_length"])
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(sequence_length * n_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(float(params.get("dropout", 0.0))),
            )
        else:
            recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[model_type]
            self.encoder = recurrent_cls(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model_type == "mlp":
            encoded = self.encoder(inputs)
        else:
            outputs, _ = self.encoder(inputs)
            encoded = outputs[:, -1, :]
        return self.head(encoded).squeeze(-1)


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
        sequences.append(x_rows[index - sequence_length + 1 : index + 1])
        targets.append(target[index])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def prediction_sequence(frame: pd.DataFrame, as_of_date, feature_cols: list[str], sequence_length: int) -> np.ndarray | None:
    required = ["date", *feature_cols]
    ordered = frame.loc[frame["date"] <= as_of_date, required].dropna().sort_values("date")
    if len(ordered) < sequence_length or pd.Timestamp(ordered.iloc[-1]["date"]) != pd.Timestamp(as_of_date):
        return None
    return ordered[feature_cols].tail(sequence_length).astype(float).to_numpy(dtype=np.float32)


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    params: dict,
    device: torch.device,
) -> tuple[SequenceRegressor, StandardScaler, dict]:
    validation_rows = max(1, min(int(params.get("validation_rows", 6)), len(x_train) // 4))
    if len(x_train) <= validation_rows:
        raise ValueError("Not enough sequence samples to create a time-ordered validation window.")
    x_fit, x_val = x_train[:-validation_rows], x_train[-validation_rows:]
    y_fit, y_val = y_train[:-validation_rows], y_train[-validation_rows:]

    scaler = StandardScaler()
    scaler.fit(x_fit.reshape(-1, x_fit.shape[-1]))

    def scaled(values: np.ndarray) -> np.ndarray:
        shape = values.shape
        return scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    x_fit_tensor = torch.tensor(scaled(x_fit))
    y_fit_tensor = torch.tensor(y_fit, dtype=torch.float32)
    x_val_tensor = torch.tensor(scaled(x_val), device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(x_fit_tensor, y_fit_tensor),
        batch_size=int(params.get("batch_size", 16)),
        shuffle=False,
    )

    model = SequenceRegressor(model_type, x_train.shape[-1], params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(params.get("lr_factor", 0.5)),
        patience=int(params.get("lr_patience", 2)),
    )
    loss_fn = nn.MSELoss()
    max_epochs = int(params.get("max_epochs", 30))
    early_stopping_patience = int(params.get("early_stopping_patience", 5))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())
        scheduler.step(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, {
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "early_stopping_used": epoch < max_epochs,
    }


def predict(model: SequenceRegressor, scaler: StandardScaler, sequence: np.ndarray, device: torch.device) -> float:
    shape = sequence.shape
    scaled = scaler.transform(sequence.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    tensor = torch.tensor(scaled[None, :, :], device=device)
    model.eval()
    with torch.no_grad():
        return float(model(tensor).item())


def model_configs_from_config(config: dict) -> list[dict]:
    rows = []
    for model_type, details in (config.get("models") or {}).items():
        if details.get("enabled", False):
            for params in details.get("param_grid") or [{}]:
                rows.append({"model_type": model_type, "params": params or {}})
    return rows


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
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_type = config["model_type"]
    params = dict(config["params"])
    sequence_length = int(params.get("sequence_length", 12))
    params["sequence_length"] = sequence_length
    hyperparameters_json = json.dumps(params, sort_keys=True)
    run_config_id = config_id(model_type, mode, feature_family_name, params)
    current_feature_set_id = feature_set_id(feature_family_name, mode, feature_cols, "none")
    feature_sets = [feature_set_row(experiment_id, current_feature_set_id, feature_family_name, mode, "none", feature_cols)]
    predictions = []
    model_runs = []

    for _, eval_row in evaluation_frame.iterrows():
        as_of_date = eval_row["date"]
        train_df = full_table[full_table["date"] < as_of_date].copy()
        x_train, y_train = sequence_samples(train_df, feature_cols, target_col, sequence_length, mode)
        sequence = prediction_sequence(full_table, as_of_date, feature_cols, sequence_length)
        if len(x_train) < min_train_rows or sequence is None:
            continue

        set_seed(seed)
        started = time.perf_counter()
        model, scaler, fit_metadata = fit_model(x_train, y_train, model_type, params, device)
        train_seconds = time.perf_counter() - started
        predict_started = time.perf_counter()
        prediction = predict(model, scaler, sequence, device)
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
            "feature_policy": "none",
            "feature_set_id": current_feature_set_id,
            "n_features": len(feature_cols),
            "n_features_before_policy": len(feature_cols),
            "n_features_after_policy": len(feature_cols),
            "representation_policy": "sequence_raw",
            "n_representation_features": len(feature_cols),
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
                "selected_feature_names_json": json.dumps(feature_cols, sort_keys=True),
                "dropped_feature_names_json": "[]",
                "feature_policy_params_json": "{}",
                "representation_params_json": json.dumps({"sequence_length": sequence_length}, sort_keys=True),
                "training_window_months": np.nan,
                "validation_strategy": "rolling_as_of_ordered_holdout",
                "early_stopping_used": fit_metadata["early_stopping_used"],
                "epochs_trained": fit_metadata["epochs_trained"],
                "best_epoch": fit_metadata["best_epoch"],
                "framework": "pytorch",
                "framework_version": torch.__version__,
                "hardware_type": "gpu" if device.type == "cuda" else "cpu",
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
                "cuda_version": torch.version.cuda or "",
                "refit_frequency_months": 1,
                "model_refit": True,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "status": "succeeded",
                "artifact_uri": "",
                "metric_extras_json": json.dumps({"validation_loss": fit_metadata["validation_loss"]}),
            }
        )

    return pd.DataFrame(predictions), pd.DataFrame(model_runs), pd.DataFrame(feature_sets)


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
    device = resolve_device(str(execution.get("device", "auto")))
    seed = int(execution.get("random_seed", 42))
    configs = model_configs_from_config(config)
    included_families = (config.get("feature_families") or {}).get("include") or []
    modes = config.get("modes") or ["raw"]

    logger.info("Using device: %s", device)
    logger.info("Feature table: %s rows, %s columns", len(feature_table), len(feature_table.columns))
    logger.info("Neural configs: %s", len(configs))

    chunks = []
    for family_name in included_families:
        feature_cols = feature_families[family_name]
        missing = [column for column in feature_cols if column not in full_table]
        if missing:
            raise ValueError(f"Missing configured features for {family_name}: {missing}")
        for mode in modes:
            for model_config in configs:
                logger.info("Running %s | %s | %s", model_config["model_type"], mode, family_name)
                chunks.append(
                    run_config(
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
                        device,
                        seed,
                    )
                )

    predictions = pd.concat([chunk[0] for chunk in chunks if not chunk[0].empty], ignore_index=True)
    model_runs = pd.concat([chunk[1] for chunk in chunks if not chunk[1].empty], ignore_index=True)
    feature_sets = pd.concat([chunk[2] for chunk in chunks if not chunk[2].empty], ignore_index=True).drop_duplicates("feature_set_id")
    feature_importance = pd.DataFrame(columns=FEATURE_IMPORTANCE_COLUMNS)
    if predictions.empty:
        raise ValueError("No neural predictions were produced.")

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
        "feature_policies": ["none"],
        "requested_feature_families": included_families,
        "model_config_count": int(len(metrics[metrics["evaluation_scope"] == "overall"])),
        "prediction_count": int(len(predictions)),
        "model_run_count": int(len(model_runs)),
        "metric_count": int(len(metrics)),
        "complexity_profile_count": int(len(complexity_profile)),
        "champion_config_id": champion["config_id"],
        "selection_rule": champion["selection_rule"],
        "runtime": {
            "compute_context": os.environ.get("COMPUTE_CONTEXT", "local"),
            "framework": "pytorch",
            "framework_version": torch.__version__,
            "hardware_type": "gpu" if device.type == "cuda" else "cpu",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            "cuda_version": torch.version.cuda or "",
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
    logger.info("Produced %s predictions across %s neural model/as-of records.", len(predictions), len(model_runs))
    logger.info("Wrote neural results to %s", results_base_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
