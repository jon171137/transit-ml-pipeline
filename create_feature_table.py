import argparse
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")


MODEL_BASE_COLS = [
    "date",
    "upt",
    "vrm",
    "vrh",
    "voms",
    "seattle_gas_price_avg",
    "seattle_gas_price_std",
    "cpi_all_items_sa",
    "cpi_core_sa",
]

CORE_OBSERVED_COLS = ["date", "upt", "vrm", "vrh", "voms"]
EXOGENOUS_COLS = [
    "seattle_gas_price_avg",
    "seattle_gas_price_std",
    "cpi_all_items_sa",
    "cpi_core_sa",
]

BUCKET_NAME = os.environ.get("BUCKET_NAME", "jolese-transit-ml-portfolio-367995857052-us-east-1-an")
INTEGRATED_OUTPUT_PREFIX = os.environ.get("INTEGRATED_OUTPUT_PREFIX", "integrated/monthly_base")
FEATURE_OUTPUT_PREFIX = os.environ.get("FEATURE_OUTPUT_PREFIX", "features/integrated_monthly_h3")
INTEGRATED_FILENAME = "integrated_monthly_base.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a monthly transit modeling feature table from integrated data."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Local path or s3:// URI to integrated monthly base Parquet. "
            "Defaults to latest integrated_monthly_base.parquet in S3."
        ),
    )
    parser.add_argument(
        "--bucket",
        default=BUCKET_NAME,
        help="S3 bucket for default input discovery and default S3 outputs.",
    )
    parser.add_argument(
        "--integrated-prefix",
        default=INTEGRATED_OUTPUT_PREFIX,
        help="S3 prefix containing integrated monthly base run partitions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Local output directory or s3:// URI. "
            "Defaults to s3://<bucket>/<feature-prefix>/run_id=<timestamp>/."
        ),
    )
    parser.add_argument(
        "--feature-prefix",
        default=FEATURE_OUTPUT_PREFIX,
        help="S3 prefix for default feature-table outputs.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecast horizon in months.",
    )
    parser.add_argument(
        "--leading-trim",
        type=int,
        default=16,
        help="Number of leading rows to trim after continuity validation.",
    )
    return parser.parse_args()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    bucket_and_key = uri[len("s3://"):]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


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
        raise FileNotFoundError(
            f"No {filename!r} objects found under s3://{bucket}/{normalized_prefix}/"
        )

    latest_key = max(candidates)
    logger.info("Resolved latest integrated input to s3://%s/%s", bucket, latest_key)
    return latest_key


def resolve_input_uri(args: argparse.Namespace) -> str:
    if args.input:
        return args.input

    key = find_latest_s3_key(args.bucket, args.integrated_prefix, INTEGRATED_FILENAME)
    return f"s3://{args.bucket}/{key}"


def load_integrated_data(input_uri: str) -> pd.DataFrame:
    if input_uri.startswith("s3://"):
        bucket, key = parse_s3_uri(input_uri)
        logger.info("Reading integrated input from s3://%s/%s", bucket, key)
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    else:
        path = Path(input_uri)
        if not path.exists():
            raise FileNotFoundError(f"Integrated input not found: {path}")
        logger.info("Reading integrated input from %s", path)
        df = pd.read_parquet(path)

    missing_cols = [col for col in MODEL_BASE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Integrated input missing required columns: {missing_cols}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def validate_monthly_continuity(df: pd.DataFrame) -> dict:
    if df["date"].isna().any():
        raise ValueError("Integrated input contains null dates.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Integrated input contains duplicate dates: {dups[:10]}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Integrated input dates are not sorted ascending.")

    expected_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    actual_months = df["date"].dt.to_period("M")
    missing_months = expected_months.difference(actual_months)
    if len(missing_months) > 0:
        raise ValueError(f"Integrated input is missing monthly periods: {missing_months.tolist()[:20]}")

    return {
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "row_count": int(len(df)),
        "missing_month_count": int(len(missing_months)),
    }


def trim_leading_rows(df: pd.DataFrame, leading_trim: int) -> pd.DataFrame:
    if leading_trim < 0:
        raise ValueError("--leading-trim must be non-negative.")
    if leading_trim >= len(df):
        raise ValueError(f"--leading-trim={leading_trim} would remove all rows.")

    return df.iloc[leading_trim:].copy().reset_index(drop=True)


def fill_trailing_with_recent_trend(series: pd.Series, window: int = 5) -> pd.Series:
    filled = series.copy()

    while filled.isna().any():
        missing_positions = np.flatnonzero(filled.isna().to_numpy())
        pos = missing_positions[0]

        # Interior gaps are handled by interpolation; this branch is only for trailing gaps.
        if pos < len(filled) - 1 and filled.iloc[pos + 1:].notna().any():
            break

        history = filled.iloc[:pos].dropna().tail(window)
        if len(history) >= 2:
            x = np.arange(len(history), dtype=float)
            slope, intercept = np.polyfit(x, history.to_numpy(dtype=float), 1)
            filled.iloc[pos] = intercept + slope * len(history)
        elif len(history) == 1:
            filled.iloc[pos] = history.iloc[-1]
        else:
            break

    return filled


def impute_exogenous_monthly_values(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    imputation_records = []

    for col in cols:
        original = working[col].copy()
        original_missing = original.isna()

        working[f"{col}_was_imputed"] = 0
        working[f"{col}_imputed_interpolated"] = 0
        working[f"{col}_imputed_trailing_trend"] = 0

        interpolated = original.interpolate(method="linear", limit_area="inside")
        interpolated_positions = original_missing & interpolated.notna()

        trended = fill_trailing_with_recent_trend(interpolated, window=5)
        trended_positions = original_missing & interpolated.isna() & trended.notna()

        working[col] = trended
        working.loc[original_missing & working[col].notna(), f"{col}_was_imputed"] = 1
        working.loc[interpolated_positions, f"{col}_imputed_interpolated"] = 1
        working.loc[trended_positions, f"{col}_imputed_trailing_trend"] = 1

        for idx in working.index[original_missing & working[col].notna()]:
            method = "linear_interpolation_between_observed_months"
            if trended_positions.loc[idx]:
                method = "five_month_recent_trend_projection"

            imputation_records.append(
                {
                    "date": working.loc[idx, "date"],
                    "column": col,
                    "method": method,
                    "imputed_value": float(working.loc[idx, col]),
                }
            )

    imputation_log = pd.DataFrame(
        imputation_records,
        columns=["date", "column", "method", "imputed_value"],
    )
    return working, imputation_log


def add_time_and_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date").reset_index(drop=True)

    df["time_index_months"] = np.arange(len(df))
    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    covid_start = pd.Timestamp("2020-03-01")
    post_covid_start = pd.Timestamp("2021-03-01")

    df["is_covid_disruption"] = (
        (df["date"] >= covid_start) & (df["date"] < post_covid_start)
    ).astype(int)
    df["is_post_covid"] = (df["date"] >= post_covid_start).astype(int)
    df["months_since_covid_impact"] = np.where(
        df["date"] >= covid_start,
        (df["date"].dt.year - covid_start.year) * 12 + (df["date"].dt.month - covid_start.month),
        0,
    )

    return df


def add_lagged_features_for_group(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
    rolling_windows: list[int],
    diff_lags: list[int],
    add_yoy_diff: bool = True,
    add_yoy_pct: bool = False,
    add_rollstd: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date").reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column not found in dataframe: {col}")

        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

        for lag in diff_lags:
            df[f"{col}_diff_lag{lag}"] = df[col].shift(lag) - df[col].shift(lag + 1)

        lagged = df[col].shift(1)
        for window in rolling_windows:
            df[f"{col}_rollmean_{window}"] = lagged.rolling(window).mean()
            if add_rollstd:
                df[f"{col}_rollstd_{window}"] = lagged.rolling(window).std()

        if add_yoy_diff:
            df[f"{col}_yoy_diff"] = df[col].shift(1) - df[col].shift(13)

        if add_yoy_pct:
            prev_year = df[col].shift(13)
            df[f"{col}_yoy_pct"] = np.where(
                prev_year != 0,
                (df[col].shift(1) - prev_year) / prev_year,
                np.nan,
            )

    return df


def add_target_horizon(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(date_col).reset_index(drop=True)

    df[f"{target_col}_target_h{horizon}"] = df[target_col].shift(-horizon)
    df[f"target_date_h{horizon}"] = df[date_col] + pd.DateOffset(months=horizon)
    df[f"target_month_num_h{horizon}"] = df[f"target_date_h{horizon}"].dt.month
    df[f"target_month_sin_h{horizon}"] = np.sin(2 * np.pi * df[f"target_month_num_h{horizon}"] / 12)
    df[f"target_month_cos_h{horizon}"] = np.cos(2 * np.pi * df[f"target_month_num_h{horizon}"] / 12)

    return df


def build_feature_table(df: pd.DataFrame, target_col: str = "upt", horizon: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date").reset_index(drop=True)

    df = add_time_and_regime_features(df)

    service_cols = [col for col in ["vrm", "vrh", "voms"] if col in df.columns]
    exogenous_cols = [col for col in EXOGENOUS_COLS if col in df.columns]

    df = add_lagged_features_for_group(
        df,
        cols=["upt"],
        lags=list(range(1, 13)) + [18, 24],
        rolling_windows=[3, 6],
        diff_lags=[1, 2, 3],
        add_yoy_diff=True,
        add_yoy_pct=False,
        add_rollstd=True,
    )
    df = add_lagged_features_for_group(
        df,
        cols=service_cols,
        lags=[1, 3, 6, 12],
        rolling_windows=[3, 6],
        diff_lags=[1, 3],
        add_yoy_diff=True,
        add_yoy_pct=False,
        add_rollstd=False,
    )
    df = add_lagged_features_for_group(
        df,
        cols=exogenous_cols,
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[3],
        diff_lags=[1, 3],
        add_yoy_diff=True,
        add_yoy_pct=False,
        add_rollstd=False,
    )
    df = add_target_horizon(df, target_col=target_col, horizon=horizon, date_col="date")

    return df


def build_feature_families(horizon: int) -> dict[str, list[str]]:
    feature_families = {
        "history_core": [
            "upt_lag1",
            "upt_lag2",
            "upt_lag3",
            "upt_lag6",
            "upt_lag12",
            "upt_lag18",
            "upt_lag24",
        ],
        "history_recent_deltas": ["upt_diff_lag1", "upt_diff_lag2", "upt_diff_lag3"],
        "history_rolls": ["upt_rollmean_3", "upt_rollstd_3", "upt_rollmean_6", "upt_rollstd_6"],
        "history_yoy": ["upt_yoy_diff"],
        "seasonality": ["month_sin", "month_cos", f"target_month_sin_h{horizon}", f"target_month_cos_h{horizon}"],
        "regime": ["is_covid_disruption", "is_post_covid", "months_since_covid_impact"],
        "time_trend": ["time_index_months"],
        "gas": [
            "seattle_gas_price_avg_lag1",
            "seattle_gas_price_avg_lag2",
            "seattle_gas_price_avg_lag3",
            "seattle_gas_price_avg_lag6",
            "seattle_gas_price_avg_lag12",
            "seattle_gas_price_avg_diff_lag1",
            "seattle_gas_price_avg_diff_lag3",
            "seattle_gas_price_avg_rollmean_3",
            "seattle_gas_price_avg_yoy_diff",
            "seattle_gas_price_std_lag1",
            "seattle_gas_price_std_lag2",
            "seattle_gas_price_std_lag3",
            "seattle_gas_price_std_lag6",
            "seattle_gas_price_std_lag12",
            "seattle_gas_price_std_diff_lag1",
            "seattle_gas_price_std_diff_lag3",
            "seattle_gas_price_std_rollmean_3",
            "seattle_gas_price_std_yoy_diff",
        ],
        "cpi": [
            "cpi_all_items_sa_lag1",
            "cpi_all_items_sa_lag2",
            "cpi_all_items_sa_lag3",
            "cpi_all_items_sa_lag6",
            "cpi_all_items_sa_lag12",
            "cpi_all_items_sa_diff_lag1",
            "cpi_all_items_sa_diff_lag3",
            "cpi_all_items_sa_rollmean_3",
            "cpi_all_items_sa_yoy_diff",
            "cpi_core_sa_lag1",
            "cpi_core_sa_lag2",
            "cpi_core_sa_lag3",
            "cpi_core_sa_lag6",
            "cpi_core_sa_lag12",
            "cpi_core_sa_diff_lag1",
            "cpi_core_sa_diff_lag3",
            "cpi_core_sa_rollmean_3",
            "cpi_core_sa_yoy_diff",
        ],
        "service": [
            "vrm_lag1",
            "vrm_lag3",
            "vrm_lag6",
            "vrm_lag12",
            "vrm_diff_lag1",
            "vrm_diff_lag3",
            "vrm_rollmean_3",
            "vrm_rollmean_6",
            "vrm_yoy_diff",
            "vrh_lag1",
            "vrh_lag3",
            "vrh_lag6",
            "vrh_lag12",
            "vrh_diff_lag1",
            "vrh_diff_lag3",
            "vrh_rollmean_3",
            "vrh_rollmean_6",
            "vrh_yoy_diff",
            "voms_lag1",
            "voms_lag3",
            "voms_lag6",
            "voms_lag12",
            "voms_diff_lag1",
            "voms_diff_lag3",
            "voms_rollmean_3",
            "voms_rollmean_6",
            "voms_yoy_diff",
        ],
    }

    feature_set_library = {
        "history_only": feature_families["history_core"] + feature_families["seasonality"],
        "history_recent": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["seasonality"]
        ),
        "history_rolls": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["seasonality"]
        ),
        "history_full": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
        ),
        "history_regime": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
        ),
        "history_regime_time": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["time_trend"]
        ),
        "history_regime_gas": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["gas"]
        ),
        "history_regime_cpi": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["cpi"]
        ),
        "history_regime_service": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["service"]
        ),
        "history_regime_gas_cpi": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["gas"]
            + feature_families["cpi"]
        ),
        "history_regime_gas_service": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["gas"]
            + feature_families["service"]
        ),
        "history_regime_cpi_service": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["cpi"]
            + feature_families["service"]
        ),
        "history_regime_all_exog": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["gas"]
            + feature_families["cpi"]
            + feature_families["service"]
        ),
        "history_regime_time_all_exog": (
            feature_families["history_core"]
            + feature_families["history_recent_deltas"]
            + feature_families["history_rolls"]
            + feature_families["history_yoy"]
            + feature_families["seasonality"]
            + feature_families["regime"]
            + feature_families["time_trend"]
            + feature_families["gas"]
            + feature_families["cpi"]
            + feature_families["service"]
        ),
    }

    return {
        name: list(dict.fromkeys(cols))
        for name, cols in feature_set_library.items()
    }


def build_feature_family_audit(feature_table: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_family_name": name,
                "requested_features": len(cols),
                "available_features": sum(col in feature_table.columns for col in cols),
                "missing_features": [col for col in cols if col not in feature_table.columns],
            }
            for name, cols in families.items()
        ]
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def current_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def runtime_metadata() -> dict:
    return {
        "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
        "image_uri": os.environ.get("IMAGE_URI"),
    }


def resolve_output_uri(args: argparse.Namespace, run_id: str) -> str:
    if args.output_dir:
        return args.output_dir.rstrip("/")

    prefix = args.feature_prefix.strip("/")
    return f"s3://{args.bucket}/{prefix}/run_id={run_id}"


def write_dataframe_parquet(df: pd.DataFrame, output_uri: str) -> None:
    if output_uri.startswith("s3://"):
        bucket, key = parse_s3_uri(output_uri)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
    else:
        path = Path(output_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def write_json_uri(output_uri: str, payload: dict) -> None:
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    if output_uri.startswith("s3://"):
        bucket, key = parse_s3_uri(output_uri)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    else:
        path = Path(output_uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def join_uri(base_uri: str, filename: str) -> str:
    return f"{base_uri.rstrip('/')}/{filename}"


def main() -> int:
    args = parse_args()
    input_uri = resolve_input_uri(args)
    run_id = current_run_id()
    output_uri = resolve_output_uri(args, run_id)
    if not output_uri.startswith("s3://"):
        Path(output_uri).mkdir(parents=True, exist_ok=True)

    integrated_df = load_integrated_data(input_uri)
    continuity = validate_monthly_continuity(integrated_df)
    null_counts_before = integrated_df[MODEL_BASE_COLS].isna().sum().astype(int).to_dict()

    model_base_df = integrated_df[MODEL_BASE_COLS].copy()
    trimmed_df = trim_leading_rows(model_base_df, args.leading_trim)
    null_counts_after_trim = trimmed_df.isna().sum().astype(int).to_dict()

    feature_input_df, imputation_log = impute_exogenous_monthly_values(trimmed_df, EXOGENOUS_COLS)
    null_counts_after_imputation = feature_input_df.isna().sum().astype(int).to_dict()

    core_nulls = feature_input_df[CORE_OBSERVED_COLS].isna().sum()
    if core_nulls.any():
        raise ValueError(f"Core observed transit columns contain nulls after trim:\n{core_nulls}")

    feature_table = build_feature_table(feature_input_df, target_col="upt", horizon=args.horizon)
    feature_families = build_feature_families(args.horizon)
    feature_family_audit = build_feature_family_audit(feature_table, feature_families)
    missing_family_features = feature_family_audit[feature_family_audit["missing_features"].map(bool)]
    if not missing_family_features.empty:
        raise ValueError(f"Feature families reference unavailable columns:\n{missing_family_features}")

    feature_table_uri = join_uri(output_uri, "feature_table.parquet")
    imputation_log_uri = join_uri(output_uri, "imputation_log.parquet")
    feature_family_audit_uri = join_uri(output_uri, "feature_family_audit.parquet")
    feature_families_uri = join_uri(output_uri, "feature_families.json")
    metadata_uri = join_uri(output_uri, "feature_metadata.json")

    write_dataframe_parquet(feature_table, feature_table_uri)
    write_dataframe_parquet(imputation_log, imputation_log_uri)
    write_dataframe_parquet(feature_family_audit, feature_family_audit_uri)
    write_json_uri(feature_families_uri, feature_families)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "input_uri": input_uri,
        "output_uri": output_uri,
        "horizon": int(args.horizon),
        "leading_trim": int(args.leading_trim),
        "continuity": continuity,
        "model_base_columns": MODEL_BASE_COLS,
        "core_observed_columns": CORE_OBSERVED_COLS,
        "exogenous_imputed_columns": EXOGENOUS_COLS,
        "null_counts_before_trim": null_counts_before,
        "null_counts_after_trim": null_counts_after_trim,
        "null_counts_after_imputation": null_counts_after_imputation,
        "imputation_count": int(len(imputation_log)),
        "feature_table": {
            "uri": feature_table_uri,
            "row_count": int(len(feature_table)),
            "column_count": int(len(feature_table.columns)),
            "date_min": feature_table["date"].min().date().isoformat(),
            "date_max": feature_table["date"].max().date().isoformat(),
            "target_column": f"upt_target_h{args.horizon}",
            "target_null_count": int(feature_table[f"upt_target_h{args.horizon}"].isna().sum()),
        },
        "feature_families_uri": feature_families_uri,
        "feature_family_audit_uri": feature_family_audit_uri,
        "imputation_log_uri": imputation_log_uri,
        "runtime": runtime_metadata(),
    }
    write_json_uri(metadata_uri, metadata)

    logger.info("Wrote feature table to %s", feature_table_uri)
    logger.info("Wrote feature families to %s", feature_families_uri)
    logger.info("Wrote metadata to %s", metadata_uri)
    logger.info("Feature table shape: %s", feature_table.shape)
    logger.info("Imputation count: %s", len(imputation_log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
