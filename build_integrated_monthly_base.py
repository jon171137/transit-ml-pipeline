import io
import json
import logging
import os
from datetime import datetime, timezone

import boto3
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

BUCKET_NAME = os.environ["BUCKET_NAME"]

TRANSIT_NORMALIZED_PREFIX = os.environ.get("TRANSIT_NORMALIZED_PREFIX", "normalized/transit")
GAS_NORMALIZED_PREFIX = os.environ.get("GAS_NORMALIZED_PREFIX", "normalized/gas")
INFLATION_NORMALIZED_PREFIX = os.environ.get("INFLATION_NORMALIZED_PREFIX", "normalized/inflation")

OUTPUT_PREFIX = os.environ.get("INTEGRATED_OUTPUT_PREFIX", "integrated/monthly_base")


def current_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def runtime_metadata() -> dict:
    return {
        "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
        "image_uri": os.environ.get("IMAGE_URI"),
    }


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
    logger.info("Resolved latest S3 key under s3://%s/%s/ to %s", bucket, normalized_prefix, latest_key)
    return latest_key


TRANSIT_KEY = os.environ.get("TRANSIT_KEY") or find_latest_s3_key(
    BUCKET_NAME,
    TRANSIT_NORMALIZED_PREFIX,
    "transit_normalized.parquet",
)
GAS_KEY = os.environ.get("GAS_KEY") or find_latest_s3_key(
    BUCKET_NAME,
    GAS_NORMALIZED_PREFIX,
    "gas_monthly_normalized.parquet",
)
INFLATION_KEY = os.environ.get("INFLATION_KEY") or find_latest_s3_key(
    BUCKET_NAME,
    INFLATION_NORMALIZED_PREFIX,
    "inflation_normalized.parquet",
)


def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    logger.info("Reading Parquet from s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def write_dataframe_to_s3_parquet(df: pd.DataFrame, bucket: str, key: str) -> None:
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )


def write_json_to_s3(payload: dict, bucket: str, key: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def standardize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    working = df.copy()
    for col in cols:
        if col in working.columns:
            working[col] = pd.to_datetime(working[col], errors="coerce")
    return working


def validate_monthly_dates(df: pd.DataFrame, name: str) -> None:
    if "date" not in df.columns:
        raise ValueError(f"{name} missing 'date' column.")

    if df["date"].isna().any():
        raise ValueError(f"{name} contains null dates.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"{name} has duplicate dates: {dups[:10]}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{name} dates are not sorted ascending.")


def main() -> int:
    # -----------------------------
    # 1. Read normalized source tables
    # -----------------------------
    transit_df = read_parquet_from_s3(BUCKET_NAME, TRANSIT_KEY)
    gas_df = read_parquet_from_s3(BUCKET_NAME, GAS_KEY)
    inflation_df = read_parquet_from_s3(BUCKET_NAME, INFLATION_KEY)

    # -----------------------------
    # 2. Parse dates
    # -----------------------------
    transit_df = standardize_dates(transit_df, ["date", "period_end", "available_at"])
    gas_df = standardize_dates(gas_df, ["date", "period_end", "available_at"])
    inflation_df = standardize_dates(inflation_df, ["date", "period_end", "available_at"])

    # -----------------------------
    # 3. Rename source-specific availability columns
    # -----------------------------
    if "available_at" in transit_df.columns:
        transit_df = transit_df.rename(columns={"available_at": "transit_available_at"})
    if "available_at" in gas_df.columns:
        gas_df = gas_df.rename(columns={"available_at": "gas_available_at"})
    if "available_at" in inflation_df.columns:
        inflation_df = inflation_df.rename(columns={"available_at": "inflation_available_at"})

    # Optional: rename source_name / series_id columns so they don't collide
    if "series_id" in transit_df.columns:
        transit_df = transit_df.rename(columns={"series_id": "transit_series_id"})
    if "source_name" in transit_df.columns:
        transit_df = transit_df.rename(columns={"source_name": "transit_source_name"})

    if "series_id" in gas_df.columns:
        gas_df = gas_df.rename(columns={"series_id": "gas_series_id"})
    if "source_name" in gas_df.columns:
        gas_df = gas_df.rename(columns={"source_name": "gas_source_name"})

    if "series_id" in inflation_df.columns:
        inflation_df = inflation_df.rename(columns={"series_id": "inflation_series_id"})
    if "source_name" in inflation_df.columns:
        inflation_df = inflation_df.rename(columns={"source_name": "inflation_source_name"})

    # -----------------------------
    # 4. Keep useful columns only
    # -----------------------------
    transit_keep = [
        c for c in [
            "date",
            "period_end",
            "transit_available_at",
            "transit_series_id",
            "transit_source_name",
            "upt",
            "vrm",
            "vrh",
            "voms",
        ] if c in transit_df.columns
    ]
    gas_keep = [
        c for c in [
            "date",
            "gas_available_at",
            "gas_series_id",
            "gas_source_name",
            "seattle_gas_price_avg",
            "seattle_gas_price_std",
            "weekly_obs_count",
        ] if c in gas_df.columns
    ]
    inflation_keep = [
        c for c in [
            "date",
            "inflation_available_at",
            "inflation_series_id",
            "inflation_source_name",
            "cpi_all_items_sa",
            "cpi_core_sa",
        ] if c in inflation_df.columns
    ]

    transit_df = transit_df[transit_keep].copy()
    gas_df = gas_df[gas_keep].copy()
    inflation_df = inflation_df[inflation_keep].copy()

    # -----------------------------
    # 5. Validate each source table
    # -----------------------------
    transit_df = transit_df.sort_values("date").reset_index(drop=True)
    gas_df = gas_df.sort_values("date").reset_index(drop=True)
    inflation_df = inflation_df.sort_values("date").reset_index(drop=True)

    validate_monthly_dates(transit_df, "transit_df")
    validate_monthly_dates(gas_df, "gas_df")
    validate_monthly_dates(inflation_df, "inflation_df")

    # -----------------------------
    # 6. Merge on monthly date
    # -----------------------------
    integrated_df = (
        transit_df
        .merge(gas_df, on="date", how="left")
        .merge(inflation_df, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------
    # 7. Create combined "all sources available" timestamp
    #    This is the earliest point when the row could be used as-of.
    # -----------------------------
    availability_cols = [
        c for c in ["transit_available_at", "gas_available_at", "inflation_available_at"]
        if c in integrated_df.columns
    ]
    integrated_df["data_available_at"] = integrated_df[availability_cols].max(axis=1)

    # -----------------------------
    # 8. Final validation / inspection metrics
    # -----------------------------
    if integrated_df["date"].duplicated().any():
        raise ValueError("Integrated table has duplicate dates.")

    missing_summary = integrated_df.isna().sum()
    logger.info("Integrated table shape: %s", integrated_df.shape)
    logger.info("Date range: %s to %s", integrated_df["date"].min(), integrated_df["date"].max())
    logger.info("Missing values summary:\n%s", missing_summary)

    # -----------------------------
    # 9. Write output
    # -----------------------------
    run_id = current_run_id()
    output_key = f"{OUTPUT_PREFIX}/run_id={run_id}/integrated_monthly_base.parquet"
    metadata_key = f"{OUTPUT_PREFIX}/run_id={run_id}/metadata.json"

    write_dataframe_to_s3_parquet(integrated_df, BUCKET_NAME, output_key)

    metadata = {
        "run_id": run_id,
        "output_key": output_key,
        "row_count": int(len(integrated_df)),
        "date_min": str(integrated_df["date"].min().date()),
        "date_max": str(integrated_df["date"].max().date()),
        "transit_key": TRANSIT_KEY,
        "gas_key": GAS_KEY,
        "inflation_key": INFLATION_KEY,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_value_counts": {k: int(v) for k, v in missing_summary.to_dict().items()},
        "runtime": runtime_metadata(),
    }
    write_json_to_s3(metadata, BUCKET_NAME, metadata_key)

    logger.info("Wrote integrated base Parquet to s3://%s/%s", BUCKET_NAME, output_key)
    logger.info("Wrote metadata to s3://%s/%s", BUCKET_NAME, metadata_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
