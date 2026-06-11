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
RAW_PREFIX = os.environ.get("FRED_RAW_PREFIX", "raw/fred_inflation")
OUTPUT_PREFIX = os.environ.get("INFLATION_OUTPUT_PREFIX", "normalized/inflation")
SOURCE_NAME = os.environ.get("FRED_SOURCE_NAME", "fred_inflation")
SERIES_ID = os.environ.get("FRED_SERIES_ID", "fred_monthly_inflation")


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
                candidates.append(obj)

    if not candidates:
        raise FileNotFoundError(f"No {filename!r} objects found under s3://{bucket}/{normalized_prefix}/")

    latest_obj = max(candidates, key=lambda obj: obj["LastModified"])
    latest_key = latest_obj["Key"]
    logger.info(
        "Resolved latest S3 key under s3://%s/%s/ to %s (LastModified=%s)",
        bucket,
        normalized_prefix,
        latest_key,
        latest_obj["LastModified"],
    )
    return latest_key


RAW_KEY = os.environ.get("FRED_RAW_KEY") or find_latest_s3_key(BUCKET_NAME, RAW_PREFIX, f"{SOURCE_NAME}.json")


def read_json_from_s3(bucket: str, key: str) -> dict:
    logger.info("Reading raw JSON from s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def normalize_fred_raw(fred_raw: dict) -> pd.DataFrame:
    series_frames = []

    for series_name, series_payload in fred_raw["series"].items():
        observations = series_payload["response"]["observations"]

        df_series = pd.DataFrame(observations)[["date", "value"]].copy()
        df_series["date"] = pd.to_datetime(df_series["date"], errors="coerce")
        df_series[series_name] = pd.to_numeric(df_series["value"], errors="coerce")
        df_series = df_series[["date", series_name]].copy()

        series_frames.append(df_series.set_index("date"))

    df = (
        pd.concat(series_frames, axis=1)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    value_cols = [c for c in df.columns if c != "date"]

    # Keep the date row even if values are missing for a real source gap like 2025-10.
    # Only drop rows with null date.
    df = df[df["date"].notna()].copy()

    df["series_id"] = SERIES_ID
    df["source_name"] = SOURCE_NAME
    df["period_end"] = df["date"] + pd.offsets.MonthEnd(0)
    df["available_at"] = df["period_end"] + pd.Timedelta(days=7)

    ordered_cols = [
        "date",
        "period_end",
        "available_at",
        "series_id",
        "source_name",
    ] + value_cols

    df = df[ordered_cols].copy()
    return df


def validate_monthly_table(df: pd.DataFrame) -> None:
    if df["date"].isna().any():
        raise ValueError("Found null dates in normalized inflation table.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Found duplicate months: {dups[:10]}")

    expected_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    actual_months = df["date"].dt.to_period("M")
    missing_months = expected_months.difference(actual_months)
    if len(missing_months) > 0:
        raise ValueError(f"Missing months detected: {missing_months.tolist()[:12]}")

    # Allow source-level missing inflation values, but log them clearly.
    value_cols = [c for c in df.columns if c.startswith("cpi_") or c.startswith("pce_")]
    missing_counts = df[value_cols].isna().sum()
    if missing_counts.any():
        logger.warning("Missing inflation values detected:\n%s", missing_counts)


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


def main() -> int:
    fred_raw = read_json_from_s3(BUCKET_NAME, RAW_KEY)
    inflation_normalized_df = normalize_fred_raw(fred_raw)
    validate_monthly_table(inflation_normalized_df)

    run_id = current_run_id()
    output_key = f"{OUTPUT_PREFIX}/run_id={run_id}/inflation_normalized.parquet"
    metadata_key = f"{OUTPUT_PREFIX}/run_id={run_id}/metadata.json"

    write_dataframe_to_s3_parquet(inflation_normalized_df, BUCKET_NAME, output_key)

    value_cols = [c for c in inflation_normalized_df.columns if c.startswith("cpi_") or c.startswith("pce_")]
    missing_counts = inflation_normalized_df[value_cols].isna().sum().to_dict()

    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "series_id": SERIES_ID,
        "raw_key": RAW_KEY,
        "output_key": output_key,
        "row_count": int(len(inflation_normalized_df)),
        "date_min": str(inflation_normalized_df["date"].min().date()),
        "date_max": str(inflation_normalized_df["date"].max().date()),
        "value_columns": value_cols,
        "missing_value_counts": missing_counts,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
    }
    write_json_to_s3(metadata, BUCKET_NAME, metadata_key)

    logger.info("Wrote normalized inflation Parquet to s3://%s/%s", BUCKET_NAME, output_key)
    logger.info("Wrote metadata to s3://%s/%s", BUCKET_NAME, metadata_key)
    logger.info("Shape: %s", inflation_normalized_df.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
