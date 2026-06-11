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
RAW_PREFIX = os.environ.get("GAS_RAW_PREFIX", "raw/eia_seattle_gas_prices")
OUTPUT_PREFIX = os.environ.get("GAS_OUTPUT_PREFIX", "normalized/gas")
SOURCE_NAME = os.environ.get("GAS_SOURCE_NAME", "eia_seattle_gas_prices")
SERIES_ID = os.environ.get("GAS_SERIES_ID", "seattle_all_grades_gas_price_monthly")
TARGET_SERIES_CODE = os.environ.get("TARGET_SERIES_CODE", "EMM_EPM0_PTE_Y48SE_DPG")


def current_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def runtime_metadata() -> dict:
    return {
        "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
        "image_uri": os.environ.get("IMAGE_URI"),
    }


def find_latest_s3_key(bucket: str, prefix: str, suffix: str) -> str:
    normalized_prefix = prefix.strip("/")
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized_prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffix):
                candidates.append(obj)

    if not candidates:
        raise FileNotFoundError(f"No objects ending in {suffix!r} found under s3://{bucket}/{normalized_prefix}/")

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


RAW_KEY = os.environ.get("GAS_RAW_KEY") or find_latest_s3_key(BUCKET_NAME, RAW_PREFIX, ".xls")


def read_excel_sheet_from_s3(bucket: str, key: str, sheet_name: str) -> pd.DataFrame:
    logger.info("Reading sheet '%s' from s3://%s/%s", sheet_name, bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="xlrd")


def extract_target_weekly_series(df_raw: pd.DataFrame, target_series_code: str) -> pd.DataFrame:
    """
    EIA workbook layout:
    - row 0 = series codes
    - row 1 = human-readable series names
    - row 2 onward = data
    - first column = Sourcekey / Date
    """
    working = df_raw.copy()

    series_codes = working.iloc[0].tolist()
    working.columns = series_codes
    working = working.iloc[2:].copy()

    if "Sourcekey" not in working.columns:
        raise ValueError("Expected 'Sourcekey' column not found in EIA workbook.")

    if target_series_code not in working.columns:
        raise ValueError(f"Target series code '{target_series_code}' not found in EIA workbook.")

    gas_weekly = working[["Sourcekey", target_series_code]].copy()
    gas_weekly = gas_weekly.rename(
        columns={
            "Sourcekey": "date",
            target_series_code: "seattle_gas_price",
        }
    )

    gas_weekly["date"] = pd.to_datetime(gas_weekly["date"], errors="coerce")
    gas_weekly["seattle_gas_price"] = pd.to_numeric(gas_weekly["seattle_gas_price"], errors="coerce")
    gas_weekly = gas_weekly.sort_values("date").reset_index(drop=True)

    return gas_weekly


def validate_weekly_table(df: pd.DataFrame) -> None:
    if df["date"].isna().any():
        raise ValueError("Found null dates in weekly gas table.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Found duplicate weekly dates: {dups[:10]}")

    if df["seattle_gas_price"].isna().any():
        missing_count = int(df["seattle_gas_price"].isna().sum())
        raise ValueError(f"Found {missing_count} missing gas price values.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Weekly gas dates are not sorted ascending.")


def weekly_to_monthly(df_weekly: pd.DataFrame) -> pd.DataFrame:
    working = df_weekly.copy()
    working["month"] = working["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        working.groupby("month", as_index=False)
        .agg(
            seattle_gas_price_avg=("seattle_gas_price", "mean"),
            seattle_gas_price_std=("seattle_gas_price", "std"),
            weekly_obs_count=("seattle_gas_price", "size"),
        )
        .rename(columns={"month": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    # std is NaN if only one weekly observation exists in a month; set to 0.0
    monthly["seattle_gas_price_std"] = monthly["seattle_gas_price_std"].fillna(0.0)

    return monthly


def validate_monthly_table(df: pd.DataFrame) -> None:
    if df["date"].isna().any():
        raise ValueError("Found null dates in monthly gas table.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Found duplicate monthly dates: {dups[:10]}")

    expected_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    actual_months = df["date"].dt.to_period("M")
    missing_months = expected_months.difference(actual_months)
    if len(missing_months) > 0:
        raise ValueError(f"Missing months detected: {missing_months.tolist()[:12]}")

    metric_cols = ["seattle_gas_price_avg", "seattle_gas_price_std", "weekly_obs_count"]
    missing_counts = df[metric_cols].isna().sum()
    if missing_counts.any():
        raise ValueError(f"Missing monthly gas values detected:\n{missing_counts}")


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
    # Load workbook sheet
    eia_raw_df = read_excel_sheet_from_s3(BUCKET_NAME, RAW_KEY, "Data 1")

    # Extract target weekly series
    gas_weekly_df = extract_target_weekly_series(eia_raw_df, TARGET_SERIES_CODE)
    validate_weekly_table(gas_weekly_df)

    # Aggregate to monthly
    gas_monthly_df = weekly_to_monthly(gas_weekly_df)

    # Add metadata fields
    gas_monthly_df["series_id"] = SERIES_ID
    gas_monthly_df["source_name"] = SOURCE_NAME
    gas_monthly_df["period_end"] = gas_monthly_df["date"] + pd.offsets.MonthEnd(0)
    gas_monthly_df["available_at"] = gas_monthly_df["period_end"] + pd.Timedelta(days=7)

    gas_monthly_df = gas_monthly_df[
        [
            "date",
            "period_end",
            "available_at",
            "series_id",
            "source_name",
            "seattle_gas_price_avg",
            "seattle_gas_price_std",
            "weekly_obs_count",
        ]
    ].copy()

    validate_monthly_table(gas_monthly_df)

    run_id = current_run_id()
    output_key = f"{OUTPUT_PREFIX}/run_id={run_id}/gas_monthly_normalized.parquet"
    metadata_key = f"{OUTPUT_PREFIX}/run_id={run_id}/metadata.json"

    write_dataframe_to_s3_parquet(gas_monthly_df, BUCKET_NAME, output_key)

    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "series_id": SERIES_ID,
        "target_series_code": TARGET_SERIES_CODE,
        "raw_key": RAW_KEY,
        "output_key": output_key,
        "row_count": int(len(gas_monthly_df)),
        "date_min": str(gas_monthly_df["date"].min().date()),
        "date_max": str(gas_monthly_df["date"].max().date()),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
    }
    write_json_to_s3(metadata, BUCKET_NAME, metadata_key)

    logger.info("Wrote normalized gas Parquet to s3://%s/%s", BUCKET_NAME, output_key)
    logger.info("Wrote metadata to s3://%s/%s", BUCKET_NAME, metadata_key)
    logger.info("Shape: %s", gas_monthly_df.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
