import io
import json
import logging
import os
import re
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
RAW_PREFIX = os.environ.get("TRANSIT_RAW_PREFIX", "raw/transit")
OUTPUT_PREFIX = os.environ.get("TRANSIT_OUTPUT_PREFIX", "normalized/transit")
SOURCE_NAME = os.environ.get("TRANSIT_SOURCE_NAME", "ntd_complete_monthly_ridership_workbook")
SERIES_ID = os.environ.get("TRANSIT_SERIES_ID", "king_county_mb_do_bus")


def current_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def runtime_metadata() -> dict:
    return {
        "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
        "image_uri": os.environ.get("IMAGE_URI"),
    }


def find_latest_s3_key(bucket: str, prefix: str, suffixes: tuple[str, ...]) -> str:
    normalized_prefix = prefix.strip("/")
    paginator = s3.get_paginator("list_objects_v2")
    candidates = []

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{normalized_prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(suffixes):
                candidates.append(obj)

    if not candidates:
        raise FileNotFoundError(f"No workbook objects found under s3://{bucket}/{normalized_prefix}/")

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


RAW_KEY = os.environ.get("TRANSIT_RAW_KEY") or find_latest_s3_key(BUCKET_NAME, RAW_PREFIX, (".xlsx", ".xls"))


def read_excel_sheet_from_s3(bucket: str, key: str, sheet_name: str) -> pd.DataFrame:
    logger.info("Reading sheet '%s' from s3://%s/%s", sheet_name, bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def filter_kcb_mb_do(df: pd.DataFrame) -> pd.DataFrame:
    df_filt = df.copy()
    df_filt.columns = [str(c).strip() for c in df_filt.columns]

    required_cols = ["Agency", "Mode", "TOS", "3 Mode", "Mode/Type of Service Status"]
    missing = [c for c in required_cols if c not in df_filt.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_filt = df_filt[
        df_filt["Agency"].astype(str).str.strip().eq("King County")
        & df_filt["Mode"].astype(str).str.strip().eq("MB")
        & df_filt["TOS"].astype(str).str.strip().eq("DO")
        & df_filt["3 Mode"].astype(str).str.strip().eq("Bus")
        & df_filt["Mode/Type of Service Status"].astype(str).str.strip().eq("Active")
    ].copy()

    if len(df_filt) != 1:
        raise ValueError(f"Expected exactly 1 filtered row, found {len(df_filt)}")

    return df_filt


def detect_month_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if isinstance(c, str) and re.fullmatch(r"\d{1,2}/\d{4}", c)
    ]


def one_row_wide_to_long(df_one_row: pd.DataFrame, month_cols: list[str], value_name: str) -> pd.DataFrame:
    long_df = df_one_row.melt(
        value_vars=month_cols,
        var_name="month",
        value_name=value_name
    ).copy()

    long_df["date"] = pd.to_datetime(long_df["month"], format="%m/%Y", errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")

    long_df = long_df[["date", value_name]].sort_values("date").reset_index(drop=True)
    return long_df


def validate_monthly_table(df: pd.DataFrame) -> None:
    if df["date"].isna().any():
        raise ValueError("Found null dates in normalized transit table.")

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].tolist()
        raise ValueError(f"Found duplicate months: {dups[:10]}")

    expected_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    actual_months = df["date"].dt.to_period("M")
    missing_months = expected_months.difference(actual_months)
    if len(missing_months) > 0:
        raise ValueError(f"Missing months detected: {missing_months.tolist()[:12]}")

    metric_cols = ["upt", "vrm", "vrh", "voms"]
    missing_metrics = df[metric_cols].isna().sum()
    if missing_metrics.any():
        raise ValueError(f"Missing metric values detected:\n{missing_metrics}")


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
    # Load workbook sheets
    transit_upt_df = read_excel_sheet_from_s3(BUCKET_NAME, RAW_KEY, "UPT")
    transit_vrm_df = read_excel_sheet_from_s3(BUCKET_NAME, RAW_KEY, "VRM")
    transit_vrh_df = read_excel_sheet_from_s3(BUCKET_NAME, RAW_KEY, "VRH")
    transit_voms_df = read_excel_sheet_from_s3(BUCKET_NAME, RAW_KEY, "VOMS")

    # Filter to the target row in each sheet
    upt_kcb_mb_do = filter_kcb_mb_do(transit_upt_df)
    vrm_kcb_mb_do = filter_kcb_mb_do(transit_vrm_df)
    vrh_kcb_mb_do = filter_kcb_mb_do(transit_vrh_df)
    voms_kcb_mb_do = filter_kcb_mb_do(transit_voms_df)

    # Month columns
    month_cols = detect_month_cols(upt_kcb_mb_do)
    if not month_cols:
        raise ValueError("No month columns detected in filtered UPT sheet.")

    # Reshape
    upt_long = one_row_wide_to_long(upt_kcb_mb_do, month_cols, "upt")
    vrm_long = one_row_wide_to_long(vrm_kcb_mb_do, month_cols, "vrm")
    vrh_long = one_row_wide_to_long(vrh_kcb_mb_do, month_cols, "vrh")
    voms_long = one_row_wide_to_long(voms_kcb_mb_do, month_cols, "voms")

    # Merge
    transit_normalized_df = (
        upt_long
        .merge(vrm_long, on="date", how="inner")
        .merge(vrh_long, on="date", how="inner")
        .merge(voms_long, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    transit_normalized_df["series_id"] = SERIES_ID
    transit_normalized_df["source_name"] = SOURCE_NAME
    transit_normalized_df["period_end"] = transit_normalized_df["date"] + pd.offsets.MonthEnd(0)
    transit_normalized_df["available_at"] = transit_normalized_df["period_end"] + pd.Timedelta(days=7)

    transit_normalized_df = transit_normalized_df[
        ["date", "period_end", "available_at", "series_id", "source_name", "upt", "vrm", "vrh", "voms"]
    ].copy()

    validate_monthly_table(transit_normalized_df)

    run_id = current_run_id()
    output_key = f"{OUTPUT_PREFIX}/run_id={run_id}/transit_normalized.parquet"
    metadata_key = f"{OUTPUT_PREFIX}/run_id={run_id}/metadata.json"

    write_dataframe_to_s3_parquet(transit_normalized_df, BUCKET_NAME, output_key)

    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "series_id": SERIES_ID,
        "raw_key": RAW_KEY,
        "output_key": output_key,
        "row_count": int(len(transit_normalized_df)),
        "date_min": str(transit_normalized_df['date'].min().date()),
        "date_max": str(transit_normalized_df['date'].max().date()),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
    }
    write_json_to_s3(metadata, BUCKET_NAME, metadata_key)

    logger.info("Wrote normalized transit Parquet to s3://%s/%s", BUCKET_NAME, output_key)
    logger.info("Wrote metadata to s3://%s/%s", BUCKET_NAME, metadata_key)
    logger.info("Shape: %s", transit_normalized_df.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
