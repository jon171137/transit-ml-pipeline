from __future__ import annotations

import argparse
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

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
INCOME_NORMALIZED_PREFIX = os.environ.get("INCOME_NORMALIZED_PREFIX", "normalized/income")

OUTPUT_PREFIX = os.environ.get("INTEGRATED_OUTPUT_PREFIX", "integrated/monthly_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated monthly transit modeling base table.")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="S3 bucket for default inputs and S3 outputs.")
    parser.add_argument("--transit-input", default=os.environ.get("TRANSIT_KEY"), help="Local path, S3 key, or s3:// URI.")
    parser.add_argument("--gas-input", default=os.environ.get("GAS_KEY"), help="Local path, S3 key, or s3:// URI.")
    parser.add_argument("--inflation-input", default=os.environ.get("INFLATION_KEY"), help="Local path, S3 key, or s3:// URI.")
    parser.add_argument("--income-input", default=os.environ.get("INCOME_KEY"), help="Optional local path, S3 key, or s3:// URI.")
    parser.add_argument("--transit-prefix", default=TRANSIT_NORMALIZED_PREFIX)
    parser.add_argument("--gas-prefix", default=GAS_NORMALIZED_PREFIX)
    parser.add_argument("--inflation-prefix", default=INFLATION_NORMALIZED_PREFIX)
    parser.add_argument("--income-prefix", default=INCOME_NORMALIZED_PREFIX)
    parser.add_argument("--output-dir", default=None, help="Local output directory or s3:// URI. Defaults to S3.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    return parser.parse_args()


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


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    bucket_and_key = uri[len("s3://") :]
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def resolve_input(input_uri: str | None, bucket: str, prefix: str, filename: str) -> str:
    if input_uri:
        return input_uri
    return find_latest_s3_key(bucket, prefix, filename)


def read_parquet_input(bucket: str, uri_or_key: str) -> pd.DataFrame:
    if uri_or_key.startswith("s3://"):
        read_bucket, key = parse_s3_uri(uri_or_key)
        logger.info("Reading Parquet from s3://%s/%s", read_bucket, key)
        obj = s3.get_object(Bucket=read_bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    path = Path(uri_or_key)
    if path.exists():
        logger.info("Reading Parquet from %s", path)
        return pd.read_parquet(path)

    logger.info("Reading Parquet from s3://%s/%s", bucket, uri_or_key)
    obj = s3.get_object(Bucket=bucket, Key=uri_or_key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def write_dataframe_to_s3_parquet(df: pd.DataFrame, bucket: str, key: str) -> None:
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )


def write_dataframe_parquet(df: pd.DataFrame, bucket: str, uri_or_key: str) -> None:
    if uri_or_key.startswith("s3://"):
        write_bucket, key = parse_s3_uri(uri_or_key)
        write_dataframe_to_s3_parquet(df, write_bucket, key)
    elif uri_or_key.endswith(".parquet") or "/" in uri_or_key:
        path = Path(uri_or_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    else:
        write_dataframe_to_s3_parquet(df, bucket, uri_or_key)


def write_json_to_s3(payload: dict, bucket: str, key: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def write_json_uri(payload: dict, bucket: str, uri_or_key: str) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    if uri_or_key.startswith("s3://"):
        write_bucket, key = parse_s3_uri(uri_or_key)
        s3.put_object(Bucket=write_bucket, Key=key, Body=body, ContentType="application/json")
    elif uri_or_key.endswith(".json") or "/" in uri_or_key:
        path = Path(uri_or_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    else:
        write_json_to_s3(payload, bucket, uri_or_key)


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
    args = parse_args()
    transit_key = resolve_input(args.transit_input, args.bucket, args.transit_prefix, "transit_normalized.parquet")
    gas_key = resolve_input(args.gas_input, args.bucket, args.gas_prefix, "gas_monthly_normalized.parquet")
    inflation_key = resolve_input(args.inflation_input, args.bucket, args.inflation_prefix, "inflation_normalized.parquet")
    income_key = args.income_input
    if income_key is None:
        try:
            income_key = resolve_input(None, args.bucket, args.income_prefix, "income_normalized.parquet")
        except FileNotFoundError:
            logger.warning("No normalized income input found. Building integrated table without income.")

    # -----------------------------
    # 1. Read normalized source tables
    # -----------------------------
    transit_df = read_parquet_input(args.bucket, transit_key)
    gas_df = read_parquet_input(args.bucket, gas_key)
    inflation_df = read_parquet_input(args.bucket, inflation_key)
    income_df = read_parquet_input(args.bucket, income_key) if income_key else None

    # -----------------------------
    # 2. Parse dates
    # -----------------------------
    transit_df = standardize_dates(transit_df, ["date", "period_end", "available_at"])
    gas_df = standardize_dates(gas_df, ["date", "period_end", "available_at"])
    inflation_df = standardize_dates(inflation_df, ["date", "period_end", "available_at"])
    if income_df is not None:
        income_df = standardize_dates(income_df, ["date", "period_end", "available_at"])

    # -----------------------------
    # 3. Rename source-specific availability columns
    # -----------------------------
    if "available_at" in transit_df.columns:
        transit_df = transit_df.rename(columns={"available_at": "transit_available_at"})
    if "available_at" in gas_df.columns:
        gas_df = gas_df.rename(columns={"available_at": "gas_available_at"})
    if "available_at" in inflation_df.columns:
        inflation_df = inflation_df.rename(columns={"available_at": "inflation_available_at"})
    if income_df is not None and "available_at" in income_df.columns:
        income_df = income_df.rename(columns={"available_at": "income_available_at"})

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

    if income_df is not None:
        if "series_id" in income_df.columns:
            income_df = income_df.rename(columns={"series_id": "income_series_id"})
        if "source_name" in income_df.columns:
            income_df = income_df.rename(columns={"source_name": "income_source_name"})

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
    income_keep = []
    if income_df is not None:
        income_keep = [
            c for c in [
                "date",
                "income_available_at",
                "income_series_id",
                "income_source_name",
                "income_reference_year",
                "king_county_median_household_income_prior_year",
                "king_county_monthly_household_income_prior_year",
                "king_county_income_yoy_pct_prior_year",
                "king_county_income_2yr_pct_prior_year",
            ] if c in income_df.columns
        ]

    transit_df = transit_df[transit_keep].copy()
    gas_df = gas_df[gas_keep].copy()
    inflation_df = inflation_df[inflation_keep].copy()
    if income_df is not None:
        income_df = income_df[income_keep].copy()

    # -----------------------------
    # 5. Validate each source table
    # -----------------------------
    transit_df = transit_df.sort_values("date").reset_index(drop=True)
    gas_df = gas_df.sort_values("date").reset_index(drop=True)
    inflation_df = inflation_df.sort_values("date").reset_index(drop=True)
    if income_df is not None:
        income_df = income_df.sort_values("date").reset_index(drop=True)

    validate_monthly_dates(transit_df, "transit_df")
    validate_monthly_dates(gas_df, "gas_df")
    validate_monthly_dates(inflation_df, "inflation_df")
    if income_df is not None:
        validate_monthly_dates(income_df, "income_df")

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
    if income_df is not None:
        integrated_df = (
            integrated_df
            .merge(income_df, on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )

    # -----------------------------
    # 7. Create combined "all sources available" timestamp
    #    This is the earliest point when the row could be used as-of.
    # -----------------------------
    availability_cols = [
        c for c in ["transit_available_at", "gas_available_at", "inflation_available_at", "income_available_at"]
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
    if args.output_dir:
        output_base = args.output_dir.rstrip("/")
        output_key = f"{output_base}/integrated_monthly_base.parquet"
        metadata_key = f"{output_base}/metadata.json"
    else:
        output_key = f"s3://{args.bucket}/{args.output_prefix}/run_id={run_id}/integrated_monthly_base.parquet"
        metadata_key = f"s3://{args.bucket}/{args.output_prefix}/run_id={run_id}/metadata.json"

    write_dataframe_parquet(integrated_df, args.bucket, output_key)

    metadata = {
        "run_id": run_id,
        "output_key": output_key,
        "row_count": int(len(integrated_df)),
        "date_min": str(integrated_df["date"].min().date()),
        "date_max": str(integrated_df["date"].max().date()),
        "transit_key": transit_key,
        "gas_key": gas_key,
        "inflation_key": inflation_key,
        "income_key": income_key,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "missing_value_counts": {k: int(v) for k, v in missing_summary.to_dict().items()},
        "runtime": runtime_metadata(),
    }
    write_json_uri(metadata, args.bucket, metadata_key)

    logger.info("Wrote integrated base Parquet to %s", output_key)
    logger.info("Wrote metadata to %s", metadata_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
