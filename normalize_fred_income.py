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

BUCKET_NAME = os.environ.get("BUCKET_NAME")
RAW_PREFIX = os.environ.get("INCOME_RAW_PREFIX", "raw/fred_income")
OUTPUT_PREFIX = os.environ.get("INCOME_OUTPUT_PREFIX", "normalized/income")
SOURCE_NAME = os.environ.get("INCOME_SOURCE_NAME", "fred_income")
SERIES_ID = os.environ.get("INCOME_SERIES_ID", "king_county_median_household_income")
RAW_FILENAME = f"{SOURCE_NAME}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize annual FRED King County income data to monthly context.")
    parser.add_argument("--raw-json", default=None, help="Local raw FRED JSON path. Defaults to latest S3 raw income JSON.")
    parser.add_argument("--bucket", default=BUCKET_NAME, help="S3 bucket for default raw input and S3 outputs.")
    parser.add_argument("--raw-prefix", default=RAW_PREFIX, help="S3 prefix containing raw income JSON.")
    parser.add_argument("--output-dir", default=None, help="Local output directory or s3:// URI.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="S3 output prefix when --output-dir is omitted.")
    parser.add_argument("--monthly-start", default="2002-01-01", help="First monthly row to emit.")
    parser.add_argument("--monthly-end", default=None, help="Optional last monthly row. Defaults to latest reference year + 1.")
    return parser.parse_args()


def current_run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def runtime_metadata() -> dict:
    return {
        "pipeline_run_id": os.environ.get("PIPELINE_RUN_ID"),
        "image_uri": os.environ.get("IMAGE_URI"),
    }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    bucket_and_key = uri[len("s3://") :]
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


def read_json_uri(uri: str, bucket: str | None = None) -> dict:
    if uri.startswith("s3://"):
        read_bucket, key = parse_s3_uri(uri)
        logger.info("Reading raw JSON from s3://%s/%s", read_bucket, key)
        obj = s3.get_object(Bucket=read_bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    if bucket and not Path(uri).exists():
        logger.info("Reading raw JSON from s3://%s/%s", bucket, uri)
        obj = s3.get_object(Bucket=bucket, Key=uri)
        return json.loads(obj["Body"].read().decode("utf-8"))

    path = Path(uri)
    logger.info("Reading raw JSON from %s", path)
    return json.loads(path.read_text(encoding="utf-8"))


def fred_raw_to_annual(fred_raw: dict) -> pd.DataFrame:
    frames = []
    for series_name, series_payload in fred_raw["series"].items():
        observations = series_payload["response"]["observations"]
        series = pd.DataFrame(observations)[["date", "value"]].copy()
        series["date"] = pd.to_datetime(series["date"], errors="coerce")
        series["year"] = series["date"].dt.year
        series[series_name] = pd.to_numeric(series["value"], errors="coerce")
        frames.append(series[["year", series_name]].set_index("year"))

    annual = pd.concat(frames, axis=1).reset_index().sort_values("year").reset_index(drop=True)
    annual = annual.dropna(subset=["year"]).copy()
    annual["year"] = annual["year"].astype(int)
    return annual


def annual_to_prior_year_monthly(annual: pd.DataFrame, monthly_start: str, monthly_end: str | None) -> pd.DataFrame:
    income_col = "king_county_median_household_income"
    if income_col not in annual.columns:
        raise ValueError(f"Expected income column {income_col!r} in annual FRED data.")

    start = pd.Timestamp(monthly_start).to_period("M").to_timestamp()
    if monthly_end:
        end = pd.Timestamp(monthly_end).to_period("M").to_timestamp()
    else:
        end = pd.Timestamp(year=int(annual["year"].max()) + 1, month=12, day=1)

    annual = annual.sort_values("year").copy()
    annual["income_reference_method"] = "observed"
    required_years = range(start.year - 1, end.year)
    missing_future_years = [year for year in required_years if year > int(annual["year"].max())]
    if missing_future_years:
        recent_diffs = annual[income_col].diff().dropna().tail(5)
        annual_step = float(recent_diffs.mean())
        last_year = int(annual["year"].max())
        last_income = float(annual.loc[annual["year"] == last_year, income_col].iloc[0])
        projected_rows = []
        for year in missing_future_years:
            last_income = last_income + annual_step
            projected_rows.append(
                {
                    "year": year,
                    income_col: last_income,
                    "income_reference_method": "projected_5yr_dollar_trend",
                }
            )
        annual = pd.concat([annual, pd.DataFrame(projected_rows)], ignore_index=True)

    annual = annual.sort_values("year").copy()
    annual["income_yoy_pct"] = annual[income_col].pct_change(1, fill_method=None)
    annual["income_2yr_pct"] = annual[income_col].pct_change(2, fill_method=None)

    annual_by_year = annual.set_index("year")

    monthly = pd.DataFrame({"date": pd.date_range(start, end, freq="MS")})
    monthly["income_reference_year"] = monthly["date"].dt.year - 1
    first_available_year = int(annual["year"].min())
    if (monthly["income_reference_year"] < first_available_year).any():
        raise ValueError("Monthly start requires income reference years before FRED series starts.")

    monthly["king_county_median_household_income_prior_year"] = monthly["income_reference_year"].map(
        annual_by_year[income_col]
    )
    monthly["income_reference_method"] = monthly["income_reference_year"].map(
        annual_by_year["income_reference_method"]
    )
    monthly["king_county_monthly_household_income_prior_year"] = (
        monthly["king_county_median_household_income_prior_year"] / 12
    )
    monthly["king_county_income_yoy_pct_prior_year"] = monthly["income_reference_year"].map(
        annual_by_year["income_yoy_pct"]
    )
    monthly["king_county_income_2yr_pct_prior_year"] = monthly["income_reference_year"].map(
        annual_by_year["income_2yr_pct"]
    )

    monthly["series_id"] = SERIES_ID
    monthly["source_name"] = SOURCE_NAME
    monthly["period_end"] = monthly["date"] + pd.offsets.MonthEnd(0)
    monthly["available_at"] = monthly["period_end"]

    ordered_cols = [
        "date",
        "period_end",
        "available_at",
        "series_id",
        "source_name",
        "income_reference_year",
        "income_reference_method",
        "king_county_median_household_income_prior_year",
        "king_county_monthly_household_income_prior_year",
        "king_county_income_yoy_pct_prior_year",
        "king_county_income_2yr_pct_prior_year",
    ]
    return monthly[ordered_cols].copy()


def validate_income_monthly(df: pd.DataFrame) -> None:
    if df["date"].isna().any():
        raise ValueError("Found null dates in normalized income table.")
    if df["date"].duplicated().any():
        raise ValueError("Found duplicate months in normalized income table.")

    expected_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    actual_months = df["date"].dt.to_period("M")
    missing_months = expected_months.difference(actual_months)
    if len(missing_months) > 0:
        raise ValueError(f"Missing months detected: {missing_months.tolist()[:12]}")

    value_cols = [c for c in df.columns if c.startswith("king_county_")]
    missing_counts = df[value_cols].isna().sum()
    if missing_counts.any():
        raise ValueError(f"Missing normalized income values detected:\n{missing_counts}")


def output_base_uri(args: argparse.Namespace, run_id: str) -> str:
    if args.output_dir:
        return args.output_dir.rstrip("/")
    if not args.bucket:
        raise ValueError("BUCKET_NAME or --bucket is required for S3 output.")
    return f"s3://{args.bucket}/{args.output_prefix.strip('/')}/run_id={run_id}"


def write_dataframe_parquet(df: pd.DataFrame, uri: str) -> None:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue(), ContentType="application/vnd.apache.parquet")
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def write_json_uri(payload: dict, uri: str) -> None:
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def join_uri(base_uri: str, filename: str) -> str:
    return f"{base_uri.rstrip('/')}/{filename}"


def main() -> int:
    args = parse_args()
    if args.raw_json:
        raw_uri = args.raw_json
    else:
        if not args.bucket:
            raise ValueError("BUCKET_NAME or --bucket is required when --raw-json is omitted.")
        raw_uri = find_latest_s3_key(args.bucket, args.raw_prefix, RAW_FILENAME)

    fred_raw = read_json_uri(raw_uri, bucket=args.bucket)
    annual_df = fred_raw_to_annual(fred_raw)
    income_normalized_df = annual_to_prior_year_monthly(annual_df, args.monthly_start, args.monthly_end)
    validate_income_monthly(income_normalized_df)

    run_id = current_run_id()
    base_uri = output_base_uri(args, run_id)
    output_uri = join_uri(base_uri, "income_normalized.parquet")
    metadata_uri = join_uri(base_uri, "metadata.json")

    write_dataframe_parquet(income_normalized_df, output_uri)

    value_cols = [c for c in income_normalized_df.columns if c.startswith("king_county_")]
    metadata = {
        "run_id": run_id,
        "source_name": SOURCE_NAME,
        "series_id": SERIES_ID,
        "raw_uri": raw_uri,
        "output_uri": output_uri,
        "row_count": int(len(income_normalized_df)),
        "date_min": str(income_normalized_df["date"].min().date()),
        "date_max": str(income_normalized_df["date"].max().date()),
        "annual_date_min": int(annual_df["year"].min()),
        "annual_date_max": int(annual_df["year"].max()),
        "value_columns": value_cols,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_metadata(),
    }
    write_json_uri(metadata, metadata_uri)

    logger.info("Wrote normalized income Parquet to %s", output_uri)
    logger.info("Wrote metadata to %s", metadata_uri)
    logger.info("Shape: %s", income_normalized_df.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
