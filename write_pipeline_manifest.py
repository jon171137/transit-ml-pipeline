import json
import logging
import os
from datetime import datetime, timezone

import boto3

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

TRANSIT_OUTPUT_PREFIX = os.environ.get("TRANSIT_OUTPUT_PREFIX", "normalized/transit")
GAS_OUTPUT_PREFIX = os.environ.get("GAS_OUTPUT_PREFIX", "normalized/gas")
INFLATION_OUTPUT_PREFIX = os.environ.get("INFLATION_OUTPUT_PREFIX", "normalized/inflation")
INTEGRATED_OUTPUT_PREFIX = os.environ.get("INTEGRATED_OUTPUT_PREFIX", "integrated/monthly_base")
FEATURE_OUTPUT_PREFIX = os.environ.get("FEATURE_OUTPUT_PREFIX", "features/integrated_monthly_h3")
PIPELINE_RUNS_PREFIX = os.environ.get("PIPELINE_RUNS_PREFIX", "pipeline_runs")


def require_run_id() -> str:
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if not run_id:
        raise ValueError("PIPELINE_RUN_ID is required to write a pipeline manifest.")
    return run_id


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def read_json_from_s3(bucket: str, key: str) -> dict:
    logger.info("Reading metadata from s3://%s/%s", bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def write_json_to_s3(payload: dict, bucket: str, key: str) -> None:
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def step_paths(run_id: str) -> dict:
    return {
        "normalize_transit": {
            "script": "normalize_transit.py",
            "metadata_key": f"{TRANSIT_OUTPUT_PREFIX}/run_id={run_id}/metadata.json",
            "output_key": f"{TRANSIT_OUTPUT_PREFIX}/run_id={run_id}/transit_normalized.parquet",
        },
        "normalize_gas": {
            "script": "normalize_eia_gas.py",
            "metadata_key": f"{GAS_OUTPUT_PREFIX}/run_id={run_id}/metadata.json",
            "output_key": f"{GAS_OUTPUT_PREFIX}/run_id={run_id}/gas_monthly_normalized.parquet",
        },
        "normalize_inflation": {
            "script": "normalize_fred_inflation.py",
            "metadata_key": f"{INFLATION_OUTPUT_PREFIX}/run_id={run_id}/metadata.json",
            "output_key": f"{INFLATION_OUTPUT_PREFIX}/run_id={run_id}/inflation_normalized.parquet",
        },
        "build_integrated_monthly_base": {
            "script": "build_integrated_monthly_base.py",
            "metadata_key": f"{INTEGRATED_OUTPUT_PREFIX}/run_id={run_id}/metadata.json",
            "output_key": f"{INTEGRATED_OUTPUT_PREFIX}/run_id={run_id}/integrated_monthly_base.parquet",
        },
        "create_feature_table": {
            "script": "create_feature_table.py",
            "metadata_key": f"{FEATURE_OUTPUT_PREFIX}/run_id={run_id}/feature_metadata.json",
            "output_key": f"{FEATURE_OUTPUT_PREFIX}/run_id={run_id}/feature_table.parquet",
        },
    }


def build_step_manifest(name: str, paths: dict, metadata: dict) -> dict:
    return {
        "script": paths["script"],
        "metadata_uri": s3_uri(BUCKET_NAME, paths["metadata_key"]),
        "output_uri": s3_uri(BUCKET_NAME, metadata.get("output_key", paths["output_key"])),
        "row_count": metadata.get("row_count") or metadata.get("feature_table", {}).get("row_count"),
        "date_min": metadata.get("date_min") or metadata.get("feature_table", {}).get("date_min"),
        "date_max": metadata.get("date_max") or metadata.get("feature_table", {}).get("date_max"),
        "metadata": metadata,
    }


def build_summary(step_manifests: dict) -> dict:
    feature_metadata = step_manifests["create_feature_table"]["metadata"]
    integrated_metadata = step_manifests["build_integrated_monthly_base"]["metadata"]
    feature_table = feature_metadata.get("feature_table", {})

    return {
        "integrated_rows": integrated_metadata.get("row_count"),
        "integrated_date_min": integrated_metadata.get("date_min"),
        "integrated_date_max": integrated_metadata.get("date_max"),
        "feature_table_rows": feature_table.get("row_count"),
        "feature_table_columns": feature_table.get("column_count"),
        "feature_table_date_min": feature_table.get("date_min"),
        "feature_table_date_max": feature_table.get("date_max"),
        "target_column": feature_table.get("target_column"),
        "target_null_count": feature_table.get("target_null_count"),
        "imputation_count": feature_metadata.get("imputation_count"),
    }


def main() -> int:
    run_id = require_run_id()
    image_uri = os.environ.get("IMAGE_URI")
    execution_arn = os.environ.get("STEP_FUNCTION_EXECUTION_ARN")
    paths_by_step = step_paths(run_id)

    steps = {}
    for name, paths in paths_by_step.items():
        metadata = read_json_from_s3(BUCKET_NAME, paths["metadata_key"])
        if metadata.get("run_id") != run_id:
            raise ValueError(
                f"Metadata run_id mismatch for {name}: expected {run_id!r}, got {metadata.get('run_id')!r}"
            )
        steps[name] = build_step_manifest(name, paths, metadata)

    manifest = {
        "run_id": run_id,
        "status": "succeeded",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_name": os.environ.get("PIPELINE_NAME", "transit-ml-pipeline"),
        "pipeline_stage": "feature_table",
        "runtime": {
            "image_uri": image_uri,
            "pipeline_run_id": run_id,
        },
        "aws": {
            "region": os.environ.get("AWS_REGION", "us-east-1"),
            "bucket": BUCKET_NAME,
            "step_function_name": os.environ.get("STEP_FUNCTION_NAME", "transit-ml-pipeline"),
            "step_function_execution_arn": execution_arn,
        },
        "steps": steps,
        "summary": build_summary(steps),
    }

    manifest_key = f"{PIPELINE_RUNS_PREFIX}/run_id={run_id}/manifest.json"
    write_json_to_s3(manifest, BUCKET_NAME, manifest_key)

    logger.info("Wrote pipeline manifest to s3://%s/%s", BUCKET_NAME, manifest_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
