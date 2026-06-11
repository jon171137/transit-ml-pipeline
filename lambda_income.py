import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
SOURCE_NAME = os.environ.get("INCOME_SOURCE_NAME", "fred_income")
FRED_API_KEY = os.environ["FRED_API_KEY"]

STATE_KEY = f"state/{SOURCE_NAME}/latest_ingested.json"
RAW_PREFIX = os.environ.get("INCOME_RAW_PREFIX", f"raw/{SOURCE_NAME}")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SERIES_MAP = {
    "king_county_median_household_income": os.environ.get(
        "FRED_INCOME_SERIES_ID",
        os.environ.get("INCOME_FRED_SERIES_ID", "MHIWA53033A052NCEN"),
    ),
}


def http_get_json(url: str) -> dict:
    with urlopen(url, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_fred_url(series_id: str) -> str:
    params = {
        "series_id": series_id,
        "observation_start": "1989-01-01",
        "file_type": "json",
        "api_key": FRED_API_KEY,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize_observations(response_json: dict) -> dict:
    observations = response_json.get("observations", [])
    observation_dates = [obs.get("date") for obs in observations if obs.get("date")]
    return {
        "observation_count": len(observations),
        "observation_date_min": min(observation_dates) if observation_dates else None,
        "observation_date_max": max(observation_dates) if observation_dates else None,
    }


def load_state() -> Optional[dict]:
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=STATE_KEY)
        body = obj["Body"].read().decode("utf-8")
        state = json.loads(body)
        logger.info("Loaded prior state: %s", state)
        return state
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("NoSuchKey", "404"):
            logger.info("No prior state found. This looks like the first run.")
            return None
        raise


def put_json(key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def lambda_handler(event, context):
    try:
        filename = f"{SOURCE_NAME}.json"

        series_payload = {}
        for name, series_id in SERIES_MAP.items():
            url = build_fred_url(series_id)
            logger.info("Fetching FRED series %s (%s)", name, series_id)
            response_json = http_get_json(url)

            if "observations" not in response_json:
                raise ValueError(f"Unexpected FRED response for {series_id}: {response_json}")

            series_payload[name] = {
                "series_id": series_id,
                "request": {
                    "base_url": BASE_URL,
                    "series_id": series_id,
                    "observation_start": "1989-01-01",
                    "file_type": "json",
                },
                "metadata": summarize_observations(response_json),
                "response": response_json,
            }

        content_bytes = json.dumps(series_payload, sort_keys=True).encode("utf-8")
        raw_hash = sha256_hex(content_bytes)

        raw_payload = {
            "source_name": SOURCE_NAME,
            "provider": "FRED",
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
            "content_sha256": raw_hash,
            "series": series_payload,
        }

        raw_bytes = json.dumps(raw_payload, sort_keys=True).encode("utf-8")

        prior_state = load_state()
        prior_hash = prior_state.get("sha256") if prior_state else None

        if prior_hash == raw_hash:
            logger.info("No new dataset content detected. Current response hash matches prior state.")
            return {
                "status": "no_change",
                "source_name": SOURCE_NAME,
                "filename": filename,
            }

        ingest_time = datetime.now(timezone.utc)
        ingest_time_str = ingest_time.isoformat()
        load_date = ingest_time.date().isoformat()

        raw_key = f"{RAW_PREFIX}/load_date={load_date}/{filename}"
        metadata_key = f"{RAW_PREFIX}/load_date={load_date}/metadata.json"

        metadata = {
            "source_name": SOURCE_NAME,
            "provider": "FRED",
            "series_ids": SERIES_MAP,
            "series_metadata": {
                name: payload["metadata"]
                for name, payload in series_payload.items()
            },
            "ingested_at_utc": ingest_time_str,
            "s3_key": raw_key,
            "file_size_bytes": len(raw_bytes),
            "sha256": raw_hash,
        }

        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=raw_key,
            Body=raw_bytes,
            ContentType="application/json",
        )
        put_json(metadata_key, metadata)
        put_json(STATE_KEY, metadata)

        logger.info("Stored raw file at s3://%s/%s", BUCKET_NAME, raw_key)
        logger.info("Updated state at s3://%s/%s", BUCKET_NAME, STATE_KEY)

        return {
            "status": "downloaded",
            "source_name": SOURCE_NAME,
            "filename": filename,
            "raw_key": raw_key,
            "metadata_key": metadata_key,
        }

    except HTTPError:
        logger.exception("HTTP error while calling FRED API.")
        raise
    except URLError:
        logger.exception("URL/network error while calling FRED API.")
        raise
    except Exception:
        logger.exception("Unhandled error in FRED income ingestion Lambda.")
        raise
