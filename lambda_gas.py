import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
SOURCE_NAME = os.environ["GAS_SOURCE_NAME"]
SOURCE_URL = os.environ["GAS_SOURCE_URL"]

STATE_KEY = f"state/{SOURCE_NAME}/latest_ingested.json"
RAW_PREFIX = f"raw/{SOURCE_NAME}"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def http_get_bytes(url: str) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def put_bytes(key: str, payload: bytes, content_type: str) -> None:
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=payload,
        ContentType=content_type,
    )


def lambda_handler(event, context):
    try:
        download_url = SOURCE_URL
        filename = f"{SOURCE_NAME}.xls"

        file_bytes = http_get_bytes(download_url)
        if not file_bytes:
            raise ValueError("Downloaded file is empty.")

        file_hash = sha256_hex(file_bytes)
        prior_state = load_state()
        prior_hash = prior_state.get("sha256") if prior_state else None

        if prior_hash == file_hash:
            logger.info("No new dataset content detected. Current file hash matches prior state.")
            return {
                "status": "no_change",
                "source_name": SOURCE_NAME,
                "filename": filename,
            }

        logger.info("New dataset content detected: %s", filename)

        ingest_time = datetime.now(timezone.utc)
        ingest_time_str = ingest_time.isoformat()
        load_date = ingest_time.date().isoformat()

        raw_key = f"{RAW_PREFIX}/load_date={load_date}/{filename}"
        metadata_key = f"{RAW_PREFIX}/load_date={load_date}/metadata.json"

        metadata = {
            "source_name": SOURCE_NAME,
            "download_url": download_url,
            "original_filename": filename,
            "ingested_at_utc": ingest_time_str,
            "s3_key": raw_key,
            "file_size_bytes": len(file_bytes),
            "sha256": file_hash,
        }

        put_bytes(
            key=raw_key,
            payload=file_bytes,
            content_type="application/vnd.ms-excel",
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
        }

    except HTTPError:
        logger.exception("HTTP error while fetching source file.")
        raise
    except URLError:
        logger.exception("URL/network error while fetching source file.")
        raise
    except Exception:
        logger.exception("Unhandled error in gas ingestion Lambda.")
        raise
