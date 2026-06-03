from __future__ import annotations

from io import BytesIO
from typing import Any

import boto3
import joblib
from botocore.client import Config

from .config import Settings
from .logging_setup import get_logger

log = get_logger(__name__)


def _s3_client(settings: Settings) -> Any:
    if not settings.s3_endpoint:
        raise ValueError("MCP_S3_ENDPOINT is required for artifact load/save")
    if not settings.s3_access_key or not settings.s3_secret_key:
        raise ValueError("MCP_S3_ACCESS_KEY / MCP_S3_SECRET_KEY (or *_FILE variants) are required")
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def _key(detector: str, uc: str) -> str:
    return f"{detector}/{uc}.joblib"


def save_model(settings: Settings, detector: str, uc: str, obj: Any) -> None:
    buf = BytesIO()
    joblib.dump(obj, buf)
    body = buf.getvalue()
    key = _key(detector, uc)
    _s3_client(settings).put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=body,
        ContentType="application/octet-stream",
    )
    log.info("model_saved", bucket=settings.s3_bucket, key=key, bytes=len(body))


def load_model(settings: Settings, detector: str, uc: str) -> Any | None:
    """Returns None when the object does not exist — score-jobs no-op in
    cold-start until the first train-job has run."""
    key = _key(detector, uc)
    client = _s3_client(settings)
    try:
        resp = client.get_object(Bucket=settings.s3_bucket, Key=key)
    except client.exceptions.NoSuchKey:
        log.info("model_missing", bucket=settings.s3_bucket, key=key)
        return None
    body = resp["Body"].read()
    log.info("model_loaded", bucket=settings.s3_bucket, key=key, bytes=len(body))
    return joblib.load(BytesIO(body))
