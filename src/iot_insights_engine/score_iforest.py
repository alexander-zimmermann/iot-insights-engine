"""Score the last completed 1h-bucket against the persisted IsolationForest
model for each (uc, group). INSERT idempotent into mcp_anomalies, publish
on `anomaly.<uc>.<severity>` only on new inserts or severity escalations.

Warm-up: until `warmup_days` elapsed since trained_at, demote critical
and warning to info — early models trained on partial baselines produce
false-positive spikes that swamp the bus on day 1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import psycopg

from . import artifacts, iforest_common, nats_publisher
from .config import Settings
from .db_write import write_connection
from .iforest_common import DETECTOR_NAME, ModelEnvelope
from .logging_setup import get_logger
from .registry import IFOREST_USECASES, IsolationForestUseCase
from .severity import escalated, meets_floor

log = get_logger(__name__)


def _load_last_bucket(
    conn: psycopg.Connection[Any], uc: IsolationForestUseCase
) -> list[dict[str, Any]]:
    """Newest closed 1h-bucket per group. CAGG refresh end_offset is 1h,
    so we filter `bucket <= date_trunc('hour', now()) - interval '1 hour'`."""
    sql = iforest_common.select_features_sql(
        uc,
        where_extra=(
            "bucket = ("
            f"  SELECT max(bucket) FROM {uc.source_cagg}"
            "   WHERE bucket <= date_trunc('hour', now()) - interval '1 hour'"
            ")"
        ),
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def _classify(
    envelope: ModelEnvelope, score: float, severity_floor: str = "info"
) -> str | None:
    if score < envelope.threshold_critical:
        severity = "critical"
    elif score < envelope.threshold_warning:
        severity = "warning"
    else:
        return None
    if not meets_floor(severity, severity_floor):
        return None
    return severity


def _warmup_demote(envelope: ModelEnvelope, severity: str, warmup_days: int) -> str:
    trained_at = datetime.fromisoformat(envelope.trained_at)
    if datetime.now(tz=UTC) - trained_at < timedelta(days=warmup_days):
        return "info"
    return severity


def _insert_anomaly(
    conn: psycopg.Connection[Any],
    uc: IsolationForestUseCase,
    bucket: Any,
    group_values: tuple[Any, ...],
    score: float,
    severity: str,
    features: dict[str, float],
) -> tuple[bool, str | None]:
    """Returns (inserted, old_severity) — old_severity is the
    pre-statement value on conflict (NULL on fresh insert) so the
    caller can re-publish on escalation."""
    metric_with_group = (
        f"{uc.uc}[{','.join(str(v) for v in group_values)}]" if group_values else uc.uc
    )
    payload = {
        "group": dict(zip(uc.group_cols, group_values, strict=True)),
        "score_samples": score,
        "features": features,
    }
    sql = """
        WITH existing AS (
            SELECT severity FROM mcp_anomalies
            WHERE time = %s AND source = %s AND metric = %s AND detector = %s
        )
        INSERT INTO mcp_anomalies (
            time, source, metric, detector, severity, uc,
            actual, expected, score, payload
        )
        VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL, %s, %s::jsonb)
        ON CONFLICT (time, source, metric, detector) DO UPDATE
        SET severity = EXCLUDED.severity,
            score    = EXCLUDED.score,
            payload  = EXCLUDED.payload,
            uc       = EXCLUDED.uc
        RETURNING xmax = 0 AS inserted,
                  (SELECT severity FROM existing) AS old_severity
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                bucket,
                uc.source_cagg,
                metric_with_group,
                DETECTOR_NAME,
                bucket,
                uc.source_cagg,
                metric_with_group,
                DETECTOR_NAME,
                severity,
                uc.uc,
                score,
                json.dumps(payload),
            ),
        )
        result = cur.fetchone()
    if not result:
        return False, None
    return bool(result["inserted"]), result["old_severity"]


def _score_group(
    settings: Settings,
    conn: psycopg.Connection[Any],
    uc: IsolationForestUseCase,
    rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """Returns (inserted, published)."""
    fnames = iforest_common.feature_names(uc)
    inserted_count = 0
    published_count = 0
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        gvals = tuple(row[c] for c in uc.group_cols)
        by_group.setdefault(gvals, []).append(row)
    for gvals, group_rows in by_group.items():
        key = iforest_common.group_key(uc, gvals)
        envelope: ModelEnvelope | None = artifacts.load_model(settings, DETECTOR_NAME, key)
        if envelope is None:
            continue
        feature_matrix = np.asarray(
            [[float(r[c]) for c in fnames] for r in group_rows], dtype=np.float64
        )
        scores = envelope.pipeline.score_samples(feature_matrix)
        for row, score in zip(group_rows, scores, strict=True):
            severity = _classify(envelope, float(score), uc.severity_floor)
            if severity is None:
                continue
            severity = _warmup_demote(envelope, severity, uc.warmup_days)
            row_features = {c: float(row[c]) for c in fnames}
            inserted, old_severity = _insert_anomaly(
                conn, uc, row["bucket"], gvals, float(score), severity, row_features
            )
            if not inserted and not escalated(old_severity, severity):
                continue
            if inserted:
                inserted_count += 1
            try:
                nats_publisher.publish_anomaly(
                    settings,
                    uc=uc.uc,
                    severity=severity,
                    payload={
                        "source": uc.source_cagg,
                        "score_samples": float(score),
                        "group": dict(zip(uc.group_cols, gvals, strict=True)),
                        "bucket": row["bucket"].isoformat(),
                        "features": row_features,
                    },
                )
                published_count += 1
            except Exception:
                log.exception("nats_publish_failed", uc=uc.uc)
    return inserted_count, published_count


def run(settings: Settings, _argv: Sequence[str]) -> int:
    total_inserted = 0
    total_published = 0
    with write_connection(settings) as conn:
        for uc in IFOREST_USECASES:
            if uc.silenced:
                log.info("uc_silenced", uc=uc.uc)
                continue
            try:
                rows = _load_last_bucket(conn, uc)
            except psycopg.Error:
                log.exception("iforest_load_failed", uc=uc.uc)
                continue
            if not rows:
                log.info("iforest_no_bucket", uc=uc.uc)
                continue
            inserted, published = _score_group(settings, conn, uc, rows)
            total_inserted += inserted
            total_published += published
    log.info(
        "score_iforest_done",
        scanned_ucs=len(IFOREST_USECASES),
        inserted=total_inserted,
        published=total_published,
    )
    return 0
