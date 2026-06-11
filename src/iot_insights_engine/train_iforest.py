"""Train one IsolationForest per (uc, group) over the last `lookback_days`
of 1h-CAGG rows. Persists pipeline + severity thresholds to S3.

Skips a UC when fewer than `MIN_TRAIN_SAMPLES` clean rows exist — IF on
e.g. 40 samples produces meaningless decision boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import psycopg
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import artifacts, iforest_common
from .config import Settings
from .db_write import write_connection
from .iforest_common import DETECTOR_NAME, ModelEnvelope
from .logging_setup import get_logger
from .registry import IFOREST_USECASES, IsolationForestUseCase

log = get_logger(__name__)

MIN_TRAIN_SAMPLES = 200
WARNING_QUANTILE = 0.005
CRITICAL_QUANTILE = 0.001


def _load_training_rows(
    conn: psycopg.Connection[Any], uc: IsolationForestUseCase
) -> dict[tuple[Any, ...], np.ndarray]:
    """Returns {group_value_tuple: feature_matrix}. Empty group_cols →
    one entry keyed by `()`."""
    sql = iforest_common.select_features_sql(
        uc, where_extra=f"bucket > now() - interval '{uc.lookback_days} days'"
    )
    fnames = iforest_common.feature_names(uc)
    buckets: dict[tuple[Any, ...], list[list[float]]] = {}
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            gvals = tuple(row[c] for c in uc.group_cols)
            feats = [float(row[c]) for c in fnames]
            buckets.setdefault(gvals, []).append(feats)
    return {g: np.asarray(rows, dtype=np.float64) for g, rows in buckets.items()}


def _fit_and_save(
    settings: Settings,
    uc: IsolationForestUseCase,
    group_values: tuple[Any, ...],
    features: np.ndarray,
) -> bool:
    n_samples = int(features.shape[0])
    if n_samples < MIN_TRAIN_SAMPLES:
        log.info(
            "iforest_insufficient_samples",
            uc=uc.uc,
            group=group_values,
            n=n_samples,
            min_required=MIN_TRAIN_SAMPLES,
        )
        return False
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    contamination=uc.contamination,
                    random_state=42,
                    n_estimators=100,
                ),
            ),
        ]
    )
    pipeline.fit(features)
    scores = pipeline.score_samples(features)
    threshold_warning = float(np.quantile(scores, WARNING_QUANTILE))
    threshold_critical = float(np.quantile(scores, CRITICAL_QUANTILE))
    envelope = ModelEnvelope(
        pipeline=pipeline,
        feature_names=iforest_common.feature_names(uc),
        threshold_warning=threshold_warning,
        threshold_critical=threshold_critical,
        n_train_samples=n_samples,
        trained_at=datetime.now(tz=UTC).isoformat(),
    )
    key = iforest_common.group_key(uc, group_values)
    artifacts.save_model(settings, DETECTOR_NAME, key, envelope)
    log.info(
        "iforest_trained",
        uc=uc.uc,
        group=group_values,
        n_samples=n_samples,
        threshold_warning=threshold_warning,
        threshold_critical=threshold_critical,
    )
    return True


def run(settings: Settings, _argv: Sequence[str]) -> int:
    trained = 0
    skipped = 0
    with write_connection(settings) as conn:
        for uc in IFOREST_USECASES:
            if uc.silenced:
                log.info("uc_silenced", uc=uc.uc)
                continue
            try:
                groups = _load_training_rows(conn, uc)
            except psycopg.Error:
                log.exception("iforest_load_failed", uc=uc.uc)
                continue
            if not groups:
                log.info("iforest_no_training_data", uc=uc.uc)
                skipped += 1
                continue
            for gvals, group_features in groups.items():
                ok = _fit_and_save(settings, uc, gvals, group_features)
                if ok:
                    trained += 1
                else:
                    skipped += 1
    log.info("train_iforest_done", trained=trained, skipped=skipped)
    return 0
