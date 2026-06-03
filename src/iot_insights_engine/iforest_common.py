"""Shared helpers for train/score IsolationForest jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import IsolationForestUseCase

# Pickle envelope written by train_iforest, consumed by score_iforest.
# `thresholds` are score_samples quantiles computed on the training set:
# score_samples(new) < threshold → that severity (sklearn returns
# higher = less anomalous).
DETECTOR_NAME = "iforest"


@dataclass(frozen=True)
class ModelEnvelope:
    """Persisted alongside the sklearn pipeline so the score-job knows the
    cutoff per severity without re-reading training data."""

    pipeline: Any  # sklearn Pipeline[StandardScaler, IsolationForest]
    feature_names: tuple[str, ...]
    threshold_warning: float
    threshold_critical: float
    n_train_samples: int
    trained_at: str  # ISO-8601


def feature_names(uc: IsolationForestUseCase) -> tuple[str, ...]:
    """Order MUST match the SELECT in `select_features_sql` so the model
    receives columns in the same position at train + score time."""
    cols = list(uc.feature_cols)
    if uc.include_hour_of_day:
        cols.append("hour_of_day")
    return tuple(cols)


def select_features_sql(
    uc: IsolationForestUseCase,
    *,
    where_extra: str,
) -> str:
    """Returns a SELECT that yields `bucket, *group_cols, *feature_names`
    in stable column order.

    `where_extra` is appended to the WHERE clause (no leading AND) — caller
    is responsible for parameterizing the lookback window vs the "last
    completed bucket" filter.
    """
    group_select = "".join(f"{c}, " for c in uc.group_cols)
    feature_select = ", ".join(uc.feature_cols)
    extras = ", EXTRACT(hour FROM bucket)::int AS hour_of_day" if uc.include_hour_of_day else ""
    not_null = " AND ".join(f"{c} IS NOT NULL" for c in uc.feature_cols)
    return f"""
        SELECT bucket, {group_select}{feature_select}{extras}
        FROM {uc.source_cagg}
        WHERE {not_null} AND {where_extra}
    """


def group_key(uc: IsolationForestUseCase, group_values: tuple[Any, ...]) -> str:
    """S3 artifact key suffix: `iforest/<uc>/<group_value_concat>.joblib`.

    No group_cols → key is just `<uc>.joblib`. Otherwise the group values
    are joined with `__` after URL-safe stringification.
    """
    if not group_values:
        return uc.uc
    safe = [str(v).replace("/", "_").replace("__", "_") for v in group_values]
    return f"{uc.uc}/{'__'.join(safe)}"
