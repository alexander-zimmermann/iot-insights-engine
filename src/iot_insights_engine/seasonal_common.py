"""Shared envelope for train/score statsforecast jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DETECTOR_NAME = "seasonal"


@dataclass(frozen=True)
class ModelEnvelope:
    """Persisted alongside the fitted statsforecast model so the
    score-job can classify without re-reading training data."""

    sf: Any  # statsforecast.StatsForecast (joblib-pickleable)
    season_length: tuple[int, ...]
    forecast_horizon_hours: int
    sigma_threshold: float
    residual_stddev: float
    n_train_samples: int
    trained_at: str  # ISO-8601
