"""Daily fit of statsforecast MSTL+AutoARIMA per registered seasonal UC.

Loads the last `lookback_days` of the source-CAGG metric, fits one
univariate model per UC, computes residual stddev from the in-sample
fit, and persists the bundle to rustfs. Score-jobs pick it up on the
next tick.

Skips a UC whose lookback returned fewer than `MIN_TRAIN_SAMPLES`
hours of usable data — fitting MSTL with [24, 168] season-length on a
short series produces meaningless forecasts.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA

from . import artifacts
from .config import Settings
from .db_write import write_connection
from .logging_setup import get_logger
from .registry import SEASONAL_MODELS, SeasonalModel
from .seasonal_common import DETECTOR_NAME, ModelEnvelope

log = get_logger(__name__)

# MSTL on (24, 168) needs at least two full weekly cycles; require a
# margin so AutoARIMA has signal beyond seasonality.
MIN_TRAIN_SAMPLES = 24 * 14


def _load_training_frame(
    conn: psycopg.Connection[Any], uc: SeasonalModel
) -> pd.DataFrame:
    sql = f"""
        SELECT bucket AS ds, {uc.metric}::float AS y
        FROM {uc.source_cagg}
        WHERE bucket > now() - interval '{uc.lookback_days} days'
          AND {uc.metric} IS NOT NULL
        ORDER BY bucket
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert(None)
    df["unique_id"] = uc.uc
    return df[["unique_id", "ds", "y"]]


def _fit_and_save(settings: Settings, uc: SeasonalModel, df: pd.DataFrame) -> bool:
    n_samples = int(df.shape[0])
    if n_samples < MIN_TRAIN_SAMPLES:
        log.info(
            "seasonal_insufficient_samples",
            uc=uc.uc,
            n=n_samples,
            min_required=MIN_TRAIN_SAMPLES,
        )
        return False
    sf = StatsForecast(
        models=[MSTL(season_length=list(uc.season_length), trend_forecaster=AutoARIMA())],
        freq="h",
    )
    # fit() persists the model for score_seasonal's predict(); forecast(fitted=True)
    # is the only way to populate forecast_fitted_values() for residuals.
    sf.fit(df=df)
    sf.forecast(df=df, h=uc.forecast_horizon_hours, fitted=True)
    fitted = sf.forecast_fitted_values()
    residuals = fitted["y"].to_numpy() - fitted["MSTL"].to_numpy()
    residual_stddev = float(np.nanstd(residuals, ddof=1))
    envelope = ModelEnvelope(
        sf=sf,
        season_length=uc.season_length,
        forecast_horizon_hours=uc.forecast_horizon_hours,
        sigma_threshold=uc.sigma_threshold,
        residual_stddev=residual_stddev,
        n_train_samples=n_samples,
        trained_at=datetime.now(tz=UTC).isoformat(),
    )
    artifacts.save_model(settings, DETECTOR_NAME, uc.uc, envelope)
    log.info(
        "seasonal_trained",
        uc=uc.uc,
        n_samples=n_samples,
        residual_stddev=residual_stddev,
        season_length=list(uc.season_length),
    )
    return True


def run(settings: Settings, _argv: Sequence[str]) -> int:
    trained = 0
    skipped = 0
    with write_connection(settings) as conn:
        for uc in SEASONAL_MODELS:
            if uc.silenced:
                log.info("uc_silenced", uc=uc.uc)
                continue
            try:
                df = _load_training_frame(conn, uc)
            except psycopg.Error:
                log.exception("seasonal_load_failed", uc=uc.uc)
                continue
            ok = _fit_and_save(settings, uc, df)
            if ok:
                trained += 1
            else:
                skipped += 1
    log.info("train_seasonal_done", trained=trained, skipped=skipped)
    return 0
