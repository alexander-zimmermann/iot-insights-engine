"""Single source of truth for anomaly use-cases.

Each UC owns a stable slug — the slug becomes part of the NATS subject
(`anomaly.<slug>.<severity>`) which the knx-nats-bridge writer-rules
resolves to a KNX-GA. Once published, do NOT rename a slug; the mapping
ConfigMap and Basalte notification rules pin to it.

Identifier-typed fields (CAGG/column names) are interpolated into SQL
as-is — they MUST stay plain `[a-z0-9_]` identifiers defined in this
file, never user/DB input. `tests/test_registry.py` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnivariateMetric:
    """A single (CAGG, column) pair scored by z-score against the matching
    `<source>_baseline_30d` hour×weekday profile.

    `group_cols` carries the extra columns the source-CAGG groups by beyond
    `bucket` (e.g. `inverter_id` for solaredge, `ga` for knx). The detector
    joins per group on both sides and emits one anomaly row per group.

    No `warmup_days` here — there is no trained model to mature; the
    30d baseline-CAGG is populated from existing raw data the moment a
    metric is registered.
    """

    uc: str
    source_cagg: str
    baseline_cagg: str
    metric: str
    stats_field: str
    group_cols: tuple[str, ...] = ()
    severity_floor: str = "info"
    silenced: bool = False
    # Robustness knobs — all default to 0.0, reducing the detector to the
    # textbook z-score for existing metrics. The effective stddev is
    # `max(stddev, min_stddev_rel*|mean|, min_stddev_abs)` (floors a
    # near-constant baseline so it can't explode); a hit is dropped unless
    # `|actual-mean| >= max(deadband_abs, deadband_rel*|mean|)`.
    min_stddev_rel: float = 0.0
    min_stddev_abs: float = 0.0
    deadband_abs: float = 0.0
    deadband_rel: float = 0.0
    # Optional SQL predicate on the source CAGG scoping which rows are scored
    # (trusted registry input). Used to keep bursty channels out of a
    # stationary z-score that a dedicated detector handles instead.
    source_filter: str | None = None
    # When False, the published anomaly carries `entity=None` (subject
    # `anomaly.<uc>`) even though `group_cols` is set. Use for house-level
    # metrics that the source CAGG happens to store per-group (e.g.
    # consumer_total/grid_power duplicated under both inverter_ids): the
    # grouping/baseline join stays correct while a `source_filter` dedups to
    # one row, but the routing entity must be None so it maps to one KNX-GA.
    emit_entity: bool = True


@dataclass(frozen=True)
class IsolationForestUseCase:
    """Multivariate features fit by IsolationForest, scored hourly.

    One model per group-value combination (e.g. per inverter_id, per
    meter_id) — a 2nd inverter doesn't poison the existing fit.

    `include_hour_of_day=True` appends EXTRACT(hour FROM bucket) — used
    for diurnal metrics (PV) where the same value can be normal at noon
    and anomalous at midnight.
    """

    uc: str
    source_cagg: str
    feature_cols: tuple[str, ...]
    group_cols: tuple[str, ...] = ()
    include_hour_of_day: bool = False
    lookback_days: int = 60
    contamination: float = 0.005
    severity_floor: str = "info"
    warmup_days: int = 7
    silenced: bool = False


@dataclass(frozen=True)
class KnxJoinUseCase:
    """Rule-based detector against per-room KNX-Joins.

    Each UC's SQL + threshold logic lives in `detect_knx_join`; this
    registry only enumerates slugs so the dispatcher can iterate them
    and ops can silence individual UCs without code changes.
    """

    uc: str
    silenced: bool = False


@dataclass(frozen=True)
class SeasonalModel:
    """statsforecast MSTL+AutoARIMA target metric.

    Univariate for now — score_seasonal fits inline each hour over the
    last `lookback_days` of the named CAGG column (no persisted model).
    Exogenous variables (outdoor temp for heating, etc.) are a
    follow-up once the framework proves itself.

    `forecast_horizon_hours` is how far each run projects ahead.
    `sigma_threshold` only sets the width of the published forecast
    bounds (± n·residual_stddev); anomaly severity uses the fixed
    1.0/1.5/2.5σ tiers in `score_seasonal._classify`.
    """

    uc: str
    source_cagg: str
    metric: str
    season_length: tuple[int, ...] = (24, 168)
    lookback_days: int = 365
    forecast_horizon_hours: int = 24
    sigma_threshold: float = 3.0
    severity_floor: str = "info"
    warmup_days: int = 14
    silenced: bool = False


UNIVARIATE_METRICS: tuple[UnivariateMetric, ...] = (
    # Heating boiler — `ems_esp_boiler_1h` × `ems_esp_boiler_baseline_30d`.
    UnivariateMetric(
        uc="boiler_curburnpow",
        source_cagg="ems_esp_boiler_1h",
        baseline_cagg="ems_esp_boiler_baseline_30d",
        metric="curburnpow_avg",
        stats_field="curburnpow_stats",
    ),
    UnivariateMetric(
        uc="boiler_curflowtemp",
        source_cagg="ems_esp_boiler_1h",
        baseline_cagg="ems_esp_boiler_baseline_30d",
        metric="curflowtemp_avg",
        stats_field="curflowtemp_stats",
    ),
    UnivariateMetric(
        uc="boiler_rettemp",
        source_cagg="ems_esp_boiler_1h",
        baseline_cagg="ems_esp_boiler_baseline_30d",
        metric="rettemp_avg",
        stats_field="rettemp_stats",
    ),
    UnivariateMetric(
        uc="boiler_heatingactive",
        source_cagg="ems_esp_boiler_1h",
        baseline_cagg="ems_esp_boiler_baseline_30d",
        metric="heatingactive_samples",
        stats_field="heatingactive_stats",
    ),
    # DHW — `ems_esp_dhw_1h` × `ems_esp_dhw_baseline_30d`.
    UnivariateMetric(
        uc="dhw_curtemp",
        source_cagg="ems_esp_dhw_1h",
        baseline_cagg="ems_esp_dhw_baseline_30d",
        metric="curtemp_avg",
        stats_field="curtemp_stats",
    ),
    UnivariateMetric(
        uc="dhw_curflow",
        source_cagg="ems_esp_dhw_1h",
        baseline_cagg="ems_esp_dhw_baseline_30d",
        metric="curflow_avg",
        stats_field="curflow_stats",
    ),
    # SolarEdge inverter — group on `inverter_id`.
    UnivariateMetric(
        uc="solar_ac_power",
        source_cagg="solaredge_inverter_1h",
        baseline_cagg="solaredge_inverter_baseline_30d",
        metric="ac_power_avg",
        stats_field="ac_power_stats",
        group_cols=("inverter_id",),
    ),
    UnivariateMetric(
        uc="solar_inverter_temperature",
        source_cagg="solaredge_inverter_1h",
        baseline_cagg="solaredge_inverter_baseline_30d",
        metric="temperature_avg",
        stats_field="temperature_stats",
        group_cols=("inverter_id",),
    ),
    # SolarEdge powerflow — group on `inverter_id`.
    UnivariateMetric(
        uc="pv_production",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="pv_production_avg",
        stats_field="pv_production_stats",
        group_cols=("inverter_id",),
    ),
    # consumer_total/grid_power are house-level but the powerflow CAGG stores
    # them under BOTH inverter_ids (duplicate values). Scope to inverter 1 so
    # the anomaly fires once; `emit_entity=False` keeps the routing entity None
    # (subject `anomaly.<uc>`) so it maps to one house-level KNX-GA.
    # SILENCED: like grid_consumption/-delivery/pv_self_consumption these read
    # SolarEdge grid/consumer fields that are invalid until the SolarEdge meter
    # is live (battery + 400 V) — real grid data is on the KNX Energiezähler
    # 15/1. Flip `silenced=False` when the SolarEdge meter feeds the powerflow.
    UnivariateMetric(
        uc="consumer_total",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="consumer_total_avg",
        stats_field="consumer_total_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    UnivariateMetric(
        uc="grid_power",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="grid_power_avg",
        stats_field="grid_power_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    # Grid import/export and PV self-consumption — same house-level pattern as
    # grid_power/consumer_total (powerflow stores them under both inverter_ids).
    # SILENCED: these read SolarEdge grid/consumer fields, which stay invalid
    # until the SolarEdge meter goes live (battery + 400 V). The real grid data
    # lives on the KNX Energiezähler 15/1 meanwhile. Flip `silenced=False` once
    # the SolarEdge meter feeds the powerflow.
    UnivariateMetric(
        uc="grid_consumption",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="grid_consumption_avg",
        stats_field="grid_consumption_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    UnivariateMetric(
        uc="grid_delivery",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="grid_delivery_avg",
        stats_field="grid_delivery_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    UnivariateMetric(
        uc="pv_self_consumption",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="consumer_used_pv_production_avg",
        stats_field="consumer_used_pv_production_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    # Battery — silenced until the storage unit is connected and the
    # solaredge_battery / powerflow battery columns accumulate ~30 d of
    # history. Flip `silenced=False` once the baseline is mature. Charge +
    # discharge come from the powerflow CAGG (house-level, inverter 1), SOC
    # from the dedicated battery baseline.
    UnivariateMetric(
        uc="battery_charge",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="battery_charge_avg",
        stats_field="battery_charge_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    UnivariateMetric(
        uc="battery_discharge",
        source_cagg="solaredge_powerflow_1h",
        baseline_cagg="solaredge_powerflow_baseline_30d",
        metric="consumer_used_battery_production_avg",
        stats_field="consumer_used_battery_production_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        silenced=True,
    ),
    UnivariateMetric(
        uc="battery_soc",
        source_cagg="solaredge_battery_1h",
        baseline_cagg="solaredge_battery_baseline_30d",
        metric="soc_avg",
        stats_field="soc_stats",
        group_cols=("inverter_id",),
        source_filter="inverter_id = 1",
        emit_entity=False,
        deadband_abs=5.0,  # %, ignore sub-5% SOC noise
        silenced=True,
    ),
    # Wallbox meter — group on `meter_id`.
    UnivariateMetric(
        uc="wallbox_power_total",
        source_cagg="warp_meter_1h",
        baseline_cagg="warp_meter_baseline_30d",
        metric="power_total_avg",
        stats_field="power_total_stats",
        group_cols=("meter_id",),
    ),
    # KNX — group on `(ga, knx_name)`. One UC for all values; the anomaly
    # row carries the GA so insights tools can join against ga_catalog
    # for the human-readable name and the source room/function.
    UnivariateMetric(
        uc="knx_value",
        source_cagg="knx_1h",
        baseline_cagg="knx_baseline_30d",
        metric="avg_value",
        stats_field="value_stats",
        group_cols=("ga", "knx_name"),
        # Heterogeneous units (lux, °C, azimuth, setpoints) → only *relative*
        # knobs are safe here. The relative stddev floor caps the variance
        # explosion on near-constant channels (lux ~0 overnight scoring 1e15);
        # the relative deadband drops deviations below 25% of the baseline mean.
        min_stddev_rel=0.15,
        deadband_rel=0.25,
        # Bursty appliance power is handled by the dedicated knx_appliance_*
        # path (standby-drift + left-on); a stationary z-score here would just
        # flag every normal use of the oven/coffee machine/etc.
        source_filter="knx_name NOT LIKE '%Stromwert'",
    ),
    # KNX appliances — standby-drift on the hourly idle floor (`min(value)`).
    # The source CAGG is pre-filtered to `%Stromwert` channels and carries a
    # per-(ga, knx_name) min, so the baseline models each appliance's idle
    # draw; a creep from e.g. 43 → 300 (phantom load / stuck relay) fires while
    # normal on/off usage does not. Single native unit, so absolute floor +
    # deadband are safe (standby valley sits well under 100; floors ≤ ~55).
    UnivariateMetric(
        uc="appliance_standby",
        source_cagg="knx_appliance_1h",
        baseline_cagg="knx_appliance_baseline_30d",
        metric="idle_floor",
        stats_field="idle_floor_stats",
        group_cols=("ga", "knx_name"),
        min_stddev_rel=0.10,
        min_stddev_abs=5.0,
        deadband_abs=20.0,
    ),
)
IFOREST_USECASES: tuple[IsolationForestUseCase, ...] = (
    # Heating boiler — single boiler, no group_cols. heatingactive_samples
    # gives the IF a coarse "is the burner firing this hour" signal that
    # disambiguates legitimate low-power idle from a sensor stall at zero.
    IsolationForestUseCase(
        uc="heating_iforest",
        source_cagg="ems_esp_boiler_1h",
        feature_cols=(
            "curburnpow_avg",
            "curflowtemp_avg",
            "rettemp_avg",
            "outdoortemp_avg",
            "heatingactive_samples",
        ),
    ),
    # PV powerflow — group on inverter_id; hour-of-day matters because PV
    # production is zero by 22:00 and very different at 06:00 vs 12:00.
    IsolationForestUseCase(
        uc="pv_iforest",
        source_cagg="solaredge_powerflow_1h",
        feature_cols=(
            "pv_production_avg",
            "consumer_total_avg",
            "grid_power_avg",
            "grid_consumption_avg",
            "grid_delivery_avg",
        ),
        group_cols=("inverter_id",),
        include_hour_of_day=True,
    ),
    # Battery — single storage unit (no group_cols → house-level subject
    # `anomaly.battery_iforest`). SOC + power are diurnal (charge midday,
    # discharge evening), so include hour-of-day. Silenced until the battery
    # is connected and has cleared warmup; flip `silenced=False` then.
    IsolationForestUseCase(
        uc="battery_iforest",
        source_cagg="solaredge_battery_1h",
        feature_cols=(
            "soc_avg",
            "power_avg",
        ),
        include_hour_of_day=True,
        silenced=True,
    ),
    # Wallbox meter — group on meter_id. Voltage + current per phase
    # catches phase imbalance and brownouts that power_total alone hides.
    IsolationForestUseCase(
        uc="wallbox_iforest",
        source_cagg="warp_meter_1h",
        feature_cols=(
            "power_total_avg",
            "voltage_l1_avg",
            "voltage_l2_avg",
            "voltage_l3_avg",
            "current_l1_avg",
            "current_l2_avg",
            "current_l3_avg",
        ),
        group_cols=("meter_id",),
    ),
)
KNX_JOIN_USECASES: tuple[KnxJoinUseCase, ...] = (
    # Per-room rule: FBH-Stellwert open >50% AND room stays >=1°C below
    # setpoint for >=2h. Detects stuck valves, bled circuits, or
    # sensor mismatches the IF would not isolate.
    KnxJoinUseCase(uc="fbh_cold"),
    # Per-room rule: window open while FBH is heating with outdoor
    # temp <12°C. Catches "heater pumping into open window" — a
    # comfort + energy waste pattern that the Basalte UI shouldn't
    # need to remember to surface.
    KnxJoinUseCase(uc="window_while_heating"),
    # Appliance left on: current draw above the standby valley for several
    # consecutive hours. Restricted to normally-idle appliances — always-on
    # loads (fridge, network rack, circulation pump) drop out by their high
    # 30d active-rate, so no per-GA exclusion list is needed.
    KnxJoinUseCase(uc="appliance_runtime"),
    # Freezer evaporator icing: the median compressor run time at warm kitchen
    # ambient creeps up over weeks as frost insulates the coil. Compared against
    # a fixed healthy baseline (a rolling one would absorb the drift). Goes
    # quiet in winter — the signal only exists under thermal load.
    KnxJoinUseCase(uc="freezer_icing"),
)
SEASONAL_MODELS: tuple[SeasonalModel, ...] = (
    # Heating burner activity per hour — the most directly weather-
    # sensitive signal in the house. With <2 years of data the model
    # only learns daily + weekly seasonality (24 + 168); annual
    # seasonality (8766) gets added once we have the data.
    SeasonalModel(
        uc="heating_activity_seasonal",
        source_cagg="ems_esp_boiler_1h",
        metric="heatingactive_samples",
    ),
)


def all_slugs() -> set[str]:
    return (
        {m.uc for m in UNIVARIATE_METRICS}
        | {u.uc for u in IFOREST_USECASES}
        | {k.uc for k in KNX_JOIN_USECASES}
        | {s.uc for s in SEASONAL_MODELS}
    )
