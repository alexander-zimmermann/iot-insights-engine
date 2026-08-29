"""The catalog scope on the KNX univariate metrics.

`knx_1h` carries no DPT, so every KNX entry restricts itself to one unit
family through a `ga_catalog` subquery. The predicate is plain SQL, so it is
executed here against an in-memory SQLite fixture holding one address per DPT
class the real catalog uses — this is the only place the filter runs outside
the cluster.

SQLite is close enough for `IN` / `LIKE` / subqueries; it differs from
Postgres in that `LIKE` is case-insensitive for ASCII, so a casing bug in a
carve-out pattern would slip through here and only show up in the cluster.
"""

from __future__ import annotations

import sqlite3

from iot_insights_engine import registry

# (ga, dpt, name) — one address per DPT class in the production catalog, plus
# the two addresses the digest reported and the carve-outs inside DPT 9.001.
_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("0/0/111", "1.001", "Allgemein.Zentral.EG.Alles-Aus"),
    ("10/1/1", "1.019", "Sicherheit.KG.Garage.Fenster.Geöffnet-Status"),
    ("3/1/13", "3.007", "Beleuchtung.KG.Flur.Indirekt.Dimmen-Relativ"),
    ("12/2/105", "5.001", "Entertainment.EG.Wohnzimmer.Sony-TV.Lautstärke"),
    ("12/3/148", "5.010", "Entertainment.OG.Badezimmer-Eltern.Asano.Quelle-Status"),
    ("2/2/227", "7.012", "Schalten.EG.Küche.K15-L1.Gefrierschrank.Stromwert"),
    ("3/2/218", "7.600", "Beleuchtung.EG.Küche.Kücheninsel.Farbtemparatur"),
    ("8/2/121", "9.001", "Sensorik.EG.Esszimmer.Sensor.Temperatur"),
    ("8/2/127", "9.001", "Sensorik.EG.Esszimmer.Sensor.Taupunkt"),
    ("15/3/26", "9.001", "Versorgungstechnik.KWL.Temperatur-Abluft"),
    ("15/2/24", "9.001", "Versorgungstechnik.Gastherme.Heizung.Vorlauf.Soll-Temperatur"),
    ("15/2/22", "9.001", "Versorgungstechnik.Gastherme.Heizung.Vorlauf.Ist-Temperatur"),
    ("15/4/23", "9.001", "Versorgungstechnik.Photovoltaik.Wechselrichter-1.Temperatur"),
    ("8/3/183", "9.001", "Sensorik.OG.Begehbarer-Schrank.Sensor.Temperatur-Alarm-2"),
    ("17/1/72", "9.001", "Haushaltstechnik.Backofen.Ist-Temperatur"),
    ("8/0/140", "9.004", "Sensorik.Zentral.A.Helligkeit-SO"),
    ("8/1/124", "9.007", "Sensorik.KG.Hauswirtschaftsraum.Sensor.Luftfeuchtigkeit"),
    ("8/1/131", "9.008", "Sensorik.KG.Hauswirtschaftsraum.Sensor.CO2"),
    ("0/0/245", "10.001", "Allgemein.Zentral.Anwesen.Uhrzeit"),
    ("0/0/246", "11.001", "Allgemein.Zentral.Anwesen.Datum"),
    ("2/2/225", "13.013", "Schalten.EG.Küche.K15-L1.Gefrierschrank.Kilowattstunde"),
    ("2/2/223", "13.100", "Schalten.EG.Küche.K15-L1.Gefrierschrank.Betriebsstunden"),
    ("15/1/4", "14.056", "Versorgungstechnik.Energiezähler.Strom.Netzbezug.Wirkleistung-L1"),
    ("15/4/21", "14.056", "Versorgungstechnik.Photovoltaik.Wechselrichter-1.Leistung"),
    ("15/6/12", "14.056", "Versorgungstechnik.Wallbox.Ladeleistung"),
    ("15/1/21", "14.027", "Versorgungstechnik.Energiezähler.Strom.Spannung-L1"),
    ("8/1/105", "14.058", "Sensorik.KG.Vorratsraum.Sensor.Luftdruck-Relativ"),
    ("8/5/57", "14.007", "Sensorik.A.Dach.Wetterstation.Azimut"),
    ("15/1/59", "16.000", "Versorgungstechnik.Energiezähler.Wasser.Seriennummer"),
    ("0/0/100", "17.001", "Allgemein.Zentral.KG.Szenen"),
    ("0/0/247", "19.001", "Allgemein.Zentral.Anwesen.Datum-Uhrzeit"),
    ("6/2/105", "20.102", "Raumklima.EG.Küche.FBH.HVAC"),
    ("3/3/226", "232.600", "Beleuchtung.OG.Badezimmer-Eltern.RGB.Farbsteuerung"),
)

_KNX_METRICS = {m.uc: m for m in registry.UNIVARIATE_METRICS if m.source_cagg == "knx_1h"}


def _scoped_gas(source_filter: str) -> set[str]:
    """The GAs a `source_filter` leaves in the source CAGG, as the detector
    interpolates it: `... FROM knx_1h WHERE bucket = ... AND <filter>`."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE ga_catalog (ga TEXT, dpt TEXT, name TEXT)")
    con.executemany("INSERT INTO ga_catalog VALUES (?, ?, ?)", _CATALOG)
    # knx_1h holds one row per scored channel; the catalog is its GA universe.
    con.execute("CREATE TABLE knx_1h (ga TEXT, knx_name TEXT)")
    con.execute("INSERT INTO knx_1h SELECT ga, name FROM ga_catalog")
    rows = con.execute(f"SELECT ga FROM knx_1h WHERE TRUE AND {source_filter}").fetchall()
    con.close()
    return {ga for (ga,) in rows}


def test_temperature_scope_keeps_measurements_and_drops_setpoints() -> None:
    """Setpoints, alarm thresholds and appliance temperatures are not room
    climate; the boiler and PV channels are already scored from their own
    sources and would otherwise fire twice for one event."""
    assert _scoped_gas(_KNX_METRICS["knx_temperature"].source_filter or "") == {
        "8/2/121",  # room sensor
        "8/2/127",  # dew point
        "15/3/26",  # KWL, covered by no other detector
    }


def test_unit_family_scopes() -> None:
    assert _scoped_gas(_KNX_METRICS["knx_humidity"].source_filter or "") == {"8/1/124"}
    assert _scoped_gas(_KNX_METRICS["knx_air_quality"].source_filter or "") == {"8/1/131"}
    assert _scoped_gas(_KNX_METRICS["knx_voltage"].source_filter or "") == {"15/1/21"}
    # PV and wallbox power are scored from their own sources, not the KNX mirror.
    assert _scoped_gas(_KNX_METRICS["knx_power"].source_filter or "") == {"15/1/4"}


def test_non_continuous_classes_are_scored_by_nobody() -> None:
    """Booleans, enums, scenes, strings, dates and cumulative counters must not
    reach a z-score — `avg_value` over them is not a physical quantity."""
    scored: set[str] = set()
    for metric in _KNX_METRICS.values():
        scored |= _scoped_gas(metric.source_filter or "")
    non_continuous = {"1", "3", "5", "7", "10", "11", "13", "16", "17", "19", "20", "232"}
    out_of_scope = {ga for ga, dpt, _ in _CATALOG if dpt.split(".")[0] in non_continuous}
    assert not scored & out_of_scope


def test_reported_offender_is_out_of_scope() -> None:
    """`12/3/148` is DPT 5.010 — a source selector. Its mean is a source that
    does not exist, and it produced nine of the digest's top ten."""
    for metric in _KNX_METRICS.values():
        assert "12/3/148" not in _scoped_gas(metric.source_filter or "")


def test_families_do_not_overlap() -> None:
    """A GA in two entries would be scored — and alerted — twice."""
    seen: set[str] = set()
    for metric in _KNX_METRICS.values():
        gas = _scoped_gas(metric.source_filter or "")
        assert not seen & gas, f"{metric.uc} overlaps an earlier family"
        seen |= gas


def test_every_knx_metric_scopes_through_the_catalog() -> None:
    """A future KNX entry without a catalog scope silently scores all 2500+
    addresses again, enums included."""
    for metric in _KNX_METRICS.values():
        assert metric.source_filter, f"{metric.uc} has no source_filter"
        assert "ga_catalog" in metric.source_filter, metric.uc


def test_knx_metrics_carry_absolute_floors() -> None:
    """Relative knobs cannot protect a near-zero baseline — 15 % of ~0 is ~0.
    Splitting by unit family is what makes the absolute knobs usable."""
    for metric in _KNX_METRICS.values():
        assert metric.min_stddev_abs > 0.0, metric.uc
        assert metric.deadband_abs > 0.0, metric.uc
