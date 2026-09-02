"""Fault-file loader tests — seam 1 of the detection rebuild.

Prior art: the writer-rules loader tests in knx-nats-bridge. Every test
feeds an invented YAML file and asserts only what comes out: typed fault
objects, or a load error naming the fault and the field.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from iot_insights_engine.faults import (
    _SCHEMA_PATH,
    Dormant,
    FaultList,
    MeasurementKind,
    Target,
)

# A minimal valid file: the tracer fault plus one drift fault, matching the
# shape the lares fault list uses.
_VALID = """
faults:
  - name: channel_silence
    sentence: "Ein Kanal, der sonst regelmäßig sendet, schweigt länger als das
      Fünffache seiner üblichen Sendepause."
    unit: "× der üblichen Sendepause"
    kind: silence
    parameters:
      gap_factor: 5
    scope:
      name_like: "%"
    target:
      per_main_group: true
  - name: appliance_standby
    sentence: "Der Ruhestrom eines Geräts steigt dauerhaft um mehr als 40 mA."
    unit: "mA"
    kind: drift
    parameters:
      healthy: 43
      slack: 5
      budget: 840
    scope:
      dpt: "7.012"
      name_like: "%.Stromwert"
    target:
      ga: "2/2/229"
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "faults.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_file(tmp_path: Path) -> None:
    faults = FaultList.load(_write(tmp_path, _VALID))
    assert len(faults) == 2
    assert [f.name for f in faults] == ["channel_silence", "appliance_standby"]

    silence = faults.get("channel_silence")
    assert silence.kind is MeasurementKind.SILENCE
    assert silence.parameters == {"gap_factor": 5}
    assert silence.unit == "× der üblichen Sendepause"
    assert silence.target is not None
    assert silence.target.per_main_group is True
    assert silence.target.ga is None
    assert silence.dormant is None

    standby = faults.get("appliance_standby")
    assert standby.kind is MeasurementKind.DRIFT
    assert standby.parameters == {"healthy": 43, "slack": 5, "budget": 840}
    assert standby.target is not None
    assert standby.target.ga == "2/2/229"
    assert standby.target.per_main_group is False


def test_scope_is_carried_not_resolved(tmp_path: Path) -> None:
    # The loader carries the catalog query verbatim; resolution happens where
    # the catalog lives, so the query text must survive the round trip.
    faults = FaultList.load(_write(tmp_path, _VALID))
    standby = faults.get("appliance_standby")
    assert standby.scope.dpt == ("7.012",)
    assert standby.scope.name_like == ("%.Stromwert",)
    assert standby.scope.exclude_name_like == ()


def test_scope_accepts_lists(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in K."
            unit: "K"
            kind: constancy
            parameters:
              window_hours: 72
            scope:
              dpt: ["9.001", "9.007"]
              exclude_name_like: "%.Alarm%"
            target:
              ga: "8/0/1"
        """,
    )
    [f] = FaultList.load(path)
    assert f.scope.dpt == ("9.001", "9.007")
    assert f.scope.exclude_name_like == ("%.Alarm%",)


def test_missing_sentence_names_fault_and_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            unit: "h"
            kind: silence
            parameters:
              gap_factor: 5
            scope:
              name_like: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*sentence"):
        FaultList.load(path)


def test_missing_unit_names_fault_and_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            sentence: "Ein Kanal schweigt."
            kind: silence
            parameters:
              gap_factor: 5
            scope:
              name_like: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*unit"):
        FaultList.load(path)


def test_missing_parameter_names_fault_and_field(tmp_path: Path) -> None:
    # silence requires gap_factor specifically, not just any parameter
    path = _write(
        tmp_path,
        """
        faults:
          - name: channel_silence
            sentence: "Ein Kanal schweigt."
            unit: "h"
            kind: silence
            parameters:
              other: 1
            scope:
              name_like: "%"
            target:
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"channel_silence.*gap_factor"):
        FaultList.load(path)


def test_empty_parameters_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters: {}
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*parameters"):
        FaultList.load(path)


def test_non_numeric_parameter_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: "high"
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*healthy"):
        FaultList.load(path)


def test_unknown_kind_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in ppm."
            unit: "ppm"
            kind: threshold
            parameters:
              limit: 1400
            scope:
              name_like: "%.CO2"
            target:
              ga: "8/0/1"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*kind"):
        FaultList.load(path)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
            severity: critical
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*severity"):
        FaultList.load(path)


def test_target_requires_exactly_one_form(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
              per_main_group: true
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


def test_invalid_ga_format_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2.2.229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


def test_empty_scope_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope: {}
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*scope"):
        FaultList.load(path)


def test_duplicate_name_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
          - name: x
            sentence: "Ein anderer Satz mit Einheit in mA."
            unit: "mA"
            kind: duration
            parameters:
              limit: 240
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/230"
        """,
    )
    with pytest.raises(ValueError, match=r"duplicate.*'x'"):
        FaultList.load(path)


def test_dormant_loads_but_is_not_schedulable(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _VALID
        + """
  - name: battery_soc_stuck
    sentence: "Der Batterie-Ladestand ändert sich über 24 h nicht (%)."
    unit: "%"
    kind: constancy
    parameters:
      window_hours: 24
    scope:
      name_like: "%.Batterie.%"
    target:
      ga: "15/4/40"
    dormant:
      reason: "SolarEdge meter not connected until the 400 V supply is in"
      active_when: "the meter reports non-zero values"
""",
    )
    faults = FaultList.load(path)
    assert len(faults) == 3
    dormant = faults.get("battery_soc_stuck")
    assert dormant.dormant == Dormant(
        reason="SolarEdge meter not connected until the 400 V supply is in",
        active_when="the meter reports non-zero values",
    )
    assert [f.name for f in faults.schedulable()] == ["channel_silence", "appliance_standby"]


def test_dormant_requires_reason_and_condition(tmp_path: Path) -> None:
    # `dormant: true` is exactly the boolean-with-a-comment this field replaces
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
            dormant: true
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*dormant"):
        FaultList.load(path)


def test_dormant_missing_active_when_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
            dormant:
              reason: "meter not connected"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*active_when"):
        FaultList.load(path)


def test_missing_name_reports_position(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
            target:
              ga: "2/2/229"
        """,
    )
    with pytest.raises(ValueError, match=r"#1.*name"):
        FaultList.load(path)


def test_empty_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="faults"):
        FaultList.load(_write(tmp_path, ""))


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        FaultList.load(_write(tmp_path, "- just\n- a\n- list\n"))


def test_fault_objects_are_frozen(tmp_path: Path) -> None:
    faults = FaultList.load(_write(tmp_path, _VALID))
    silence = faults.get("channel_silence")
    with pytest.raises(AttributeError):
        silence.unit = "h"  # type: ignore[misc]


def test_target_union_holds_for_python_construction() -> None:
    # The schema's oneOf, mirrored for Python-side construction
    with pytest.raises(ValueError, match="never both or neither"):
        Target()
    with pytest.raises(ValueError, match="never both or neither"):
        Target(ga="1/2/3", per_main_group=True)


_EXTERNAL = """
faults:
  - name: system_pressure_low
    sentence: "Der Systemdruck der Gastherme liegt unter 1,0 bar."
    unit: "bar"
    kind: external
    scope:
      name_like: "%.Gastherme.System-Druck-Anomalie"
"""


def test_external_fault_loads_without_target_and_parameters(tmp_path: Path) -> None:
    # Basalte owns threshold and delivery; the entry only declares the
    # sentence and where the severity writes appear.
    [fault] = FaultList.load(_write(tmp_path, _EXTERNAL))
    assert fault.kind is MeasurementKind.EXTERNAL
    assert fault.sentence == "Der Systemdruck der Gastherme liegt unter 1,0 bar."
    assert fault.parameters == {}
    assert fault.target is None
    assert fault.scope.name_like == ("%.Gastherme.System-Druck-Anomalie",)


def test_external_fault_rejects_a_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _EXTERNAL
        + """    target:
      ga: "15/2/2"
""",
    )
    with pytest.raises(ValueError, match=r"'system_pressure_low'.*target"):
        FaultList.load(path)


def test_external_fault_rejects_parameters(tmp_path: Path) -> None:
    # A threshold on an external fault would be a second source of truth
    # beside the Basalte logic.
    path = _write(
        tmp_path,
        _EXTERNAL
        + """    parameters:
      threshold: 1
""",
    )
    with pytest.raises(ValueError, match=r"'system_pressure_low'.*parameters"):
        FaultList.load(path)


def test_engine_kind_still_requires_target(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        faults:
          - name: x
            sentence: "Ein Satz mit Einheit in mA."
            unit: "mA"
            kind: drift
            parameters:
              healthy: 43
            scope:
              name_like: "%.Stromwert"
        """,
    )
    with pytest.raises(ValueError, match=r"'x'.*target"):
        FaultList.load(path)


def test_bundled_schema_is_valid() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
