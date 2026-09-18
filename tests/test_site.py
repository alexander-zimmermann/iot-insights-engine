"""Site-file loader tests: what the house is, read from an invented
`site.yaml`. Every test feeds a file and asserts only what comes out — the
typed site, or a load error naming the field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iot_insights_engine.site import Location, Plane, Site

# The shape the lares site file uses: the two roof planes with the inverter
# each of them feeds, in the Open-Meteo azimuth convention.
_VALID = """
location:
  latitude: 50.62598
  longitude: 6.02435
timezone: Europe/Berlin
pv:
  planes:
    West:
      inverter_id: 1
      tilt: 17
      azimuth: 129
      kwp: 6.435
    Ost:
      inverter_id: 2
      tilt: 17
      azimuth: -51
      kwp: 6.175
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "site.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_file(tmp_path: Path) -> None:
    site = Site.load(_write(tmp_path, _VALID))
    assert site.location == Location(latitude=50.62598, longitude=6.02435)
    assert site.timezone == "Europe/Berlin"
    assert site.planes == (
        Plane(key="West", inverter_id=1, tilt=17.0, azimuth=129.0, kwp=6.435),
        Plane(key="Ost", inverter_id=2, tilt=17.0, azimuth=-51.0, kwp=6.175),
    )


def test_two_planes_on_one_inverter_rejected(tmp_path: Path) -> None:
    # A counter belongs to one inverter; two planes claiming it would count
    # the same rises twice.
    path = _write(tmp_path, _VALID.replace("inverter_id: 2", "inverter_id: 1"))
    with pytest.raises(ValueError, match=r"inverter 1.*West.*Ost"):
        Site.load(path)


def test_unknown_timezone_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID.replace("Europe/Berlin", "Europe/Nowhere"))
    with pytest.raises(ValueError, match=r"timezone.*Europe/Nowhere"):
        Site.load(path)


def test_missing_kwp_names_the_plane_and_the_field(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID.replace("      kwp: 6.175\n", ""))
    with pytest.raises(ValueError, match=r"pv\.planes\.Ost.*kwp"):
        Site.load(path)


def test_non_positive_kwp_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID.replace("kwp: 6.175", "kwp: 0"))
    with pytest.raises(ValueError, match=r"pv\.planes\.Ost\.kwp"):
        Site.load(path)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _VALID + "      colour: red\n")
    with pytest.raises(ValueError, match=r"colour"):
        Site.load(path)


def test_empty_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"location"):
        Site.load(_write(tmp_path, ""))


def test_site_is_frozen(tmp_path: Path) -> None:
    site = Site.load(_write(tmp_path, _VALID))
    with pytest.raises(AttributeError):
        site.timezone = "UTC"  # type: ignore[misc]
