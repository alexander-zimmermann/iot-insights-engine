"""The engine's job surface.

With the old detection switched off, only the forecast pulls and the
energy balance remain dispatchable — the detectors return as fault
entries, not as subcommands.
"""

from __future__ import annotations

import importlib

from iot_insights_engine.__main__ import SUBCOMMANDS


def test_no_detection_jobs_remain() -> None:
    assert set(SUBCOMMANDS) == {"forecast-solar", "forecast-weather", "energy-balance"}


def test_every_subcommand_resolves_to_a_module() -> None:
    for subcommand in SUBCOMMANDS:
        module = importlib.import_module(f"iot_insights_engine.{subcommand.replace('-', '_')}")
        assert callable(module.run)
