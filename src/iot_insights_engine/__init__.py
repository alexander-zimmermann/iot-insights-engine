"""The package's import-stable surface: the entity slug the lares generator
writes the writer rules with, imported at the deployed tag. The fault-list
loader stays at `iot_insights_engine.faults`."""

from .slug import entity_slug

__all__ = ["entity_slug"]
