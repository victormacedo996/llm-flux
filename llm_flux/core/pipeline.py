"""
core/pipeline.py — Declarative pipeline definition.

Users describe WHAT to run and IN WHAT ORDER using plain Pydantic models.
No callables here — the registry wires the config_ref strings to Port instances.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator




class PipelineStep(BaseModel):
    """
    Declarative descriptor for one DAG node.
    """

    label: str  # human label rendered in the DAG graph and logs
    port: Any   # The actual port instance (ModelHandle, CompressionPort, etc.)

    @property
    def kind(self) -> str:
        """
        Infer the step type from the port's interface to keep the
        pipeline definition cleaner.
        """
        from llm_flux.core.compression import CompressionPort
        from llm_flux.core.healing import HealingPort
        from llm_flux.core.model import ModelHandle
        from llm_flux.core.profiling import ProfilingPort

        if isinstance(self.port, ModelHandle):
            return "load"
        if isinstance(self.port, CompressionPort):
            return "compress"
        if isinstance(self.port, ProfilingPort):
            return "profile"
        if isinstance(self.port, HealingPort):
            return "heal"

        raise ValueError(
            f"Step '{self.label}' has an unknown port type: {type(self.port).__name__}. "
            "Must inherit from ModelHandle, CompressionPort, ProfilingPort, or HealingPort."
        )


class Pipeline(BaseModel):
    """
    The complete user-defined pipeline.
    """

    name: str
    description: str = ""
    steps: list[PipelineStep]

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("steps")
    @classmethod
    def _must_start_with_load(cls, steps: list[PipelineStep]) -> list[PipelineStep]:
        if not steps or steps[0].kind != "load":
            raise ValueError("A Pipeline must begin with a LOAD step.")
        return steps

    @field_validator("steps")
    @classmethod
    def _labels_must_be_unique(cls, steps: list[PipelineStep]) -> list[PipelineStep]:
        labels = [s.label for s in steps]
        duplicates = {l for l in labels if labels.count(l) > 1}
        if duplicates:
            raise ValueError(
                f"PipelineStep labels must be unique. Duplicates found: {duplicates}"
            )
        return steps
