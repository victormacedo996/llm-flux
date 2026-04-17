"""Pydantic types for LLM structural profiling results."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class ParameterInfo(BaseModel):
    total: int
    trainable: int
    non_trainable: int
    total_millions: float
    total_billions: float
    by_dtype_counts: dict[str, int] = Field(default_factory=dict)
    by_dtype_bytes: dict[str, int] = Field(default_factory=dict)


class LayerConnectionInfo(BaseModel):
    name: str
    type: str
    parameters: int
    depth: int
    input_layers: list[str] = Field(default_factory=list)
    output_layers: list[str] = Field(default_factory=list)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None


class ConnectionAnalysisInfo(BaseModel):
    total_connections: int
    analysis_method: str
    has_skip_connections: bool
    max_fan_in: int
    max_fan_out: int
    connection_graph: dict[str, LayerConnectionInfo] = Field(default_factory=dict)


class ArchitectureInfo(BaseModel):
    total_layers: int
    max_depth: int
    layer_types_count: dict[str, int] = Field(default_factory=dict)
    layer_details: list[dict[str, Any]] = Field(default_factory=list)
    connections: ConnectionAnalysisInfo | None = None


class AttentionLayerAnalysisInfo(BaseModel):
    num_attention_layers: int
    attention_layers: list[dict[str, Any]] = Field(default_factory=list)


class MemoryEstimate(BaseModel):
    precision: str
    bytes_per_parameter: float
    model_weights_mb: float
    kv_cache_mb: float | None = None
    activation_memory_mb: float | None = None
    gradient_memory_mb: float
    optimizer_memory_mb: float
    # training_memory_mb = gradient + optimizer overhead
    training_memory_mb: float
    total_memory_mb: float
    total_memory_gb: float
    total_inference_memory_mb: float


class MemoryEstimationInfo(BaseModel):
    base_parameters: int
    estimates: dict[str, MemoryEstimate] = Field(default_factory=dict)


class ModelSummary(BaseModel):
    device: str
    model_class: str
    pytorch_version: str


class LLMInfo(BaseModel):
    summary: ModelSummary
    parameters: ParameterInfo
    architecture: ArchitectureInfo
    attention_layers: AttentionLayerAnalysisInfo
    memory_estimation: MemoryEstimationInfo | None = None


# ── Optional profiling trigger configs ───────────────────────────────────────


class AnalyzeConnections(BaseModel):
    """Pass to LLMProfiler.profile_complete() to enable connection analysis."""

    input_shape: tuple | None = None  # type: ignore[type-arg]
    sample_input: Any | None = None   # callable or tensor

    model_config = {"arbitrary_types_allowed": True}


class EstimateMemory(BaseModel):
    """Pass to LLMProfiler.profile_complete() to enable memory estimation."""

    sequence_length: int = 2048
    batch_size: int = 1
