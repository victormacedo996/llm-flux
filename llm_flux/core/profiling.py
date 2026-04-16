"""
core/profiling.py — Port: Profiling abstraction + structured output types.

ProfilingResult is the single source of truth for all profiling data.
It is serializable to JSON and can generate dissertation-ready Markdown tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from llm_flux.core.model import ModelHandle

from llm_flux.core.task_metrics import TaskMetrics


# ── Output sub-models ─────────────────────────────────────────────────────────


class MemoryMetrics(BaseModel):
    """GPU and CPU peak memory captured during a profiling pass."""

    peak_gpu_mb: float | None = None
    peak_cpu_mb: float | None = None


class LatencyMetrics(BaseModel):
    """Inference latency statistics across benchmark runs (milliseconds)."""

    mean_ms: float
    std_ms: float = 0.0
    min_ms: float = 0.0
    p5_ms: float = 0.0
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float = 0.0


class AccuracyMetrics(BaseModel):
    """Task-specific accuracy / quality measures."""

    perplexity: float | None = None
    task_score: float | None = None
    task_name: str | None = None
    task_metrics: list[TaskMetrics] = Field(default_factory=list)


# ── Top-level result ──────────────────────────────────────────────────────────


class ProfilingResult(BaseModel):
    """
    Structured output of a single profiling run.

    This model is the dissertation's single source of truth for any
    measurement. The ``extra`` field stores arbitrary raw dumps (hardware
    profile, full LLM architecture info, benchmark lists) so that nothing
    is lost in the JSON export while keeping the top-level fields clean.
    """

    profiler_name: str
    model_label: str
    pipeline_stage: str  # e.g. "Baseline", "After GPTQ-4bit"
    timestamp: datetime = Field(default_factory=datetime.now)

    memory: MemoryMetrics = Field(default_factory=MemoryMetrics)
    latency: LatencyMetrics
    accuracy: AccuracyMetrics | None = None

    # Raw dumps keyed by: "hardware", "llm_profile", "inference", "benchmarks"
    extra: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """One-liner for log output and dissertation tables."""
        ppl = (
            f"{self.accuracy.perplexity:.3f}"
            if self.accuracy and self.accuracy.perplexity is not None
            else "N/A"
        )
        gpu = f"{self.memory.peak_gpu_mb:.0f} MB" if self.memory.peak_gpu_mb is not None else "N/A"
        # Build task metrics string
        tm_parts: list[str] = []
        if self.accuracy and self.accuracy.task_metrics:
            for tm in self.accuracy.task_metrics:
                tm_parts.append(str(tm))
        tm_str = " | ".join(tm_parts) if tm_parts else "N/A"
        return (
            f"[{self.pipeline_stage}] "
            f"Latency p95={self.latency.p95_ms:.1f} ms | "
            f"GPU peak={gpu} | PPL={ppl} | {tm_str}"
        )


# ── Port ──────────────────────────────────────────────────────────────────────


class ProfilingConfig(BaseModel):
    """
    Base configuration for any ProfilingPort implementation.
    Subclass to add profiler-specific parameters.
    """

    name: str
    description: str = ""
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10


class ProfilingPort(ABC):
    """
    Port: a profiling strategy that produces a ``ProfilingResult``.

    Every concrete profiler (whether it wraps torch.profiler, deepspeed,
    or the custom LLMProfiler) must implement this single method.
    """

    config: ProfilingConfig

    @abstractmethod
    def profile(
        self,
        stage_label: str,
        model_handle: ModelHandle | None = None,
    ) -> ProfilingResult:
        """
        Run profiling against ``model`` and return a fully-populated
        ``ProfilingResult``.

        Args:
            model: The model to profile (raw ``nn.Module`` or equivalent).
            stage_label: Human-readable label describing the pipeline stage
                         at which this profiling run occurs
                         (e.g. "After GPTQ-4bit").  Stored verbatim in
                         ``ProfilingResult.pipeline_stage``.
            model_handle: Optional ``ModelHandle`` that owns the loaded model.
                          Useful when profiling requires tokenizer or other
                          metadata kept by the handle.
        """

    @property
    def label(self) -> str:
        return self.config.name
