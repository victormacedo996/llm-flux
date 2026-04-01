"""Pydantic types for inference performance benchmarking."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class InferencePerformanceInfo(BaseModel):
    # ── Central tendency ─────────────────────────────────────────────────────
    avg_time: float        # seconds — arithmetic mean
    std_time: float = 0.0  # seconds — sample std dev (ddof=1)
    # ── Order statistics ─────────────────────────────────────────────────────
    min_time: float        # seconds
    p5_time: float = 0.0   # seconds
    p50_time: float = 0.0  # seconds — median
    p95_time: float = 0.0  # seconds
    p99_time: float = 0.0  # seconds
    max_time: float        # seconds
    # ── Throughput & bookkeeping ──────────────────────────────────────────────
    tokens_per_second: float
    num_runs: int
    generated_tokens: int
    #: Raw per-run wall-clock times (seconds) for offline analysis.
    raw_times: List[float] = Field(default_factory=list)


class CompareBenchmark(BaseModel):
    speedup: float                  # compressed_time / original_time ratio
    tps_improvement_percent: float  # positive = faster
