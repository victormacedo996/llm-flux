"""Pydantic types for inference performance benchmarking."""
from __future__ import annotations

from pydantic import BaseModel


class InferencePerformanceInfo(BaseModel):
    avg_time: float       # seconds
    min_time: float       # seconds
    max_time: float       # seconds
    tokens_per_second: float
    num_runs: int
    generated_tokens: int


class CompareBenchmark(BaseModel):
    speedup: float                  # compressed_time / original_time ratio
    tps_improvement_percent: float  # positive = faster
