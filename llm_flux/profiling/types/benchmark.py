"""Pydantic types for model accuracy / perplexity benchmarks."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm_flux.profiling.types.stats import DescriptiveStats


class ComputePerplexityForBatchReturn(BaseModel):
    perplexities: list[float]
    mean_perplexity: float


class ComputePerplexityForDatasetReturn(BaseModel):
    all_perplexities: list[float]
    mean_perplexity: float
    #: Full distributional summary derived from all_perplexities.
    stats: DescriptiveStats | None = None


class PerplexityTestResult(BaseModel):
    test_name: str
    result: ComputePerplexityForDatasetReturn


class AccuracyTestResult(BaseModel):
    test_name: str
    metric_name: str = "accuracy"
    accuracy: float
    num_examples: int
    details: dict[str, Any] = {}
    #: Individual per-example scores (F1, 0/1 correctness, MC2 mass, etc.).
    per_example_scores: list[float] = Field(default_factory=list)
    #: Full distributional summary derived from per_example_scores.
    stats: DescriptiveStats | None = None
