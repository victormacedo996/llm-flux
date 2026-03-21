"""Pydantic types for model accuracy / perplexity benchmarks."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class ComputePerplexityForBatchReturn(BaseModel):
    perplexities: List[float]
    mean_perplexity: float


class ComputePerplexityForDatasetReturn(BaseModel):
    all_perplexities: List[float]
    mean_perplexity: float


class PerplexityTestResult(BaseModel):
    test_name: str
    result: ComputePerplexityForDatasetReturn


class AccuracyTestResult(BaseModel):
    test_name: str
    accuracy: float
    num_examples: int
    details: Dict[str, Any] = {}
