"""
profiling/types/stats.py — Shared descriptive statistics model.

DescriptiveStats is the single contract for any metric that is sampled
more than once (inference latency runs, per-example accuracy/perplexity).
All benchmark result types embed this model rather than duplicating fields.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class DescriptiveStats(BaseModel):
    """
    Full distributional summary for a sequence of numeric measurements.

    Suitable for both computational metrics (latency, tokens/s) and
    model-performance metrics (perplexity, accuracy, F1, MC2).

    Fields
    ------
    n        : number of samples
    mean     : arithmetic mean
    std      : sample standard deviation (ddof=1); 0.0 when n ≤ 1
    min      : minimum value
    p5       : 5th percentile
    p25      : 25th percentile (Q1)
    median   : 50th percentile (Q2 / p50)
    p75      : 75th percentile (Q3)
    p95      : 95th percentile
    p99      : 99th percentile
    max      : maximum value
    """

    n: int
    mean: float
    std: float
    min: float
    p5: float
    p25: float
    median: float  # p50
    p75: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_values(cls, values: list[float]) -> DescriptiveStats:
        """
        Compute descriptive statistics from a list of floats.

        Returns an all-zero instance with n=0 for empty input so that
        callers never need to guard against None.
        """
        if not values:
            return cls(
                n=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                p5=0.0,
                p25=0.0,
                median=0.0,
                p75=0.0,
                p95=0.0,
                p99=0.0,
                max=0.0,
            )
        arr = np.array(values, dtype=float)
        return cls(
            n=int(len(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min=float(np.min(arr)),
            p5=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            median=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max=float(np.max(arr)),
        )
