"""
core/task_metrics.py — Structured metrics for individual evaluation tasks.

TaskMetrics holds a single (task, metric, value) tuple as returned by
lm-evaluation-harness or any other benchmarker that produces per-task scores.
"""

from __future__ import annotations

from pydantic import BaseModel


class TaskMetrics(BaseModel):
    """
    A single task-level metric.

    lm-evaluation-harness returns one or more (metric_name, value) pairs per
    task (e.g.  "acc": 0.642,  "acc_stderr": 0.009).  This model captures the
    primary value and, when available, its bootstrap standard error.
    """

    task_name: str
    """Task identifier as registered in the benchmarker (e.g. "mmlu_pro")."""

    metric_name: str
    """Metric reported by the task (e.g. "acc", "f1", "mc2", "perplexity")."""

    metric_value: float
    """Numeric value of the metric."""

    stderr: float | None = None
    """Bootstrap standard error, if reported by the benchmarker."""

    num_fewshot: int = 0
    """Number of few-shot examples used for this task."""

    def __str__(self) -> str:
        se = f" ±{self.stderr:.4f}" if self.stderr is not None else ""
        return f"{self.task_name}/{self.metric_name}={self.metric_value:.4f}{se}"
