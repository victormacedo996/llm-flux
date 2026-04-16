"""
core/results.py — Top-level pipeline run result.

PipelineRunResult is what the executor returns after a full run.
It carries all ProfilingResults and can generate dissertation-ready output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from llm_flux.core.profiling import ProfilingResult

if TYPE_CHECKING:
    from llm_flux.core.html_reporter import generate_html_report


class PipelineRunResult(BaseModel):
    """
    Aggregated result of a complete pipeline execution.

    - ``profiling_records`` contains one ``ProfilingResult`` per PROFILE step.
    - ``to_markdown_table()`` emits a paste-ready dissertation results table.
    - ``save_json()`` writes the full result (including ``extra`` raw dumps) to disk.
    """

    pipeline_name: str
    pipeline_description: str = ""
    started_at: datetime
    finished_at: datetime
    profiling_records: list[ProfilingResult | list[ProfilingResult]] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_records(
        records: list[ProfilingResult | list[ProfilingResult]],
    ) -> list[ProfilingResult]:
        """Un-nest lists so every item is a single ProfilingResult."""
        result: list[ProfilingResult] = []
        for r in records:
            if isinstance(r, list):
                result.extend(r)
            else:
                result.append(r)
        return result

    def _record_rows(
        self,
    ) -> list[tuple[ProfilingResult, str]]:
        """Yield (record, metrics_summary) pairs for table/log rendering."""
        for r in self._flatten_records(self.profiling_records):
            if r.accuracy and r.accuracy.task_metrics:
                metric_str = " | ".join(str(tm) for tm in r.accuracy.task_metrics)
            elif r.accuracy and r.accuracy.task_name:
                metric_str = f"{r.accuracy.task_name}={r.accuracy.task_score}"
            elif r.accuracy and r.accuracy.perplexity is not None:
                metric_str = f"ppl={r.accuracy.perplexity:.3f}"
            else:
                metric_str = "—"
            yield r, metric_str

    def to_markdown_table(self) -> str:
        """
        Generate a Markdown table ready to paste into a dissertation.

        Example output::

            | Stage                  | Latency p95 (ms) | GPU Peak (MB) | Metrics                     |
            | ---------------------- | ---------------- | ------------- | ---------------------------- |
            | Baseline Profiling     | 42.3             | 14,500        | hellaswag/acc=0.6432 ±0.009  |
            | Post-GPTQ Profiling    | 18.7             | 5,200         | hellaswag/acc=0.5891          |
            | Post-Healing Profiling | 19.1             | 5,300         | hellaswag/acc=0.6312 ±0.008  |
        """
        header = (
            "| Stage | Latency p95 (ms) | GPU Peak (MB) | Metrics |\n"
            "| ----- | :--------------: | :-----------: | ------- |"
        )
        rows: list[str] = [header]
        for r, metric_str in self._record_rows():
            gpu = f"{r.memory.peak_gpu_mb:.0f}" if r.memory.peak_gpu_mb else "—"
            rows.append(f"| {r.pipeline_stage} | {r.latency.p95_ms:.1f} | {gpu} | {metric_str} |")
        return "\n".join(rows)

    def to_dissertation_log(self) -> str:
        """
        Generate a human-readable experiment log suitable for a dissertation
        Methods section or appendix.
        """
        lines: list[str] = [
            f"Experiment: {self.pipeline_name}",
            f"Objective:  {self.pipeline_description}",
            f"Duration:   {self.duration_seconds:.1f} s",
            "",
        ]
        idx = 0
        for r, metric_str in self._record_rows():
            idx += 1
            lines.append(f"Profiling checkpoint {idx} — {r.pipeline_stage}")
            base = f"  [{r.pipeline_stage}] Latency p95={r.latency.p95_ms:.1f} ms"
            if r.memory.peak_gpu_mb is not None:
                base += f" | GPU peak={r.memory.peak_gpu_mb:.0f} MB"
            base += f" | {metric_str}"
            lines.append(base)
        return "\n".join(lines)

    def save_json(self, path: str | Path = "pipeline_result.json") -> Path:
        """Persist the full result (including raw ``extra`` dumps) to JSON."""
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2))
        return out

    def save_html_report(
        self,
        path: str | Path = "pipeline_report.html",
        dag_image_path: str | Path | None = None,
        dag_echarts_data: dict | None = None,
    ) -> Path:
        """
        Generate an interactive HTML report from the pipeline results.

        Creates a single self-contained HTML file with:
        - Jinja2-templated structure
        - Apache ECharts visualizations
        - Embedded DAG image or interactive ECharts graph
        - Model architecture information
        - Hardware profiling data

        Args:
            path: Output path for the HTML file (default: pipeline_report.html)
            dag_image_path: Optional path to DAG PNG image to embed in report
            dag_echarts_data: Optional ECharts data dict for interactive DAG visualization

        Returns:
            Path: The absolute path to the generated HTML file
        """
        from llm_flux.core.html_reporter import generate_html_report

        return generate_html_report(
            self, output_path=path, dag_image_path=dag_image_path, dag_echarts_data=dag_echarts_data
        )
