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
    profiling_records: list[ProfilingResult] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def to_markdown_table(self) -> str:
        """
        Generate a Markdown table ready to paste into a dissertation.

        Example output::

            | Stage                  | Latency p95 (ms) | GPU Peak (MB) | Perplexity |
            | ---------------------- | ---------------- | ------------- | ---------- |
            | Baseline Profiling     | 42.3             | 14,500        | 6.840      |
            | Post-GPTQ Profiling    | 18.7             | 5,200         | 7.520      |
            | Post-Healing Profiling | 19.1             | 5,300         | 7.010      |
        """
        header = (
            "| Stage | Latency p95 (ms) | GPU Peak (MB) | Perplexity |\n"
            "| ----- | :--------------: | :-----------: | :--------: |"
        )
        rows: list[str] = [header]
        for r in self.profiling_records:
            ppl = (
                f"{r.accuracy.perplexity:.3f}"
                if r.accuracy and r.accuracy.perplexity is not None
                else "—"
            )
            gpu = f"{r.memory.peak_gpu_mb:.0f}" if r.memory.peak_gpu_mb else "—"
            rows.append(
                f"| {r.pipeline_stage} "
                f"| {r.latency.p95_ms:.1f} "
                f"| {gpu} "
                f"| {ppl} |"
            )
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
        for i, r in enumerate(self.profiling_records, start=1):
            lines.append(f"Profiling checkpoint {i} — {r.pipeline_stage}")
            lines.append(f"  {r.summary()}")
        return "\n".join(lines)

    def save_json(self, path: str | Path = "pipeline_result.json") -> Path:
        """Persist the full result (including raw ``extra`` dumps) to JSON."""
        out = Path(path)
        out.write_text(self.model_dump_json(indent=2))
        return out

    def save_html_report(self, path: str | Path = "pipeline_report.html", dag_image_path: str | Path | None = None, dag_echarts_data: dict | None = None) -> Path:
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
        return generate_html_report(self, output_path=path, dag_image_path=dag_image_path, dag_echarts_data=dag_echarts_data)
