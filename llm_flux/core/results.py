"""
core/results.py — Top-level pipeline run result.

PipelineRunResult is what the executor returns after a full run.
It carries all ProfilingResults and can generate dissertation-ready output.
"""

from __future__ import annotations

import csv
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from llm_flux.core.profiling import ProfilingResult

if TYPE_CHECKING:
    pass


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
    ) -> Generator[tuple[ProfilingResult, str], None, None]:
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

    # ── CSV output ─────────────────────────────────────────────────────────────

    def _extract_benchmark_metrics(self, record: ProfilingResult) -> dict[str, float | str | None]:
        """Extract benchmark metrics from a ProfilingResult for CSV output."""
        metrics: dict[str, float | str | None] = {}

        # ── native benchmarks ──────────────────────────────────────────────────
        if record.extra and "benchmarks" in record.extra:
            for b in record.extra.get("benchmarks", []):
                test_name = b.get("test_name", "unknown")
                res = b.get("result", {})
                if "mean_perplexity" in res:
                    metrics[f"{test_name}_perplexity"] = res["mean_perplexity"]
                elif "accuracy" in b:
                    metric_name = b.get("metric_name", "acc")
                    metrics[f"{test_name}_{metric_name}"] = b["accuracy"]

        # ── lm-eval results ────────────────────────────────────────────────────
        if record.extra and "lm_eval" in record.extra:
            lm = record.extra["lm_eval"]
            lm_results = lm.get("results", {})
            for task_name, metrics_dict in lm_results.items():
                if not isinstance(metrics_dict, dict):
                    continue
                for k, v in metrics_dict.items():
                    if isinstance(v, str):
                        if v == "N/A":
                            v_val: float | None = None
                        else:
                            try:
                                v_val = float(v)
                            except ValueError:
                                continue
                    elif isinstance(v, (int, float)):
                        v_val = float(v)
                    else:
                        continue
                    if k.endswith("_stderr"):
                        continue
                    metrics[f"{task_name}_{k}"] = v_val

        return metrics

    def _get_comparison_metrics(
        self, record: ProfilingResult
    ) -> dict[str, float | int | str | None]:
        """Extract and format metrics for a single profiling record for comparison CSV."""
        metrics_data: dict[str, float | int | str | None] = {}

        if record.extra and "llm_profile" in record.extra:
            params_info = record.extra["llm_profile"].get("parameters", {})
            metrics_data["Parameters"] = params_info.get("total_parameters")

        if record.latency and record.latency.p95_ms is not None:
            metrics_data["Inference Time"] = round(record.latency.p95_ms / 1000.0, 3)

        benchmark_metrics = self._extract_benchmark_metrics(record)
        for key, value in benchmark_metrics.items():
            display_key = key.replace("_perplexity", "").replace("_acc", "")
            metrics_data[display_key] = value

        return metrics_data

    def to_comparison_csv(self, path: str | Path = "pipeline_comparison.csv") -> Path:
        """
        Generate a Metric / Model comparison CSV.

        Compares the first and last profiling checkpoints. Format::

            Metric,Model1,Model2,Change
            Parameters,268098176,256950912,-4.16%
            Inference Time,4.651s,4.181s,+10.1% faster
            arc_easy,0.55,0.46,-16.36%
            ...

        Change is expressed as a signed percentage. For Inference Time,
        positive means faster (lower latency is better).
        """
        all_records = self._flatten_records(self.profiling_records)
        if len(all_records) < 2:
            raise ValueError(
                "to_comparison_csv requires at least two profiling records "
                "(first and last) to compare."
            )

        record_a = all_records[0]
        record_b = all_records[-1]
        model_a = record_a.model_label
        model_b = record_b.model_label

        metrics_a = self._get_comparison_metrics(record_a)
        metrics_b = self._get_comparison_metrics(record_b)

        all_metric_keys: set[str] = set(metrics_a.keys()) | set(metrics_b.keys())
        metric_order = ["Parameters", "Inference Time"]
        sorted_keys = metric_order + sorted(k for k in all_metric_keys if k not in metric_order)

        out = Path(path)
        with open(out, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", model_a, model_b, "Change"])

            for key in sorted_keys:
                val_a = metrics_a.get(key)
                val_b = metrics_b.get(key)

                if val_a is None or val_b is None:
                    change_str = "N/A"
                elif key == "Inference Time":
                    if (
                        isinstance(val_a, (int, float))
                        and isinstance(val_b, (int, float))
                        and val_a != 0
                    ):
                        pct = ((val_a - val_b) / val_a) * 100
                        sign = "+" if pct > 0 else ""
                        change_str = f"{sign}{pct:.1f}% faster"
                    else:
                        change_str = "N/A"
                elif (
                    isinstance(val_a, (int, float))
                    and isinstance(val_b, (int, float))
                    and val_a != 0
                ):
                    pct = ((val_b - val_a) / val_a) * 100
                    sign = "+" if pct > 0 else ""
                    change_str = f"{sign}{pct:.2f}%"
                else:
                    change_str = "N/A"

                writer.writerow(
                    [
                        key,
                        val_a if val_a is not None else "",
                        val_b if val_b is not None else "",
                        change_str,
                    ]
                )

        return out.absolute()

    def save_csv(self, path: str | Path = "pipeline_results.csv") -> Path:
        """
        Persist profiling results as a CSV with one row per profiling checkpoint.

        Columns are dynamic: benchmark test names become columns
        (e.g. lambada_perplexity, hellaswag_acc). Static columns are:
        model, stage, tokens_per_second, latency_p95_ms, memory_peak_mb.
        """
        out = Path(path)
        all_records = self._flatten_records(self.profiling_records)

        benchmark_headers: set[str] = set()
        for record in all_records:
            benchmark_headers.update(self._extract_benchmark_metrics(record).keys())

        static_headers = [
            "model",
            "stage",
            "tokens_per_second",
            "latency_p95_ms",
            "memory_peak_mb",
        ]
        fieldnames = static_headers + sorted(benchmark_headers)

        with open(out, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in all_records:
                row: dict[str, Any] = {
                    "model": record.model_label,
                    "stage": record.pipeline_stage,
                    "tokens_per_second": (
                        record.extra["inference"]["tokens_per_second"]
                        if record.extra and "inference" in record.extra
                        else ""
                    ),
                    "latency_p95_ms": record.latency.p95_ms,
                    "memory_peak_mb": (
                        int(record.memory.peak_gpu_mb)
                        if record.memory.peak_gpu_mb is not None
                        else ""
                    ),
                }
                row.update(self._extract_benchmark_metrics(record))
                writer.writerow(row)

        return out.absolute()

    def save_html_report(
        self,
        path: str | Path = "pipeline_report.html",
        dag_image_path: str | Path | None = None,
        dag_echarts_data: dict[str, Any] | None = None,
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
            dag_echarts_data: Optional ECharts data dict for interactive DAG
                              visualization

        Returns:
            Path: The absolute path to the generated HTML file
        """
        from llm_flux.core.html_reporter import generate_html_report

        return generate_html_report(
            self, output_path=path, dag_image_path=dag_image_path, dag_echarts_data=dag_echarts_data
        )
