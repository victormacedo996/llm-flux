"""
core/html_reporter.py — Generate interactive HTML reports using Jinja2 and Apache ECharts.

Converts PipelineRunResult to a self-contained HTML file with:
- Jinja2 templating for clean code/template separation
- Apache ECharts for advanced visualizations
- Embedded DAG image
- Model architecture and hardware information
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Template, Environment, FileSystemLoader

if TYPE_CHECKING:
    from llm_flux.core.results import PipelineRunResult

from loguru import logger


def generate_html_report(
    result: PipelineRunResult,
    output_path: str | Path = "pipeline_report.html",
    dag_image_path: str | Path | None = None,
) -> Path:
    """
    Generate an interactive HTML report from a PipelineRunResult.

    Args:
        result: The PipelineRunResult object from pipeline execution.
        output_path: Where to save the HTML file.
        dag_image_path: Optional path to DAG PNG image to embed.

    Returns:
        Path: The absolute path to the generated HTML file.
    """
    output_path = Path(output_path)
    dag_image_b64 = _encode_image_to_base64(dag_image_path)
    html_content = _render_template(result, dag_image_b64)
    output_path.write_text(html_content)
    return output_path.absolute()


def _encode_image_to_base64(image_path: str | Path | None) -> str:
    """Encode image file to base64 data URL."""
    if not image_path or not Path(image_path).exists():
        return ""
    image_data = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_model_info(record: Any) -> dict[str, Any]:
    """Extract model information from profiling record."""
    if not record.extra or "llm_profile" not in record.extra:
        return {}
    profile = record.extra.get("llm_profile", {})
    return {
        "model_class": profile.get("summary", {}).get("model_class", "Unknown"),
        "total_parameters": profile.get("parameters", {}).get("total", 0),
        "total_parameters_millions": profile.get("parameters", {}).get("total_millions", 0),
        "trainable_parameters": profile.get("parameters", {}).get("trainable", 0),
        "total_layers": profile.get("architecture", {}).get("total_layers", 0),
    }


def _extract_hardware_info(record: Any) -> dict[str, Any]:
    """Extract hardware information from profiling record."""
    if not record.extra or "hardware" not in record.extra:
        return {}
    hw = record.extra.get("hardware", {})
    cpu_info = hw.get("cpu", {})
    gpu_info = hw.get("gpu", {})
    ram_info = hw.get("ram", {})
    return {
        "cpu_name": cpu_info.get("name", "Unknown"),
        "cpu_cores": cpu_info.get("total_cores", 0),
        "cuda_available": gpu_info.get("cuda_available", False),
        "gpu_count": gpu_info.get("device_count", 0),
        "gpu_devices": gpu_info.get("gpus", []),
        "total_ram_gb": ram_info.get("total_memory_gb", 0),
        "available_ram_gb": ram_info.get("available_memory_gb", 0),
    }


def _render_template(result: PipelineRunResult, dag_image_data_url: str) -> str:
    """Render HTML using Jinja2 template."""
    # Extract metrics
    stages = [r.pipeline_stage for r in result.profiling_records]
    latencies_mean = [r.latency.mean_ms for r in result.profiling_records]
    latencies_p50 = [r.latency.p50_ms for r in result.profiling_records]
    latencies_p95 = [r.latency.p95_ms for r in result.profiling_records]
    gpu_peaks = [r.memory.peak_gpu_mb if r.memory.peak_gpu_mb else 0 for r in result.profiling_records]
    perplexities = [
        r.accuracy.perplexity if r.accuracy and r.accuracy.perplexity else None
        for r in result.profiling_records
    ]
    has_perplexity = any(p is not None for p in perplexities)


    # Build profiling data with model and hardware info
    profiling_data = []
    for record in result.profiling_records:
        profiling_data.append({
            "stage": record.pipeline_stage,
            "latency_mean": record.latency.mean_ms,
            "latency_p50": record.latency.p50_ms,
            "latency_p95": record.latency.p95_ms,
            "latency_p99": record.latency.p99_ms,
            "gpu_memory": record.memory.peak_gpu_mb,
            "cpu_memory": record.memory.peak_cpu_mb,
            "perplexity": record.accuracy.perplexity if record.accuracy else None,
            "model": _extract_model_info(record),
            "hardware": _extract_hardware_info(record),
        })

    template_dir = Path(__file__).parent / "templates"
    loader = FileSystemLoader(searchpath=template_dir)
    env = Environment(loader=loader, autoescape=True)
    template = env.get_template('html_reporter.html.jinja2')
    return template.render(
        pipeline_name=result.pipeline_name,
        pipeline_description=result.pipeline_description,
        duration_seconds=result.duration_seconds,
        started_at=result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        finished_at=result.finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        num_stages=len(result.profiling_records),
        stages_json=json.dumps(stages),
        latencies_mean_json=json.dumps(latencies_mean),
        latencies_p50_json=json.dumps(latencies_p50),
        latencies_p95_json=json.dumps(latencies_p95),
        gpu_peaks_json=json.dumps(gpu_peaks),
        perplexities_json=json.dumps([p if p is not None else 0 for p in perplexities]),
        has_perplexity=has_perplexity,
        profiling_data_json=json.dumps(profiling_data),
        dag_image_data_url=dag_image_data_url,
    )


