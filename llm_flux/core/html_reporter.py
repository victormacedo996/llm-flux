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
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from llm_flux.core.results import PipelineRunResult

from loguru import logger

# ── Layer-type colour palette (ECharts) ─────────────────────────────────────
_LAYER_COLORS: dict[str, str] = {
    "Embedding": "#5470c6",
    "Linear": "#91cc75",
    "LlamaRMSNorm": "#fac858",
    "LayerNorm": "#fac858",
    "SiLUActivation": "#ee6666",
    "GELU": "#ee6666",
    "LlamaRotaryEmbedding": "#73c0de",
    "MultiheadAttention": "#3ba272",
    "Dropout": "#fc8452",
    "ParameterDict": "#9a60b4",
    "ModuleDict": "#ea7ccc",
    "Placeholder": "#b0b0b0",
}
_COLOR_DEFAULT = "#aaa"


def generate_html_report(
    result: PipelineRunResult,
    output_path: str | Path = "pipeline_report.html",
    dag_image_path: str | Path | None = None,
    dag_echarts_data: dict | None = None,
) -> Path:
    """
    Generate an interactive HTML report from a PipelineRunResult.

    Args:
        result: The PipelineRunResult object from pipeline execution.
        output_path: Where to save the HTML file.
        dag_image_path: Optional path to DAG PNG image to embed.
        dag_echarts_data: Optional ECharts data dict for interactive DAG visualization.

    Returns:
        Path: The absolute path to the generated HTML file.
    """
    output_path = Path(output_path)
    dag_image_b64 = _encode_image_to_base64(dag_image_path)
    html_content = _render_template(result, dag_image_b64, dag_echarts_data)
    output_path.write_text(html_content)
    return output_path.absolute()


def _encode_image_to_base64(image_path: str | Path | None) -> str:
    """Encode image file to base64 data URL."""
    if not image_path or not Path(image_path).exists():
        return ""
    image_data = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── Architecture graph builder (truncated, ECharts) ─────────────────────────


def _block_index(name: str) -> int | None:
    """Return the transformer block index if the layer lives inside one."""
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _build_arch_graph_data(layer_details: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a memory-efficient ECharts graph option from layer_details.

    Strategy:
    1. Detect the unique transformer block *template* (block 0).
    2. Count how many identical blocks exist.
    3. Render: [pre-block layers] -> [Block 0 layers] -> ellipsis node
       -> [post-block layers].
    4. Sequential edges connect nodes in execution order.
    """
    if not layer_details:
        return {}

    # Split layers into pre, block groups, post
    pre: list[dict] = []
    blocks: dict[int, list[dict]] = defaultdict(list)
    post: list[dict] = []

    for ld in layer_details:
        idx = _block_index(ld["name"])
        if idx is None:
            if blocks:
                post.append(ld)
            else:
                pre.append(ld)
        else:
            blocks[idx].append(ld)

    num_blocks = len(blocks)
    block0 = blocks.get(0, [])

    # Build the rendered node list
    rendered_layers: list[dict] = list(pre)
    if block0:
        rendered_layers.extend(block0)
    if num_blocks > 1:
        rendered_layers.append(
            {
                "name": "__ellipsis__",
                "type": "Placeholder",
                "parameters": 0,
                "depth": 0,
                "_label": f"\u22ef {num_blocks - 1} more block{'s' if num_blocks - 1 > 1 else ''}",
            }
        )
    rendered_layers.extend(post)

    # Assign x/y positions — horizontal layout, evenly spaced
    total = len(rendered_layers)
    x_step = 1000 / max(total - 1, 1)

    echarts_nodes: list[dict] = []
    for i, ld in enumerate(rendered_layers):
        node_name = ld.get("name", f"node_{i}")
        label = ld.get("_label") or node_name.split(".")[-1] or node_name
        layer_type = ld.get("type", "Unknown")
        params = ld.get("parameters", 0)
        symbol_size = max(14, min(44, 14 + int((params**0.35) * 0.5))) if params else 16
        color = _LAYER_COLORS.get(layer_type, _COLOR_DEFAULT)

        tooltip_lines = [
            f"<b>{node_name}</b>",
            f"Type: {layer_type}",
            f"Parameters: {params:,}",
        ]
        if ld.get("input_size"):
            tooltip_lines.append(f"Shape: {ld['input_size']} \u2192 {ld.get('output_size', '?')}")
        if ld.get("vocab_size"):
            tooltip_lines.append(f"Vocab: {ld['vocab_size']} \u00d7 {ld.get('embedding_dim', '?')}")

        echarts_nodes.append(
            {
                "id": str(i),
                "name": label,
                "x": round(i * x_step, 1),
                "y": 0,
                "symbolSize": symbol_size,
                "itemStyle": {"color": color},
                "tooltip": {"formatter": "<br>".join(tooltip_lines)},
                "category": layer_type,
            }
        )

    # Sequential edges
    echarts_edges: list[dict] = []
    for i in range(len(echarts_nodes) - 1):
        echarts_edges.append(
            {
                "source": str(i),
                "target": str(i + 1),
                "lineStyle": {"color": "#ccc", "width": 1.5},
            }
        )

    # Unique categories for legend
    seen_types: list[str] = []
    for node in echarts_nodes:
        t = node["category"]
        if t not in seen_types:
            seen_types.append(t)
    categories = [
        {"name": t, "itemStyle": {"color": _LAYER_COLORS.get(t, _COLOR_DEFAULT)}}
        for t in seen_types
    ]

    return {
        "tooltip": {"show": True, "enterable": True},
        "legend": [
            {"data": [c["name"] for c in categories], "top": 5, "textStyle": {"fontSize": 10}}
        ],
        "animationDurationUpdate": 500,
        "_meta": {
            "total_original": len(layer_details),
            "total_rendered": len(echarts_nodes),
            "num_blocks": num_blocks,
        },
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "roam": True,
                "draggable": True,
                "label": {
                    "show": True,
                    "position": "bottom",
                    "fontSize": 9,
                    "formatter": "{b}",
                },
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [0, 6],
                "categories": categories,
                "data": echarts_nodes,
                "edges": echarts_edges,
            }
        ],
    }


# ── Per-record extractors ────────────────────────────────────────────────────


def _extract_model_info(record: Any) -> dict[str, Any]:
    """Extract full model / LLM-profile information from a profiling record."""
    if not record.extra or "llm_profile" not in record.extra:
        return {}
    profile = record.extra.get("llm_profile", {})
    params = profile.get("parameters", {})
    arch = profile.get("architecture", {})
    attn = profile.get("attention_layers", {})
    summary = profile.get("summary", {})
    mem_est = profile.get("memory_estimation") or {}
    layer_details = arch.get("layer_details") or []
    return {
        "model_class": summary.get("model_class", "Unknown"),
        "device": summary.get("device", "—"),
        "pytorch_version": summary.get("pytorch_version", "—"),
        "total_parameters": params.get("total", 0),
        "total_parameters_millions": params.get("total_millions", 0),
        "total_parameters_billions": params.get("total_billions", 0),
        "trainable_parameters": params.get("trainable", 0),
        "non_trainable_parameters": params.get("non_trainable", 0),
        "by_dtype_counts": params.get("by_dtype_counts", {}),
        "by_dtype_bytes": params.get("by_dtype_bytes", {}),
        "total_layers": arch.get("total_layers", 0),
        "max_depth": arch.get("max_depth", 0),
        "layer_types_count": arch.get("layer_types_count", {}),
        "layer_details": layer_details,
        "num_attention_layers": attn.get("num_attention_layers", 0),
        "attention_layers": attn.get("attention_layers", []),
        "memory_estimates": mem_est.get("estimates", {}),
        "arch_graph": _build_arch_graph_data(layer_details),
    }


def _extract_hardware_info(record: Any) -> dict[str, Any]:
    """Extract full hardware information from a profiling record."""
    if not record.extra or "hardware" not in record.extra:
        return {}
    hw = record.extra.get("hardware", {})
    cpu = hw.get("cpu", {})
    gpu = hw.get("gpu", {})
    ram = hw.get("ram", {})
    return {
        # CPU
        "cpu_name": cpu.get("name", "Unknown"),
        "cpu_architecture": cpu.get("architecture", "—"),
        "cpu_platform": cpu.get("platform", "—"),
        "cpu_physical_cores": cpu.get("physical_cores", 0),
        "cpu_total_cores": cpu.get("total_cores", 0),
        "cpu_max_freq_mhz": cpu.get("max_freq", 0),
        # GPU
        "cuda_available": gpu.get("cuda_available", False),
        "cuda_version": gpu.get("cuda_version"),
        "cudnn_version": gpu.get("cudnn_version"),
        "gpu_count": gpu.get("device_count", 0),
        "gpu_devices": gpu.get("gpus", []),
        # RAM
        "total_ram_gb": ram.get("total_memory_gb", 0),
        "available_ram_gb": ram.get("available_memory_gb", 0),
        "free_ram_gb": ram.get("free_memory_gb", 0),
        "swap_total_gb": ram.get("swap_total_gb", 0),
        "swap_free_gb": ram.get("swap_free_gb", 0),
    }


def _extract_inference_info(record: Any) -> dict[str, Any]:
    """Extract inference benchmarking data from a profiling record."""
    if not record.extra or "inference" not in record.extra:
        return {}
    inf = record.extra.get("inference", {})
    return {
        "avg_time_ms": inf.get("avg_time", 0) * 1000,
        "std_time_ms": inf.get("std_time", 0) * 1000,
        "min_time_ms": inf.get("min_time", 0) * 1000,
        "p5_time_ms": inf.get("p5_time", 0) * 1000,
        "p50_time_ms": inf.get("p50_time", 0) * 1000,
        "p95_time_ms": inf.get("p95_time", 0) * 1000,
        "p99_time_ms": inf.get("p99_time", 0) * 1000,
        "max_time_ms": inf.get("max_time", 0) * 1000,
        "tokens_per_second": inf.get("tokens_per_second", 0),
        "num_runs": inf.get("num_runs", 0),
        "generated_tokens": inf.get("generated_tokens", 0),
        "raw_times_ms": [t * 1000 for t in inf.get("raw_times", [])],
    }


def _extract_benchmarks_info(record: Any) -> list[dict[str, Any]]:
    """
    Extract all benchmark test results from a profiling record.

    Also handles lm-eval results stored in ``record.extra['lm_eval']``
    by converting them to the same benchmark-entry format expected by
    ``benchmarks_detail.html.jinja2``.
    """
    results: list[dict[str, Any]] = []

    # ── native benchmarks (existing) ────────────────────────────────────────
    if record.extra and "benchmarks" in record.extra:
        for b in record.extra.get("benchmarks", []):
            test_name = b.get("test_name", "unknown")
            res = b.get("result", {})
            entry: dict[str, Any] = {"test_name": test_name}
            if "mean_perplexity" in res:
                entry["mean_perplexity"] = res["mean_perplexity"]
                entry["all_perplexities"] = res.get("all_perplexities", [])
                entry["stats"] = res.get("stats")
                entry["kind"] = "perplexity"
            elif "accuracy" in b:
                entry["accuracy"] = b["accuracy"]
                entry["num_examples"] = b.get("num_examples", 0)
                entry["metric_name"] = b.get("metric_name", "accuracy")
                entry["per_example_scores"] = b.get("per_example_scores", [])
                entry["stats"] = b.get("stats")
                entry["details"] = b.get("details", {})
                entry["kind"] = "accuracy"
            else:
                entry["kind"] = "unknown"
            results.append(entry)

    # ── lm-eval results ────────────────────────────────────────────────────────
    if record.extra and "lm_eval" in record.extra:
        lm = record.extra["lm_eval"]
        lm_results = lm.get("results", {})
        n_shot = lm.get("n-shot", {})
        n_samples = lm.get("n-samples", {})
        higher_is_better = lm.get("higher_is_better", {})

        for task_name, metrics in lm_results.items():
            if not isinstance(metrics, dict):
                continue

            # Collect (metric_name, value) pairs, separating primary from stderr
            primary: dict[str, float] = {}
            stderr_map: dict[str, float | None] = {}
            for k, v in metrics.items():
                if not isinstance(v, (int, float, str)):
                    continue
                if isinstance(v, str):
                    if v == "N/A":
                        v = None
                    else:
                        try:
                            v = float(v)
                        except ValueError:
                            continue
                if "," in k:
                    metric_part, _ = k.split(",", 1)
                else:
                    metric_part = k
                if metric_part.endswith("_stderr"):
                    base = metric_part[: -len("_stderr")]
                    stderr_map[base] = float(v) if v is not None else None
                else:
                    primary[metric_part] = float(v) if v is not None else 0.0

            for metric_name, metric_value in primary.items():
                stderr = stderr_map.get(metric_name)
                ns = n_shot.get(task_name, 0)
                eff_samples = (
                    n_samples.get(task_name, {}).get("effective")
                    if isinstance(n_samples.get(task_name), dict)
                    else 0
                )
                hib = higher_is_better.get(task_name, {}).get(metric_name, True)

                entry: dict[str, Any] = {
                    "test_name": f"{task_name}/{metric_name}",
                    "kind": "lm_eval",
                    "accuracy": metric_value,
                    "stderr": stderr,
                    "num_examples": eff_samples,
                    "metric_name": metric_name,
                    "num_fewshot": ns,
                    "higher_is_better": hib,
                    "task_name": task_name,
                }
                results.append(entry)

    return results


# ── Template rendering ───────────────────────────────────────────────────────


def _render_template(
    result: PipelineRunResult, dag_image_data_url: str, dag_echarts_data: dict | None
) -> str:
    """Render HTML using Jinja2 template."""
    stages = [r.pipeline_stage for r in result.profiling_records]
    latencies_mean = [r.latency.mean_ms for r in result.profiling_records]
    latencies_std = [r.latency.std_ms for r in result.profiling_records]
    latencies_min = [r.latency.min_ms for r in result.profiling_records]
    latencies_p5 = [r.latency.p5_ms for r in result.profiling_records]
    latencies_p50 = [r.latency.p50_ms for r in result.profiling_records]
    latencies_p95 = [r.latency.p95_ms for r in result.profiling_records]
    latencies_p99 = [r.latency.p99_ms for r in result.profiling_records]
    latencies_max = [r.latency.max_ms for r in result.profiling_records]
    gpu_peaks = [
        r.memory.peak_gpu_mb if r.memory.peak_gpu_mb else 0 for r in result.profiling_records
    ]
    perplexities = [
        r.accuracy.perplexity if r.accuracy and r.accuracy.perplexity else None
        for r in result.profiling_records
    ]
    has_perplexity = any(p is not None for p in perplexities)

    profiling_data: list[dict[str, Any]] = []
    for record in result.profiling_records:
        profiling_data.append(
            {
                "stage": record.pipeline_stage,
                "latency_mean": record.latency.mean_ms,
                "latency_std": record.latency.std_ms,
                "latency_min": record.latency.min_ms,
                "latency_p5": record.latency.p5_ms,
                "latency_p50": record.latency.p50_ms,
                "latency_p95": record.latency.p95_ms,
                "latency_p99": record.latency.p99_ms,
                "latency_max": record.latency.max_ms,
                "gpu_memory": record.memory.peak_gpu_mb,
                "cpu_memory": record.memory.peak_cpu_mb,
                "perplexity": record.accuracy.perplexity if record.accuracy else None,
                "test_name": record.accuracy.task_name if record.accuracy else None,
                "model": _extract_model_info(record),
                "hardware": _extract_hardware_info(record),
                "inference": _extract_inference_info(record),
                "benchmarks": _extract_benchmarks_info(record),
            }
        )

    tps_values = [pd["inference"].get("tokens_per_second", 0) for pd in profiling_data]

    template_dir = Path(__file__).parent / "templates"
    loader = FileSystemLoader(searchpath=template_dir)
    env = Environment(loader=loader, autoescape=True)
    env.filters["from_json"] = json.loads
    template = env.get_template("report/html_reporter.html.jinja2")
    return template.render(
        pipeline_name=result.pipeline_name,
        pipeline_description=result.pipeline_description,
        duration_seconds=result.duration_seconds,
        started_at=result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
        finished_at=result.finished_at.strftime("%Y-%m-%d %H:%M:%S"),
        num_stages=len(result.profiling_records),
        stages_json=json.dumps(stages),
        latencies_mean_json=json.dumps(latencies_mean),
        latencies_std_json=json.dumps(latencies_std),
        latencies_min_json=json.dumps(latencies_min),
        latencies_p5_json=json.dumps(latencies_p5),
        latencies_p50_json=json.dumps(latencies_p50),
        latencies_p95_json=json.dumps(latencies_p95),
        latencies_p99_json=json.dumps(latencies_p99),
        latencies_max_json=json.dumps(latencies_max),
        gpu_peaks_json=json.dumps(gpu_peaks),
        perplexities_json=json.dumps([p if p is not None else 0 for p in perplexities]),
        tps_json=json.dumps(tps_values),
        has_perplexity=has_perplexity,
        profiling_data_json=json.dumps(profiling_data),
        dag_image_data_url=dag_image_data_url,
        dag_echarts_data_json=json.dumps(dag_echarts_data) if dag_echarts_data else None,
    )
