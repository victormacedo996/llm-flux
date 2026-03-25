"""
Test script to demonstrate HTML report generation from a PipelineRunResult.
"""
from datetime import datetime, timedelta
from llm_flux.core.results import PipelineRunResult
from llm_flux.core.profiling import ProfilingResult, LatencyMetrics, MemoryMetrics, AccuracyMetrics


def create_demo_result() -> PipelineRunResult:
    """Create a demo PipelineRunResult for testing."""
    now = datetime.now()
    
    records = [
        ProfilingResult(
            profiler_name="baseline-profiler",
            model_label="TinyLlama-1.1B",
            pipeline_stage="Baseline Profile",
            timestamp=now,
            memory=MemoryMetrics(peak_gpu_mb=2048.5, peak_cpu_mb=512.3),
            latency=LatencyMetrics(mean_ms=45.2, p50_ms=44.1, p95_ms=52.3, p99_ms=58.7),
            accuracy=AccuracyMetrics(perplexity=6.840, task_name="lambada"),
        ),
        ProfilingResult(
            profiler_name="fast-profiler",
            model_label="TinyLlama-1.1B-pruned",
            pipeline_stage="Post-Pruning (20%)",
            timestamp=now + timedelta(minutes=5),
            memory=MemoryMetrics(peak_gpu_mb=1536.2, peak_cpu_mb=384.1),
            latency=LatencyMetrics(mean_ms=32.1, p50_ms=31.5, p95_ms=38.9, p99_ms=42.1),
            accuracy=AccuracyMetrics(perplexity=7.210, task_name="lambada"),
        ),
        ProfilingResult(
            profiler_name="fast-profiler",
            model_label="TinyLlama-1.1B-pruned-healed",
            pipeline_stage="Post-LoRA Healing",
            timestamp=now + timedelta(minutes=15),
            memory=MemoryMetrics(peak_gpu_mb=1600.7, peak_cpu_mb=400.5),
            latency=LatencyMetrics(mean_ms=33.5, p50_ms=32.8, p95_ms=40.2, p99_ms=44.3),
            accuracy=AccuracyMetrics(perplexity=6.950, task_name="lambada"),
        ),
    ]
    
    result = PipelineRunResult(
        pipeline_name="llama_1b_prune_20pct_awq",
        pipeline_description="1B LLaMA Depth Pruning (20%) → LoRA Healing (C4) → AWQ Pipeline.",
        started_at=now,
        finished_at=now + timedelta(minutes=20),
        profiling_records=records,
    )
    
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Create demo result
    result = create_demo_result()
    
    # Generate HTML report
    output_path = Path("/tmp/pipeline_report_demo.html")
    html_path = result.save_html_report(output_path)
    
    print(f"✓ HTML report generated: {html_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nReport includes:")
    print("  - Interactive latency chart (mean, p50, p95)")
    print("  - GPU memory usage chart")
    print("  - Model perplexity chart")
    print("  - Detailed metrics table with clickable rows")
    print("  - Individual stage detail panels")
