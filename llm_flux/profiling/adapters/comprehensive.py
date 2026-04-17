"""
profiling/adapters/comprehensive.py — ComprehensiveProfilingAdapter.

Bridges HardwareProfiler + LLMProfiler + InferencePerformanceBenchmarker +
ModelPerformanceBenchmarker to the ProfilingPort interface expected by the
DAG executor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import Field

from llm_flux.core.model import ModelHandle
from llm_flux.core.profiling import (
    AccuracyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    ProfilingConfig,
    ProfilingPort,
    ProfilingResult,
)
from llm_flux.profiling.hardware_profiler import HardwareProfiler
from llm_flux.profiling.inference_benchmarker import InferencePerformanceBenchmarker
from llm_flux.profiling.llm_profiler import LLMProfiler
from llm_flux.profiling.model_benchmarker import ModelPerformanceBenchmarker
from llm_flux.profiling.types.benchmark import AccuracyTestResult, PerplexityTestResult
from llm_flux.profiling.types.llm import EstimateMemory


class ComprehensiveProfilingConfig(ProfilingConfig):
    """
    Configuration for the all-in-one profiling adapter.

    Feature flags allow disabling expensive steps for quick passes.
    For example, disable ``run_hardware_profile`` after the first stage
    since the hardware does not change between compression steps.
    """

    prompt: str = "The quick brown fox jumps over the lazy dog."
    max_new_tokens: int = 100
    benchmark_tests: set[str] = Field(default_factory=lambda: {"lambada"})
    limit_test_samples: int | None = None

    # Feature flags
    run_hardware_profile: bool = True
    run_llm_profile: bool = True
    run_inference_benchmark: bool = True
    run_model_benchmark: bool = True
    estimate_memory: bool = True


class ComprehensiveProfilingAdapter(ProfilingPort):
    """
    Runs all available profiling tools in sequence and maps results into
    a single ``ProfilingResult``.

    Full raw data is stored in ``ProfilingResult.extra`` under the keys:
    ``"hardware"``, ``"llm_profile"``, ``"inference"``, ``"benchmarks"``.
    This ensures zero data loss in the dissertation JSON export while the
    structured top-level fields remain clean for Markdown table generation.

    Usage::

        profiler = ComprehensiveProfilingAdapter(
            config=ComprehensiveProfilingConfig(name="full-profiler"),
            tokenizer=tokenizer,
        )
        results = profiler.profile(stage_label="Baseline", model_handle=handle)
    """

    def __init__(
        self,
        config: ComprehensiveProfilingConfig,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def profile(
        self,
        stage_label: str,
        model_handle: ModelHandle | None = None,
    ) -> list[ProfilingResult]:
        if model_handle is None:
            raise ValueError("ComprehensiveProfilingAdapter requires a model_handle.")

        model = model_handle.get_model_instance()
        extra: dict[str, Any] = {}

        # Resolve tokenizer: prefer model_handle, fall back to self.tokenizer
        if hasattr(model_handle, "get_tokenizer"):
            actual_tokenizer = model_handle.get_tokenizer()
        elif self.tokenizer is not None:
            actual_tokenizer = (
                self.tokenizer.get_tokenizer()
                if hasattr(self.tokenizer, "get_tokenizer")
                else self.tokenizer
            )
        else:
            raise ValueError("ComprehensiveProfilingAdapter could not resolve a tokenizer.")

        # ── Hardware profile ──────────────────────────────────────────────────
        if self.config.run_hardware_profile:
            logger.info("  📡 Hardware profiling...")
            hw = HardwareProfiler().retrieve_hardware_information()
            extra["hardware"] = hw.model_dump()

        # ── LLM static profile ────────────────────────────────────────────────
        if self.config.run_llm_profile:
            logger.info("  🔬 LLM structural profiling...")
            llm_p = LLMProfiler(model=model, tokenizer=actual_tokenizer, verbose=False)
            llm_info = llm_p.profile_complete(
                estimate_memory=EstimateMemory() if self.config.estimate_memory else None
            )
            extra["llm_profile"] = llm_info.model_dump()

        # ── Inference benchmark ───────────────────────────────────────────────
        latency = LatencyMetrics(
            mean_ms=0.0,
            std_ms=0.0,
            min_ms=0.0,
            p5_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            max_ms=0.0,
        )
        memory = MemoryMetrics()

        if self.config.run_inference_benchmark:
            logger.info("  ⏱️  Inference benchmarking...")
            import torch

            bench = InferencePerformanceBenchmarker()
            inf = bench.time_inference(
                model=model,
                tokenizer=actual_tokenizer,
                prompt=self.config.prompt,
                max_new_tokens=self.config.max_new_tokens,
                num_runs=self.config.num_benchmark_runs,
                warmup_runs=self.config.num_warmup_runs,
            )
            extra["inference"] = inf.model_dump()

            avg_ms = inf.avg_time * 1000
            latency = LatencyMetrics(
                mean_ms=avg_ms,
                std_ms=inf.std_time * 1000,
                min_ms=inf.min_time * 1000,
                p5_ms=inf.p5_time * 1000,
                p50_ms=inf.p50_time * 1000,
                p95_ms=inf.p95_time * 1000,
                p99_ms=inf.p99_time * 1000,
                max_ms=inf.max_time * 1000,
            )
            if torch.cuda.is_available():
                memory = MemoryMetrics(peak_gpu_mb=torch.cuda.max_memory_allocated() / (1024**2))
                torch.cuda.reset_peak_memory_stats()

        # ── Model accuracy / perplexity ───────────────────────────────────────
        accuracy: AccuracyMetrics | None = None

        if self.config.run_model_benchmark and self.config.benchmark_tests:
            logger.info("  📊 Model benchmarking...")
            benchmarker = ModelPerformanceBenchmarker()
            results = benchmarker.benchmark(
                model=model,
                tokenizer=actual_tokenizer,
                tests=self.config.benchmark_tests,
                num_examples=self.config.limit_test_samples,
            )
            extra["benchmarks"] = [r.model_dump() for r in results]

            ppl_results = [r for r in results if isinstance(r, PerplexityTestResult)]
            if ppl_results:
                accuracy = AccuracyMetrics(
                    perplexity=ppl_results[0].result.mean_perplexity,
                    task_name=ppl_results[0].test_name,
                )
            else:
                score_results = [r for r in results if isinstance(r, AccuracyTestResult)]
                if score_results:
                    top = score_results[0]
                    accuracy = AccuracyMetrics(
                        task_score=top.accuracy,
                        task_name=f"{top.test_name}:{top.metric_name}",
                    )

        return [
            ProfilingResult(
                profiler_name=self.config.name,
                model_label=type(model).__name__,
                pipeline_stage=stage_label,
                timestamp=datetime.now(),
                memory=memory,
                latency=latency,
                accuracy=accuracy,
                extra=extra,
            )
        ]
