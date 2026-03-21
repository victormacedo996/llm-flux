"""
profiling/inference_benchmarker.py — Inference latency domain service.

BUG FIX from original: compare_models_inference divided the Pydantic object
instead of .avg_time.  Fixed in this version.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedModel

from llm_flux.profiling.types.inference import CompareBenchmark, InferencePerformanceInfo


class InferencePerformanceBenchmarker:
    """
    Measures inference generation latency for a model over multiple runs.

    Usage::

        bench = InferencePerformanceBenchmarker()
        info = bench.time_inference(model, tokenizer, prompt="Hello world")
        print(info.tokens_per_second)
    """

    def time_inference(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> InferencePerformanceInfo:
        """
        Measure inference generation time (wall-clock) for a single prompt.

        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer to use.
            prompt: Input prompt for generation.
            max_new_tokens: Maximum number of tokens to generate.
            num_runs: Number of timed iterations.
            warmup_runs: Discarded warm-up iterations.

        Returns:
            InferencePerformanceInfo with timing statistics and token throughput.
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        # Timed runs
        times: list[float] = []
        last_output = None
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                last_output = model.generate(
                    inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=False
                )
            times.append(time.perf_counter() - start)

        generated_tokens = last_output.size(1) - inputs.input_ids.size(1)
        avg = float(np.mean(times))

        return InferencePerformanceInfo(
            avg_time=avg,
            min_time=float(np.min(times)),
            max_time=float(np.max(times)),
            tokens_per_second=generated_tokens / avg if avg > 0 else 0.0,
            num_runs=num_runs,
            generated_tokens=generated_tokens,
        )

    def compare_models_inference(
        self,
        original: InferencePerformanceInfo,
        compressed: InferencePerformanceInfo,
    ) -> CompareBenchmark:
        """
        Compute speedup and token throughput improvement between two benchmark runs.

        Args:
            original: Baseline (uncompressed) model benchmark.
            compressed: Post-compression model benchmark.
        """
        # BUG FIX: was `original / compressed.avg_time` (divided the object).
        speedup = (
            original.avg_time / compressed.avg_time
            if compressed.avg_time > 0
            else float("inf")
        )
        tps_improvement = (
            (compressed.tokens_per_second / original.tokens_per_second - 1) * 100
            if original.tokens_per_second > 0
            else float("inf")
        )
        return CompareBenchmark(speedup=speedup, tps_improvement_percent=tps_improvement)
