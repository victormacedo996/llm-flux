from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from loguru import logger

from llm_flux.core.model import ModelHandle
from llm_flux.core.profiling import (
    AccuracyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    ProfilingConfig,
    ProfilingPort,
    ProfilingResult,
)
from llm_flux.core.task_metrics import TaskMetrics


class LmEvalConfig(ProfilingConfig):
    tasks: str = "hellaswag"
    num_fewshot: int = 0
    limit: int | float | None = None
    model_args: str = ""
    no_cache: bool = True
    device: str | None = None
    batch_size: int | str | None = None
    bootstrap_iters: int = 100_000
    verbosity: str = "WARNING"

    run_hardware_profile: bool = True
    run_inference_benchmark: bool = True
    run_llm_profile: bool = True
    estimate_memory: bool = False
    prompt: str = "The quick brown fox jumps over the lazy dog."
    max_new_tokens: int = 100

    gen_kwargs: dict[str, Any] | None = None


class LmEvalAdapter(ProfilingPort):
    def __init__(
        self,
        config: LmEvalConfig,
    ) -> None:
        self.config = config

    def profile(
        self,
        stage_label: str,
        model_handle: ModelHandle | None = None,
    ) -> list[ProfilingResult]:
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
        except ImportError as e:
            raise ImportError(
                "lm-eval is not installed. Install with: pip install 'lm_eval[hf]'"
            ) from e

        if model_handle is None:
            raise ValueError("LmEvalAdapter requires a model_handle.")

        model = model_handle.get_model_instance()
        tokenizer = model_handle.get_tokenizer()

        gen_kwargs = self._resolve_gen_kwargs(model)

        extra: dict[str, Any] = {}

        # ── Hardware profile ─────────────────────────────────────────────────
        if self.config.run_hardware_profile:
            from llm_flux.profiling.hardware_profiler import HardwareProfiler

            logger.info("  📡 Hardware profiling...")
            hw = HardwareProfiler().retrieve_hardware_information()
            extra["hardware"] = hw.model_dump()

        # ── LLM static profile ────────────────────────────────────────────────
        if self.config.run_llm_profile:
            from llm_flux.profiling.llm_profiler import LLMProfiler
            from llm_flux.profiling.types.llm import EstimateMemory

            logger.info("  🔬 LLM structural profiling...")
            llm_p = LLMProfiler(model=model, tokenizer=tokenizer, verbose=False)
            estimate_mem = EstimateMemory() if self.config.estimate_memory else None
            llm_info = llm_p.profile_complete(estimate_memory=estimate_mem)
            extra["llm_profile"] = llm_info.model_dump()

        memory = MemoryMetrics()
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

        if self.config.run_inference_benchmark:
            import torch

            from llm_flux.profiling.inference_benchmarker import (
                InferencePerformanceBenchmarker,
            )

            logger.info("  ⏱️  Inference benchmarking...")
            bench = InferencePerformanceBenchmarker()
            inf = bench.time_inference(
                model=model,
                tokenizer=tokenizer,
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
                memory = MemoryMetrics(
                    peak_gpu_mb=torch.cuda.max_memory_allocated() / (1024**2),
                )
                torch.cuda.reset_peak_memory_stats()

        # ── lm-eval ────────────────────────────────────────────────────────────
        hf_model = HFLM(pretrained=model, tokenizer=tokenizer)

        logger.info(
            f"  Running lm-eval | tasks={self.config.tasks} "
            f"| num_fewshot={self.config.num_fewshot} | limit={self.config.limit}"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Failed to get model SHA")
            raw = lm_eval.simple_evaluate(
                model=hf_model,
                tasks=[self.config.tasks],
                num_fewshot=self.config.num_fewshot,
                batch_size=self.config.batch_size,
                device=self.config.device,
                limit=self.config.limit,
                bootstrap_iters=self.config.bootstrap_iters,
                verbosity=self.config.verbosity,
                gen_kwargs=gen_kwargs,
            )

        return self._normalize_results(raw, stage_label, model_handle, extra, latency, memory)

    # ── private helpers ────────────────────────────────────────────────────────

    def _resolve_gen_kwargs(self, model: Any) -> dict[str, Any]:
        """
        Resolve generation kwargs for lm-eval, automatically adapting the max
        tokens to generate to the model's context window size.

        lm-eval task YAMLs (e.g. ``mmlu_pro``) often set ``max_gen_toks: 2048``
        which — on models with ``max_position_embeddings == 2048`` — leaves no
        room for the prompt and triggers:

            AssertionError: Invalid configuration: requested max tokens to
            generate (2048) must be less than model's maximum sequence
            length (2048).

        We override with a safe value based on the model's actual context
        size minus a reserved prompt window.

        IMPORTANT: lm-eval's ``normalize_gen_kwargs`` prioritises
        ``max_gen_toks`` over ``max_new_tokens``. Since task YAMLs use
        ``max_gen_toks``, we must set the same key to guarantee the override.
        """
        PROMPT_RESERVED_SPACE = 512

        max_seq_len = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
            or getattr(model.config, "n_ctx", None)
            or 2048
        )

        safe_max_gen_toks = max(1, max_seq_len - PROMPT_RESERVED_SPACE)

        gen_kwargs: dict[str, Any] = dict(self.config.gen_kwargs) if self.config.gen_kwargs else {}

        # Normalise any user-provided token-limit alias to max_gen_toks so it
        # wins against the task YAML's max_gen_toks default.
        user_limit = (
            gen_kwargs.pop("max_gen_toks", None)
            or gen_kwargs.pop("max_new_tokens", None)
            or gen_kwargs.pop("max_tokens", None)
            or gen_kwargs.pop("max_completion_tokens", None)
        )

        chosen = int(user_limit) if user_limit is not None else safe_max_gen_toks
        # Clamp to safe bound to avoid the max_ctx_len <= 0 assertion in lm-eval.
        if chosen >= max_seq_len:
            logger.warning(
                f"  Requested max_gen_toks={chosen} >= model max_context={max_seq_len}; "
                f"clamping to {safe_max_gen_toks}."
            )
            chosen = safe_max_gen_toks

        gen_kwargs["max_gen_toks"] = chosen
        logger.info(
            f"  Using max_gen_toks={chosen} "
            f"(model max_context={max_seq_len}, reserved={PROMPT_RESERVED_SPACE})"
        )

        return gen_kwargs

    def _normalize_results(
        self,
        lm_results: dict,
        stage_label: str,
        model_handle: ModelHandle | None = None,
        extra_base: dict[str, Any] | None = None,
        latency: LatencyMetrics | None = None,
        memory: MemoryMetrics | None = None,
    ) -> list[ProfilingResult]:
        records: list[ProfilingResult] = []
        results_dict = lm_results.get("results", {})
        cfg = lm_results.get("config", {})
        model_label = cfg.get("model") or (model_handle.name if model_handle else "unknown")

        extra: dict[str, Any] = dict(extra_base) if extra_base else {}

        for task_name, metrics in results_dict.items():
            if not isinstance(metrics, dict):
                continue

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

                raw_base = k.split(",")[0]

                if raw_base.endswith("_stderr"):
                    stem = raw_base[: -len("_stderr")]
                    stderr_map[stem] = v
                elif v is not None:
                    primary[raw_base] = float(v)

            for metric_name, metric_value in primary.items():
                stderr = stderr_map.get(metric_name)

                task_metric = TaskMetrics(
                    task_name=task_name,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    stderr=stderr,
                    num_fewshot=lm_results.get("n-shot", {}).get(task_name, 0),
                )

                accuracy = AccuracyMetrics(
                    task_score=metric_value,
                    task_name=f"{task_name}/{metric_name}",
                    task_metrics=[task_metric],
                )

                lm_extra = {
                    "results": lm_results.get("results", {}),
                    "versions": lm_results.get("versions", {}),
                    "n-shot": lm_results.get("n-shot", {}),
                    "higher_is_better": lm_results.get("higher_is_better", {}),
                    "n-samples": lm_results.get("n-samples", {}),
                    "config": {
                        "model": cfg.get("model"),
                        "model_num_parameters": cfg.get("model_num_parameters"),
                        "lm_eval_version": cfg.get("lm_eval_version"),
                        "transformers_version": cfg.get("transformers_version"),
                        "batch_size": cfg.get("batch_size"),
                        "device": cfg.get("device"),
                        "limit": cfg.get("limit"),
                        "bootstrap_iters": cfg.get("bootstrap_iters"),
                        "num_fewshot": self.config.num_fewshot,
                    },
                }
                record_extra: dict[str, Any] = dict(extra)
                record_extra["lm_eval"] = lm_extra

                records.append(
                    ProfilingResult(
                        profiler_name=self.config.name,
                        model_label=model_label,
                        pipeline_stage=stage_label,
                        timestamp=datetime.now(),
                        memory=memory or MemoryMetrics(),
                        latency=latency
                        or LatencyMetrics(
                            mean_ms=0.0,
                            std_ms=0.0,
                            min_ms=0.0,
                            p5_ms=0.0,
                            p50_ms=0.0,
                            p95_ms=0.0,
                            p99_ms=0.0,
                            max_ms=0.0,
                        ),
                        accuracy=accuracy,
                        extra=record_extra,
                    )
                )

        if not records:
            logger.warning(
                f"  [LmEvalAdapter] No metrics parsed from lm-eval results "
                f"for tasks {self.config.tasks}. Check task names."
            )

        return records
