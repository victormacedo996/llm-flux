"""
examples/lm_eval_study.py

Experiment: Depth Pruning + LoRA Healing + AWQ Quantization with lm-eval profiling.

Profiles a 1B LLaMA model using lm-evaluation-harness before and after
each compression stage, producing one ProfilingResult per (task × metric)
so every benchmark gets its own row in the results table.

Tasks (lm-eval):
    mmlu_pro, hellaswag, gsm8k, math_500, arc_challenge,
    lambada_openai, truthfulqa_mc2

Pipeline:
    Load → Baseline lm-eval → Depth Prune → Post-Prune lm-eval →
    LoRA Heal → Post-Heal lm-eval → AWQ Quantize → Final lm-eval

Run:
    # Install lm-eval first (optional but recommended):
    pip install 'lm_eval[hf]'

    # Run the example:
    uv run python examples/lm_eval_study.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

TINY_MODE = True  # Set to True for lightning-fast validation on CPU

import torch

HAS_GPU = torch.cuda.is_available()

CACHE_DIR = str(Path(__file__).parent.parent / "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

if TINY_MODE:
    MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    # MODEL_ID = "Qwen/Qwen3-0.6B"
    CALIBRATION_DATASET = "dummy_dataset.jsonl"
    HEALING_DATASET = "dummy_dataset.jsonl"
    CALIBRATION_SAMPLES = 2
    HEALING_SAMPLES = 2
    LIMIT_TEST_SAMPLES = 2
    NUM_BENCHMARK_RUNS = 1
    HEALING_STEPS = 1
    USE_FP16 = False
    AWQ_CALIBRATION_SAMPLES = 2
    TEST_LIMIT = 10
else:
    MODEL_ID = "Qwen/Qwen3-0.6B"
    CALIBRATION_DATASET = "allenai/c4"
    HEALING_DATASET = "tatsu-lab/alpaca"
    CALIBRATION_SAMPLES = 32
    HEALING_SAMPLES = 500
    LIMIT_TEST_SAMPLES = 3
    NUM_BENCHMARK_RUNS = 3
    HEALING_STEPS = 100
    USE_FP16 = HAS_GPU
    AWQ_CALIBRATION_SAMPLES = 32
    TEST_LIMIT = 100

# ───────────────────────────────────────────────────────────────────────────────
from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter,
    DepthPruningConfig,
    angular_distance_importance,
)
from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.profiling import LmEvalAdapter, LmEvalConfig
from llm_flux.core.model import ModelSource
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.datasets.port import DatasetConfig
from llm_flux.runner import run_pipeline

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>")


def run_experiment_for_ratio(ratio: float):
    logger.info(
        f"\n{'=' * 80}\n"
        f"Starting lm-eval Experiment | Ratio: {ratio * 100:.0f}% | Model: {MODEL_ID}\n"
        f"{'=' * 80}"
    )

    # ── 1. Model handle ─────────────────────────────────────────────────────────
    model_handle = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )

    # ── 2. Compression adapters ────────────────────────────────────────────────
    depth_pruner = DepthPruningAdapter(
        config=DepthPruningConfig(
            name=f"depth-pruning-{int(ratio * 100)}pct",
            description=f"Drops {ratio * 100:.0f}% of layers via angular distance.",
            pruning_ratio=ratio,
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source=CALIBRATION_DATASET,
                subset=None if TINY_MODE else "en",
                split="train",
                streaming=not TINY_MODE,
                max_samples=CALIBRATION_SAMPLES,
            ),
        ),
        tokenizer=model_handle,
        importance_fn=angular_distance_importance,
    )

    # awq = AWQAdapter(
    #     config=AWQConfig(
    #         name="awq-4bit",
    #         description="Activation-aware weight quantization (4-bit GEMM).",
    #         bits=4,
    #         group_size=128,
    #         zero_point=True,
    #         version="GEMM",
    #         calibration_samples=AWQ_CALIBRATION_SAMPLES,
    #         calibration_dataset=DatasetConfig(
    #             source=CALIBRATION_DATASET,
    #             subset=None if TINY_MODE else "en",
    #             split="train",
    #             streaming=not TINY_MODE,
    #             max_samples=AWQ_CALIBRATION_SAMPLES,
    #         ),
    #     ),
    # )

    # ── 3. Healing adapter ───────────────────────────────────────────────────────
    # healer = HFTrainerAdapter(
    #     config=HealingConfig(
    #         name="lora-recovery",
    #         description=f"LoRA fine-tuning (r=8) on {HEALING_SAMPLES} samples.",
    #         dataset=DatasetConfig(
    #             source=HEALING_DATASET,
    #             max_samples=HEALING_SAMPLES,
    #             split="train",
    #             streaming=not TINY_MODE,
    #         ),
    #         trainer_class=None,
    #         max_steps=HEALING_STEPS,
    #         learning_rate=2e-4,
    #         per_device_train_batch_size=2,
    #         gradient_accumulation_steps=8,
    #         fp16=USE_FP16,
    #         lora=LoRAConfig(r=8, lora_alpha=32),
    #     ),
    #     tokenizer=model_handle,
    # )

    # ── 4. Build pipeline ───────────────────────────────────────────────────────
    # Build one LmEvalAdapter per task programmatically so the same config
    # (limit, num_fewshot, etc.) is applied consistently to every task and
    # the set of evaluated tasks is trivially extensible via the TASKS list.
    TASKS = [
        "gsm8k",
        "lambada",
    ]

    def build_profiler(task: str) -> LmEvalAdapter:
        return LmEvalAdapter(
            config=LmEvalConfig(
                name=task,
                description=f"lm-evaluation-harness {task}",
                tasks=task,
                limit=TEST_LIMIT,
                no_cache=True,
                num_fewshot=5,
                # Enable profiling so every LmEvalAdapter step captures
                # latency, memory and hardware info independently.
                run_hardware_profile=True,
                run_llm_profile=True,
                run_inference_benchmark=True,
            ),
        )

    task_profilers = {task: build_profiler(task) for task in TASKS}

    def eval_steps(label_prefix: str) -> list[PipelineStep]:
        return [
            PipelineStep(
                label=f"{label_prefix} lm-eval {task}",
                port=profiler,
            )
            for task, profiler in task_profilers.items()
        ]

    pipeline = Pipeline(
        name=f"lm_eval_prune_{int(ratio * 100)}pct",
        description=(
            f"lm-eval profiling across compression pipeline. "
            f"Model={MODEL_ID} | Pruning ratio={ratio * 100:.0f}% | "
            f"Tasks={', '.join(TASKS)}"
        ),
        steps=[
            PipelineStep(label="Load Model", port=model_handle),
            *eval_steps("Baseline"),
            PipelineStep(label="Depth Prune", port=depth_pruner),
            *eval_steps("Post-Prune"),
        ],
    )

    # ── 5. Execute ─────────────────────────────────────────────────────────────
    result_dir = Path(__file__).parent.parent / "compression_results" / pipeline.name
    result = run_pipeline(
        pipeline=pipeline,
        dag_output=str(result_dir / "dag_graph.png"),
        result_output=str(result_dir / "results.json"),
        html_report_output=str(result_dir / "report.html"),
        csv_output=str(result_dir / "results.csv"),
        skip_confirmation=True,
        show_dag=False,
    )

    return result


if __name__ == "__main__":
    RATIOS = [0.10]

    logger.info(f"TINY_MODE : {TINY_MODE}")
    logger.info(f"Model     : {MODEL_ID}")
    logger.info(f"Limit     : {LIMIT_TEST_SAMPLES} samples/task")
    logger.info(f"GPU available: {HAS_GPU}")

    for ratio in RATIOS:
        try:
            run_experiment_for_ratio(ratio)
        except Exception as e:
            raise e
