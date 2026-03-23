"""
examples/prune_gptq_experiment.py

Experiment: Depth Pruning + LoRA Healing + GPTQ Quantization
Executes a 7-step pipeline across multiple pruning ratios on a 1B LLaMA model.

Ratios tested: 5%, 10%, 15%, 20%
Target model: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from loguru import logger

TINY_MODE = True  # Set to True for lightning-fast end-to-end pipeline validation

import torch
HAS_GPU = torch.cuda.is_available()
CACHE_DIR = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/hf_cache"

if TINY_MODE:
    # Use a tiny-random model that takes < 10MB of RAM and runs in seconds on CPU
    MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    DATASET_SOURCE = "dummy_dataset.jsonl"
    C4_SAMPLES_FOR_HEALING = 2
    LIMIT_TEST_SAMPLES = 2
    NUM_BENCHMARK_RUNS = 1
    HEALING_STEPS = 1
    CALIBRATION_SAMPLES = 2
    USE_FP16 = False # CPU is slow with FP16 emulation
else:
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    DATASET_SOURCE = "allenai/c4"
    C4_SAMPLES_FOR_HEALING = 5000
    LIMIT_TEST_SAMPLES = 50
    NUM_BENCHMARK_RUNS = 3
    HEALING_STEPS = 50
    CALIBRATION_SAMPLES = 32
    USE_FP16 = HAS_GPU # Only use FP16 if we have a GPU


# Set Hugging Face environment variables globally so ALL sub-libraries (datasets, peft, auto-gptq) respect it
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.compression.depth_pruning import DepthPruningAdapter, DepthPruningConfig
from llm_flux.adapters.compression.gptq import GPTQAdapter, GPTQConfig
from llm_flux.adapters.healing.hf_trainer import HFTrainerAdapter
from llm_flux.core.model import ModelSource
from llm_flux.core.healing import HealingConfig, LoRAConfig
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.datasets.port import DatasetConfig
from llm_flux.profiling.adapters.comprehensive import (
    ComprehensiveProfilingAdapter,
    ComprehensiveProfilingConfig,
)
from llm_flux.runner import run_pipeline

# Configure global logger
logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>")

RATIOS = [
    # 0.05, 
    # 0.10, 
    # 0.15, 
    0.20
    ]

def run_experiment_for_ratio(ratio: float):
    logger.info(f"\n{'='*80}\nStarting GPTQ Experiment Pipeline for Pruning Ratio: {ratio*100:.0f}%\n{'='*80}")
    
    # ── 1. Model Source ────────────────────────────────────────────────────────
    model_handle = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
    )

    # ── 2. Profilers ───────────────────────────────────────────────────────────
    # Full baseline scan (includes hardware)
    baseline_profiler = ComprehensiveProfilingAdapter(
        config=ComprehensiveProfilingConfig(
            name="profiler-baseline",
            description="Runs heavy structural and baseline inference tracking.",
            benchmark_tests={"lambada"},
            run_hardware_profile=True,
            limit_test_samples=LIMIT_TEST_SAMPLES,
            num_benchmark_runs=NUM_BENCHMARK_RUNS,
            num_warmup_runs=1,
        ),
        tokenizer=model_handle,
    )

    # Fast scan for intermediate steps (skips hardware)
    fast_profiler = ComprehensiveProfilingAdapter(
        config=ComprehensiveProfilingConfig(
            name="profiler-fast",
            description="Quick check, drops hardware scanning",
            benchmark_tests={"lambada"},
            run_hardware_profile=False,
            limit_test_samples=LIMIT_TEST_SAMPLES,
            num_benchmark_runs=NUM_BENCHMARK_RUNS,
            num_warmup_runs=1,
        ),
        tokenizer=model_handle,
    )

    # ── 3. Depth Pruning Adapter ───────────────────────────────────────────────
    depth_pruner = DepthPruningAdapter(
        config=DepthPruningConfig(
            name=f"depth-pruning-{ratio*100:.0f}pct",
            description=f"Drops {ratio*100:.0f}% of layers using angular distance calculation.",
            pruning_ratio=ratio,
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source=DATASET_SOURCE,
                subset=None if TINY_MODE else "en",
                split="train",
                streaming=not TINY_MODE,
                max_samples=CALIBRATION_SAMPLES,
            ),
        ),
        tokenizer=model_handle,
    )

    # ── 4. Healing Adapter ─────────────────────────────────────────────────────
    healer = HFTrainerAdapter(
        config=HealingConfig(
            name="lora-healing-c4",
            description=f"LoRA fine-tuning on {C4_SAMPLES_FOR_HEALING} C4 samples to recover accuracy.",
            dataset=DatasetConfig(
                source=DATASET_SOURCE,
                subset=None if TINY_MODE else "en",
                split="train",
                streaming=not TINY_MODE,
                max_samples=C4_SAMPLES_FOR_HEALING,
            ),
            max_steps=HEALING_STEPS,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            fp16=USE_FP16, # Ensure efficient dtype for target device
            lora=LoRAConfig(r=8, lora_alpha=32),
        ),
        tokenizer=model_handle
    )

    # ── 5. GPTQ Quantization Adapter ────────────────────────────────────────────
    gptq = GPTQAdapter(
        config=GPTQConfig(
            name="gptq-4bit",
            description="Post-training quantization via auto-gptq (4-bit).",
            bits=4,
            group_size=128,
            desc_act=False,
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source=DATASET_SOURCE,
                subset=None if TINY_MODE else "en",
                split="train",
                streaming=not TINY_MODE,
                max_samples=CALIBRATION_SAMPLES,
            ),
        ),
        tokenizer=model_handle
    )

    pipeline = Pipeline(
        name=f"llama_1b_prune_{ratio*100:.0f}pct_gptq",
        description=(
            f"1B LLaMA Depth Pruning ({ratio*100:.0f}%) -> LoRA Healing (C4) -> GPTQ Pipeline."
        ),
        steps=[
            PipelineStep(label="Load Model",            port=model_handle),
            PipelineStep(label="Baseline Profile",      port=baseline_profiler),
            PipelineStep(label="Depth Prune",           port=depth_pruner),
            PipelineStep(label="Post-Prune Profile",    port=fast_profiler),
            PipelineStep(label="LoRA Heal",             port=healer),
            PipelineStep(label="Post-Heal Profile",     port=fast_profiler),
            PipelineStep(label="GPTQ Quantize",         port=gptq),
            PipelineStep(label="Final GPTQ Profile",    port=fast_profiler),
        ]
    )

    # ── 7. Execute ─────────────────────────────────────────────────────────────
    result = run_pipeline(
        pipeline=pipeline,
        dag_output=f"dag_prune_{ratio*100:.0f}pct_gptq.png",
        result_output=f"result_prune_{ratio*100:.0f}pct_gptq.json",
        skip_confirmation=True,  # Automated experiment script
        show_dag=False,          # Don't block on diagram window
    )
    return result

if __name__ == "__main__":
    logger.info("Initializing multi-ratio depth pruning GPTQ experiment...")
    for r in RATIOS:
        try:
            run_experiment_for_ratio(r)
        except Exception as e:
            logger.error(f"Experiment for ratio {r} failed: {e}")
            raise e
