"""
examples/prune_awq_experiment.py

Experiment: Depth Pruning + LoRA Healing + AWQ Quantization
Executes a 7-step pipeline across multiple pruning ratios on a 1B LLaMA model.

Ratios tested: 5%, 10%, 15%, 20%
Target model: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from loguru import logger

# Target a 1B model (as requested by user)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
CACHE_DIR = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/hf_cache"  # Used to store downloaded HF models
TINY_MODE = True  # Set to True for lightning-fast end-to-end pipeline validation

if TINY_MODE:
    C4_SAMPLES_FOR_HEALING = 5
    LIMIT_TEST_SAMPLES = 5
    NUM_BENCHMARK_RUNS = 1
    HEALING_STEPS = 1
    CALIBRATION_SAMPLES = 4
else:
    C4_SAMPLES_FOR_HEALING = 5000
    LIMIT_TEST_SAMPLES = 50
    NUM_BENCHMARK_RUNS = 3
    HEALING_STEPS = 50
    CALIBRATION_SAMPLES = 32


# Set Hugging Face environment variables globally so ALL sub-libraries (datasets, peft, AutoAWQ) respect it
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.compression.depth_pruning import DepthPruningAdapter, DepthPruningConfig
# from modelforge.adapters.compression.mlp_pruning import MLPPrunin`gAdapter
from llm_flux.adapters.compression.awq import AWQAdapter, AWQConfig
from llm_flux.adapters.healing.hf_trainer import HFTrainerAdapter
from llm_flux.core.model import ModelSource
from llm_flux.core.healing import HealingConfig, LoRAConfig
from llm_flux.core.pipeline import Pipeline, PipelineStep, StepKind
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
    logger.info(f"\n{'='*80}\nStarting Experiment Pipeline for Pruning Ratio: {ratio*100:.0f}%\n{'='*80}")
    
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
                source="allenai/c4",
                subset="en",
                split="train",
                streaming=True,
                max_samples=CALIBRATION_SAMPLES * 2,
            ),
        ),
        tokenizer=model_handle,
    )

    # # ── 4. MLP Pruning Adapter (with 64/128 alignment) ─────────────────────────
    # mlp_pruner = MLPPruningAdapter(
    #     config=MLPPruningConfig(
    #         name=f"mlp-pruning-{ratio*100:.0f}pct",
    #         description=f"Prunes {ratio*100:.0f}% of MLP neurons aligned to 128-dimension blocks.",
    #         pruning_ratio=ratio,
    #         alignment=128,
    #         importance_strategy="max_abs_weight"
    #     )
    # )

    # ── 5. Healing Adapter ─────────────────────────────────────────────────────
    healer = HFTrainerAdapter(
        config=HealingConfig(
            name="lora-healing-c4",
            description=f"LoRA fine-tuning on {C4_SAMPLES_FOR_HEALING} C4 samples to recover accuracy.",
            dataset=DatasetConfig(
                source="allenai/c4",
                subset="en",
                split="train",
                streaming=True,
                max_samples=C4_SAMPLES_FOR_HEALING * 2,
            ),
            max_steps=HEALING_STEPS,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            fp16=True, # Ensure efficient dtype for GPU
            lora=LoRAConfig(r=8, lora_alpha=32),
        ),
        tokenizer=model_handle
    )

    # ── 5. AWQ Quantization Adapter ────────────────────────────────────────────
    # User requested AWQ quantization after healing
    awq = AWQAdapter(
        config=AWQConfig(
            name="awq-4bit",
            description="Activation-aware weight quantization (4-bit GEMM).",
            bits=4,
            group_size=128,
            zero_point=True,
            version="GEMM",
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source="allenai/c4",
                subset="en",
                split="train",
                streaming=True,
                max_samples=CALIBRATION_SAMPLES * 2,
            ),
        )
    )

    pipeline = Pipeline(
        name=f"llama_1b_prune_{ratio*100:.0f}pct_awq",
        description=(
            f"1B LLaMA Depth+MLP Pruning ({ratio*100:.0f}%) -> LoRA Healing (C4) -> AWQ Pipeline."
        ),
        steps=[
            PipelineStep(kind=StepKind.LOAD,     label="Load Model",            port=model_handle),
            PipelineStep(kind=StepKind.PROFILE,  label="Baseline Profile",      port=baseline_profiler),
            PipelineStep(kind=StepKind.COMPRESS, label="Depth Prune",           port=depth_pruner),
            # PipelineStep(kind=StepKind.COMPRESS, label="MLP Prune",             port=mlp_pruner),
            PipelineStep(kind=StepKind.PROFILE,  label="Post-Prune Profile",    port=fast_profiler),
            PipelineStep(kind=StepKind.HEAL,     label="LoRA Heal",             port=healer),
            PipelineStep(kind=StepKind.PROFILE,  label="Post-Heal Profile",     port=fast_profiler),
            PipelineStep(kind=StepKind.COMPRESS, label="AWQ Quantize",          port=awq),
            PipelineStep(kind=StepKind.PROFILE,  label="Final AWQ Profile",     port=fast_profiler),
        ]
    )

    # ── 7. Execute ─────────────────────────────────────────────────────────────
    result = run_pipeline(
        pipeline=pipeline,
        dag_output=f"dag_prune_{ratio*100:.0f}pct.png",
        result_output=f"result_prune_{ratio*100:.0f}pct.json",
        skip_confirmation=True,  # Automated experiment script
        show_dag=False,          # Don't block on diagram window
    )
    return result

if __name__ == "__main__":
    logger.info("Initializing multi-ratio depth pruning experiment...")
    for r in RATIOS:
        try:
            run_experiment_for_ratio(r)
        except Exception as e:
            logger.error(f"Experiment for ratio {r} failed: {e}")
            raise e
