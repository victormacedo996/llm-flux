"""
examples/bnb_compression_study.py

Dissertation Experiment — BitsAndBytes 4-bit quantization + LoRA healing.

This example demonstrates how to use ModelForge to:
1. Load a model from HuggingFace Hub (or a local path).
2. Profile it at baseline.
3. Apply NF4 quantization (BitsAndBytes).
4. Profile again post-compression.
5. Apply LoRA healing on a small dataset.
6. Profile the healed model.

Run:
    uv run python examples/bnb_compression_study.py
"""
from __future__ import annotations

from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.compression.bitsandbytes import BitsAndBytesAdapter, BitsAndBytesConfig
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

# ── 1. Model source ────────────────────────────────────────────────────────────
# Change the identifier to a local path or another HF model as needed.
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"   # small model for quick experimentation

model_handle = HFModelHandle(
    source=ModelSource(identifier=MODEL_ID, trust_remote_code=True)
)

# ── 2. Profiler ────────────────────────────────────────────────────────────────
# We re-use the same profiler across stages.
# Note: run_hardware_profile=True only on baseline (hardware doesn't change).
baseline_profiler = ComprehensiveProfilingAdapter(
    config=ComprehensiveProfilingConfig(
        name="profiler-baseline",
        description="Full hardware + LLM profile + LAMBADA perplexity.",
        benchmark_tests={"lambada"},
        prompt="The theory of relativity was first proposed by",
        max_new_tokens=50,
        num_warmup_runs=2,
        num_benchmark_runs=5,
        run_hardware_profile=True,
    ),
    tokenizer=model_handle,   # HFModelHandle exposes get_tokenizer() after load()
)

fast_profiler = ComprehensiveProfilingAdapter(
    config=ComprehensiveProfilingConfig(
        name="profiler-fast",
        description="Latency + perplexity only (no hardware re-scan).",
        benchmark_tests={"lambada"},
        prompt="The theory of relativity was first proposed by",
        max_new_tokens=50,
        num_warmup_runs=2,
        num_benchmark_runs=5,
        run_hardware_profile=False,  # skip — hardware unchanged
    ),
    tokenizer=model_handle,
)

# ── 3. Compression ─────────────────────────────────────────────────────────────
bnb = BitsAndBytesAdapter(
    config=BitsAndBytesConfig(
        name="NF4-quantization",
        description=(
            "4-bit NF4 quantization via BitsAndBytes. "
            "Reduces model VRAM footprint by ~75% vs FP32."
        ),
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
    model_handle=model_handle,
)

# ── 4. Healing ─────────────────────────────────────────────────────────────────
healer = HFTrainerAdapter(
    config=HealingConfig(
        name="lora-recovery",
        description=(
            "LoRA fine-tuning (r=16) on 1K Alpaca samples to recover "
            "perplexity degraded by NF4 quantization."
        ),
        dataset=DatasetConfig(
            source="tatsu-lab/alpaca",
            max_samples=1000,
            split="train",
        ),
        max_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=True,
        lora=LoRAConfig(r=16, lora_alpha=64),
    )
)

# ── 6. Pipeline ────────────────────────────────────────────────────────────────
pipeline = Pipeline(
    name="bnb-nf4-compression-study",
    description="Profiles baseline Qwen, quantizes to NF4, and runs LoRA healing.",
    steps=[
        PipelineStep(kind=StepKind.LOAD,     label="Load Model",             port=model_handle),
        PipelineStep(kind=StepKind.PROFILE,  label="Baseline Profiling",     port=baseline_profiler),
        PipelineStep(kind=StepKind.COMPRESS, label="Apply NF4 Quantization", port=bnb),
        PipelineStep(kind=StepKind.PROFILE,  label="Post-NF4 Profiling",     port=fast_profiler),
        PipelineStep(kind=StepKind.HEAL,     label="LoRA Recovery",          port=healer),
        PipelineStep(kind=StepKind.PROFILE,  label="Post-Healing Profiling", port=fast_profiler),
    ]
)

# ── 7. Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_pipeline(
        pipeline=pipeline,
        dag_output="bnb_study_dag.png",
        show_dag=True,
    )
