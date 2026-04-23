"""
examples/kd_random_llama.py

Knowledge Distillation validation using a tiny random Llama model.
Uses precompute mode for memory efficiency.

Flow:
  1. Load original model (teacher)
  2. Profile baseline
  3. Compress model (depth pruning)
  4. Profile post-compression
  5. KD healing using original model as teacher
  6. Profile after healing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HAS_GPU = torch.cuda.is_available()
CACHE_DIR = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/hf_cache"

MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DATASET_SOURCE = "tatsu-lab/alpaca"
KD_DATASET = (
    "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/dummy_dataset.jsonl"
)
KD_SAMPLES = 10
COMPRESSION_RATIO = 0.2
CALIBRATION_SAMPLES = 5
LIMIT_TEST_SAMPLES = 5
KD_STEPS = 3
USE_FP16 = HAS_GPU
TEST_LIMIT = 5

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

from loguru import logger

from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter,
    DepthPruningConfig,
    angular_distance_importance,
)
from llm_flux.adapters.healing.knowledge_distillation import (
    KnowledgeDistillationAdapter,
)
from llm_flux.core.model import ModelSource
from llm_flux.core.healing import DistillationConfig, LoRAConfig
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.datasets.port import DatasetConfig
from llm_flux.adapters.profiling import LmEvalAdapter, LmEvalConfig
from llm_flux.runner import run_pipeline

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>")


teacher_handle = HFModelHandle(
    source=ModelSource(
        identifier=MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
)
teacher_handle.load()
logger.info(f"  📦 Teacher model loaded: {teacher_handle.name}")

student_handle = HFModelHandle(
    source=ModelSource(
        identifier=MODEL_ID,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
)

gsm8k_profiler = LmEvalAdapter(
    config=LmEvalConfig(
        name="lm-eval-profiling gsm8k",
        description=f"lm-evaluation-harness profiling on {MODEL_ID} with gsm8k dataset",
        tasks="gsm8k",
        limit=TEST_LIMIT,
        no_cache=True,
        # Enable profiling so every LmEvalAdapter step captures
        # latency, memory and hardware info independently.
        run_hardware_profile=True,
        run_llm_profile=True,
        run_inference_benchmark=True,
    ),
)

lambada_profiler = LmEvalAdapter(
    config=LmEvalConfig(
        name="lm-eval-profiling lambada",
        description=f"lm-evaluation-harness profiling on {MODEL_ID} with lambada dataset",
        tasks="lambada",
        limit=TEST_LIMIT,
        no_cache=True,
        run_hardware_profile=True,
        run_llm_profile=True,
        run_inference_benchmark=True,
    ),
)

depth_pruner = DepthPruningAdapter(
    config=DepthPruningConfig(
        name=f"depth-pruning-{COMPRESSION_RATIO * 100:.0f}pct",
        description=f"Drops {COMPRESSION_RATIO * 100:.0f}% of layers",
        pruning_ratio=COMPRESSION_RATIO,
        calibration_samples=CALIBRATION_SAMPLES,
        calibration_dataset=DatasetConfig(
            source=KD_DATASET,
            subset=None,
            split="train",
            streaming=False,
            max_samples=CALIBRATION_SAMPLES,
        ),
    ),
    tokenizer=student_handle,
    importance_fn=angular_distance_importance,
)

kd_adapter = KnowledgeDistillationAdapter(
    config=DistillationConfig(
        name="kd-healing",
        description=f"KD healing on {MODEL_ID}",
        dataset=DatasetConfig(
            source=KD_DATASET,
            subset=None,
            split="train",
            streaming=False,
            max_samples=KD_SAMPLES,
        ),
        temperature=2.0,
        alpha=0.5,
        precompute_teacher_logits=True,
        teacher_logits_output_dir="./kd_logits",
        max_steps=KD_STEPS,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=USE_FP16,
        output_dir="./kd_output",
        logging_steps=1,
        save_steps=KD_STEPS,
        lora=LoRAConfig(r=8, lora_alpha=32),
    ),
    tokenizer=student_handle,
    teacher_model_handle=teacher_handle,
)

pipeline = Pipeline(
    name="kd-random-llama-validation",
    description=f"KD validation: {MODEL_ID}",
    steps=[
        PipelineStep(label="Load Model (Student)", port=student_handle),
        PipelineStep(label="GSM8K Profile Baseline", port=gsm8k_profiler),
        PipelineStep(label="Lambada Profile Baseline", port=lambada_profiler),
        PipelineStep(label="Depth Pruning", port=depth_pruner),
        PipelineStep(label="GSM8K Profile Post Pruning", port=gsm8k_profiler),
        PipelineStep(label="Lambada Profile Post Pruning", port=lambada_profiler),
        PipelineStep(label="KD Healing", port=kd_adapter),
        PipelineStep(label="GSM8K Profile Post Pruning and KD", port=gsm8k_profiler),
        PipelineStep(label="Lambada Profile Post Pruning and KD", port=lambada_profiler),
    ],
)


if __name__ == "__main__":
    result_dir = Path(__file__).parent.parent / "compression_results" / pipeline.name
    result_dir.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(
        pipeline=pipeline,
        dag_output=str(result_dir / "dag_graph.png"),
        result_output=str(result_dir / "results.json"),
        html_report_output=str(result_dir / "report.html"),
        skip_confirmation=True,
        show_dag=False,
    )
    logger.info(f"\n  KD validation complete! Results saved to {result_dir}")
