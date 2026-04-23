"""
examples/manual_kd_random_llama.py

Knowledge Distillation example using Qwen2-0.6B.

This demonstrates:
1. Load teacher model (uncompressed) - loaded BEFORE pipeline
2. Load student model (same model, will be compressed)
3. Apply compression (BitsAndBytes NF4)
4. Profile post-compression
5. Apply Knowledge Distillation healing using teacher as guide
6. Profile post-KD
7. Generate HTML, CSV, and JSON reports

Run:
    uv run python examples/manual_kd_random_llama.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter,
    DepthPruningConfig,
    angular_distance_importance,
)
from llm_flux.adapters.profiling.lm_eval_adapter import LmEvalAdapter, LmEvalConfig

from llm_flux.adapters.profiling.lm_eval_adapter import LmEvalConfig

HAS_GPU = torch.cuda.is_available()
CACHE_DIR = os.path.join(Path(__file__).parent.parent, "hf_cache")
OUTPUT_DIR = Path(__file__).parent.parent / "kd_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-0.6B"
# MODEL_ID  = "hf-internal-testing/tiny-random-LlamaForCausalLM"
KD_DATASET = "tatsu-lab/alpaca"
KD_SAMPLES = 500
COMPRESSION_SAMPLES = 128
KD_STEPS = 100
USE_FP16 = HAS_GPU
TEST_LIMIT = 10
COMPRESSION_RATIO = 0.2
CALIBRATION_SAMPLES = 5



os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

from loguru import logger

from llm_flux.adapters.healing.knowledge_distillation import KnowledgeDistillationAdapter
from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.core.healing import DistillationConfig, LoRAConfig, SaveFormat
from llm_flux.core.model import ModelSource
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.datasets.port import DatasetConfig
from llm_flux.runner import run_pipeline

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>")


def main():
    # ── 1. Teacher Model (uncompressed) ────────────────────────────────────────
    # The teacher must be loaded BEFORE the pipeline runs.
    # This is the original model that will guide the student's learning.
    logger.info("=" * 60)
    logger.info("  Loading Teacher Model (uncompressed)")
    logger.info("=" * 60)

    teacher_handle = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )
    teacher_handle.load()
    logger.info(f"  📦 Teacher loaded: {teacher_handle.name}")

    # ── 2. Student Model Handle ────────────────────────────────────────────────
    # The student starts as a copy of the teacher model.
    # It will be compressed and then healed via KD.
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
            num_fewshot=2,
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

    # ── 5. Knowledge Distillation Adapter ────────────────────────────────────
    # The teacher is the uncompressed model, student is the compressed model.
    kd_adapter = KnowledgeDistillationAdapter(
        config=DistillationConfig(
            name="kd-healing",
            description=f"KD healing on {MODEL_ID}",
            dataset=DatasetConfig(
                source=KD_DATASET,
                max_samples=KD_SAMPLES,
                split="train",
            ),
            temperature=2.0,
            alpha=0.5,
            precompute_teacher_logits=True,
            teacher_logits_output_dir=str(OUTPUT_DIR / "kd_logits"),
            max_steps=KD_STEPS,
            learning_rate=2e-4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            fp16=USE_FP16,
            output_dir=str(OUTPUT_DIR / "kd_output"),
            logging_steps=10,
            save_steps=KD_STEPS,
            save_format=SaveFormat.AUTO,
        ),
        tokenizer=student_handle,
        teacher_model_handle=teacher_handle,
    )

    # ── 6. Pipeline Definition ───────────────────────────────────────────────
    pipeline = Pipeline(
        name="qwen-0.6b-kd-compression",
        description=f"KD Compression Study: {MODEL_ID}",
        steps=[
            PipelineStep(label="Load Student Model", port=student_handle),
            PipelineStep(label="Baseline Profiling", port=gsm8k_profiler),
            PipelineStep(label="Apply Depth Pruning", port=depth_pruner),
            PipelineStep(label="Post-Compression Profiling", port=gsm8k_profiler),
            PipelineStep(label="KD Healing", port=kd_adapter),
            PipelineStep(label="Post-KD Profiling", port=gsm8k_profiler),
        ],
    )

    # ── 7. Execute Pipeline with Report Generation ───────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Starting KD Compression Pipeline")
    logger.info("=" * 60)

    result = run_pipeline(
        pipeline=pipeline,
        dag_output=str(OUTPUT_DIR / "dag_graph.png"),
        result_output=str(OUTPUT_DIR / "results.json"),
        html_report_output=str(OUTPUT_DIR / "report.html"),
        csv_output=str(OUTPUT_DIR / "results.csv"),
        comparison_csv_output=str(OUTPUT_DIR / "comparison.csv"),
        skip_confirmation=True,
        show_dag=False,
    )

    # ── 8. Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  KD Compression Study Complete!")
    logger.info("=" * 60)
    logger.info(f"  Results saved to: {OUTPUT_DIR}")
    logger.info("  - JSON:   results.json")
    logger.info("  - HTML:   report.html")
    logger.info("  - CSV:    results.csv")
    logger.info("  - DAG:    dag_graph.png")
    logger.info("\n" + result.to_markdown_table())

    # Unload teacher to free memory
    teacher_handle.unload()
    logger.info("  Teacher model unloaded. Pipeline complete.")

    return result


if __name__ == "__main__":
    main()
