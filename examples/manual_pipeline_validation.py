"""
examples/manual_pipeline_validation.py

Manual pipeline example that instantiates classes directly without using run_pipeline().
Demonstrates full workflow: load -> LMEval baseline -> depth prune -> profile ->
knowledge distillation recovery -> final profile -> generate reports.

Run:
    uv run python examples/manual_pipeline_validation.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter,
    DepthPruningConfig,
    angular_distance_importance,
)
from llm_flux.adapters.healing.knowledge_distillation import KnowledgeDistillationAdapter
from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.profiling import LmEvalAdapter, LmEvalConfig
from llm_flux.core.healing import DistillationConfig, LoRAConfig
from llm_flux.core.model import ModelSource
from llm_flux.core.results import PipelineRunResult
from llm_flux.datasets.port import DatasetConfig

HAS_GPU = torch.cuda.is_available()

CACHE_DIR = str(Path(__file__).parent.parent / "hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

MODEL_ID = "Qwen/Qwen3-0.6B"
CALIBRATION_DATASET = "tatsu-lab/alpaca"
KD_DATASET = "tatsu-lab/alpaca"
CALIBRATION_SAMPLES = 32
KD_SAMPLES = 100
KD_STEPS = 100
TEST_LIMIT = 10
PRUNING_RATIO = 0.10
USE_FP16 = HAS_GPU

logger.remove()
logger.add(sys.stdout, format="<level>{message}</level>")


def main() -> PipelineRunResult:
    started_at = datetime.now()
    logger.info("=" * 80)
    logger.info("Manual Pipeline Validation")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"GPU available: {HAS_GPU}")
    logger.info("=" * 80)

    profiling_records: list = []

    # ── Step 1: Load uncompressed model (teacher) ─────────────────────────────
    logger.info("\n[Step 1] Loading uncompressed model...")
    teacher_handle = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )
    teacher_handle.load()
    logger.info(f"  Teacher model loaded: {teacher_handle.name}")

    tokenizer = teacher_handle.get_tokenizer()
    uncompressed_model = teacher_handle.get_model_instance()

    # ── Step 2: Baseline LMEval profiling (uncompressed) ───────────────────────
    logger.info("\n[Step 2] Baseline LMEval profiling (uncompressed model)...")

    def make_lm_eval_adapter(task: str) -> LmEvalAdapter:
        return LmEvalAdapter(
            config=LmEvalConfig(
                name=f"lm-eval-{task}",
                description=f"lm-evaluation-harness {task}",
                tasks=task,
                limit=TEST_LIMIT,
                no_cache=True,
                run_hardware_profile=True,
                run_llm_profile=True,
                run_inference_benchmark=True,
            ),
        )

    gsm8k_profiler = make_lm_eval_adapter("gsm8k")
    lambada_profiler = make_lm_eval_adapter("lambada")

    baseline_gsm8k_results = gsm8k_profiler.profile("baseline-gsm8k", model_handle=teacher_handle)
    profiling_records.append(baseline_gsm8k_results)
    logger.info(f"  Baseline GSM8K: {len(baseline_gsm8k_results)} result(s)")

    baseline_lambada_results = lambada_profiler.profile(
        "baseline-lambada", model_handle=teacher_handle
    )
    profiling_records.append(baseline_lambada_results)
    logger.info(f"  Baseline Lambada: {len(baseline_lambada_results)} result(s)")

    # ── Step 3: Depth Pruning (10%) ─────────────────────────────────────────────
    logger.info(f"\n[Step 3] Applying depth pruning ({PRUNING_RATIO * 100:.0f}%)...")
    depth_pruner = DepthPruningAdapter(
        config=DepthPruningConfig(
            name=f"depth-pruning-{int(PRUNING_RATIO * 100)}pct",
            description=f"Drops {PRUNING_RATIO * 100:.0f}% of layers via angular distance",
            pruning_ratio=PRUNING_RATIO,
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source=CALIBRATION_DATASET,
                split="train",
                max_samples=CALIBRATION_SAMPLES,
            ),
        ),
        tokenizer=tokenizer,
        importance_fn=angular_distance_importance,
    )

    compressed_model = depth_pruner.compress(uncompressed_model)
    logger.info("  Depth pruning complete.")

    # ── Step 4: Save compressed model to disk ───────────────────────────────────
    compressed_path = Path("./compressed_model")
    logger.info(f"\n[Step 4] Saving compressed model to {compressed_path}...")
    compressed_model.save_pretrained(compressed_path)
    tokenizer.save_pretrained(compressed_path)
    logger.info("  Compressed model saved.")

    # ── Step 5: Load compressed model (new handle) ─────────────────────────────
    logger.info("\n[Step 5] Loading compressed model from disk...")
    compressed_handle = HFModelHandle(
        source=ModelSource(
            identifier=str(compressed_path),
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )
    compressed_handle.load()
    logger.info(f"  Compressed model loaded: {compressed_handle.name}")

    # ── Step 6: Post-pruning LMEval profiling ───────────────────────────────────
    logger.info("\n[Step 6] Post-pruning LMEval profiling...")

    post_prune_gsm8k_results = gsm8k_profiler.profile(
        "post-prune-gsm8k", model_handle=compressed_handle
    )
    profiling_records.append(post_prune_gsm8k_results)
    logger.info(f"  Post-prune GSM8K: {len(post_prune_gsm8k_results)} result(s)")

    post_prune_lambada_results = lambada_profiler.profile(
        "post-prune-lambada", model_handle=compressed_handle
    )
    profiling_records.append(post_prune_lambada_results)
    logger.info(f"  Post-prune Lambada: {len(post_prune_lambada_results)} result(s)")

    # ── Step 7: Knowledge Distillation Recovery ─────────────────────────────────
    logger.info("\n[Step 7] Knowledge distillation recovery...")

    # Teacher handle needs to be re-loaded since it was unloaded during precompute
    teacher_handle_for_kd = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )
    teacher_handle_for_kd.load()
    logger.info("  Teacher model re-loaded for KD.")

    kd_adapter = KnowledgeDistillationAdapter(
        config=DistillationConfig(
            name="kd-recovery",
            description="Knowledge distillation healing using uncompressed model as teacher",
            dataset=DatasetConfig(
                source=KD_DATASET,
                split="train",
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
            logging_steps=10,
            save_steps=KD_STEPS,
            # lora=LoRAConfig(r=8, lora_alpha=32),
        ),
        tokenizer=tokenizer,
        teacher_model_handle=teacher_handle_for_kd,
    )

    healed_model = kd_adapter.heal(compressed_model)
    logger.info("  Knowledge distillation complete.")

    # ── Step 8: Save healed model to disk ───────────────────────────────────────
    healed_path = Path("./healed_model")
    logger.info(f"\n[Step 8] Saving healed model to {healed_path}...")
    healed_model.save_pretrained(healed_path)
    tokenizer.save_pretrained(healed_path)
    logger.info("  Healed model saved.")

    # ── Step 9: Load healed model (new handle) ─────────────────────────────────
    logger.info("\n[Step 9] Loading healed model from disk...")
    healed_handle = HFModelHandle(
        source=ModelSource(
            identifier=str(healed_path),
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )
    healed_handle.load()
    logger.info(f"  Healed model loaded: {healed_handle.name}")

    # ── Step 10: Post-KD LMEval profiling ───────────────────────────────────────
    logger.info("\n[Step 10] Post-KD LMEval profiling...")

    final_gsm8k_results = gsm8k_profiler.profile("final-gsm8k", model_handle=healed_handle)
    profiling_records.append(final_gsm8k_results)
    logger.info(f"  Final GSM8K: {len(final_gsm8k_results)} result(s)")

    final_lambada_results = lambada_profiler.profile("final-lambada", model_handle=healed_handle)
    profiling_records.append(final_lambada_results)
    logger.info(f"  Final Lambada: {len(final_lambada_results)} result(s)")

    # ── Step 11: Generate reports ────────────────────────────────────────────────
    finished_at = datetime.now()
    logger.info("\n[Step 11] Generating reports...")

    result = PipelineRunResult(
        pipeline_name="manual-pipeline-validation",
        pipeline_description=(
            f"Manual pipeline: LMEval baseline -> depth prune {PRUNING_RATIO * 100:.0f}% -> "
            f"KD recovery -> final profile. Model={MODEL_ID}, Tasks=gsm8k,lambada"
        ),
        started_at=started_at,
        finished_at=finished_at,
        profiling_records=profiling_records,
    )

    output_dir = Path("./manual_pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = result.save_json(output_dir / "manual_validation_results.json")
    logger.info(f"  JSON saved: {json_path}")

    csv_path = result.save_csv(output_dir / "manual_validation_results.csv")
    logger.info(f"  CSV saved: {csv_path}")

    html_path = result.save_html_report(output_dir / "manual_validation_report.html")
    logger.info(f"  HTML report saved: {html_path}")

    comparison_csv = result.to_comparison_csv(output_dir / "manual_validation_comparison.csv")
    logger.info(f"  Comparison CSV saved: {comparison_csv}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed in {result.duration_seconds:.1f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    print("\n" + result.to_markdown_table())

    return result


if __name__ == "__main__":
    main()
