"""
examples/e2e_option_c_validation.py

CPU-friendly end-to-end validation for Option C (Perplexity + QA first).
Uses a tiny random LLaMA model and first-N dataset slicing via limit_test_samples.
"""
from __future__ import annotations

import os
from pathlib import Path
from llm_flux.core.model import ModelSource
from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.datasets.port import DatasetConfig
from llm_flux.profiling.adapters.comprehensive import (
    ComprehensiveProfilingAdapter,
    ComprehensiveProfilingConfig,
)
from llm_flux.runner import run_pipeline
from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter, 
    DepthPruningConfig,
    angular_distance_importance
)
from loguru import logger

CACHE_DIR = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_ID = "Qwen/Qwen3-0.6B"
FIRST_N = 5

RATIOS = [
    0.05, 
    0.10, 
    0.15, 
    0.20
]
CALIBRATION_SAMPLES = 2
DATASET_SOURCE = "dummy_dataset.jsonl"



def run_experiment_for_ratio(ratio: float):
    model = HFModelHandle(
        source=ModelSource(
            identifier=MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    )

    profiler = ComprehensiveProfilingAdapter(
        config=ComprehensiveProfilingConfig(
            name="option-c-profiler",
            description="Option C validation: perplexity + QA tests with first-N rows.",
            benchmark_tests={"lambada", "squad", "arc_challenge", "gsm8k", "truthfulqa_mc2"},
            limit_test_samples=FIRST_N,
            run_hardware_profile=True,
            num_warmup_runs=2,
            num_benchmark_runs=5,
            max_new_tokens=24,
            prompt="Summarize compression in one sentence.",
        ),
        tokenizer=model,
    )

    depth_pruner = DepthPruningAdapter(
        config=DepthPruningConfig(
            name=f"depth-pruning-{ratio*100:.0f}pct",
            description=f"Drops {ratio*100:.0f}% of layers using angular distance calculation.",
            pruning_ratio=ratio,
            calibration_samples=CALIBRATION_SAMPLES,
            calibration_dataset=DatasetConfig(
                source=DATASET_SOURCE,
                subset="en",
                split="train",
                streaming=False,
                max_samples=CALIBRATION_SAMPLES,
            ),
        ),
        tokenizer=model,
        importance_fn=angular_distance_importance,
    )

    pipeline = Pipeline(
        name="option-c-first-n-validation",
        description="Quick validation run with first-N benchmark examples.",
        steps=[
            PipelineStep(label="Load Model", port=model),
            PipelineStep(label="Profile Benchmarks", port=profiler),
            PipelineStep(label="Apply Depth Pruning", port=depth_pruner),
            PipelineStep(label="Profile Post-Pruning", port=profiler),
        ],
    )

    output_dir = Path("compression_results") / pipeline.name
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(
        pipeline=pipeline,
        dag_output=str(output_dir / "dag_graph.png"),
        result_output=str(output_dir / "results.json"),
        html_report_output=str(output_dir / "report.html"),
        skip_confirmation=True,
        show_dag=False,
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
