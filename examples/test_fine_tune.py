import os
import sys
from pathlib import Path
from loguru import logger

TINY_MODE = False  # Set to True for lightning-fast end-to-end pipeline validation

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
    MODEL_ID = "Qwen/Qwen3-0.6B"
    DATASET_SOURCE = "allenai/c4"
    C4_SAMPLES_FOR_HEALING = 2
    LIMIT_TEST_SAMPLES = 2
    NUM_BENCHMARK_RUNS = 1
    HEALING_STEPS = 1
    CALIBRATION_SAMPLES = 2
    USE_FP16 = HAS_GPU # Only use FP16 if we have a GPU


# Set Hugging Face environment variables globally so ALL sub-libraries (datasets, peft, AutoAWQ) respect it
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.adapters.compression.depth_pruning import (
    DepthPruningAdapter, 
    DepthPruningConfig,
    angular_distance_importance
)
# from modelforge.adapters.compression.mlp_pruning import MLPPrunin`gAdapter
from llm_flux.adapters.compression.awq import AWQAdapter, AWQConfig
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


model_handle = HFModelHandle(
    source=ModelSource(
        identifier=MODEL_ID, 
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
)

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
        trainer_class=None,
        max_steps=HEALING_STEPS,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=USE_FP16, # Ensure efficient dtype for target device
        lora=LoRAConfig(r=8, lora_alpha=32),
    ),
    tokenizer=model_handle
)


pipeline = Pipeline(
    name=f"test fine tune",
    description=(
        f"Test compresion pipeline"
    ),
    steps=[
        PipelineStep(label="Load Model",            port=model_handle),
        PipelineStep(label="LoRA Heal",             port=healer),
    ]
)


if __name__ == "__main__":
    result_report_output = Path(__file__).parent.parent / f"compression_results" / pipeline.name
    result = run_pipeline(
        pipeline=pipeline,
        dag_output=f"{result_report_output}/dag_graph.png",
        result_output=f"{result_report_output}/results.json",
        html_report_output=f"{result_report_output}/report.html",
        skip_confirmation=True,  # Automated experiment script
        show_dag=False,          # Don't block on diagram window
    )