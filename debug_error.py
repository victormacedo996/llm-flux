
import torch
from llm_flux.adapters.model.huggingface import HFModelHandle
from llm_flux.core.model import ModelSource
from llm_flux.adapters.compression.depth_pruning import DepthPruningAdapter, DepthPruningConfig
from llm_flux.datasets.port import DatasetConfig
from llm_flux.profiling.adapters.comprehensive import ComprehensiveProfilingAdapter, ComprehensiveProfilingConfig
import os

CACHE_DIR = "/mnt/3c2f822b-db13-4837-ba6e-3d7b256042cc/repositorios/mestrado/ag_test/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

model_handle = HFModelHandle(source=ModelSource(identifier=MODEL_ID, cache_dir=CACHE_DIR))
model = model_handle.load()

# Prune
adapter = DepthPruningAdapter(
    config=DepthPruningConfig(
        pruning_ratio=0.2,
        calibration_samples=4,
        calibration_dataset=DatasetConfig(source="allenai/c4", subset="en", split="train", streaming=True, max_samples=4)
    ),
    tokenizer=model_handle
)
model = adapter.compress(model)

# Profile
profiler = ComprehensiveProfilingAdapter(
    config=ComprehensiveProfilingConfig(name="test", run_hardware_profile=False, limit_test_samples=2, num_benchmark_runs=1, num_warmup_runs=0),
    tokenizer=model_handle
)

try:
    profiler.profile(model, "Post-Prune")
except Exception:
    import traceback
    traceback.print_exc()
