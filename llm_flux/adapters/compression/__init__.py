"""Compression adapters."""

from llm_flux.adapters.compression.awq import AWQAdapter, AWQConfig
from llm_flux.adapters.compression.bitsandbytes import BitsAndBytesAdapter, BitsAndBytesConfig
from llm_flux.adapters.compression.depth_pruning import DepthPruningAdapter, DepthPruningConfig
from llm_flux.adapters.compression.gptq import GPTQAdapter, GPTQConfig

__all__ = [
    "BitsAndBytesAdapter", "BitsAndBytesConfig",
    "GPTQAdapter", "GPTQConfig",
    "AWQAdapter", "AWQConfig",
    "DepthPruningAdapter", "DepthPruningConfig",
]
