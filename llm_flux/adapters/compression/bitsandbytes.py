"""
adapters/compression/bitsandbytes.py — BitsAndBytes quantization adapter.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_flux.core.compression import (
    CompressionConfig,
    CompressionNotSupportedError,
    CompressionPort,
)


class BitsAndBytesConfig(CompressionConfig):
    """
    Configuration for BitsAndBytes quantization.

    Set ``load_in_4bit=True`` for NF4 quantization (QLoRA-style),
    or ``load_in_8bit=True`` for INT8.
    """

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"  # e.g. "float16", "bfloat16"
    bnb_4bit_quant_type: str = "nf4"           # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True


class BitsAndBytesAdapter(CompressionPort):
    """
    Re-loads the model with BitsAndBytes quantization config applied.

    Note: BitsAndBytes quantization is applied at load time (not post-training).
    If the model was already loaded, this adapter unloads and reloads it.
    This requires access to a ``ModelHandle`` — pass it via ``model_handle``.
    """

    def __init__(self, config: BitsAndBytesConfig, model_handle: Any) -> None:
        self.config = config
        self.model_handle = model_handle

    def compress(self, model: Any) -> Any:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig as HFBnBConfig, AutoModelForCausalLM
        except ImportError as e:
            raise CompressionNotSupportedError(
                f"bitsandbytes is not installed. Run: uv sync --extra bitsandbytes. "
                f"Original error: {e}"
            )

        cfg = self.config
        bnb_cfg = HFBnBConfig(
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

        # Re-load the model with quantization config
        source = self.model_handle.source
        quantized = AutoModelForCausalLM.from_pretrained(
            source.identifier,
            quantization_config=bnb_cfg,
            trust_remote_code=source.trust_remote_code,
            device_map="auto",
        )
        return quantized
