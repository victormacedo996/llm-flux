"""
adapters/compression/bitsandbytes.py — BitsAndBytes quantization adapter.
"""

from __future__ import annotations

import tempfile
from typing import Any

from llm_flux.core.compression import (
    CompressionConfig,
    CompressionNotSupportedError,
    CompressionPort,
)
from llm_flux.core.model import CompressedModelHandle, ModelHandle


class BitsAndBytesConfig(CompressionConfig):
    """
    Configuration for BitsAndBytes quantization.

    Set ``load_in_4bit=True`` for NF4 quantization (QLoRA-style),
    or ``load_in_8bit=True`` for INT8.
    """

    name: str = "bitsandbytes-compression"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    output_dir: str | None = None


class BitsAndBytesAdapter(CompressionPort):
    """
    Re-loads the model with BitsAndBytes quantization config applied,
    saves the quantized model to disk, and returns a CompressedModelHandle.

    Note: BitsAndBytes quantization is applied at load time (not post-training).
    If the model was already loaded, this adapter unloads and reloads it.
    This requires access to a ``ModelHandle`` — pass it via ``model_handle``.
    """

    def __init__(self, config: BitsAndBytesConfig, model_handle: Any) -> None:
        self.config = config
        self.model_handle = model_handle

    def compress(self, model_handle: ModelHandle) -> ModelHandle:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import BitsAndBytesConfig as HFBnBConfig
        except ImportError as e:
            raise CompressionNotSupportedError(
                f"bitsandbytes is not installed. Run: uv sync --extra bitsandbytes. "
                f"Original error: {e}"
            )

        source = self.model_handle.source
        bnb_cfg = HFBnBConfig(
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        quantized = AutoModelForCausalLM.from_pretrained(
            source.identifier,
            quantization_config=bnb_cfg,
            trust_remote_code=source.trust_remote_code,
            device_map="auto",
        )

        output_dir = self.config.output_dir or tempfile.mkdtemp(prefix="bnb_compressed_")
        quantized.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            source.identifier,
            trust_remote_code=source.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return CompressedModelHandle(
            original_handle=self.model_handle,
            compressed_path=output_dir,
            compressed_model=quantized,
            tokenizer=tokenizer,
        )
