"""
adapters/compression/gptq.py — GPTQ quantization adapter (via auto-gptq).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field

from llm_flux.core.compression import (
    CompressionConfig,
    CompressionNotSupportedError,
    CompressionPort,
)
from llm_flux.core.model import CompressedModelHandle, ModelHandle
from llm_flux.datasets.port import DatasetConfig


class GPTQConfig(CompressionConfig):
    name: str = "gptq-compression"
    bits: int = Field(4, ge=2, le=8, description="Quantization bit-width")
    group_size: int = Field(128, description="Group size for quantization")
    desc_act: bool = False
    calibration_samples: int = 128
    calibration_dataset: DatasetConfig
    output_dir: str | None = None


class GPTQAdapter(CompressionPort):
    """
    Post-training quantization via auto-gptq.

    Raises CompressionNotSupportedError if:
    - auto-gptq is not installed, or
    - the model does not expose linear layers compatible with GPTQ.
    """

    def __init__(self, config: GPTQConfig, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def compress(self, model_handle: ModelHandle) -> ModelHandle:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError as e:
            raise CompressionNotSupportedError(
                f"auto-gptq is not installed. Run: uv sync --extra gptq. Error: {e}"
            )

        model = model_handle.get_model_instance()
        quantize_config = BaseQuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
        )

        try:
            from llm_flux.datasets.huggingface import HFDatasetAdapter
            from llm_flux.datasets.local import LocalDatasetAdapter

            cfg = self.config.calibration_dataset
            if Path(cfg.source).exists():
                dataset = LocalDatasetAdapter(cfg).load()
            else:
                dataset = HFDatasetAdapter(cfg).load()

            actual_tokenizer = (
                self.tokenizer.get_tokenizer()
                if hasattr(self.tokenizer, "get_tokenizer")
                else self.tokenizer
            )
            examples = []
            for ex in dataset:
                text = ex.get("text", "")
                if text and text.strip():
                    examples.append(actual_tokenizer(text, return_tensors="pt"))
                if len(examples) >= self.config.calibration_samples:
                    break

            gptq_model = AutoGPTQForCausalLM.from_pretrained(
                model.config._name_or_path,
                quantize_config=quantize_config,
            )
            gptq_model.quantize(examples)

            output_dir = self.config.output_dir or tempfile.mkdtemp(prefix="gptq_compressed_")
            gptq_model.save_pretrained(output_dir)

            return CompressedModelHandle(
                original_handle=model_handle,
                compressed_path=output_dir,
                compressed_model=gptq_model,
                tokenizer=actual_tokenizer,
            )
        except Exception as e:
            raise CompressionNotSupportedError(f"GPTQ compression failed on this model: {e}") from e
