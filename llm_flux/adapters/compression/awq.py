"""
adapters/compression/awq.py — AWQ quantization adapter (via autoawq).
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


class AWQConfig(CompressionConfig):
    name: str = "awq-compression"
    bits: int = Field(4, ge=2, le=8)
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"
    calibration_samples: int = 128
    calibration_dataset: DatasetConfig
    output_dir: str | None = None


class AWQAdapter(CompressionPort):
    """
    Activation-aware Weight Quantization via autoawq.

    Raises CompressionNotSupportedError if autoawq is not installed or
    the model architecture is not supported.
    """

    def __init__(self, config: AWQConfig, tokenizer: Any | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def compress(self, model_handle: ModelHandle) -> ModelHandle:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError as e:
            raise CompressionNotSupportedError(
                f"autoawq is not installed. Run: uv sync --extra awq. Error: {e}"
            )

        model = model_handle.get_model_instance()
        quant_config = {
            "zero_point": self.config.zero_point,
            "q_group_size": self.config.group_size,
            "w_bit": self.config.bits,
            "version": self.config.version,
        }

        try:
            from llm_flux.datasets.huggingface import HFDatasetAdapter
            from llm_flux.datasets.local import LocalDatasetAdapter

            awq_model = AutoAWQForCausalLM.from_pretrained(model.config._name_or_path)

            cfg = self.config.calibration_dataset
            if Path(cfg.source).exists():
                dataset = LocalDatasetAdapter(cfg).load()
            else:
                dataset = HFDatasetAdapter(cfg).load()

            texts = []
            for ex in dataset:
                text = ex.get("text", "")
                if text and text.strip():
                    texts.append(text)
                if len(texts) >= self.config.calibration_samples:
                    break

            awq_model.quantize(quant_config=quant_config, calib_data=texts)

            output_dir = self.config.output_dir or tempfile.mkdtemp(prefix="awq_compressed_")
            awq_model.save_pretrained(output_dir)

            actual_tokenizer = (
                self.tokenizer.get_tokenizer()
                if hasattr(self.tokenizer, "get_tokenizer")
                else self.tokenizer
            )

            return CompressedModelHandle(
                original_handle=model_handle,
                compressed_path=output_dir,
                compressed_model=awq_model,
                tokenizer=actual_tokenizer,
            )
        except Exception as e:
            raise CompressionNotSupportedError(f"AWQ compression failed on this model: {e}") from e
