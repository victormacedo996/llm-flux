"""
adapters/compression/awq.py — AWQ quantization adapter (via autoawq).
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from llm_flux.core.compression import (
    CompressionConfig,
    CompressionNotSupportedError,
    CompressionPort,
)
from llm_flux.datasets.port import DatasetConfig


class AWQConfig(CompressionConfig):
    name: str = "awq-compression"
    bits: int = Field(4, ge=2, le=8)
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"
    calibration_samples: int = 128
    calibration_dataset: DatasetConfig


class AWQAdapter(CompressionPort):
    """
    Activation-aware Weight Quantization via autoawq.

    Raises CompressionNotSupportedError if autoawq is not installed or
    the model architecture is not supported.
    """

    def __init__(self, config: AWQConfig) -> None:
        self.config = config

    def compress(self, model: Any) -> Any:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError as e:
            raise CompressionNotSupportedError(
                f"autoawq is not installed. Run: uv sync --extra awq. Error: {e}"
            )

        quant_config = {
            "zero_point": self.config.zero_point,
            "q_group_size": self.config.group_size,
            "w_bit": self.config.bits,
            "version": self.config.version,
        }

        try:
            awq_model = AutoAWQForCausalLM.from_pretrained(model.config._name_or_path)
            
            from pathlib import Path

            cfg = self.config.calibration_dataset
            if Path(cfg.source).exists():
                from llm_flux.datasets.local import LocalDatasetAdapter
                dataset = LocalDatasetAdapter(cfg).load()
            else:
                from llm_flux.datasets.huggingface import HFDatasetAdapter
                dataset = HFDatasetAdapter(cfg).load()
                
            # AWQ expects a list of strings or a tokenized dataset for calibration
            texts = []
            for ex in dataset:
                text = ex.get("text", "")
                if text and text.strip():
                    texts.append(text)
                if len(texts) >= self.config.calibration_samples:
                    break
                    
            awq_model.quantize(quant_config=quant_config, calib_data=texts)
            return awq_model
        except Exception as e:
            raise CompressionNotSupportedError(
                f"AWQ compression failed on this model: {e}"
            ) from e
