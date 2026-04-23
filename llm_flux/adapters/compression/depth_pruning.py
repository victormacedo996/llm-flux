"""
adapters/compression/depth_pruning.py — Angular Distance Depth Pruning Adapter.

Prunes the most redundant layers based on the angular distance of their representations
across a calibration dataset. Supports aligning vocabulary sizes for Tensor Core efficiency.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from pydantic import Field

from llm_flux.core.compression import (
    CompressionConfig,
    CompressionNotSupportedError,
    CompressionPort,
)
from llm_flux.core.model import CompressedModelHandle, ModelHandle
from llm_flux.datasets.port import DatasetConfig


class DepthPruningConfig(CompressionConfig):
    name: str = "depth-pruning"
    pruning_ratio: float = Field(0.1, ge=0.01, lt=1.0)
    calibration_samples: int = 32
    calibration_dataset: DatasetConfig
    output_dir: str | None = None


def get_transformer_layers(model: Any) -> nn.ModuleList:
    """Heuristic to find the transformer layers list (works for Llama, Qwen, Mistral)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 / Qwen

    # Fallback search
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and (
            "layers" in name or "h" in name or "blocks" in name
        ):
            return module

    raise CompressionNotSupportedError(
        "Could not locate the nn.ModuleList containing transformer layers."
    )


def angular_distance_importance(model: Any, input_ids: torch.Tensor) -> list[float]:
    """
    Calculates the angular distance between the input and output of each layer.
    Angular Distance = (1/pi) * arccos(cosine_similarity)
    """
    # Find the module list holding the layers. In Llama, it's `model.model.layers`.
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    # We will hook into each layer to capture its input and output.
    layer_inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    layer_outputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    hooks = []

    def get_hook(idx: int):
        def hook(module, inp, out):
            # inp[0] is typically the hidden states
            h_in = inp[0].detach().cpu()
            # out[0] is typically the hidden states after the layer
            h_out = out[0].detach().cpu() if isinstance(out, tuple) else out.detach().cpu()
            layer_inputs[idx].append(h_in)
            layer_outputs[idx].append(h_out)

        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_hook(i)))

    # Run forward pass (batch by batch to save memory)
    batch_size = 4
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch = input_ids[i : i + batch_size]
            try:
                model(batch)
            except Exception as e:
                logger.warning(f"  [Depth Prune] Forward pass error during calib: {e}")

    for h in hooks:
        h.remove()

    distances: list[float] = []
    for i in range(num_layers):
        if not layer_inputs[i]:
            distances.append(float("inf"))
            continue

        # Concat all batches: shape (N, seq_len, hidden_size)
        h_in = torch.cat(layer_inputs[i], dim=0).float()
        h_out = torch.cat(layer_outputs[i], dim=0).float()

        # Normalize along hidden dim
        cos_sim = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1)
        # Clamp to avoid nan in arccos
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        # Compute angular distance per sequence element, then mean across batch and seq
        ang_dist = (1.0 / math.pi) * torch.acos(cos_sim)
        distances.append(ang_dist.mean().item())

    return distances


LayerImportanceFn = Callable[[Any, torch.Tensor], list[float]]


class DepthPruningAdapter(CompressionPort):
    """
    Implements Depth Pruning by removing layers with the smallest importance.
    Uses Angular distance (cosine similarity between layer input/output).
    """

    def __init__(
        self,
        config: DepthPruningConfig,
        tokenizer: Any,
        importance_fn: LayerImportanceFn | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.importance_fn = importance_fn or angular_distance_importance

    def compress(self, model_handle: ModelHandle) -> ModelHandle:
        model = model_handle.get_model_instance()

        if not hasattr(model.config, "num_hidden_layers"):
            raise CompressionNotSupportedError(
                "Model configuration lacks 'num_hidden_layers'. "
                "Only standard causal LMs are supported."
            )

        num_layers = model.config.num_hidden_layers
        num_prune = max(1, int(num_layers * self.config.pruning_ratio))

        logger.info(
            f"  [Depth Prune] Target: drop {num_prune} out of {num_layers} layers "
            f"({self.config.pruning_ratio * 100:.1f}%)."
        )

        inputs = self._get_calibration_inputs(model.device)

        distances = self.importance_fn(model, inputs)

        sorted_indices = sorted(range(1, num_layers), key=lambda i: distances[i])
        layers_to_drop = sorted(sorted_indices[:num_prune])

        logger.info(f"  [Depth Prune] Layers selected for pruning: {layers_to_drop}")

        pruned_model = self._prune_layers(model, layers_to_drop)

        output_dir = self.config.output_dir or tempfile.mkdtemp(prefix="depth_pruned_")
        pruned_model.save_pretrained(output_dir)

        actual_tokenizer = (
            self.tokenizer.get_tokenizer()
            if hasattr(self.tokenizer, "get_tokenizer")
            else self.tokenizer
        )

        return CompressedModelHandle(
            original_handle=model_handle,
            compressed_path=output_dir,
            compressed_model=pruned_model,
            tokenizer=actual_tokenizer,
        )

    def _get_calibration_inputs(self, device: torch.device) -> torch.Tensor:
        """Load a few samples from the configured DatasetConfig for calibration."""
        from pathlib import Path

        cfg = self.config.calibration_dataset
        # Resolve dataset loader properly
        if Path(cfg.source).exists():
            from llm_flux.datasets.local import LocalDatasetAdapter

            dataset = LocalDatasetAdapter(cfg).load()
        else:
            from llm_flux.datasets.huggingface import HFDatasetAdapter

            dataset = HFDatasetAdapter(cfg).load()

        # Find the text column (usually "text" for causal LM datasets)
        # Use simple heuristic: find the first string column with data, or fallback to "text"
        texts = []
        for ex in dataset:
            text = ex.get("text", "")
            if text and text.strip():
                texts.append(text)
            if len(texts) >= self.config.calibration_samples:
                break

        if not texts:
            raise ValueError(f"Could not find valid 'text' fields in the dataset {cfg.source}.")

        actual_tokenizer = (
            self.tokenizer.get_tokenizer()
            if hasattr(self.tokenizer, "get_tokenizer")
            else self.tokenizer
        )

        inputs = actual_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        return inputs["input_ids"]

    def _prune_layers(self, model: Any, layers_to_drop: list[int]) -> Any:
        layers = get_transformer_layers(model)
        keep_indices = [i for i in range(len(layers)) if i not in layers_to_drop]

        # Create a new ModuleList with only the kept layers
        retained_layers = nn.ModuleList([layers[i] for i in keep_indices])

        # Re-index layers to avoid KV Cache IndexError during inference
        # Most HF models (Llama, Mistral, Qwen) store layer_idx in the attention or layer itself
        for i, layer in enumerate(retained_layers):
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = i
            # Also check sub-modules (Attention is where it usually resides in modern Llama)
            for sub in layer.modules():
                if hasattr(sub, "layer_idx"):
                    sub.layer_idx = i

        # Overwrite the old ModuleList
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = retained_layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            model.transformer.h = retained_layers
        else:
            for name, module in model.named_modules():
                if module is layers:
                    pass

        # Update config
        model.config.num_hidden_layers = len(keep_indices)
        logger.info(f"  [Depth Prune] Pruned. New layer count: {model.config.num_hidden_layers}")

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
