"""
adapters/model/huggingface.py — HuggingFace Hub + local model adapter.

Detects local vs Hub automatically based on ModelSource.is_local.
"""
from __future__ import annotations

from loguru import logger

from llm_flux.core.model import ModelHandle, ModelSource
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModelHandle(ModelHandle):
    """
    Load a causal LM from the HuggingFace Hub or a local directory.

    The same class handles both sources — ModelSource.is_local controls the path.
    """

    def __init__(self, source: ModelSource) -> None:
        self.source = source
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return self.source.identifier.split("/")[-1]

    def load(self) -> object:
        

        kwargs = {
            "pretrained_model_name_or_path": self.source.identifier,
            "trust_remote_code": self.source.trust_remote_code,
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if not self.source.is_local:
            kwargs["revision"] = self.source.revision
        if self.source.cache_dir:
            kwargs["cache_dir"] = self.source.cache_dir

        logger.info(f"  Loading model: {self.source.identifier} ({'local' if self.source.is_local else 'HF Hub'})")
        self._model = AutoModelForCausalLM.from_pretrained(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(**kwargs)
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        logger.info(f"  Model loaded on device: {next(self._model.parameters()).device}")
        return self._model

    def get_tokenizer(self) -> object:
        """Return the tokenizer loaded alongside the model."""
        if self._tokenizer is None:
            raise RuntimeError("Call load() before accessing the tokenizer.")
        return self._tokenizer

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"  Model '{self.name}' unloaded.")
    

    def get_model_instance(self) -> Any:
        """Return the loaded model instance, if already loaded."""
        if self._model is None:
            raise RuntimeError("Call load() before accessing the model instance.")
        return self._model

