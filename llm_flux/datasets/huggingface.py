"""
datasets/huggingface.py — Adapter: HuggingFace Hub dataset loader.
"""
from __future__ import annotations

from llm_flux.datasets.port import DatasetConfig, DatasetPort


class HFDatasetAdapter(DatasetPort):
    """Load a dataset from the HuggingFace Hub."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(self) -> object:
        from datasets import load_dataset  # lazy import

        kwargs: dict = {
            "path": self.config.source, 
            "split": self.config.split,
            "streaming": self.config.streaming
        }
        if self.config.subset:
            kwargs["name"] = self.config.subset

        dataset = load_dataset(**kwargs)

        if self.config.max_samples is not None:
            if self.config.streaming:
                dataset = dataset.take(self.config.max_samples)
            else:
                dataset = dataset.shuffle(seed=self.config.seed).select(
                    range(min(self.config.max_samples, len(dataset)))
                )

        return dataset
