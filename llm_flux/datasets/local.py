"""
datasets/local.py — Adapter: local directory / JSON / CSV dataset loader.
"""
from __future__ import annotations

from pathlib import Path

from llm_flux.datasets.port import DatasetConfig, DatasetPort


class LocalDatasetAdapter(DatasetPort):
    """
    Load a dataset from a local file or directory.

    Supported formats (auto-detected by extension): JSON, JSONL, CSV, Parquet.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(self) -> object:
        from datasets import load_dataset  # lazy import

        path = Path(self.config.source)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {path}")

        suffix = path.suffix.lower()
        format_map = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".parquet": "parquet",
        }
        data_format = format_map.get(suffix, "json")

        dataset = load_dataset(
            data_format,
            data_files=str(path),
            split=self.config.split,
        )

        if self.config.max_samples is not None:
            dataset = dataset.shuffle(seed=self.config.seed).select(
                range(min(self.config.max_samples, len(dataset)))
            )

        return dataset
