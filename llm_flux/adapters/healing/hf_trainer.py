"""
adapters/healing/hf_trainer.py — Default HuggingFace Trainer + optional PEFT/LoRA adapter.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from llm_flux.core.healing import HealingConfig, HealingPort
from llm_flux.datasets.huggingface import HFDatasetAdapter
from llm_flux.datasets.local import LocalDatasetAdapter
from llm_flux.datasets.port import DatasetConfig


def _load_dataset(config: DatasetConfig) -> Any:
    """Auto-select the right dataset adapter based on the source path."""
    from pathlib import Path

    if Path(config.source).exists():
        return LocalDatasetAdapter(config).load()
    return HFDatasetAdapter(config).load()


class HFTrainerAdapter(HealingPort):
    """
    Default model healing adapter using HuggingFace Trainer.

    Features:
    - Automatically applies LoRA / QLoRA via PEFT if ``config.lora`` is set.
    - Supports a custom ``config.trainer_class`` (duck-typed) to swap the Trainer.
    - Loads the training dataset from HuggingFace Hub or a local path.

    Minimal usage::

        healer = HFTrainerAdapter(HealingConfig(
            dataset=DatasetConfig(source="tatsu-lab/alpaca", max_samples=2000),
            max_steps=300,
            lora=LoRAConfig(r=16),
        ))
    """

    def __init__(self, config: HealingConfig, tokenizer: Any = None) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def heal(self, model: Any) -> Any:
        from transformers import DataCollatorForLanguageModeling, TrainingArguments

        cfg = self.config

        # ── Load and tokenize dataset ──────────────────────────────────────────
        logger.info(f"  📂 Loading healing dataset: {cfg.dataset.source}")
        raw_dataset = _load_dataset(cfg.dataset)
        
        # Resolve tokenizer handle
        actual_tokenizer = self.tokenizer.get_tokenizer() if hasattr(self.tokenizer, "get_tokenizer") else self.tokenizer
        if actual_tokenizer is None:
            raise ValueError("A tokenizer must be provided to HFTrainerAdapter (via constructor) to heal the model.")

        if actual_tokenizer.pad_token is None:
            actual_tokenizer.pad_token = actual_tokenizer.eos_token

        # Determine max sequence length
        max_seq_len = cfg.max_seq_length
        if max_seq_len is None:
            max_seq_len = self._get_max_seq_length(model, actual_tokenizer)
            logger.info(f"  🔍 Auto-detected max_seq_length: {max_seq_len}")
        else:
            logger.info(f"  📍 Using configured max_seq_length: {max_seq_len}")

        def tokenize_function(examples):
            return actual_tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=max_seq_len
            )

        logger.info("  🔤 Tokenizing dataset...")
        # For IterableDataset, column_names might be None until actually peeked
        # We explicitly remove "text" which is the only one we know is there
        tokenized_dataset = raw_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"] if "text" in (raw_dataset.column_names or ["text"]) else None
        )

        # ── Materialise streaming datasets ────────────────────────────────────
        # IterableDatasets are lazy: .map() sets up a pipeline but does NO work.
        # The DataLoader would then stream (and re-stream) data from the internet
        # during training, causing apparent hangs and repeated downloads.
        # Converting to a regular in-memory Dataset avoids all of that.
        try:
            from datasets import Dataset as _Dataset
            from datasets import IterableDataset as _IterableDataset
            if isinstance(tokenized_dataset, _IterableDataset):
                logger.info("  🔄 Materialising streaming dataset into memory (avoids lazy I/O during training)...")
                tokenized_dataset = _Dataset.from_list(list(tokenized_dataset))
                logger.info(f"  ✔ Materialised {len(tokenized_dataset)} samples.")
        except ImportError:
            pass

        # Warn if there are too few samples to complete even a single optimiser step
        min_needed = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        if hasattr(tokenized_dataset, "__len__") and len(tokenized_dataset) < min_needed:
            logger.warning(
                f"  ⚠️  Dataset has only {len(tokenized_dataset)} sample(s), but "
                f"per_device_train_batch_size={cfg.per_device_train_batch_size} × "
                f"gradient_accumulation_steps={cfg.gradient_accumulation_steps} requires "
                f"at least {min_needed} samples to complete one optimiser step. "
                "Consider reducing gradient_accumulation_steps or increasing max_samples."
            )

        # ── Apply LoRA if requested ───────────────────────────────────────────
        if cfg.lora:
            logger.info("  🔧 Applying LoRA (PEFT)...")
            try:
                from peft import LoraConfig as PeftLoraConfig
                from peft import TaskType, get_peft_model

                lc = cfg.lora
                peft_cfg = PeftLoraConfig(
                    r=lc.r,
                    lora_alpha=lc.lora_alpha,
                    target_modules=lc.target_modules,
                    lora_dropout=lc.lora_dropout,
                    bias=lc.bias,
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, peft_cfg)
                model.print_trainable_parameters()
            except ImportError as e:
                raise RuntimeError(
                    f"PEFT is not installed. Run: uv add peft. Error: {e}"
                ) from e

        # ── Training arguments ────────────────────────────────────────────────
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            fp16=cfg.fp16,
            save_steps=cfg.save_steps,
            logging_steps=cfg.logging_steps,
            report_to="none",  # disable wandb/etc by default
            remove_unused_columns=True, # Standard behavior is best
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=actual_tokenizer, mlm=False)
        trainer_cls = cfg.trainer_class or self._default_trainer()

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        logger.info(f"  🏋️  Starting training for {cfg.max_steps} steps...")
        trainer.train()
        logger.info("  ✅ Healing complete.")
        return model

    @staticmethod
    def _get_max_seq_length(model: Any, tokenizer: Any, default: int = 512) -> int:
        """
        Heuristic to find the model's maximum supported sequence length.
        """
        # 1. Try tokenizer property
        res = getattr(tokenizer, "model_max_length", None)
        if res and isinstance(res, (int, float)) and res < 1e6:
            return int(res)

        # 2. Try model config attributes
        if hasattr(model, "config"):
            for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
                res = getattr(model.config, attr, None)
                if res and isinstance(res, (int, float)):
                    return int(res)

        return default

    @staticmethod
    def _default_trainer() -> type:
        from transformers import Trainer

        return Trainer
