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
        from transformers import TrainingArguments, DataCollatorForLanguageModeling

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

        def tokenize_function(examples):
            return actual_tokenizer(examples["text"], truncation=True, max_length=512)

        logger.info("  🔤 Tokenizing dataset...")
        # For IterableDataset, column_names might be None until actually peeked
        # We explicitly remove "text" which is the only one we know is there
        tokenized_dataset = raw_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"] if "text" in (raw_dataset.column_names or ["text"]) else None
        )

        # ── Apply LoRA if requested ───────────────────────────────────────────
        if cfg.lora:
            logger.info("  🔧 Applying LoRA (PEFT)...")
            try:
                from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

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
    def _default_trainer() -> type:
        from transformers import Trainer

        return Trainer
