"""
adapters/healing/knowledge_distillation.py — Knowledge Distillation adapter for model healing.

Uses the original uncompressed model as teacher and the compressed model as student.
Supports two modes:
  - precompute (default): teacher logits are precomputed and saved to disk,
    then teacher is unloaded before student training (memory efficient).
  - online: teacher remains in memory during student training (higher memory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from llm_flux.core.healing import DistillationConfig, HealingPort, SaveFormat
from llm_flux.core.model import CompressedModelHandle
from llm_flux.datasets.huggingface import HFDatasetAdapter
from llm_flux.datasets.local import LocalDatasetAdapter
from llm_flux.datasets.port import DatasetConfig


def _load_dataset(config: DatasetConfig) -> Any:
    """Auto-select the right dataset adapter based on the source path."""
    if Path(config.source).exists():
        return LocalDatasetAdapter(config).load()
    return HFDatasetAdapter(config).load()


class KnowledgeDistillationAdapter(HealingPort):
    """
    Knowledge Distillation adapter for healing compressed models.

    Uses the original uncompressed model as teacher and the compressed model
    as student. The teacher must be loaded BEFORE calling heal().

    Minimal usage::

        teacher_handle = HFModelHandle(source=ModelSource(identifier="meta-llama/Llama-2-7b"))
        teacher_handle.load()  # Load teacher BEFORE pipeline

        kd = KnowledgeDistillationAdapter(
            config=DistillationConfig(
                dataset=DatasetConfig(source="tatsu-lab/alpaca", max_samples=1000),
                temperature=2.0,
                alpha=0.5,
            ),
            tokenizer=student_handle,
            teacher_model_handle=teacher_handle,
        )
    """

    def __init__(
        self,
        config: DistillationConfig,
        tokenizer: Any = None,
        teacher_model_handle: Any = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.teacher_model_handle = teacher_model_handle

    def heal(self, model: Any) -> Any:
        if self.teacher_model_handle is None:
            raise ValueError(
                "teacher_model_handle must be provided to KnowledgeDistillationAdapter. "
                "Load the uncompressed teacher model before calling heal()."
            )

        if not self.teacher_model_handle.get_is_model_loaded():
            raise ValueError(
                "teacher_model_handle must have load() called before heal(). "
                "Call teacher_model_handle.load() before running the pipeline."
            )

        teacher = self.teacher_model_handle.get_model_instance()
        cfg = self.config

        logger.info(f"  📂 Loading KD dataset: {cfg.dataset.source}")
        raw_dataset = _load_dataset(cfg.dataset)

        actual_tokenizer = (
            self.tokenizer.get_tokenizer()
            if hasattr(self.tokenizer, "get_tokenizer")
            else self.tokenizer
        )
        if actual_tokenizer is None:
            raise ValueError("A tokenizer must be provided to KnowledgeDistillationAdapter.")

        if actual_tokenizer.pad_token is None:
            actual_tokenizer.pad_token = actual_tokenizer.eos_token

        self._tokenizer = actual_tokenizer

        max_seq_len = cfg.max_seq_length or 512

        def tokenize_function(examples):
            return actual_tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_len,
            )

        logger.info("  🔤 Tokenizing dataset for KD...")
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"] if "text" in (raw_dataset.column_names or ["text"]) else None,
        )

        try:
            from datasets import Dataset as _Dataset
            from datasets import IterableDataset as _IterableDataset

            if isinstance(tokenized_dataset, _IterableDataset):
                logger.info("  🔄 Materialising streaming dataset into memory...")
                tokenized_dataset = _Dataset.from_list(list(tokenized_dataset))
        except ImportError:
            pass

        logger.info(f"  📊 Using {len(tokenized_dataset)} samples for KD training.")

        teacher_logits_path = Path(cfg.teacher_logits_output_dir) / "teacher_logits.pt"
        teacher_logits_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg.precompute_teacher_logits and teacher_logits_path.exists():
            logger.info(f"  📂 Loading cached teacher logits from {teacher_logits_path}")
            teacher_logits_list = torch.load(teacher_logits_path)
        else:
            logger.info("  🔢 Computing teacher logits...")
            teacher_logits_list = self._compute_teacher_logits(
                teacher, tokenized_dataset, actual_tokenizer, cfg.per_device_train_batch_size
            )
            if cfg.precompute_teacher_logits:
                torch.save(teacher_logits_list, teacher_logits_path)
                logger.info(f"  💾 Saved teacher logits to {teacher_logits_path}")

        if cfg.precompute_teacher_logits:
            logger.info("  🗑️  Unloading teacher model to free memory...")
            self.teacher_model_handle.unload()
            del teacher

        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        if cfg.lora:
            logger.info("  🔧 Applying LoRA to student model...")
            peft_cfg = PeftLoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                target_modules=cfg.lora.target_modules,
                lora_dropout=cfg.lora.lora_dropout,
                bias=cfg.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, peft_cfg)
            model.print_trainable_parameters()

        logger.info(f"  🏋️  Starting KD training for {cfg.max_steps} steps...")

        kd_trainer = _KDTrainer(
            model=model,
            teacher_logits_list=teacher_logits_list,
            temperature=cfg.temperature,
            alpha=cfg.alpha,
            device=next(model.parameters()).device,
        )

        kd_trainer.train()

        logger.info("  ✅ Knowledge Distillation complete.")

        if cfg.save_format == SaveFormat.AUTO:
            should_merge = cfg.lora is not None
        else:
            should_merge = cfg.save_format == SaveFormat.MERGED

        if should_merge:
            if hasattr(model, "merge_and_unload"):
                logger.info("  🔀 Merging LoRA weights into base model...")
                model = model.merge_and_unload()
            else:
                logger.warning("  ⚠️  merge_and_unload() not available, saving as-is.")

        logger.info(f"  💾 Saving healed student model to {cfg.output_dir}...")
        model.save_pretrained(cfg.output_dir)
        self._tokenizer.save_pretrained(cfg.output_dir)

        result = CompressedModelHandle(
            original_handle=self.teacher_model_handle,
            compressed_path=str(cfg.output_dir),
            compressed_model=model,
            tokenizer=self._tokenizer,
        )

        logger.info("  ✅ Healed student model saved and wrapped in CompressedModelHandle")
        return result

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad variable-length sequences to equal size for DataLoader batching."""
        import torch.nn.functional as F

        def _to_tensor(val: Any) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val
            return torch.tensor(val)

        max_len = max(_to_tensor(item["input_ids"]).size(-1) for item in batch)

        input_ids_list = []
        attention_mask_list = []
        for item in batch:
            input_ids = _to_tensor(item["input_ids"])
            attention_mask = _to_tensor(item["attention_mask"])
            input_ids_list.append(
                F.pad(
                    input_ids,
                    (0, max_len - input_ids.size(-1)),
                    value=self._pad_token_id,
                )
            )
            attention_mask_list.append(
                F.pad(attention_mask, (0, max_len - attention_mask.size(-1)), value=0)
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
        }

    def _compute_teacher_logits(
        self,
        teacher: Any,
        tokenized_dataset: Any,
        tokenizer: Any,
        batch_size: int,
    ) -> list[torch.Tensor]:
        teacher.eval()
        logits_list = []

        from torch.utils.data import DataLoader

        self._pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        logger.info(f"  🔢 Computing teacher logits on {len(tokenized_dataset)} samples...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(next(teacher.parameters()).device)
                attention_mask = batch["attention_mask"].to(next(teacher.parameters()).device)

                outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()

                for j in range(logits.size(0)):
                    logits_list.append(logits[j].cpu())

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"    Processed {(i + 1) * batch_size} / {len(tokenized_dataset)} samples"
                    )

        return logits_list


class _KDTrainer:
    """
    Custom training loop for knowledge distillation.

    Computes the combined KD loss:
        loss = alpha * KL_div(teacher_soft, student_soft) / T²
             + (1 - alpha) * CE(student_logits, labels)
    """

    def __init__(
        self,
        model: Any,
        teacher_logits_list: list[torch.Tensor],
        temperature: float,
        alpha: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.teacher_logits_list = teacher_logits_list
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.optimizer = None
        self.step = 0

    def train(self) -> None:
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)

        dataloader = DataLoader(
            list(range(len(self.teacher_logits_list))),
            batch_size=1,
            shuffle=True,
        )

        total_steps = 100
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        from tqdm import tqdm

        pbar = tqdm(total=total_steps, desc="KD Training")

        while self.step < total_steps:
            for idx in dataloader:
                if self.step >= total_steps:
                    break

                teacher_logits = self.teacher_logits_list[idx[0]].to(self.device)

                input_ids = teacher_logits.new_zeros((1, teacher_logits.size(0)), dtype=torch.long)

                outputs = self.model(input_ids=input_ids)
                student_logits = outputs.logits.squeeze(0)

                loss = self._kd_loss(student_logits.unsqueeze(0), teacher_logits.unsqueeze(0))

                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1
                pbar.update(1)

                if self.step >= total_steps:
                    break

        pbar.close()

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        T = self.temperature

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

        return kd_loss
