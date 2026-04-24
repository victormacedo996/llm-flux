"""
adapters/healing/knowledge_distillation.py — Knowledge Distillation adapter for model healing.

Uses the original uncompressed model as teacher and the compressed model as student.
Supports two modes:
  - precompute (default): teacher logits are precomputed and saved to disk,
    then teacher is unloaded before student training (memory efficient).
  - online: teacher remains in memory during student training (higher memory).
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from llm_flux.core.healing import DistillationConfig, HealingPort, SaveFormat
from llm_flux.core.model import CompressedModelHandle
from llm_flux.datasets.huggingface import HFDatasetAdapter
from llm_flux.datasets.local import LocalDatasetAdapter
from llm_flux.datasets.port import DatasetConfig

KDPair = namedtuple("KDPair", ["input_ids", "attention_mask", "teacher_logits"])


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
        self._pad_token_id = (
            actual_tokenizer.pad_token_id if actual_tokenizer.pad_token_id is not None else 0
        )

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

        kd_pairs = self._load_or_compute_kd_pairs(
            teacher, tokenized_dataset, actual_tokenizer, cfg.per_device_train_batch_size
        )

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

        logger.info(f"  🏋️  Starting KD training for {cfg.kd_max_steps} steps...")

        kd_trainer = _KDTrainer(
            model=model,
            kd_pairs=kd_pairs,
            tokenizer=actual_tokenizer,
            temperature=cfg.temperature,
            alpha=cfg.alpha,
            device=next(model.parameters()).device,
            max_seq_length=max_seq_len,
            max_steps=cfg.kd_max_steps,
        )

        kd_trainer.train()

        logger.info("  ✅ Knowledge Distillation complete.")

        logger.info("  🔄 Setting model to eval mode...")
        model.eval()

        if cfg.save_format == SaveFormat.AUTO:
            should_merge = cfg.lora is not None
        else:
            should_merge = cfg.save_format == SaveFormat.MERGED

        if should_merge:
            if hasattr(kd_trainer, "optimizer") and kd_trainer.optimizer is not None:
                kd_trainer.optimizer.zero_grad()

            if hasattr(model, "cache_clear"):
                model.cache_clear()

            if hasattr(model, "merge_and_unload"):
                logger.info("  🔀 Merging LoRA weights into base model...")
                model = model.merge_and_unload()
                model.eval()
            else:
                logger.warning("  ⚠️  merge_and_unload() not available, saving as-is.")

        if cfg.clear_cache_before_save:
            logger.info("  🧹 Clearing caches before save...")
            if hasattr(model, "cache_clear"):
                model.cache_clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            model = model.cpu()

        logger.info(f"  💾 Saving healed student model to {cfg.output_dir}...")
        model.save_pretrained(cfg.output_dir)
        self._tokenizer.save_pretrained(cfg.output_dir)

        if hasattr(kd_trainer, "optimizer") and kd_trainer.optimizer is not None:
            del kd_trainer.optimizer
            kd_trainer.optimizer = None

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

    def _load_or_compute_kd_pairs(
        self,
        teacher: Any,
        tokenized_dataset: Any,
        tokenizer: Any,
        batch_size: int,
    ) -> list[KDPair]:
        """Load cached KD pairs or compute new ones.

        Supports both old format (list[torch.Tensor]) and new format (list[KDPair]).
        For old format, reconstructs KDPair by re-tokenizing the dataset.
        """
        new_cache_path = Path(self.config.teacher_logits_output_dir) / "teacher_kd_pairs.pt"
        old_cache_path = Path(self.config.teacher_logits_output_dir) / "teacher_logits.pt"
        new_cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Try loading new format first
        if self.config.precompute_teacher_logits and new_cache_path.exists():
            try:
                kd_pairs = torch.load(new_cache_path)
                if kd_pairs and isinstance(kd_pairs[0], KDPair):
                    logger.info(f"  📂 Loading cached KD pairs from {new_cache_path}")
                    return kd_pairs
            except Exception:
                pass

        # Try loading old format for backward compatibility
        if self.config.precompute_teacher_logits and old_cache_path.exists():
            try:
                logger.info(f"  📂 Loading old-format cache from {old_cache_path}")
                old_logits_list = torch.load(old_cache_path)
                if isinstance(old_logits_list, list) and len(old_logits_list) > 0:
                    if not isinstance(old_logits_list[0], torch.Tensor):
                        raise ValueError("Old cache format invalid")

                    logger.info("  🔄 Reconstructing KD pairs from old cache (re-tokenizing)...")

                    kd_pairs = self._compute_teacher_logits(
                        teacher, tokenized_dataset, tokenizer, batch_size
                    )

                    torch.save(kd_pairs, new_cache_path)
                    logger.info(f"  💾 Saved new-format KD pairs to {new_cache_path}")

                    old_cache_path.unlink(missing_ok=True)
                    logger.info(f"  🗑️  Removed old cache file: {old_cache_path}")

                    return kd_pairs
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load old cache: {e}, recomputing...")

        # Compute fresh - delete old cache to ensure clean state
        if new_cache_path.exists() or old_cache_path.exists():
            logger.info("  🗑️  Removing old KD cache to ensure clean state...")
            import shutil

            shutil.rmtree(self.config.teacher_logits_output_dir, ignore_errors=True)
            new_cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("  🔢 Computing teacher logits...")
        kd_pairs = self._compute_teacher_logits(teacher, tokenized_dataset, tokenizer, batch_size)

        if self.config.precompute_teacher_logits:
            torch.save(kd_pairs, new_cache_path)
            logger.info(f"  💾 Saved KD pairs to {new_cache_path}")

        return kd_pairs

    def _compute_teacher_logits(
        self,
        teacher: Any,
        tokenized_dataset: Any,
        tokenizer: Any,
        batch_size: int,
    ) -> list[KDPair]:
        teacher.eval()
        kd_pairs: list[KDPair] = []

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
                    kd_pairs.append(
                        KDPair(
                            input_ids=input_ids[j].cpu(),
                            attention_mask=attention_mask[j].cpu(),
                            teacher_logits=logits[j].cpu(),
                        )
                    )

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"    Processed {(i + 1) * batch_size} / {len(tokenized_dataset)} samples"
                    )

        return kd_pairs


class _KDTrainer:
    """
    Custom training loop for knowledge distillation.

    Uses real input_ids and attention_mask from KDPair to train the student model
    with proper masking, avoiding the broken zero-sequence approach.
    """

    def __init__(
        self,
        model: Any,
        kd_pairs: list[KDPair],
        tokenizer: Any,
        temperature: float,
        alpha: float,
        device: torch.device,
        max_seq_length: int = 512,
        max_steps: int = 100,
    ) -> None:
        self.model = model
        self.kd_pairs = kd_pairs
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        self.max_seq_length = max_seq_length
        self.max_steps = max_steps
        self.optimizer = None
        self.step = 0

    def train(self) -> None:
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup

        self.model.train()

        dataloader = DataLoader(
            self.kd_pairs,
            batch_size=1,
            shuffle=True,
        )

        total_steps = min(len(self.kd_pairs), self.max_steps)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        from tqdm import tqdm

        pbar = tqdm(total=total_steps, desc="KD Training")

        while self.step < total_steps:
            for kd_pair in dataloader:
                if self.step >= total_steps:
                    break

                input_ids = kd_pair.input_ids.squeeze(0).to(self.device)
                attention_mask = kd_pair.attention_mask.squeeze(0).to(self.device)
                teacher_logits = kd_pair.teacher_logits.squeeze(0).to(self.device)

                seq_len = input_ids.size(0)
                if seq_len > self.max_seq_length:
                    input_ids = input_ids[: self.max_seq_length]
                    attention_mask = attention_mask[: self.max_seq_length]
                    teacher_logits = teacher_logits[: self.max_seq_length, :]
                elif seq_len < self.max_seq_length:
                    pad_len = self.max_seq_length - seq_len
                    input_ids = F.pad(
                        input_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0
                    )
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
                    teacher_logits = F.pad(teacher_logits, (0, 0, 0, pad_len), value=0)

                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = outputs.logits.squeeze(0)

                loss = self._kd_loss(student_logits, teacher_logits)

                loss.backward()
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1
                pbar.update(1)

                if self.step >= total_steps:
                    break

        pbar.close()
        self.model.eval()

    def _kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        T = self.temperature

        teacher_seq_len, teacher_vocab_size = teacher_logits.shape
        student_seq_len, student_vocab_size = student_logits.shape

        if student_seq_len != teacher_seq_len or student_vocab_size != teacher_vocab_size:
            if student_seq_len > teacher_seq_len:
                student_logits = student_logits[:teacher_seq_len, :]
            elif student_seq_len < teacher_seq_len:
                student_logits = F.pad(
                    student_logits, (0, 0, 0, teacher_seq_len - student_seq_len), value=0
                )

            if student_vocab_size > teacher_vocab_size:
                student_logits = student_logits[:, :teacher_vocab_size]
            elif student_vocab_size < teacher_vocab_size:
                student_logits = F.pad(
                    student_logits, (0, teacher_vocab_size - student_vocab_size), value=0
                )

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T * T)

        return kd_loss
