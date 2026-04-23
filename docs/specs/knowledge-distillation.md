# Feature Spec — Knowledge Distillation as Healing Step

## Metadata
- Spec ID: `SPEC-KD-001`
- Title: Knowledge Distillation for Model Healing
- Author: Victor Macedo
- Date: 2026-04-22
- Status: `implemented`
- Related decisions: N/A

## 1. Problem statement

After compression (quantization, pruning, etc.), models lose quality. The framework supports healing via fine-tuning (LoRA/QLoRA), but some scenarios require recovering knowledge from the original uncompressed model rather than just re-training on a dataset. Knowledge Distillation (KD) addresses this by using the original model as a "teacher" that provides soft probability targets (logits) to guide the compressed "student" model's training.

## 2. Goals
- Provide KD as a first-class healing strategy in the pipeline
- Support memory-efficient workflow via pre-computed teacher logits
- Integrate seamlessly with existing LoRA/QLoRA infrastructure
- Keep `run_pipeline()` interface unchanged

## 3. Non-goals
- Multi-teacher distillation (single teacher only)
- Intermediate-layer distillation
- End-to-end differentiation (no gradient through teacher)

## 4. Scope
### In scope
- `DistillationConfig` in `core/healing.py`
- `KnowledgeDistillationAdapter` in `adapters/healing/knowledge_distillation.py`
- Precompute and online KD modes
- LoRA/QLoRA integration during student training
- Spec document and healing docs updates
- Validation example with random Llama

### Out of scope
- Custom trainer subclassing beyond overriding `compute_loss`
- Dynamic dataset re-loading during training
- Streaming teacher logits (precompute only for now)

## 5. User scenarios
- **Scenario A — Memory-constrained KD**: User compresses a large model and wants to heal it via KD but has limited GPU memory. They enable `precompute_teacher_logits=True`. Teacher model is loaded, logits are saved to disk, teacher is unloaded, then student is trained.

- **Scenario B — Quick KD**: User has enough memory to keep both models loaded. They set `precompute_teacher_logits=False`. Teacher and student are both in memory during training.

- **Scenario C — KD + LoRA**: User combines KD with LoRA for parameter-efficient student training, reducing memory further.

## 6. Proposed solution

### Architecture
- `KnowledgeDistillationAdapter` implements `HealingPort`
- Constructor: `__init__(config, tokenizer=None, teacher_model_handle=None)`
- `heal(student_model)` — trains student using teacher logits

### Two KD modes
1. **Precompute mode (default)**: `precompute_teacher_logits=True`
   - Load teacher → generate logits for all dataset samples → save `.pt` to disk
   - Unload teacher model (free GPU memory)
   - Train student using cached logits

2. **Online mode**: `precompute_teacher_logits=False`
   - Both teacher and student remain in memory during training
   - Teacher logits computed on-the-fly per batch

### KD Loss
```
loss = alpha * KL_div(teacher_soft, student_soft) / T^2
     + (1 - alpha) * CE(student_logits, labels)
```

### Pipeline flow
```
Load Model → Profile → Compress → Profile → KD Heal → Profile
                        ↑
                  teacher_model_handle must be loaded
                  before KD Heal step runs
```

## 7. Decision choices
- **Choice**: Teacher model reference in adapter
  - Selected: `teacher_model_handle` passed via `__init__`
  - Rationale: Explicit, no executor modification needed

- **Choice**: Precompute as default
  - Selected: `precompute_teacher_logits=True`
  - Rationale: Memory efficiency is critical for dissertation experiments

- **Choice**: LoRA integration
  - Selected: Via `config.lora: LoRAConfig | None`
  - Rationale: Reuses existing LoRA infrastructure

## 8. Technical impact
- Modules affected: `core/healing.py`, `adapters/healing/knowledge_distillation.py`, `adapters/healing/__init__.py`
- New dependencies: None (uses existing `transformers`, `torch`, `datasets`)
- Data model changes: `DistillationConfig` added to `core/healing.py`

## 9. Acceptance criteria
- [x] `DistillationConfig` validates `teacher_model_id` (via handle), `dataset`, `temperature`, `alpha`
- [x] Precompute mode generates `.pt` teacher logits, then unloads teacher before student training
- [x] Online mode keeps teacher in memory during student training
- [x] `alpha=0` → standard supervised training, `alpha=1` → pure KD
- [x] LoRA is applied to student model during KD training
- [x] Example runs end-to-end with random Llama
- [x] Existing `test_fine_tune.py` continues to work unchanged

## 10. Validation plan
- Run `examples/kd_random_llama.py` end-to-end
- Verify teacher model is unloaded after pre-compute (check GPU memory)
- Verify `alpha=0` and `alpha=1` produce different loss curves

## 11. Rollout plan
- Step 1: Implement `DistillationConfig` + `KnowledgeDistillationAdapter`
- Step 2: Create `examples/kd_random_llama.py`
- Step 3: Update `docs/modules/healing.md`
- Step 4: Run full validation

## 12. Risks and mitigations
- **Risk**: Precompute mode saves logits to disk that can become stale if dataset changes
  - **Mitigation**: Include a cache-busting check (recompute if dataset source differs)

## 13. Open questions
- None currently
