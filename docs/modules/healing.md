# Módulo Healing — Reparação de Modelos via Fine-Tuning

## Visão Geral

O módulo `llm_flux/adapters/healing/` implementa a estratégia de **reparação de modelos compressados** através de fine-tuning. O objetivo é recuperar qualidade perdida durante a compressão (especialmente quantização) sem reintroduzir o custo de memória/velocidade do modelo original.

O adapter principal (`HFTrainerAdapter`) utiliza o **HuggingFace Trainer** com suporte a **LoRA/QLoRA** via biblioteca PEFT.

---

## 1. `hf_trainer.py` — HFTrainerAdapter

### 1.1 Visão Geral

**Responsabilidade:** Fine-tuning de modelos de linguagem compressados usando HuggingFace Trainer, com suporte opcional a LoRA para eficiência de memória.

**Fluxo:**
```
compress(model)
    ↓
HFTrainerAdapter.heal(model)
    ↓
1. Carregar dataset (HF Hub ou local)
2. Tokenizar textos
3. Materializar streaming datasets (se aplicável)
4. Aplicar LoRA via PEFT (se config.lora não for None)
5. Treinar com transformers.Trainer
6. Retornar modelo fine-tuned
```

### 1.2 Construtor

```python
def __init__(self, config: HealingConfig, tokenizer: Any = None) -> None:
    self.config = config
    self.tokenizer = tokenizer  # expõe get_tokenizer() após load()
```

**Nota:** O tokenizer é opcional no construtor, mas **obrigatório** em `heal()` — verificado em runtime:
```python
actual_tokenizer = self.tokenizer.get_tokenizer() if hasattr(self.tokenizer, "get_tokenizer") else self.tokenizer
if actual_tokenizer is None:
    raise ValueError("A tokenizer must be provided to HFTrainerAdapter (via constructor) to heal the model.")
```

### 1.3 `heal(model)` — Fluxo Detalhado

#### Passo 1: Carregamento do Dataset

```python
def _load_dataset(config: DatasetConfig) -> Any:
    from pathlib import Path
    if Path(config.source).exists():
        return LocalDatasetAdapter(config).load()
    return HFDatasetAdapter(config).load()
```

**Nota:** Mesma convenção de auto-detecção usada nos compression adapters.

#### Passo 2: Tokenização

```python
actual_tokenizer = self.tokenizer.get_tokenizer() if hasattr(self.tokenizer, "get_tokenizer") else self.tokenizer
if actual_tokenizer.pad_token is None:
    actual_tokenizer.pad_token = actual_tokenizer.eos_token

def tokenize_function(examples):
    return actual_tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_len
    )

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

**Coluna esperada:** `"text"` — texto em linguagem natural para language modeling.

#### Passo 3: Max Sequence Length

```python
max_seq_len = cfg.max_seq_length
if max_seq_len is None:
    max_seq_len = self._get_max_seq_length(model, actual_tokenizer)
    # Usa tokenizer.model_max_length ou model.config.max_position_embeddings/n_positions/seq_length
else:
    max_seq_len = configured value
```

**Fallback:** `_get_max_seq_length()` tenta (1) `tokenizer.model_max_length`, (2) `model.config.max_position_embeddings`, (3) `model.config.n_positions`, (4) `model.config.seq_length`, (5) default=512.

#### Passo 4: Materialização de Streaming Datasets

```python
if isinstance(tokenized_dataset, IterableDataset):
    tokenized_dataset = Dataset.from_list(list(tokenized_dataset))
```

**Problema:** Streaming datasets (`IterableDataset`) são lazy — `.map()` apenas configura uma pipeline, mas não executa. Durante o DataLoader, os dados seriam re-streamados da internet repetidamente, causando hangs.

**Solução:** Materializa em memória converting para `Dataset` regular antes do treino.

#### Passo 5: Validação de Tamanho do Dataset

```python
min_needed = per_device_train_batch_size * gradient_accumulation_steps
if len(tokenized_dataset) < min_needed:
    logger.warning("Dataset has only N samples, but batch_size * grad_accum requires M for one optimizer step.")
```

**Aviso apenas** — não impede execução.

#### Passo 6: Aplicação de LoRA (se configurado)

```python
if cfg.lora:
    from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType

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
```

**Parâmetros LoRA padrão** (em `LoRAConfig`):
```python
r: int = 8
lora_alpha: int = 32
target_modules: list[str] = ["q_proj", "v_proj"]
lora_dropout: float = 0.05
bias: str = "none"
task_type: str = "CAUSAL_LM"
```

**Se `config.lora = None`:** Treina todos os parâmetros (full fine-tuning).

#### Passo 7: HuggingFace Trainer

```python
training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    max_steps=cfg.max_steps,
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    fp16=cfg.fp16,
    save_steps=cfg.save_steps,
    logging_steps=cfg.logging_steps,
    report_to="none",  # disable wandb/etc
    remove_unused_columns=True,
)

trainer_cls = cfg.trainer_class or self._default_trainer()  # transformers.Trainer
trainer = trainer_cls(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
return model
```

**Configurações importantes:**
- `fp16=True` por padrão — acelera treino em GPUs com tensor cores
- `report_to="none"` — desabilita integração com wandb, mlflow, etc.
- `remove_unused_columns=True` — evita erros com colunas extras no dataset
- `trainer_class` configurável — permite injetar Trainer customizado (duck-typed)

### 1.4 Output

**Diretório de saída padrão:** `./healing_output/`

Após training, checkpoints são salvos a cada `save_steps` nesse diretório.

---

## 2. LoRAConfig — Parâmetros de LoRA

**Arquivo:** `llm_flux/core/healing.py`

```python
class LoRAConfig(BaseModel):
    r: int = 8                    # rank da decomposição baixa
    lora_alpha: int = 32         # fator de escala (alpha = 2*r é comum)
    target_modules: list[str] = ["q_proj", "v_proj"]  # matrizes para injetar LoRA
    lora_dropout: float = 0.05    # dropout nas camadas LoRA
    bias: str = "none"            # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"
```

**Módulos-alvo típicos para modelos transformer:**
- `q_proj` — projeção de queries
- `v_proj` — projeção de values
- Opcionalmente: `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## 3. HealingConfig — Configuração Completa

**Arquivo:** `llm_flux/core/healing.py`

```python
class HealingConfig(BaseModel):
    name: str = "model-healing"
    description: str = ""

    # Dataset
    dataset: DatasetConfig

    # Training knobs
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "./healing_output"
    save_steps: int = 100
    logging_steps: int = 10
    max_seq_length: int | None = None  # None → auto-detect

    # LoRA
    lora: LoRAConfig | None = None      # None → full fine-tune

    # Advanced
    trainer_class: Any | None = Field(None, exclude=True)
```

---

## 4. Exports

```python
from llm_flux.adapters.healing import HFTrainerAdapter
from llm_flux.core.healing import HealingConfig, LoRAConfig, HealingPort
```

---

## 5. Contrato — HealingPort

**Interface em `core/healing.py`:**

```python
class HealingPort(ABC):
    config: HealingConfig

    @abstractmethod
    def heal(self, model: Any) -> Any:
        """
        Fine-tune/repara model e retorna versão healing.

        O objeto retornado substitui o modelo no estado do DAG,
        então passos subsequentes recebem automaticamente o modelo healing.
        """

    @property
    def label(self) -> str:
        return self.config.name
```

---

## 6. Fluxo no Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE                                  │
│                                                                  │
│  Load Model → Profile → Compress → Profile → Heal → Profile     │
│                                                  │              │
│                                                  ↓              │
│                                    HFTrainerAdapter.heal(model) │
│                                    1. Load dataset             │
│                                    2. Tokenize                  │
│                                    3. Materialize (if streaming)│
│                                    4. Apply LoRA (if configured)│
│                                    5. Train                     │
│                                                  ↓              │
│                                    healed_model (com LoRA ou FT)│
│                                                  │              │
│                                                  ↓              │
│                                    Post-Healing Profiling        │
└─────────────────────────────────────────────────────────────────┘
```

**Thread de estado no executor (`dag/executor.py`):**
```python
case "heal":
    model = port.heal(model)
    logger.info(f"  ✅ Healing complete: {port.label}")
```

---

## 7. Riscos e Limitações

1. **Coluna "text" hardcoded** — o dataset deve ter coluna "text" para tokenização. Não há fallback para outros nomes de coluna.

2. **`remove_unapped_columns=True`** — remove "text" após tokenização, mas se o dataset original não tiver "text", o mapeamento pode falhar silenciosamente.

3. **`fp16=True` por padrão** — pode falhar em CPUs ou GPUs antigas sem suporte FP16. Não há detecção automática.

4. **Sem Early Stopping** — `TrainingArguments` não configura `early_stopping`. Pode overfit.

5. **`trainer_class` duck-typed** — não há validação de interface. Se o `trainer_class` não tiver a assinatura esperada, falhará em runtime.

6. **`max_seq_length` auto-detect** — heurística pode falhar para modelos com estruturas diferentes.

7. **Streaming dataset materialization** — `Dataset.from_list(list(dataset))` pode consumir toda a memória se o dataset for grande.

8. **Sem suporte a evaluation** — o Trainer é configurado apenas para training, sem evaluation.

9. **`save_steps` vs. `max_steps`** — checkpoints intermediários podem não ser necessários se o objetivo final é apenas o modelo final.

10. **LoRA `target_modules` default** — `["q_proj", "v_proj"]` funciona para Llama/Qwen, mas pode não ser ideal para todos os modelos.

---

## 8. Exemplo de Uso

```python
from llm_flux.adapters.healing import HFTrainerAdapter
from llm_flux.core.healing import HealingConfig, LoRAConfig
from llm_flux.datasets.port import DatasetConfig

healer = HFTrainerAdapter(
    config=HealingConfig(
        name="lora-recovery",
        description="LoRA fine-tuning (r=16) para recuperar perplexidade",
        dataset=DatasetConfig(source="tatsu-lab/alpaca", max_samples=1000),
        max_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        lora=LoRAConfig(r=16, lora_alpha=64),
    ),
    tokenizer=model_handle,  # HFModelHandle com get_tokenizer()
)

healed_model = healer.heal(compressed_model)
```
