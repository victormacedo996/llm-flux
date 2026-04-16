# Módulo Datasets — Carregamento de Dados

## Visão Geral

O módulo `llm_flux/datasets/` implementa a abstração para **carregamento de datasets** de qualsquer fonte. Segue a mesma arquitetura de ports/adapters usada no resto do framework.

```
DatasetPort (contract — interface abstrata)
        ↑
        │
   ┌────┴────────────┐
   │                 │
HFDatasetAdapter  LocalDatasetAdapter
   (HF Hub)         (arquivos locais)
```

---

## 1. `port.py` — DatasetPort e DatasetConfig

### 1.1 `DatasetConfig`

**Arquivo:** `llm_flux/datasets/port.py`

**Responsabilidade:** Descritor validado para uma fonte de dataset.

**Campos:**
```python
source: str                   # HF dataset name OU path local
split: str = "train"          # split do dataset
subset: str | None = None    # nome do subset/config (HF datasets com configs)
max_samples: int | None = None  # None = todas; int = primeiras N
streaming: bool = False       # se True, usa streaming mode
seed: int = 42                # mantido por retrocompatibilidade (não usado em first-N mode)
```

**Validação:**
```python
@model_validator(mode="after")
def _max_samples_positive(self) -> DatasetConfig:
    if self.max_samples is not None and self.max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    return self
```

**Detecção de tipo de fonte:**
- Se `Path(source).exists()` → local path → `LocalDatasetAdapter`
- Caso contrário → HF dataset name → `HFDatasetAdapter`

### 1.2 `DatasetPort`

**Responsabilidade:** Interface abstrata para qualquer loader de dataset.

**Contrato:**
```python
class DatasetPort(ABC):
    config: DatasetConfig

    @abstractmethod
    def load(self) -> object:
        """Retorna dataset pronto para uso pelo Trainer."""
```

**Retorno esperado:** Objeto compatível com `transformers.Trainer` — tipicamente `datasets.Dataset` ou `torch.utils.data.Dataset`.

---

## 2. `huggingface.py` — HFDatasetAdapter

### 2.1 Responsabilidade

Carregar datasets do **HuggingFace Hub** via `datasets.load_dataset()`.

### 2.2 `load()`

```python
def load(self) -> object:
    kwargs = {
        "path": self.config.source,
        "split": self.config.split,
        "streaming": self.config.streaming,
    }
    if self.config.subset:
        kwargs["name"] = self.config.subset

    dataset = load_dataset(**kwargs)

    if self.config.max_samples is not None:
        if self.config.streaming:
            dataset = dataset.take(self.config.max_samples)
        else:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

    return dataset
```

**Tratamento de `max_samples`:**
- Streaming mode: usa `.take()` (lazy)
- Regular mode: usa `.select()` (sliceing)

**Tratamento de `subset`:**
- Usado para datasets com múltiplas configs (e.g., `"allenai/ai2_arc"` tem `"ARC-Challenge"`, `"ARC-Easy"`)

**Tratamento de splits:** O split fornecido em `config.split` é passado diretamente para `load_dataset()`. Se o split não existir, a biblioteca levanta exceção.

---

## 3. `local.py` — LocalDatasetAdapter

### 3.1 Responsabilidade

Carregar datasets de **arquivos locais** — JSON, JSONL, CSV ou Parquet.

### 3.2 Formatos Suportados

Detecção por extensão:
| Extensão | Formato |
|---|---|
| `.json`, `.jsonl` | `json` |
| `.csv` | `csv` |
| `.parquet` | `parquet` |

### 3.3 `load()`

```python
def load(self) -> object:
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
        dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

    return dataset
```

**Nota:** Para JSON/JSONL, usa `data_format="json"` — a biblioteca do datasets interpreta o arquivo como um dataset de JSON lines.

---

## 4. Exports

```python
from llm_flux.datasets import DatasetConfig, DatasetPort, HFDatasetAdapter, LocalDatasetAdapter
```

---

## 5. Fluxo de Uso

### 5.1 No Compression Adapter (ex: GPTQAdapter)

```python
from llm_flux.datasets.port import DatasetConfig
from llm_flux.datasets.local import LocalDatasetAdapter
from llm_flux.datasets.huggingface import HFDatasetAdapter
from pathlib import Path

cfg = self.config.calibration_dataset
if Path(cfg.source).exists():
    dataset = LocalDatasetAdapter(cfg).load()
else:
    dataset = HFDatasetAdapter(cfg).load()
```

### 5.2 No Healing Adapter (HFTrainerAdapter)

```python
def _load_dataset(config: DatasetConfig) -> Any:
    if Path(config.source).exists():
        return LocalDatasetAdapter(config).load()
    return HFDatasetAdapter(config).load()
```

### 5.3 No DepthPruningAdapter

Mesmo padrão — `_get_calibration_inputs()` usa a mesma lógica de auto-detecção.

---

## 6. Detecção Automática de Fonte

A convenção estabelecida é que todos os adapters que precisam de datasets usam:

```python
from pathlib import Path
if Path(config.source).exists():
    adapter = LocalDatasetAdapter(config)
else:
    adapter = HFDatasetAdapter(config)
dataset = adapter.load()
```

Isso permite que o usuário forneça qualquer path local ou nome de HF dataset sem alterar o código.

---

## 7. Casos de Uso Comuns

### 7.1 Dataset de HF Hub

```python
DatasetConfig(
    source="tatsu-lab/alpaca",
    split="train",
    max_samples=1000,
)
```

### 7.2 Dataset Local JSONL

```python
DatasetConfig(
    source="/path/to/my/custom_dataset.jsonl",
    split="train",
)
```

### 7.3 Dataset com Subset

```python
DatasetConfig(
    source="allenai/ai2_arc",
    subset="ARC-Challenge",  # nome do config
    split="test",
)
```

### 7.4 Dataset de HuggingFace com Streaming

```python
DatasetConfig(
    source="allenai/c4",
    subset="en",
    split="train",
    streaming=True,
    max_samples=10000,
)
```

---

## 8. Riscos e Limitações

1. **`seed` não é usado** — O campo `seed` existe em `DatasetConfig` mas não é usado em nenhum adapter (apenas para "backward compatibility"). Se for útil para reprodutibilidade, deveria ser aplicado em `.shuffle()`.

2. **Fallback de split não documentado** — Apenas `HFDatasetAdapter._load_dataset()` (no benchmarker, não no dataset adapter) tem fallback de split. O `LocalDatasetAdapter` não tem fallback.

3. **Coluna "text" hardcoded** — `HFTrainerAdapter.heal()` espera coluna `"text"` no dataset. Se o dataset não tiver essa coluna, a tokenização falha.

4. **Streaming + `max_samples`** — `.take()` é lazy em streaming mode, o que é eficiente. Porém, se o dataset for iterado múltiplas vezes (e.g., em múltiplos epochs), cada iteração vai re-stream. A materialização em `HFTrainerAdapter` resolve isso.

5. **`remove_unused_columns=True` no Trainer** — remove "text" após tokenização. Se o adapter de healing tentasse usar alguma coluna do dataset original que não fosse preservada, haveria erro.

6. **JSON/JSONL sem validação de schema** — O formato é inferido pela extensão. Arquivos malformados podem produzir erros obscuros.

7. **`LocalDatasetAdapter` não detecta HF datasets** — Se alguém passar um path que não existe, mas que seria um HF dataset name válido, o adapter lançará `FileNotFoundError` em vez de delegar para `HFDatasetAdapter`.

8. **Sem cache de dataset** — Cada chamada a `load()` recarrega o dataset. Para múltiplos adapters usando o mesmo dataset, pode haver redundância.

---

## 9. Hipóteses

1. **Datasets para calibração de quantização** — Todos os compression adapters que usam `calibration_dataset` tokenizam textos (para GPTQ e DepthPruning) ou usam textos crus (AWQ). Não há suporte para datasets estruturados (QA,闲聊).

2. **Datasets para fine-tuning** — `HFTrainerAdapter` assume column `"text"` com texto em linguagem natural para causal language modeling.

3. **`max_samples` sempre das primeiras N linhas** — Não há aleatorização. `seed` não é usado.

4. **Split padrão é "train"** — Razoável para fine-tuning, mas pode não ser correto para evaluation datasets (muitos usam "test" ou "validation").
