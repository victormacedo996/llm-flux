# Módulo Core — Ports e Interfaces

## Visão Geral

O módulo `core/` define as **portas** (interfaces abstratas) do framework LLM-Flux. Cada port é uma classe abstrata (ABC) que estabelece um **contrato** entre o domínio puro do framework e suas implementações concretas nos adapters. Os ports são **puramente Python** — nenhum import de `torch`, `transformers` ou outras bibliotecas de ML é permitido neste módulo.

Esta decisão arquitetural permite:
- Testar o domínio sem dependências de ML
- Manter as interfaces stables mesmo quando bibliotecas externas mudam
- Garantir que novos adapters sigam contratos bem definidos

---

## 1. `model.py` — ModelHandle e ModelSource

### 1.1 `ModelSource`

**Arquivo:** `llm_flux/core/model.py`

**Responsabilidade:** Descrever de forma validada a origem de um modelo de ML.

**Campos:**
```python
identifier: str                    # HF repo id ("org/repo") ou caminho local
revision: str = "main"            # git revision no HF Hub
trust_remote_code: bool = False   # permite código customizado (e.g. tokenizer)
cache_dir: str | None = None      # diretório de cache do HF
```

**Validação:**
- Se `Path(identifier).exists()` → tratado como diretório local
- Caso contrário → validado como HF repo id (deve conter `/`)
- Lança `ValueError` se nenhum dos dois for válido

**Propriedade derivada:**
```python
is_local: bool  # True se identifier é caminho existente
```

### 1.2 `ModelHandle`

**Arquivo:** `llm_flux/core/model.py`

**Responsabilidade:** Abstração universal para carregar e descarregar modelos.

**Contrato:**
```python
@abstractmethod def load() -> object:
    """Carrega e retorna o modelo cru (e.g. nn.Module).
    Deve mover o modelo para o device apropriado antes de retornar."""

@abstractmethod def unload() -> None:
    """Libera recursos (GPU memory, file handles, etc.)."""

@property @abstractmethod def name() -> str:
    """Label legível para nós do DAG e logs."""
```

**Propriedades expostas:**
```python
source: ModelSource  # configuração da fonte do modelo
```

**Nota:** O retorno de `load()` é tipicamente um `torch.nn.Module`, mas o tipo exato não é especificado no contrato para permitir diferentes implementações.

---

## 2. `compression.py` — CompressionPort e CompressionConfig

### 2.1 `CompressionConfig`

**Arquivo:** `llm_flux/core/compression.py`

**Responsabilidade:** Base para configurações de algoritmos de compressão. Subclassada para cada técnica.

**Campos obrigatórios:**
```python
name: str              # identificador da técnica
description: str = ""  # descrição legível
```

**Exemplo de subclassing:**
```python
class GPTQConfig(CompressionConfig):
    bits: int = Field(4, ge=2, le=8)           # largura de quantização
    group_size: int = Field(128)                # grupo de quantização
    desc_act: bool = False                     # activation order
    calibration_samples: int = 128            # amostras de calibração
    calibration_dataset: DatasetConfig        # dataset de calibração
```

### 2.2 `CompressionNotSupportedError`

**Arquivo:** `llm_flux/core/compression.py`

**Responsabilidade:** Exceção lançada quando uma técnica de compressão é incompatível com a arquitetura do modelo.

**Tratamento:** O `PipelineExecutor` captura esta exceção, loga erro descritivo e aborta o pipeline.

### 2.3 `CompressionPort`

**Responsabilidade:** Interface abstrata para qualquer técnica de compressão.

**Contrato:**
```python
config: CompressionConfig  # configuração específica da técnica

@abstractmethod def compress(model: Any) -> Any:
    """
    Aplica compressão e retorna o modelo (possivelmente novo).

    Args:
        model: Modelo cru retornado por ModelHandle.load()
               ou por CompressionPort.compress() anterior.

    Returns:
        Modelo compressado (in-place ou novo objeto).

    Raises:
        CompressionNotSupportedError: Se a técnica não pode processar o modelo.
    """

@property def label() -> str:
    """Short label para nós do DAG e log lines."""
    return self.config.name
```

---

## 3. `profiling.py` — ProfilingPort, ProfilingConfig e Tipos de Resultado

### 3.1 Tipos de Métricas

#### `MemoryMetrics`

**Arquivo:** `llm_flux/core/profiling.py`

```python
peak_gpu_mb: float | None = None  # pico de memória GPU em MB
peak_cpu_mb: float | None = None  # pico de memória CPU em MB
```

#### `LatencyMetrics`

```python
mean_ms: float
std_ms: float = 0.0
min_ms: float = 0.0
p5_ms: float = 0.0
p50_ms: float
p95_ms: float
p99_ms: float
max_ms: float = 0.0
```

Percentis: p5, p50, p95, p99. Todas as unidades são **milissegundos**.

#### `AccuracyMetrics`

```python
perplexity: float | None = None   # para benchmarks de perplexidade
task_score: float | None = None   # para benchmarks de acurácia
task_name: str | None = None      # nome do benchmark (e.g. "lambada")
```

**Nota:** Os campos são mutualmente exclusivos na prática — ou perplexity ou task_score é definido, nunca ambos.

### 3.2 `ProfilingResult`

**Responsabilidade:** Resultado estruturado de uma única execução de profiling. **Fonte de verdade** para qualquer medição na dissertação.

**Campos:**
```python
profiler_name: str                # nome do profiler
model_label: str                  # label do modelo (e.g. tipo do modelo)
pipeline_stage: str               # label do estágio (e.g. "Baseline", "Post-GPTQ")
timestamp: datetime = factory     # UTC now

memory: MemoryMetrics = factory
latency: LatencyMetrics
accuracy: AccuracyMetrics | None = None

# Raw dumps — preserva TODOS os dados sem poluir campos principais
extra: dict[str, Any] = factory   # chaves: "hardware", "llm_profile", "inference", "benchmarks"
```

**Métodos:**
```python
def summary() -> str:
    """One-liner para log e tabelas de dissertação.
    Formato: '[{pipeline_stage}] Latency p95={X} ms | GPU peak={Y} MB | PPL={Z}'"""
```

**Estratégia de serialização:** Dados brutos são armazenados em `extra` para preservar informação completa no JSON export, enquanto campos top-level são mantidos limpos para geração de tabelas Markdown.

### 3.3 `ProfilingConfig`

**Responsabilidade:** Base para configurações de profiling.

**Campos:**
```python
name: str
description: str = ""
num_warmup_runs: int = 3
num_benchmark_runs: int = 10
```

### 3.4 `ProfilingPort`

**Responsabilidade:** Interface para qualquer estratégia de profiling.

**Contrato:**
```python
config: ProfilingConfig

@abstractmethod def profile(model: Any, stage_label: str) -> ProfilingResult:
    """
    Executa profiling contra model e retorna ProfilingResult completo.

    Args:
        model: Modelo a ser profileado (nn.Module ou equivalente).
        stage_label: Label descritivo do estágio do pipeline
                     (e.g. "After GPTQ-4bit"). Armazenado verbatim em
                     ProfilingResult.pipeline_stage.
    """

@property def label() -> str:
    return self.config.name
```

---

## 4. `healing.py` — HealingPort, HealingConfig e LoRAConfig

### 4.1 `LoRAConfig`

**Responsabilidade:** Hyperparâmetros LoRA para fine-tuning eficiente.

**Campos:**
```python
r: int = 8                              # rank da decomposição
lora_alpha: int = 32                    # escala
target_modules: list[str] = ["q_proj", "v_proj"]  # módulos alvo
lora_dropout: float = 0.05
bias: str = "none"                      # "none", "all", "lora_only"
task_type: str = "CAUSAL_LM"
```

**Uso:** `healing.lora = None` desabilita LoRA → fine-tuning completo.

### 4.2 `HealingConfig`

**Responsabilidade:** Configuração declarativa para treinamento/reparação de modelo.

**Campos:**
```python
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
max_seq_length: int | None = None

# PEFT/LoRA
lora: LoRAConfig | None = None   # None → full fine-tune

# Avançado: Trainer custom (não JSON-serializável)
trainer_class: Any | None = Field(None, exclude=True)
```

### 4.3 `HealingPort`

**Responsabilidade:** Interface abstrata para qualquer estratégia de treinamento.

**Contrato:**
```python
config: HealingConfig

@abstractmethod def heal(model: Any) -> Any:
    """
    Fine-tuna/repara model e retorna versão healing.

    O objeto retornado substitui o modelo no estado do DAG,
    então passos subsequentes (profiling, compressão adicional)
    recebem automaticamente o modelo healing.
    """

@property def label() -> str:
    return self.config.name
```

---

## 5. `pipeline.py` — Pipeline e PipelineStep

### 5.1 `PipelineStep`

**Responsabilidade:** Descriptor declarativo para um nó do DAG.

**Campos:**
```python
label: str  # nome legível renderizado no grafo DAG e logs
port: Any   # instância do port (ModelHandle, CompressionPort, etc.)
```

**Propriedade derivada `kind`:**
```python
@property def kind(self) -> str:
    """Infere o tipo do passo a partir da interface do port."""
    if isinstance(self.port, ModelHandle):      → "load"
    if isinstance(self.port, CompressionPort):  → "compress"
    if isinstance(port, ProfilingPort):        → "profile"
    if isinstance(self.port, HealingPort):     → "heal"
    # raises ValueError se tipo desconhecido
```

### 5.2 `Pipeline`

**Responsabilidade:** Definição completa do pipeline do experimento.

**Campos:**
```python
name: str
description: str = ""
steps: list[PipelineStep]
```

**Validações (via `@field_validator`):**
```python
_must_start_with_load:  steps[0].kind deve ser "load"
_labels_must_be_unique:  todos os labels devem ser distintos
```

**Config:** `model_config = {"arbitrary_types_allowed": True}` — permite que `port` seja de qualquer tipo (incluindo classes abstratas).

---

## 6. `results.py` — PipelineRunResult

### `PipelineRunResult`

**Responsabilidade:** Agregar resultado de uma execução completa de pipeline.

**Campos:**
```python
pipeline_name: str
pipeline_description: str = ""
started_at: datetime
finished_at: datetime
profiling_records: list[ProfilingResult] = factory
```

**Propriedade:**
```python
duration_seconds: float  # (finished_at - started_at).total_seconds()
```

**Métodos:**

```python
def to_markdown_table() -> str:
    """
    Gera tabela Markdown pronta para dissertação.

    Colunas: Stage | Latency p95 (ms) | GPU Peak (MB) | Perplexity
    Uma linha por ProfilingResult em profiling_records.
    """

def to_dissertation_log() -> str:
    """
    Gera log de experimento legível para Methods section ou appendix.
    Inclui: pipeline_name, description, duration, e summary() de cada record.
    """

def save_json(path: str | Path) -> Path:
    """Serializa resultado completo (incluindo extra dumps) para JSON."""

def save_html_report(path, dag_image_path, dag_echarts_data) -> Path:
    """Gera relatório HTML interativo via generate_html_report()."""
```

---

## 7. `html_reporter.py` — Gerador de Relatório HTML

### `generate_html_report()`

**Responsabilidade:** Converter `PipelineRunResult` em HTML autocontido com visualizações ECharts.

**Assinatura:**
```python
def generate_html_report(
    result: PipelineRunResult,
    output_path: str | Path = "pipeline_report.html",
    dag_image_path: str | Path | None = None,
    dag_echarts_data: dict | None = None,
) -> Path
```

**Dependências:**
- `jinja2.Environment` com `FileSystemLoader`
- Template: `llm_flux/core/templates/report/html_reporter.html.jinja2`
- Apache ECharts (via CDN no template)

**Dados extraídos por registro (`_extract_*`):**
- `_extract_model_info()` — de `extra["llm_profile"]`
- `_extract_hardware_info()` — de `extra["hardware"]`
- `_extract_inference_info()` — de `extra["inference"]`
- `_extract_benchmarks_info()` — de `extra["benchmarks"]`

**ECharts no template:**
- **Gráfico de arquitetura** — usa estratégia de ellipsis para blocos repetidos (transformation layers → Block 0 → ... → Block N-1 → post-block)
- **Latência por estágio** — linha chart com mean/std/p95
- **Memória GPU** — bar chart por estágio
- **Perplexidade** — linha chart (se disponível)

---

## Contratos de Interface — Resumo

| Port | Método Principal | Retorno | Exceção |
|---|---|---|---|
| `ModelHandle` | `load()` | `object` (nn.Module) | — |
| `ModelHandle` | `unload()` | `None` | — |
| `CompressionPort` | `compress(model)` | `Any` (model) | `CompressionNotSupportedError` |
| `ProfilingPort` | `profile(model, stage_label)` | `ProfilingResult` | — |
| `HealingPort` | `heal(model)` | `Any` (model) | — |
| `DatasetPort` | `load()` | `object` (dataset) | — |
| `Pipeline` | valida steps | validação Pydantic | `ValueError` |

---

## Riscos e Limitações

1. **Validação fraca de `port` em `PipelineStep`** — `arbitrary_types_allowed` permite qualquer tipo sem verificação em tempo de validação do Pydantic. Um port inválido só falha em runtime no executor.

2. **Tipo de retorno de `load()` não especificado** — `object` é genérico demais. Qualquer adapter pode retornar qualquer coisa, e o `Executor` confia no tipo sem verificação.

3. **`get_tokenizer()` como convenção** — não é parte do contrato de `ModelHandle`, mas é esperado por `GPTQAdapter`, `HFTrainerAdapter`, `DepthPruningAdapter` e `ComprehensiveProfilingAdapter`. Se um `ModelHandle` concreto não implementar este método, resulta em erro em runtime.

4. **`CompressionNotSupportedError` apenas em compressores** — `HealingPort` e `ProfilingPort` não têm mecanismo de erro documentado para falhas.

5. **Validação de `steps[0]` é `kind == "load"`** — `PipelineStep.kind` é uma propriedade que infere o tipo via `isinstance`. Se um port customizado não for reconhecido, levanta `ValueError` no momento da definição do pipeline (não no parsing).

---

## Exports Públicos de `core/`

```python
from llm_flux.core import (
    # compression
    CompressionConfig, CompressionNotSupportedError, CompressionPort,
    # healing
    HealingConfig, HealingPort, LoRAConfig,
    # model
    ModelHandle, ModelSource,
    # pipeline
    Pipeline, PipelineStep,
    # profiling
    AccuracyMetrics, LatencyMetrics, MemoryMetrics,
    ProfilingConfig, ProfilingPort, ProfilingResult,
    # results
    PipelineRunResult,
)
```
