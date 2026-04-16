# Arquitetura — LLM-Flux

## Visão Geral

LLM-Flux é um framework de pesquisa para compressão de LLMs baseado em uma **arquitetura hexagonal** (ports and adapters) com um **DAG pipeline** declarativo. O design prioriza:

1. **Separação de concerns** — portas puras em Python (sem imports de ML) vs. adaptadores concretos.
2. **Extensibilidade** — novas técnicas de compressão/healing são adicionadas via subclassing, sem modificar código existente (Open/Closed Principle).
3. **Dissertation-readiness** — toda saída é projetada para ser diretamente utilizável em dissertações (tabelas Markdown, JSON completo, relatórios HTML).

---

## Camadas Arquiteturais

```
┌─────────────────────────────────────────────────────────────────┐
│                         USAGE LAYER                              │
│   run_pipeline() em runner.py — ponto de entrada de alto nível   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Pipeline (declarativo)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DAG EXECUTION LAYER                           │
│   build_dag()  →  nx.DiGraph  →  PipelineExecutor (topological)  │
│   render_dag() (PNG)  +  render_dag_echarts() (interactive)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Para cada nó: port.{method}(model)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PORT LAYER (core/)                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │ ModelHandle  │ │CompressionPort│ │ ProfilingPort│ │HealingPort│ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
│                                                                   │
│  contracts: CompressionConfig, HealingConfig, ProfilingConfig,   │
│             ModelSource, DatasetConfig, PipelineRunResult        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ implementação concreta
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ADAPTER LAYER (adapters/)                      │
│                                                                   │
│  ┌────────────────────┐  ┌────────────────────────────────────┐  │
│  │ adapters/model/    │  │ adapters/compression/               │  │
│  │  └─ HFModelHandle  │  │  ├─ GPTQAdapter  (auto-gptq)        │  │
│  │                    │  │  ├─ AWQAdapter    (autoawq)        │  │
│  │ adapters/healing/  │  │  ├─ BitsAndBytesAdapter            │  │
│  │  └─ HFTrainerAdapter│ │  └─ DepthPruningAdapter           │  │
│  │                    │  │                                     │  │
│  │ adapters/datasets/ │  │                                     │  │
│  │  ├─ HFDatasetAdapter│ │                                     │  │
│  │  └─ LocalDatasetAdapter│                                    │  │
│  └────────────────────┘  └────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ profiling/adapters/comprehensive.py                         │  │
│  │   → orquestra HardwareProfiler + LLMProfiler +              │  │
│  │          InferenceBenchmarker + ModelPerformanceBenchmarker  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

---

## Componentes Principais

### 1. Core Ports (`llm_flux/core/`)

Todos os ports são definidos como **classes abstratas (ABC)** em Python puro, sem imports de frameworks de ML. Isso garante que o núcleo do framework seja testável isoladamente.

#### 1.1 `ModelHandle` + `ModelSource` (`core/model.py`)

**Responsabilidade:** Abstração para carregar um modelo de **qualquer fonte** (HuggingFace Hub ou diretório local).

**Contrato:**
```python
class ModelHandle(ABC):
    source: ModelSource
    @abstractmethod def load() -> object: ...
    @abstractmethod def unload() -> None: ...
    @property @abstractmethod def name() -> str: ...
```

**Fluxo:**
- `ModelSource` detecta automaticamente local vs. Hub via `Path(identifier).exists()`
- `HFModelHandle.load()` usa `AutoModelForCausalLM.from_pretrained()` + `AutoTokenizer`
- O tokenizer é exposto via `get_tokenizer()` para uso por outros componentes

**Fronteiras:**
- `load()` retorna um `nn.Module` cru — todos os adapters de compressão recebem este objeto.
- O `ModelHandle` é o ÚNICO ponto de entrada para modelos no pipeline.

#### 1.2 `CompressionPort` + `CompressionConfig` (`core/compression.py`)

**Responsabilidade:** Interface abstrata para qualquer técnica de compressão.

**Contrato:**
```python
class CompressionPort(ABC):
    config: CompressionConfig
    @abstractmethod def compress(model: Any) -> Any: ...
```

**Fluxo:**
- Aceita modelo carregado (de `ModelHandle.load()` ou de outro `CompressionPort.compress()`)
- Retorna modelo compressado (in-place ou novo objeto)
- Lança `CompressionNotSupportedError` se a técnica for incompatível com a arquitetura

**Fronteiras:**
- Cada adapter de compressão é completamente independente — não há ordem forçada entre etapas de compressão.
- É possível encadear múltiplas técnicas de compressão (ex: BitsAndBytes + Depth Pruning).

#### 1.3 `ProfilingPort` + `ProfilingConfig` (`core/profiling.py`)

**Responsabilidade:** Interface para coleta de métricas de performance.

**Sub-tipos de resultado (`ProfilingResult`):**
```python
MemoryMetrics:  peak_gpu_mb | peak_cpu_mb
LatencyMetrics: mean_ms, std_ms, min_ms, p5_ms, p50_ms, p95_ms, p99_ms, max_ms
AccuracyMetrics: perplexity | task_score + task_name
ProfilingResult: profiler_name, model_label, pipeline_stage, timestamp,
                memory, latency, accuracy, extra (dict with raw dumps)
```

**Contrato:**
```python
class ProfilingPort(ABC):
    config: ProfilingConfig
    @abstractmethod def profile(model: Any, stage_label: str) -> ProfilingResult: ...
```

**Fluxo:**
- Recebe modelo (pós-compressão ou pós-healing)
- Produz um `ProfilingResult` com métricas estruturadas
- Dados brutos (hardware profile, LLM architecture dump, inference timings) ficam em `extra`
- `PipelineRunResult` agrega múltiplos `ProfilingResult` (um por etapa de profiling)

#### 1.4 `HealingPort` + `HealingConfig` (`core/healing.py`)

**Responsabilidade:** Interface para reparação de modelo via fine-tuning.

**Sub-tipos:**
```python
LoRAConfig:  r, lora_alpha, target_modules, lora_dropout, bias, task_type
HealingConfig:  dataset, max_steps, learning_rate, per_device_train_batch_size,
                gradient_accumulation_steps, fp16, output_dir, lora (Optional[LoRAConfig])
```

**Contrato:**
```python
class HealingPort(ABC):
    config: HealingConfig
    @abstractmethod def heal(model: Any) -> Any: ...
```

**Fluxo:**
- Recebe modelo compressado
- Aplica fine-tuning (full ou LoRA/QLoRA)
- Retorna modelo re-treinado que substitui o modelo compressado no estado do DAG

#### 1.5 `Pipeline` + `PipelineStep` (`core/pipeline.py`)

**Responsabilidade:** Definição declarativa do pipeline completo.

```python
class PipelineStep(BaseModel):
    label: str           # nome legível para DAG e logs
    port: Any           # instância do port (ModelHandle, CompressionPort, etc.)
    @property def kind() -> str:  # "load" | "compress" | "profile" | "heal"
```

**Validações:**
- `steps[0]` deve ser do tipo `load` (ModelHandle) — validado por `_must_start_with_load`
- Labels devem ser únicos — validado por `_labels_must_be_unique`

**Propriedade derivada `kind`:**
```python
if isinstance(port, ModelHandle)      → "load"
if isinstance(port, CompressionPort)  → "compress"
if isinstance(port, ProfilingPort)  → "profile"
if isinstance(port, HealingPort)    → "heal"
```

### 2. DAG Layer (`llm_flux/dag/`)

#### 2.1 `build_dag()` (`dag/builder.py`)

Converte um `Pipeline` em um `nx.DiGraph`:

```python
def build_dag(pipeline: Pipeline) -> nx.DiGraph:
    # nós: id = "{index}_{label}", dados: step, kind, label, index
    # arestas: linear sequential (step[i] → step[i+1])
    # grafo: metadata = {name, description}
```

**Nota:** O builder produz um DAG **linear** (sem branches). Suporte a fan-out está marcado como extensão futura (`depends_on: list[str]` em `PipelineStep`).

#### 2.2 `PipelineExecutor` (`dag/executor.py`)

Executa o DAG via **topological sort** do NetworkX:

```python
for node_id in nx.topological_sort(dag):
    step = dag.nodes[node_id]["step"]
    match step.kind:
        case "load":     model = port.load()
        case "compress":  model = port.compress(model)
        case "profile":   result = port.profile(model, stage_label=step.label)
        case "heal":      model = port.heal(model)
```

**Thread de estado:**
- LOAD inicializa `model`
- COMPRESS e HEAL substituem `model`
- PROFILE lê `model` e appenda `ProfilingResult` à lista

**Tratamento de erros:**
- `CompressionNotSupportedError` → loga erro e aborta pipeline
- Exceções genéricas → loga e relança

#### 2.3 `render_dag()` (`dag/renderer.py`)

Renderiza o DAG em **PNG** via matplotlib + networkx:
- Layout hierarchical (graphviz dot ou spring fallback)
- Cores por tipo de passo: load=blue, compress=orange, profile=green, heal=purple
- Fundo escuro, legendas coloridas

`render_dag_echarts()` produz estrutura de dados para visualização interativa no HTML.

### 3. Adapter Layer (`llm_flux/adapters/`)

#### 3.1 Model Adapter

**`HFModelHandle`** — o único adapter de modelo implementado:
- Detecta local vs. Hub via `ModelSource.is_local`
- `device_map="auto"` para alocação automática em GPU
- Tokenizer exposto via `get_tokenizer()`
- `unload()` deleta modelo e faz `torch.cuda.empty_cache()`

#### 3.2 Compression Adapters

| Adapter | Técnica | Importação tardia | Requer re-load |
|---|---|---|---|
| `GPTQAdapter` | GPTQ post-training quantization | `auto_gptq` (lazy) | Não |
| `AWQAdapter` | AWQ activation-aware quantization | `autoawq` (lazy) | Não |
| `BitsAndBytesAdapter` | NF4/INT8 | `bitsandbytes` (lazy) | **Sim** (reload com `quantization_config`) |
| `DepthPruningAdapter` | Structural depth pruning | custom hooks | Não |

**Contrato comum:**
```python
class XxxAdapter(CompressionPort):
    def __init__(self, config: XxxConfig, ...) -> None: ...
    def compress(self, model: Any) -> Any: ...  # raises CompressionNotSupportedError
```

#### 3.3 Healing Adapter

**`HFTrainerAdapter`** — HF Trainer com LoRA/QLoRA:
- Aceita tokenizer via construtor
- `_load_dataset()` auto-detecta HF Hub vs. local (mesma lógica de DatasetConfig)
- Tokenização com `max_seq_length` auto-detectado ou configurado
- Materializa streaming datasets para evitar lazy I/O durante training
- Aplica LoRA via `peft.LoraConfig` se `config.lora` for definido
- Fallback: `_default_trainer()` retorna `transformers.Trainer`

#### 3.4 Dataset Adapters

**`DatasetPort`** — interface para carregar datasets:
```python
class DatasetPort(ABC):
    config: DatasetConfig
    @abstractmethod def load() -> object: ...  # retorno compatível com HF Trainer
```

**Adaptadores concretos:**
- `HFDatasetAdapter`: `load_dataset(path, split, streaming=...)`
- `LocalDatasetAdapter`: `load_dataset(format, data_files=...)` — formatos: JSON, JSONL, CSV, Parquet

### 4. Profiling Domain Services (`llm_flux/profiling/`)

```
ProfilingPort (contract)
    ↑
    │ (bridge)
ComprehensiveProfilingAdapter
    │
    ├── HardwareProfiler      → CPU/GPU/RAM info (linux-only)
    ├── LLMProfiler           → Static analysis (params, arch, attention, memory est.)
    ├── InferenceBenchmarker   → Latency timing (wall-clock, token throughput)
    └── ModelPerformanceBenchmarker → Perplexity + accuracy benchmarks
```

#### 4.1 `LLMProfiler` (`profiling/llm_profiler.py`)

Serviço de domínio para **análise estrutural profunda** de LLMs PyTorch.

**Análises disponíveis:**
1. **Parameter counting** — total, trainable, non-trainable, by_dtype
2. **Architecture analysis** — layer_types, layer_details, max_depth
3. **Attention layer analysis** — num_attention_layers, per-layer attributes
4. **Memory estimation** — teórica por precisão (FP32→INT4), incluindo KV-cache e ativações
5. **Connection analysis** — torch.fx tracing → graph edges + shapes; fallback hooks; fallback basic

**Fluxo de `analyze_connections()`:**
```
sample_input ou input_shape
    │
    ├─→ _analyze_connections_fx()     ← tenta torch.fx.symbolic_trace
    │         se falhar → retorna None
    │
    ├─→ _analyze_connections_hooks()  ← register_forward_hook por layer
    │         se falhar → retorna None
    │
    └─→ _analyze_connections_basic()   ← build_empty_graph (sem edge info)
```

**Saída:** `LLMInfo` (Pydantic model) com sub-modelos:
- `ArchitectureInfo` (com `connections: ConnectionAnalysisInfo | None`)
- `AttentionLayerAnalysisInfo`
- `ParameterInfo`
- `ModelSummary`
- `MemoryEstimationInfo` (um `MemoryEstimate` por `PrecisionType`)

#### 4.2 `HardwareProfiler` (`profiling/hardware_profiler.py`)

- **CPU:** `/proc/cpuinfo` (Linux-only) → name, architecture, cores, freq
- **GPU:** `torch.cuda` API → device_name, memory_info, properties
- **RAM:** `psutil.virtual_memory()` → total, available, free, swap

Plataforma suportada: **Linux apenas**. `PlatformNotSupportedException` em outras plataformas.

#### 4.3 `InferencePerformanceBenchmarker` (`profiling/inference_benchmarker.py`)

**Entrada:** model, tokenizer, prompt, max_new_tokens, num_runs, warmup_runs

**Processo:**
1. Warmup runs (descartadas)
2. Timed runs — wall-clock via `time.perf_counter()`
3. Cálculo de estatísticas (mean, std, percentis, min, max)
4. Token throughput (generated_tokens / avg_time)

**Saída:** `InferencePerformanceInfo` (Pydantic)

**Método auxiliar:** `compare_models_inference(original, compressed)` → speedup ratio + TPS improvement %

#### 4.4 `ModelPerformanceBenchmarker` (`profiling/model_benchmarker.py`)

**Arquitetura:**
- Auto-registro de métodos de benchmark via `hasattr(self, test_name)`
- Registro manual via `register(name, fn)`
- `benchmark()` itera sobre os tests solicitados e agrega resultados

**Tipos de teste:**
- **Perplexidade:** `lambada`, `wikitext_103_v1`, `c4` → `PerplexityTestResult`
- **Multiple-choice accuracy:** mmlu_pro, arc_challenge, hellaswag, winogrande, piqa, sciq, truthfulqa_mc2
- **Extractive QA:** squad (F1 token-level)
- **Math extraction:** gsm8k, math_500 (exact match com parser de número)
- **Generation:** big_bench_hard
- **Code:** humaneval_pass1, mbpp_pass1 (Pass@1 via exec)

**Métodos de acurácia:**
- `_multiple_choice_accuracy()` — log-prob de cada escolha via `_choice_logprob()`
- `_exact_match_generation_accuracy()` — geração + normalização + comparison
- `_truthfulqa_mc2()` — normalized probability mass (calibration-style)
- `_squad_f1()` — max token F1 sobre gold answers

### 5. Results & Reporting

#### `PipelineRunResult` (`core/results.py`)

Agrega todos os `ProfilingResult` de uma execução completa:
```python
pipeline_name, pipeline_description, started_at, finished_at,
profiling_records: list[ProfilingResult]
```

**Métodos:**
- `to_markdown_table()` → tabela Markdown pronta para dissertação
- `to_dissertation_log()` → log legível (Methods section / appendix)
- `save_json()` → JSON completo com `extra` dumps
- `save_html_report()` → HTML interativo (Jinja2 + ECharts)

#### `generate_html_report()` (`core/html_reporter.py`)

Gera HTML autocontido com:
- Jinja2 template (`html_reporter.html.jinja2`)
- Apache ECharts para gráficos
- Gráfico de arquitetura de modelo (com estratégia de ellipsis para blocos repetidos)
- DAG image ou visualização ECharts interativa
- Dados de hardware, inference e benchmarks extraídos de cada `ProfilingResult.extra`

---

## Conexões Entre Módulos

```
User Code (examples/)
    │
    └─→ run_pipeline(pipeline, ...)
            │
            ├─→ build_dag(pipeline) → nx.DiGraph
            ├─→ render_dag(dag) → PNG
            ├─→ render_dag_echarts(dag) → dict
            │
            └─→ PipelineExecutor(dag).run()
                    │
                    ├─→ model_handle.load()
                    │       └─→ HFModelHandle → AutoModelForCausalLM + AutoTokenizer
                    │
                    ├─→ compressor.compress(model)
                    │       ├─→ GPTQAdapter → auto_gptq → quantize()
                    │       ├─→ AWQAdapter → autoawq → quantize()
                    │       ├─→ BitsAndBytesAdapter → reload with quantization_config
                    │       └─→ DepthPruningAdapter → angular_distance_importance
                    │                                     + _prune_layers()
                    │
                    ├─→ profiler.profile(model)
                    │       └─→ ComprehensiveProfilingAdapter
                    │               ├─→ HardwareProfiler.retrieve_hardware_information()
                    │               ├─→ LLMProfiler(model, tokenizer).profile_complete()
                    │               ├─→ InferencePerformanceBenchmarker.time_inference()
                    │               └─→ ModelPerformanceBenchmarker.benchmark()
                    │
                    └─→ healer.heal(model)
                            └─→ HFTrainerAdapter
                                    ├─→ LocalDatasetAdapter or HFDatasetAdapter
                                    ├─→ tokenization
                                    ├─→ peft.LoraConfig (se lora is not None)
                                    └─→ transformers.Trainer.train()
```

---

## Fronteiras Entre Módulos

| De | Para | Tipo de dados |
|---|---|---|
| `ModelHandle.load()` | PipelineExecutor | `nn.Module` + `_tokenizer` interno |
| `CompressionPort.compress()` | PipelineExecutor | `nn.Module` (mesmo ou novo objeto) |
| `HealingPort.heal()` | PipelineExecutor | `nn.Module` (modelo fine-tuned) |
| `ProfilingPort.profile()` | PipelineExecutor | `ProfilingResult` |
| `HFModelHandle` | `GPTQAdapter` | `tokenizer.get_tokenizer()` |
| `HFModelHandle` | `HFTrainerAdapter` | `tokenizer.get_tokenizer()` |
| `HFModelHandle` | `DepthPruningAdapter` | `tokenizer.get_tokenizer()` |
| `DatasetConfig` | `GPTQAdapter/AWQAdapter/DepthPruningAdapter` | `calibration_dataset` |
| `ProfilingResult.extra` | `html_reporter.py` | dict com raw dumps |

---

## Decisões Arquiteturais Chave

1. **Ports puras em Python (core/)** — sem imports de torch/transformers. Permite testar o domínio sem dependências de ML.

2. **Pipeline declarativo** — usuário descreve steps como dados (não callables), permitindo validação estática (labels únicos, primeiro passo é LOAD).

3. **Estado threadado no executor** — o modelo é passado implicitamente de passo a passo, sem armazenamento global.

4. **ComprehensiveProfilingAdapter como facade** — orquestra 4 serviços de profiling em um único `ProfilingResult`, evitando múltiplos nós de profiling no DAG.

5. **extra dict para dados brutos** — `ProfilingResult.extra` mantém dados raw (hardware, llm_profile, inference, benchmarks) sem poluir o modelo Pydantic principal, mas preserva tudo no JSON.

6. **BitsAndBytes re-load** — diferente das outras técnicas, BitsAndBytes requer re-carregar o modelo com `quantization_config`. O adapter recebe `model_handle` para ter acesso ao `source`.

---

## Hipóteses e Pontos de Incerteza

1. **mlp_pruning.py** está vazio — não implementado, apenas placeholder.
2. **Suporte a non-Decoder-only LLMs** não foi verificado — todas as técnicas assumem causal LM.
3. **Threads do tokenizer** — `tokenizer.get_tokenizer()` é chamado em múltiplos adapters. Se o HFModelHandle não for passado como tokenizer, pode falhar em runtime.
4. **Auto-detecção de LayerList** em `DepthPruningAdapter.get_transformer_layers()` usa heuristics ( Llama→`model.model.layers`, GPT-2→`transformer.h`). Modelos com arquiteturas diferentes podem não funcionar.
5. **`healing_output/` e `compression_results/`** são diretórios de output mas não estão formalizados como parte da arquitetura de resultados.
6. **`dag/builder.py` suporta apenas DAGs lineares** — branches estão como extensão futura.
