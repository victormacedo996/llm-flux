# Módulo Profiling — Serviços de Domínio e Tipos

## Visão Geral

O módulo `llm_flux/profiling/` implementa todos os serviços de **avaliação, profiling e benchmark** do framework. A arquitetura é organizada em:

1. **Domain Services** (`*.py`) — classes que executam profiling concreto
2. **Types** (`types/`) — modelos Pydantic com todos os tipos de dado
3. **Adapters** (`adapters/`) — bridges para o port `ProfilingPort`

```
ProfilingPort (contract — em core/profiling.py)
         ↑
         │ (wraps and orchestrates)
ComprehensiveProfilingAdapter (adapters/comprehensive.py)
         │
         ├── HardwareProfiler (hardware_profiler.py)
         ├── LLMProfiler (llm_profiler.py)
         ├── InferencePerformanceBenchmarker (inference_benchmarker.py)
         └── ModelPerformanceBenchmarker (model_benchmarker.py)
```

---

## 1. Domain Services

### 1.1 `HardwareProfiler` (`profiling/hardware_profiler.py`)

**Responsabilidade:** Coletar snapshot da configuração de hardware do host.

**Plataforma suportada:** **Linux apenas**. Lança `PlatformNotSupportedException` em outras plataformas.

**Interface pública:**
```python
def retrieve_hardware_information(self) -> HardwareProfile
def retrieve_cpu_information(self) -> CPUInfo
def retrieve_gpu_information(self) -> SystemGPUInfo
def retrieve_ram_information(self) -> RAMInfo
```

**CPU Info (`_get_linux_cpu_info()`):**
- Lê `/proc/cpuinfo` para `model name`
- `psutil.cpu_freq()` para frequência
- `psutil.cpu_count(logical=False)` / `psutil.cpu_count(logical=True)` para cores

**GPU Info (`_get_gpu_info(device_id)`):**
- `torch.cuda.get_device_name(device_id)`
- `torch.cuda.get_device_properties(device_id)`
- `torch.cuda.memory_allocated/reserved()` snapshot

**RAM Info (`_get_ram_information()`):**
- `psutil.virtual_memory()` → total, available, free
- `psutil.swap_memory()` → swap total, free

**Saída:** `HardwareProfile` com sub-modelos:
```python
cpu: CPUInfo          # name, architecture, platform, physical_cores, total_cores, max_freq
gpu: SystemGPUInfo    # cuda_available, cuda_version, cudnn_version, device_count, gpus[]
ram: RAMInfo          # total_memory_gb, available_memory_gb, free_memory_gb, swap_*
```

### 1.2 `LLMProfiler` (`profiling/llm_profiler.py`)

**Responsabilidade:** Análise estrutural profunda de LLMs PyTorch. Combina análise estática (parâmetros, arquitetura) com análise dinâmica opcional (conexões de camadas).

**Construtor:**
```python
def __init__(self, model: nn.Module, tokenizer: Any, verbose: bool = True) -> None:
    self.model = model
    self.tokenizer = tokenizer
    self.verbose = verbose
    self.device = self._detect_device()
    self.profile_data: Optional[LLMInfo] = None
```

**Entrada principal:**
```python
def profile_complete(
    self,
    analyze_connections: Optional[AnalyzeConnections] = None,
    estimate_memory: Optional[EstimateMemory] = None,
) -> LLMInfo
```

**Análises executadas sequencialmente:**

#### 1.2.1 Parameter Counting — `count_parameters()`

```python
def count_parameters(self) -> ParameterInfo:
    # Itera todos os parâmetros do modelo
    # Conta: total, trainable, non_trainable
    # Agrega por dtype: counts e bytes
```

**Retorno:** `ParameterInfo`
```python
total: int
trainable: int
non_trainable: int
total_millions: float
total_billions: float
by_dtype_counts: dict[str, int]      # e.g. {"torch.float32": 1240000}
by_dtype_bytes: dict[str, int]
```

#### 1.2.2 Architecture Analysis — `analyze_architecture()`

```python
def analyze_architecture(self) -> ArchitectureInfo:
    # named_modules() — apenas leaf modules (não containers)
    # Extrai: layer_type, name, parameters, depth
    # Extrai atributos: in/out_features, vocab_size, embedding_dim, num_heads, head_dim, embed_dim
```

**Retorno:** `ArchitectureInfo`
```python
total_layers: int
layer_types_count: dict[str, int]   # e.g. {"Linear": 32, "LayerNorm": 1}
layer_details: list[dict]           # [{name, type, parameters, depth, input_size, ...}]
max_depth: int
connections: ConnectionAnalysisInfo | None  # populado se analyze_connections fornecido
```

#### 1.2.3 Attention Layer Analysis — `analyze_attention_layers()`

```python
def _looks_like_attention(module) -> bool:
    # isinstance(module, nn.MultiheadAttention) → True
    # hasattr(module, "num_attention_heads"|"num_heads"|"head_dim") → True
    # type name contém "attention"|"attn"|"multihead" → True
```

**Filtro:** only leaf modules que são "attention-like", depois filtra para manter apenas leaf candidates (não pais de outros candidates).

**Retorno:** `AttentionLayerAnalysisInfo`
```python
num_attention_layers: int
attention_layers: list[dict]  # [{name, type, num_attention_heads, head_dim, embed_dim, ...}]
```

#### 1.2.4 Memory Estimation — `estimate_memory_requirements()`

Calcula requisitos teóricos de memória para **cada precisão** (FP32, FP16, BF16, INT8, INT4):

```python
def estimate_memory_requirements(
    self,
    sequence_length: int = 2048,
    batch_size: int = 1,
    include_kv_cache: bool = True,
    include_activations: bool = True,
    gradient_accumulation_steps: int = 1,
    optimizer_type: str = "adamw",
) -> MemoryEstimationInfo
```

**Para cada precisão:**
- `weights_mb` — `total_params * bytes_per_param / 1024²`
- `kv_mb` — estimado via `_estimate_kv_cache_memory()` (2 * batch * heads * seq * head_dim * bpp por layer)
- `act_mb` — estimado via `_estimate_activation_memory()` (heurística com embed_dim e attn_layers)
- `grad_mb` — `total * bpp / 1024² * grad_accum`
- `opt_mb` — `total * opt_bytes / 1024²` (AdamW=8, SGD=bpp, Adafactor=4)
- `training_mb` — weights + activations + gradients + optimizer
- `total_inference_mb` — weights * 1.2 (overhead)

**Retorno:** `MemoryEstimationInfo`
```python
base_parameters: int
estimates: dict[str, MemoryEstimate]  # chave = precision.value (e.g. "FP32", "INT4")
```

Cada `MemoryEstimate`:
```python
precision: str
bytes_per_parameter: float
model_weights_mb: float
kv_cache_mb: float | None
activation_memory_mb: float | None
gradient_memory_mb: float
optimizer_memory_mb: float
training_memory_mb: float
total_memory_mb: float
total_memory_gb: float
total_inference_memory_mb: float
```

#### 1.2.5 Connection Analysis — `analyze_connections()`

```python
def analyze_connections(
    self,
    sample_input: Optional[Any] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    input_sample: Optional[Callable[[], Any]] = None,
) -> ConnectionAnalysisInfo
```

**Estratégia de fallback em cascata:**

```
_analyze_connections_fx(input)
    falha → retorna None
        ↓
_analyze_connections_hooks(input)
    falha → retorna None
        ↓
_analyze_connections_basic()
    retorna graph vazio (apenas estrutura, sem edges)
```

**FX tracing (`_analyze_connections_fx`):**
1. `torch.fx.symbolic_trace(self.model)`
2. Itera nodes com `op == "call_module"`
3. Popula `LayerConnectionInfo` por target
4. Reconstrói arestas via `node.args`
5. `_attach_shapes_via_hooks()` para input/output shapes
6. `_analyze_connection_patterns(graph, "torch_fx")`

**Hooks (`_analyze_connections_hooks`):**
1. Registra `register_forward_hook` em cada leaf module
2. Forward pass com input
3. Captura `input.shape` e `output.shape` por hook
4. Reconstrói ordem de execução + arestas
5. `_analyze_connection_patterns(graph, "hooks")`

**Basic (`_analyze_connections_basic`):**
1. `build_empty_graph()` — nodes sem edges
2. `_analyze_connection_patterns(graph, "basic")`

**Retorno:** `ConnectionAnalysisInfo`
```python
total_connections: int
connection_graph: dict[str, LayerConnectionInfo]
analysis_method: str              # "torch_fx", "hooks", ou "basic"
has_skip_connections: bool
max_fan_in: int
max_fan_out: int
```

**`LayerConnectionInfo`:**
```python
name: str
type: str
parameters: int
depth: int
input_layers: list[str]           # nomes das layers que alimentam esta
output_layers: list[str]          # nomes das layers que esta alimenta
input_shape: list[int] | None
output_shape: list[int] | None
```

### 1.3 `InferencePerformanceBenchmarker` (`profiling/inference_benchmarker.py`)

**Responsabilidade:** Medir latência de inferência (wall-clock) e throughput de tokens.

**Interface:**
```python
def time_inference(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> InferencePerformanceInfo
```

**Processo:**
1. `tokenizer(prompt, return_tensors="pt").to(model.device)` — tokenização
2. `warmup_runs` iterações de `model.generate()` sem timed (descartadas)
3. `num_runs` iterações timed via `time.perf_counter()`
4. Contagem de tokens gerados: `output.size(1) - input_ids.size(1)`
5. Estatísticas com `numpy` (mean, std, percentis, min, max)
6. Token throughput: `generated_tokens / avg_time`

**Retorno:** `InferencePerformanceInfo`
```python
avg_time: float            # segundos
std_time: float
min_time: float
p5_time: float
p50_time: float
p95_time: float
p99_time: float
max_time: float
tokens_per_second: float
num_runs: int
generated_tokens: int
raw_times: list[float]
```

**Helper:** `compare_models_inference(original, compressed)` → `CompareBenchmark`
```python
speedup: float                      # original.avg_time / compressed.avg_time
tps_improvement_percent: float      # (compressed.tps / original.tps - 1) * 100
```

**Nota:** Havia um bug na versão original onde `compare_models_inference` dividia o objeto Pydantic em vez de `.avg_time`. Corrigido na versão atual.

### 1.4 `ModelPerformanceBenchmarker` (`profiling/model_benchmarker.py`)

**Responsabilidade:** Executar benchmarks de perplexidade e acurácia em modelos.

**Arquitetura de auto-descoberta:**
```python
class ModelPerformanceBenchmarker:
    def __init__(self):
        self._test_methods: dict[str, Callable] = {}
        # Auto-registra métodos que casam com AVAILABLE_TESTS
        for test in AVAILABLE_TESTS:
            if hasattr(self, test):
                self._test_methods[test] = getattr(self, test)

    def register(self, name, fn):
        self._test_methods[name] = fn
```

**Conjunto de testes disponíveis (`AVAILABLE_TESTS`):**

Perplexity tests:
- `lambada` — cimec/lambada, test split
- `wikitext_103_v1` — Salesforce/wikitext, wikitext-103-v1 subset, test split
- `c4` — allenai/c4, en subset, validation split

Accuracy tests (multiple-choice):
- `mmlu_pro` — TIGER-Lab/MMLU-Pro
- `arc_challenge` — allenai/ai2_arc, ARC-Challenge subset
- `hellaswag` — Rowan/hellaswag, validation
- `winogrande` — allenai/winogrande, winogrande_xl subset
- `piqa` — ybisk/piqa, validation
- `sciq` — allenai/sciq, test
- `truthfulqa_mc2` — domenicrosati/TruthfulQA, validation

Accuracy tests (generation/extractive):
- `gsm8k` — openai/gsm8k, main subset, test — exact match com número
- `math_500` — HuggingFaceH4/MATH-500, test
- `squad` — rajpurkar/squad, validation — token F1
- `big_bench_hard` — maveriq/bigbenchhard, test
- `humaneval_pass1` — openai/openai_humaneval, test — Pass@1 via exec
- `mbpp_pass1` — Muennighoff/mbpp, test — Pass@1 via exec

**Perplexity computation (`_compute_perplexity_for_batch`):**
```python
def _compute_perplexity_for_batch(input_texts, tokenizer, model) -> ComputePerplexityForBatchReturn:
    # Tokeniza textos (batch)
    # Forward pass → logits
    # shift_logits = logits[:, :-1, :]
    # shift_labels = input_ids[:, 1:]
    # shift_mask = attention_mask[:, 1:]
    # NLL = -sum(log_probs * shift_labels) / sum(shift_mask)
    # Perplexity = exp(NLL)
```

**Métodos de acurácia:**

`_choice_logprob(tokenizer, model, prompt, choice)`:
- Decodifica log-probability de uma escolha via `log_softmax` + gather

`_multiple_choice_accuracy(dataset, tokenizer, model, prompt_builder, choices_builder, gold_builder)`:
- Itera dataset
- Para cada row: constrói prompt, calcula log-prob de cada escolha
- Predição = argmax dos scores
- Retorna (accuracy, total, per_example_scores)

`_exact_match_generation_accuracy(dataset, ...)`:
- Gera texto com `model.generate()`
- Parser (e.g. `_extract_last_number()` para GSM8K)
- Normaliza e compara com referência

`_truthfulqa_mc2(dataset, ...)`:
- Para cada questão: soma probability mass em true_answers e false_answers
- MC2 = `P(true) / (P(true) + P(false))`

`_squad_f1(dataset, ...)`:
- Para cada row: prompt = context + question
- Gera resposta
- F1 token-level com gold answers (max sobre todos os gold texts)

**`_token_f1(prediction, reference)`:**
- Normaliza textos (lowercase, remove punct, split)
- Computa overlap (common tokens / pred_len + ref_len)
- F1 = 2 * precision * recall / (precision + recall)

**`_extract_last_number(text)`:**
```python
re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))[-1]
# Retorna último número encontrado (para extrair resposta de GSM8K)
```

**`_humaneval_pass_at_1` / `_mbpp_pass_at_1`:**
- Gera código com `model.generate()`
- Executa código + tests com `exec()`
- Pass@1 = se todos os tests passam

**API de benchmark:**
```python
def benchmark(
    self,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tests: Set[str],
    batch_size: int = 16,
    num_examples: int | None = None,
) -> List[TestResult]
```

**Retornos por tipo:**
- Perplexity tests → `PerplexityTestResult(test_name, result: ComputePerplexityForDatasetReturn)`
- Accuracy tests → `AccuracyTestResult(test_name, metric_name, accuracy, num_examples, per_example_scores, stats)`

**Stats computados:** `DescriptiveStats.from_values()` (mean, std, min, max, percentis) para ambos os tipos.

---

## 2. Tipos Pydantic (`profiling/types/`)

### 2.1 `hardware.py`

```python
CPUInfo:         name, architecture, platform, physical_cores, total_cores, max_freq
GPUInfo:         device_id, device_name, is_available, memory_info, properties
GPUMemoryInfo:  total, allocated, cached, reserved, free; + gb versions
GPUProperties:  name, major, minor, total_memory, multi_processor_count,
                 max_threads_*, max_block_dim, max_grid_dim, warp_size
SystemGPUInfo:  cuda_available, cuda_version, cudnn_version, device_count,
                 current_device, gpus, timestamp
RAMInfo:        total_memory_gb, available_memory_gb, free_memory_gb,
                 swap_total_gb, swap_free_gb
HardwareProfile: cpu, gpu, ram
```

### 2.2 `llm.py`

```python
PrecisionType:  Literal["FP32", "FP16", "BFLOAT16", "INT8", "INT4"]

ParameterInfo: total, trainable, non_trainable, total_millions, total_billions,
                by_dtype_counts, by_dtype_bytes

ModelSummary:   device, model_class, pytorch_version

LayerConnectionInfo:  name, type, parameters, depth,
                      input_layers, output_layers, input_shape, output_shape

ConnectionAnalysisInfo: total_connections, connection_graph, analysis_method,
                         has_skip_connections, max_fan_in, max_fan_out

ArchitectureInfo: total_layers, layer_types_count, layer_details, max_depth,
                   connections (optional)

AttentionLayerAnalysisInfo: num_attention_layers, attention_layers

MemoryEstimate:  precision, bytes_per_parameter, model_weights_mb, kv_cache_mb,
                  activation_memory_mb, gradient_memory_mb, optimizer_memory_mb,
                  training_memory_mb, total_memory_mb, total_memory_gb,
                  total_inference_memory_mb

MemoryEstimationInfo: estimates (dict por PrecisionType), base_parameters

AnalyzeConnections: input_shape, sample_input (optional)
EstimateMemory: sequence_length, batch_size (com defaults)

LLMInfo: architecture, attention_layers, parameters, summary,
          memory_estimation (optional)
```

### 2.3 `inference.py`

```python
InferencePerformanceInfo: avg_time, std_time, min_time, p5_time, p50_time,
                           p95_time, p99_time, max_time, tokens_per_second,
                           num_runs, generated_tokens, raw_times

CompareBenchmark: speedup, tps_improvement_percent
```

### 2.4 `benchmark.py`

```python
ComputePerplexityForBatchReturn: perplexities (list), mean_perplexity

ComputePerplexityForDatasetReturn: all_perplexities, mean_perplexity, stats (optional)

PerplexityTestResult: test_name, result

AccuracyTestResult: test_name, metric_name, accuracy, num_examples,
                    details, per_example_scores, stats (optional)

DescriptiveStats (types/stats.py): mean, std, min, max, p5, p50, p95, p99
```

---

## 3. Adapter — `ComprehensiveProfilingAdapter`

**Arquivo:** `profiling/adapters/comprehensive.py`

**Responsabilidade:** Orquestrar todos os serviços de profiling em uma única chamada e produzir um `ProfilingResult` completo.

### 3.1 `ComprehensiveProfilingConfig`

**Campos:**
```python
name: str = "comprehensive"
description: str = ""
prompt: str = "The quick brown fox jumps over the lazy dog."
max_new_tokens: int = 100
benchmark_tests: Set[str] = {"lambada"}
limit_test_samples: int | None = None

# Feature flags
run_hardware_profile: bool = True
run_llm_profile: bool = True
run_inference_benchmark: bool = True
run_model_benchmark: bool = True
estimate_memory: bool = True

# Herdados de ProfilingConfig
num_warmup_runs: int = 3
num_benchmark_runs: int = 10
```

### 3.2 `profile(model, stage_label)` — Fluxo

```python
def profile(self, model, stage_label) -> ProfilingResult:
    extra: dict[str, Any] = {}

    # 1. Hardware
    if run_hardware_profile:
        hw = HardwareProfiler().retrieve_hardware_information()
        extra["hardware"] = hw.model_dump()

    # 2. LLM static profile
    if run_llm_profile:
        llm_p = LLMProfiler(model, tokenizer).profile_complete(
            estimate_memory=EstimateMemory() if estimate_memory else None
        )
        extra["llm_profile"] = llm_p.model_dump()

    # 3. Inference benchmark
    if run_inference_benchmark:
        inf = InferencePerformanceBenchmarker().time_inference(...)
        extra["inference"] = inf.model_dump()
        # Popula LatencyMetrics

    # 4. Model benchmark
    if run_model_benchmark and benchmark_tests:
        results = ModelPerformanceBenchmarker().benchmark(...)
        extra["benchmarks"] = [r.model_dump() for r in results]
        # Extrai perplexity ou task_score para AccuracyMetrics

    # Monta e retorna ProfilingResult
    return ProfilingResult(
        profiler_name=name,
        model_label=type(model).__name__,
        pipeline_stage=stage_label,
        memory=MemoryMetrics(...),
        latency=LatencyMetrics(...),
        accuracy=AccuracyMetrics(...),
        extra=extra,
    )
```

---

## Riscos e Limitações

1. **`analyze_connections()` pode falhar silenciosamente** — se torch.fx e hooks ambos falharem, retorna graph sem edges (análise "basic").

2. **`_estimate_kv_cache_memory()` usa heurística** — assume que todos os attention layers têm a mesma configuração. Pode super/subestimar para arquiteturas não-padrão.

3. **Perplexity usa `max_length=512`** hardcoded na tokenização em `_compute_perplexity_for_batch`.

4. **`_get_max_seq_length()`** em `HFTrainerAdapter` tem heurística limited — pode não detectar corretamente para todos os modelos.

5. **Hardware profiler é Linux-only** — em macOS/Windows, levanta `PlatformNotSupportedException`.

6. **`torch.cuda.memory_allocated()` snapshot** — se chamado após o garbage collector rodar, pode mostrar valores incorretos.

7. **`ModelPerformanceBenchmarker.register()`** é runtime — se um teste registrado manualmente tiver assinatura diferente, `benchmark()` pode falhar.

8. **Sem validação de dataset split** — `_load_dataset()` tenta fallback splits ("validation", "test", "train") se o split original falhar, mas sem controle claro de qual foi usado.

9. **GSM8K/Math500 parser** (`_extract_last_number`) extrai apenas o último número — pode falhar em problemas com múltiplas etapas numéricas.

10. **`humaneval_pass1` / `mbpp_pass1`** executam código gerado via `exec()` — risco de segurança em ambientes não isolados.

---

## 4. Adapter — `LmEvalAdapter`

**Arquivo:** `adapters/profiling/lm_eval_adapter.py`

**Responsabilidade:** Bridge para o **EleutherAI lm-evaluation-harness** (`lm_eval`). Executa benchmarks padronizados (MMLU-Pro, Hellaswag, GSM8K, etc.) através da API `simple_evaluate()` e normaliza os resultados para `list[ProfilingResult]`.

**Dependência opcional:** `lm_eval[hf]` — instalável via `pip install 'lm_eval[hf]'`.

**Arquitetura de auto-descoberta:** lm-eval mantém seu próprio registry interno de tasks. O adapter apenas delega tasks names para `simple_evaluate()`, que os resolve. Não há registro manual de methods como no `ModelPerformanceBenchmarker`.

### 4.1 `LmEvalConfig`

```python
class LmEvalConfig(ProfilingConfig):
    name: str = "lm-eval"
    tasks: list[str] = Field(default_factory=lambda: ["hellaswag"])
    num_fewshot: int = 0                    # few-shot examples (0 = zero-shot)
    limit: int | float | None = None         # cap examples per task
    model_type: str = "hf"                    # backend: "hf", "vllm", "sglang", etc.
    model_args: str = ""                       # comma-separated k=v for model constructor
    no_cache: bool = True                     # disable result caching
    device: str | None = None                 # device override (None = auto)
    batch_size: int | str | None = None       # int, "auto", or None
    bootstrap_iters: int = 100_000            # bootstrap iterations for stderr
```

**Tasks suportadas (exemplos — lm-eval cobre 200+ tasks):**

| Task | lm-eval name | Tipo | Few-shot |
|---|---|---|---|
| MMLU-Pro | `mmlu_pro` | Multiple-choice (STEM) | 0 |
| GSM8K | `gsm8k` | Math (exact-number match) | 0 |
| MATH-500 | `math_500` | Math | 0 |
| ARC-C | `arc_challenge` | Multiple-choice (science) | 0 |
| Hellaswag | `hellaswag` | Multiple-choice (commonsense) | 0 |
| LAMBADA | `lambada_openai` | Fill-in-blank perplexity | 0 |
| TruthfulQA MC2 | `truthfulqa_mc2` | Multiple-choice (truthfulness) | 0 |
| SQuAD | `squad` | Extractive QA (F1) | 0 |
| BIG-Bench-Hard | `bigbench_hard` | Miscellaneous | 0 |
| HumanEval Pass@1 | `humaneval_pass1` | Code (Pass@1 via exec) | 0 |
| MBPP Pass@1 | `mbpp_pass1` | Code (Pass@1 via exec) | 0 |

**Tasks já cobertas pelo `ModelPerformanceBenchmarker` interno:**
- `lambada` (via HF dataset cimec/lambada)
- `wikitext_103_v1`, `c4` (perplexity)
- `squad` (via HF dataset rajpurkar/squad)
- `gsm8k`, `math_500`, `arc_challenge`, `hellaswag`, `truthfulqa_mc2`

**Nota:** `LmEvalAdapter` e `ModelPerformanceBenchmarker` são **independentes**. Ambos podem estar presentes no mesmo pipeline. Use `LmEvalAdapter` para tasks que não são cobertas pelo benchmarker interno, ou para ter acesso ao ecosistema completo de tasks do lm-eval.

### 4.2 `LmEvalAdapter`

```python
class LmEvalAdapter(ProfilingPort):
    def __init__(
        self,
        config: LmEvalConfig,
        tokenizer: Any = None,
        model_handle: Any = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model_handle = model_handle

    def profile(
        self, model: Any, stage_label: str
    ) -> list[ProfilingResult]:
        # Lazy import de lm_eval
        # Resolve model input (PreTrainedModel direto ou str path)
        # Chama simple_evaluate(tasks=[...])
        # Retorna list[ProfilingResult] — um por (task, metric) pair
```

### 4.3 Model Resolution (Híbrido 1A)

O adapter aceita três formas de modelo:

```python
# 1. PreTrainedModel direto (in-memory, após compressão in-place)
#    → lm-eval recebe o objeto diretamente
adapter.profile(compressed_model, stage_label="Post-GPTQ")

# 2. Path string
#    → lm-eval carrega do disco com "pretrained=<path>"
adapter.profile("./output_model", stage_label="Post-Healing")

# 3. None (usa model_args do config)
#    → lm-eval usa model_args diretamente
adapter.profile(None, stage_label="Evaluation")
```

### 4.4 Fluxo de `profile()`

```
profile(model, stage_label)
    │
    ├─→ _lazy_import_lm_eval()   # deferred until first call
    │
    ├─→ _resolve_model(model)
    │       PreTrainedModel → model direto (no reload)
    │       str            → "pretrained=<path>"
    │       None          → usa model_args do config
    │
    ├─→ simple_evaluate(
    │       model=model_arg,
    │       model_args=model_kwargs,
    │       tasks=["mmlu_pro", "hellaswag", ...],
    │       num_fewshot=...,
    │       limit=...,
    │       ...
    │   )
    │
    └─→ _normalize_results(results, stage_label)
            lm-eval results structure:
                {
                  "results": {
                    "mmlu_pro":  {"acc": 0.512, "acc_stderr": 0.007},
                    "hellaswag": {"acc": 0.643, "acc_stderr": 0.009},
                  },
                  "versions": {...},
                  "config": {...},
                }
            Para cada task × metric:
                → TaskMetrics(task_name, metric_name, value, stderr)
                → AccuracyMetrics(task_metrics=[TaskMetrics(...)])
                → ProfilingResult(task_metric)
            Retorna list[ProfilingResult]
```

### 4.5 Integração com o DAG

O executor detecta que `LmEvalAdapter.profile()` retorna `list[ProfilingResult]` e faz `extend()`:

```python
case "profile":
    result_or_list = port.profile(model, stage_label=step.label)
    if isinstance(result_or_list, list):
        profiling_records.extend(result_or_list)  # ← multi-result case
    else:
        profiling_records.append(result_or_list)  # ← single-result case (backward compat)
```

Cada `ProfilingResult` gera uma linha separada na tabela Markdown:

```
| Stage           | Latency p95 (ms) | GPU Peak (MB) | Metrics                    |
| --------------- | ---------------- | ------------- | -------------------------- |
| Baseline-mmlu   | 0.0              | —             | mmlu_pro/acc=0.5120 ±0.007 |
| Baseline-hella  | 0.0              | —             | hellaswag/acc=0.6432 ±0.009 |
| Post-GPTQ-mmlu  | 0.0              | —             | mmlu_pro/acc=0.4871 ±0.008 |
| Post-GPTQ-hella | 0.0              | —             | hellaswag/acc=0.6011 ±0.010 |
```

### 4.6 Exemplo de Uso

```python
from llm_flux.adapters.profiling import LmEvalAdapter, LmEvalConfig
from llm_flux.core.pipeline import Pipeline, PipelineStep
from llm_flux.adapters.model import HFModelHandle
from llm_flux.core.model import ModelSource

# Model
model_handle = HFModelHandle(
    source=ModelSource(identifier="Qwen/Qwen2-0.5B-Instruct")
)

# Profilers — um por task para granularidade
mmlu_profiler = LmEvalAdapter(
    config=LmEvalConfig(
        name="lm-eval-mmlu",
        tasks=["mmlu_pro"],
        num_fewshot=0,
        model_args="pretrained=Qwen/Qwen2-0.5B-Instruct",
    )
)

hella_profiler = LmEvalAdapter(
    config=LmEvalConfig(
        name="lm-eval-hellaswag",
        tasks=["hellaswag"],
        num_fewshot=0,
        model_args="pretrained=Qwen/Qwen2-0.5B-Instruct",
    )
)

gsm8k_profiler = LmEvalAdapter(
    config=LmEvalConfig(
        name="lm-eval-gsm8k",
        tasks=["gsm8k"],
        num_fewshot=0,
        model_args="pretrained=Qwen/Qwen2-0.5B-Instruct",
    )
)

# Pipeline
pipeline = Pipeline(
    name="qwen-compression-study",
    steps=[
        PipelineStep("Load Model", model_handle),
        PipelineStep("MMLU-Pro Baseline", mmlu_profiler),
        PipelineStep("Hellaswag Baseline", hella_profiler),
        PipelineStep("GSM8K Baseline", gsm8k_profiler),
        # ... compressão ...
        PipelineStep("MMLU-Pro Post-Compression", mmlu_profiler),
        PipelineStep("Hellaswag Post-Compression", hella_profiler),
        PipelineStep("GSM8K Post-Compression", gsm8k_profiler),
    ]
)
```

### 4.7 Lazy Import

`lm_eval` é importado apenas quando `LmEvalAdapter` é usado:

```python
class SimpleEvaluateWrapper:
    """Deferred import de lm_eval.simple_evaluate até primeiro uso."""
    def __call__(self, **kwargs):
        from lm_eval import simple_evaluate
        self._fn = simple_evaluate
        return self._fn(**kwargs)
```

Isso significa que o framework **funciona sem `lm-eval`** instalado — só quebra se alguém tentar usar `LmEvalAdapter`.

### 4.8 Riscos e Limitações

1. **`lm-eval` não instalado** — se `LmEvalAdapter` for usado sem `pip install 'lm_eval[hf]'`, um `ImportError` é levantado com instrução de instalação.

2. **Task names inválidos** — se um task name não existir no lm-eval, `simple_evaluate()` pode falhar ou retornar resultados vazios. O adapter loga um warning se nenhum metrics for parsed.

3. **Sem latência nativa** — `LmEvalAdapter` não mede latência de inferência (latency fields são zeros). Para latência, use `ComprehensiveProfilingAdapter` ou `InferencePerformanceBenchmarker`.

4. **Sem hardware profile nativo** — similarmente, memory fields são zeros. Para hardware info, use `ComprehensiveProfilingAdapter`.

5. **Modelo em memória vs. path** — se `model` é `PreTrainedModel` após compressão in-place, o modelo é passado diretamente. lm-eval pode não suportar todos os wrappers/modificadores. Para modelos com quantization in-place (ex: bitsandbytes), prefira passar o path se o modelo foi salvo.

6. **vLLM/SGLang batch_size** — para backends vLLM/SGLang, `batch_size` deve ser `"auto"` ou omitido, não um int. O adapter não faz validação específica por backend.

7. **`no_cache=True` default** — caching do lm-eval é desabilitado por default para evitar dependência de sistema de arquivos. Para datasets grandes, isso pode resultar em re-download a cada execução.

