# Workflows — Fluxos de Execução do Framework

## Visão Geral

Este documento descreve os principais fluxos de execução do LLM-Flux, desde o uso básico até workflows avançados de pesquisa.

---

## 1. Workflow Principal — Pipeline Completo

### 1.1 Definição

O usuário define um `Pipeline` com passos declarativos, cada passo sendo um `PipelineStep` com um `label` e um `port` (instância de port).

### 1.2 Execução via `run_pipeline()`

```python
from llm_flux.runner import run_pipeline

result = run_pipeline(
    pipeline=pipeline,
    dag_output="pipeline_dag.png",
    result_output="result.json",
    html_report_output="report.html",
    skip_confirmation=False,
    show_dag=True,
)
```

**Fluxo interno:**
```
1. build_dag(pipeline) → nx.DiGraph
2. render_dag(dag) → PNG (show ou save)
3. render_dag_echarts(dag) → dict (para HTML)
4. print step summary
5. confirmation prompt [y/N]
6. PipelineExecutor(dag).run()
   ├─ topological sort
   ├─ para cada nó:
   │   ├─ LOAD: model = port.load()
   │   ├─ COMPRESS: model = port.compress(model)
   │   ├─ PROFILE: result = port.profile(model, label) → profiling_records
   │   └─ HEAL: model = port.heal(model)
   └─ retorna PipelineRunResult
7. print result.to_markdown_table()
8. print result.to_dissertation_log()
9. result.save_json(output)
10. result.save_html_report() (se solicitado)
```

### 1.3 Resultado

- `PipelineRunResult` com todos os `ProfilingResult`
- Tabela Markdown para dissertação
- JSON com dados completos
- HTML interativo opcional

---

## 2. Workflow de Experimento — BitsAndBytes + LoRA

### 2.1 Cenário

Pesquisador quer avaliar impacto de quantização NF4 e recuperação via LoRA.

### 2.2 Passos

1. **Carregar modelo base** — `HFModelHandle` do HuggingFace Hub
2. **Profile baseline** — `ComprehensiveProfilingAdapter` (com hardware profile)
3. **Aplicar NF4** — `BitsAndBytesAdapter`
4. **Profile pós-compressão** — `ComprehensiveProfilingAdapter` (sem hardware profile)
5. **Aplicar LoRA healing** — `HFTrainerAdapter` com dataset Alpaca
6. **Profile pós-healing** — `ComprehensiveProfilingAdapter` (sem hardware profile)

### 2.3 Configuração

```python
# Model
model_handle = HFModelHandle(source=ModelSource(identifier="Qwen/Qwen2-0.5B-Instruct"))

# Profilers (baseline vs. fast)
baseline_profiler = ComprehensiveProfilingAdapter(
    config=ComprehensiveProfilingConfig(
        name="profiler-baseline",
        benchmark_tests={"lambada"},
        run_hardware_profile=True,  # Only on baseline
    ),
    tokenizer=model_handle,
)

fast_profiler = ComprehensiveProfilingAdapter(
    config=ComprehensiveProfilingConfig(
        name="profiler-fast",
        benchmark_tests={"lambada"},
        run_hardware_profile=False,  # Hardware unchanged
    ),
    tokenizer=model_handle,
)

# Compression
bnb = BitsAndBytesAdapter(
    config=BitsAndBytesConfig(name="NF4", load_in_4bit=True),
    model_handle=model_handle,
)

# Healing
healer = HFTrainerAdapter(
    config=HealingConfig(
        dataset=DatasetConfig(source="tatsu-lab/alpaca", max_samples=1000),
        max_steps=200,
        lora=LoRAConfig(r=16),
    ),
    tokenizer=model_handle,
)

# Pipeline
pipeline = Pipeline(
    name="bnb-nf4-compression-study",
    steps=[
        PipelineStep("Load Model", model_handle),
        PipelineStep("Baseline", baseline_profiler),
        PipelineStep("NF4 Quantization", bnb),
        PipelineStep("Post-NF4", fast_profiler),
        PipelineStep("LoRA Recovery", healer),
        PipelineStep("Post-Healing", fast_profiler),
    ]
)
```

### 2.4 Output Esperado

- 3 tabelas Markdown (baseline, post-NF4, post-healing)
- Comparison: perplexity pré/pós compressão e pós-healing
- Speedup de inferência pós-compressão
- Memory reduction footprint

---

## 3. Workflow de Validação Rápida (CPU)

### 3.1 Propósito

Validação lightweight sem GPU, usando tiny model e first-N samples.

### 3.2 Cenário de Uso

CI/CD ou quick iteration antes de rodar em GPU real.

### 3.3 Características

- Modelo: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- `limit_test_samples = FIRST_N` (e.g., 10-100 samples)
- Benchmark tests: `lambada`, `squad`, `arc_challenge`, `gsm8k`, `truthfulqa_mc2`
- `skip_confirmation = True` (para automação)

### 3.4 Exemplo

```bash
uv run python examples/e2e_option_c_validation.py
```

Este exemplo implementa Option C (Perplexity + QA first) — uma estratégia de validação incremental.

---

## 4. Workflow de Quantização Seletiva

### 4.1 Cenário

Pesquisador quer comparar GPTQ vs AWQ vs BitsAndBytes no mesmo modelo e dataset.

### 4.2 Estrutura

Cada técnica é um step de compressão separado, seguido de profiling:

```
Load → Profile → Compress_GPTQ → Profile → [Heal] → Profile
                        ↓
                   Compress_AWQ → Profile
                        ↓
                   Compress_BnB → Profile
```

**Nota:** O DAG é linear, então as compressões são sequenciais. O modelo é modificado após cada compressão, então não é um comparison fair. Um workflow de comparação fair exigiria:
- 3 pipelines separados (um por técnica)
- Mesmos pesos iniciais
- Mesmos datasets de calibração

### 4.3 Recomendação

Para comparison fair, usar 3 pipelines separados com o mesmo `seed` de random state.

---

## 5. Workflow de Depth Pruning

### 5.1 Cenário

Remover camadas de transformer redundantes via distância angular.

### 5.2 Passos

1. Load modelo
2. Calibration dataset pass (captura inputs/outputs de cada camada)
3. Cálculo de distância angular por camada
4. Remoção das camadas menos transformativas (menor distância)
5. Atualização de `num_hidden_layers` no config
6. Re-indexação de `layer_idx` para evitar KV cache errors

### 5.3 Customização

```python
def custom_importance(model, input_ids):
    """Exemplo: magnitude-based em vez de angular distance."""
    layers = get_transformer_layers(model)
    return [layer.weight.abs().mean() for layer in layers]

adapter = DepthPruningAdapter(
    config=DepthPruningConfig(pruning_ratio=0.2),
    tokenizer=tokenizer,
    importance_fn=custom_importance,
)
```

---

## 6. Workflow de Heuristic — Auto-Detecção

### 6.1 Max Sequence Length

O framework auto-detecta `max_seq_length` na ausência de configuração explícita:

```python
# HFTrainerAdapter.heal()
if max_seq_length is None:
    max_seq_length = _get_max_seq_length(model, tokenizer)
    # tenta: tokenizer.model_max_length
    # depois: model.config.max_position_embeddings
    # depois: model.config.n_positions
    # depois: model.config.seq_length
    # fallback: 512
```

### 6.2 Dataset Adapter Selection

```python
if Path(source).exists():
    adapter = LocalDatasetAdapter(config)
else:
    adapter = HFDatasetAdapter(config)
```

### 6.3 Transformer Layer Detection

```python
# DepthPruningAdapter.get_transformer_layers()
if hasattr(model, "model") and hasattr(model.model, "layers"):
    return model.model.layers  # Llama/Mistral
if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
    return model.transformer.h  # GPT-2/Qwen
# fallback: percorre named_modules
```

---

## 7. Workflow de Extração de Resultados

### 7.1 Tabela Markdown

```python
result.to_markdown_table()
```

Output:
```
| Stage                  | Latency p95 (ms) | GPU Peak (MB) | Perplexity |
| ---------------------- | :--------------: | :-----------: | :--------: |
| Baseline Profiling     | 42.3             | 14,500        | 6.840      |
| Post-NF4 Profiling     | 11.2             | 5,200         | 12.340     |
| Post-Healing Profiling | 11.5             | 5,400         | 7.210      |
```

### 7.2 Log para Dissertação

```python
result.to_dissertation_log()
```

Output:
```
Experiment: bnb-nf4-compression-study
Objective:  Profiles baseline Qwen, quantizes to NF4, and runs LoRA healing.
Duration:   1234.5 s

Profiling checkpoint 1 — Baseline Profiling
  [Baseline Profiling] Latency p95=42.3 ms | GPU peak=14500 MB | PPL=6.840
Profiling checkpoint 2 — Post-NF4 Profiling
  [Post-NF4 Profiling] Latency p95=11.2 ms | GPU peak=5200 MB | PPL=12.340
Profiling checkpoint 3 — Post-Healing Profiling
  [Post-Healing Profiling] Latency p95=11.5 ms | GPU peak=5400 MB | PPL=7.210
```

### 7.3 JSON Completo

```python
result.save_json("experiment_result.json")
```

Contém `extra` com todos os raw dumps (hardware, llm_profile, inference, benchmarks).

### 7.4 HTML Interativo

```python
result.save_html_report(
    path="report.html",
    dag_image_path="pipeline_dag.png",
    dag_echarts_data=echarts_data,
)
```

Gera HTML autocontido com ECharts.

---

## 8. Fluxo de Extensão — Adicionar Nova Técnica

### 8.1 Compression

1. Criar `CompressionConfig` subclass em `adapters/compression/`
2. Criar `XxxAdapter(CompressionPort)` com `compress(model)`
3. Exportar em `adapters/compression/__init__.py`
4. Adicionar em `examples/` para demonstração

### 8.2 Benchmark

1. Adicionar método `test_name()` em `ModelPerformanceBenchmarker`
2. Retornar `PerplexityTestResult` ou `AccuracyTestResult`
3. O teste é automaticamente elegível via auto-registro

### 8.3 Healing

1. Criar `XxxHealingAdapter(HealingPort)` com `heal(model)`
2. Adicionar exports em `adapters/healing/__init__.py`

---

## 9. Riscos de Workflows

1. **Comparação entre técnicas** — o DAG linear não permite fan-out, então múltiplas técnicas não podem ser aplicadas ao mesmo modelo base simultaneamente.

2. **Healing após cada compressão** — se o pipeline tiver compressão → heal → compressão → heal, o segundo heal está fine-tunando sobre o modelo já healado, não o original.

3. **BitsAndBytes re-load** — o adapter re-carrega o modelo, perdendo qualquer modificação feita por compressões anteriores (e.g., depth pruning).

4. **Benchmark overlap** — se `benchmark_tests` incluir múltiplos testes, cada um é executado sequencialmente. Para pipelines com muitas etapas, o tempo total pode ser longo.

5. **Streaming datasets sem materialização** — HFTrainerAdapter materializa streaming datasets, mas compression adapters (GPTQ, AWQ, DepthPruning) não — eles iteram diretamente.

6. **Ordem de profiling** — se `run_hardware_profile=False` em etapas subsequentes, informações de hardware não são logadas. Se o hardware mudar (e.g., outra GPU), não há detecção.
