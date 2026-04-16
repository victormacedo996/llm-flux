# LLM-Flux

**Framework de metodologia de dissertação para pesquisa em compressão de LLMs.**

---

## Visão Geral

LLM-Flux é um framework de pesquisa que abstrai e orquestra técnicas de compressão, profiling e reparação (healing) de modelos de linguagem através de um pipeline definido por DAG (Directed Acyclic Graph). O framework é construído sobre uma arquitetura de **ports e adapters** (hexagonal), onde:

- **Ports** (`llm_flux/core/`) definem interfaces puras em Python — sem dependências de ML.
- **Adapters** (`llm_flux/adapters/`) implementam essas interfaces com tecnologias concretas (auto-gptq, autoawq, bitsandbytes, HuggingFace Trainer, etc.).
- O **DAG** (`llm_flux/dag/`) conecta os passos do pipeline e é executado topologicalmente.
- **Profiling** (`llm_flux/profiling/`) coleta métricas de latência, memória e acurácia.
- **Resultados** são serializados em JSON e exportados como tabelas Markdown prontas para dissertação.

## Estrutura do Repositório

```
llm_flux/
├── core/                    # Ports (interfaces puras em Python)
│   ├── compression.py       # CompressionPort, CompressionConfig
│   ├── healing.py           # HealingPort, HealingConfig, LoRAConfig
│   ├── model.py             # ModelHandle, ModelSource
│   ├── pipeline.py          # Pipeline, PipelineStep
│   ├── profiling.py         # ProfilingPort, ProfilingConfig, ProfilingResult
│   ├── results.py           # PipelineRunResult
│   └── html_reporter.py     # Gerador de relatório HTML interativo
│
├── adapters/                # Implementações concretas dos ports
│   ├── compression/         # GPTQ, AWQ, BitsAndBytes, DepthPruning
│   ├── healing/            # HFTrainerAdapter (HF Trainer + LoRA/QLoRA)
│   └── model/              # HFModelHandle (HuggingFace Hub + local)
│
├── datasets/               # DatasetPort + adapters (HF Hub, local)
│   ├── port.py             # DatasetConfig, DatasetPort
│   ├── huggingface.py      # HFDatasetAdapter
│   └── local.py           # LocalDatasetAdapter (JSON/JSONL/CSV/Parquet)
│
├── dag/                    # Builder, renderer, executor do DAG
│   ├── builder.py         # build_dag() → nx.DiGraph
│   ├── renderer.py        # render_dag() (PNG via matplotlib)
│   └── executor.py       # PipelineExecutor (walk topológico)
│
├── profiling/             # Serviços de domínio + tipos Pydantic
│   ├── llm_profiler.py       # Análise estrutural profunda de LLMs
│   ├── hardware_profiler.py   # CPU, GPU, RAM
│   ├── inference_benchmarker.py  # Latência de inferência
│   ├── model_benchmarker.py    # Perplexity + benchmarks de acurácia
│   ├── adapters/comprehensive.py  # ComprehensiveProfilingAdapter
│   └── types/             # Tipos Pydantic (hardware, llm, inference, benchmark)
│
├── runner.py             # run_pipeline() — ponto de entrada de alto nível
└── __init__.py
```

## Portas e Adaptadores

### Arquitetura de Portes

O framework segue o padrão **hexagonal** (ports and adapters):

| Port (interface) | Arquivo | Responsabilidade |
|---|---|---|
| `ModelHandle` | `core/model.py` | Carregar/descarregar modelo de qualquer fonte |
| `CompressionPort` | `core/compression.py` | Aplicar técnica de compressão |
| `ProfilingPort` | `core/profiling.py` | Coleta de métricas (latência, memória, acurácia) |
| `HealingPort` | `core/healing.py` | Reparação de modelo via fine-tuning |
| `DatasetPort` | `datasets/port.py` | Carregar dataset de qualquer fonte |

### Adaptadores Implementados

**Model Loading:**
- `HFModelHandle` — HuggingFace Hub ou diretório local via `AutoModelForCausalLM`

**Compression:**
- `GPTQAdapter` — Post-training quantization via `auto-gptq`
- `AWQAdapter` — Activation-Aware Weight Quantization via `autoawq`
- `BitsAndBytesAdapter` — 4-bit NF4 ou 8-bit INT8 via `bitsandbytes`
- `DepthPruningAdapter` — Depth Pruning baseado em distância angular

**Healing:**
- `HFTrainerAdapter` — HuggingFace Trainer com suporte LoRA/QLoRA via PEFT

**Datasets:**
- `HFDatasetAdapter` — HuggingFace Hub datasets
- `LocalDatasetAdapter` — Arquivos locais (JSON, JSONL, CSV, Parquet)

**Profiling:**
- `ComprehensiveProfilingAdapter` — Orquestra todos os profilers em um único `ProfilingResult`

## Técnicas de Compressão Suportadas

| Técnica | Módulo | Tipo | Biblioteca |
|---|---|---|---|
| GPTQ | `adapters/compression/gptq.py` | Quantização PTQ | auto-gptq |
| AWQ | `adapters/compression/awq.py` | Quantização PTQ | autoawq |
| BitsAndBytes NF4 | `adapters/compression/bitsandbytes.py` | Quantização PTQ | bitsandbytes |
| Depth Pruning | `adapters/compression/depth_pruning.py` | Pruning estrutural | custom (hooks) |

## Métricas e Benchmarks

### Benchmarks de Perplexidade
- `lambada` (cimec/lambada)
- `wikitext_103_v1` (Salesforce/wikitext)
- `c4` (allenai/c4)

### Benchmarks de Acurácia
- **Multiple-choice:** mmlu_pro, arc_challenge, hellaswag, winogrande, piqa, sciq, truthfulqa_mc2
- **Extractive QA:** squad
- **Math:** gsm8k (extração de número), math_500
- **Generation:** big_bench_hard
- **Code:** humaneval_pass1, mbpp_pass1

### Métricas Coletadas
- **Latência:** mean, std, min, p5, p50, p95, p99, max (ms)
- **Memória:** peak_gpu_mb, peak_cpu_mb
- **Acurácia:** perplexity ou task_score + task_name

## Dependências

### Core (sempre instaladas)
- `pydantic>=2.7` — Validação de configurações e tipos
- `networkx>=3.3` — Construção e manipulação do DAG
- `matplotlib>=3.9` — Renderização do DAG em PNG
- `torch>=2.3` — Backend de ML
- `transformers>=4.42` — Modelos e tokenizers
- `datasets>=2.20` — Carregamento de datasets
- `peft>=0.12` — LoRA/QLoRA via PEFT
- `psutil>=6.0` — Informações de hardware
- `loguru>=0.7` — Logging
- `numpy>=1.26` — Operações numéricas

### Extras Opcionais
```
gptq         = ["auto-gptq>=0.7"]
awq          = ["autoawq>=0.2"]
bitsandbytes = ["bitsandbytes>=0.43"]
cli          = ["typer>=0.12"]
dev          = ["pytest>=8.2", "pytest-mock>=3.14", "ruff>=0.5", "mypy>=1.10"]
```

## Conceitos Centrais

### Pipeline Declarativo

O usuário define um `Pipeline` com passos declarativos (`PipelineStep`). Cada passo contém um `label` (string legível) e um `port` (instância de port). A ordem de execução é determinada pela ordem na lista — o primeiro passo deve ser sempre um `ModelHandle` (LOAD).

```python
pipeline = Pipeline(
    name="bnb-nf4-compression-study",
    steps=[
        PipelineStep(label="Load Model",          port=model_handle),
        PipelineStep(label="Baseline Profiling",  port=baseline_profiler),
        PipelineStep(label="Apply NF4 Quantization", port=bnb),
        PipelineStep(label="Post-NF4 Profiling",  port=fast_profiler),
        PipelineStep(label="LoRA Recovery",       port=healer),
        PipelineStep(label="Post-Healing Profiling", port=fast_profiler),
    ]
)
```

### Fluxo de Dados no DAG

```
LOAD → PROFILE → COMPRESS → PROFILE → HEAL → PROFILE
│        │          │          │        │         │
│   (model)    (model)    (model)   (model)   (model)
│                                                │
                                    PipelineRunResult
                                   (profiling_records[])
```

- **LOAD:** `ModelHandle.load()` → retorna modelo cru (PyTorch `nn.Module`)
- **COMPRESS:** `CompressionPort.compress(model)` → retorna modelo compressado
- **PROFILE:** `ProfilingPort.profile(model, stage_label)` → retorna `ProfilingResult`
- **HEAL:** `HealingPort.heal(model)` → retorna modelo re-treinado

O estado do modelo é threadado através do walk topológico: LOAD inicializa, COMPRESS/HEAL substituem, PROFILE lê.

### Healing e LoRA/QLoRA

A etapa de healing (reparação) compõe com compressão: após comprimir um modelo, a qualidade (perplexidade) geralmente degrada. O healer aplica fine-tuning — tipicamente com LoRA — para recuperar a qualidade usando um dataset de calibração.

**Hipótese central:** compressão (quantização) degrada qualidade → healing via fine-tuning recupera qualidade sem reintroduzir o custo de memória/velocidade original.

## Quick Start

```bash
# Instalação
uv sync

# Exemplo completo: compressão BitsAndBytes + LoRA healing
uv run python examples/bnb_compression_study.py

# Validação leve (CPU, tiny model, first-N samples)
uv run python examples/e2e_option_c_validation.py
```

## Output do Pipeline

Após execução, `run_pipeline()` produz:

1. **DAG PNG** — visualização do grafo de pipeline (`pipeline_dag.png`)
2. **JSON** — resultados completos com todos os `extra` dumps (`{pipeline.name}_result.json`)
3. **Console** — tabela Markdown e log de experimento para copy-paste na dissertação
4. **HTML (opcional)** — relatório interativo com Apache ECharts

## Hipóteses e Lacunas de Informação

### Hipóteses
1. O modelo é sempre um Causal LM (arquitetura decoder-only). Não há suporte explícito para encoders ou modelos de sequência.
2. O primeiro passo do pipeline deve ser sempre LOAD (validado por `Pipeline._must_start_with_load`).
3. O healer (fine-tuning) sempre recebe o tokenizer do `HFModelHandle` via `get_tokenizer()`.
4. Benchmarks de acurácia são auto-descobertos via `hasattr(self, test_name)` — novos métodos adicionados à classe são automaticamente elegíveis.

### Lacunas Identificadas
1. `mlp_pruning.py` está vazio (0 bytes) — parece ser um placeholder não implementado.
2. A versão exata de `transformers` com suporte completo a todas as features não está documentada.
3. Não há testes de integração no repositório (apenas pytest básico em dev dependencies).
4. O mecanismo de "extension-playbook" mencionado no README.md existe em `docs/technical/extension-playbook.md` mas não foi verificado.
