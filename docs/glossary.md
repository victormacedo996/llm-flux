# Glossary — Glossário de Termos e Conceitos

## Visão Geral

Este documento define os termos técnicos usados no framework LLM-Flux, organizados por domínio.

---

## Conceitos de Compressão

### PTQ (Post-Training Quantization)

**Definição:** Quantização aplicada após o treinamento do modelo, sem re-treinamento.

**Contexto no framework:** GPTQ, AWQ e BitsAndBytes são técnicas PTQ.

**Referências:** Oposto de QAT (Quantization-Aware Training), que simula quantization durante o treinamento.

---

### QLoRA (Quantized Low-Rank Adaptation)

**Definição:** Técnica de fine-tuning eficiente que combina quantização (tipicamente NF4) com LoRA.

**Contexto no framework:** BitsAndBytes (NF4) + LoRA (via PEFT) é a configuração de healing padrão para recovery de qualidade.

---

### NF4 (4-bit NormalFloat)

**Definição:** Tipo de quantização 4-bit proposto por QLoRA, otimizado para weights normalmente distributed.

**Contexto no framework:** Configurado via `bnb_4bit_quant_type="nf4"` em `BitsAndBytesConfig`.

---

### GPTQ (Generative Pre-Training Quantization)

**Definição:** Técnica de quantização post-training que usa aproximação de segunda ordem para calibrar quantized weights.

**Contexto no framework:** `GPTQAdapter` via `auto-gptq`.

---

### AWQ (Activation-Aware Weight Quantization)

**Definição:** Técnica de quantização que considera a distribuição de ativações (não apenas weights) para melhor calibragem.

**Contexto no framework:** `AWQAdapter` via `autoawq`.

---

### Group Size (GPTQ/AWQ)

**Definição:** Número de elementos weights em cada grupo de quantização. Group size menor = granularidade maior = qualidade melhor, mas overhead maior.

**Valor típico:** 128 (default em ambos).

---

### Calibration Dataset

**Definição:** Dataset usado para calibrar a quantização — o modelo passa por esse dataset para determinar as scales de quantização.

**Contexto no framework:** `calibration_dataset: DatasetConfig` em `GPTQConfig`, `AWQConfig`, `DepthPruningConfig`.

---

### desc_act (GPTQ)

**Definição:** Se True, orderna a quantização por activation order em vez de magnitude. Útil para modelos com padrões de ativação específicos.

**Default:** False.

---

### Depth Pruning (Poda de Profundidade)

**Definição:** Remoção de camadas inteiras (typicamente transformer layers) com base em métricas de importância.

**Contexto no framework:** `DepthPruningAdapter` — remove camadas com menor "distância angular" entre input e output.

---

### Angular Distance (Distância Angular)

**Definição:** Métrica de importância de camada calculada como `(1/π) * arccos(cosine_similarity(input, output))`. Distância próxima de 0 indica que a camada transforma pouco.

**Contexto no framework:** Usada em `DepthPruningAdapter` para determinar quais camadas remover.

---

## Conceitos de Profiling e Métricas

### Perplexity (PPL)

**Definição:** Métrica de qualidade de language models — `exp(cross_entropy)`. Perplexidade baixa = modelo melhor.

**Fórmula:** `PPL = exp(-(1/N) * sum(log_p(token_i)))`

**Contexto no framework:** Benchmarks `lambada`, `wikitext_103_v1`, `c4` computam perplexidade.

---

### Latency Metrics

**Definição:** Estatísticas de latência de inferência em milissegundos.

**Métricas no framework:**
| Métrica | Descrição |
|---|---|
| mean_ms | Média |
| std_ms | Desvio padrão |
| min_ms | Mínimo |
| p5_ms | Percentil 5 |
| p50_ms | Percentil 50 (mediana) |
| p95_ms | Percentil 95 |
| p99_ms | Percentil 99 |
| max_ms | Máximo |

---

### Tokens Per Second (TPS)

**Definição:** Throughput de geração — número de tokens gerados por segundo.

**Cálculo:** `generated_tokens / avg_inference_time`

---

### Speedup

**Definição:** Razão de speedup entre dois modelos: `original_latency / compressed_latency`.

**Valores:** > 1 significa compressed é mais rápido.

---

### KV Cache

**Definição:** Cache de key-value activations das attention layers durante generation. Reduz computation mas aumenta memory usage.

**Estimativa no framework:** `_estimate_kv_cache_memory()` em `LLMProfiler`.

---

### Memory Estimation Teórica

**Definição:** Cálculo de memory footprint baseado em parâmetros + ativações + KV cache + gradients + optimizer states.

**Precisions suportadas:** FP32, FP16, BF16, INT8, INT4

---

## Conceitos de Fine-Tuning e Healing

### LoRA (Low-Rank Adaptation)

**Definição:** Técnica de parameter-efficient fine-tuning (PEFT) que injeta rank-decomposed matrices em camadas de atenção.

**Parâmetros:**
| Parâmetro | Descrição | Default |
|---|---|---|
| r | Rank da decomposição | 8 |
| lora_alpha | Fator de escala | 32 |
| target_modules | Camadas-alvo | ["q_proj", "v_proj"] |
| lora_dropout | Dropout | 0.05 |

---

### QLoRA (no healing context)

**Definição:** Combinação de NF4 quantization + LoRA para fine-tuning memória-eficiente.

**Contexto no framework:** Healing com `BitsAndBytesAdapter` (compressão NF4) + `HFTrainerAdapter` (LoRA fine-tuning).

---

### Healing (Reparação)

**Definição:** Processo de fine-tuning de um modelo compressado para recuperar qualidade perdida na compressão.

**Nota:** O framework usa "healing" como conceito de pesquisa — não é um termo padrão da literatura.

---

### PEFT (Parameter-Efficient Fine-Tuning)

**Definição:** Família de técnicas para fine-tuning eficiente — LoRA, Adapters, Prompt Tuning, etc.

**Contexto no framework:** Suporte via `peft` library.

---

## Conceitos de Benchmark

### MC2 (TruthfulQA)

**Definição:** Metric para TruthfulQA — `P(true) / (P(true) + P(false))` onde P é a probability mass assigned pelo modelo.

**Contexto no framework:** `_truthfulqa_mc2()` em `ModelPerformanceBenchmarker`.

---

### Pass@1 (Code Generation)

**Definição:** Probabilidade de o primeiro código gerado passar em todos os tests.

**Cálculo:** `count(passed) / total`

**Contexto no framework:** `humaneval_pass1` e `mbpp_pass1` em `ModelPerformanceBenchmarker`.

---

### Token F1 (SQuAD)

**Definição:** F1 token-level entre prediction e gold answers — max sobre todos os gold texts.

**Contexto no framework:** `_squad_f1()` em `ModelPerformanceBenchmarker`.

---

### Multiple Choice Accuracy

**Definição:** Métrica para benchmarks de escolha múltipla — soma de acertos / total.

**Método:** `_choice_logprob()` calcula log-probability de cada opção via `log_softmax + gather`.

---

## Conceitos de Arquitetura

### Port (Interface)

**Definição:** Interface abstrata que define um contrato no padrão hexagonal.

**Exemplos no framework:**
- `CompressionPort`
- `ProfilingPort`
- `HealingPort`
- `ModelHandle`
- `DatasetPort`

---

### Adapter (Implementação Concreta)

**Definição:** Implementação concreta de uma Port usando uma biblioteca específica.

**Exemplos:**
- `GPTQAdapter` implementa `CompressionPort` via `auto-gptq`
- `HFTrainerAdapter` implementa `HealingPort` via `transformers.Trainer`
- `HFModelHandle` implementa `ModelHandle` via `transformers.AutoModelForCausalLM`

---

### DAG (Directed Acyclic Graph)

**Definição:** Grafo direcionado acíclico representando a ordem de execução do pipeline.

**Contexto no framework:** Construído via `build_dag()` → executado via `PipelineExecutor.run()`.

---

### Pipeline Declarativo

**Definição:** Pipeline definido como dados (lista de `PipelineStep`) em vez de código (callables).

**Vantagem:** Serializável, validável antes da execução.

---

### Topological Sort

**Definição:** Ordenação de nós de um DAG onde todo nó vem depois de suas dependências.

**Uso no framework:** `nx.topological_sort(self.dag)` no `PipelineExecutor.run()`.

---

### NetworkX DiGraph

**Definição:** Estrutura de grafo do NetworkX usada para representar o pipeline.

**Nós:** cada passo do pipeline
**Arestas:** dependência sequencial linear

---

## Conceitos de Modelo

### Causal LM (Decoder-only)

**Definição:** Language model autoregressivo que prediction tokens sequencialmente usando apenas contexto left-to-right.

**Contexto no framework:** Todos os modelos suportados são causal LMs (Llama, Qwen, Mistral, GPT-2).

---

### HuggingFace Hub

**Definição:** Repositório de modelos e datasets da HuggingFace (huggingface.co).

**Contexto no framework:** `HFModelHandle` e `HFDatasetAdapter` carregam do Hub.

---

### ModelHandle

**Definição:** Port abstrata para carregar/descarregar modelos de qualquer fonte.

**Implementação:** `HFModelHandle` (Hub + local)

---

### Transformer Layers

**Definição:** Blocos de atenção/multi-layer perceptron que formam o core de LLMs modernos.

**Contexto no framework:** `get_transformer_layers()` heuristics para Llama (`model.model.layers`) e GPT-2 (`model.transformer.h`).

---

### Layer Index (layer_idx)

**Definição:** Índice numérico da camada de atenção dentro de um transformer block.

**Importância:** Usado internamente para KV cache indexing. Deve ser atualizado após depth pruning para evitar index errors.

---

## Conceitos de Sistema

### CUDA

**Definição:** Plataforma de computação paralela da NVIDIA.

**Contexto no framework:** `torch.cuda` API para GPU info e memory tracking.

---

### PyTorch Module (nn.Module)

**Definição:** Classe base para todos os módulos PyTorch (layers, models).

**Contexto no framework:** Retorno de `ModelHandle.load()` é tipicamente `nn.Module`.

---

### torch.fx

**Definição:** Framework de PyTorch para transformações de grafos computationais.

**Uso no framework:** `LLMProfiler._analyze_connections_fx()` para tracing de camadas.

---

### Forward Hook

**Definição:** Callback registrado em `nn.Module` executado durante forward pass.

**Uso no framework:** `LLMProfiler._analyze_connections_hooks()` e `DepthPruningAdapter.angular_distance_importance()`.

---

### Apache ECharts

**Definição:** Biblioteca JavaScript para visualização de dados.

**Uso no framework:** `html_reporter.py` gera HTML com gráficos ECharts para DAG interativo e métricas.

---

### Jinja2

**Definição:** Motor de templating Python.

**Uso no framework:** `html_reporter.py` usa Jinja2 para renderizar template de relatório HTML.

---

## Termos Específicos do Framework

### PipelineRunResult

**Definição:** Agregação de todos os `ProfilingResult` de uma execução de pipeline completa.

**Local:** `core/results.py`

---

### ProfilingResult

**Definição:** Resultado de uma única execução de profiling.

**Campos:** `profiler_name`, `model_label`, `pipeline_stage`, `timestamp`, `memory`, `latency`, `accuracy`, `extra`

---

### PipelineStep

**Definição:** Um passo individual no pipeline, com `label` e `port`.

**Propriedade derivada:** `kind` ("load", "compress", "profile", "heal")

---

### CompressionNotSupportedError

**Definição:** Exceção lançada quando uma técnica de compressão não pode ser aplicada (arquitetura incompatível ou biblioteca não instalada).

**Tratamento:** Causa abort do pipeline no `PipelineExecutor`.

---

### FIRST_N (Validation Mode)

**Definição:** Modo de validação rápida usando apenas as primeiras N amostras.

**Uso:** `limit_test_samples` em `ComprehensiveProfilingConfig` e `max_samples` em `DatasetConfig`.

---

## Hipóteses Registradas

### H1: Decoder-only Assumption

O framework assume que todos os modelos são causal decoder-only LLMs. Suporte para encoders ou modelos de sequência não foi verificado.

### H2: "text" Column Convention

Datasets para fine-tuning devem ter coluna "text" com texto em linguagem natural. Não há fallback para outros nomes de coluna.

### H3: First-N Sem Randomization

`max_samples=N` sempre usa as primeiras N linhas, sem aleatorização. O campo `seed` em `DatasetConfig` não é usado.

### H4: Layer 0 Preservation

`DepthPruningAdapter` assume que layer 0 (embedding) é critical c deve ser preservada mesmo se tiver baixa distância angular.

### H5: tokenizer.get_tokenizer()

Todos os adapters que precisam de tokenizer chamam `get_tokenizer()` — convenção estabelecida mas não verificada via interface.

### H6: Heuristic Layer Detection

`get_transformer_layers()` heuristics funcionam para Llama, Mistral, Qwen e GPT-2. Outros modelos podem não funcionar.

---

## Abreviaturas

| Abreviação | Significado |
|---|---|
| PTQ | Post-Training Quantization |
| QLoRA | Quantized Low-Rank Adaptation |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| NF4 | 4-bit NormalFloat |
| AWQ | Activation-Aware Weight Quantization |
| GPTQ | Generative Pre-Training Quantization |
| DAG | Directed Acyclic Graph |
| HF | HuggingFace |
| TPS | Tokens Per Second |
| PPL | Perplexity |
| KV Cache | Key-Value Cache |
| MC2 | TruthfulQA metric (multiple choice corrected) |
| Pass@1 | Pass at 1 (code generation) |
| F1 | Token-level F1 score (SQuAD) |
