# Decisões Técnicas — Log de Decisões Arquiteturais

## Visão Geral

Este documento registra as principais decisões técnicas do framework LLM-Flux, com justificativas, alternativas consideradas e consequências.

---

## 1. Arquitetura Hexagonal (Ports & Adapters)

### Decisão

Ports definidos em `core/` como classes abstratas puras em Python, sem imports de `torch`/`transformers`.

### Justificativa

- **Testabilidade** — portas podem ser testadas sem dependências de ML
- **Estabilidade de interface** — bibliotecas externas mudam frequentemente, ports são estáveis
- **Extensibilidade** — novos adapters não exigem modificação em core

### Alternativas Consideradas

1. **Ports como Protocolos (Python 3.8+)** — `typing.Protocol` em vez de ABC. Rejeitada: ABC é mais explícito e `isinstance` check funciona.
2. **Ports no nível de módulos** — sem classe abstrata, apenas funções. Rejeitada: sem contrato claro, validação harder.
3. **Dependency Injection via construtor** — passar ports no `__init__`. Já feito parcialmente (tokenizer, model_handle).

### Consequências

- Adapter que não respeita o contrato falha em runtime no `PipelineExecutor` — sem validação estática.
- Todos os ports que precisam de tokenizer/confirmação o recebem via construtor (não via pipeline state).

---

## 2. Pipeline Declarativo

### Decisão

Usar `Pipeline` e `PipelineStep` como modelos Pydantic (dados) em vez de funções/callables.

### Justificativa

- **Validação estática** — `_must_start_with_load`, `_labels_must_be_unique` validam antes da execução
- **Serialização** — `Pipeline` pode ser salvo em JSON para reprodutibilidade
- **DAG derivation** — a lista linear é naturalmente convertida em DAG

### Alternativas Consideradas

1. **Callables** — lista de funções lambda. Problema: não há como validar antes, impossível serializar.
2. **Configuration DSL** — YAML/TOML como config file. Problema: perdia a expressividade de Python para casos complexos.

### Consequências

- O campo `port: Any` em `PipelineStep` usa `arbitrary_types_allowed` — sem type checking em runtime.
- O `kind` é derivado via `isinstance()` — um port customizado que não é reconhecido levanta `ValueError` na definição do pipeline.

---

## 3. Thread de Estado via Variável Local no Executor

### Decisão

O `model` é passado entre passos como variável local no `PipelineExecutor.run()`, não como atributo do grafo ou storage global.

### Justificativa

- **Simplicidade** — não há estado global para gerenciar
- **Thread-safety** — cada execução de pipeline tem seu próprio estado
- **Testabilidade** — executor pode ser testado com mock ports

### Alternativas Consideradas

1. **Estado no nx.DiGraph** — nodes armazenam o modelo. Problema: grafo deve ser serializável para checkpointing, modelo não é.
2. **PipelineContext object** — objeto com estado mutável passado entre passos. Problema: mais complexo, mesmo benefício.
3. **Retorno em tuple** — `(model, result)` por passo. Problema: complica a lógica do executor.

### Consequências

- Se um passo falhar, o modelo carregado permanece em memória (sem cleanup automático).
- `HFModelHandle.unload()` nunca é chamado automaticamente pelo executor.

---

## 4. ComprehensiveProfilingAdapter como Facade

### Decisão

Um único `ProfilingPort` que orquestra `HardwareProfiler`, `LLMProfiler`, `InferenceBenchmarker` e `ModelPerformanceBenchmarker`.

### Justificativa

- **Simplicidade no DAG** — apenas um nó de profiling por estágio
- **Co-localização de dados** — todos os dados related estão no mesmo `ProfilingResult`
- **Feature flags** — cada sub-profiler pode ser habilitado/desabilitado individualmente

### Alternativas Consideradas

1. **Múltiplos nós de profiling** — um por tipo de profiler. Problema: DAG poluído, dados fragmentados em múltiplos `ProfilingResult`.
2. **Profiler registry** — registrador de profilers adicionados dinamicamente. Problema: mais complexo, sem benefício claro.

### Consequências

- Se `run_hardware_profile=False` em etapas subsequentes, dados de hardware não estão no JSON para comparação.
- Todos os 4 profilers são sempre instanciados mesmo se alguns não executam.

---

## 5. extra dict para Dados Brutos

### Decisão

`ProfilingResult.extra` armazena todos os raw dumps (hardware, llm_profile, inference, benchmarks) em um dict com chaves string.

### Justificativa

- **Zero data loss** — tudo é preservado no JSON export
- **Extensibilidade** — adapters podem adicionar dados customizados sem alterar o schema Pydantic
- **Clean top-level fields** — campos principais não são poluídos

### Alternativas Consideradas

1. **Campos flatten no Pydantic** — cada métrica como campo separado. Problema: dezenas de campos, schema inflation.
2. **Sub-models aninhados** — `ProfilingResult.hardware`, `.llm_profile`, etc. Problema: acoplamento, nem todos os profilers populam todos os sub-models.

### Consequências

- A extração de dados no `html_reporter.py` depende de chaves fixas (`"hardware"`, `"llm_profile"`). Se um adapter usar chave diferente, o HTML não渲染.
- O acesso a `extra["hardware"]` sem validação de presença pode causar `KeyError`.

---

## 6. BitsAndBytes Re-load Strategy

### Decisão

`BitsAndBytesAdapter` re-carrega o modelo do HuggingFace Hub com `quantization_config` em vez de aplicar quantização in-place.

### Justificativa

- **API constraint** — `bitsandbytes` só funciona com `BitsAndBytesConfig` no `from_pretrained`
- **Consistência** — quantização é aplicada de forma determinística

### Alternativas Consideradas

1. **Patch in-place** — tentar patchar o modelo já carregado. Problema: complicado, comportamento imprevisível.
2. **ModelHandle com quantization config** — passar quantization config para o ModelHandle. Problema: violação do princípio de responsabilidade única.

### Consequências

- `model_handle.source` precisa ser acessível — adapter recebe `model_handle` no construtor.
- Re-carregamento consume tempo e memória adicional.
- Modelo compressado anterior é descartado (garbage collected).

---

## 7. Depth Pruning — Angular Distance como Métrica

### Decisão

Usar distância angular (não magnitude) como métrica de importância de camada.

### Justificativa

- **Invariância a escala** — distância angular é normalize, não afetada por magnitudes absolutas
- **Literatura** — abordagem inspirada em métodos de análise de representação
- **Simplicidade** — computável via cosine similarity de input/output

### Alternativas Consideradas

1. **Taylor expansion** — derivadas de perda. Problema: requer forward pass com loss compute.
2. **Magnitude-based** — `L2 norm` dos pesos. Problema: sensível a escala, menos interpretável.
3. **Attention-based** — análise de padrões de atenção. Problema: não disponível em todos os modelos.

### Consequências

- Calibration dataset é usado para coletar ativações (tokenizado) — overhead.
- Layer 0 é sempre preservada — assumption de que primeira camada é critical.
- A heurística de `get_transformer_layers()` pode falhar para arquiteturas não-Llama/GPT-2.

---

## 8. Streaming Dataset Materialization

### Decisão

`HFTrainerAdapter` materializa streaming datasets em memória antes do treinamento.

### Justificativa

- **Evitar lazy I/O** — streaming datasets re-stream em cada epoch, causando repeated downloads
- **Checkpoint compatibility** — Trainer pode fazer checkpointing com dataset materializado

### Alternativas Consideradas

1. **Manter streaming** — trusting HF streaming. Problema: aparente hang quando dataset é iterado múltiplas vezes.
2. **Cache em disco** — `datasets.save_to_disk()`. Problema: mais complexo, disco é mais lento que memória para datasets pequenos.

### Consequências

- Se o streaming dataset for muito grande, pode consumir toda a memória.
- `Dataset.from_list(list(dataset))` itera o dataset inteiro — sem progress indication.

---

## 9. HTML Reporter com Jinja2 + ECharts

### Decisão

Gerar HTML autocontido via Jinja2 template + Apache ECharts (CDN).

### Justificativa

- **Autocontido** — um único arquivo HTML para compartilhar
- **Interativo** — ECharts permite zoom, hover tooltips
- **Sem servidor** — CDN funciona offline se cacheado

### Alternativas Consideradas

1. **Plotly Dash** — web app interativo. Problema: requer servidor Python em execução.
2. **Static HTML + Chart.js** — mais leve. Problema: Chart.js é less powerful que ECharts para grafos.
3. **Jupyter notebook** — mas não é "single file".

### Consequências

- O template Jinja2 (`html_reporter.html.jinja2`) está em `llm_flux/core/templates/report/` — caminho hardcoded.
- ECharts via CDN requer internet para carregar libraries na primeira vez.

---

## 10. LoRA como Padrão de Healing

### Decisão

`HealingConfig.lora` é `Optional[LoRAConfig]` — `None` significa full fine-tuning.

### Justificativa

- **QLoRA compatibility** — LoRA + BitsAndBytes é a técnica state-of-art para fine-tuning eficiente
- **Flexibilidade** — usuário escolhe entre full FT e LoRA via configuração
- **Padrão da literatura** — maioria dos experimentos de compressão usa LoRA recovery

### Alternativas Consideradas

1. **LoRA-only** — sempre aplica LoRA. Problema: se usuário quiser full FT, precisa de adapter diferente.
2. **Adapter de LoRA separado** — `LoRAHealingAdapter` vs `FullHealingAdapter`. Problema: mais adapters.

### Consequências

- O adapter `HFTrainerAdapter` detecta se LoRA está configurado e aplica `peft.get_peft_model()` conditionally.
- Se `lora=None`, todos os parâmetros são treináveis — memory footprint maior.

---

## 11. HuggingFace Trainer como Trainer Padrão

### Decisão

O framework usa `transformers.Trainer` como trainer padrão, com `trainer_class` configurável.

### Justificativa

- **Battle-tested** — HF Trainer é amplamente usado e bem otimizado
- **PEFT integration** — suporte nativo a LoRA/QLoRA
- **Configurable** — qualquer objeto duck-typed pode substituir

### Alternativas Consideradas

1. **PyTorch Lightning** — mais opinionado. Problema: opinionated demais para pesquisa.
2. **Raw training loop** — máximo controle. Problema: reinventing the wheel.
3. **DeepSpeed** — para multi-GPU. Problema: mais complexidade.

### Consequências

- `trainer_class` não é validado — qualquer callable com a assinatura correta é aceito.
- Sem suporte nativo a multi-GPU distributed training (DeepSpeed, FSDP).

---

## 12. ModelHandle.get_tokenizer() como Convenção

### Decisão

O tokenizer é exposto via `get_tokenizer()` method (não como propriedade) no `HFModelHandle`.

### Justificativa

- **Lazy loading** — tokenizer é carregado junto com o modelo, mas acessível separadamente
- **Compatibility** — vários adapters esperam este método

### Alternativas Consideradas

1. **Tokenize property** — `tokenizer` como property. Problema: se alguém substituir `load()`, o tokenizer pode não estar disponível.
2. **TokenizerPort separado** — interface própria. Problema: overkill — tokenizer é sempre carregado junto.

### Consequências

- Todos os adapters que precisam de tokenizer chamam `get_tokenizer()` — se não implementado, falha em runtime.
- Não há interface `TokenizerPort` — o framework assume `HFModelHandle`.

---

## 13. Calibração Dataset em Compression Adapters

### Decisão

GPTQ, AWQ e DepthPruning usam `calibration_dataset: DatasetConfig` com carregamento automático (local vs HF Hub).

### Justificativa

- **Consistência** — mesma interface para todos os compression adapters
- **Flexibilidade** — usuário pode usar dataset local ou HF Hub
- **Dissertation-ready** — configuração de dataset é serializável em JSON

### Alternativas Consideradas

1. **Dataset como path string** — simples. Problema: sem validação, sem HF Hub support.
2. **Dataset direto em memória** — `List[str]`. Problema: não serializável.

### Consequências

- Cada adapter implementa a mesma lógica de auto-detecção (copy-paste de fato).
- Se o dataset não tiver column "text", pode haver erro silencioso ou falha.

---

## 14. Validação de `steps[0] is load`

### Decisão

`Pipeline._must_start_with_load` valida que o primeiro passo é `load` kind.

### Justificativa

- **DAG semantics** — o primeiro nó deve produzir o modelo; não há modelo prévio para comprimir/heal/profile
- **Early error** — falha em definição de pipeline, não em execução

### Alternativas Consideradas

1. **Inferir LOAD se não especificado** — criar ModelHandle implícito. Problema: surpresa para usuário.
2. **Permitir qualquer ordem** — o executor tenta fazer load se model is None. Problema: mais difícil de debug.

### Consequências

- Se usuário acidentalmente colocar compress antes de load, recebe `ValueError` na construção do pipeline.

---

## 15. Grafico de Arquitetura com Ellipsis para Blocos Repetidos

### Decisão

O HTML reporter renderiza apenas Block 0 e adiciona nó de ellipsis (`... N more blocks`) para arquiteturas transformer com camadas repetidas.

### Justificativa

- **Legibilidade** — renderizar 32+ camadas idênticas polui o grafo
- **Informativo** — ainda comunica que há N blocos
- **Memória** — dados completos preservados no JSON, não no grafo visual

### Alternativas Consideradas

1. **Render all layers** — máximo detalhe. Problema: grafo poluído, ilegível.
2. **Aggregated block node** — um nó por bloco. Problema: perde granularidade intra-block.
3. **Truncation sem ellipsis** — confuso.

### Consequências

- `_block_index()` regex pode falhar para modelos com convenções de nomenclatura diferentes.
- O grafo ECharts é aproximação visual — dados reais estão no JSON.
