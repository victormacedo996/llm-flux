# Módulo Compression — Técnicas de Compressão e Adaptadores

## Visão Geral

O módulo `adapters/compression/` contém as **implementações concretas** das técnicas de compressão suportadas pelo framework. Cada adapter implementa o port `CompressionPort` definido em `core/compression.py`.

Todas as técnicas são **Post-Training Quantization (PTQ)** ou **pruning estrutural**, ou seja, não requerem re-treinamento do zero — apenas fine-tuning opcional posterior (healing).

---

## 1. GPTQ — `adapters/compression/gptq.py`

### 1.1 `GPTQConfig`

**Responsabilidade:** Configuração para GPTQ quantization.

**Campos:**
```python
name: str = "gptq-compression"
bits: int = Field(4, ge=2, le=8)           # largura de quantização (2-8 bits)
group_size: int = Field(128)               # tamanho do grupo de quantização
desc_act: bool = False                     # se True, ativa ordenação por activation
calibration_samples: int = 128             # número de amostras para calibração
calibration_dataset: DatasetConfig         # dataset de calibração
```

### 1.2 `GPTQAdapter`

**Responsabilidade:** Aplicar quantização GPTQ via `auto-gptq`.

**Construtor:**
```python
def __init__(self, config: GPTQConfig, tokenizer: Any) -> None:
    self.config = config
    self.tokenizer = tokenizer  # deve expor get_tokenizer()
```

**Fluxo de `compress(model)`:**
1. Importação tardia de `auto_gptq` (lazy). Fallback: `CompressionNotSupportedError`
2. Construção de `BaseQuantizeConfig(bits, group_size, desc_act)`
3. Carregamento do dataset de calibração (local ou HF Hub via `DatasetConfig.source`)
4. Tokenização das primeiras `calibration_samples` amostras
5. `AutoGPTQForCausalLM.from_pretrained(model.config._name_or_path, quantize_config)`
6. `gptq_model.quantize(examples)` — calibração in-place
7. Retorno do modelo quantizado

**Dependências:**
- `auto-gptq` (instalável via `uv sync --extra gptq`)
- `transformers` (já em deps core)

**Entrada:** `model` — um `PreTrainedModel` (nn.Module com `.config`)

**Saída:** `AutoGPTQForCausalLM` (objeto quantizado)

**Limitações e Riscos:**
- O adapter usa `model.config._name_or_path` para recarregar — assumes que o modelo foi originalmente carregado via HF Hub path.
- Compatível apenas com arquiteturas que `auto_gptq` suporta (Llama, Mistral, Qwen, GPT-2, etc.).
- Erro é capturado em blanket `except Exception` → `CompressionNotSupportedError` genérico.

---

## 2. AWQ — `adapters/compression/awq.py`

### 2.1 `AWQConfig`

**Responsabilidade:** Configuração para AWQ (Activation-Aware Weight Quantization).

**Campos:**
```python
name: str = "awq-compression"
bits: int = Field(4, ge=2, le=8)
group_size: int = 128
zero_point: bool = True                   # usa zero-point quantization
version: str = "GEMM"                    # versão do algoritmo ("GEMM" ou "GEMV")
calibration_samples: int = 128
calibration_dataset: DatasetConfig
```

### 2.2 `AWQAdapter`

**Responsabilidade:** Aplicar AWQ via `autoawq`.

**Construtor:**
```python
def __init__(self, config: AWQConfig) -> None:
    self.config = config
```

**Fluxo de `compress(model)`:**
1. Importação tardia de `awq` (lazy). Fallback: `CompressionNotSupportedError`
2. Construção de `quant_config` dict com `zero_point`, `q_group_size`, `w_bit`, `version`
3. `AutoAWQForCausalLM.from_pretrained(model.config._name_or_path)`
4. Carregamento do dataset de calibração
5. Extração de textos (`text` field) — diferente de GPTQ que tokeniza
6. `awq_model.quantize(quant_config=quant_config, calib_data=texts)`
7. Retorno do modelo quantizado

**Diferenças de GPTQ:**
- AWQ não requer tokenização do dataset de calibração — usa textos crus
- AWQ suporta apenas certain architectures (verificar `autoawq` docs)
- AWQ não tem suporte a `desc_act`

**Dependências:**
- `autoawq` (instalável via `uv sync --extra awq`)

**Entrada:** `model` — PreTrainedModel com `config._name_or_path`

**Saída:** `AutoAWQForCausalModel` (objeto quantizado)

---

## 3. BitsAndBytes — `adapters/compression/bitsandbytes.py`

### 3.1 `BitsAndBytesConfig`

**Responsabilidade:** Configuração para quantização BitsAndBytes.

**Campos:**
```python
load_in_4bit: bool = True     # NF4 quantization (QLoRA-style)
load_in_8bit: bool = False   # INT8 quantization
bnb_4bit_compute_dtype: str = "bfloat16"   # dtype para computação
bnb_4bit_quant_type: str = "nf4"           # "nf4" ou "fp4"
bnb_4bit_use_double_quant: bool = True    # double quantization
```

**Restrição:** `load_in_4bit` e `load_in_8bit` são mutuamente exclusivos na prática.

### 3.2 `BitsAndBytesAdapter`

**Responsabilidade:** Re-carregar o modelo com quantização BitsAndBytes aplicada.

**Construtor:**
```python
def __init__(self, config: BitsAndBytesConfig, model_handle: Any) -> None:
    self.config = config
    self.model_handle = model_handle  # precisa de .source para re-load
```

**Fluxo de `compress(model)`:**
1. Importação tardia de `bitsandbytes` + `transformers.BitsAndBytesConfig`
2. Construção de `HFBnBConfig` com todos os campos de configuração
3. **Re-carregamento** do modelo: `AutoModelForCausalLM.from_pretrained(source.identifier, quantization_config=bnb_cfg)`
4. Retorno do modelo quantizado

**Diferença crítica vs. outras técnicas:**
- BitsAndBytes **requer re-carregar** o modelo com `quantization_config` — não pode ser aplicado a um modelo já carregado in-place.
- O adapter armazena `model_handle` para ter acesso a `source` ( HF repo id ou path local).
- `device_map="auto"` é sempre usado.

**Dependências:**
- `bitsandbytes` (instalável via `uv sync --extra bitsandbytes`)

**Entrada:** `model` — qualquer PreTrainedModel (desconsiderado, apenas `model_handle.source` é usado)

**Saída:** Novo modelo quantizado (re-carregado)

**Riscos:**
- O modelo original (`model` passado como argumento) é **desconsiderado**. O modelo quantizado vem de um novo `from_pretrained`.
- Se `model_handle` não for um `HFModelHandle` (ou equivalente com atributo `.source`), a execução falha.
- Re-carregar consome tempo e memória adicional durante o pipeline.

---

## 4. Depth Pruning — `adapters/compression/depth_pruning.py`

### 4.1 `DepthPruningConfig`

**Responsabilidade:** Configuração para depth pruning estrutural.

**Campos:**
```python
name: str = "depth-pruning"
pruning_ratio: float = Field(0.1, ge=0.01, lt=1.0)  # % de camadas para remover
calibration_samples: int = 32
calibration_dataset: DatasetConfig
```

**Restrição:** `pruning_ratio` deve estar entre 0.01 (1%) e 0.99 (99%).

### 4.2 `DepthPruningAdapter`

**Responsabilidade:** Remover camadas de transformer redundantes com base na distância angular entre input/output.

**Construtor:**
```python
def __init__(self, config: DepthPruningConfig, tokenizer: Any,
             importance_fn: Optional[LayerImportanceFn] = None) -> None:
    self.config = config
    self.tokenizer = tokenizer
    self.importance_fn = importance_fn or angular_distance_importance
```

**Tipo de função de importância (customizável):**
```python
LayerImportanceFn = Callable[[Any, torch.Tensor], List[float]]
# Retorna lista de floats (um por camada), menor = menos importante
```

### 4.3 Algoritmo — Distância Angular

**Função `angular_distance_importance(model, input_ids)`:**
1. Registrar forward hooks em cada camada do transformer para capturar input/output
2. Forward pass com batches de `input_ids` (batch_size=4)
3. Para cada camada `i`:
   - `h_in = layer_input[i]` — shape (N, seq_len, hidden)
   - `h_out = layer_output[i]`
   - `cos_sim = cosine_similarity(h_in, h_out, dim=-1)`
   - `angular_distance = (1/π) * arccos(cos_sim)`
   - média sobre batch e seq_len
4. Retorna lista de distâncias angulares (uma por camada)

**Interpretação:** Distância angular próxima de 0 → camada transforma pouco (input ≈ output) → baixa importância → candidata a poda.

### 4.4 Fluxo de `compress(model)`

1. Verifica `model.config.num_hidden_layers`
2. `num_prune = max(1, int(num_layers * pruning_ratio))`
3. Coleta calibration inputs via `_get_calibration_inputs(device)`
4. Calcula importância: `distances = self.importance_fn(model, inputs)`
5. Identifica camadas para poda — menores distâncias (exclui layer 0):
   ```python
   sorted_indices = sorted(range(1, num_layers), key=lambda i: distances[i])
   layers_to_drop = sorted(sorted_indices[:num_prune])
   ```
6. `model = self._prune_layers(model, layers_to_drop)`
7. Atualiza `model.config.num_hidden_layers`

### 4.5 `_prune_layers(model, layers_to_drop)`

1. Encontra `nn.ModuleList` via `get_transformer_layers()` (heurística)
2. Cria novo `ModuleList` com camadas retidas (índice `keep_indices`)
3. **Re-indexa** `layer_idx` em atenção e sub-módulos para evitar KV Cache index errors
4. Substitui o ModuleList no modelo (atualiza `model.model.layers` ou `model.transformer.h`)
5. Coleta garbage + `torch.cuda.empty_cache()`

### 4.6 Heurística `get_transformer_layers()`

```python
def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # Llama / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h       # GPT-2 / Qwen
    # Fallback: percorre named_modules procurando ModuleList com "layers"/"h"/"blocks"
    # se não encontrar → CompressionNotSupportedError
```

**Limitações:**
- Assume arquitetura de transformer padrão (decoder-only)
- Modelos com estrutura não-padrão podem não ser suportados
- Heurística pode falhar silenciosamente se a estrutura do modelo for diferente

### 4.7 Customização da Função de Importância

O adapter aceita `importance_fn` customizado via construtor:

```python
def my_custom_importance(model, input_ids):
    # retorna lista de floats (uma importância por camada)
    return [1.0 / (i + 1) for i in range(num_layers)]

adapter = DepthPruningAdapter(config, tokenizer, importance_fn=my_custom_importance)
```

---

## Resumo Comparativo

| Técnica | Biblioteca | Requer re-load | Dataset | Tipo |
|---|---|---|---|---|
| GPTQ | `auto-gptq` | Não | Tokenizado | PTQ |
| AWQ | `autoawq` | Não | Textos crus | PTQ |
| BitsAndBytes | `bitsandbytes` | **Sim** | N/A | PTQ |
| Depth Pruning | custom (hooks) | Não | Tokenizado | Pruning estrutural |

---

## Dataset Loading nos Compression Adapters

Todos os adapters que usam `calibration_dataset` seguem o mesmo padrão:

```python
from pathlib import Path
cfg = self.config.calibration_dataset
if Path(cfg.source).exists():
    dataset = LocalDatasetAdapter(cfg).load()
else:
    dataset = HFDatasetAdapter(cfg).load()
```

- Se `source` é path existente → `LocalDatasetAdapter` (JSON/JSONL/CSV/Parquet)
- Caso contrário → `HFDatasetAdapter` (HF Hub dataset)

---

## Contratos de Interface

### `CompressionPort.compress(model)`

**Pré-condições:**
- `model` é um `nn.Module` ou `PreTrainedModel` carregado via `ModelHandle`
- Para `BitsAndBytesAdapter`: `self.model_handle.source` deve ser válido

**Pós-condições:**
- Retorna modelo compressado (pode ser o mesmo objeto mutado ou novo objeto)
- O modelo deve ser capaz de fazer forward pass normalmente

**Exceções levantadas:**
- `CompressionNotSupportedError` — técnica incompatível com arquitetura ou biblioteca não instalada
- `ValueError` — dataset não encontrado ou configuração inválida
- Para `DepthPruningAdapter`: `CompressionNotSupportedError` se não conseguir localizar transformer layers

---

## Riscos e Limitações

1. **GPTQ/AWQ usam `model.config._name_or_path`** — propriedade interna do transformers. Pode mudar em versões futuras.

2. **AWQ usa textos crus** vs. **GPTQ tokeniza** — comportamento diferente para mesma `calibration_dataset`.

3. **BitsAndBytes re-load é custoso** — ao contrário das outras técnicas que modificam in-place, um re-carregamento completo é necessário.

4. **`DepthPruningAdapter.get_transformer_layers()` é heurística** — pode falhar para modelos que não seguem as convenções Llama/GPT-2.

5. **`DepthPruningAdapter._prune_layers()` atualiza apenas duas estruturas** (Llama e GPT-2). Modelos como Mistral ou BLOOM podem não ser tratados.

6. **`mlp_pruning.py` está vazio** — não há adapter de MLP pruning implementado.

7. **Nenhum adapter para pruning por magnitude (weight pruning)** — apenas depth pruning estrutural está implementado.

8. **Sem suporte a distillation** — não há adapter que implemente knowledge distillation explicitamente.

9. **Compressão encadeada** — não há validação de compatibilidade entre técnicas encadeadas (ex: BitsAndBytes + DepthPruning).

---

## Exports do Módulo

```python
from llm_flux.adapters.compression import (
    BitsAndBytesAdapter, BitsAndBytesConfig,
    GPTQAdapter, GPTQConfig,
    AWQAdapter, AWQConfig,
    DepthPruningAdapter, DepthPruningConfig,
)
```
