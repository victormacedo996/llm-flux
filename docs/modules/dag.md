# Módulo DAG — Builder, Executor e Renderer

## Visão Geral

O módulo `llm_flux/dag/` implementa a camada de **orquestração do pipeline**. Baseado em NetworkX, gerencia a construção do grafo, renderização visual e execução topológica.

---

## 1. `builder.py` — build_dag()

### 1.1 Responsabilidade

Converter um `Pipeline` (lista de `PipelineStep`) em um `nx.DiGraph` (Directed Acyclic Graph) do NetworkX.

### 1.2 Assinatura

```python
def build_dag(pipeline: Pipeline) -> nx.DiGraph
```

### 1.3 Lógica

```python
def build_dag(pipeline: Pipeline) -> nx.DiGraph:
    g = nx.DiGraph(name=pipeline.name, description=pipeline.description)
    prev_id = None

    for i, step in enumerate(pipeline.steps):
        node_id = f"{i}_{step.label.replace(' ', '_')}"
        g.add_node(
            node_id,
            step=step,
            kind=step.kind,
            label=step.label,
            index=i,
        )
        if prev_id is not None:
            g.add_edge(prev_id, node_id)
        prev_id = node_id

    return g
```

### 1.4 Estrutura do Grafo

**Nós:**
- `node_id`: `"{index}_{label}"` (e.g., `"0_Load_Model"`)
- Atributos: `step` (PipelineStep), `kind` (str), `label` (str), `index` (int)

**Arestas:**
- Lineares e sequenciais — cada nó conecta ao próximo
- `step[i]` → `step[i+1]`

**Metadata do grafo:**
- `g.graph["name"]` = `pipeline.name`
- `g.graph["description"]` = `pipeline.description`

### 1.5 Limitações

- O builder produz um DAG **estritamente linear** (sem branches)
- Extensão futura documentada: `depends_on: list[str]` em `PipelineStep` para fan-out
- O código fonte já menciona suporte a branches como "Future extension"

---

## 2. `executor.py` — PipelineExecutor

### 2.1 Responsabilidade

Executar o DAG via **topological sort** do NetworkX, threadando o estado do modelo através dos passos.

### 2.2 Construtor

```python
class PipelineExecutor:
    def __init__(self, dag: nx.DiGraph) -> None:
        self.dag = dag
```

### 2.3 `run()`

```python
def run(self) -> PipelineRunResult:
    model: Optional[Any] = None
    profiling_records: list[ProfilingResult] = []
    started_at = datetime.utcnow()
    step_count = len(self.dag.nodes)

    for node_id in nx.topological_sort(self.dag):
        node = self.dag.nodes[node_id]
        step = node["step"]
        port = step.port
        idx = node["index"] + 1

        match step.kind:
            case "load":
                model = port.load()
            case "compress":
                model = port.compress(model)
            case "profile":
                result = port.profile(model, stage_label=step.label)
                profiling_records.append(result)
            case "heal":
                model = port.heal(model)

    return PipelineRunResult(
        pipeline_name=...,
        profiling_records=profiling_records,
        ...
    )
```

### 2.4 Fluxo de Dados

```
LOAD          → model = port.load()
COMPRESS     → model = port.compress(model)
PROFILE       → result = port.profile(model, stage_label)
HEAL         → model = port.heal(model)
```

- `model` é `None` inicializado
- LOAD define `model`
- COMPRESS e HEAL **substituem** `model`
- PROFILE **lê** `model` mas não modifica

### 2.5 Tratamento de Erros

```python
except CompressionNotSupportedError as e:
    logger.error(f"Compression step '{step.label}' failed: {e}")
    raise  # aborta pipeline

except Exception as e:
    logger.error(f"Step '{step.label}' raised an unexpected error: {e}")
    raise  # relança
```

**Nota:** Erros em PROFILE e HEAL também são relançados — não há graceful degradation.

### 2.6 Retorno

`PipelineRunResult` com:
- `pipeline_name` e `pipeline_description` do grafo
- `started_at` e `finished_at`
- `profiling_records` de todos os nós de profiling

---

## 3. `renderer.py` — Visualização do DAG

### 3.1 `render_dag()`

**Responsabilidade:** Renderizar o DAG em PNG usando matplotlib + networkx.

```python
def render_dag(
    dag: nx.DiGraph,
    output_path: str | Path = "pipeline_dag.png",
    show: bool = True
) -> Path
```

**Parâmetros:**
- `output_path`: onde salvar o PNG
- `show`: se True, chama `plt.show()` (bloqueante em terminais não-interativos)

**Cores por tipo de passo:**
```python
KIND_COLORS = {
    "load":     "#4A90D9",  # blue
    "compress": "#E67E22",  # orange
    "profile":  "#27AE60",  # green
    "heal":     "#8E44AD",  # purple
}
```

**Layout:**
- Preferência: `nx.nx_agraph.graphviz_layout(dag, prog="dot")` (hierárquico left-to-right)
- Fallback: `nx.spring_layout(dag, seed=42)`

**Estilo:**
- Fundo escuro (`#1A1A2E`)
- Nodes: `node_size=4000`, `alpha=0.95`
- Font: white, bold
- Legenda: canto superior esquerdo

### 3.2 `render_dag_echarts()`

**Responsabilidade:** Produzir estrutura de dados para visualização interativa no HTML.

```python
def render_dag_echarts(dag: nx.DiGraph) -> dict
```

**Retorno:** Dict com opção ECharts (graph chart):
```python
{
    "title": {"text": "Pipeline: {name}", ...},
    "series": [{
        "type": "graph",
        "layout": "none",
        "data": [...],  # nodes com x, y, color, symbolSize
        "links": [...],  # edges com source, target, lineStyle
    }]
}
```

**Transformação de posições:**
- Calcula pos via graphviz/spring (igual a `render_dag()`)
- Escala para canvas 800x600 (100-700 para x, 100-500 para y)

**Dados dos nós:**
```python
{"name": label, "x": x, "y": y, "itemStyle": {"color": color}, "symbolSize": 50}
```

**Dados das arestas:**
```python
{"source": source_label, "target": target_label, "lineStyle": {"curveness": 0.1}}
```

---

## 4. Exports

```python
from llm_flux.dag import build_dag, render_dag, render_dag_echarts, PipelineExecutor
```

---

## 5. Fluxo Completo no DAG Layer

```
Pipeline (declarativo)
    │
    ├─→ build_dag(pipeline) → nx.DiGraph
    │         │
    │         │ linear sequential edges
    │         │ nós com step, kind, label, index
    │         │
    │         ▼
    │
    ├─→ render_dag(dag) → PNG
    │
    ├─→ render_dag_echarts(dag) → dict (para HTML)
    │
    └─→ PipelineExecutor(dag).run()
              │
              │ topological_sort
              │
              ▼
         Para cada nó:
           LOAD     → port.load() → model
           COMPRESS → port.compress(model) → model
           PROFILE  → port.profile(model) → ProfilingResult
           HEAL     → port.heal(model) → model
```

---

## 6. Contrato com Camadas Adjacentes

### 6.1 Interface com Core (`pipeline.py`)

- `PipelineStep.kind` é usado para dispatch no executor
- Validações (`_must_start_with_load`, `_labels_must_be_unique`) garantem que o DAG construído será válido

### 6.2 Interface com Adapters

- Cada port é chamado via `port.{method}(model)` onde `method` é determinado por `step.kind`
- O executor não conhece detalhes dos ports — só dispatch

### 6.3 Interface com Results

- `PipelineExecutor.run()` retorna `PipelineRunResult`
- `profiling_records` é populado por nós de tipo "profile"

---

## 7. Riscos e Limitações

1. **DAG linear apenas** — não suporta fan-out (um passo alimentando múltiplos). Extensão futura existe no código mas não implementada.

2. **Sem execução paralela** — topological sort garante ordem, mas não há parallelization mesmo para passos independentes (se houvesse branches).

3. **Erro em qualquer passo aborta pipeline** — não há checkpointing ou resume. Se um passo falhar, todo o progresso é perdido.

4. **`CompressionNotSupportedError` é relançada** — o executor captura mas relança, impedindo qualquer continuação.

5. **`render_dag()` usa matplotlib** — requer ambiente gráfico ou `matplotlib.use("Agg")` para evitar display. O código já tenta isso via `show=False`.

6. **`graphviz_layout` pode não estar disponível** — requer graphviz instalado. O fallback é `spring_layout`, que pode produzir layout diferente.

7. **Node IDs com espaços** — `step.label.replace(' ', '_')` remove espaços mas não outros caracteres especiais. Labels com caracteres especiais podem gerar IDs imprevisíveis.

8. **Nenhum suporte a cache de modelo** — se o mesmo modelo for usado em múltiplos branches (futuro), não haveria compartilhamento de estado.

9. **`torch.cuda.empty_cache()` em `HFModelHandle.unload()`** — é chamado apenas se o usuário chamar `unload()` explicitamente. O executor não chama `unload()` após cada passo.

10. **Ordem de profiling vs. compressão** — se múltiplos profiling steps existirem sem compressão entre eles, o modelo é o mesmo. Isso é válido para o padrão de avaliação, mas pode ser confuso.
