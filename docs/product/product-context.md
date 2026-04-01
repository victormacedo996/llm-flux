# Product Context

## Purpose
LLM-Flux is a research framework to run repeatable **LLM compression pipelines** and collect structured results for analysis.

The product enables users to compose workflows such as:
- load model
- profile baseline
- compress (e.g., pruning, quantization)
- profile intermediate state
- heal (fine-tune)
- profile final state

## Target users
- ML researchers running compression studies.
- Engineers evaluating compression trade-offs before deployment.
- Students generating reproducible experiment artifacts.

## Core value proposition
- **Composable pipelines** with reusable steps.
- **Port/adapters architecture** for pluggable implementations.
- **Structured outputs** (`results.json`, markdown summary, HTML report).
- **Research-friendly experimentation** (quick tiny mode + larger full mode).

## Product goals
1. Make new compression/healing/profiling methods easy to integrate.
2. Keep pipeline authoring declarative and simple.
3. Preserve comparability of results across experiments.
4. Support reproducibility by capturing full run outputs.

## Non-goals (current)
- Distributed orchestration across clusters.
- Multi-tenant serving/runtime deployment.
- Full experiment tracking platform replacement.

## Primary user journeys
1. **Author experiment** in `examples/` using `Pipeline` and `PipelineStep`.
2. **Run pipeline** with `run_pipeline()`.
3. **Inspect results** in markdown/json/html outputs.
4. **Iterate** by changing adapters/configs and re-running.

## Product constraints
- Python 3.14+.
- `transformers`, `torch`, and `datasets` ecosystem alignment.
- Optional dependencies required for specific techniques (AWQ/GPTQ/BNB).
- Hardware variability influences benchmark outputs.

## Product-level acceptance criteria for new features
A feature is accepted only when:
1. It has an approved spec (from templates in `docs/specs/`).
2. It does not break existing examples.
3. It preserves current pipeline semantics (load/compress/profile/heal).
4. It documents usage and expected outcomes.
