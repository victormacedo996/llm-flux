# Technical Context (LLM-Oriented)

This document is optimized for code-generation assistants and contributors that need fast, accurate repository understanding.

## Repository identity
- Package/module root: `llm_flux/`
- Language: Python 3.14
- Domain: LLM compression experimentation
- Core style: ports + adapters with declarative pipeline orchestration

## Key modules
- `llm_flux/core/`: stable contracts and canonical models
- `llm_flux/adapters/`: concrete implementations for model/compression/healing
- `llm_flux/datasets/`: dataset ports/adapters (HF and local)
- `llm_flux/profiling/`: profiling services and profiling adapter
- `llm_flux/dag/`: pipeline-to-DAG conversion + execution
- `llm_flux/runner.py`: high-level `run_pipeline()` entrypoint
- `examples/`: composition patterns and experiment scripts

## Architectural invariants
1. New techniques SHOULD be introduced as adapters implementing existing ports.
2. `core/` should avoid concrete ML dependency coupling as much as possible.
3. Pipeline step semantics are inferred by port type (`load`, `compress`, `profile`, `heal`).
4. Execution state uses a threaded `model` object through topological DAG order.
5. Profiling outputs MUST map into `ProfilingResult`.

## Canonical contracts
- `ModelHandle` (`load`, `unload`, `name`)
- `CompressionPort` (`compress`)
- `HealingPort` (`heal`)
- `ProfilingPort` (`profile`)
- `DatasetPort` (`load`)

## Primary concrete adapters
- Model: `HFModelHandle`
- Compression: `DepthPruningAdapter`, `GPTQAdapter`, `AWQAdapter`, `BitsAndBytesAdapter`
- Healing: `HFTrainerAdapter`
- Profiling: `ComprehensiveProfilingAdapter`
- Datasets: `HFDatasetAdapter`, `LocalDatasetAdapter`

## Data and control flow
1. User declares `Pipeline(steps=[PipelineStep(...)])`.
2. `build_dag()` converts it to linear DiGraph.
3. `PipelineExecutor.run()` processes each step in topological order.
4. Profile steps append `ProfilingResult` records.
5. `PipelineRunResult` aggregates records and exports markdown/json/html.

## Extension guidance for LLMs
When adding new functionality, default strategy should be:
1. Reuse an existing `core` port if semantics match.
2. Create adapter in matching domain folder.
3. Add/update config model with clear defaults.
4. Update an example under `examples/`.
5. Add/update spec docs in `docs/specs/` and decision entry if architecture is impacted.

## Risk hotspots
- Optional dependency mismatch (AWQ/GPTQ/BNB extras).
- Tokenizer/model coupling assumptions in healing/profiling.
- Hardware-sensitive benchmark variability.
- Large-model memory pressure and device_map behavior.
