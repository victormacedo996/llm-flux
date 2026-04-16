# AGENTS.md

## Dev Commands

```bash
uv sync                          # Install dependencies
uv run python examples/bnb_compression_study.py        # Full compression study
uv run python examples/e2e_option_c_validation.py       # CPU-friendly quick validation (first-N samples)
uv run pytest                                           # Run tests
uv run ruff check .                                     # Lint
uv run mypy llm_flux                                    # Typecheck (strict, py314)
```

## Architecture

- **Ports** (`llm_flux/core/`): Pure interfaces — no ML dependencies
- **Adapters** (`llm_flux/adapters/`): Concrete implementations (GPTQ, AWQ, BitsAndBytes, LoRA)
- **Entry point**: `llm_flux/runner.py::run_pipeline()`

Pipeline steps execute in list order via topological DAG walk. Model state is threaded through: LOAD sets it, COMPRESS/HEAL replace it, PROFILE reads it.

## Key Constraints

- Pipeline MUST start with a LOAD step (`_must_start_with_load` validator at `core/pipeline.py:64`)
- `mlp_pruning.py` is an empty placeholder — do not use
- `HFModelHandle.get_tokenizer()` only available after `load()` is called
- Benchmark tests auto-discovered via `hasattr(self, test_name)` — new methods on `ModelBenchmarker` are automatically eligible

## Caching

Examples set `HF_HOME` and `HF_DATASETS_CACHE` for model/dataset caching. Pass `cache_dir` to `ModelSource` for local caching.

## Quick Validation

For fast CPU validation, use `limit_test_samples=N` in `ComprehensiveProfilingConfig` and small models (e.g., `Qwen/Qwen3-0.6B`).

## Optional Extras

```bash
uv sync --extra gptq         # GPTQ quantization support
uv sync --extra awq          # AWQ quantization support
uv sync --extra bitsandbytes  # BitsAndBytes 4-bit support
uv sync --extra dev           # pytest, ruff, mypy
```

## Extension

See `docs/technical/extension-playbook.md` for adding new adapters. Spec templates live in `docs/specs/`.