# LLM-Flux

A dissertation methodology framework for LLM compression research.

LLM-Flux abstracts and orchestrates compression, profiling, and model healing techniques over a user-defined DAG pipeline. It generates structured logs and tables ready for dissertation documentation.

## Quick Start (uv)

```bash
uv sync
uv run python examples/compression_study.py
```

## Local Validation (First N rows)

For lightweight CPU-only validation, use first-N benchmark slicing:

```bash
uv run python examples/e2e_option_c_validation.py
```

This runs Option C (Perplexity + QA first) with:
- tiny model (`hf-internal-testing/tiny-random-LlamaForCausalLM`)
- `limit_test_samples = FIRST_N`
- benchmark tests: `lambada`, `squad`, `arc_challenge`, `gsm8k`, `truthfulqa_mc2`

## Implemented benchmark datasets and metrics

- Perplexity: `lambada`, `wikitext_103_v1`, `c4`
- Accuracy: `mmlu_pro`, `math_500`, `arc_challenge`, `hellaswag`, `winogrande`, `piqa`, `sciq`, `big_bench_hard`
- Number extraction accuracy: `gsm8k`
- F1: `squad`
- MC2: `truthfulqa_mc2`
- Pass@1: `humaneval_pass1`, `mbpp_pass1`

## Project Structure

```
modelforge/
├── core/           # Pure-Python ports (interfaces) — no ML imports
├── profiling/      # Profiling domain services + types
├── adapters/       # Concrete implementations (GPTQ, AWQ, HF Trainer, …)
├── datasets/       # Dataset loading adapters
├── dag/            # DAG builder, renderer, executor
└── cli.py          # Optional typer CLI
```

## Optional Extras

```bash
uv sync --extra gptq          # GPTQ compression
uv sync --extra awq           # AWQ compression
uv sync --extra bitsandbytes  # BitsAndBytes quantization
uv sync --extra cli           # CLI (typer)
uv sync --extra dev           # Development tools
```

## Documentation (Spec-Driven Development)

- [docs/README.md](docs/README.md) — documentation hub
- [docs/product/product-context.md](docs/product/product-context.md) — product scope and goals
- [docs/product/spec-driven-workflow.md](docs/product/spec-driven-workflow.md) — SDD process
- [docs/product/decision-log.md](docs/product/decision-log.md) — decision choices and trade-offs
- [docs/technical/technical-context.md](docs/technical/technical-context.md) — LLM-oriented technical context
- [docs/technical/architecture.md](docs/technical/architecture.md) — architecture and Mermaid diagrams
- [docs/technical/extension-playbook.md](docs/technical/extension-playbook.md) — how to add new adapters
- [docs/specs/feature-spec-template.md](docs/specs/feature-spec-template.md) — feature spec template
- [docs/specs/adapter-spec-template.md](docs/specs/adapter-spec-template.md) — adapter spec template
