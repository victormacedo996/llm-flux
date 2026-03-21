# LLM-Flux

A dissertation methodology framework for LLM compression research.

LLM-Flux abstracts and orchestrates compression, profiling, and model healing techniques over a user-defined DAG pipeline. It generates structured logs and tables ready for dissertation documentation.

## Quick Start (uv)

```bash
uv sync
uv run python examples/compression_study.py
```

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
