"""
modelforge/runner.py — High-level pipeline entry point.

Combines build_dag + render_dag + confirmation prompt + PipelineExecutor
into a single `run_pipeline()` function.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from llm_flux.core.pipeline import Pipeline
from llm_flux.core.results import PipelineRunResult
from llm_flux.dag.builder import build_dag
from llm_flux.dag.executor import PipelineExecutor
from llm_flux.dag.renderer import render_dag


def run_pipeline(
    pipeline: Pipeline,
    dag_output: Path | str = "pipeline_dag.png",
    result_output: Path | str | None = None,
    skip_confirmation: bool = False,
    show_dag: bool = True,
) -> PipelineRunResult:
    """
    Build the DAG, render it, ask for confirmation, then execute.

    Args:
        pipeline:          Validated Pipeline instance (registry must be populated).
        dag_output:        Path where the DAG PNG is saved.
        result_output:     Path for the JSON result file.  Defaults to
                           ``"{pipeline.name}_result.json"``.
        skip_confirmation: If True, skip the [y/N] prompt (useful for scripts/CI).
        show_dag:          If True, attempt to display the DAG interactively.

    Returns:
        PipelineRunResult with all profiling records.
    """
    # ── 1. Build & render DAG ───────────────────────────────────────────────
    dag = build_dag(pipeline)
    logger.info(f"\n{'═' * 60}")
    logger.info(f"  Pipeline: {pipeline.name}")
    if pipeline.description:
        logger.info(f"  {pipeline.description}")
    logger.info(f"  Steps: {len(pipeline.steps)}")
    logger.info(f"{'═' * 60}\n")

    render_dag(dag, output_path=dag_output, show=show_dag)

    # ── 3. Print step summary ───────────────────────────────────────────────
    print("\nPlanned execution order:")
    for i, step in enumerate(pipeline.steps, 1):
        print(f"  {i:2}. [{step.kind.value.upper():8}] {step.label}")

    # ── 4. Confirmation ────────────────────────────────────────────────────
    if not skip_confirmation:
        print()
        answer = input("Execute this pipeline? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted by user.")
            raise SystemExit(0)

    # ── 5. Execute ─────────────────────────────────────────────────────────
    executor = PipelineExecutor(dag)
    try:
        result = executor.run()
    except Exception as e:
        logger.error(f"Pipeline execution aborted due to error: {e}")
        return None
        
    print("\n" + "═" * 60)
    print("  Results")
    print("═" * 60)
    print(result.to_markdown_table())
    print()
    print(result.to_dissertation_log())

    out_path = Path(result_output or f"{pipeline.name}_result.json")
    result.save_json(out_path)
    logger.info(f"\n  Full results saved → {out_path.resolve()}")

    return result
