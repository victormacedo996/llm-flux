"""
dag/executor.py — Walks the DAG and runs each step via its registered Port.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import networkx as nx
from loguru import logger

from llm_flux.core.compression import CompressionNotSupportedError
from llm_flux.core.model import ModelHandle
from llm_flux.core.results import PipelineRunResult
from llm_flux.core.profiling import ProfilingResult


class PipelineExecutor:
    """
    Topologically walks the DAG and executes each step via its Port.

    The running model state is threaded through the walk:
    LOAD sets it, COMPRESS and HEAL replace it, PROFILE reads it.

    Args:
        dag: DiGraph produced by ``build_dag()``.
    """

    def __init__(self, dag: nx.DiGraph) -> None:
        self.dag = dag

    def run(self) -> PipelineRunResult:
        model: Optional[Any] = None
        model_handle: Optional[ModelHandle] = None
        profiling_records: list[ProfilingResult] = []
        started_at = datetime.now()
        step_count = len(self.dag.nodes)

        logger.info(f"🚀 Starting pipeline: {self.dag.graph.get('name', '')}")
        logger.info(f"   {step_count} step(s) to execute\n")
        for node_id in nx.topological_sort(self.dag):
            node = self.dag.nodes[node_id]
            step = node["step"]
            port = step.port
            idx = node["index"] + 1

            logger.info(f"[{idx}/{step_count}] ▶ {step.kind.upper()} — {step.label}")

            try:
                match step.kind:
                    case "load":
                        model = port.load()
                        model_handle = port
                        logger.info(f"  ✅ Model loaded: {port.name}")

                    case "compress":
                        model = port.compress(model)
                        logger.info(f"  ✅ Compression applied: {port.label}")

                    case "profile":
                        result_or_list = port.profile(
                            stage_label=step.label,
                            model_handle=model_handle,
                        )
                        if isinstance(result_or_list, list):
                            profiling_records.extend(result_or_list)
                            logger.info(
                                f"  ✅ {len(result_or_list)} profiling result(s) — {step.label}"
                            )
                            for r in result_or_list:
                                logger.info(f"  ✅ {r.summary()}")
                        else:
                            profiling_records.append(result_or_list)
                            logger.info(f"  ✅ {result_or_list.summary()}")

                    case "heal":
                        model = port.heal(model)
                        logger.info(f"  ✅ Healing complete: {port.label}")

            except CompressionNotSupportedError as e:
                logger.error(
                    f"  ❌ Compression step '{step.label}' failed: {e}\n"
                    "     Technique is incompatible with the current model. "
                    "Aborting pipeline."
                )
                raise

            except Exception as e:
                logger.error(f"  ❌ Step '{step.label}' raised an unexpected error: {e}")
                raise e

            logger.info("")  # blank line between steps

        finished_at = datetime.now()
        logger.info(f"🏁 Pipeline complete in {(finished_at - started_at).total_seconds():.1f} s")

        return PipelineRunResult(
            pipeline_name=self.dag.graph.get("name", "unnamed"),
            pipeline_description=self.dag.graph.get("description", ""),
            started_at=started_at,
            finished_at=finished_at,
            profiling_records=profiling_records,
        )
