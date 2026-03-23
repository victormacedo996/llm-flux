"""
dag/builder.py — Converts a Pipeline into a networkx DiGraph.
"""
from __future__ import annotations

import networkx as nx

from llm_flux.core.pipeline import Pipeline


def build_dag(pipeline: Pipeline) -> nx.DiGraph:
    """
    Convert a ``Pipeline`` into a linear Directed Acyclic Graph.

    Each step becomes a node (id = ``"{index}_{label}"``).
    Edges flow sequentially: step[0] → step[1] → … → step[n].

    Future extension: support fan-out branches by adding an optional
    ``depends_on: list[str]`` field to ``PipelineStep``.
    """
    g: nx.DiGraph = nx.DiGraph(name=pipeline.name, description=pipeline.description)
    prev_id: str | None = None

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
