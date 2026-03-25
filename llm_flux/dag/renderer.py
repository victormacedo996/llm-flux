"""
dag/renderer.py — Renders a pipeline DAG to a PNG file using matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import os

KIND_COLORS: dict[str, str] = {
    "load":     "#4A90D9",  # blue
    "compress": "#E67E22",  # orange
    "profile":  "#27AE60",  # green
    "heal":     "#8E44AD",  # purple
}

KIND_LABELS: dict[str, str] = {
    "load":     "Load",
    "compress": "Compress",
    "profile":  "Profile",
    "heal":     "Heal",
}


def render_dag(
    dag: nx.DiGraph, output_path: str | Path = "pipeline_dag.png", show: bool = True
) -> None:
    """
    Render the DAG using matplotlib + networkx layout.

    Args:
        dag: DiGraph produced by ``build_dag()``.
        output_path: Where to save the PNG.
        show: If True, call ``plt.show()`` (blocks in non-interactive terminals).

    Returns:
        Absolute path to the saved PNG.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_path = Path(output_path)
    nodes = list(dag.nodes)

    colors = [KIND_COLORS.get(dag.nodes[n]["kind"], "#888888") for n in nodes]
    display_labels = {n: dag.nodes[n]["label"] for n in nodes}

    # Use a left-to-right hierarchical layout for linear pipelines
    try:
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(dag, seed=42)

    fig, ax = plt.subplots(figsize=(max(12, len(nodes) * 1.8), 5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    nx.draw_networkx_nodes(
        dag, pos, node_color=colors, node_size=4000, alpha=0.95, ax=ax
    )
    nx.draw_networkx_edges(
        dag, pos, edge_color="#AAAAAA", width=2, arrows=True,
        arrowsize=25, ax=ax,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_labels(
        dag, pos, labels=display_labels, font_size=8,
        font_color="white", font_weight="bold", ax=ax
    )

    legend_handles = [
        mpatches.Patch(color=c, label=KIND_LABELS.get(k, k))
        for k, c in KIND_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles, loc="upper left",
        facecolor="#2A2A4A", labelcolor="white", framealpha=0.9,
    )
    ax.set_title(
        f"Pipeline: {dag.graph.get('name', '')}",
        fontsize=14, color="white", pad=12,
    )
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)

    print(f"  DAG saved → {output_path.resolve()}")
    return output_path.resolve()
