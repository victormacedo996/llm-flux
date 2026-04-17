"""
dag/renderer.py — Renders a pipeline DAG to a PNG file using matplotlib or ECharts-compatible data.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

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


def render_dag_echarts(dag: nx.DiGraph) -> dict:
    """
    Render the DAG as ECharts-compatible data structure.

    Args:
        dag: DiGraph produced by ``build_dag()``.

    Returns:
        Dict containing ECharts option for graph visualization.
    """
    # Compute positions using networkx
    try:
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(dag, seed=42)

    # Scale positions to fit ECharts canvas (assuming 800x600 canvas)
    if pos:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Scale to fit within 100-700 range for x, 100-500 for y
        scaled_pos = {}
        for node, (x, y) in pos.items():
            scaled_x = 100 + (x - x_min) / x_range * 600
            scaled_y = 100 + (y - y_min) / y_range * 400
            scaled_pos[node] = (scaled_x, scaled_y)
        pos = scaled_pos

    # Prepare nodes data
    nodes_data = []
    for node in dag.nodes:
        x, y = pos.get(node, (300, 300))  # fallback position
        kind = dag.nodes[node]["kind"]
        color = KIND_COLORS.get(kind, "#888888")
        nodes_data.append({
            "name": dag.nodes[node]["label"],
            "x": x,
            "y": y,
            "itemStyle": {"color": color},
            "symbolSize": 50,
        })

    # Prepare links data
    links_data = []
    for source, target in dag.edges:
        source_label = dag.nodes[source]["label"]
        target_label = dag.nodes[target]["label"]
        links_data.append({
            "source": source_label,
            "target": target_label,
            "lineStyle": {
                "width": 2,
                "curveness": 0.1
            }
        })

    # Build ECharts option
    option = {
        "title": {
            "text": f"Pipeline: {dag.graph.get('name', '')}",
            "left": "center",
            "textStyle": {"color": "#333"}
        },
        "tooltip": {},
        "animationDurationUpdate": 1500,
        "animationEasingUpdate": "quinticInOut",
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "symbolSize": 50,
                "roam": True,
                "label": {
                    "show": True,
                    "fontSize": 12,
                    "color": "#333"
                },
                "edgeSymbol": ["circle", "arrow"],
                "edgeSymbolSize": [4, 10],
                "edgeLabel": {
                    "fontSize": 12
                },
                "data": nodes_data,
                "links": links_data,
                "lineStyle": {
                    "opacity": 0.9,
                    "width": 2,
                    "curveness": 0
                }
            }
        ]
    }
    
    return option
