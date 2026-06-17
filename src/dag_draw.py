"""Directed DAG rendering with visible parent → child arrows."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from pgmpy.models import DiscreteBayesianNetwork

from .config import (
    LEGACY_INPUT_GROUPS,
    OPTIMIZED_INPUT_GROUPS,
    TARGET,
)


def _layered_positions(model: DiscreteBayesianNetwork) -> dict[str, tuple[float, float]]:
    """Top-down layout: parents above children (directed flow downward)."""
    if set(OPTIMIZED_INPUT_GROUPS["Demographics"]) <= set(model.nodes()):
        layers = {
            3: OPTIMIZED_INPUT_GROUPS["Demographics"],
            2: OPTIMIZED_INPUT_GROUPS["Symptoms & exercise"],
            1: OPTIMIZED_INPUT_GROUPS["Clinical test results"],
            0: [TARGET],
        }
    else:
        layers = {
            3: LEGACY_INPUT_GROUPS["Demographics"],
            2: LEGACY_INPUT_GROUPS["Symptoms"] + LEGACY_INPUT_GROUPS["Vitals & labs"],
            1: LEGACY_INPUT_GROUPS["Clinical findings"],
            0: [TARGET],
        }

    pos: dict[str, tuple[float, float]] = {}
    for y, vars_in_layer in layers.items():
        present = [n for n in vars_in_layer if n in model.nodes()]
        for i, n in enumerate(present):
            x = (i + 1) / (len(present) + 1)
            pos[n] = (x, float(y))
    for n in model.nodes():
        if n not in pos:
            pos[n] = (0.5, 1.5)
    return pos


def draw_directed_dag(
    model: DiscreteBayesianNetwork,
    title: str,
    *,
    layered: bool = True,
    figsize: tuple[float, float] = (11, 8),
    dpi: int = 130,
) -> bytes:
    """Render DAG with bold directed edges (parent → child)."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    pos = _layered_positions(model) if layered else nx.spring_layout(G, seed=42, k=1.8)

    fig, ax = plt.subplots(figsize=figsize)
    node_colors = ["#c0392b" if n == TARGET else "#2980b9" for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=2200, ax=ax, alpha=0.95, edgecolors="white"
    )
    nx.draw_networkx_labels(
        G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax
    )

    for parent, child in G.edges():
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.0,
            color="#e67e22",
            shrinkA=28,
            shrinkB=28,
            connectionstyle="arc3,rad=0.08",
            zorder=1,
        )
        ax.add_patch(arrow)

    ax.set_title(f"{title}\n(arrows: parent → child)", fontsize=13, fontweight="bold")
    ax.text(
        0.02, 0.02, "Directed acyclic graph (DAG)",
        transform=ax.transAxes, fontsize=9, color="#555",
    )
    ax.axis("off")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.35, 3.6 if layered else 1.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def save_directed_dag(model: DiscreteBayesianNetwork, path: Path, title: str, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(draw_directed_dag(model, title, **kwargs))
