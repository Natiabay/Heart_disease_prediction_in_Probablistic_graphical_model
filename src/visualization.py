"""Figures for reports, notebooks, and demo slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

from .evaluation import EvalResult, confusion_matrix_labels, roc_curve_data


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dag(
    model: DiscreteBayesianNetwork,
    output_path: Path,
    title: str,
    highlight_target: str = "HeartDisease",
) -> None:
    """Draw Bayesian Network DAG."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())

    pos = nx.spring_layout(G, seed=42, k=1.8)
    colors = ["#e74c3c" if n == highlight_target else "#3498db" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2200, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrows=True, arrowsize=18, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    _save(fig, output_path)


def plot_metrics_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of accuracy / F1 across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"{r['Model']}\n({r['Inference']})" for _, r in df.iterrows()]
    x = np.arange(len(labels))

    axes[0].bar(x, df["Accuracy"], color="#2ecc71", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Score")

    axes[1].bar(x, df["F1"], color="#9b59b6", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("F1 Score")

    fig.suptitle("Heart Disease BN — Evaluation Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def plot_confusion_matrix(result: EvalResult, output_path: Path) -> None:
    cm, labels = confusion_matrix_labels(result)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_title(f"Confusion Matrix\n{result.model_name} ({result.method})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, output_path)


def plot_roc_curve(result: EvalResult, output_path: Path) -> None:
    fpr, tpr = roc_curve_data(result)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {result.roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {result.model_name} ({result.method})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, output_path)


def plot_inference_benchmark(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['model']}\n{r['method']}" for _, r in df.iterrows()]
    x = np.arange(len(labels))
    ax.bar(x, df["mean_ms"], yerr=df["std_ms"], capsize=4, color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Milliseconds per query")
    ax.set_title("Inference Efficiency — VE vs Belief Propagation")
    _save(fig, output_path)


def plot_posterior_bar(posterior: dict[str, float], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(posterior.keys())
    vals = list(posterior.values())
    colors = ["#2ecc71" if l == "No" else "#e74c3c" for l in labels]
    ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    _save(fig, output_path)
