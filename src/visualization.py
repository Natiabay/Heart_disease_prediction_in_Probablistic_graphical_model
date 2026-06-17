"""Figures for reports, notebooks, and demo slides."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork

from .config import TARGET
from .dag_draw import save_directed_dag
from .evaluation import EvalResult, confusion_matrix_labels, roc_curve_data
from .inference import disease_probability, predict_disease


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
    """Draw Bayesian Network DAG with visible directed edges."""
    del highlight_target
    save_directed_dag(model, output_path, title, layered=True)


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


def plot_layered_dag(
    model: DiscreteBayesianNetwork,
    output_path: Path,
    title: str = "Manual Structure BN — Heart Disease",
) -> None:
    """Layered color-coded DAG with directed arrows."""
    save_directed_dag(model, output_path, title, layered=True)


def _p_yes_given_parent(model: DiscreteBayesianNetwork, parent: str) -> dict[str, float]:
    """Marginal P(Yes | parent state) by querying each parent value."""
    probs: dict[str, float] = {}
    parent_states: list[str] = []
    for cpd in model.get_cpds():
        if cpd.variable == parent:
            parent_states = [str(s) for s in cpd.state_names[parent]]
            break
    for state in parent_states:
        trace = predict_disease(model, {parent: state}, method="ve")
        probs[state] = disease_probability(trace)
    return probs


def plot_cpt_analysis(
    model: DiscreteBayesianNetwork,
    output_path: Path,
    nodes: list[str] | None = None,
) -> None:
    """Conditional probability bar charts P(HD | feature) for selected nodes."""
    nodes = nodes or ["Ca", "Thal", "CP", "Trestbps", "Exang"]
    present = [n for n in nodes if n in model.nodes()]
    n = len(present)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, var in enumerate(present):
        probs = _p_yes_given_parent(model, var)
        labels = list(probs.keys())
        vals = list(probs.values())
        colors = ["#e74c3c" if v >= 0.5 else "#f39c12" for v in vals]
        axes[i].bar(range(len(labels)), vals, color=colors, edgecolor="black")
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axes[i].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axes[i].set_ylim(0, 1.05)
        axes[i].set_title(f"P(HD | {var})", fontweight="bold")
        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    for j in range(len(present), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Conditional Probability Tables (Selected Nodes)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


INFERENCE_SCENARIOS = {
    "Prior P(HD)": {},
    "High Cholesterol": {"Chol": "High"},
    "Asym. CP + High BP": {"CP": "Asymptomatic", "Trestbps": "High"},
    "Exercise Angina": {"Exang": "Yes"},
    "Multi High-Risk": {
        "CP": "Typical angina",
        "Exang": "Yes",
        "Ca": "2",
        "Thal": "Reversible defect",
        "Chol": "High",
    },
    "Low-Risk Profile": {
        "Age": "Young",
        "Sex": "Female",
        "CP": "Asymptomatic",
        "Chol": "Low",
        "Exang": "No",
        "Thalach": "High",
    },
}


def plot_inference_scenarios(
    model: DiscreteBayesianNetwork,
    output_path: Path,
    threshold: float = 0.5,
    scenarios: dict[str, dict[str, str]] | None = None,
) -> None:
    """Posterior probabilities under different evidence profiles."""
    scenarios = scenarios or INFERENCE_SCENARIOS
    rows = []
    for name, evidence in scenarios.items():
        trace = predict_disease(model, evidence, method="ve")
        p_yes = disease_probability(trace)
        p_no = float(trace.posterior.get("No", 1.0 - p_yes))
        rows.append({"Evidence": name, "P(HD=0)": p_no, "P(HD=1)": p_yes})

    df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    y_pos = np.arange(len(df))
    colors = ["#e74c3c" if p >= threshold else "#f39c12" for p in df["P(HD=1)"]]
    ax1.barh(y_pos, df["P(HD=1)"], color=colors, edgecolor="black")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["Evidence"])
    ax1.axvline(threshold, color="#c0392b", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("P(Heart Disease = Yes)")
    ax1.set_title("Posterior Probabilities Under Different Evidence", fontweight="bold")
    ax1.legend()
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    table_df = df.copy()
    table_df["Risk"] = [
        "Baseline" if i == 0 else ("↑↑↑" if r["P(HD=1)"] >= 0.55 else "↑" if r["P(HD=1)"] > 0.49 else "↓↓↓")
        for i, r in table_df.iterrows()
    ]
    cell_text = [
        [f"{r['P(HD=0)']:.3f}", f"{r['P(HD=1)']:.3f}", r["Risk"]]
        for _, r in table_df.iterrows()
    ]
    table = ax2.table(
        cellText=cell_text,
        rowLabels=table_df["Evidence"].tolist(),
        colLabels=["P(HD=0)", "P(HD=1)", "Risk"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    ax2.set_title("Inference Summary", fontweight="bold", pad=20)

    fig.suptitle("Probabilistic Inference Results", fontsize=14, fontweight="bold")
    _save(fig, output_path)


def plot_mle_vs_bayesian_evaluation(
    mle_result: EvalResult,
    bayes_result: EvalResult,
    output_path: Path,
) -> None:
    """Side-by-side confusion matrices, ROC curves, and metric bars."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for col, result, cmap, label in [
        (0, mle_result, "Blues", "MLE"),
        (1, bayes_result, "Greens", "Bayesian Estimation"),
    ]:
        cm, labels = confusion_matrix_labels(result)
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(f"Confusion Matrix — {label}\nTN={tn} FP={fp} FN={fn} TP={tp}", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontweight="bold")

    ax_roc = fig.add_subplot(gs[0, 2])
    for result, color, label in [
        (mle_result, "#3498db", "MLE"),
        (bayes_result, "#27ae60", "Bayesian"),
    ]:
        fpr, tpr = roc_curve_data(result)
        ax_roc.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={result.roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    ax_met = fig.add_subplot(gs[1, :])
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    mle_vals = [mle_result.accuracy, mle_result.precision, mle_result.recall, mle_result.f1, mle_result.roc_auc]
    bayes_vals = [bayes_result.accuracy, bayes_result.precision, bayes_result.recall, bayes_result.f1, bayes_result.roc_auc]
    x = np.arange(len(metrics))
    w = 0.35
    ax_met.bar(x - w / 2, mle_vals, w, label="MLE", color="#3498db", edgecolor="black")
    ax_met.bar(x + w / 2, bayes_vals, w, label="Bayesian Estimation", color="#27ae60", edgecolor="black")
    ax_met.set_xticks(x)
    ax_met.set_xticklabels(metrics)
    ax_met.set_ylim(0, 1.08)
    ax_met.set_title("MLE vs Bayesian Estimation Metrics", fontweight="bold")
    ax_met.legend()
    for i, (mv, bv) in enumerate(zip(mle_vals, bayes_vals)):
        ax_met.text(i - w / 2, mv + 0.02, f"{mv:.2f}", ha="center", fontsize=8)
        ax_met.text(i + w / 2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)

    fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")
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
