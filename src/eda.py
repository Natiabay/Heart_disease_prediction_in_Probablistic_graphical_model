"""Exploratory data analysis — UCI Heart Disease (Cleveland subset)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from .config import TARGET
from .data import load_raw_combined


def load_cleveland_raw() -> pd.DataFrame:
    """Cleveland-only raw UCI records for EDA."""
    raw = load_raw_combined()
    df = raw[raw["source"] == "cleveland"].copy()
    df["target"] = (df["num"] > 0).astype(int)
    return df.dropna(subset=["num"])


def plot_eda_dashboard(output_path: Path) -> None:
    """Multi-panel EDA dashboard (matches proposal / reference style)."""
    df = load_cleveland_raw()
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    no_hd = df[df["target"] == 0]
    hd = df[df["target"] == 1]

    def _hist(ax, col, title, xlabel):
        ax.hist(no_hd[col].dropna(), bins=15, alpha=0.65, color="#3498db", label="No Disease")
        ax.hist(hd[col].dropna(), bins=15, alpha=0.65, color="#e74c3c", label="Disease")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    _hist(fig.add_subplot(gs[0, 0]), "age", "Age Distribution", "Age (years)")
    _hist(fig.add_subplot(gs[0, 1]), "chol", "Cholesterol (mg/dl)", "Cholesterol")
    _hist(fig.add_subplot(gs[0, 2]), "trestbps", "Resting BP (mm Hg)", "Blood Pressure")
    _hist(fig.add_subplot(gs[0, 3]), "thalach", "Max Heart Rate", "Max HR")

    cp_labels = {1: "Typical", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"}
    cp_counts = (
        df.assign(cp_label=df["cp"].map(cp_labels))
        .groupby(["cp_label", "target"])
        .size()
        .unstack(fill_value=0)
    )
    ax = fig.add_subplot(gs[1, 0])
    cp_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Chest Pain Type", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(["No Disease", "Disease"], fontsize=7)
    ax.tick_params(axis="x", rotation=30)

    sex_counts = df.groupby(["sex", "target"]).size().unstack(fill_value=0)
    sex_counts.index = ["Female", "Male"]
    ax = fig.add_subplot(gs[1, 1])
    sex_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Sex", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(["No Disease", "Disease"], fontsize=7)
    ax.tick_params(axis="x", rotation=0)

    exang_counts = df.groupby(["exang", "target"]).size().unstack(fill_value=0)
    exang_counts.index = ["No", "Yes"]
    ax = fig.add_subplot(gs[1, 2])
    exang_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Exercise Angina", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(["No Disease", "Disease"], fontsize=7)

    fbs_counts = df.groupby(["fbs", "target"]).size().unstack(fill_value=0)
    fbs_counts.index = ["≤120", ">120"]
    ax = fig.add_subplot(gs[1, 3])
    fbs_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Fasting BS > 120", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(["No Disease", "Disease"], fontsize=7)

    corr_cols = ["age", "chol", "trestbps", "thalach", "oldpeak", "target"]
    corr = df[corr_cols].corr()
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    target_counts = df["target"].value_counts().sort_index()
    ax.bar(["No Disease (0)", "Disease (1)"], target_counts.values, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Target Variable Distribution", fontweight="bold")
    ax.set_ylabel("Count")

    ax = fig.add_subplot(gs[2, 2])
    data_bp = [no_hd["oldpeak"].dropna(), hd["oldpeak"].dropna()]
    bp = ax.boxplot(data_bp, tick_labels=["No Disease", "Disease"], patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#aed6f1")
    for med in bp["medians"]:
        med.set(color="black")
    ax.set_title("ST Depression by Outcome", fontweight="bold")
    ax.set_ylabel("Oldpeak")

    thal_labels = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    thal_counts = (
        df.assign(thal_label=df["thal"].map(thal_labels))
        .groupby(["thal_label", "target"])
        .size()
        .unstack(fill_value=0)
    )
    ax = fig.add_subplot(gs[2, 3])
    thal_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Thalassemia by Outcome", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(["No Disease", "Disease"], fontsize=7)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Exploratory Data Analysis — UCI Heart Disease (Cleveland)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
