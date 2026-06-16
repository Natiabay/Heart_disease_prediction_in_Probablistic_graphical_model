"""Build PGM_EndToEnd_Pipeline.ipynb from project modules."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "PGM_EndToEnd_Pipeline.ipynb"

C = []


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in s.strip().split("\n")]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": [line + "\n" for line in s.strip().split("\n")]}


C.append(code("pip install -r ../requirements.txt"))

C.append(md("""
# Medical Diagnosis Using Bayesian Networks — Heart Disease Prediction

## PGM project overview

This notebook walks through all **three pillars** of Probabilistic Graphical Models:

| Pillar | Implementation |
|--------|----------------|
| **Representation** | Expert + learned DAGs over symptoms and HeartDisease |
| **Learning** | MLE/Bayesian CPTs; Hill Climb structure learning |
| **Inference** | Variable Elimination & Belief Propagation |

**Query answered:** P(Heart Disease | observed symptoms and risk factors)
"""))

C.append(code("""
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").is_dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from src.config import EXPERT_EDGES, TARGET, STATE_LABELS
from src.data import load_discretized_dataset, train_test_split_data, dataset_summary, records_to_evidence
from src.model import build_expert_structure, representation_summary
from src.learning import learn_expert_bn, learn_structure_and_parameters, get_learning_summary
from src.inference import predict_disease, disease_probability
from src.evaluation import evaluate_model, results_to_dataframe, benchmark_inference_methods
from src.config import ProjectConfig
from src.visualization import plot_dag

OUT = ROOT / "outputs" / "notebook"
OUT.mkdir(parents=True, exist_ok=True)
print("Root:", ROOT)
"""))

C.append(md("## PART 1 — Data & preprocessing (UCI multi-source)"))

C.append(code("""
df = load_discretized_dataset()
summary = dataset_summary(df)
print("Samples:", summary["n_samples"])
print("By source:", summary["by_source"])
print("Prevalence:", f"{summary['prevalence']:.1%}")
df.head()
"""))

C.append(md("## PART 2 — Representation (expert Bayesian Network)"))

C.append(code("""
meta = build_expert_structure()
print(meta["description"])
print("Edges:", meta["edges"])
"""))

C.append(md("## PART 3 — Learning (CPTs + structure)"))

C.append(code("""
train_df, test_df = train_test_split_data(df)
config = ProjectConfig()
expert_lr = learn_expert_bn(train_df)
learned_lr = learn_structure_and_parameters(train_df, config)
print(get_learning_summary(expert_lr))
print(get_learning_summary(learned_lr))
plot_dag(expert_lr.model, OUT / "expert_dag.png", "Expert BN")
plot_dag(learned_lr.model, OUT / "learned_dag.png", "Learned BN")
"""))

C.append(md("## PART 4 — Inference (VE vs BP)"))

C.append(code("""
row = test_df.iloc[0]
evidence = records_to_evidence(row)
for method in ("ve", "bp"):
    trace = predict_disease(expert_lr.model, evidence, method=method)
    print(method.upper(), "P(Yes)=", f"{disease_probability(trace):.3f}", f"({trace.elapsed_ms:.1f} ms)")
    print("  Notes:", trace.notes[0])
"""))

C.append(md("## PART 5 — Evaluation"))

C.append(code("""
results = []
for lr, name in [(expert_lr, "Expert"), (learned_lr, "Learned")]:
    for method in ("ve", "bp"):
        results.append(evaluate_model(lr.model, test_df, name, method=method))
metrics = results_to_dataframe(results)
metrics
"""))

C.append(md("## PART 6 — Summary"))

C.append(md("""
| Part | PGM pillar | Deliverable |
|------|------------|-------------|
| 1 | Data | UCI heart disease merged & discretized |
| 2 | Representation | Expert DAG |
| 3 | Learning | CPTs + Hill Climb structure |
| 4 | Inference | VE & BP queries |
| 5 | Evaluation | Accuracy, precision, recall, F1, ROC-AUC |

**Live demo:** `streamlit run app/streamlit_app.py`
"""))


def main():
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": C,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(C)} cells -> {OUT}")


if __name__ == "__main__":
    main()
