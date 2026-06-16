"""Model evaluation — accuracy, precision, recall, F1, ROC, inference timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import TARGET
from .data import records_to_evidence
from .inference import disease_probability, map_label, predict_disease


@dataclass
class EvalResult:
    """Evaluation metrics for one model + inference method."""

    model_name: str
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    mean_inference_ms: float
    n_test: int
    y_true: list[str] = field(default_factory=list)
    y_pred: list[str] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)


def evaluate_model(
    model: DiscreteBayesianNetwork,
    test_df: pd.DataFrame,
    model_name: str,
    method: str = "ve",
) -> EvalResult:
    """
    Evaluate BN on held-out patients.

    For each test row, all features except HeartDisease are evidence;
    predict MAP label and P(Yes) for ROC-AUC.
    """
    y_true, y_pred, y_score = [], [], []
    timings = []

    for _, row in test_df.iterrows():
        evidence = records_to_evidence(row)
        t0 = time.perf_counter()
        trace = predict_disease(model, evidence, method=method)
        timings.append((time.perf_counter() - t0) * 1000.0)

        y_true.append(str(row[TARGET]))
        y_pred.append(map_label(trace))
        y_score.append(disease_probability(trace))

    # Binary metrics: positive class = "Yes"
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    try:
        auc = roc_auc_score([1 if y == "Yes" else 0 for y in y_true], y_score)
    except ValueError:
        auc = float("nan")

    return EvalResult(
        model_name=model_name,
        method=method.upper(),
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        roc_auc=float(auc),
        mean_inference_ms=float(np.mean(timings)),
        n_test=len(test_df),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )


def benchmark_inference_methods(
    model: DiscreteBayesianNetwork,
    test_df: pd.DataFrame,
    model_name: str,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Compare VE vs BP mean latency on test evidence."""
    rows = []
    sample = test_df.head(min(50, len(test_df)))
    for method in ("ve", "bp"):
        times = []
        for _ in range(n_repeats):
            for _, row in sample.iterrows():
                evidence = records_to_evidence(row)
                t0 = time.perf_counter()
                predict_disease(model, evidence, method=method)
                times.append((time.perf_counter() - t0) * 1000.0)
        rows.append({
            "model": model_name,
            "method": method.upper(),
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "n_queries": len(times),
        })
    return pd.DataFrame(rows)


def results_to_dataframe(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Model": r.model_name,
            "Inference": r.method,
            "Accuracy": r.accuracy,
            "Precision": r.precision,
            "Recall": r.recall,
            "F1": r.f1,
            "ROC-AUC": r.roc_auc,
            "Mean inference (ms)": r.mean_inference_ms,
            "N test": r.n_test,
        }
        for r in results
    ])


def confusion_matrix_labels(result: EvalResult) -> tuple[np.ndarray, list[str]]:
    labels = ["No", "Yes"]
    cm = confusion_matrix(result.y_true, result.y_pred, labels=labels)
    return cm, labels


def roc_curve_data(result: EvalResult) -> tuple[np.ndarray, np.ndarray]:
    y_bin = [1 if y == "Yes" else 0 for y in result.y_true]
    fpr, tpr, _ = roc_curve(y_bin, result.y_score)
    return fpr, tpr
