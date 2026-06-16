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
    threshold: float = 0.5
    y_true: list[str] = field(default_factory=list)
    y_pred: list[str] = field(default_factory=list)
    y_score: list[float] = field(default_factory=list)


def _predict_label(score: float, threshold: float) -> str:
    return "Yes" if score >= threshold else "No"


def collect_scores(
    model: DiscreteBayesianNetwork,
    df: pd.DataFrame,
    method: str = "ve",
    feature_vars: list[str] | None = None,
) -> tuple[list[str], list[float], float]:
    """Run inference on all rows; return labels, P(Yes) scores, mean latency."""
    y_true, y_score = [], []
    timings = []
    for _, row in df.iterrows():
        evidence = records_to_evidence(row, feature_vars=feature_vars)
        t0 = time.perf_counter()
        trace = predict_disease(model, evidence, method=method)
        timings.append((time.perf_counter() - t0) * 1000.0)
        y_true.append(str(row[TARGET]))
        y_score.append(disease_probability(trace))
    return y_true, y_score, float(np.mean(timings))


def find_optimal_threshold(y_true: list[str], y_score: list[float]) -> float:
    """Pick threshold on validation data that maximizes F1."""
    y_bin = [1 if y == "Yes" else 0 for y in y_true]
    if len(set(y_bin)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 71):
        preds = [1 if s >= t else 0 for s in y_score]
        f1 = f1_score(y_bin, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def find_optimal_threshold_balanced(y_true: list[str], y_score: list[float]) -> float:
    """Pick threshold that maximizes the minimum of accuracy, precision, recall, and F1."""
    y_bin = [1 if y == "Yes" else 0 for y in y_true]
    if len(set(y_bin)) < 2:
        return 0.5
    best_t, best_min = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = [1 if s >= t else 0 for s in y_score]
        vals = [
            accuracy_score(y_bin, preds),
            precision_score(y_bin, preds, zero_division=0),
            recall_score(y_bin, preds, zero_division=0),
            f1_score(y_bin, preds, zero_division=0),
        ]
        mn = min(vals)
        if mn > best_min:
            best_min, best_t = mn, float(t)
    return best_t


def _metrics_from_scores(
    y_true: list[str],
    y_score: list[float],
    threshold: float,
) -> tuple[float, float, float, float, float]:
    y_pred = [_predict_label(s, threshold) for s in y_score]
    y_bin = [1 if y == "Yes" else 0 for y in y_true]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    try:
        auc = roc_auc_score(y_bin, y_score)
    except ValueError:
        auc = float("nan")
    return float(acc), float(prec), float(rec), float(f1), float(auc)


def evaluate_model(
    model: DiscreteBayesianNetwork,
    test_df: pd.DataFrame,
    model_name: str,
    method: str = "ve",
    threshold: float | None = None,
    tune_threshold_df: pd.DataFrame | None = None,
    feature_vars: list[str] | None = None,
    balanced_threshold: bool = False,
) -> EvalResult:
    """
    Evaluate BN on held-out patients.

    If tune_threshold_df is provided, threshold is tuned on that set (e.g. train)
    then applied to test_df — avoids optimistic bias and improves recall.
    """
    if threshold is None and tune_threshold_df is not None:
        ty, ts, _ = collect_scores(
            model, tune_threshold_df, method=method, feature_vars=feature_vars
        )
        if balanced_threshold:
            threshold = find_optimal_threshold_balanced(ty, ts)
        else:
            threshold = find_optimal_threshold(ty, ts)
    elif threshold is None:
        threshold = 0.5

    y_true, y_score, mean_ms = collect_scores(
        model, test_df, method=method, feature_vars=feature_vars
    )
    acc, prec, rec, f1, auc = _metrics_from_scores(y_true, y_score, threshold)
    y_pred = [_predict_label(s, threshold) for s in y_score]

    return EvalResult(
        model_name=model_name,
        method=method.upper(),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        mean_inference_ms=mean_ms,
        n_test=len(test_df),
        threshold=threshold,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
    )


def benchmark_inference_methods(
    model: DiscreteBayesianNetwork,
    test_df: pd.DataFrame,
    model_name: str,
    n_repeats: int = 1,
    feature_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Compare VE vs BP mean latency on test evidence."""
    rows = []
    sample = test_df.head(min(15, len(test_df)))
    for method in ("ve", "bp"):
        times = []
        for _ in range(n_repeats):
            for _, row in sample.iterrows():
                evidence = records_to_evidence(row, feature_vars=feature_vars)
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
            "Threshold": round(r.threshold, 3),
            "Accuracy": round(r.accuracy, 3),
            "Precision": round(r.precision, 3),
            "Recall": round(r.recall, 3),
            "F1": round(r.f1, 3),
            "ROC-AUC": round(r.roc_auc, 3),
            "Mean inference (ms)": round(r.mean_inference_ms, 2),
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
