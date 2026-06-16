"""Probabilistic inference — Variable Elimination and Belief Propagation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from .config import TARGET


@dataclass
class InferenceTrace:
    """Explainable inference output for demos and notebooks."""

    method: str
    query_var: str
    evidence: dict[str, str]
    posterior: dict[str, float]
    elapsed_ms: float
    elimination_order: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _posterior_to_dict(factor, query_var: str) -> dict[str, float]:
    """Convert pgmpy factor to {state: probability}."""
    states = factor.state_names[query_var]
    values = factor.values.flatten()
    return {str(s): float(v) for s, v in zip(states, values)}


def _states_for_var(model: DiscreteBayesianNetwork, var: str) -> list[str]:
    for cpd in model.get_cpds():
        if cpd.variable == var:
            return [str(s) for s in cpd.state_names[var]]
    return []


def sanitize_evidence(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
) -> dict[str, str]:
    """Map evidence values to states the trained model actually supports."""
    clean: dict[str, str] = {}
    for var, val in evidence.items():
        if var not in model.nodes():
            continue
        states = _states_for_var(model, var)
        v = str(val)
        if v in states:
            clean[var] = v
        elif "Unknown" in states:
            clean[var] = "Unknown"
        elif states:
            clean[var] = states[0]
    return clean


_infer_cache: dict[int, tuple[VariableElimination, BeliefPropagation]] = {}


def _get_engines(model: DiscreteBayesianNetwork):
    key = id(model)
    if key not in _infer_cache:
        bp = BeliefPropagation(model)
        bp.calibrate()
        _infer_cache[key] = (VariableElimination(model), bp)
    return _infer_cache[key]


def infer_variable_elimination(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
    query_var: str = TARGET,
) -> InferenceTrace:
    """
    Variable Elimination inference (PGM Inference pillar).

    VE sums out non-query variables in an order that minimizes
    intermediate factor size (min-neighbors heuristic in pgmpy).
    """
    notes = [
        "Variable Elimination: multiply CPTs matching evidence, sum out hidden vars.",
        f"Query: P({query_var} | evidence)",
        f"Evidence variables: {list(evidence.keys())}",
    ]
    t0 = time.perf_counter()
    ve, _ = _get_engines(model)
    factor = ve.query(variables=[query_var], evidence=evidence, show_progress=False)
    elapsed = (time.perf_counter() - t0) * 1000.0

    # pgmpy does not expose elimination order directly; document heuristic used
    order = sorted(
        [n for n in model.nodes() if n not in evidence and n != query_var],
        key=lambda x: len(list(model.neighbors(x))),
    )
    notes.append(f"Approx. elimination order (min-neighbors): {order}")

    return InferenceTrace(
        method="Variable Elimination",
        query_var=query_var,
        evidence=evidence,
        posterior=_posterior_to_dict(factor, query_var),
        elapsed_ms=elapsed,
        elimination_order=order,
        notes=notes,
    )


def infer_belief_propagation(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
    query_var: str = TARGET,
) -> InferenceTrace:
    """
    Belief Propagation on the junction tree (PGM Inference pillar).

    Exact on trees; for loopy graphs pgmpy uses junction-tree calibration.
    """
    notes = [
        "Belief Propagation: messages passed on junction tree / clique tree.",
        f"Query: P({query_var} | evidence)",
        "Calibration propagates evidence through cliques until convergence.",
    ]
    t0 = time.perf_counter()
    _, bp = _get_engines(model)
    factor = bp.query(variables=[query_var], evidence=evidence, show_progress=False)
    elapsed = (time.perf_counter() - t0) * 1000.0

    return InferenceTrace(
        method="Belief Propagation",
        query_var=query_var,
        evidence=evidence,
        posterior=_posterior_to_dict(factor, query_var),
        elapsed_ms=elapsed,
        notes=notes,
    )


def predict_disease(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
    method: str = "ve",
) -> InferenceTrace:
    """Unified API: method in {'ve', 'bp'}."""
    evidence = sanitize_evidence(model, evidence)
    if method.lower() in ("ve", "variable_elimination"):
        return infer_variable_elimination(model, evidence)
    if method.lower() in ("bp", "belief_propagation"):
        return infer_belief_propagation(model, evidence)
    raise ValueError(f"Unknown inference method: {method}")


def compare_inference(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
) -> tuple[InferenceTrace, InferenceTrace]:
    """Run VE and BP on the same evidence for side-by-side demo."""
    return (
        predict_disease(model, evidence, method="ve"),
        predict_disease(model, evidence, method="bp"),
    )


def sensitivity_analysis(
    model: DiscreteBayesianNetwork,
    evidence: dict[str, str],
    method: str = "ve",
) -> list[dict]:
    """
    What-if: flip each evidence variable to an alternate state;
    measure delta in P(Heart Disease = Yes).
    """
    base = disease_probability(predict_disease(model, evidence, method=method))
    rows = []
    for var, current in evidence.items():
        if var == TARGET:
            continue
        states = _states_for_var(model, var)
        alts = [s for s in states if s != current]
        for alt in alts:
            ev = dict(evidence)
            ev[var] = alt
            p = disease_probability(predict_disease(model, ev, method=method))
            rows.append({
                "variable": var,
                "from": current,
                "to": alt,
                "p_yes": p,
                "delta": p - base,
            })
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows


def map_label(trace: InferenceTrace) -> str:
    """Maximum a posteriori class label."""
    return max(trace.posterior, key=trace.posterior.get)


def disease_probability(trace: InferenceTrace) -> float:
    """P(HeartDisease = Yes | evidence)."""
    return float(trace.posterior.get("Yes", 0.0))
