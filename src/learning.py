"""Parameter and structure learning for the Heart Disease BN."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import BIC

from .config import EXPERT_VARS, TARGET, ProjectConfig
from .fit_manual import fit_cpds_sequential
from .model import build_expert_structure, expert_model_skeleton


@dataclass
class LearningResult:
    """Output of a learning procedure (PGM Learning pillar)."""

    name: str
    model: DiscreteBayesianNetwork
    method: str
    description: str
    score: float | None = None
    iterations: int | None = None


def learn_expert_bn(
    train_df: pd.DataFrame,
    use_bayesian: bool = True,
) -> LearningResult:
    """
    Learn CPTs on the fixed expert DAG (structure fixed, parameters learned).

    PGM Learning: MLE / Laplace-smoothed MLE on discretized UCI data.
    """
    meta = build_expert_structure()
    model = expert_model_skeleton()
    cols = [c for c in EXPERT_VARS if c in train_df.columns]
    data = train_df[cols].astype(str).copy()

    pseudo = 1.0 if use_bayesian else 0.0
    model = fit_cpds_sequential(model, data, pseudo_count=pseudo)
    method = (
        "Sequential MLE + Laplace smoothing (ESS≈1)"
        if use_bayesian
        else "Sequential Maximum Likelihood Estimation"
    )

    return LearningResult(
        name=meta["name"],
        model=model,
        method=method,
        description=(
            f"Parameters learned on the expert DAG using {method}. "
            "Structure is fixed by medical/clinical intuition."
        ),
    )


def _ensure_target_parents(model: DiscreteBayesianNetwork, cols: list[str]) -> None:
    if TARGET not in model.nodes():
        model.add_node(TARGET)
    if not model.get_parents(TARGET):
        for parent in ["CP", "Exang", "Ca", "Thal", "Oldpeak", "Chol"]:
            if parent in cols and parent != TARGET:
                if not model.has_edge(parent, TARGET):
                    model.add_edge(parent, TARGET)


def learn_tree_bn(train_df: pd.DataFrame) -> LearningResult:
    """
    Chow-Liu tree structure learning (PGM Learning pillar).

    Learns a maximum-weight spanning tree using mutual information.
    """
    cols = [c for c in EXPERT_VARS if c in train_df.columns]
    data = train_df[cols].astype(str).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = TreeSearch(data)
        tree = ts.estimate(
            estimator_type="chow-liu",
            class_node=TARGET,
            show_progress=False,
        )

    model = DiscreteBayesianNetwork(list(tree.edges()))
    model.add_nodes_from(cols)
    _ensure_target_parents(model, cols)
    model = fit_cpds_sequential(model, data, pseudo_count=0.5)
    bic = BIC(data).score(model)

    return LearningResult(
        name="Chow-Liu Tree BN",
        model=model,
        method="TreeSearch (Chow-Liu) + Sequential MLE",
        description=(
            "Tree-structured BN from Chow-Liu algorithm (mutual information MST), "
            "with CPTs fit via sequential MLE."
        ),
        score=float(bic),
    )


def learn_structure_and_parameters(
    train_df: pd.DataFrame,
    config: ProjectConfig,
    use_hillclimb: bool = False,
) -> LearningResult:
    """Learn data-driven tree BN (fast and reliable for deployment)."""
    del use_hillclimb, config  # Hill climb optional; tree used for reliability
    return learn_tree_bn(train_df)


def get_learning_summary(result: LearningResult) -> dict:
    return {
        "name": result.name,
        "method": result.method,
        "description": result.description,
        "score": result.score,
        "n_edges": len(result.model.edges()),
        "n_nodes": len(result.model.nodes()),
    }
