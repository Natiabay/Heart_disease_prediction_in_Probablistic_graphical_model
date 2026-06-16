"""Bayesian Network representation — expert DAG and pgmpy model builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from .config import EXPERT_EDGES, EXPERT_VARS, NAIVE_BAYES_EDGES, TARGET


@dataclass
class BNRepresentation:
    """Container for a trained/static Bayesian Network."""

    name: str
    model: DiscreteBayesianNetwork
    edges: list[tuple[str, str]]
    description: str
    pillar: str = "Representation"


def build_expert_structure() -> dict:
    """Return expert DAG metadata (PGM Representation pillar)."""
    return {
        "name": "Expert Heart Disease BN",
        "variables": list(EXPERT_VARS),
        "edges": list(EXPERT_EDGES),
        "description": (
            "Expert-defined DAG: demographic and clinical risk factors "
            f"probabilistically influence {TARGET}. Age affects cholesterol, "
            "blood pressure, and exercise capacity; symptoms and test results "
            "directly inform the disease node."
        ),
    }


def naive_bayes_skeleton() -> DiscreteBayesianNetwork:
    """Naive Bayes DAG: every symptom/risk factor is a direct parent of disease."""
    model = DiscreteBayesianNetwork(NAIVE_BAYES_EDGES)
    model.add_nodes_from(EXPERT_VARS)
    return model


def expert_model_skeleton() -> DiscreteBayesianNetwork:
    """Instantiate expert DAG without parameters (structure only)."""
    model = DiscreteBayesianNetwork(EXPERT_EDGES)
    model.add_nodes_from(EXPERT_VARS)
    return model


def uniform_cpds(model: DiscreteBayesianNetwork) -> list[TabularCPD]:
    """Placeholder uniform CPTs for structure-only demos."""
    cpds = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        n_parents = len(parents)
        if n_parents == 0:
            card = 2
            cpd = TabularCPD(node, card, [[0.5], [0.5]])
        else:
            parent_cards = [len({0, 1}) for _ in parents]
            card = 2
            n_cols = int(pd.np.prod(parent_cards)) if hasattr(pd, "np") else 2 ** n_parents
            import numpy as np
            n_cols = int(np.prod([2] * n_parents))
            vals = np.full((card, n_cols), 0.5)
            cpd = TabularCPD(node, card, vals, evidence=parents, evidence_card=[2] * n_parents)
        cpds.append(cpd)
    return cpds


def learned_model_from_edges(edges: list[tuple[str, str]], nodes: list[str]) -> DiscreteBayesianNetwork:
    """Build pgmpy model from learned edge list."""
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(nodes)
    return model


def representation_summary(model: DiscreteBayesianNetwork) -> dict:
    """Graph statistics for logging and notebooks."""
    return {
        "n_nodes": len(model.nodes()),
        "n_edges": len(model.edges()),
        "nodes": list(model.nodes()),
        "edges": list(model.edges()),
        "is_dag": nx.is_directed_acyclic_graph(model),
    }


def wrap_trained_model(
    model: DiscreteBayesianNetwork,
    name: str,
    description: str,
) -> BNRepresentation:
    return BNRepresentation(
        name=name,
        model=model,
        edges=list(model.edges()),
        description=description,
    )
