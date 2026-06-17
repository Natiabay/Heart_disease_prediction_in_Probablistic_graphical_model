"""Bayesian Network representation — manual structure DAG and pgmpy model builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork

from .config import MANUAL_STRUCTURE_EDGES, MANUAL_STRUCTURE_VARS, NAIVE_BAYES_EDGES, TARGET


@dataclass
class BNRepresentation:
    """Container for a trained/static Bayesian Network."""

    name: str
    model: DiscreteBayesianNetwork
    edges: list[tuple[str, str]]
    description: str
    pillar: str = "Representation"


def build_manual_structure() -> dict:
    """Return hand-built DAG metadata (PGM Representation pillar)."""
    return {
        "name": "Manual Structure BN",
        "variables": list(MANUAL_STRUCTURE_VARS),
        "edges": list(MANUAL_STRUCTURE_EDGES),
        "description": (
            "Manually constructed DAG for heart-disease diagnosis: demographics and "
            f"clinical variables are linked by directed edges to {TARGET}. Structure is "
            "fixed by domain knowledge; CPTs are learned from data (Learning pillar)."
        ),
    }


def naive_bayes_skeleton() -> DiscreteBayesianNetwork:
    """Naive Bayes DAG: every symptom/risk factor is a direct parent of disease."""
    model = DiscreteBayesianNetwork(NAIVE_BAYES_EDGES)
    model.add_nodes_from(MANUAL_STRUCTURE_VARS)
    return model


def manual_structure_skeleton() -> DiscreteBayesianNetwork:
    """Instantiate hand-built DAG without parameters (structure only)."""
    model = DiscreteBayesianNetwork(MANUAL_STRUCTURE_EDGES)
    model.add_nodes_from(MANUAL_STRUCTURE_VARS)
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
            import numpy as np
            n_cols = int(np.prod([2] * n_parents))
            vals = np.full((2, n_cols), 0.5)
            cpd = TabularCPD(node, 2, vals, evidence=parents, evidence_card=[2] * n_parents)
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
