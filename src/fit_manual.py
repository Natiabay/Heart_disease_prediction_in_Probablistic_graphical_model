"""Manual CPT fitting — avoids joblib hangs in some pgmpy versions."""

from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_state_counts


def fit_cpds_sequential(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    pseudo_count: float = 0.0,
) -> DiscreteBayesianNetwork:
    """
    Fit TabularCPDs sequentially (MLE with optional Laplace smoothing).

    Works reliably when pgmpy Parallel backend hangs.
    """
    state_names: dict[str, list] = {}
    for col in data.columns:
        if col in model.nodes():
            vals = {str(x) for x in data[col].tolist() if pd.notna(x) and str(x) not in ("nan", "None")}
            state_names[col] = sorted(vals)

    cpds = []
    for node in model.nodes():
        parents = sorted(model.get_parents(node))
        counts = get_state_counts(
            data=data.astype(str),
            state_names=state_names,
            variable=node,
            parents=parents,
        )
        arr = counts.values.astype(float)
        if pseudo_count > 0:
            arr += pseudo_count
        # zero columns -> uniform
        zero_cols = (arr.sum(axis=0) == 0)
        if zero_cols.any():
            card = len(state_names[node])
            arr[:, zero_cols] = 1.0
            arr[:, zero_cols] /= card

        parent_cards = [len(state_names[p]) for p in parents]
        cpd = TabularCPD(
            node,
            len(state_names[node]),
            arr,
            evidence=parents or None,
            evidence_card=parent_cards or None,
            state_names={v: state_names[v] for v in chain([node], parents)},
        )
        cpd.normalize()
        cpds.append(cpd)

    model.add_cpds(*cpds)
    if not model.check_model():
        raise ValueError("Invalid CPD configuration after sequential fit.")
    return model
