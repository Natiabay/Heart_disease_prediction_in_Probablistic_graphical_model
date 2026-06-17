"""Save and load trained Bayesian Networks for Streamlit / deployment."""

from __future__ import annotations

import json
from pathlib import Path

from pgmpy.models import DiscreteBayesianNetwork

from .config import ARTIFACTS_DIR


def save_model(model: DiscreteBayesianNetwork, name: str, metadata: dict | None = None) -> Path:
    """Save model structure + metadata as JSON (lightweight, Git-friendly)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_")
    payload = {
        "name": name,
        "nodes": list(model.nodes()),
        "edges": [list(e) for e in model.edges()],
        **(metadata or {}),
    }
    meta_path = ARTIFACTS_DIR / f"{slug}.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def load_model_spec(name: str) -> dict:
    slug = name.lower().replace(" ", "_")
    path = ARTIFACTS_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model spec not found: {path}. Run `python run.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_model_from_spec(spec: dict) -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork([tuple(e) for e in spec["edges"]])
    model.add_nodes_from(spec["nodes"])
    return model


def artifacts_exist() -> bool:
    return (ARTIFACTS_DIR / "manual_structure_bn.json").exists()
