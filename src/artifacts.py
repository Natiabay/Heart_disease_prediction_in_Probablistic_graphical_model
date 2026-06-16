"""Save and load trained Bayesian Networks for Streamlit / deployment."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from pgmpy.models import DiscreteBayesianNetwork

from .config import ARTIFACTS_DIR


def save_model(model: DiscreteBayesianNetwork, name: str, metadata: dict | None = None) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "_")
    model_path = ARTIFACTS_DIR / f"{slug}.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    if metadata:
        meta_path = ARTIFACTS_DIR / f"{slug}_meta.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_model(name: str) -> DiscreteBayesianNetwork:
    slug = name.lower().replace(" ", "_")
    model_path = ARTIFACTS_DIR / f"{slug}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run `python run.py` first."
        )
    with model_path.open("rb") as f:
        return pickle.load(f)


def artifacts_exist() -> bool:
    return (ARTIFACTS_DIR / "expert_heart_disease_bn.pkl").exists()
