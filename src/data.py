"""Load, merge, discretize, and split UCI Heart Disease datasets."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR,
    EXPERT_VARS,
    RAW_DIR,
    TARGET,
    UCI_COLUMNS,
    UCI_SOURCES,
)


def _download_uci_sources() -> None:
    """Download all four UCI heart-disease files if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in UCI_SOURCES.items():
        dest = RAW_DIR / f"{name}.data"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"  Downloading UCI heart disease ({name}) ...")
        urllib.request.urlretrieve(url, dest)


def _read_uci_file(path: Path, source: str) -> pd.DataFrame:
    """Read one UCI .data file; missing values marked as '?'."""
    df = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?")
    df["source"] = source
    return df


def load_raw_combined() -> pd.DataFrame:
    """Load and vertically stack Cleveland, Hungarian, Switzerland, and VA data."""
    _download_uci_sources()
    frames = [_read_uci_file(RAW_DIR / f"{name}.data", name) for name in UCI_SOURCES]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["num"])
    return combined


def _bin_age(age: float) -> str:
    if age < 45:
        return "Young"
    if age < 55:
        return "Middle"
    if age < 65:
        return "Senior"
    return "Elderly"


def _bin_trestbps(v: float) -> str:
    if v < 120:
        return "Low"
    if v <= 140:
        return "Normal"
    return "High"


def _bin_chol(v: float) -> str:
    if v < 200:
        return "Low"
    if v <= 240:
        return "Borderline"
    return "High"


def _bin_thalach(series: pd.Series, value: float) -> str:
  q1, q2 = series.quantile([0.33, 0.66])
  if value <= q1:
      return "Low"
  if value <= q2:
      return "Medium"
  return "High"


def _bin_oldpeak(v: float) -> str:
    if v <= 0.0:
        return "None"
    if v <= 2.0:
        return "Mild"
    return "Severe"


def discretize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw UCI attributes to categorical BN node states."""
    out = pd.DataFrame(index=df.index)
    out["Age"] = df["age"].apply(_bin_age)
    out["Sex"] = df["sex"].map({0.0: "Female", 1.0: "Male"})
    cp_map = {
        1.0: "Typical angina",
        2.0: "Atypical angina",
        3.0: "Non-anginal pain",
        4.0: "Asymptomatic",
    }
    out["CP"] = df["cp"].map(cp_map).fillna("Unknown")
    out["Trestbps"] = df["trestbps"].apply(
        lambda v: _bin_trestbps(v) if pd.notna(v) else "Unknown"
    )
    out["Chol"] = df["chol"].apply(
        lambda v: _bin_chol(v) if pd.notna(v) else "Unknown"
    )
    out["Fbs"] = df["fbs"].map({0.0: "Normal", 1.0: "High"}).fillna("Unknown")
    restecg_map = {
        0.0: "Normal",
        1.0: "ST-T abnormality",
        2.0: "LV hypertrophy",
    }
    out["Restecg"] = df["restecg"].map(restecg_map).fillna("Unknown")
    thalach_ref = df["thalach"].dropna()
    out["Thalach"] = df["thalach"].apply(
        lambda v: _bin_thalach(thalach_ref, v) if pd.notna(v) else "Unknown"
    )
    out["Exang"] = df["exang"].map({0.0: "No", 1.0: "Yes"}).fillna("Unknown")
    out["Oldpeak"] = df["oldpeak"].apply(
        lambda v: _bin_oldpeak(v) if pd.notna(v) else "Unknown"
    )
    slope_map = {1.0: "Upsloping", 2.0: "Flat", 3.0: "Downsloping"}
    out["Slope"] = df["slope"].map(slope_map).fillna("Unknown")
    ca_map = {0.0: "0", 1.0: "1", 2.0: "2", 3.0: "3+"}
    out["Ca"] = df["ca"].map(ca_map).fillna("Unknown")
    thal_map = {3.0: "Normal", 6.0: "Fixed defect", 7.0: "Reversible defect"}
    out["Thal"] = df["thal"].map(thal_map).fillna("Unknown")
    out[TARGET] = df["num"].apply(lambda x: "Yes" if x > 0 else "No")
    out["source"] = df["source"]
    # Drop rows only if core demographics or target are missing
    out = out.dropna(subset=["Age", "Sex", "CP", TARGET])
    return out


def load_discretized_dataset() -> pd.DataFrame:
    """Full pipeline: download, merge, discretize."""
    raw = load_raw_combined()
    disc = discretize_dataframe(raw)
    return disc


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on heart-disease label."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def records_to_evidence(row: pd.Series, query_var: str = TARGET) -> dict[str, str]:
    """Convert one patient row to evidence dict (all features except query)."""
    vars_used = [v for v in EXPERT_VARS if v != query_var]
    evidence = {}
    for v in vars_used:
        if v not in row.index:
            continue
        val = row[v]
        if pd.isna(val) or str(val).lower() == "nan":
            evidence[v] = "Unknown"
        else:
            evidence[v] = str(val)
    return evidence


def dataset_summary(df: pd.DataFrame) -> dict:
    """Summary stats for reports and Streamlit sidebar."""
    by_source = df.groupby("source").size().to_dict()
    prevalence = (df[TARGET] == "Yes").mean()
    return {
        "n_samples": len(df),
        "n_features": len(EXPERT_VARS) - 1,
        "prevalence": float(prevalence),
        "by_source": by_source,
        "columns": list(EXPERT_VARS),
    }


def ensure_data_cached() -> Path:
    """Ensure processed CSV exists for fast Streamlit cold start."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "heart_disease_discretized.csv"
    if not cache.exists():
        df = load_discretized_dataset()
        df.to_csv(cache, index=False)
    return cache


def load_cached_or_build() -> pd.DataFrame:
    cache = ensure_data_cached()
    return pd.read_csv(cache)
