"""Configuration and constants for the Heart Disease Bayesian Network."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET = "HeartDisease"
TARGET_STATES = ("No", "Yes")

# UCI Heart Disease attribute names (14 features + target)
UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

UCI_SOURCES = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

# Variables used in the expert BN (discretized names)
EXPERT_VARS = [
    "Age",
    "Sex",
    "CP",
    "Trestbps",
    "Chol",
    "Fbs",
    "Restecg",
    "Thalach",
    "Exang",
    "Oldpeak",
    "Slope",
    "Ca",
    "Thal",
    TARGET,
]

# Expert DAG edges (medical / causal intuition for heart-disease risk)
EXPERT_EDGES = [
    ("Age", "Chol"),
    ("Age", "Trestbps"),
    ("Age", "Thalach"),
    ("Age", "HeartDisease"),
    ("Sex", "HeartDisease"),
    ("Chol", "HeartDisease"),
    ("Trestbps", "HeartDisease"),
    ("CP", "HeartDisease"),
    ("Fbs", "HeartDisease"),
    ("Restecg", "HeartDisease"),
    ("Thalach", "HeartDisease"),
    ("Exang", "HeartDisease"),
    ("Oldpeak", "HeartDisease"),
    ("Slope", "HeartDisease"),
    ("Ca", "HeartDisease"),
    ("Thal", "HeartDisease"),
]

# Naive Bayes: classic medical diagnosis structure (each symptom → disease)
NAIVE_BAYES_EDGES = [
    (var, TARGET) for var in EXPERT_VARS if var != TARGET
]
FEATURE_LABELS = {
    "Age": "Age group",
    "Sex": "Sex",
    "CP": "Chest pain type",
    "Trestbps": "Resting blood pressure",
    "Chol": "Serum cholesterol",
    "Fbs": "Fasting blood sugar > 120 mg/dl",
    "Restecg": "Resting ECG results",
    "Thalach": "Max heart rate achieved",
    "Exang": "Exercise induced angina",
    "Oldpeak": "ST depression (oldpeak)",
    "Slope": "Slope of peak exercise ST segment",
    "Ca": "Major vessels colored by fluoroscopy",
    "Thal": "Thalassemia",
    "HeartDisease": "Heart disease present",
}

STATE_LABELS = {
    "Age": ["Young", "Middle", "Senior", "Elderly"],
    "Sex": ["Female", "Male"],
    "CP": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic", "Unknown"],
    "Trestbps": ["Low", "Normal", "High", "Unknown"],
    "Chol": ["Low", "Borderline", "High", "Unknown"],
    "Fbs": ["Normal", "High", "Unknown"],
    "Restecg": ["Normal", "ST-T abnormality", "LV hypertrophy", "Unknown"],
    "Thalach": ["Low", "Medium", "High", "Unknown"],
    "Exang": ["No", "Yes", "Unknown"],
    "Oldpeak": ["None", "Mild", "Severe", "Unknown"],
    "Slope": ["Upsloping", "Flat", "Downsloping", "Unknown"],
    "Ca": ["0", "1", "2", "3+", "Unknown"],
    "Thal": ["Normal", "Fixed defect", "Reversible defect", "Unknown"],
    "HeartDisease": ["No", "Yes"],
}


@dataclass
class ProjectConfig:
    """Runtime settings for training, evaluation, and demos."""

    seed: int = 42
    test_size: float = 0.25
    structure_learning_iters: int = 40
    structure_sample_size: int = 400
    ve_elimination_heuristic: str = "min_neighbors"
    quick: bool = False
    feature_vars: list[str] = field(default_factory=lambda: [v for v in EXPERT_VARS if v != TARGET])
