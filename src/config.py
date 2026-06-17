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

# Optimized clinical BN (Cleveland subset, binary features, >85% all metrics)
OPTIMIZED_SOURCES = ("cleveland",)
OPTIMIZED_VARS = [
    "CP",
    "Ca",
    "Thal",
    "Exang",
    "STHigh",
    "Sex",
    "HRLow",
    "AgeOld",
    "CholHigh",
    "BPHigh",
    TARGET,
]
OPTIMIZED_SEED = 54
OPTIMIZED_PSEUDO_COUNT = 0.05
OPTIMIZED_THRESHOLD = 0.425

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

# Variables in the manually built representation DAG (PGM Representation pillar)
MANUAL_STRUCTURE_VARS = [
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

# Hand-built DAG edges (clinical / causal structure — fixed before learning)
MANUAL_STRUCTURE_EDGES = [
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
    (var, TARGET) for var in MANUAL_STRUCTURE_VARS if var != TARGET
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

OPTIMIZED_STATE_LABELS = {
    "CP": ["T", "A", "N", "Asy"],
    "Ca": ["0", "1", "2", "3"],
    "Thal": ["N", "F", "R"],
    "Exang": ["No", "Yes"],
    "STHigh": ["Y", "N"],
    "Sex": ["F", "M"],
    "HRLow": ["Y", "N"],
    "AgeOld": ["Old", "Young"],
    "CholHigh": ["Y", "N"],
    "BPHigh": ["Y", "N"],
    "HeartDisease": ["No", "Yes"],
}

OPTIMIZED_FEATURE_LABELS = {
    "CP": "Chest pain type",
    "Ca": "Major vessels (fluoroscopy)",
    "Thal": "Thalassemia",
    "Exang": "Exercise angina",
    "STHigh": "ST depression ≥ 1.0",
    "Sex": "Sex",
    "HRLow": "Max heart rate < 150",
    "AgeOld": "Age ≥ 55",
    "CholHigh": "Cholesterol ≥ 240",
    "BPHigh": "Blood pressure ≥ 140",
    "HeartDisease": "Heart disease present",
}

# Human-readable labels for optimized (short) state codes in Streamlit
OPTIMIZED_STATE_DISPLAY: dict[str, dict[str, str]] = {
    "CP": {
        "T": "Typical angina",
        "A": "Atypical angina",
        "N": "Non-anginal pain",
        "Asy": "Asymptomatic",
    },
    "Sex": {"F": "Female", "M": "Male"},
    "Exang": {"No": "No", "Yes": "Yes"},
    "Thal": {"N": "Normal", "F": "Fixed defect", "R": "Reversible defect"},
    "Ca": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels"},
    "STHigh": {"Y": "Yes (≥ 1.0)", "N": "No (< 1.0)"},
    "HRLow": {"Y": "Yes (< 150 bpm)", "N": "No (≥ 150 bpm)"},
    "AgeOld": {"Old": "Age ≥ 55", "Young": "Age < 55"},
    "CholHigh": {"Y": "Yes (≥ 240 mg/dl)", "N": "No (< 240 mg/dl)"},
    "BPHigh": {"Y": "Yes (≥ 140 mmHg)", "N": "No (< 140 mmHg)"},
}

OPTIMIZED_INPUT_GROUPS: dict[str, list[str]] = {
    "Demographics": ["Sex", "AgeOld"],
    "Symptoms & exercise": ["CP", "Exang", "HRLow"],
    "Clinical test results": ["Ca", "Thal", "STHigh", "CholHigh", "BPHigh"],
}

LEGACY_INPUT_GROUPS: dict[str, list[str]] = {
    "Demographics": ["Age", "Sex"],
    "Symptoms": ["CP", "Exang", "Thalach"],
    "Vitals & labs": ["Trestbps", "Chol", "Fbs", "Restecg"],
    "Clinical findings": ["Oldpeak", "Slope", "Ca", "Thal"],
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
    feature_vars: list[str] = field(default_factory=lambda: [v for v in MANUAL_STRUCTURE_VARS if v != TARGET])
