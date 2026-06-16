# Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction

Probabilistic Graphical Models (PGM) course project — **interactive heart disease diagnosis** from symptoms using Bayesian Networks, with a **deployable Streamlit demo**.

> Educational demo only — not for clinical use.

## What makes this project stand out

| Feature | Description |
|---------|-------------|
| **Dual BN representation** | Expert medical DAG + Chow-Liu tree learned from data |
| **Full PGM pillars** | Representation, Learning (MLE + structure), Inference (VE & BP) |
| **Interactive web demo** | Presets, random patients, what-if sensitivity, VE vs BP |
| **Multi-source UCI data** | Cleveland + Hungarian + Switzerland + VA (~920 patients) |
| **Explainable inference** | Posterior probabilities + elimination order + timing |

## Quick start

```bash
cd heart-disease-bn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --quick
streamlit run app/streamlit_app.py
```

## Deploy (public demo)

See **[DEPLOY.md](DEPLOY.md)** — push to [Natiabay/micro_research-](https://github.com/Natiabay/micro_research-) and connect on [Streamlit Cloud](https://share.streamlit.io) with main file `app/streamlit_app.py`.

## PGM pillars

### 1. Representation
- **Expert BN**: 14 nodes, clinically motivated edges (`src/config.py`)
- **Chow-Liu Tree BN**: structure from mutual information (`TreeSearch`)

### 2. Learning
- Sequential MLE + Laplace smoothing on expert DAG
- Chow-Liu tree structure + MLE CPTs

### 3. Inference
- **Variable Elimination** and **Belief Propagation** (`src/inference.py`)
- Cached inference engines for demo speed

## Instructor demo flow

1. **Diagnosis** tab → preset *Classic angina* → Run inference → show P(Yes)
2. Expand **What-if sensitivity** — symptom impact on probability
3. **Algorithm Lab** → VE vs BP agreement
4. **Network Explorer** → DAG visualization + metrics
5. **PGM Guide** → pillar summary

## Layout

```
heart-disease-bn/
├── run.py
├── app/streamlit_app.py      # Deploy this file
├── notebooks/PGM_EndToEnd_Pipeline.ipynb
├── src/                      # data, model, learning, inference, evaluation
├── outputs/                  # figures, metrics, report.json
└── artifacts/                # model metadata (models train on first run)
```

## Dataset

[UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease) — four clinical databases merged.

## Results (example)

| Model | Inference | Accuracy | F1 | ROC-AUC |
|-------|-----------|----------|-----|---------|
| Expert BN | VE | ~0.50 | ~0.14 | ~0.57 |
| Chow-Liu Tree BN | VE | ~0.76 | ~0.78 | ~0.82 |

*Chow-Liu tree fits data better; expert BN demonstrates interpretable medical structure.*
