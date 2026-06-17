# Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction

Probabilistic Graphical Models (PGM) course project — **interactive heart disease diagnosis** with Streamlit.

**Live app:** [heartdiseasepredictiondemo.streamlit.app](https://heartdiseasepredictiondemo.streamlit.app/)

> Educational demo only — not for clinical use.

**Authors:** Abiy Alemu & Natnael Abayneh — Addis Ababa University, M.Sc. in AI

## Results (hold-out test)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Manual Structure BN (920 pts) | 0.60 | 0.58 | 0.96 | 0.72 | 0.63 |
| Naive Bayes BN (920 pts) | 0.60 | 0.58 | 0.96 | 0.72 | 0.63 |
| Chow-Liu Tree BN (920 pts) | 0.78 | 0.73 | 0.95 | 0.83 | 0.89 |
| **Optimized Clinical BN** ★ | **0.895** | **0.886** | **0.886** | **0.886** | **0.925** |

**Optimized Clinical BN** (recommended): Cleveland subset, 10 binary clinical features, Chow-Liu tree — **all metrics ≥ 85%**.

## Quick start

```bash
cd heart-disease-bn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
streamlit run app/streamlit_app.py
```

Or from repo root: `streamlit run streamlit_app.py`

## Live demo (Streamlit Cloud)

**Try the deployed app:** [https://heartdiseasepredictiondemo.streamlit.app/](https://heartdiseasepredictiondemo.streamlit.app/)

Hosted on [Streamlit Community Cloud](https://share.streamlit.io) from this repository:

| Setting | Value |
|---------|--------|
| **Live URL** | [heartdiseasepredictiondemo.streamlit.app](https://heartdiseasepredictiondemo.streamlit.app/) |
| **Repository** | [`Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model`](https://github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model) |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |

### Redeploy after code changes

After code changes:

```bash
cd heart-disease-bn
git add .
git commit -m "Your message"
git push origin main
```

Streamlit Cloud **redeploys automatically** on each push to `main`.

**Note:** The first visitor after a cold start may wait ~1–3 minutes while all 4 Bayesian Networks load; later visits use Streamlit cache.

See **DEPLOY.md** for the instructor demo script and troubleshooting.

## Streamlit interactive demo

Six tabs:

| Tab | Purpose |
|-----|---------|
| **Diagnosis** | Enter evidence, run inference, **compare all 4 models** |
| **Metrics** | Accuracy, precision, recall, F1, ROC-AUC |
| **Analysis Gallery** | EDA, CPT, inference figures |
| **Algorithm Lab** | Variable Elimination vs Belief Propagation |
| **Network Explorer** | **Directed** DAG (parent → child arrows) |
| **PGM Guide** | Three-pillar walkthrough |

**Demo tip:** Preset *Classic angina (high risk)* → **Compare all 4 models** — each BN returns a different P(Heart Disease).

## Why metrics differ across models

- **Manual Structure / Naive Bayes** — PGM representation on multi-source UCI data (920 patients).
- **Chow-Liu Tree** — learned structure on multi-source data.
- **Optimized Clinical BN** — Cleveland-only, best ML performance (default in app).

## PGM pillars

| Pillar | Implementation |
|--------|----------------|
| **Representation** | Manual Structure DAG, Naive Bayes, Chow-Liu tree, Optimized Clinical BN |
| **Learning** | MLE + Laplace smoothing; Chow-Liu TreeSearch |
| **Inference** | Variable Elimination & Belief Propagation |

## Layout

```
heart-disease-bn/
├── run.py
├── streamlit_app.py
├── app/streamlit_app.py
├── presentation-overleaf/
├── src/
├── outputs/
└── data/
```

## Presentation

Upload `presentation-overleaf.zip` to [Overleaf](https://www.overleaf.com), compile `main.tex`.

GitHub: [Heart_disease_prediction_in_Probablistic_graphical_model](https://github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model) · Live demo: [heartdiseasepredictiondemo.streamlit.app](https://heartdiseasepredictiondemo.streamlit.app/)
