# Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction

Probabilistic Graphical Models (PGM) course project — **interactive heart disease diagnosis** with a **deployable Streamlit demo**.

> Educational demo only — not for clinical use.

## Results (full test set, 230 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Expert BN | 0.59 | 0.58 | 0.97 | 0.72 | 0.62 |
| Naive Bayes BN | 0.59 | 0.58 | 0.97 | 0.72 | 0.62 |
| **Chow-Liu Tree BN** | **0.80** | **0.76** | **0.95** | **0.84** | **0.90** |

Decision thresholds are **tuned on the training set** to maximize F1 (standard for imbalanced diagnosis).

## Why metrics differ across models

- **Expert BN** — fixed clinical DAG; good for *interpretability*, weaker fit to noisy multi-source UCI data.
- **Naive Bayes BN** — classic symptom → disease structure from your project proposal; strong recall with tuned threshold.
- **Chow-Liu Tree BN** — structure *learned* from data; **best accuracy** and the default recommendation in the demo.

## Quick start

```bash
cd heart-disease-bn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
streamlit run app/streamlit_app.py
```

## Deploy

Push to [Natiabay/micro_research-](https://github.com/Natiabay/micro_research-.git) → [Streamlit Cloud](https://share.streamlit.io) → main file: `app/streamlit_app.py`

See **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions.

## PGM pillars

| Pillar | Implementation |
|--------|----------------|
| **Representation** | Expert DAG, Naive Bayes, Chow-Liu tree |
| **Learning** | MLE + Laplace smoothing; TreeSearch structure learning |
| **Inference** | Variable Elimination & Belief Propagation |

## Layout

```
heart-disease-bn/
├── run.py
├── app/streamlit_app.py
├── notebooks/PGM_EndToEnd_Pipeline.ipynb
├── src/
├── outputs/
└── data/heart_disease_discretized.csv
```
