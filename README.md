# Medical Diagnosis System Using Bayesian Networks for Heart Disease Prediction

Probabilistic Graphical Models (PGM) course project — **interactive heart disease diagnosis** with a **deployable Streamlit demo**.

> Educational demo only — not for clinical use.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Expert BN (multi-source) | 0.59 | 0.58 | 0.97 | 0.72 | 0.62 |
| Naive Bayes BN (multi-source) | 0.59 | 0.58 | 0.97 | 0.72 | 0.62 |
| Chow-Liu Tree BN (multi-source) | 0.80 | 0.76 | 0.95 | 0.84 | 0.90 |
| **Optimized Clinical BN** ★ | **0.90** | **0.89** | **0.89** | **0.89** | **0.93** |

**Optimized Clinical BN** (primary / recommended): Cleveland subset, clinical binary features, Chow-Liu tree, balanced threshold tuning — **all metrics ≥ 85%**.

Decision thresholds are tuned on train (+ validation for the optimized model).

## Why metrics differ across models

- **Expert BN** — fixed clinical DAG; good for *interpretability*, weaker fit to noisy multi-source UCI data.
- **Naive Bayes BN** — classic symptom → disease structure from your project proposal; strong recall with tuned threshold.
- **Chow-Liu Tree BN** — structure *learned* from multi-source data; strong baseline on the full dataset.
- **Optimized Clinical BN** — Cleveland + binary clinical features; **best overall metrics** and the default in the demo.

## Quick start

```bash
cd heart-disease-bn
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
streamlit run app/streamlit_app.py
```

## Deploy

**Live app:** after Streamlit Cloud setup, your URL will look like  
`https://micro-research.streamlit.app` (name chosen at deploy time).

### Streamlit Cloud (recommended)

1. Open **[share.streamlit.io](https://share.streamlit.io)** and sign in with **GitHub**
2. Click **Create app** (or **New app**)
3. Fill in:
   - **Repository:** `Natiabay/micro_research-`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Click **Deploy** — first build takes ~2–5 minutes (trains BNs on load)
5. When status is **Running**, open the public URL and demo the **Diagnosis** tab

No secrets or API keys are required. Datasets are bundled in `data/`.

### Push updates

Use single quotes so zsh does not break on `!` in the folder name:

```bash
cd '/home/natnael/Desktop/PGM!!!!/heart-disease-bn'
git push origin main
```

Streamlit Cloud redeploys automatically on each push to `main`.

See **[DEPLOY.md](DEPLOY.md)** for the instructor demo script.

## PGM pillars

| Pillar | Implementation |
|--------|----------------|
| **Representation** | Expert DAG, Naive Bayes, Chow-Liu tree, Optimized Clinical BN |
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
