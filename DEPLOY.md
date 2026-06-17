# Deploy to Streamlit Community Cloud

**Live app:** [https://heartdiseasepredictiondemo.streamlit.app/](https://heartdiseasepredictiondemo.streamlit.app/)

Repository: [Heart_disease_prediction_in_Probablistic_graphical_model](https://github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model)

## 1. Push to GitHub

```bash
cd heart-disease-bn
chmod +x push_to_github.sh
./push_to_github.sh
```

Or manually:

```bash
git add .
git commit -m "Update"
git push origin main
```

**Important:** Run git commands inside `heart-disease-bn` (this folder has its own `.git`), not the parent `PGM!!!!` folder.

If push fails with `src refspec main does not match any`, remove any stray parent `.git` and retry from `heart-disease-bn`.

## 2. Deploy on Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Sign in with **GitHub** (`Natiabay`)
3. Click **Create app** (or **Manage app** → reconnect if you previously used another repo)
4. Use these settings:

| Field | Value |
|-------|--------|
| Repository | `Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model` |
| Branch | `main` |
| Main file path | **`streamlit_app.py`** |

5. Click **Deploy** — public URL: [heartdiseasepredictiondemo.streamlit.app](https://heartdiseasepredictiondemo.streamlit.app/)

### Why `streamlit_app.py`?

`streamlit_app.py` at the repo root is the **Streamlit Cloud entry point**. It loads `app/streamlit_app.py` (the full six-tab demo). Use the root file path in Cloud settings.

### Redeploy

Every `git push` to `main` triggers an automatic rebuild. Check **Manage app → Logs** if a deploy fails.

### First load

The app trains/loads **4 distinct Bayesian Networks** on first visit (~1–3 min). Bundled CSVs in `data/` avoid live UCI downloads. No `secrets.toml` required.

## 3. Instructor demo script

1. Open [heartdiseasepredictiondemo.streamlit.app](https://heartdiseasepredictiondemo.streamlit.app/)
2. **Diagnosis** → **Optimized Clinical BN (recommended)** → preset **Classic angina (high risk)** → **Run inference**
3. Show **P(Heart Disease = Yes)** and risk band (Optimized model: all metrics ≥ 85%)
4. Click **Compare all 4 models** — Manual Structure, Naive Bayes, Chow-Liu, and Optimized should differ
5. **Network Explorer** → directed DAG (parent → child arrows)
6. **Algorithm Lab** → Variable Elimination vs Belief Propagation
7. **PGM Guide** — Representation, Learning, Inference pillars

## 4. Local run (before deploy)

```bash
pip install -r requirements.txt
python run.py --quick
streamlit run streamlit_app.py
```

## 5. Troubleshooting

| Issue | Fix |
|-------|-----|
| App still shows **Expert BN** or old UI | Streamlit Cloud is on an old build. In [share.streamlit.io](https://share.streamlit.io) → **Manage app** → confirm repo is `Heart_disease_prediction_in_Probablistic_graphical_model`, branch `main`, main file `streamlit_app.py`, then **Reboot app** and **Clear cache**. Sidebar should show version `2.1-manual-structure-dag`. |
| `ModuleNotFoundError: pgmpy` | Check `requirements.txt` is at repo root |
| Long startup | Normal on first load; models are cached with `@st.cache_resource` |
| Wrong repo on Cloud | Settings → reconnect to `Heart_disease_prediction_in_Probablistic_graphical_model` |
