# Deploy to Streamlit Community Cloud

## 1. Push to GitHub

Repository: [https://github.com/Natiabay/micro_research-](https://github.com/Natiabay/micro_research-.git)

```bash
cd heart-disease-bn
git init
git add .
git commit -m "Heart Disease BN — PGM medical diagnosis project"
git branch -M main
git remote add origin https://github.com/Natiabay/micro_research-.git
git push -u origin main
```

## 2. Create Streamlit app

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. **New app** → Repository: `Natiabay/micro_research-`
4. Branch: `main`
5. **Main file path:** `app/streamlit_app.py`
6. Click **Deploy**

## 3. Demo script for instructor

1. Open the public URL
2. Tab **Diagnosis** → model **Optimized Clinical BN (recommended)** → preset **Classic angina (high risk)** → **Run Bayesian inference**
3. Show **P(Heart Disease = Yes)** and risk band (all evaluation metrics ≥ 85%)
4. Expand **What-if sensitivity** — which symptom changes probability most
5. Tab **Algorithm Lab** → compare VE vs BP on same evidence
6. Tab **Network Explorer** → show Optimized Clinical BN vs Expert DAG
7. Tab **PGM Guide** — three pillars summary

## 4. Local run

```bash
pip install -r requirements.txt
python run.py --quick
streamlit run app/streamlit_app.py
```

Pre-trained models in `artifacts/` speed up cold start on Streamlit Cloud.
