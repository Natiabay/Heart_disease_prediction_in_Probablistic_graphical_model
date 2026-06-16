# Deploy to Streamlit Community Cloud

## 1. Push to GitHub

Repository: [https://github.com/Natiabay/micro_research-](https://github.com/Natiabay/micro_research-.git)

**Important:** Push from the `heart-disease-bn` folder only — not from the parent `PGM!!!!` folder.

```bash
cd ~/Desktop/PGM\!!!!/heart-disease-bn
chmod +x push_to_github.sh
./push_to_github.sh
```

If you see `src refspec main does not match any`, you are in the **wrong folder** or the parent folder has an empty git repo. Fix:

```bash
# Remove accidental empty git repo in parent (safe)
rm -rf ~/Desktop/PGM\!!!!/.git

# Then push from heart-disease-bn
cd ~/Desktop/PGM\!!!!/heart-disease-bn
git push -u origin main
```

First-time GitHub login on this machine:

```bash
sudo apt install gh
gh auth login
cd ~/Desktop/PGM\!!!!/heart-disease-bn
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
