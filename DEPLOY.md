# Deploy to Streamlit Community Cloud

## 1. Push to GitHub

Repository: [Heart_disease_prediction_in_Probablistic_graphical_model](https://github.com/Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model)

**Important:** Push from the `heart-disease-bn` folder only — not from the parent `PGM!!!!` folder.

```bash
cd ~/Desktop/PGM\!!!!/heart-disease-bn
chmod +x push_to_github.sh
./push_to_github.sh
```

If you see `src refspec main does not match any`, you are in the **wrong folder** or the parent folder has an empty git repo. Fix:

```bash
rm -rf ~/Desktop/PGM\!!!!/.git
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
2. Sign in with **GitHub** (same account: `Natiabay`)
3. Click **Create app**
4. Set:
   | Field | Value |
   |-------|--------|
   | Repository | `Natiabay/Heart_disease_prediction_in_Probablistic_graphical_model` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` or `app/streamlit_app.py` |
5. Click **Deploy** — wait until status is **Running**
6. Share the public URL

**Note:** First load trains 4 Bayesian Networks (~1–3 min). Later visits are cached.

**Redeploy:** Any push to `main` triggers an automatic rebuild.

## 3. Demo script for instructor

1. Open the public URL
2. Tab **Diagnosis** → **Optimized Clinical BN (recommended)** → preset **Classic angina (high risk)** → **Run inference**
3. Show **P(Heart Disease = Yes)** and risk band (all evaluation metrics ≥ 85%)
4. Expand **What-if sensitivity**
5. Tab **Algorithm Lab** → compare VE vs BP
6. Tab **Network Explorer** → Optimized Clinical BN vs Manual Structure DAG
7. Tab **PGM Guide** — three pillars summary

## 4. Local run

```bash
pip install -r requirements.txt
python run.py --quick
streamlit run app/streamlit_app.py
```
