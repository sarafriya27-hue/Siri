# Claim Settlement Bias Audit Dashboard

A Streamlit dashboard built to investigate potential bias in an insurance
claim settlement process — descriptive cross-tabs, statistical bias
diagnostics (age / income / team / etc.), and 4 supervised classification
models (KNN, Decision Tree, Random Forest, Gradient Boosting) with full
evaluation (accuracy, precision/recall/F1, ROC curves, confusion matrices,
and FP/FN % contribution).

> ⚠️ **This repo ships with a SYNTHETIC sample dataset** (`data/claims_data_SAMPLE.csv`)
> so you can see the whole dashboard working immediately. It is fake data with
> deliberately injected bias patterns, generated purely for demo purposes —
> **it is not real claims data.** Upload your own CSV from the sidebar to run
> a genuine audit on your company's data.

---

## 1. What it does

| # | Objective | Where |
|---|-----------|-------|
| 1 | Descriptive cross-tabulation vs Policy Status | Tab 1 |
| 2 | Diagnostic bias analysis (age/income/team-wise, with Chi² significance test) | Tab 2 |
| 3 | Feature engineering + KNN / Decision Tree / Random Forest / Gradient Boosting | Tab 3 |
| 4 | Train/test accuracy, precision/recall/F1, ROC curves, confusion matrices, FP/FN % | Tab 4 |
| 5 | Auto-generated findings & recommendations | Tab 5 |

## 2. Project structure

```
claim_bias_project/
├── app.py                      # Main Streamlit app (run this)
├── generate_sample_data.py     # Recreates the synthetic demo dataset
├── requirements.txt
├── data/
│   └── claims_data_SAMPLE.csv  # Synthetic demo data
├── utils/
│   ├── data_prep.py            # Column auto-mapping + feature engineering
│   ├── bias_analysis.py        # Cross-tabs, Chi-square test, hot-spot flagging
│   ├── ml_models.py            # Train/evaluate KNN, DT, RF, GBM
│   └── findings.py             # Auto-generated narrative findings
└── .streamlit/config.toml      # Theme config
```

## 3. Run it locally

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

It will open at `http://localhost:8501`. Use the **sidebar** to either keep
the bundled sample data, or upload your real claims CSV.

### Using your own dataset

1. In the sidebar, upload your CSV.
2. Expand **"Map your columns"** — the app tries to auto-detect columns like
   Age, Income, Settlement Team, Policy Status, and the settlement outcome
   column, but you should verify/correct the mapping.
3. The **Claim Settlement Outcome** column is required — it should contain
   values like `Approved`/`Rejected` (or `Settled`/`Denied`, `Yes`/`No`, etc.
   — the app recognizes common positive labels automatically).
4. Everything downstream (cross-tabs, bias diagnostics, ML training,
   evaluation, findings) recalculates automatically against your real data.

## 4. Deploy to Streamlit Community Cloud via GitHub

1. **Push this folder to a new GitHub repo:**

   ```bash
   cd claim_bias_project
   git init
   git add .
   git commit -m "Initial commit: claim settlement bias audit dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with
   GitHub → **"New app"**.
3. Select your repo, branch `main`, and main file path `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app — you'll get a public URL to share.
5. To update later, just `git push` — Streamlit Cloud redeploys automatically.

## 5. Regenerating the synthetic sample data (optional)

```bash
python3 generate_sample_data.py
```

This recreates `data/claims_data_SAMPLE.csv` with a fresh random seed.

## 6. Notes on methodology

- **Bias diagnostics** use a Chi-square test of independence (segment vs.
  outcome) plus Cramer's V as an effect-size measure, alongside a simple
  "approval-rate gap vs. portfolio average" metric per segment.
- **Feature engineering**: numeric fields are median-imputed; categoricals
  are one-hot encoded; Age and Income are additionally binned for the
  diagnostics view. KNN features are standardized (z-scored); tree-based
  models use raw scale.
- **Train/test split** is stratified to preserve class balance.
- **FP/FN %** = false positives / false negatives as a percentage of *all*
  test predictions — a quick way to see which model's errors are biggest,
  and which type of error (wrongly approving vs. wrongly rejecting) dominates.
- This dashboard surfaces **statistical association**, not legal or causal
  proof of discriminatory intent. Pair findings with a qualitative process
  review (SOPs, training, workload) before drawing conclusions.
