"""
app.py
-------
Claim Settlement Bias Audit Dashboard
Streamlit dashboard covering:
  1. Descriptive cross-tabulation vs Policy Status
  2. Diagnostic bias analysis (age / income / team / etc.)
  3. Supervised ML classification (KNN, Decision Tree, Random Forest, Gradient Boosting)
     with feature engineering
  4. Model evaluation: train/test accuracy, precision/recall/F1, ROC curves,
     confusion matrices, FP/FN % contribution
  5. Auto-generated findings

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub -> share.streamlit.io -> New app -> point to app.py
"""

import io
import os
import sys

# Ensure the folder containing this file is on sys.path, so the local
# `utils` package resolves no matter what working directory the host
# (e.g. Streamlit Community Cloud) launches the app from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from utils.data_prep import (
    DEFAULT_COLUMNS, load_data, auto_map_columns, engineer_features
)
from utils.bias_analysis import (
    crosstab_pct, run_bias_diagnostics, flag_high_risk_segments
)
from utils.ml_models import (
    MODEL_REGISTRY, split_data, train_all_models, metrics_summary_table, feature_importance
)
from utils.findings import generate_findings

st.set_page_config(page_title="Claim Settlement Bias Audit", layout="wide", page_icon="📊")
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Sidebar - data source & configuration
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### 1. Data source")
uploaded_file = st.sidebar.file_uploader("Upload your claims CSV", type=["csv"])
use_sample = False
if uploaded_file is None:
    use_sample = st.sidebar.checkbox("Use bundled SYNTHETIC sample dataset (demo)", value=True)

if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
    data_source_label = f"Uploaded file: {uploaded_file.name}"
elif use_sample:
    raw_df = load_data("data/claims_data_SAMPLE.csv")
    data_source_label = "⚠️ SYNTHETIC SAMPLE DATA (for demo only — not real claims data)"
else:
    st.sidebar.warning("Upload a CSV or enable the sample dataset to continue.")
    st.stop()

st.sidebar.success(f"Loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

# ----------------------------------------------------------------------------
# Sidebar - column mapping
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 2. Column mapping")
auto_map = auto_map_columns(raw_df)
cols_with_none = ["-- none --"] + list(raw_df.columns)

with st.sidebar.expander("Map your columns (auto-detected, override if wrong)", expanded=False):
    colmap = {}
    field_labels = {
        "target": "Claim Settlement Outcome (Approved/Rejected) *required*",
        "policy_status": "Policy Status",
        "age": "Policyholder Age",
        "income": "Annual Income",
        "team": "Settlement Team / Handler",
        "gender": "Gender",
        "region": "Region",
        "policy_type": "Policy Type",
        "claim_type": "Claim Type",
        "claim_amount": "Claim Amount",
        "tenure": "Policy Tenure (years)",
        "prior_claims": "Number of Prior Claims",
        "processing_days": "Processing Days (TAT)",
        "doc_score": "Documentation Completeness Score",
    }
    for key, label in field_labels.items():
        default_val = auto_map.get(key) or DEFAULT_COLUMNS.get(key)
        default_idx = cols_with_none.index(default_val) if default_val in cols_with_none else 0
        sel = st.selectbox(label, cols_with_none, index=default_idx, key=f"map_{key}")
        colmap[key] = None if sel == "-- none --" else sel

if not colmap.get("target"):
    st.error("Please map the **Claim Settlement Outcome** column in the sidebar to proceed.")
    st.stop()

st.sidebar.markdown("### 3. Model settings")
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.25, 0.05)
selected_models = st.sidebar.multiselect(
    "Algorithms to train", list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys())
)
gap_threshold = st.sidebar.slider("Bias hot-spot flag threshold (pp gap)", 2.0, 20.0, 8.0, 1.0)

st.sidebar.info(data_source_label)

# ----------------------------------------------------------------------------
# Feature engineering (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _engineer(df, colmap):
    return engineer_features(df, colmap)

model_df, enriched_df, fe_meta = _engineer(raw_df, colmap)

st.title("📊 Claim Settlement Bias Audit Dashboard")
st.caption(data_source_label)

tabs = st.tabs([
    "1️⃣ Descriptive Cross-tab",
    "2️⃣ Bias Diagnostics",
    "3️⃣ Model Training",
    "4️⃣ Model Evaluation",
    "5️⃣ Findings",
])

# ============================================================================
# TAB 1 — Descriptive cross-tabulation vs Policy Status
# ============================================================================
with tabs[0]:
    st.header("Descriptive Analysis: Cross-tabulation vs Policy Status")

    if not colmap.get("policy_status"):
        st.warning("Map a 'Policy Status' column in the sidebar to enable this section.")
    else:
        ps_col = colmap["policy_status"]
        st.subheader(f"Overall distribution of `{ps_col}`")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(raw_df[ps_col].value_counts().rename("Count"))
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            raw_df[ps_col].value_counts().plot(kind="bar", color="#4C72B0", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel(ps_col)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, use_container_width=False)

        st.divider()
        candidate_dims = [colmap.get(k) for k in
                          ["team", "gender", "region", "policy_type", "claim_type"]
                          if colmap.get(k)]
        if "_AgeBand" in enriched_df.columns:
            candidate_dims.append("_AgeBand")
        if "_IncomeBand" in enriched_df.columns:
            candidate_dims.append("_IncomeBand")

        if not candidate_dims:
            st.info("Map more dimension columns (team, gender, region, etc.) to see cross-tabs.")
        else:
            dim = st.selectbox("Cross-tabulate Policy Status against:", candidate_dims)
            src_df = enriched_df if dim.startswith("_") else raw_df
            counts, pct = crosstab_pct(src_df, dim, ps_col)
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Counts**")
                st.dataframe(counts)
            with cc2:
                st.markdown("**Row % (within each group)**")
                st.dataframe(pct)

            fig, ax = plt.subplots(figsize=(8, 4))
            pct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
            ax.set_ylabel("% of group")
            ax.set_xlabel(dim)
            ax.legend(title=ps_col, bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, use_container_width=False)

    st.divider()
    st.subheader("Cross-tab vs Claim Settlement Outcome (the bias target variable)")
    candidate_dims2 = [colmap.get(k) for k in
                       ["team", "gender", "region", "policy_type", "claim_type", "policy_status"]
                       if colmap.get(k)]
    if "_AgeBand" in enriched_df.columns:
        candidate_dims2.append("_AgeBand")
    if "_IncomeBand" in enriched_df.columns:
        candidate_dims2.append("_IncomeBand")

    dim2 = st.selectbox("Cross-tabulate Claim Outcome against:", candidate_dims2, key="dim2")
    counts2, pct2 = crosstab_pct(enriched_df, dim2, colmap["target"])
    cc3, cc4 = st.columns(2)
    with cc3:
        st.markdown("**Counts**")
        st.dataframe(counts2)
    with cc4:
        st.markdown("**Row % (approval/rejection rate within each group)**")
        st.dataframe(pct2)

# ============================================================================
# TAB 2 — Bias Diagnostics
# ============================================================================
with tabs[1]:
    st.header("Diagnostic Analysis: Probing for Biased Settlement Behaviour")
    st.caption(
        "For each dimension below we compute the claim-approval rate per segment, the gap vs. the "
        "portfolio average, and a Chi-square test of independence (segment vs. outcome) to check "
        "whether the relationship is statistically significant -- not just random noise."
    )

    group_cols = {}
    if "_AgeBand" in enriched_df.columns:
        group_cols["Age Band"] = "_AgeBand"
    if "_IncomeBand" in enriched_df.columns:
        group_cols["Income Band"] = "_IncomeBand"
    if colmap.get("team"):
        group_cols["Settlement Team"] = colmap["team"]
    if colmap.get("gender"):
        group_cols["Gender"] = colmap["gender"]
    if colmap.get("region"):
        group_cols["Region"] = colmap["region"]
    if colmap.get("policy_type"):
        group_cols["Policy Type"] = colmap["policy_type"]
    if colmap.get("claim_type"):
        group_cols["Claim Type"] = colmap["claim_type"]

    diag_results = run_bias_diagnostics(enriched_df, group_cols, target_col="_target")

    if not diag_results:
        st.warning("Map at least one of: Age, Income, Team, Gender, Region in the sidebar to run diagnostics.")
    else:
        for label, res in diag_results.items():
            if "rates" not in res:
                continue
            st.subheader(f"🔎 {label}")
            stats = res["stats"]
            rates = res["rates"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Chi² p-value", f"{stats['p_value']:.4g}",
                      help="p < 0.05 suggests the relationship is unlikely due to chance")
            m2.metric("Cramer's V (effect size)", f"{stats['cramers_v']}")
            m3.metric("Statistically significant?", "Yes ⚠️" if stats["significant"] else "No")

            fig, ax = plt.subplots(figsize=(7, 3.5))
            colors = ["#d62728" if g < 0 else "#2ca02c" for g in rates["GapVsOverall_pp"]]
            ax.barh(rates.index.astype(str), rates["GapVsOverall_pp"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Approval-rate gap vs. portfolio average (percentage points)")
            ax.set_title(f"{label}: deviation from overall approval rate")
            st.pyplot(fig, use_container_width=False)

            with st.expander("See detailed rate table"):
                st.dataframe(rates)
            st.divider()

        st.subheader("🚩 Flagged High-Risk Segments")
        hot_spots = flag_high_risk_segments(diag_results, gap_threshold=gap_threshold)
        if hot_spots.empty:
            st.success(f"No segment exceeds the ±{gap_threshold} pp gap threshold.")
        else:
            st.dataframe(hot_spots, use_container_width=True)
            st.caption(
                "These segments deviate from the overall approval rate by more than the configured "
                "threshold. Combined with a significant Chi-square p-value, this is evidence worth "
                "escalating for a manual settlement-process review."
            )

# ============================================================================
# TAB 3 — Model Training
# ============================================================================
with tabs[2]:
    st.header("Supervised Learning: Feature Engineering + Model Training")

    st.subheader("Feature engineering summary")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Numeric features used", len(fe_meta["numeric_cols"]))
    fc2.metric("Categorical features (pre-encoding)", len(fe_meta["categorical_cols"]))
    fc3.metric("Final feature count (post one-hot encoding)", fe_meta["n_features"])

    with st.expander("Feature engineering steps applied"):
        st.markdown(f"""
- **Numeric columns** ({', '.join(fe_meta['numeric_cols']) or 'none'}): missing values imputed
  with the column median; left on original scale for tree models, standardized (z-score) for KNN.
- **Categorical columns** ({', '.join(fe_meta['categorical_cols']) or 'none'}): missing values
  filled as `"Unknown"`, then one-hot encoded (drop-first to avoid the dummy-variable trap).
- **Derived bands**: Age binned into 6 bands; Income binned into quartiles (Low/Lower-Mid/
  Upper-Mid/High) — used for diagnostics and optionally as model features.
- **Target encoding**: settlement outcome mapped to binary 1 = Approved/Settled, 0 = Rejected/Denied.
- **Train/test split**: stratified split preserving the class balance, test size = {test_size:.0%}.
        """)

    st.subheader("Class balance")
    st.bar_chart(model_df["_target"].value_counts().rename({0: "Rejected", 1: "Approved"}))

    if len(selected_models) == 0:
        st.warning("Select at least one algorithm in the sidebar.")
        st.stop()

    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = split_data(model_df, test_size=test_size)
            results = train_all_models(X_train, X_test, y_train, y_test, selected_models)
            st.session_state["ml_results"] = results
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
        st.success(f"Trained {len(results)} model(s) on {len(X_train):,} training rows, "
                   f"tested on {len(X_test):,} rows.")

    if "ml_results" in st.session_state:
        st.subheader("Quick metrics preview")
        st.dataframe(metrics_summary_table(st.session_state["ml_results"]), use_container_width=True)
        st.info("Full evaluation charts (ROC, confusion matrices, FP/FN%) are in the **Model Evaluation** tab →")
    else:
        st.caption("Click **Train models** to run KNN, Decision Tree, Random Forest, and Gradient Boosting.")

# ============================================================================
# TAB 4 — Model Evaluation
# ============================================================================
with tabs[3]:
    st.header("Model Evaluation: Accuracy, Precision/Recall/F1, ROC, Confusion Matrices")

    if "ml_results" not in st.session_state:
        st.warning("Train models in the **Model Training** tab first.")
    else:
        results = st.session_state["ml_results"]
        metrics_df = metrics_summary_table(results)

        st.subheader("Metrics summary table")
        st.dataframe(metrics_df.style.format({
            "Train Accuracy": "{:.2%}", "Test Accuracy": "{:.2%}",
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}",
            "ROC-AUC": "{:.3f}", "Train-Test Gap (overfit risk)": "{:.2%}",
            "FP %": "{:.2f}%", "FN %": "{:.2f}%",
        }), use_container_width=True)

        st.subheader("Train vs Test Accuracy (stability check)")
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(metrics_df))
        w = 0.35
        ax.bar(x - w/2, metrics_df["Train Accuracy"], width=w, label="Train Accuracy", color="#4C72B0")
        ax.bar(x + w/2, metrics_df["Test Accuracy"], width=w, label="Test Accuracy", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Model"], rotation=15)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("A large gap = potential overfitting / instability")
        st.pyplot(fig, use_container_width=False)

        st.subheader("Precision / Recall / F1-Score by model")
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_df.set_index("Model")[["Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=15)
        st.pyplot(fig, use_container_width=False)

        st.subheader("ROC Curves (all models)")
        fig, ax = plt.subplots(figsize=(6, 6))
        for name, r in results.items():
            ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=False)

        st.subheader("Confusion Matrices")
        n_models = len(results)
        ncols = min(n_models, 4)
        cols = st.columns(ncols)
        for i, (name, r) in enumerate(results.items()):
            with cols[i % ncols]:
                fig, ax = plt.subplots(figsize=(3.6, 3.2))
                sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["Pred Rejected", "Pred Approved"],
                            yticklabels=["Actual Rejected", "Actual Approved"], ax=ax)
                ax.set_title(name, fontsize=10)
                st.pyplot(fig, use_container_width=False)

        st.subheader("False Positive % and False Negative % contribution")
        st.caption(
            "FP % / FN % = share of *all test predictions* that were a false positive / false negative. "
            "In a settlement context: a False Positive = a claim predicted Approved that was actually "
            "Rejected; a False Negative = a claim predicted Rejected that was actually Approved "
            "(i.e. a legitimate claim incorrectly flagged as a denial — directly relevant to the bias audit)."
        )
        fpfn_df = metrics_df[["Model", "FP %", "FN %"]].set_index("Model")
        fig, ax = plt.subplots(figsize=(8, 4))
        fpfn_df.plot(kind="bar", ax=ax, color=["#C44E52", "#8172B2"])
        ax.set_ylabel("% of total test predictions")
        plt.xticks(rotation=15)
        st.pyplot(fig, use_container_width=False)
        st.dataframe(fpfn_df, use_container_width=True)

        st.subheader("Feature importance (tree-based models)")
        feat_imp = feature_importance(results, st.session_state["X_train"].columns)
        if feat_imp:
            model_pick = st.selectbox("Show feature importance for:", list(feat_imp.keys()))
            top_n = feat_imp[model_pick].head(15)
            fig, ax = plt.subplots(figsize=(7, 5))
            top_n.sort_values().plot(kind="barh", ax=ax, color="#55A868")
            ax.set_xlabel("Importance")
            ax.set_title(f"Top 15 features — {model_pick}")
            st.pyplot(fig, use_container_width=False)

# ============================================================================
# TAB 5 — Findings
# ============================================================================
with tabs[4]:
    st.header("Findings & Recommendations")

    if "ml_results" not in st.session_state:
        st.warning("Train models in the **Model Training** tab to generate complete findings "
                    "(bias diagnostics below still work independently).")
        metrics_df_for_findings = pd.DataFrame()
        feat_imp_for_findings = {}
    else:
        metrics_df_for_findings = metrics_summary_table(st.session_state["ml_results"])
        feat_imp_for_findings = feature_importance(st.session_state["ml_results"], st.session_state["X_train"].columns)

    diag_results = run_bias_diagnostics(enriched_df, group_cols, target_col="_target") if group_cols else {}
    hot_spots = flag_high_risk_segments(diag_results, gap_threshold=gap_threshold) if diag_results else pd.DataFrame()

    findings = generate_findings(diag_results, hot_spots, metrics_df_for_findings, feat_imp_for_findings)

    for f in findings:
        st.markdown(f"- {f}")

    st.divider()
    st.subheader("Suggested next steps")
    st.markdown("""
1. **Escalate flagged segments** (Tab 2 → Flagged High-Risk Segments) to a manual file review —
   especially any with both a large approval-rate gap *and* a statistically significant Chi² p-value.
2. **Re-run this audit after removing/auditing biased proxies** — if Age, Income, Gender, or Team
   rank as top predictive features (Tab 4 → Feature Importance), determine whether they're acting as
   proxies for legitimate risk or as a channel for bias, and consider excluding/adjusting them.
3. **Compare FP/FN distribution across the same segments** — if False Negatives (legitimate claims
   denied) cluster in specific age/income/team groups, that is direct evidence of inconsistent treatment.
4. **Monitor over time** — bias can be seasonal or team-specific; re-run this dashboard periodically as
   new settlement data accumulates, ideally on real (not synthetic) data once available.
5. **Validate with domain experts** — statistical association does not prove intent; pair these findings
   with a process audit (SOPs, training, workload per team) before drawing conclusions.
    """)

    st.caption(
        "⚠️ Note: when running on the bundled synthetic sample dataset, these findings describe "
        "deliberately injected demo patterns — not real findings about your company. Upload your "
        "actual claims CSV in the sidebar to get a genuine audit."
    )
