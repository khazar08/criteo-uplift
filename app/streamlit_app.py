"""
Criteo Uplift — Interactive Dashboard

One screen with three panels:
  1. Qini curves for all learners (interactive legend)
  2. Targeting slider → incremental captured vs spend (live update)
  3. Uplift-by-decile bar chart for the selected model
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
from pathlib import Path

from src.metrics import (
    qini_curve,
    qini_coefficient,
    uplift_at_k,
    uplift_by_decile,
    evaluate_all,
)
from src.policy import targeting_curve, sleeping_dogs_analysis

DATA_PATH = Path(__file__).parent.parent / "data" / "meta_learner_predictions.pkl"

st.set_page_config(
    page_title="Criteo Uplift — CATE Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@st.cache_data
def load_predictions():
    if not DATA_PATH.exists():
        return None
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


saved = load_predictions()

st.title("Incrementality & Heterogeneous Treatment Effect Estimation")
st.caption("Criteo Uplift Benchmark  ·  14M-row randomized experiment  ·  CATE via S/T/X/R/DR + Causal Forest")

if saved is None:
    st.warning(
        "No predictions found. Run notebooks 03 and 04 first to generate "
        "`data/meta_learner_predictions.pkl`."
    )
    st.stop()

predictions = saved["predictions"]
y_te = saved["y_te"]
t_te = saved["t_te"]

model_names = list(predictions.keys())

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.header("Controls")
selected_models = st.sidebar.multiselect(
    "Models to show on Qini chart",
    options=model_names,
    default=model_names,
)
targeting_model = st.sidebar.selectbox(
    "Model for targeting policy",
    options=model_names,
    index=model_names.index("X-learner") if "X-learner" in model_names else 0,
)
spend_threshold = st.sidebar.slider(
    "Targeting spend threshold (% of population)",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="Top-k% of users by predicted uplift will receive the ad.",
)
n_decile_bins = st.sidebar.slider("Decile bins", 5, 20, 10, step=5)

# ------------------------------------------------------------------
# Metrics summary
# ------------------------------------------------------------------
st.subheader("Model Performance Summary")
rows = []
for name in model_names:
    tau = predictions[name]
    m = evaluate_all(y_te, tau, t_te, name)
    rows.append(m)

metric_df = pd.DataFrame(rows).set_index("model")
metric_df = metric_df.sort_values("qini_coeff", ascending=False)

best_model = metric_df["qini_coeff"].idxmax()
styled = metric_df[["qini_coeff", "uplift@10", "uplift@20", "uplift@30", "auuc"]].style.format(
    "{:.4f}"
).highlight_max(axis=0, color="#d4edda").highlight_min(axis=0, color="#f8d7da")
st.dataframe(styled, use_container_width=True)

col_l, col_r = st.columns(2)
with col_l:
    best_q = metric_df.loc[best_model, "qini_coeff"]
    st.metric("Best Qini coefficient", f"{best_q:.4f}", delta=best_model)
with col_r:
    ate = float((y_te[t_te == 1].mean() - y_te[t_te == 0].mean()))
    st.metric("Naive ATE (visit rate lift)", f"{ate:.4%}")

st.markdown("---")

# ------------------------------------------------------------------
# Row 1: Qini curves | Targeting policy
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Qini Curves")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = dict(zip(model_names, palette))

    for name in selected_models:
        tau = predictions[name]
        x, q = qini_curve(y_te, tau, t_te)
        frac = x / len(y_te)
        qc = qini_coefficient(y_te, tau, t_te)
        ax.plot(frac, q, label=f"{name} ({qc:.3f})", color=color_map[name], linewidth=1.8)

    if selected_models:
        q_last = qini_curve(y_te, predictions[selected_models[0]], t_te)[1][-1]
        ax.plot([0, 1], [0, q_last], "k--", linewidth=0.8, label="Random")

    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Qini score")
    ax.set_title("Qini curves  (Qini coeff in legend)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader(f"Targeting Policy — {targeting_model}")
    tau_pol = predictions[targeting_model]
    fractions, pct_captured, total_incr = targeting_curve(y_te, tau_pol, t_te)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(fractions * 100, pct_captured * 100, linewidth=2, color="#1f77b4", label=targeting_model)
    ax2.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="Random (proportional)")

    # current threshold
    thr_frac = spend_threshold / 100
    idx_t = min(np.searchsorted(fractions, thr_frac), len(pct_captured) - 1)
    y_thr = pct_captured[idx_t] * 100
    ax2.axvline(spend_threshold, color="red", linestyle=":", linewidth=1.2)
    ax2.axhline(y_thr, color="red", linestyle=":", linewidth=1.2)
    ax2.scatter([spend_threshold], [y_thr], color="red", zorder=5, s=60)

    ax2.set_xlabel("% of population treated (spend proxy)")
    ax2.set_ylabel("% of incremental visits captured")
    ax2.set_title("Incremental Captured vs. Spend")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # KPI callout
    st.success(
        f"**Top {spend_threshold}% by predicted uplift captures {y_thr:.1f}% of "
        f"incremental visits at {spend_threshold}% of spend.**"
    )

st.markdown("---")

# ------------------------------------------------------------------
# Row 2: Decile chart | Sleeping Dogs
# ------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader(f"Uplift by Decile — {targeting_model}")
    bins, realized = uplift_by_decile(y_te, tau_pol, t_te, n_bins=n_decile_bins)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    colors_d = ["#d62728" if v < 0 else "#1f77b4" for v in realized]
    ax3.bar(bins, realized * 100, color=colors_d)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xlabel(f"Bin (1 = highest predicted uplift), {n_decile_bins} bins")
    ax3.set_ylabel("Realized uplift (pp)")
    ax3.set_title("Monotone decrease = model ranks persuadables correctly")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col4:
    st.subheader("Sleeping Dogs (Negative Uplift Segments)")
    sd = sleeping_dogs_analysis(tau_pol, t_te, y_te)

    st.metric("Users with τ̂(x) < 0", f"{sd['n_sleeping_dogs']:,}")
    st.metric("% of test population", f"{sd['pct_sleeping_dogs']:.1f}%")
    if not np.isnan(sd["realized_uplift_sleeping_dogs"]):
        st.metric(
            "Realized uplift on Sleeping Dogs",
            f"{sd['realized_uplift_sleeping_dogs']:.4f}",
            delta="Advertising HURTS these users",
            delta_color="inverse",
        )
    st.caption(
        "Sleeping Dogs have *negative* estimated treatment effects — showing them ads "
        "reduces conversion. Excluding them improves ROI without increasing spend. "
        "This is why uplift ≠ propensity: a high-converting user can still be a Sleeping Dog."
    )

st.markdown("---")

# ------------------------------------------------------------------
# Footer: core conceptual framing
# ------------------------------------------------------------------
with st.expander("The core argument — why ROC-AUC is the wrong metric"):
    st.markdown(
        """
**Fundamental problem of causal inference:** Each user has two potential outcomes — Y(1) if shown an ad,
Y(0) if not. We observe only one. The individual treatment effect Y(1)−Y(0) is *never observed*.

**What we estimate:** CATE = τ(x) = E[Y(1)−Y(0)|X=x]. Targeting = ranking users by τ̂(x).

**The trap:** Optimizing ROC-AUC on the outcome label ranks users by *P(convert)*, not by *P(convert if treated) − P(convert if untreated)*. A Sure Thing (high baseline, near-zero uplift) has high P(convert) but wastes ad spend. A Persuadable (low baseline, high uplift) looks ordinary to the AUC model.

**Why X-learner wins here:** 85% treated / 15% control imbalance → T-learner's mu0 is noisy. X-learner imputes treatment effects from the large arm, then weights by propensity (85% on tau0 estimated from the large arm) — the exact variance fix.

**CUPED:** Reduces ATE estimate variance by ρ², shrinking required experiment size by the same factor. Directly translates to "we need X fewer users to detect the same lift."
        """
    )
