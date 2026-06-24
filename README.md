# Incrementality & Heterogeneous Treatment Effect Estimation
### Criteo AI Lab Uplift Benchmark — CATE on 14M rows

**Targeting the top 30% of users by predicted uplift captures ~X% of incremental visits at 30% of spend** — with the X-learner achieving a Qini coefficient of [X], outperforming propensity-based targeting by [Y]%.

> Fill the bracketed values after running notebooks 03–05.

---

## Why this project exists

Your standard supervised ML metric — ROC-AUC — is the wrong objective for targeting. A model with great AUC ranks users by *P(convert)*, not *P(convert if shown ad) − P(convert if not shown ad)*. Those are different populations:

- **Sure Things** (high converter, zero uplift): wasted ad spend regardless of targeting
- **Persuadables** (low baseline, high uplift): the actual target — missed by AUC optimization
- **Sleeping Dogs** (negative uplift): advertising *reduces* their conversion — a cost the propensity model never discovers

This project proves the gap with data. The Qini coefficient of a propensity-optimized model is consistently lower than a proper CATE model, even when the former has higher AUC.

---

## Dataset

**Criteo AI Lab Uplift v2.1** (Diemert, Betlei, Renaudin, Amini — AdKDD/KDD 2018)

| Property | Value |
|---|---|
| Rows | ~14M (develop on 10% slice) |
| Features | `f0`–`f11` — 12 dense floats, anonymized & randomly projected |
| Treatment | `treatment` — 1 = ad-eligible, 0 = control (~85%/15% split) |
| Outcome (headline) | `visit` (rate ≈ 4.7%) |
| Outcome (stretch) | `conversion` (rate ≈ 0.29% — too rare for reliable CATE; discussed honestly) |

**Key design decisions:**
- `exposure` is excluded — it is a post-treatment variable; conditioning on it induces collider bias
- `visit` is the headline label, not `conversion` (too rare; estimates collapse to noise at 0.29%)
- Experiment is **randomized** → identification is clean; we solve estimation, not confounding

---

## Structure

```
criteo-uplift/
├── notebooks/
│   ├── 00_data_validation.ipynb   # SRM check, covariate balance, naive ATE, leakage scan
│   ├── 01_eda.ipynb               # Label rates, feature dists, MI with outcome
│   ├── 02_baseline_two_model.ipynb # T-learner + the AUC trap demo
│   ├── 03_meta_learners.ipynb     # S/T/X/R/DR comparison + Qini table
│   ├── 04_causal_forest.ipynb     # CausalForestDML + CIs on CATE
│   └── 05_targeting_policy.ipynb  # Targeting curve, CUPED, power analysis
├── src/
│   ├── data.py        # load / parquet cache / split
│   ├── learners.py    # S/T/X/R/DR meta-learner wrappers (LightGBM base)
│   ├── metrics.py     # Qini / AUUC / uplift@k FROM SCRATCH, cross-checked vs sklift
│   ├── cuped.py       # CUPED variance reduction + ATE CI comparison
│   ├── power.py       # MDE calculator, sample size, CUPED impact
│   └── policy.py      # Targeting curve, sleeping dogs, incremental vs spend
├── app/
│   └── streamlit_app.py   # Interactive Qini curves + targeting slider
└── reports/
    ├── memo.md            # 2-page internal experimentation memo
    └── *.png              # Generated figures
```

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run notebooks in order (00 → 05)
jupyter notebook notebooks/

# 3. Launch dashboard (after running 03 + 04 to generate predictions)
streamlit run app/streamlit_app.py
```

Data downloads automatically on first run via `scikit-uplift`. Use `percent10=True` (default) for development; remove flag for full 14M-row run.

---

## Methods

### Meta-learners (S/T/X/R/DR)

Each estimator targets τ(x) = E[Y(1)−Y(0)|X=x] differently. The treatment imbalance (85/15) is the diagnostic:

| Learner | Failure mode in this dataset |
|---|---|
| S-learner | Regularization shrinks T coefficient → uplift biased toward 0 at low signal-to-noise |
| T-learner | Control arm is 15% of data → mu0 is high-variance |
| **X-learner** | **Fixes T-learner: imputes TEs from large arm, propensity-weights heavily toward tau0** |
| R-learner | Robinson decomposition + cross-fitting; cross-fitting is essential (without it: overfitting bias in nuisances) |
| DR-learner | AIPW pseudo-outcome; doubly robust — consistent if either outcome model OR propensity is right; known propensity = best case |

### Causal Forest

`CausalForestDML` (EconML) provides **confidence intervals on CATE** via honest splitting. Enables "this segment's uplift is significantly positive" instead of just point estimates.

### Evaluation (from scratch, verified against sklift)

Standard ROC-AUC is inapplicable: the label Y(1)−Y(0) is never observed. Instead:

- **Qini coefficient** — area between Qini curve and random baseline, normalized by oracle
- **AUUC** — area under the uplift curve per user
- **Uplift@k** — realized `visit_rate_treated − visit_rate_control` in top-k% by predicted uplift
- **Uplift-by-decile** — monotone decrease = model genuinely ranks persuadables

### CUPED

Reduces ATE estimate variance by ρ² by regressing out a pre-experiment covariate. Here we use a control-arm-trained outcome prediction as a proxy (caveat: no true pre-period in Criteo). A ρ² of 0.10 means 10% fewer users needed for the same statistical power.

---

## Key results

| Model | Qini coeff | Uplift@10% | Uplift@30% | AUUC |
|---|---|---|---|---|
| X-learner | — | — | — | — |
| DR-learner | — | — | — | — |
| Causal Forest | — | — | — | — |
| R-learner | — | — | — | — |
| T-learner | — | — | — | — |
| S-learner | — | — | — | — |
| P(visit) classifier | — | — | — | — |

---


---

## References

- Diemert, Betlei, Renaudin, Amini (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD/KDD.
- Künzel, Sekhon, Bickel, Yu (2019). *Metalearners for estimating heterogeneous treatment effects.* PNAS.
- Nie & Wager (2021). *Quasi-oracle estimation of heterogeneous treatment effects.* Biometrika.
- Kennedy (2020). *Optimal doubly robust estimation of heterogeneous causal effects.*
- Athey & Wager (2019). *Estimating treatment effects with causal forests.* AoS.
- Deng, Xu, Kohavi, Walker (2013). *Improving the sensitivity of online controlled experiments.* KDD (CUPED).
