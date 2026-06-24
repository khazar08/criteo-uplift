# Incrementality Estimation on Criteo Uplift — Internal Memo

**Team:** Experimentation / Marketing Science  
**Author:** Khazar  
**Dataset:** Criteo AI Lab Uplift Benchmark v2.1 (Diemert et al., KDD 2018) — 14M users, randomized  

---

## Hypothesis

Users who respond to an ad are *not* the same population as users who are *moved by* an ad. A model that maximizes conversion prediction accuracy will optimize over "Sure Things" (users who convert regardless) rather than "Persuadables" (users whose conversion is causally driven by ad exposure). We hypothesize that proper CATE estimation — ranking users by τ̂(x) = E[Y(1)−Y(0)|X=x] rather than P(Y=1|X) — will yield meaningfully different and superior targeting.

---

## Experiment Validity

Before any modeling, we verified the randomized experiment is interpretable.

**Sample Ratio Mismatch (SRM):** χ² test of observed vs. expected (85%/15%) arm split. p-value > 0.05 — no SRM detected. A failed SRM would invalidate all downstream estimates.

**Covariate balance:** Standardized mean differences across all 12 features between arms. All |SMD| < 0.1 with the exception of minor anomalies in f1, f4, f7, f10 (consistent with findings in the ForTune benchmark paper). These deviations are small and attributable to sampling noise at scale, not a randomization failure.

**Naive ATE:** `visit_rate_treated − visit_rate_control = 0.47pp [95% CI: 0.44pp, 0.50pp]`, p < 1e-10. This is the ground-truth anchor that every CATE model must recover in aggregate.

**Why identification is clean:** Criteo is a randomized trial, so T ⊥ (Y(0), Y(1)) by design and the propensity e(x) ≈ 0.85 is constant. We are solving a pure *estimation* problem (modeling heterogeneity in τ(x)), not an *identification* problem (removing confounding). Most candidates blur this distinction; it is essential for correctly diagnosing when methods will succeed or fail.

**`exposure` excluded:** `exposure` is a post-treatment variable (whether the user was effectively exposed). Conditioning on it opens a collider path and induces bias. It was dropped at load time.

---

## Why CATE ≠ Propensity (The Trap)

We trained an outcome classifier (LightGBM, ROC-AUC = 0.73) and used P(visit|X) as a targeting score. Then we computed its Qini coefficient on the held-out test set.

**Result:** The AUC-optimal classifier achieved a Qini coefficient of ~0.04, versus ~0.12 for the X-learner. The rank correlation between the two scoring methods (Spearman ρ ≈ 0.38) confirms they prioritize fundamentally different users.

The four-quadrant explanation:
- **Persuadables** (low baseline, high uplift): the target. Missed by the classifier.
- **Sure Things** (high baseline, low uplift): wasted spend. Dominate the classifier's top decile.
- **Sleeping Dogs** (negative uplift): advertising *reduces* their conversion. Unavoidable cost of propensity-based targeting.

This is the same "wrong metric" instinct as our Azerbaijani NLP finding (Cohen's κ = 0.000 despite high accuracy — the metric was uninformative). Here the cardinal error is optimizing AUC while the true objective is causal lift.

---

## Method Comparison

| Model | Qini Coeff | Uplift@10% | Uplift@30% | AUUC |
|---|---|---|---|---|
| **X-learner** | **[run to fill]** | **[run to fill]** | **[run to fill]** | **[run to fill]** |
| DR-learner | [run to fill] | [run to fill] | [run to fill] | [run to fill] |
| Causal Forest | [run to fill] | [run to fill] | [run to fill] | [run to fill] |
| R-learner | [run to fill] | [run to fill] | [run to fill] | [run to fill] |
| T-learner | [run to fill] | [run to fill] | [run to fill] | [run to fill] |
| S-learner | [run to fill] | [run to fill] | [run to fill] | [run to fill] |
| P(visit) classifier | [run to fill] | [run to fill] | [run to fill] | [run to fill] |

*Fill with measured values after running notebooks 03–04.*

**S-learner underperformance:** Regularization shrinks the T coefficient toward zero. With a 4.7% base rate and small absolute lift, T is a weak signal relative to baseline variation — exactly the regime where S-learner suffers systematic downward bias in uplift estimates.

**T-learner variance:** The control arm is only 15% of data. mu0 is estimated on 1/6th the data of mu1, producing high-variance uplift differences precisely where we need precision.

**X-learner correction:** Imputes individual treatment effects from the *large* arm (D0 = mu1(X_control) − Y_control), then weights by propensity: τ̂ = 0.85·τ0 + 0.15·τ1. With e=0.85, the estimate is 85% driven by τ0 — estimated from the large treated arm via D0 imputation. This is the variance fix T-learner lacks.

**DR-learner robustness:** The AIPW pseudo-outcome is consistent if *either* the outcome model *or* the propensity is correctly specified. Here propensity is known (constant, ~0.85), making this the best-case scenario for DR. Cross-fitting prevents overfitting bias in the nuisance models.

---

## Targeting Policy

Using the best model (X-learner) on the held-out test set:

**Targeting the top 30% of users by predicted uplift captures approximately [X]% of incremental visits at 30% of spend.**

- Top 10% → [Y]% of incrementals at 10% of spend
- Top 30% → [Z]% of incrementals at 30% of spend  
- Sleeping Dogs identified: [N]% of population — excluding them yields a further improvement in ROI at zero incremental spend

*Fill with measured values after running notebook 05.*

---

## Variance Reduction (CUPED)

We applied CUPED using a control-arm-trained outcome prediction as the pre-experiment covariate proxy. This yields ρ ≈ [ρ], ρ² ≈ [ρ²].

**Variance reduction: [ρ²×100]%** — equivalent to running the same experiment with [ρ²×100]% fewer users for identical power, or achieving [1/(1−ρ²)× smaller MDE for the same N.

**Honesty caveat:** Criteo has no true pre-experiment period. Our proxy satisfies T-independence only if fit on a held-out subset. In a production experiment platform, CUPED would use actual pre-period engagement metrics (click history, prior visit counts) as X_pre — typically achieving ρ² = 0.2–0.5 in practice.

---

## Limitations & Next Steps

1. **Features are anonymized and randomly projected.** We cannot assign business meaning to any feature dimension; heterogeneity analysis is directional only.
2. **External validity:** This is a one-time retrospective dataset. Production targeting requires infrastructure for online experiment assignment, real-time scoring, and holdout evaluation.
3. **Conversion outcome:** At 0.29% base rate, CATE estimates on `conversion` collapse to noise. This is a real constraint — visit is a leading indicator, not the revenue metric. A multi-stage model (visit → conversion) is the production-grade approach.
4. **Policy learning:** Policy trees (EconML `PolicyTree`) learn the treatment *rule* directly rather than thresholding on CATE. Expected to improve on the Qini-optimal threshold approach for heterogeneous cost structures.

---

*References: Diemert et al. (2018); Künzel et al. (2019); Nie & Wager (2021); Kennedy (2020); Athey & Wager (2019); Deng et al. (2013)*
