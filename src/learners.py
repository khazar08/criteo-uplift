"""
Meta-learner wrappers: S / T / X / R / DR.

Treatment imbalance (~85% treated) is the narrative thread:
  - S-learner: regularization shrinks T coefficient toward 0 — biased in small-effect regime
  - T-learner: control arm is small (15%) → noisy mu0
  - X-learner: imputes TEs from the large arm to fix T-learner's noisy mu0;
               propensity weighting then puts 85% weight on tau0 (estimated from
               the large treated arm via imputed D0) — the correct correction
  - R/DR: cross-fitted nuisances; DR is doubly robust (known propensity here = best case)
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier, LGBMRegressor

_DEFAULT_CLF = lambda: LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    n_jobs=-1, verbose=-1, random_state=42
)
_DEFAULT_REG = lambda: LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    n_jobs=-1, verbose=-1, random_state=42
)


class SLearner:
    """Single model with T as a feature. Fast but shrinks T toward 0."""

    def __init__(self):
        self.model = _DEFAULT_CLF()

    def fit(self, X, y, t):
        Xt = np.column_stack([X, t])
        self.model.fit(Xt, y)
        return self

    def predict(self, X):
        n = len(X)
        X1 = np.column_stack([X, np.ones(n)])
        X0 = np.column_stack([X, np.zeros(n)])
        return (
            self.model.predict_proba(X1)[:, 1]
            - self.model.predict_proba(X0)[:, 1]
        )


class TLearner:
    """Two separate outcome models. Noisy in the small (control) arm."""

    def __init__(self):
        self.mu1 = _DEFAULT_CLF()
        self.mu0 = _DEFAULT_CLF()

    def fit(self, X, y, t):
        self.mu1.fit(X[t == 1], y[t == 1])
        self.mu0.fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.mu1.predict_proba(X)[:, 1] - self.mu0.predict_proba(X)[:, 1]


class XLearner:
    """
    X-learner (Künzel et al. 2019).

    Step 1: fit mu0, mu1 as in T-learner.
    Step 2: impute individual TEs:
        D1 = Y1 - mu0(X1)   [treated: what *would* they have done without ad?]
        D0 = mu1(X0) - Y0   [control: what *would* they have done with ad?]
    Step 3: fit tau1 on D1, tau0 on D0.
    Step 4: combine with propensity:
        tau(x) = e(x)*tau0(x) + (1-e(x))*tau1(x)
        With e≈0.85: weights tau0 heavily — tau0 is estimated from the
        *large treated arm via D0 imputation*, so this is the variance win.
    """

    def __init__(self, propensity: float = 0.85):
        self.e = propensity
        self.mu0 = _DEFAULT_CLF()
        self.mu1 = _DEFAULT_CLF()
        self.tau0 = _DEFAULT_REG()
        self.tau1 = _DEFAULT_REG()

    def fit(self, X, y, t):
        mask1, mask0 = t == 1, t == 0
        X1, y1 = X[mask1], y[mask1]
        X0, y0 = X[mask0], y[mask0]

        self.mu0.fit(X0, y0)
        self.mu1.fit(X1, y1)

        D1 = y1 - self.mu0.predict_proba(X1)[:, 1]
        D0 = self.mu1.predict_proba(X0)[:, 1] - y0

        self.tau1.fit(X1, D1)
        self.tau0.fit(X0, D0)
        return self

    def predict(self, X):
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


class RLearner:
    """
    R-learner (Nie & Wager 2021) with cross-fitting.

    Robinson decomposition: residualize both Y and T, then fit tau on residuals.
    Cross-fitting prevents overfitting bias in the nuisance models m(x), e(x).

    Minimizes: sum[ (Yi - m(xi)) - (Ti - e(xi)) * tau(xi) ]^2
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.tau = _DEFAULT_REG()

    def fit(self, X, y, t):
        n = len(y)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            m_fold = _DEFAULT_CLF()
            e_fold = _DEFAULT_CLF()
            m_fold.fit(X[train_idx], y[train_idx])
            e_fold.fit(X[train_idx], t[train_idx])
            m_hat[val_idx] = m_fold.predict_proba(X[val_idx])[:, 1]
            e_hat[val_idx] = e_fold.predict_proba(X[val_idx])[:, 1]

        r_y = y - m_hat
        r_t = t - e_hat
        # pseudo-outcome: r_y / r_t; weight by r_t^2
        weights = r_t ** 2
        # guard against near-zero r_t
        safe_rt = np.where(np.abs(r_t) > 1e-6, r_t, 1e-6)
        pseudo_y = r_y / safe_rt

        self.tau.fit(X, pseudo_y, sample_weight=weights)
        return self

    def predict(self, X):
        return self.tau.predict(X)


class DRLearner:
    """
    Doubly-robust / AIPW learner (Kennedy 2020) with cross-fitting.

    AIPW pseudo-outcome:
        psi_i = mu1(x) - mu0(x)
              + T*(Y - mu1(x)) / e(x)
              - (1-T)*(Y - mu0(x)) / (1-e(x))

    Doubly robust: consistent if *either* outcome model OR propensity is right.
    Known constant propensity here = best-case scenario for DR; note this explicitly.
    """

    def __init__(self, n_folds: int = 5, propensity: float | None = None):
        self.n_folds = n_folds
        self.propensity = propensity  # None = estimate from data
        self.tau = _DEFAULT_REG()

    def fit(self, X, y, t):
        n = len(y)
        mu1_hat = np.zeros(n)
        mu0_hat = np.zeros(n)
        e_hat = np.full(n, self.propensity if self.propensity else 0.85)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            mask1 = t[train_idx] == 1
            mask0 = t[train_idx] == 0

            m1 = _DEFAULT_CLF()
            m0 = _DEFAULT_CLF()
            m1.fit(X[train_idx][mask1], y[train_idx][mask1])
            m0.fit(X[train_idx][mask0], y[train_idx][mask0])
            mu1_hat[val_idx] = m1.predict_proba(X[val_idx])[:, 1]
            mu0_hat[val_idx] = m0.predict_proba(X[val_idx])[:, 1]

            if self.propensity is None:
                ep = _DEFAULT_CLF()
                ep.fit(X[train_idx], t[train_idx])
                e_hat[val_idx] = ep.predict_proba(X[val_idx])[:, 1]

        e_clip = np.clip(e_hat, 0.01, 0.99)
        psi = (
            mu1_hat - mu0_hat
            + t * (y - mu1_hat) / e_clip
            - (1 - t) * (y - mu0_hat) / (1 - e_clip)
        )
        self.tau.fit(X, psi)
        return self

    def predict(self, X):
        return self.tau.predict(X)


# convenience registry for iteration
LEARNERS = {
    "S-learner": SLearner,
    "T-learner": TLearner,
    "X-learner": XLearner,
    "R-learner": RLearner,
    "DR-learner": DRLearner,
}
