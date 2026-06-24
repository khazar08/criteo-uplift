"""
CUPED — Controlled-experiment Using Pre-Experiment Data (Deng et al. 2013).

Reduces ATE estimate variance by regressing out a covariate X_pre that is:
  (a) correlated with the outcome Y
  (b) independent of treatment T

Y_cuped = Y - theta * (X_pre - E[X_pre])
theta    = Cov(Y, X_pre) / Var(X_pre)     [OLS coefficient]

Variance reduction ≈ rho^2   where rho = Corr(Y, X_pre).

Honesty caveat (stated in writeup): Criteo has no true pre-experiment period.
We use a control-arm-trained outcome prediction as a proxy for X_pre. This
satisfies (b) if we compute it on a held-out split. The limitation is noted.
"""

import numpy as np
import pandas as pd
from scipy import stats


def theta_estimate(y: np.ndarray, x_pre: np.ndarray) -> float:
    """OLS coefficient: Cov(Y, X_pre) / Var(X_pre)."""
    return float(np.cov(y, x_pre)[0, 1] / np.var(x_pre))


def apply_cuped(
    y: np.ndarray, x_pre: np.ndarray, theta: float | None = None
) -> np.ndarray:
    """Return CUPED-adjusted outcome."""
    if theta is None:
        theta = theta_estimate(y, x_pre)
    return y - theta * (x_pre - x_pre.mean())


def variance_reduction(y: np.ndarray, x_pre: np.ndarray) -> dict:
    """
    Compute and compare ATE variance before and after CUPED.
    Returns variance reduction fraction and rho^2.
    """
    rho = float(np.corrcoef(y, x_pre)[0, 1])
    theta = theta_estimate(y, x_pre)
    y_cuped = apply_cuped(y, x_pre, theta)

    return {
        "rho": rho,
        "rho_squared": rho ** 2,
        "var_before": float(np.var(y)),
        "var_after": float(np.var(y_cuped)),
        "variance_reduction_pct": float(rho ** 2 * 100),
        "theta": theta,
    }


def cuped_ate(
    y: np.ndarray,
    t: np.ndarray,
    x_pre: np.ndarray,
) -> dict:
    """
    Compute ATE and 95% CI before and after CUPED adjustment.
    Shows concretely how CUPED shrinks the confidence interval.
    """
    theta = theta_estimate(y, x_pre)
    y_c = apply_cuped(y, x_pre, theta)

    def _ate_ci(outcome, treatment):
        y1 = outcome[treatment == 1]
        y0 = outcome[treatment == 0]
        ate = y1.mean() - y0.mean()
        se = np.sqrt(y1.var() / len(y1) + y0.var() / len(y0))
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se
        return {"ate": float(ate), "se": float(se), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}

    raw = _ate_ci(y, t)
    adjusted = _ate_ci(y_c, t)
    vr = variance_reduction(y, x_pre)

    return {
        "raw": raw,
        "cuped": adjusted,
        "rho": vr["rho"],
        "rho_squared": vr["rho_squared"],
        "variance_reduction_pct": vr["variance_reduction_pct"],
        "ci_width_reduction_pct": float(
            (1 - (adjusted["ci_hi"] - adjusted["ci_lo"]) / (raw["ci_hi"] - raw["ci_lo"])) * 100
        ),
    }
