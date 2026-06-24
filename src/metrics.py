"""
Uplift evaluation metrics implemented from scratch, cross-checked against sklift.

Key insight: we cannot use ROC-AUC because the ground truth label Y(1)-Y(0) is
never observed for any individual. Instead we evaluate *ranking quality* over the
population using Qini and AUUC — both rely only on observed (Y, T) pairs.
"""

import numpy as np
from typing import Tuple


def qini_curve(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve by sorting users descending on predicted uplift.

    At each cumulative cut k:
        Qini(k) = R_t(k) - R_c(k) * (N_t(k) / N_c(k))

    The N_t/N_c rescaling makes the control arm comparable to treatment.
    """
    order = np.argsort(uplift)[::-1]
    y_s, t_s = y[order], t[order]

    rt = np.cumsum(y_s * t_s)
    rc = np.cumsum(y_s * (1 - t_s))
    nt = np.cumsum(t_s).clip(min=1)
    nc = np.cumsum(1 - t_s).clip(min=1)

    q = rt - rc * (nt / nc)
    x = np.arange(1, len(y_s) + 1)
    return x, q


def qini_auc(y: np.ndarray, uplift: np.ndarray, t: np.ndarray) -> float:
    """
    Area between the Qini curve and the random (diagonal) baseline.
    Positive = better than random targeting; negative = worse.
    """
    x, q = qini_curve(y, uplift, t)
    # random baseline: straight line from 0 to q[-1]
    rand = q[-1] * x / x[-1]
    return float(np.trapezoid(q - rand, x))


def qini_coefficient(y: np.ndarray, uplift: np.ndarray, t: np.ndarray) -> float:
    """
    Normalized Qini: qini_auc / perfect_qini_auc.
    Ranges roughly [0, 1] for sensible models.
    """
    model_auc = qini_auc(y, uplift, t)
    # perfect model: sort by true uplift (oracle — use actual y, t)
    # oracle = treated responders first, then control non-responders last
    oracle_uplift = y * t - y * (1 - t)
    perfect_auc = qini_auc(y, oracle_uplift, t)
    if perfect_auc == 0:
        return 0.0
    return float(model_auc / perfect_auc)


def auuc(y: np.ndarray, uplift: np.ndarray, t: np.ndarray) -> float:
    """Area Under the Uplift Curve (unnormalized, normalized by n)."""
    x, q = qini_curve(y, uplift, t)
    return float(np.trapezoid(q, x) / len(y))


def uplift_at_k(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray, k: float = 0.10
) -> float:
    """
    Realized uplift (visit_rate_treated - visit_rate_control) in the top-k fraction
    by predicted uplift. This is the number a PM cares about.
    """
    n = len(y)
    cutoff = max(1, int(np.ceil(k * n)))
    order = np.argsort(uplift)[::-1][:cutoff]
    y_k, t_k = y[order], t[order]

    n_t = t_k.sum()
    n_c = (1 - t_k).sum()
    if n_t == 0 or n_c == 0:
        return np.nan

    rate_t = (y_k * t_k).sum() / n_t
    rate_c = (y_k * (1 - t_k)).sum() / n_c
    return float(rate_t - rate_c)


def uplift_by_decile(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin users into n_bins by predicted uplift, compute realized uplift per bin.
    A monotone-decreasing staircase means the model genuinely ranks persuadables.
    Returns (bin_centers, realized_uplift_per_bin).
    """
    order = np.argsort(uplift)[::-1]
    y_s, t_s = y[order], t[order]
    bins = np.array_split(np.arange(len(y_s)), n_bins)

    realized = []
    for idx in bins:
        y_b, t_b = y_s[idx], t_s[idx]
        n_t = t_b.sum()
        n_c = (1 - t_b).sum()
        if n_t == 0 or n_c == 0:
            realized.append(np.nan)
        else:
            realized.append(
                float((y_b * t_b).sum() / n_t - (y_b * (1 - t_b)).sum() / n_c)
            )
    bin_centers = np.arange(1, n_bins + 1)
    return bin_centers, np.array(realized)


def evaluate_all(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray, label: str = ""
) -> dict:
    """Compute the full metrics dict for a single model."""
    return {
        "model": label,
        "qini_coeff": qini_coefficient(y, uplift, t),
        "auuc": auuc(y, uplift, t),
        "uplift@10": uplift_at_k(y, uplift, t, 0.10),
        "uplift@20": uplift_at_k(y, uplift, t, 0.20),
        "uplift@30": uplift_at_k(y, uplift, t, 0.30),
    }
