"""
Targeting policy: turn tau_hat(x) into a business decision.

Core output: "targeting the top X% by predicted uplift captures Y% of incremental
visits at X% of spend" — the exact sentence an experimentation interviewer wants.
"""

import numpy as np
from typing import Tuple


def targeting_curve(
    y: np.ndarray,
    uplift: np.ndarray,
    t: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep the targeting threshold from 0% to 100% of population.

    Returns:
        fractions  — fraction of population treated (spend proxy)
        incremental — cumulative incremental visits captured
        total_incremental — total incremental visits in dataset (denominator)
    """
    order = np.argsort(uplift)[::-1]
    y_s, t_s, u_s = y[order], t[order], uplift[order]

    n = len(y_s)
    fractions = np.linspace(0, 1, n_points + 1)[1:]

    # total incremental = ATE * n (proxy via treated/control difference)
    n_t_total = t_s.sum()
    n_c_total = (1 - t_s).sum()
    total_rate_t = (y_s * t_s).sum() / n_t_total
    total_rate_c = (y_s * (1 - t_s)).sum() / n_c_total
    total_incremental = (total_rate_t - total_rate_c) * n

    incremental_captured = []
    for frac in fractions:
        k = max(1, int(np.ceil(frac * n)))
        subset = slice(0, k)
        y_k, t_k = y_s[subset], t_s[subset]
        nt_k = t_k.sum()
        nc_k = (1 - t_k).sum()
        if nt_k == 0 or nc_k == 0:
            incremental_captured.append(0.0)
            continue
        rate_t_k = (y_k * t_k).sum() / nt_k
        rate_c_k = (y_k * (1 - t_k)).sum() / nc_k
        incr_k = (rate_t_k - rate_c_k) * k
        incremental_captured.append(float(incr_k))

    incremental_captured = np.array(incremental_captured)
    pct_captured = incremental_captured / total_incremental if total_incremental > 0 else incremental_captured

    return fractions, pct_captured, float(total_incremental)


def sleeping_dogs_analysis(uplift: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    """
    Identify Sleeping Dogs: users with negative predicted uplift.
    Show the incremental gain from excluding them from targeting.
    """
    negative_mask = uplift < 0
    n_sleeping = negative_mask.sum()
    pct_sleeping = float(n_sleeping / len(uplift) * 100)

    # uplift among sleeping dogs in treated vs control
    y_sd, t_sd = y[negative_mask], t[negative_mask]
    nt = t_sd.sum()
    nc = (1 - t_sd).sum()
    if nt > 0 and nc > 0:
        realized_sd = float(
            (y_sd * t_sd).sum() / nt - (y_sd * (1 - t_sd)).sum() / nc
        )
    else:
        realized_sd = np.nan

    return {
        "n_sleeping_dogs": int(n_sleeping),
        "pct_sleeping_dogs": pct_sleeping,
        "mean_predicted_uplift": float(uplift[negative_mask].mean()) if n_sleeping > 0 else np.nan,
        "realized_uplift_sleeping_dogs": realized_sd,
    }


def policy_summary(
    y: np.ndarray,
    uplift: np.ndarray,
    t: np.ndarray,
    thresholds: list = [0.10, 0.20, 0.30, 0.50],
) -> list:
    """
    For each spend threshold, report incremental captures and sleeping dog exclusions.
    """
    fractions, pct_captured, _ = targeting_curve(y, uplift, t)
    rows = []
    for thr in thresholds:
        idx = np.searchsorted(fractions, thr)
        idx = min(idx, len(pct_captured) - 1)
        rows.append({
            "spend_fraction": thr,
            "pct_incremental_captured": float(pct_captured[idx] * 100),
        })
    return rows
