import numpy as np
from typing import Tuple


def qini_curve(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    x, q = qini_curve(y, uplift, t)
    # random baseline: straight line from 0 to q[-1]
    rand = q[-1] * x / x[-1]
    return float(np.trapezoid(q - rand, x))


def qini_coefficient(y: np.ndarray, uplift: np.ndarray, t: np.ndarray) -> float:
    model_auc = qini_auc(y, uplift, t)
    # perfect model: sort by true uplift (oracle — use actual y, t)
    # oracle = treated responders first, then control non-responders last
    oracle_uplift = y * t - y * (1 - t)
    perfect_auc = qini_auc(y, oracle_uplift, t)
    if perfect_auc == 0:
        return 0.0
    return float(model_auc / perfect_auc)


def auuc(y: np.ndarray, uplift: np.ndarray, t: np.ndarray) -> float:
    x, q = qini_curve(y, uplift, t)
    return float(np.trapezoid(q, x) / len(y))


def uplift_at_k(
    y: np.ndarray, uplift: np.ndarray, t: np.ndarray, k: float = 0.10
) -> float:
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
