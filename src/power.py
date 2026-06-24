"""
Sample-size and MDE calculator for two-proportion tests.
Includes CUPED variance reduction impact on required sample size.
"""

import numpy as np
from scipy import stats


def sample_size(
    p_base: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Per-arm sample size for a two-proportion z-test.

    n = (z_{1-alpha/2} + z_{1-beta})^2 * [p1(1-p1) + p0(1-p0)] / (p1-p0)^2
    """
    p1 = p_base + mde
    p0 = p_base
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p0 * (1 - p0))
    denominator = (p1 - p0) ** 2
    return int(np.ceil(numerator / denominator))


def mde(
    n: int,
    p_base: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """
    Minimum detectable effect given per-arm sample size n.
    Solved by binary search over the sample_size formula.
    """
    lo, hi = 1e-6, 1 - p_base
    for _ in range(60):
        mid = (lo + hi) / 2
        if sample_size(p_base, mid, alpha, power, two_sided) <= n:
            hi = mid
        else:
            lo = mid
    return float(hi)


def cuped_sample_size(
    p_base: float,
    mde_target: float,
    rho_sq: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Show how CUPED's rho^2 reduces the required sample size.
    CUPED reduces variance by (1 - rho^2), which shrinks required n by the same factor.
    """
    n_raw = sample_size(p_base, mde_target, alpha, power)
    n_cuped = int(np.ceil(n_raw * (1 - rho_sq)))
    return {
        "n_raw": n_raw,
        "n_cuped": n_cuped,
        "reduction_pct": float((1 - n_cuped / n_raw) * 100),
        "rho_sq": rho_sq,
    }


def power_table(
    p_base: float,
    mde_values: list,
    alpha: float = 0.05,
    power: float = 0.80,
) -> list:
    """Return a list of {mde, n_per_arm, total_n} dicts."""
    rows = []
    for d in mde_values:
        n = sample_size(p_base, d, alpha, power)
        rows.append({"mde": d, "n_per_arm": n, "total_n": 2 * n})
    return rows


def summarize(p_base: float = 0.047, rho_sq: float = 0.10):
    """Print a human-readable power summary using Criteo's real base rates."""
    print(f"Base rate (visit): {p_base:.3%}")
    print(f"ATE in Criteo ≈ 0.47pp absolute lift above {p_base:.3%}")
    print()
    print("Sample size for various MDEs (alpha=0.05, power=0.80):")
    print(f"{'MDE':>8}  {'n/arm':>10}  {'total n':>12}")
    for d in [0.001, 0.002, 0.003, 0.005, 0.010]:
        n = sample_size(p_base, d, 0.05, 0.80)
        print(f"{d:>8.3f}  {n:>10,}  {2*n:>12,}")
    print()
    print(f"CUPED variance reduction (rho^2={rho_sq:.2f}):")
    r = cuped_sample_size(p_base, 0.003, rho_sq)
    print(f"  Without CUPED: {r['n_raw']:,}/arm")
    print(f"  With CUPED:    {r['n_cuped']:,}/arm  ({r['reduction_pct']:.1f}% fewer users)")


if __name__ == "__main__":
    summarize()
