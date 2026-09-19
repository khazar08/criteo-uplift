"""
Business-impact layer on top of the fitted CATE models.

Nothing here trains a model. Every function consumes three arrays the pipeline
already produces on the held-out test set:

    uplift    — tau_hat(x), predicted incremental P(convert) per user, float
    treatment — observed treatment flag {0, 1}
    outcome   — observed conversion flag {0, 1}

Two questions get answered:

1. *Who* should we treat?  `assign_segments` bins the continuous uplift score
   into the four classical uplift segments, and `budget_report` prices each one.
2. *How many* of them?  `threshold_curve` sweeps the targeting cutoff and prices
   every operating point, so the cutoff becomes a net-value optimum rather than
   a round number someone picked in a meeting.

The outcome column in this repo is `visit` (rate ~4.7%), not `conversion`
(0.29% — too rare for reliable CATE). "Conversion" below means "the modelled
binary outcome"; swap the constants if you re-run on a different label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Campaign economics — the only two numbers a marketer has to supply.
# cost_per_treatment : fully-loaded cost of running the campaign against one user
# value_per_conversion : contribution margin of one *incremental* conversion
# Break-even uplift is their ratio: treating a user pays off iff
#     tau_hat(x) * value_per_conversion > cost_per_treatment
# --------------------------------------------------------------------------
COST_PER_TREATMENT: float = 0.05
VALUE_PER_CONVERSION: float = 5.00

SEGMENT_ORDER: tuple[str, ...] = ("Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog")


@dataclass(frozen=True)
class SegmentPolicy:
    """
    Cutoffs that turn a continuous uplift score into four actionable segments.

    Attributes:
        persuadable_cut: uplift at or above this is a Persuadable (target them).
            Defaults to the break-even uplift implied by campaign economics when
            built via :meth:`from_economics`.
        sleeping_dog_cut: uplift strictly below this is a Sleeping Dog (never
            target). Defaults to 0.0 — treating them is predicted to *reduce*
            conversions, so any negative score is disqualifying.
        sure_thing_base_p: within the neutral band, users whose control-arm
            convert probability is at or above this are Sure Things (they convert
            anyway); the rest are Lost Causes (they convert either way — never).
            Only used when ``base_convert_p`` is supplied.
    """

    persuadable_cut: float = 0.01
    sleeping_dog_cut: float = 0.0
    sure_thing_base_p: float = 0.05

    @classmethod
    def from_economics(
        cls,
        cost_per_treatment: float = COST_PER_TREATMENT,
        value_per_conversion: float = VALUE_PER_CONVERSION,
        sleeping_dog_cut: float = 0.0,
        sure_thing_base_p: float = 0.05,
    ) -> "SegmentPolicy":
        """
        Build a policy whose Persuadable cutoff is the break-even uplift,
        ``cost_per_treatment / value_per_conversion``. Below that line a treated
        user is predicted to cost more than the lift they generate.
        """
        if value_per_conversion <= 0:
            raise ValueError("value_per_conversion must be positive")
        return cls(
            persuadable_cut=cost_per_treatment / value_per_conversion,
            sleeping_dog_cut=sleeping_dog_cut,
            sure_thing_base_p=sure_thing_base_p,
        )


def assign_segments(
    uplift: np.ndarray,
    base_convert_p: np.ndarray | None = None,
    policy: SegmentPolicy = SegmentPolicy(),
) -> np.ndarray:
    """
    Bin a continuous uplift score into the four uplift segments.

    Args:
        uplift: (n,) predicted incremental conversion probability per user.
        base_convert_p: (n,) optional calibrated control-arm P(convert | x), used
            only to split the neutral band into Sure Things and Lost Causes. When
            ``None``, the whole neutral band is labelled Lost Cause — the split is
            not identified from the uplift score alone, so we do not invent it.
        policy: cutoffs; see :class:`SegmentPolicy`.

    Returns:
        (n,) array of segment names drawn from :data:`SEGMENT_ORDER`.
    """
    uplift = np.asarray(uplift, dtype=float).ravel()
    segments = np.full(uplift.shape, "Lost Cause", dtype=object)

    segments[uplift >= policy.persuadable_cut] = "Persuadable"
    segments[uplift < policy.sleeping_dog_cut] = "Sleeping Dog"

    if base_convert_p is not None:
        base_convert_p = np.asarray(base_convert_p, dtype=float).ravel()
        if base_convert_p.shape != uplift.shape:
            raise ValueError("base_convert_p must have the same shape as uplift")
        neutral = (uplift >= policy.sleeping_dog_cut) & (uplift < policy.persuadable_cut)
        segments[neutral & (base_convert_p >= policy.sure_thing_base_p)] = "Sure Thing"

    return segments.astype(str)


def budget_report(
    uplift: np.ndarray,
    segments: np.ndarray,
    cost_per_treatment: float = COST_PER_TREATMENT,
    value_per_conversion: float = VALUE_PER_CONVERSION,
) -> pd.DataFrame:
    """
    Price each segment: who they are, what treating them costs, what it returns.

    Incremental conversions are the *model's* claim — the sum of predicted uplift
    over the segment's users. That is the right quantity for a forward-looking
    budget: it is what the model expects to buy if you treat that segment.
    (`threshold_curve` cross-checks it against observed treated/control outcomes.)

    Args:
        uplift: (n,) predicted incremental conversion probability per user.
        segments: (n,) segment labels from :func:`assign_segments`.
        cost_per_treatment: cost of treating one user.
        value_per_conversion: margin of one incremental conversion.

    Returns:
        One row per segment (in :data:`SEGMENT_ORDER`) with columns
        ``segment, n_users, share_pct, incremental_conversions, spend, value, net``.
        ``df.attrs`` carries the blanket-vs-Persuadable comparison:
        ``blanket_spend, persuadable_spend, budget_saved_pct, blanket_net,
        persuadable_net, blanket_incremental, persuadable_incremental,
        incremental_retained_pct``.
    """
    uplift = np.asarray(uplift, dtype=float).ravel()
    segments = np.asarray(segments).ravel()
    if segments.shape != uplift.shape:
        raise ValueError("segments must have the same shape as uplift")

    n_total = uplift.size
    rows = []
    for name in SEGMENT_ORDER:
        mask = segments == name
        n_users = int(mask.sum())
        incremental = float(uplift[mask].sum())
        spend = n_users * cost_per_treatment
        value = incremental * value_per_conversion
        rows.append(
            {
                "segment": name,
                "n_users": n_users,
                "share_pct": 100.0 * n_users / n_total if n_total else 0.0,
                "incremental_conversions": incremental,
                "spend": spend,
                "value": value,
                "net": value - spend,
            }
        )

    df = pd.DataFrame(rows)

    persuadable = df.loc[df["segment"] == "Persuadable"].iloc[0]
    blanket_spend = n_total * cost_per_treatment
    blanket_incremental = float(uplift.sum())
    blanket_net = blanket_incremental * value_per_conversion - blanket_spend

    df.attrs.update(
        {
            "cost_per_treatment": cost_per_treatment,
            "value_per_conversion": value_per_conversion,
            "n_users_total": n_total,
            "blanket_spend": blanket_spend,
            "persuadable_spend": float(persuadable["spend"]),
            "budget_saved_pct": (
                100.0 * (1.0 - persuadable["spend"] / blanket_spend) if blanket_spend else 0.0
            ),
            "blanket_incremental": blanket_incremental,
            "persuadable_incremental": float(persuadable["incremental_conversions"]),
            "incremental_retained_pct": (
                100.0 * persuadable["incremental_conversions"] / blanket_incremental
                if blanket_incremental
                else 0.0
            ),
            "blanket_net": blanket_net,
            "persuadable_net": float(persuadable["net"]),
        }
    )
    return df


def threshold_curve(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    cost_per_treatment: float = COST_PER_TREATMENT,
    value_per_conversion: float = VALUE_PER_CONVERSION,
    n_points: int = 101,
) -> pd.DataFrame:
    """
    Sweep the targeting cutoff from 0% to 100% of the population and price each point.

    Users are sorted by descending predicted uplift. Incremental conversions at
    cut ``k`` come from the Qini cumulative-gain estimator, which needs no
    per-user counterfactual — only the observed arms inside the top-k:

        inc(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)

    where ``Y_t``/``Y_c`` are cumulative conversions and ``N_t``/``N_c`` cumulative
    user counts in the treated/control arms. Rescaling the control arm by
    ``N_t/N_c`` makes the two arms comparable at every cut.

    Args:
        uplift: (n,) predicted incremental conversion probability per user.
        treatment: (n,) observed treatment flag {0, 1}.
        outcome: (n,) observed conversion flag {0, 1}.
        cost_per_treatment: cost of treating one user.
        value_per_conversion: margin of one incremental conversion.
        n_points: number of cutoffs on the 0%-100% grid, endpoints included.

    Returns:
        DataFrame with columns ``target_fraction, users_targeted,
        incremental_conversions, pct_of_total_incremental, spend, net_value``.
        ``df.attrs`` carries ``peak_incremental``, ``peak_fraction``,
        ``endpoint_incremental`` and the optimum (``optimal_fraction``,
        ``optimal_users``, ``optimal_net_value``).
    """
    uplift = np.asarray(uplift, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    outcome = np.asarray(outcome, dtype=float).ravel()
    if not (uplift.shape == treatment.shape == outcome.shape):
        raise ValueError("uplift, treatment and outcome must have the same shape")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    order = np.argsort(uplift)[::-1]
    t_s, y_s = treatment[order], outcome[order]
    n = uplift.size

    # Cumulative arms, prefixed with a zero so index k reads "top-k users".
    y_t_cum = np.concatenate([[0.0], np.cumsum(y_s * t_s)])
    y_c_cum = np.concatenate([[0.0], np.cumsum(y_s * (1.0 - t_s))])
    n_t_cum = np.concatenate([[0.0], np.cumsum(t_s)])
    n_c_cum = np.concatenate([[0.0], np.cumsum(1.0 - t_s)])

    ks = np.unique(np.round(np.linspace(0.0, 1.0, n_points) * n).astype(int))
    safe_nc = np.where(n_c_cum[ks] > 0, n_c_cum[ks], 1.0)
    incremental = y_t_cum[ks] - y_c_cum[ks] * (n_t_cum[ks] / safe_nc)
    incremental = np.where(n_c_cum[ks] > 0, incremental, 0.0)

    # Normalize against the PEAK of the Qini curve, not its k=N endpoint: with
    # Sleeping Dogs in the population the curve rises above where it ends, so the
    # endpoint would score the best cutoff at >100% of "total" incremental.
    # (For classic full-population normalization, divide by incremental[-1] instead.)
    peak = float(incremental.max())
    pct = 100.0 * incremental / peak if peak > 0 else np.zeros_like(incremental)

    users = ks.astype(float)
    spend = users * cost_per_treatment
    net = incremental * value_per_conversion - spend

    df = pd.DataFrame(
        {
            "target_fraction": users / n,
            "users_targeted": ks,
            "incremental_conversions": incremental,
            "pct_of_total_incremental": pct,
            "spend": spend,
            "net_value": net,
        }
    )

    best = int(np.argmax(net))
    df.attrs.update(
        {
            "cost_per_treatment": cost_per_treatment,
            "value_per_conversion": value_per_conversion,
            "n_users_total": n,
            "peak_incremental": peak,
            "peak_fraction": float(df["target_fraction"].iloc[int(np.argmax(incremental))]),
            "endpoint_incremental": float(incremental[-1]),
            "optimal_fraction": float(df["target_fraction"].iloc[best]),
            "optimal_users": int(ks[best]),
            "optimal_net_value": float(net[best]),
        }
    )
    return df


def summarize_thresholds(
    curve: pd.DataFrame,
    captures: Sequence[float] = (0.8, 0.85, 0.9),
) -> list[str]:
    """
    Turn a threshold curve into the sentences that go in a deck.

    Args:
        curve: output of :func:`threshold_curve`.
        captures: capture levels to report, as fractions of peak incremental.

    Returns:
        One line per capture level ("top X% captures Y% of incremental
        conversions ..."), plus a final line for the net-value optimum.
    """
    lines: list[str] = []
    for capture in captures:
        target_pct = 100.0 * capture
        hit = curve.loc[curve["pct_of_total_incremental"] >= target_pct]
        if hit.empty:
            lines.append(
                f"Top 100% never reaches {target_pct:.0f}% of incremental conversions."
            )
            continue
        row = hit.iloc[0]
        lines.append(
            f"Top {100 * row['target_fraction']:.0f}% by predicted uplift "
            f"({int(row['users_targeted']):,} users) captures "
            f"{row['pct_of_total_incremental']:.1f}% of incremental conversions "
            f"at {100 * row['target_fraction']:.0f}% of blanket spend."
        )

    opt_frac = curve.attrs["optimal_fraction"]
    opt_row = curve.loc[curve["target_fraction"] == opt_frac].iloc[0]
    lines.append(
        f"Net-value optimum: target the top {100 * opt_frac:.0f}% "
        f"({curve.attrs['optimal_users']:,} users) for net "
        f"${curve.attrs['optimal_net_value']:,.0f}, capturing "
        f"{opt_row['pct_of_total_incremental']:.1f}% of incremental conversions."
    )
    return lines


def plot_threshold_curve(curve: pd.DataFrame, path: str) -> str:
    """
    Dual-axis chart: % incremental captured (left) and net value (right) against
    % of population targeted, with a marker at the net-value optimum.

    Args:
        curve: output of :func:`threshold_curve`.
        path: PNG destination; parent directory must exist.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = 100.0 * curve["target_fraction"].to_numpy()
    fig, ax_left = plt.subplots(figsize=(9, 5.5))

    ax_left.plot(x, curve["pct_of_total_incremental"], color="#2d6a9f", lw=2.2,
                 label="% incremental conversions captured")
    ax_left.set_xlabel("Population targeted (%, ranked by predicted uplift)")
    ax_left.set_ylabel("Incremental conversions captured (% of peak)", color="#2d6a9f")
    ax_left.tick_params(axis="y", labelcolor="#2d6a9f")
    ax_left.set_xlim(0, 100)
    ax_left.set_ylim(0, 105)
    ax_left.grid(alpha=0.25, linestyle=":")
    ax_left.plot([0, 100], [0, 100], color="#9aa0a6", lw=1.2, ls="--",
                 label="Random targeting")

    ax_right = ax_left.twinx()
    ax_right.plot(x, curve["net_value"], color="#c2513a", lw=2.2, label="Net value ($)")
    ax_right.set_ylabel("Net value ($)", color="#c2513a")
    ax_right.tick_params(axis="y", labelcolor="#c2513a")
    ax_right.axhline(0.0, color="#c2513a", lw=0.8, ls=":", alpha=0.6)

    opt_x = 100.0 * curve.attrs["optimal_fraction"]
    opt_y = curve.attrs["optimal_net_value"]
    ax_right.plot([opt_x], [opt_y], marker="o", ls="none", ms=9, color="#c2513a",
                  mec="white", mew=1.5, zorder=5, label="Net-value optimum")
    ax_right.annotate(
        f"optimum: top {opt_x:.0f}%\nnet ${opt_y:,.0f}",
        xy=(opt_x, opt_y),
        xytext=(min(opt_x + 12, 62), opt_y),
        color="#c2513a",
        fontsize=9,
        va="center",
        arrowprops=dict(arrowstyle="->", color="#c2513a", lw=1.0),
    )

    handles = ax_left.get_lines()[:2] + [ax_right.get_lines()[0], ax_right.get_lines()[-1]]
    ax_left.legend(handles, [h.get_label() for h in handles], loc="lower right", fontsize=9)
    ax_left.set_title("Cost-benefit of the targeting threshold (X-learner, Criteo test set)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(df: pd.DataFrame) -> str:
    """Compact console rendering of a report DataFrame."""
    return df.to_string(index=False, float_format=lambda v: f"{v:,.2f}")


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    PRED_PATH = ROOT / "data" / "meta_learner_predictions.pkl"
    CHART_PATH = ROOT / "reports" / "threshold_curve.png"

    with open(PRED_PATH, "rb") as fh:
        blob = pickle.load(fh)

    # The three arrays the X-learner pipeline already emits on the test set.
    uplift = np.asarray(blob["predictions"]["X-learner"], dtype=float)
    treatment = np.asarray(blob["t_te"], dtype=float)
    outcome = np.asarray(blob["y_te"], dtype=float)

    print("=== Inputs (held-out test set) ===")
    print(f"uplift     shape={uplift.shape} dtype={uplift.dtype} "
          f"range=[{uplift.min():.4f}, {uplift.max():.4f}] mean={uplift.mean():.5f}")
    print(f"treatment  shape={treatment.shape} dtype={treatment.dtype} "
          f"treated_share={treatment.mean():.4f}")
    print(f"outcome    shape={outcome.shape} dtype={outcome.dtype} "
          f"convert_rate={outcome.mean():.5f}")
    print(f"economics  cost_per_treatment=${COST_PER_TREATMENT:.2f}  "
          f"value_per_conversion=${VALUE_PER_CONVERSION:.2f}  "
          f"break-even uplift={COST_PER_TREATMENT / VALUE_PER_CONVERSION:.4f}")

    policy = SegmentPolicy.from_economics(COST_PER_TREATMENT, VALUE_PER_CONVERSION)
    # The X-learner exposes tau_hat only — no calibrated control-arm P(convert) is
    # persisted, so the Sure Thing / Lost Cause split is not available here and the
    # whole neutral band lands in Lost Cause. See README for the limitation.
    segments = assign_segments(uplift, base_convert_p=None, policy=policy)

    report = budget_report(uplift, segments, COST_PER_TREATMENT, VALUE_PER_CONVERSION)
    print("\n=== Four-quadrant budget report ===")
    print(_fmt(report))
    a = report.attrs
    print(
        f"\nBlanket spend ${a['blanket_spend']:,.0f} -> Persuadable-only "
        f"${a['persuadable_spend']:,.0f} ({a['budget_saved_pct']:.1f}% budget saved). "
        f"Predicted incremental conversions {a['blanket_incremental']:,.0f} -> "
        f"{a['persuadable_incremental']:,.0f} ({a['incremental_retained_pct']:.1f}% "
        f"of blanket -- above 100% because Sleeping Dogs subtract from the blanket "
        f"total). Net: ${a['blanket_net']:,.0f} -> ${a['persuadable_net']:,.0f}."
    )

    curve = threshold_curve(
        uplift, treatment, outcome, COST_PER_TREATMENT, VALUE_PER_CONVERSION, n_points=101
    )
    print("\n=== Threshold curve (every 10th point) ===")
    print(_fmt(curve.iloc[::10]))
    print(
        f"\nQini peak {curve.attrs['peak_incremental']:,.1f} incremental conversions at "
        f"{100 * curve.attrs['peak_fraction']:.0f}% targeted; "
        f"endpoint (k=N) {curve.attrs['endpoint_incremental']:,.1f}."
    )

    print("\n=== Threshold summary ===")
    for line in summarize_thresholds(curve):
        print(" -", line)

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nChart -> {plot_threshold_curve(curve, str(CHART_PATH))}")
