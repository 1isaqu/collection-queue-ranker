"""Ranking metrics.

Everything here takes a score where bigger means "work this first". Ties are broken by row
position (stable sort), so results are reproducible. Capacity is a fraction of the queue; the
number of rows worked is `max(1, floor(n * fraction))`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def top_count(n: int, fraction: float) -> int:
    return max(1, int(n * fraction)) if n else 0


def order_desc(score: np.ndarray) -> np.ndarray:
    """Row positions from best to worst score (stable for ties)."""
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC, NaN when only one class is present."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def precision_at(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """Share of the worked rows that are positive."""
    y = np.asarray(y)
    k = top_count(len(y), fraction)
    return float(y[order_desc(score)[:k]].mean()) if k else float("nan")


def recall_at(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """Share of all positives found in the worked rows (the gain at that capacity)."""
    y = np.asarray(y)
    k = top_count(len(y), fraction)
    total = y.sum()
    return float(y[order_desc(score)[:k]].sum() / total) if k and total else float("nan")


def lift_at(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """Precision in the worked rows divided by the base rate."""
    y = np.asarray(y)
    base = y.mean() if len(y) else 0.0
    return precision_at(y, score, fraction) / base if base else float("nan")


def gain_table(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative capture by population bin, best scores first."""
    y = np.asarray(y)
    ordered = y[order_desc(score)]
    total = max(int(y.sum()), 1)
    rows, cumulative = [], 0
    for i, idx in enumerate(np.array_split(np.arange(len(y)), n_bins), start=1):
        cumulative += int(ordered[idx].sum())
        share = i / n_bins
        rows.append({
            "bin": i, "population_share": round(share, 4), "positives_captured": cumulative,
            "captured_share": round(cumulative / total, 4),
            "lift": round(cumulative / total / share, 3),
        })
    return pd.DataFrame(rows)


def value_capture(value: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """Share of the total realised value that sits in the worked rows."""
    value = np.asarray(value, dtype=float)
    total = value.sum()
    k = top_count(len(value), fraction)
    return float(value[order_desc(score)[:k]].sum() / total) if k and total > 0 else float("nan")


def ndcg_at(grades: np.ndarray, score: np.ndarray, fraction: float) -> float:
    """NDCG over one queue with exponential gain (2^grade - 1) and a log2 discount."""
    grades = np.asarray(grades, dtype=float)
    k = top_count(len(grades), fraction)
    if not k:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(k) + 2)
    gain = (2.0 ** grades - 1.0)
    got = float((gain[order_desc(score)[:k]] * discount).sum())
    ideal = float((np.sort(gain)[::-1][:k] * discount).sum())
    return got / ideal if ideal > 0 else float("nan")


def cluster_resampler(
    groups: Sequence | np.ndarray | pd.Series, rng: np.random.Generator
) -> Callable[[], np.ndarray]:
    """Return a function drawing one bootstrap sample of row positions, resampling whole groups.

    Debtors appear in several snapshots, so rows are not independent. Resampling debtors (the
    cluster) keeps every row of a drawn debtor together, which gives honest interval widths.
    """
    codes, _ = pd.factorize(pd.Series(np.asarray(groups)))
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    def draw() -> np.ndarray:
        picked = rng.integers(0, len(counts), len(counts))
        lengths = counts[picked]
        offsets = np.cumsum(lengths) - lengths
        within = np.repeat(starts[picked] - offsets, lengths) + np.arange(lengths.sum())
        return order[within]

    return draw


def bootstrap_ci(
    n_rows: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    level: float,
    seed: int,
    groups: Sequence | np.ndarray | pd.Series | None = None,
) -> tuple[float, float]:
    """Percentile interval of `statistic(row_positions)` over bootstrap samples."""
    rng = np.random.default_rng(seed)

    def plain() -> np.ndarray:
        return rng.integers(0, n_rows, n_rows)

    draw = plain if groups is None else cluster_resampler(groups, rng)
    values = [v for v in (statistic(draw()) for _ in range(n_boot)) if np.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def auc_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 200,
    level: float = 0.95,
    seed: int = 0,
    groups: Sequence | np.ndarray | pd.Series | None = None,
) -> tuple[float, float, float]:
    """AUC with a bootstrap percentile interval (an AUC near 0.5 may just be sampling noise)."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    lo, hi = bootstrap_ci(
        len(y), lambda rows: auc(y[rows], score[rows]),
        n_boot=n_boot, level=level, seed=seed, groups=groups,
    )
    return auc(y, score), lo, hi


def discrimination_verdict(value: float, low: float, high: float, chance: float = 0.5) -> str:
    """Read an interval against chance. Without it, 0.53 on a small holdout looks like signal."""
    if not np.isfinite(value) or not np.isfinite(low):
        return "not estimable (one class or too few rows)"
    if low <= chance <= high:
        return "interval covers chance: not distinguishable from noise"
    if value > chance:
        return "interval above chance: weak but real signal"
    return "interval below chance: worse than random, investigate"


def calibration_summary(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Mean predicted vs observed rate. Read it together with discrimination.

    A model that predicts the base rate for everyone is perfectly calibrated and useless for
    ordering a queue.
    """
    y = np.asarray(y, dtype=float)
    return {"mean_predicted": float(np.mean(proba)), "observed_rate": float(y.mean())}
