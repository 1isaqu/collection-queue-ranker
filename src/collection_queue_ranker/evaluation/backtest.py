"""Monetised backtest under limited capacity.

The question a collections manager asks is not "what is the AUC" but "if the team can only work the
top X% of the queue, how much money lands in that X% under each ordering?". The backtest answers it
on holdout snapshots:

* every snapshot is **one queue**: capacity is applied inside each queue, then summed, because the
  team works one queue per period, not one giant pooled list;
* the judge is the value actually received inside the label window, which does not depend on how any
  model defines its target;
* strategies are compared against two baselines, "largest balance first" (what a manual operation
  tends to do) and random order (the floor);
* the uplift over the balance baseline gets a **debtor-level cluster bootstrap** interval, because a
  debtor can appear in several snapshots and rows are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collection_queue_ranker.config import EvaluationSettings
from collection_queue_ranker.evaluation.metrics import auc, bootstrap_ci

BASELINE = "largest_balance"
FLOOR = "random"
BASELINES = (BASELINE, FLOOR)


def top_value_per_queue(
    value: np.ndarray, score: np.ndarray, queue_codes: np.ndarray, fraction: float
) -> float:
    """Sum of `value` over the top `fraction` of every queue, ranked by `score` within the queue."""
    if len(value) == 0:
        return 0.0
    order = np.lexsort((-np.asarray(score, dtype=float), queue_codes))
    sorted_value, sorted_queue = np.asarray(value, dtype=float)[order], queue_codes[order]
    sizes = np.bincount(queue_codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    position = np.arange(len(value)) - starts[sorted_queue]
    limit = np.maximum(1, (sizes * fraction).astype(int))
    return float(sorted_value[position < limit[sorted_queue]].sum())


def _uplift_pct(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate - baseline) / baseline if baseline > 0 else float("nan")


@dataclass
class BacktestResult:
    table: pd.DataFrame
    total_value: float
    queues: int
    rows: int

    def uplift(self, strategy: str, capacity: float) -> float:
        row = self.table[(self.table["strategy"] == strategy)
                         & np.isclose(self.table["capacity"], capacity)]
        return float(row["uplift_vs_balance_pct"].iloc[0]) if len(row) else float("nan")

    def interval(self, strategy: str, capacity: float) -> tuple[float, float]:
        """Bootstrap interval of the uplift over the balance rule (NaN when not computed)."""
        row = self.table[(self.table["strategy"] == strategy)
                         & np.isclose(self.table["capacity"], capacity)]
        if not len(row):
            return float("nan"), float("nan")
        return float(row["ci_low_pct"].iloc[0]), float(row["ci_high_pct"].iloc[0])


def strategies_in(frame: pd.DataFrame) -> list[str]:
    return [c[len("score_"):] for c in frame.columns if c.startswith("score_")]


def run_backtest(
    frame: pd.DataFrame,
    capacities: tuple[float, ...],
    settings: EvaluationSettings,
    *,
    value_col: str = "realized_value",
    queue_col: str = "snapshot_at",
    debtor_cols: tuple[str, str] = ("product_line", "debtor_id"),
) -> BacktestResult:
    """Compare every `score_<strategy>` column of `frame` at each capacity."""
    queue_codes = pd.factorize(frame[queue_col])[0]
    value = frame[value_col].to_numpy(dtype=float)
    names = strategies_in(frame)
    scores = {n: frame[f"score_{n}"].to_numpy(dtype=float) for n in names}
    debtors = (frame[debtor_cols[0]].astype(str) + "|" + frame[debtor_cols[1]].astype(str)).to_numpy()
    total = float(value.sum())
    rows = []
    for capacity in capacities:
        captured = {n: top_value_per_queue(value, scores[n], queue_codes, capacity)
                    for n in names}
        for name in names:
            row = {
                "strategy": name, "capacity": capacity, "captured_value": captured[name],
                "captured_share": captured[name] / total if total else float("nan"),
                "uplift_vs_balance_pct": _uplift_pct(captured[name], captured.get(BASELINE, 0)),
                "uplift_vs_random_pct": _uplift_pct(captured[name], captured.get(FLOOR, 0)),
                "ci_low_pct": float("nan"), "ci_high_pct": float("nan"),
            }
            if name not in BASELINES and BASELINE in scores and settings.n_bootstrap > 0:
                row["ci_low_pct"], row["ci_high_pct"] = _uplift_interval(
                    value, scores[name], scores[BASELINE], queue_codes, debtors, capacity, settings
                )
            rows.append(row)
    return BacktestResult(
        pd.DataFrame(rows), total, len(np.unique(queue_codes)), len(frame)
    )


def _uplift_interval(
    value: np.ndarray,
    score: np.ndarray,
    baseline: np.ndarray,
    queues: np.ndarray,
    debtors: np.ndarray,
    capacity: float,
    settings: EvaluationSettings,
) -> tuple[float, float]:
    def statistic(rows: np.ndarray) -> float:
        # queue codes must stay dense for bincount, so re-factorize the resampled rows
        codes = pd.factorize(queues[rows])[0]
        got = top_value_per_queue(value[rows], score[rows], codes, capacity)
        base = top_value_per_queue(value[rows], baseline[rows], codes, capacity)
        return _uplift_pct(got, base)

    return bootstrap_ci(
        len(value), statistic, n_boot=settings.n_bootstrap, level=settings.ci_level,
        seed=settings.seed, groups=debtors,
    )


def verdict(
    uplift_pct: float, settings: EvaluationSettings, interval_low_pct: float | None = None
) -> str:
    """Decision rule on the headline uplift. The thresholds are policy, not statistics.

    A point estimate above the proof threshold is only called evidence when the lower end of its
    bootstrap interval is above zero. With few queues the interval is wide, and an uplift that
    could be zero should not be presented as proof. A missing interval (`None` or NaN, for example
    with `evaluation.n_bootstrap = 0`) cannot show that, so the uplift stays in the grey zone.
    """
    if not np.isfinite(uplift_pct):
        return "not estimable"
    if uplift_pct >= settings.uplift_proof_pct:
        if interval_low_pct is None or not np.isfinite(interval_low_pct):
            return ("grey zone: the uplift clears the proof threshold but no interval was "
                    "computed, so it cannot be called evidence")
        if interval_low_pct <= 0:
            return ("grey zone: the uplift clears the proof threshold but its interval includes "
                    "zero, so more queues or a controlled pilot are needed")
        return "evidence of value: worth a controlled pilot"
    if uplift_pct < settings.uplift_floor_pct:
        return "little or no gain over the balance rule: not worth the effort as is"
    return "grey zone: a business decision, ideally decided by a pilot"


def robustness_by_queue(
    frame: pd.DataFrame,
    score_col: str,
    *,
    label_col: str = "closed",
    queue_col: str = "snapshot_at",
    min_auc: float = 0.6,
) -> pd.DataFrame:
    """AUC per queue. Drift shows up as a queue whose AUC falls below `min_auc`."""
    rows = []
    for queue, part in frame.groupby(queue_col, sort=True):
        value = auc(part[label_col].to_numpy(), part[score_col].to_numpy())
        rows.append({
            queue_col: queue, "rows": len(part), "positives": int(part[label_col].sum()),
            "auc": round(value, 4) if np.isfinite(value) else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["alert"] = out["auc"] < min_auc
    return out
