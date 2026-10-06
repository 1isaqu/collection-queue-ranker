"""Two label experiments, each judged by something that does not depend on the label.

When you change how a target is defined, the models trained on different targets cannot be compared
by their own metric: a different positive class means a different AUC scale. So each experiment
fixes a **common judge** (a common set of holdout rows and a metric that is independent of the label
definition) and changes only the labels used for training.

1. `label_target_experiment` (close head): train on "an agreement started", "something was paid", or
   "a material amount was paid", then judge all three by the realised value captured in the top of
   each queue. A target that only counts paid agreements hides partial recovery inside the negative
   class; this tests whether value-aware training pays off.
2. `unresolved_experiment` (break head): how should an agreement that is still active be treated?
   As a failure, excluded, or as a success? It is right-censored (it may yet be paid or broken), so
   labelling it either way mixes a pending outcome into a class. The judge is the AUC on holdout
   rows whose outcome *is* resolved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.evaluation.backtest import top_value_per_queue
from collection_queue_ranker.evaluation.metrics import auc
from collection_queue_ranker.labels import Agreements, break_cohort
from collection_queue_ranker.models import ClassifierHead, break_space, close_space
from collection_queue_ranker.snapshots import break_features
from collection_queue_ranker.split import TimeSplit, time_split
from collection_queue_ranker.tables import RawTables


def _targets(dev: pd.DataFrame) -> dict[str, np.ndarray]:
    value = dev["realized_value"].to_numpy(dtype=float)
    positive = value[value > 0]
    material = float(np.median(positive)) if len(positive) else 0.0
    return {
        "closed": dev["closed"].to_numpy().astype(int),
        "paid_any": (value > 0).astype(int),
        "paid_material": ((value > 0) & (value >= material)).astype(int),
    }


def label_target_experiment(split: TimeSplit, settings: Settings) -> pd.DataFrame:
    """Train the close head on each target and judge by captured realised value."""
    hold = split.holdout
    codes = pd.factorize(hold["snapshot_at"])[0]
    value = hold["realized_value"].to_numpy(dtype=float)
    total = value.sum()
    rows = []
    for name, target in _targets(split.dev).items():
        if len(np.unique(target)) < 2:
            continue
        head = ClassifierHead("close", close_space(settings), dict(settings.model.close_params),
                              settings.model.seed).fit(split.dev, target)
        proba = head.predict(hold)
        row = {"target": name, "train_positive_rate": float(target.mean()),
               "auc_on_closed": auc(hold["closed"].to_numpy(), proba)}
        for capacity in settings.ranking.capacities:
            captured = top_value_per_queue(
                value, hold["balance"].to_numpy(dtype=float) * proba, codes, capacity
            )
            row[f"value_share_at_{capacity:g}"] = captured / total if total > 0 else float("nan")
        rows.append(row)
    if not rows:
        raise DataError("no target definition has both classes in the development rows")
    return pd.DataFrame(rows)


def break_experiment_frame(
    raw: RawTables, prep: Agreements, settings: Settings, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Mature break-cohort rows, resolved or not, with base features at the first agreement."""
    cohort = break_cohort(prep, as_of, settings)
    cohort = cohort[cohort["mature"]]
    if cohort.empty:
        raise DataError("no mature agreements for the break experiment")
    return break_features(raw, cohort, settings).reset_index(drop=True)


def unresolved_experiment(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compare treatments of still-active agreements, judged on resolved holdout rows only."""
    split = time_split(
        frame, "anchor_at", settings.split.holdout_fraction,
        settings.labels.break_observation_days, purge=settings.split.purge,
    )
    dev, judge = split.dev, split.holdout[split.holdout["resolved"]]
    if judge["broke"].nunique() < 2:
        raise DataError("the resolved holdout rows hold a single outcome: nothing to judge")
    failed = (dev["n_paid"] == 0).astype(int)
    variants = {
        "unresolved_as_failure": (dev, failed),
        "unresolved_excluded": (dev[dev["resolved"]], failed[dev["resolved"]]),
        "unresolved_as_success": (dev, (failed.where(dev["resolved"], 0)).astype(int)),
    }
    rows = []
    for name, (train, target) in variants.items():
        if target.nunique() < 2:
            continue
        head = ClassifierHead("break", break_space(settings), dict(settings.model.break_params),
                              settings.model.seed).fit(train, target.to_numpy())
        rows.append({
            "variant": name, "train_rows": len(train), "train_failure_rate": float(target.mean()),
            "judge_rows": len(judge), "auc_on_resolved": auc(judge["broke"].to_numpy(),
                                                             head.predict(judge)),
        })
    return pd.DataFrame(rows)
