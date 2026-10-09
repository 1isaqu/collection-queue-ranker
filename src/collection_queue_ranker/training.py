"""Training orchestration.

Order of operations, and why:

1. Split **once**, at a date boundary shared by every head, with a purge (see `split.py`). The
   holdout is not touched again until the final report.
2. Choose the close-head hyper-parameters by walk-forward cross-validation **on the development
   rows only**. The default selection metric is business value: the share of realised value
   captured in the top of each queue when ranking by `balance x P(close)`. Two models can tie on
   AUC and still differ in money, which is why AUC is not the default.
3. Fit the break head, the close head and the ranker on development rows.
4. Score the holdout and write the report. The bundle that is saved is the one trained on
   development rows: the holdout is for judging, not for refitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import ConfigError, DataError
from collection_queue_ranker.evaluation.backtest import top_value_per_queue
from collection_queue_ranker.evaluation.metrics import auc
from collection_queue_ranker.evaluation.report import build_report, jsonable, write_report
from collection_queue_ranker.models import (
    ClassifierHead,
    ModelBundle,
    RankerHead,
    break_space,
    close_space,
    new_meta,
    ranker_space,
)
from collection_queue_ranker.scoring import scored_frame
from collection_queue_ranker.split import TimeSplit, time_split, walk_forward_folds
from collection_queue_ranker.tracking import NullTracker, Tracker

log = logging.getLogger(__name__)

SELECTION_METRICS = ("value_capture", "auc")


@dataclass
class TrainResult:
    bundle: ModelBundle
    split: TimeSplit
    break_split: TimeSplit
    holdout: pd.DataFrame
    report: dict[str, Any]
    selection: pd.DataFrame
    selected_params: dict[str, Any]
    files: list[Path]


def _fold_metrics(
    train: pd.DataFrame, test: pd.DataFrame, params: dict[str, Any], settings: Settings
) -> dict[str, float]:
    head = ClassifierHead("close", close_space(settings), params, settings.model.seed)
    head.fit(train, train["closed"].to_numpy())
    proba = head.predict(test)
    codes = pd.factorize(test["snapshot_at"])[0]
    value = test["realized_value"].to_numpy(dtype=float)
    total = value.sum()
    captured = top_value_per_queue(
        value, test["balance"].to_numpy(dtype=float) * proba, codes,
        settings.model.selection_capacity,
    )
    return {
        "auc": auc(test["closed"].to_numpy(), proba),
        "value_capture": captured / total if total > 0 else float("nan"),
    }


def select_close_params(
    dev: pd.DataFrame, settings: Settings
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Walk-forward selection among `model.candidates`. Returns the winner and the full table."""
    metric = settings.model.selection_metric
    if metric not in SELECTION_METRICS:
        raise ConfigError(f"model.selection_metric must be one of {SELECTION_METRICS}")
    base = dict(settings.model.close_params)
    candidates = [{**base, **c} for c in settings.model.candidates] or [base]
    folds = walk_forward_folds(
        dev, "snapshot_at", settings.split.cv_folds, settings.labels.horizon_days,
        purge=settings.split.purge,
    )
    rows = []
    for i, params in enumerate(candidates):
        per_fold = []
        for train_idx, test_idx in folds:
            train, test = dev.iloc[train_idx], dev.iloc[test_idx]
            if train["closed"].nunique() < 2:
                continue
            per_fold.append(_fold_metrics(train, test, params, settings))
        mean = {m: float(np.nanmean([f[m] for f in per_fold])) if per_fold else float("nan")
                for m in SELECTION_METRICS}
        rows.append({"candidate": i, "folds": len(per_fold), **mean, "params": params})
    table = pd.DataFrame(rows)
    usable = table[table[metric].notna()]
    if usable.empty:
        log.warning("no usable cross-validation folds; keeping the default close-head parameters")
        return base, table
    best = usable.sort_values([metric, "candidate"], ascending=[False, True]).iloc[0]
    return dict(best["params"]), table


def _split_heads(
    snapshots: pd.DataFrame, break_df: pd.DataFrame, settings: Settings
) -> tuple[TimeSplit, TimeSplit]:
    s = settings.split
    main = time_split(
        snapshots, "snapshot_at", s.holdout_fraction, settings.labels.horizon_days, purge=s.purge
    )
    brk = time_split(
        break_df, "anchor_at", s.holdout_fraction, settings.labels.break_observation_days,
        purge=s.purge, holdout_start=main.holdout_start, require_holdout=False,
    )
    return main, brk


def fit_bundle(
    split: TimeSplit, break_split: TimeSplit, params: dict[str, Any], settings: Settings,
    as_of: pd.Timestamp,
) -> ModelBundle:
    ms = settings.model
    close = ClassifierHead("close", close_space(settings), params, ms.seed)
    close.fit(split.dev, split.dev["closed"].to_numpy())
    brk = ClassifierHead("break", break_space(settings), dict(ms.break_params), ms.seed)
    brk.fit(break_split.dev, break_split.dev["broke"].to_numpy())
    ranker = None
    if ms.train_ranker:
        ranker = RankerHead.from_settings(ranker_space(settings), ms).fit(split.dev)
    return ModelBundle(close, brk, ranker, new_meta(settings, as_of, split.holdout_start))


def _data_summary(split: TimeSplit, brk: TimeSplit) -> dict[str, Any]:
    return {
        "dev_rows": len(split.dev), "holdout_rows": len(split.holdout),
        "dev_queues": int(split.dev["snapshot_at"].nunique()),
        "holdout_queues": int(split.holdout["snapshot_at"].nunique()),
        "holdout_start": split.holdout_start.date().isoformat(),
        "purged_rows": split.purged, "break_dev_rows": len(brk.dev),
        "break_holdout_rows": len(brk.holdout), "break_purged_rows": brk.purged,
        "close_rate_dev": float(split.dev["closed"].mean()),
        "close_rate_holdout": float(split.holdout["closed"].mean()),
    }


def train(
    snapshots: pd.DataFrame,
    break_df: pd.DataFrame,
    settings: Settings,
    as_of: pd.Timestamp,
    *,
    tracker: Tracker | None = None,
    out_dir: str | Path | None = None,
) -> TrainResult:
    """Fit, judge on the holdout, save the bundle and the report under `out_dir`."""
    tracker = tracker or NullTracker()
    check_inputs(snapshots, break_df)
    split, break_split = _split_heads(snapshots, break_df, settings)
    params, selection = select_close_params(split.dev, settings)
    bundle = fit_bundle(split, break_split, params, settings, as_of)
    holdout = scored_frame(split.holdout, bundle, settings.model.seed)
    data = _data_summary(split, break_split)
    report = build_report(
        holdout, bundle, settings,
        break_holdout=break_split.holdout if len(break_split.holdout) else None, data=data,
    )
    bundle.meta["metrics"] = jsonable({
        k: v for k, v in report.items() if k in ("close_head", "break_head")
    })
    files = _persist(bundle, report, selection, out_dir, tracker, settings, params)
    return TrainResult(bundle, split, break_split, holdout, report, selection, params, files)


def _headline_metrics(report: dict[str, Any]) -> dict[str, float]:
    close = report["close_head"]
    metrics = {"close_auc": close["auc"], "close_auc_ci_low": close["auc_ci"][0],
               "close_auc_ci_high": close["auc_ci"][1]}
    for k, v in close["precision_at"].items():
        metrics[f"close_precision_at_{k}"] = v
    rank = report["ranking"]
    metrics["uplift_vs_balance_pct"] = rank["uplift_vs_balance_pct"]
    if "break_head" in report:
        metrics["break_auc"] = report["break_head"]["auc"]
    return metrics


def _persist(
    bundle: ModelBundle,
    report: dict[str, Any],
    selection: pd.DataFrame,
    out_dir: str | Path | None,
    tracker: Tracker,
    settings: Settings,
    params: dict[str, Any],
) -> list[Path]:
    files: list[Path] = []
    with tracker.run(
        "train", {"settings": settings.to_dict(), "selected_close_params": params},
        {"strategy": settings.ranking.strategy, "stage": "train"},
    ):
        tracker.log_metrics(_headline_metrics(report))
        if out_dir is None:
            return files
        out = Path(out_dir)
        files = write_report(report, out)
        files.append(bundle.save(out / "bundle.joblib"))
        selection.assign(params=selection["params"].astype(str)).to_csv(
            out / "selection.csv", index=False
        )
        files.append(out / "selection.csv")
        for head, name in ((bundle.close, "close"), (bundle.brk, "break"), (bundle.ranker, "ranker")):
            if head is not None:
                path = out / f"importance_{name}.csv"
                head.importance().to_csv(path, index=False)
                files.append(path)
        for path in files:
            if path.suffix != ".joblib":
                tracker.log_artifact(path, "reports")
    return files


def check_inputs(snapshots: pd.DataFrame, break_df: pd.DataFrame) -> None:
    """Fail early, with a readable message, when the data cannot train anything."""
    if snapshots["closed"].nunique() < 2:
        raise DataError("every snapshot row has the same close label: nothing to learn")
    if break_df["broke"].nunique() < 2:
        raise DataError("every break-cohort row has the same outcome: nothing to learn")
