"""Leakage-safe time splits.

Three rules, all about the time axis:

1. **Whole dates, never rows.** The boundary falls between two distinct dates, so rows sharing a
   date (a batch date can hold thousands) never straddle it. A split by row count has to break ties
   with a sort key, and the holdout then changes whenever that key changes. A date boundary has no
   ties to break.
2. **Purge.** A development row whose label window reaches past the first holdout date has seen
   part of the holdout period through its label. Those rows are dropped (`purge=True`).
3. **Expanding walk-forward folds** apply the same purge between each training block and the block
   that follows it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collection_queue_ranker.errors import DataError


@dataclass
class TimeSplit:
    dev: pd.DataFrame
    holdout: pd.DataFrame
    holdout_start: pd.Timestamp
    purged: int


def _sorted(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    tie = [c for c in ("product_line", "debtor_id") if c in frame.columns]
    return frame.sort_values([time_col, *tie], kind="stable").reset_index(drop=True)


def holdout_boundary(frame: pd.DataFrame, time_col: str, holdout_fraction: float) -> pd.Timestamp:
    """First date of the holdout: the latest date whose tail still holds the wanted fraction."""
    if not 0 < holdout_fraction < 1:
        raise DataError("holdout_fraction must be between 0 and 1")
    counts = frame.groupby(time_col).size().sort_index()
    if len(counts) < 2:
        raise DataError("need at least two distinct dates to split in time")
    tail = counts.to_numpy()[::-1].cumsum()
    first_enough = int(np.searchsorted(tail, holdout_fraction * counts.sum(), side="left"))
    position = len(counts) - 1 - min(first_enough, len(counts) - 1)
    return pd.Timestamp(counts.index[max(position, 1)])


def time_split(
    frame: pd.DataFrame,
    time_col: str,
    holdout_fraction: float,
    window_days: int,
    *,
    purge: bool = True,
    holdout_start: pd.Timestamp | None = None,
    require_holdout: bool = True,
) -> TimeSplit:
    """Split into development and holdout at a date boundary.

    `window_days` is how far a row's label reaches past its own date. With `purge` the development
    rows whose label window crosses the boundary are removed. Pass `holdout_start` to reuse the
    boundary of another dataset (the heads share one so the break head never sees holdout outcomes).
    With `require_holdout=False` an empty holdout is allowed (the development side never is).
    """
    ordered = _sorted(frame, time_col)
    start = pd.Timestamp(holdout_start) if holdout_start is not None else holdout_boundary(
        ordered, time_col, holdout_fraction
    )
    dev = ordered[ordered[time_col] < start]
    holdout = ordered[ordered[time_col] >= start]
    purged = 0
    if purge and window_days > 0:
        reach = dev[time_col] + pd.Timedelta(days=window_days)
        keep = reach <= start
        purged = int((~keep).sum())
        dev = dev[keep]
    if dev.empty or (holdout.empty and require_holdout):
        raise DataError(
            f"split left {len(dev)} development and {len(holdout)} holdout rows "
            f"(boundary {start.date()}, purged {purged}); the data span is too short"
        )
    return TimeSplit(dev.reset_index(drop=True), holdout.reset_index(drop=True), start, purged)


def walk_forward_folds(
    frame: pd.DataFrame, time_col: str, n_folds: int, window_days: int, *, purge: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds as (train positions, test positions) over `frame` rows.

    Distinct dates are cut into `n_folds + 1` consecutive blocks. Fold i trains on blocks 0..i and
    tests on block i + 1. Folds whose test block lacks rows are skipped.
    """
    if n_folds < 1:
        raise DataError("n_folds must be at least 1")
    dates = np.sort(frame[time_col].unique())
    if len(dates) < n_folds + 1:
        raise DataError(f"{len(dates)} distinct dates cannot make {n_folds} walk-forward folds")
    blocks = np.array_split(dates, n_folds + 1)
    values = frame[time_col].to_numpy()
    folds = []
    for i in range(n_folds):
        test_start = blocks[i + 1][0]
        test = np.flatnonzero(np.isin(values, blocks[i + 1]))
        train_mask = np.isin(values, np.concatenate(blocks[: i + 1]))
        if purge and window_days > 0:
            reach = values + np.timedelta64(window_days, "D")
            train_mask &= reach <= test_start
        train = np.flatnonzero(train_mask)
        if len(train) and len(test):
            folds.append((train, test))
    return folds
