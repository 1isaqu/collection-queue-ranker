import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.errors import DataError
from collection_queue_ranker.split import holdout_boundary, time_split, walk_forward_folds


def _frame(per_date, start="2024-01-01", step_days=30):
    """`per_date[i]` rows on the i-th date; ids are unique."""
    rows, n = [], 0
    for i, count in enumerate(per_date):
        day = pd.Timestamp(start) + pd.Timedelta(days=step_days * i)
        for _ in range(count):
            n += 1
            rows.append({"product_line": "auto", "debtor_id": n, "snapshot_at": day})
    return pd.DataFrame(rows)


def test_the_boundary_falls_between_dates_so_ties_are_never_split():
    frame = _frame([100, 100, 100, 100, 100])
    split = time_split(frame, "snapshot_at", 0.3, window_days=0)
    assert split.holdout["snapshot_at"].nunique() == 2   # whole dates only
    assert split.dev["snapshot_at"].max() < split.holdout["snapshot_at"].min()
    assert len(split.holdout) == 200


def test_the_split_does_not_depend_on_the_input_row_order():
    frame = _frame([50, 80, 60, 70, 90, 40])
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = time_split(frame, "snapshot_at", 0.3, 0)
    b = time_split(shuffled, "snapshot_at", 0.3, 0)
    pd.testing.assert_frame_equal(a.dev, b.dev)
    pd.testing.assert_frame_equal(a.holdout, b.holdout)
    assert a.holdout_start == b.holdout_start


def test_purge_drops_development_rows_whose_label_window_reaches_the_holdout():
    frame = _frame([10, 10, 10, 10, 10, 10], step_days=30)     # dates 30 days apart
    split = time_split(frame, "snapshot_at", 0.34, window_days=70)
    assert (split.dev["snapshot_at"] + pd.Timedelta(days=70) <= split.holdout_start).all()
    assert split.purged == 20   # the two dates within 70 days of the boundary
    kept = time_split(frame, "snapshot_at", 0.34, window_days=70, purge=False)
    assert kept.purged == 0 and len(kept.dev) == len(split.dev) + 20


def test_a_shared_holdout_start_is_reused_for_another_dataset():
    main = time_split(_frame([10] * 6), "snapshot_at", 0.34, 0)
    other = pd.DataFrame({"product_line": "auto", "debtor_id": range(8), "anchor_at":
                          pd.date_range("2024-01-15", periods=8, freq="30D")})
    split = time_split(other, "anchor_at", 0.9, 0, holdout_start=main.holdout_start)
    assert split.holdout_start == main.holdout_start
    assert (split.holdout["anchor_at"] >= main.holdout_start).all()
    assert (split.dev["anchor_at"] < main.holdout_start).all()


def test_an_empty_holdout_can_be_allowed_but_an_empty_development_side_never():
    other = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2],
                          "anchor_at": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    late = pd.Timestamp("2025-01-01")
    split = time_split(other, "anchor_at", 0.3, 0, holdout_start=late, require_holdout=False)
    assert split.holdout.empty and len(split.dev) == 2
    with pytest.raises(DataError):
        time_split(other, "anchor_at", 0.3, 0, holdout_start=late)
    with pytest.raises(DataError):
        time_split(other, "anchor_at", 0.3, 0, holdout_start=pd.Timestamp("2023-01-01"),
                   require_holdout=False)


def test_bad_inputs_are_reported():
    with pytest.raises(DataError, match="between 0 and 1"):
        holdout_boundary(_frame([5, 5]), "snapshot_at", 1.5)
    with pytest.raises(DataError, match="two distinct dates"):
        holdout_boundary(_frame([5]), "snapshot_at", 0.3)
    with pytest.raises(DataError):
        time_split(_frame([5, 5]), "snapshot_at", 0.5, window_days=400)  # purge empties dev


def test_walk_forward_folds_expand_and_never_train_on_the_future():
    frame = _frame([5] * 8, step_days=30)
    folds = walk_forward_folds(frame, "snapshot_at", n_folds=3, window_days=0)
    assert len(folds) == 3
    times = frame["snapshot_at"].to_numpy()
    for train, test in folds:
        assert times[train].max() < times[test].min()
    assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])
    seen = np.concatenate([t for _, t in folds])
    assert len(seen) == len(set(seen))   # test blocks never overlap


def test_walk_forward_purge_keeps_label_windows_out_of_the_test_block():
    frame = _frame([5] * 8, step_days=30)
    times = frame["snapshot_at"].to_numpy()
    for train, test in walk_forward_folds(frame, "snapshot_at", 3, window_days=45):
        assert (times[train] + np.timedelta64(45, "D") <= times[test].min()).all()


def test_walk_forward_needs_enough_dates():
    with pytest.raises(DataError):
        walk_forward_folds(_frame([5, 5]), "snapshot_at", 3, 0)
    with pytest.raises(DataError):
        walk_forward_folds(_frame([5] * 5), "snapshot_at", 0, 0)
