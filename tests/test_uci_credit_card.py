"""The public credit-card adapter: validation, the population and label rules, and no look-ahead.

Every panel here is written inline or drawn from a seeded generator with invented numbers. No test
downloads anything: `fetch` is checked with a fake `urlopen`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker import uci_credit_card as card
from collection_queue_ranker.cli import main
from collection_queue_ranker.config import Settings, load_settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.base import BASE_CATEGORICAL, BASE_NUMERIC
from collection_queue_ranker.features.contact import CONTACT_FEATURES
from collection_queue_ranker.pipeline import read_datasets, write_datasets

ROOT = Path(__file__).resolve().parents[1]

_STATUS = ("X11", "X10", "X9", "X8", "X7", "X6")
_BILL = ("X17", "X16", "X15", "X14", "X13", "X12")
_PAID = ("X23", "X22", "X21", "X20", "X19", "X18")


def raw_frame(clients: list[dict]) -> pd.DataFrame:
    """The source layout from per-client lists written oldest month first."""
    rows = []
    for c in clients:
        row = {"ID": c["id"], "X1": c.get("limit", 100_000), "X2": 1, "X3": 1, "X4": 1, "X5": 30,
               "Y": c.get("y", 0)}
        for columns, key in ((_STATUS, "status"), (_BILL, "bill"), (_PAID, "paid")):
            row.update(dict(zip(columns, c[key], strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def small_panel() -> card.Panel:
    """Six invented clients, each there to test one rule."""
    return card.panel_from_frame(raw_frame([
        # in arrears at months 0 to 4, pays now and then
        {"id": 1, "limit": 10_000, "status": [2, 2, 2, 2, 2, 0],
         "bill": [1000, 1100, 1200, 1300, 1400, 1500], "paid": [0, 300, 0, 400, 500, 600]},
        # in arrears only at months 2 and 3, newly late at month 2
        {"id": 2, "limit": 50_000, "status": [0, 0, 2, 2, 0, 0],
         "bill": [500, 500, 800, 900, 100, 100], "paid": [100, 100, 100, 0, 700, 100]},
        # never in arrears
        {"id": 3, "status": [0, -1, 0, -2, -1, 0], "bill": [10, 20, 30, 40, 50, 60],
         "paid": [5, 5, 5, 5, 5, 5]},
        # late, but the bill is zero at month 1 and negative at month 2
        {"id": 4, "status": [2, 2, 2, 0, 0, 0], "bill": [300, 0, -50, 10, 10, 10],
         "paid": [10, 10, 10, 10, 10, 10]},
        # late at month 0 only, pays at once and again the month after
        {"id": 5, "status": [3, 0, 0, 0, 0, 0], "bill": [400, 100, 100, 100, 100, 100],
         "paid": [0, 50, 20, 0, 0, 0]},
        # late all the way, never pays
        {"id": 6, "status": [2, 2, 2, 2, 2, 2], "bill": [900, 910, 920, 930, 940, 950],
         "paid": [0, 0, 0, 0, 0, 0]},
    ]))


def invented_panel(n: int, seed: int) -> card.Panel:
    """A seeded panel with a planted willingness to pay. Every number is made up."""
    rng = np.random.default_rng(seed)
    willing = rng.random(n)
    late = rng.random((n, 6)) < 0.4
    status = np.where(late, rng.integers(1, 4, (n, 6)), rng.integers(-2, 1, (n, 6)))
    pays = rng.random((n, 6)) < willing[:, None]
    return card.Panel(
        client_id=np.arange(1, n + 1), limit=rng.choice([20_000.0, 80_000.0, 200_000.0], n),
        status=status.astype(int), bill=np.round(rng.gamma(2.0, 15_000, (n, 6))),
        paid=np.where(pays, np.round(rng.gamma(2.0, 1_500, (n, 6))), 0.0),
        default_next_month=(rng.random(n) < 0.2).astype(int),
    )


def snapshot_at(ds: card.CardDatasets, month: int) -> pd.DataFrame:
    rows = ds.snapshots[ds.snapshots["snapshot_at"] == card.cutoff_dates()[month]]
    return rows.set_index("debtor_id")


# --- reading and validating -------------------------------------------------------------------


def test_months_are_arranged_oldest_first_whatever_the_order_in_the_file():
    raw = raw_frame([{"id": 1, "status": [0] * 6, "bill": [0] * 6, "paid": [0] * 6}])
    # In the source file X6 is September and X11 is April.
    for offset, column in enumerate(("X6", "X7", "X8", "X9", "X10", "X11")):
        raw[column] = 6 - offset
    for offset, column in enumerate(("X12", "X13", "X14", "X15", "X16", "X17")):
        raw[column] = 60 - 10 * offset
    for offset, column in enumerate(("X18", "X19", "X20", "X21", "X22", "X23")):
        raw[column] = 6 - offset
    panel = card.panel_from_frame(raw)
    assert panel.status[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert panel.bill[0].tolist() == [10, 20, 30, 40, 50, 60]
    assert panel.paid[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert len(panel) == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.drop(columns=["X7"]), "lacks the column"),
        (lambda f: f.assign(X7=np.nan), "missing or non-numeric"),
        (lambda f: f.assign(X7="late"), "missing or non-numeric"),
        (lambda f: pd.concat([f, f]), "repeated client ID"),
        (lambda f: f.assign(X20=-1), "negative payment"),
        (lambda f: f.assign(X6=10), "outside the documented range"),
        (lambda f: f.assign(Y=2), "other than 0 and 1"),
    ],
)
def test_a_panel_that_breaks_the_source_layout_is_refused(change, message):
    good = raw_frame([{"id": 1, "status": [0] * 6, "bill": [1] * 6, "paid": [1] * 6}])
    with pytest.raises(DataError, match=message):
        card.panel_from_frame(change(good))


def test_the_hash_check_names_both_hashes():
    card.check_hash(b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    with pytest.raises(DataError, match=r"expected 0{64}"):
        card.check_hash(b"abc", "0" * 64)


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.payload


def test_fetch_downloads_once_and_reuses_the_verified_file(tmp_path, monkeypatch):
    payload = b"ID,X1\n1,2\n"
    monkeypatch.setattr(card, "DATA_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def fake(url, timeout):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(card.urllib.request, "urlopen", fake)
    path = card.fetch(tmp_path / "data")
    assert path.read_bytes() == payload and calls == [card.DATA_URL]
    assert card.fetch(tmp_path / "data") == path and len(calls) == 1


def test_fetch_replaces_a_file_with_the_wrong_hash_and_writes_nothing_for_a_wrong_download(
    tmp_path, monkeypatch
):
    good = b"good bytes"
    monkeypatch.setattr(card, "DATA_SHA256", hashlib.sha256(good).hexdigest())
    target = tmp_path / card.DATA_FILE
    target.write_bytes(b"something else")
    monkeypatch.setattr(card.urllib.request, "urlopen", lambda url, timeout: _Response(good))
    assert card.fetch(tmp_path).read_bytes() == good

    fresh = tmp_path / "fresh"
    monkeypatch.setattr(card.urllib.request, "urlopen", lambda url, timeout: _Response(b"bad"))
    with pytest.raises(DataError, match="not the file"):
        card.fetch(fresh)
    assert not (fresh / card.DATA_FILE).exists()


# --- the population, the labels and the features ----------------------------------------------


def test_the_cutoffs_are_the_first_days_of_the_months_after_each_statement():
    assert [d.date().isoformat() for d in card.cutoff_dates()] == [
        "2005-05-01", "2005-06-01", "2005-07-01", "2005-08-01", "2005-09-01",
    ]
    assert card.as_of_date() == pd.Timestamp("2005-10-01")


def test_a_client_is_queued_only_when_late_with_a_positive_bill():
    ds = card.build_datasets(small_panel())
    by_month = {m: sorted(snapshot_at(ds, m).index) for m in range(5)}
    assert by_month[0] == [1, 4, 5, 6]
    assert by_month[1] == [1, 6]               # client 4 is late but its bill is zero
    assert by_month[2] == [1, 2, 6]            # client 4 is late but its bill is negative
    assert by_month[3] == [1, 2, 6]
    assert by_month[4] == [1, 6]               # client 2 is no longer late
    assert 3 not in ds.snapshots["debtor_id"].to_numpy()
    assert ds.quality.late_without_positive_bill == (0, 1, 1, 0, 0)


def test_the_label_is_the_payment_of_the_following_month_and_nothing_else():
    ds = card.build_datasets(small_panel())
    month2 = snapshot_at(ds, 2)
    assert month2.loc[1, "realized_value"] == 400 and month2.loc[1, "closed"] == 1
    assert month2.loc[2, "realized_value"] == 0 and month2.loc[2, "closed"] == 0
    assert month2.loc[6, "closed"] == 0
    assert (ds.snapshots["closed"] == (ds.snapshots["realized_value"] > 0)).all()


def test_features_are_what_was_known_at_the_end_of_the_statement_month():
    ds = card.build_datasets(small_panel())
    one = snapshot_at(ds, 2).loc[1]
    assert one["balance"] == 1200 and one["days_overdue_max"] == 60 == one["days_overdue_min"]
    assert one["credit_limit"] == 10_000 and one["utilisation"] == pytest.approx(0.12)
    assert one["paid_last_month"] == 0 and one["paid_two_months_ago"] == 300
    assert one["months_paid_recent"] == 1 and one["months_in_arrears"] == 3
    assert one["worst_delay_history"] == 2 and one["delay_change"] == 0
    assert one["paid_share_of_previous_bill"] == 0
    assert one["bill_growth"] == pytest.approx(100 / 1100)
    fresh = snapshot_at(ds, 2).loc[2]
    assert fresh["months_in_arrears"] == 1 and fresh["delay_change"] == 2
    assert fresh["months_paid_recent"] == 3 and fresh["worst_delay_history"] == 2


def test_the_first_cutoff_has_no_history_to_compute_from():
    row = snapshot_at(card.build_datasets(small_panel()), 0).loc[1]
    for column in ("paid_two_months_ago", "paid_share_of_previous_bill", "bill_growth",
                   "delay_change"):
        assert np.isnan(row[column])
    assert row["months_in_arrears"] == 1 and row["months_paid_recent"] == 0


def test_a_division_by_a_bill_that_is_not_positive_gives_a_missing_value():
    panel = card.panel_from_frame(raw_frame([
        {"id": 1, "status": [0, 2, 2, 2, 2, 2], "bill": [0, 100, 100, 100, 100, 100],
         "paid": [0, 10, 10, 10, 10, 10]},
    ]))
    row = snapshot_at(card.build_datasets(panel), 1).loc[1]
    assert np.isnan(row["paid_share_of_previous_bill"]) and np.isnan(row["bill_growth"])


@pytest.mark.parametrize("month", range(5))
def test_nothing_after_the_cutoff_reaches_the_features_or_the_population(month):
    panel = invented_panel(400, seed=11)
    rng = np.random.default_rng(99)
    later = slice(month + 1, 6)
    changed = dataclasses.replace(
        panel, status=panel.status.copy(), bill=panel.bill.copy(), paid=panel.paid.copy()
    )
    changed.status[:, later] = rng.integers(-2, 4, changed.status[:, later].shape)
    changed.bill[:, later] = rng.gamma(2.0, 20_000, changed.bill[:, later].shape)
    changed.paid[:, later] = rng.gamma(2.0, 3_000, changed.paid[:, later].shape)
    labels = ["closed", "realized_value"]
    before = snapshot_at(card.build_datasets(panel), month).drop(columns=labels)
    after = snapshot_at(card.build_datasets(changed), month).drop(columns=labels)
    pd.testing.assert_frame_equal(before, after)


def test_a_stricter_population_keeps_a_subset_of_the_same_rows():
    panel = invented_panel(600, seed=4)
    loose = card.build_datasets(panel).snapshots
    strict = card.build_datasets(panel, min_status=2, min_months_in_arrears=2).snapshots
    keys = ["snapshot_at", "debtor_id"]
    assert 0 < len(strict) < len(loose)
    merged = strict.merge(loose, on=keys, suffixes=("_s", ""))
    assert len(merged) == len(strict)
    assert (strict["months_in_arrears"] >= 2).all() and (strict["days_overdue_max"] >= 60).all()


# --- the break cohort -------------------------------------------------------------------------


def test_the_break_cohort_holds_each_client_once_at_the_first_month_it_paid():
    ds = card.build_datasets(small_panel())
    cohort = ds.break_df.set_index("debtor_id")
    assert sorted(cohort.index) == [1, 2, 4, 5]
    # Client 1 is late at month 0 and pays 300 in month 1, then nothing in month 2.
    assert cohort.loc[1, "anchor_at"] == pd.Timestamp("2005-05-01") and cohort.loc[1, "broke"] == 1
    # Client 5 pays 50 in month 1 and 20 in month 2.
    assert cohort.loc[5, "anchor_at"] == pd.Timestamp("2005-05-01") and cohort.loc[5, "broke"] == 0
    # Client 2 pays nothing after being late at month 2, then 700 and 100 after month 3.
    assert cohort.loc[2, "anchor_at"] == pd.Timestamp("2005-08-01") and cohort.loc[2, "broke"] == 0
    assert 6 not in cohort.index                  # never paid
    assert ds.quality.break_rows == 4


def test_the_break_cohort_stops_when_the_second_month_would_be_unobserved():
    panel = card.panel_from_frame(raw_frame([
        {"id": 1, "status": [0, 0, 0, 0, 2, 0], "bill": [10] * 6, "paid": [0, 0, 0, 0, 0, 90]},
    ]))
    ds = card.build_datasets(panel)
    assert len(snapshot_at(ds, 4)) == 1 and snapshot_at(ds, 4).loc[1, "closed"] == 1
    assert ds.break_df.empty and list(ds.break_df.columns) == ["anchor_at", "broke"]


def test_break_features_are_the_snapshot_features_of_the_same_row():
    ds = card.build_datasets(invented_panel(500, seed=8))
    keys = ["debtor_id", "snapshot_at"]
    joined = ds.break_df.rename(columns={"anchor_at": "snapshot_at"}).merge(
        ds.snapshots, on=keys, suffixes=("_b", ""), how="left"
    )
    assert len(joined) == len(ds.break_df) and joined["balance"].notna().all()
    assert (joined["balance_b"] == joined["balance"]).all()
    assert (joined["closed"] == 1).all()
    assert ds.break_df["debtor_id"].is_unique


# --- the shape the rest of the package reads --------------------------------------------------


def test_snapshots_carry_every_column_the_models_and_the_report_read(tmp_path):
    ds = card.build_datasets(invented_panel(500, seed=2))
    needed = {
        "product_line", "debtor_id", "snapshot_at", "closed", "realized_value", "contact_factor",
        *BASE_NUMERIC, *BASE_CATEGORICAL, *CONTACT_FEATURES, *card.EXTRA_NUMERIC,
    }
    assert needed <= set(ds.snapshots.columns)
    assert {"debtor_id", "anchor_at", "broke", *BASE_NUMERIC, *BASE_CATEGORICAL} <= set(
        ds.break_df.columns
    )
    assert (ds.snapshots["contact_factor"] == Settings().queue.contact_floor).all()
    assert (ds.snapshots[["phone_count", "email_count", "n_answered", "is_company"]] == 0).all().all()
    write_datasets(tmp_path, ds.snapshots, ds.break_df, ds.as_of)
    snapshots, break_df, as_of = read_datasets(tmp_path)
    assert as_of == pd.Timestamp("2005-10-01") and len(snapshots) == len(ds.snapshots)
    assert len(break_df) == len(ds.break_df)


def test_the_quality_report_adds_up():
    quality = card.build_datasets(small_panel()).quality
    assert quality.clients == 6 and quality.negative_bill_cells == 1
    assert quality.status_one_by_month == (0, 0, 0, 0, 0, 0)
    assert quality.late_by_cutoff == (4, 3, 4, 3, 2)
    assert quality.snapshot_rows == 14 and quality.snapshot_clients == 5
    assert quality.clients_at_several_cutoffs == 3


def test_no_client_in_arrears_at_any_cutoff_is_an_error():
    panel = card.panel_from_frame(raw_frame([
        {"id": 1, "status": [0, 0, 0, 0, 0, 2], "bill": [10] * 6, "paid": [1] * 6},
    ]))
    with pytest.raises(DataError, match="no client is in arrears"):
        card.build_datasets(panel)


# --- the settings and the stages --------------------------------------------------------------


def test_the_overrides_load_as_a_settings_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(card.SETTINGS_OVERRIDES), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded == card.card_settings()
    assert loaded.labels.horizon_days == 30 and loaded.split.cv_folds == 2
    assert list(loaded.features.extra_numeric) == list(card.EXTRA_NUMERIC)
    assert card.card_settings(model={"seed": 3}).model.seed == 3


def test_cqr_train_and_backtest_run_unchanged_on_the_written_files(tmp_path, capsys):
    ds = card.build_datasets(invented_panel(1500, seed=3))
    write_datasets(tmp_path, ds.snapshots, ds.break_df, ds.as_of)
    config = {
        **card.SETTINGS_OVERRIDES, "evaluation": {"n_bootstrap": 20},
        "model": {"candidates": [{"max_depth": 3, "num_leaves": 8}]},
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    common = ["--artifacts", str(tmp_path), "--config", str(path), "--no-tracking"]
    assert main(["train", *common]) == 0
    out = capsys.readouterr().out
    assert "Holdout evaluation" in out and "Ranking under limited capacity" in out
    for name in ("report.md", "metrics.json", "bundle.joblib", "importance_close.csv"):
        assert (tmp_path / name).is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["data"]["dev_queues"] == 3 and metrics["data"]["holdout_queues"] == 2
    assert main(["backtest", *common]) == 0
    assert (tmp_path / "backtest" / "report.md").is_file()
