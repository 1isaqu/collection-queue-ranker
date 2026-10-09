"""Cutoff rules for snapshot features: nothing dated at or after T may leak into a row at T."""

import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.base import base_features, make_anchors
from collection_queue_ranker.features.contact import CONTACT_FEATURES, contact_features
from collection_queue_ranker.labels import prepare_agreements
from collection_queue_ranker.snapshots import (
    build_break_dataset,
    build_snapshots,
    cutoff_dates,
    features_at,
    infer_as_of,
)
from world import NaT, agreements_df, debts_df, events_df, make_raw, phones_df, ts

T = ts("2024-03-01")
S = Settings()

DEBTS = debts_df(
    (1, 1, ts("2024-01-01"), ts("2023-01-01"), 100.0),
    (2, 2, ts("2024-02-01"), ts("2023-06-01"), 50.0),
    (3, 3, ts("2024-03-01"), ts("2023-06-01"), 70.0),  # enters exactly at T: out
)


def _features(raw, cutoff=T, prep=None):
    prep = prep or prepare_agreements(raw, S)
    return features_at(raw, prep, cutoff, S).set_index("debtor_id")


def test_a_debt_entering_at_or_after_the_cutoff_is_not_in_the_population():
    raw = make_raw(debts=DEBTS)
    f = _features(raw)
    assert sorted(f.index) == [1, 2]
    assert f.loc[1, "tenure_days"] == (T - ts("2024-01-01")).days


def test_days_overdue_are_measured_at_the_cutoff_not_at_extraction():
    debts = debts_df((1, 1, ts("2024-01-01"), ts("2023-12-01"), 10.0),
                     (2, 1, ts("2024-01-01"), ts("2024-01-21"), 10.0))
    f = _features(make_raw(debts=debts))
    assert f.loc[1, "days_overdue_max"] == (T - ts("2023-12-01")).days
    assert f.loc[1, "days_overdue_min"] == (T - ts("2024-01-21")).days
    assert f.loc[1, "overdue_spread"] == f.loc[1, "days_overdue_max"] - f.loc[1, "days_overdue_min"]
    assert f.loc[1, "balance"] == 20.0


def test_a_debt_closed_before_the_cutoff_does_not_count_but_one_closed_later_does():
    debts = debts_df((1, 1, ts("2024-01-01"), ts("2023-06-01"), 10.0, ts("2024-02-15")),
                     (2, 2, ts("2024-01-01"), ts("2023-06-01"), 10.0, ts("2024-03-20")))
    f = _features(make_raw(debts=debts))
    assert list(f.index) == [2]  # debtor 1 had nothing open at T


def test_contacts_at_or_after_the_cutoff_do_not_enter():
    events = events_df(
        (1, 1, 3, ts("2024-02-10")), (2, 1, 1, ts("2024-02-20")),   # answered, then no answer
        (3, 1, 3, ts("2024-03-01")),                                  # at T: out
        (4, 1, 2, ts("2024-04-01")),                                  # after T: out
    )
    f = _features(make_raw(debts=DEBTS, events=events))
    assert f.loc[1, "n_answered"] == 1 and f.loc[1, "n_wrong_party"] == 0
    assert f.loc[1, "answer_rate"] == 0.5
    assert f.loc[1, "days_since_answered"] == (T - ts("2024-02-10")).days


def test_no_contact_means_zero_counts_and_missing_rate_not_a_median():
    f = _features(make_raw(debts=DEBTS))
    assert f.loc[2, "n_answered"] == 0
    assert np.isnan(f.loc[2, "answer_rate"]) and np.isnan(f.loc[2, "days_since_answered"])


def test_attempt_intensity_is_not_a_feature():
    # How hard we tried measures who the operation already chose to work: a feedback loop.
    banned = {"attempts", "n_attempts", "attempts_30d", "n_voice", "voice_attempts"}
    assert not banned & set(CONTACT_FEATURES)
    events = events_df(*[(i, 1, 1, ts("2024-02-01") + pd.Timedelta(days=i)) for i in range(1, 9)])
    f = _features(make_raw(debts=DEBTS, events=events))
    assert not banned & set(f.columns)


def test_only_response_categories_count_not_automated_messages():
    results = pd.DataFrame({"result_id": [1, 2, 3], "label": ["x", "y", "z"], "answered": 0,
                            "right_party": 0, "category": ["message", "no_answer", "answered"]})
    results.loc[2, ["answered", "right_party"]] = 1
    events = events_df((1, 1, 1, ts("2024-02-01")), (2, 1, 1, ts("2024-02-02")),
                       (3, 1, 2, ts("2024-02-03")))
    f = _features(make_raw(debts=DEBTS, events=events, results=results))
    assert f.loc[1, "n_answered"] == 0 and f.loc[1, "answer_rate"] == 0.0  # 1 voice attempt, 0 answers


def test_a_debtor_with_an_agreement_before_the_cutoff_leaves_the_population():
    agreements = agreements_df(
        ("a1", 1, 0, 40.0, 0.0, ts("2024-02-15"), NaT, "ACTIVE"),
        ("a2", 2, 0, 40.0, 0.0, ts("2024-03-01"), NaT, "ACTIVE"),  # at T: still in the population
    )
    raw = make_raw(debts=DEBTS, agreements=agreements)
    assert sorted(_features(raw).index) == [2]
    prep = prepare_agreements(raw, S)
    keep = features_at(raw, prep, T, S, exclude_existing_agreements=False)
    assert sorted(keep["debtor_id"]) == [1, 2]


def test_phone_and_email_counts_respect_creation_dates_when_present():
    phones = phones_df((1, 1, 1, 0, 0, 0, ts("2024-01-05"), NaT),
                       (2, 1, 1, 0, 0, 0, ts("2024-03-05"), NaT),   # created after T
                       (3, 1, 1, 0, 0, 0, NaT, NaT))                 # undated: always counted
    f = _features(make_raw(debts=DEBTS, phones=phones))
    assert f.loc[1, "phone_count"] == 2 and f.loc[2, "phone_count"] == 0 and f.loc[1, "email_count"] == 0


def test_the_same_debtor_id_in_two_product_lines_stays_separate():
    other = debts_df((9, 1, ts("2024-01-01"), ts("2023-01-01"), 999.0)).assign(product_line="consumer")
    debts = pd.concat([DEBTS.assign(product_line="auto"), other], ignore_index=True)
    raw = make_raw(debts=debts)
    f = features_at(raw, prepare_agreements(raw, S), T, S).set_index(["product_line", "debtor_id"])
    assert f.loc[("auto", 1), "balance"] == 100.0 and f.loc[("consumer", 1), "balance"] == 999.0


def test_min_balance_filters_the_population():
    strict = S.with_overrides({"snapshots": {"min_balance": 60.0}})
    raw = make_raw(debts=DEBTS)
    out = features_at(raw, prepare_agreements(raw, strict), T, strict)
    assert out["debtor_id"].tolist() == [1]


def test_balance_column_is_configurable():
    debts = DEBTS.assign(original_balance=[500.0, 600.0, 700.0])
    cfg = S.with_overrides({"features": {"balance_column": "original_balance"}})
    raw = make_raw(cfg, debts=debts)
    out = features_at(raw, prepare_agreements(raw, cfg), T, cfg).set_index("debtor_id")
    assert out.loc[1, "balance"] == 500.0


def test_base_features_are_computed_per_anchor_date():
    anchors = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2],
                            "anchor_at": [ts("2024-02-01"), ts("2024-05-01")]})
    raw = make_raw(debts=DEBTS)
    out = base_features(raw.debts, raw.debtors, raw.phones, raw.emails, anchors).set_index("debtor_id")
    assert out.loc[1, "days_overdue_max"] == (ts("2024-02-01") - ts("2023-01-01")).days
    assert out.loc[2, "days_overdue_max"] == (ts("2024-05-01") - ts("2023-06-01")).days


def test_make_anchors_requires_an_open_debt():
    debts = make_raw(debts=debts_df(
        (1, 1, ts("2024-01-01"), ts("2023-01-01"), 1.0, ts("2024-02-01")))).debts
    assert make_anchors(debts, T).empty
    assert len(make_anchors(debts, ts("2024-01-15"))) == 1


def test_contact_features_handles_empty_inputs():
    anchors = pd.DataFrame({"product_line": ["auto"], "debtor_id": [1], "anchor_at": [T]})
    raw = make_raw(debts=DEBTS)
    assert contact_features(raw.events, anchors).empty
    assert contact_features(raw.events.iloc[0:0], anchors.iloc[0:0]).empty


def test_snapshot_labels_use_only_the_forward_window():
    agreements = agreements_df(
        ("a2", 2, 0, 40.0, 40.0, ts("2024-04-10"), ts("2024-04-20"), "PAID"),
        ("a1", 1, 0, 40.0, 0.0, ts("2024-06-15"), NaT, "ACTIVE"),  # beyond T + 60 days
    )
    debts = debts_df((1, 1, ts("2024-01-01"), ts("2023-01-01"), 100.0),
                     (2, 2, ts("2024-02-01"), ts("2023-06-01"), 50.0),
                     (4, 4, ts("2024-01-01"), ts("2023-06-01"), 50.0))
    events = events_df((1, 4, 1, ts("2024-01-05")), (2, 4, 1, ts("2024-06-30")))
    raw = make_raw(debts=debts, agreements=agreements, events=events)
    prep = prepare_agreements(raw, S)
    snaps = build_snapshots(raw, prep, S.with_overrides(
        {"snapshots": {"first_cutoff": "2024-03-01", "frequency": "MS"}}), ts("2024-07-01"))
    first = snaps[snaps["snapshot_at"] == T].set_index("debtor_id")
    assert first.loc[2, "closed"] == 1 and first.loc[2, "realized_value"] == 40.0
    assert first.loc[1, "closed"] == 0 and first.loc[4, "closed"] == 0
    # the cutoffs stop at as_of - horizon so every label window is fully observed
    assert snaps["snapshot_at"].max() == ts("2024-05-01")


def test_cutoff_dates_are_derived_from_the_data_and_validated():
    raw = make_raw(debts=DEBTS, events=events_df((1, 1, 3, ts("2024-02-10"))))
    dates = cutoff_dates(raw, S, ts("2024-09-01"))
    assert dates[0] >= ts("2024-01-01") + pd.Timedelta(days=S.snapshots.warmup_days)
    assert dates[-1] <= ts("2024-09-01") - pd.Timedelta(days=S.labels.horizon_days)
    with pytest.raises(DataError, match="too short"):
        cutoff_dates(raw, S, ts("2024-01-15"))


def test_as_of_defaults_to_the_day_after_the_latest_data():
    raw = make_raw(debts=DEBTS, events=events_df((1, 1, 3, ts("2024-05-10 13:00"))))
    assert infer_as_of(raw) == ts("2024-05-11")


def test_break_dataset_uses_features_at_the_first_agreement_date_and_drops_unresolved():
    agreements = agreements_df(
        ("a1", 1, 0, 40.0, 40.0, ts("2024-02-01"), ts("2024-02-09"), "PAID"),
        ("a2", 2, 0, 40.0, 0.0, ts("2024-02-10"), NaT, "ACTIVE"),
    )
    raw = make_raw(debts=DEBTS, agreements=agreements)
    prep = prepare_agreements(raw, S)
    data = build_break_dataset(raw, prep, S, ts("2024-09-01"))
    assert data["debtor_id"].tolist() == [1] and data["broke"].tolist() == [0]
    assert data["days_overdue_max"].iloc[0] == (ts("2024-02-01") - ts("2023-01-01")).days
    with pytest.raises(DataError):
        build_break_dataset(make_raw(debts=DEBTS), prepare_agreements(make_raw(debts=DEBTS), S),
                            S, ts("2024-09-01"))
