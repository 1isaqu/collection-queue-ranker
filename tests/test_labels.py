import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.labels import (
    agreements_before,
    break_cohort,
    prepare_agreements,
    valid_installments,
    window_labels,
)
from world import NaT, agreements_df, debts_df, make_raw, ts

S = Settings()
DEBTS = debts_df((1, 10, ts("2024-01-01"), ts("2023-06-01"), 100.0),
                 (2, 11, ts("2024-01-01"), ts("2023-06-01"), 100.0))


def _prep(agreements, links=None, settings=S):
    raw = make_raw(settings, debts=DEBTS, agreements=agreements,
                   **({"links": links} if links is not None else {}))
    return raw, prepare_agreements(raw, settings)


def test_only_valid_statuses_count_as_agreements():
    rows = agreements_df(
        ("a1", 10, 0, 50.0, 0.0, ts("2024-02-01"), NaT, "PENDING_EXCEPTION"),
        ("a2", 11, 0, 50.0, 0.0, ts("2024-02-01"), NaT, "REJECTED"),
        ("a3", 10, 0, 50.0, 0.0, ts("2024-02-05"), NaT, "ACTIVE"),
    )
    raw, prep = _prep(rows)
    assert prep.first["agreement_id"].tolist() == ["a3"]
    assert valid_installments(raw, S.statuses)["agreement_id"].tolist() == ["a3"]


def test_the_latest_rewrite_of_an_installment_wins():
    rows = agreements_df(
        ("a1", 10, 0, 50.0, 0.0, ts("2024-02-01"), NaT, "ACTIVE"),
        ("a1", 10, 0, 50.0, 50.0, ts("2024-02-03"), ts("2024-02-20"), "PAID"),
    )
    _, prep = _prep(rows)
    assert prep.first["status"].tolist() == ["PAID"] and prep.first["outcome"].tolist() == ["paid"]
    assert len(prep.installments) == 1


def test_a_later_out_of_scope_rewrite_cannot_hide_a_valid_row():
    # filter on status first, then keep the latest: the PENDING re-write is simply not a candidate
    rows = agreements_df(
        ("a1", 10, 0, 50.0, 0.0, ts("2024-02-01"), NaT, "ACTIVE"),
        ("a1", 10, 0, 50.0, 0.0, ts("2024-02-09"), NaT, "PENDING_EXCEPTION"),
    )
    _, prep = _prep(rows)
    assert prep.first["status"].tolist() == ["ACTIVE"]


def test_only_the_first_installment_defines_the_agreement_but_all_are_kept_for_payments():
    rows = agreements_df(
        ("a1", 10, 0, 50.0, 50.0, ts("2024-02-01"), ts("2024-02-10"), "PAID"),
        ("a1", 10, 1, 50.0, 50.0, ts("2024-02-01"), ts("2024-03-10"), "PAID"),
    )
    _, prep = _prep(rows)
    assert len(prep.first) == 1 and len(prep.installments) == 2
    assert (prep.installments["first_issued_at"] == ts("2024-02-01")).all()


def test_outcome_mapping_follows_the_configured_labels():
    rows = agreements_df(
        ("p", 10, 0, 1.0, 1.0, ts("2024-02-01"), ts("2024-02-02"), "PAID_MANUAL"),
        ("b", 10, 0, 1.0, 0.0, ts("2024-02-01"), NaT, "AUTO_BROKEN"),
        ("a", 11, 0, 1.0, 0.0, ts("2024-02-01"), NaT, "ACTIVE"),
    )
    _, prep = _prep(rows)
    assert prep.first.set_index("agreement_id")["outcome"].to_dict() == {
        "p": "paid", "b": "broken", "a": "active"}


def test_agreements_far_older_than_the_debt_are_legacy_and_dropped():
    rows = agreements_df(
        ("old", 10, 0, 50.0, 50.0, ts("2023-01-01"), ts("2023-01-15"), "PAID"),
        ("near", 11, 0, 50.0, 0.0, ts("2023-12-01"), NaT, "ACTIVE"),
    )
    links = pd.DataFrame({"agreement_id": ["old", "near"], "portfolio_id": 1, "debt_id": [1, 2]})
    _, prep = _prep(rows, links)
    assert prep.first["agreement_id"].tolist() == ["near"]  # 31 days before entry: inside tolerance


def test_an_agreement_without_a_debt_link_cannot_be_judged_and_is_kept():
    rows = agreements_df(("old", 10, 0, 50.0, 50.0, ts("2020-01-01"), ts("2020-01-15"), "PAID"))
    _, prep = _prep(rows)
    assert prep.first["agreement_id"].tolist() == ["old"]


def test_legacy_tolerance_is_configurable():
    rows = agreements_df(("a", 10, 0, 50.0, 0.0, ts("2023-12-01"), NaT, "ACTIVE"))
    links = pd.DataFrame({"agreement_id": ["a"], "portfolio_id": 1, "debt_id": [1]})
    strict = S.with_overrides({"labels": {"legacy_tolerance_days": 10}})
    assert _prep(rows, links, strict)[1].first.empty
    assert len(_prep(rows, links)[1].first) == 1


def test_window_labels_count_agreements_started_and_money_received_inside_the_window():
    rows = agreements_df(
        ("a1", 10, 0, 40.0, 40.0, ts("2024-03-10"), ts("2024-03-15"), "PAID"),
        ("a1", 10, 1, 40.0, 40.0, ts("2024-03-10"), ts("2024-04-15"), "PAID"),
        ("a1", 10, 2, 40.0, 40.0, ts("2024-03-10"), ts("2024-07-15"), "PAID"),  # after the window
        ("a2", 11, 0, 40.0, 0.0, ts("2024-06-20"), NaT, "ACTIVE"),               # after the window
    )
    _, prep = _prep(rows)
    out = window_labels(prep, ts("2024-03-01"), 60).set_index("debtor_id")
    assert out.loc[10, "closed"] == 1 and out.loc[10, "realized_value"] == 80.0
    assert 11 not in out.index


def test_window_boundaries_are_half_open():
    rows = agreements_df(
        ("on_start", 10, 0, 40.0, 0.0, ts("2024-03-01"), NaT, "ACTIVE"),
        ("on_end", 11, 0, 40.0, 0.0, ts("2024-04-30"), NaT, "ACTIVE"),
    )
    _, prep = _prep(rows)
    out = window_labels(prep, ts("2024-03-01"), 60)
    assert out["debtor_id"].tolist() == [10]  # cutoff + 60 days is excluded, the cutoff included


def test_agreements_before_a_cutoff_exclude_the_cutoff_itself():
    rows = agreements_df(("a1", 10, 0, 40.0, 0.0, ts("2024-03-01"), NaT, "ACTIVE"))
    _, prep = _prep(rows)
    assert agreements_before(prep, ts("2024-03-01")).empty
    assert agreements_before(prep, ts("2024-03-02"))["debtor_id"].tolist() == [10]


def test_break_cohort_flags_resolved_mature_and_the_failure_label():
    rows = agreements_df(
        ("paid", 10, 0, 40.0, 40.0, ts("2024-02-01"), ts("2024-02-05"), "PAID"),
        ("broke", 11, 0, 40.0, 0.0, ts("2024-02-01"), NaT, "BROKEN"),
        ("act", 12, 0, 40.0, 0.0, ts("2024-02-01"), NaT, "ACTIVE"),
        ("late", 13, 0, 40.0, 0.0, ts("2024-05-20"), NaT, "BROKEN"),
    )
    _, prep = _prep(rows)
    cohort = break_cohort(prep, ts("2024-06-01"), S).set_index("debtor_id")
    assert cohort.loc[10, ["resolved", "mature", "broke"]].tolist() == [True, True, 0]
    assert cohort.loc[11, ["resolved", "mature", "broke"]].tolist() == [True, True, 1]
    assert cohort.loc[12, "resolved"] == False  # noqa: E712 - still active: right-censored
    assert cohort.loc[13, "mature"] == False  # noqa: E712 - too young to trust its outcome


def test_a_debtor_with_one_paid_and_one_broken_agreement_is_not_a_failure():
    rows = agreements_df(
        ("a", 10, 0, 40.0, 40.0, ts("2024-01-10"), ts("2024-01-12"), "PAID"),
        ("b", 10, 0, 40.0, 0.0, ts("2024-02-10"), NaT, "BROKEN"),
    )
    _, prep = _prep(rows)
    row = break_cohort(prep, ts("2024-09-01"), S).iloc[0]
    assert row["broke"] == 0 and row["anchor_at"] == ts("2024-01-10")
