import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.labels import prepare_agreements
from collection_queue_ranker.monitoring import (
    MIN_ATTEMPTED,
    agreements_since,
    by_predicted_band,
    category_counts,
    cold_baseline,
    delivered_overlap,
    funnel,
    last_attempt_before,
    monitor,
    render_monitor,
)
from collection_queue_ranker.util import debtor_key, text_id
from world import agreements_df, events_df, make_raw, ts

SINCE = ts("2024-06-01")


def _delivered(n):
    return pd.DataFrame({"product_line": "auto", "debtor_id": [str(i) for i in range(1, n + 1)],
                         "collector": 1, "order": range(1, n + 1)})


def _world(n=40):
    """Odd ids are attempted in the window; every third of those answers; ids 9k sign."""
    rows, eid = [], 0
    for d in range(1, n + 1):
        if d % 2:
            eid += 1
            rows.append((eid, d, 3 if d % 3 == 0 else 1, ts("2024-06-02"), 1))
            rows.append((1000 + d, d, 1, ts("2024-01-05"), 1))   # an older attempt: cold at SINCE
    agreements = [(f"a{d}", d, 0, 100.0, 0.0, ts("2024-06-10"), None, "ACTIVE")
                  for d in range(1, n + 1) if d % 9 == 0]
    return make_raw(events=events_df(*rows), agreements=agreements_df(*agreements))


def test_ids_become_text_without_decimals_or_scientific_notation():
    assert text_id(12.0) == "12" and text_id(12) == "12" and text_id("0012") == "0012"
    assert text_id(np.nan) == "" and text_id(None) == "" and text_id(1.5) == "1.5"
    frame = pd.DataFrame({"product_line": ["auto", "auto"], "debtor_id": [7, 7.0]})
    assert debtor_key(frame).tolist() == ["auto|7", "auto|7"]


def test_category_counts_pivot_the_window_by_debtor():
    raw = _world()
    counts = category_counts(raw.events, SINCE).set_index("key")
    assert counts.loc["auto|1", "attempts"] == 1 and counts.loc["auto|3", "answered"] == 1
    assert category_counts(raw.events, ts("2024-07-01")).empty


def test_agreements_since_sum_the_installment_amounts():
    rows = agreements_df(("x", 1, 0, 100.0, 0.0, ts("2024-06-10"), None, "ACTIVE"),
                         ("x", 1, 1, 100.0, 0.0, ts("2024-06-10"), None, "ACTIVE"),
                         ("y", 2, 0, 50.0, 0.0, ts("2024-05-01"), None, "ACTIVE"))
    raw = make_raw(agreements=rows)
    out = agreements_since(prepare_agreements(raw, Settings()), SINCE)
    assert out["agreement_id"].tolist() == ["x"] and out["agreed_value"].tolist() == [200.0]


def test_funnel_counts_only_debtors_on_the_delivered_queue():
    raw = _world()
    prep = prepare_agreements(raw, Settings())
    counts = category_counts(raw.events, SINCE)
    deals = agreements_since(prep, SINCE)
    keys = set(debtor_key(_delivered(40)))
    result = funnel(keys, counts, deals)
    assert result["delivered"] == 40 and result["attempted"] == 20
    assert result["answered"] == 7           # odd multiples of three up to 40: 3, 9, 15, 21, 27, 33, 39
    assert result["agreements"] == 4 and result["agreed_value"] == 400.0
    narrow = funnel({"auto|9", "auto|1"}, counts, deals)
    assert (narrow["attempted"], narrow["answered"], narrow["agreements"]) == (2, 1, 1)


def test_the_ordering_test_compares_bands_not_levels():
    n = 40
    delivered = _delivered(n).assign(score=np.arange(n, dtype=float))
    # answers and agreements are concentrated among the highest scores
    answered = {d for d in range(1, n + 1) if d > 30}
    rows = [(i, d, 3 if d in answered else 1, ts("2024-06-02"), 1)
            for i, d in enumerate(range(1, n + 1), 1)]
    counts = category_counts(make_raw(events=events_df(*rows)).events, SINCE)
    deals = pd.DataFrame({"key": [f"auto|{d}" for d in range(33, 41)],
                          "agreement_id": [f"a{d}" for d in range(33, 41)], "agreed_value": 1.0})
    table = by_predicted_band(delivered, counts, deals, "score", bands=4).set_index("band")
    assert table["answered"].tolist() == [0, 0, 0, 10]
    assert table["agreed"].tolist() == [0, 0, 0, 8]
    assert table.loc["Q4", "answer_rate_of_attempted"] == 1.0


def test_the_baseline_uses_debtors_equally_cold_at_the_start_who_were_not_listed():
    counts = pd.DataFrame({"key": ["auto|1", "auto|2", "auto|3", "auto|4"],
                           "attempts": 1, "answered": [1, 0, 1, 1]})
    last = pd.DataFrame({"key": ["auto|2", "auto|3", "auto|4"],
                         "last_attempt": [ts("2024-01-01"), ts("2024-05-28"), ts("2024-01-01")]})
    deals = pd.DataFrame({"key": ["auto|4"], "agreement_id": ["z"], "agreed_value": 1.0})
    out = cold_baseline({"auto|1"}, counts, deals, last, SINCE, cooldown_days=21)
    # debtor 1 is on the list, debtor 3 was not cold (4 days), debtors 2 and 4 qualify
    assert out == {"n": 2, "answer_rate": 0.5, "agreement_rate": 0.5}
    assert cold_baseline({"auto|2", "auto|4"}, counts, deals, last, SINCE, 21)["n"] == 0
    assert cold_baseline(set(), counts.iloc[0:0], deals, last, SINCE, 21)["n"] == 0


def test_last_attempt_before_the_window_per_debtor():
    raw = make_raw(events=events_df((1, 1, 1, ts("2024-04-01")), (2, 1, 1, ts("2024-05-01")),
                                    (3, 1, 1, ts("2024-06-05"))))
    out = last_attempt_before(raw.events, SINCE)
    assert out["last_attempt"].tolist() == [ts("2024-05-01")]
    assert last_attempt_before(raw.events.iloc[0:0], SINCE).empty


def test_a_monitor_with_too_little_volume_refuses_to_quote_rates():
    raw = _world(10)
    prep = prepare_agreements(raw, Settings())
    predicted = _delivered(10).assign(best_phone_score=1.0, ranking_score=1.0)
    report = monitor(_delivered(10), predicted, raw, prep, SINCE, 21)
    assert not report.enough_volume and report.bands == {}
    text = render_monitor(report)
    assert "Not enough volume" in text and str(MIN_ATTEMPTED) in text


def test_a_monitor_with_volume_reports_bands_and_a_baseline():
    raw = _world(80)
    prep = prepare_agreements(raw, Settings())
    delivered = _delivered(60)
    predicted = _delivered(60).assign(best_phone_score=np.arange(60.0),
                                      ranking_score=np.arange(60.0)[::-1])
    report = monitor(delivered, predicted, raw, prep, SINCE, 21)
    assert report.enough_volume and set(report.bands) == {"best_phone_score", "ranking_score"}
    assert report.baseline["n"] > 0
    text = render_monitor(report)
    assert "Ordering test by `best_phone_score`" in text and "Cold baseline" in text
    assert "comparison between bands" in text


def test_overlap_between_two_deliveries_is_a_share_of_the_first():
    a, b = _delivered(10), _delivered(10).iloc[:4]
    assert delivered_overlap(a, b) == 0.4
    assert np.isnan(delivered_overlap(a.iloc[0:0], b))
