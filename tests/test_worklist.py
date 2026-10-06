import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import ConfigError, DataError
from collection_queue_ranker.worklist.assign import assign_round_robin
from collection_queue_ranker.worklist.build import (
    COLD,
    NEVER,
    NO_CONTACT,
    RECENT,
    assign_tiers,
    build_queue,
    contact_status,
    last_human_attempt,
    write_queue,
)
from world import events_df, make_raw, ts

AS_OF = ts("2024-06-01")
AGENTS = pd.DataFrame({"agent_id": [1, 2], "login": ["agent_01", "autodialer_01"]})


def test_round_robin_deals_like_cards_not_blocks():
    collector, order = assign_round_robin(10, 3)
    assert collector.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    assert order.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    # every collector's list is spread over the whole ranking, not a contiguous slice of it
    ranks = np.arange(10)
    for c in (1, 2, 3):
        mine = ranks[collector == c]
        assert mine.min() < 3 and mine.max() >= 7
    with pytest.raises(ValueError):
        assign_round_robin(5, 0)
    assert assign_round_robin(0, 4)[0].size == 0


def test_tiers_are_rank_shares_of_the_workable_queue():
    score = pd.Series(np.arange(20.0))
    tiers = assign_tiers(score, 0.15, 0.25)
    assert tiers.value_counts().to_dict() == {"P1": 3, "P2": 5, "P3": 12}
    assert tiers[score.idxmax()] == "P1" and tiers[score.idxmin()] == "P3"


def test_tied_scores_still_split_exactly_by_share():
    tiers = assign_tiers(pd.Series([1.0] * 10), 0.2, 0.3)
    assert tiers.value_counts().to_dict() == {"P1": 2, "P2": 3, "P3": 5}


def test_cooldown_counts_human_attempts_only():
    raw = make_raw(
        events=events_df(
            (1, 1, 1, ts("2024-05-25"), 1),   # debtor 1: a human called 7 days ago -> recent
            (2, 2, 1, ts("2024-05-30"), 2),   # debtor 2: only the dialer touched it -> never
            (3, 3, 1, ts("2024-03-01"), 1),   # debtor 3: a human, long ago -> cold
        ), agents=AGENTS)
    frame = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2, 3, 4]})
    out = contact_status(frame, raw.events, AS_OF, cooldown_days=21).set_index("debtor_id")
    assert out["contact_status"].to_dict() == {1: RECENT, 2: NEVER, 3: COLD, 4: NEVER}
    assert out.loc[1, "days_since_contact"] == 7 and pd.isna(out.loc[2, "days_since_contact"])


def test_attempts_at_or_after_the_reference_date_are_ignored():
    raw = make_raw(events=events_df((1, 1, 1, AS_OF, 1), (2, 1, 1, ts("2024-07-01"), 1)),
                   agents=AGENTS)
    assert last_human_attempt(raw.events, AS_OF).empty


def test_a_missing_cooldown_history_means_everyone_is_never_contacted():
    frame = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2]})
    out = contact_status(frame, make_raw().events, AS_OF, 21)
    assert set(out["contact_status"]) == {NEVER}


@pytest.fixture(scope="module")
def queue(demo, trained):
    result, _ = trained
    settings = Settings().with_overrides({"queue": {"top_n": 300, "collectors": 4}})
    return build_queue(demo["raw"], demo["prep"], result.bundle, settings, demo["as_of"]), settings


def test_the_queue_has_one_row_per_debtor_ranked_best_workable_first(queue):
    result, _ = queue
    full = result.full
    assert not full.duplicated(["product_line", "debtor_id"]).any()
    assert full["rank"].tolist() == list(range(1, len(full) + 1))
    workable = full[full["contactable"]]
    assert workable["ranking_score"].is_monotonic_decreasing
    first_dead = full["contactable"].tolist().index(False) if (~full["contactable"]).any() else None
    if first_dead is not None:
        assert not full["contactable"].iloc[first_dead:].any()    # unreachable debtors go last
        assert set(full.loc[~full["contactable"], "tier"]) == {NO_CONTACT}


def test_tier_shares_follow_the_settings_among_workable_debtors(queue):
    result, settings = queue
    workable = result.full[result.full["contactable"]]
    share = workable["tier"].value_counts(normalize=True)
    assert share["P1"] == pytest.approx(settings.ranking.tier_p1_share, abs=0.01)
    assert share["P2"] == pytest.approx(settings.ranking.tier_p2_share, abs=0.01)


def test_debtors_inside_the_cooldown_are_not_queued(queue, demo):
    result, _ = queue
    assert set(result.full["contact_status"]) <= {NEVER, COLD}
    assert result.summary["excluded_recent_human_attempt"] >= 0
    assert result.summary["active_base"] + result.summary["excluded_recent_human_attempt"] == (
        result.summary["open_population"])
    human = demo["raw"].events[~demo["raw"].events["is_automated"]]
    recent = human[human["occurred_at"] >= demo["as_of"] - pd.Timedelta(days=20)]
    hot = set(zip(recent["product_line"], recent["debtor_id"], strict=True))
    queued = set(zip(result.full["product_line"], result.full["debtor_id"], strict=True))
    assert not hot & queued


def test_debtors_who_already_hold_an_agreement_are_not_queued(queue, demo):
    result, _ = queue
    signed = demo["prep"].first[demo["prep"].first["issued_at"] < demo["as_of"]]
    held = set(zip(signed["product_line"], signed["debtor_id"], strict=True))
    queued = set(zip(result.full["product_line"], result.full["debtor_id"], strict=True))
    assert not held & queued


def test_the_operational_file_has_no_scores_and_only_opaque_ids(queue):
    result, settings = queue
    op = result.operational
    assert len(op) <= settings.queue.top_n
    forbidden = {"ranking_score", "expected_value", "adjusted_value", "p_close", "p_break",
                 "tier", "rank", "contact_factor", "best_phone_score"}
    assert not forbidden & set(op.columns)
    assert op.columns.tolist()[:4] == ["collector", "order", "product_line", "debtor_id"]
    for col in ("debtor_id", "phone_1_id", "phone_2_id", "phone_3_id"):
        assert op[col].map(type).eq(str).all()
        assert not op[col].str.contains(r"\.0$").any() and not op[col].str.contains("e+").any()
    assert op["phone_1_id"].ne("").all()   # contactable means at least one usable phone
    assert not any("name" in c or "document" in c for c in op.columns)


def test_collectors_get_ordered_lists_drawn_from_the_whole_ranking(queue):
    result, settings = queue
    op = result.operational
    assert set(op["collector"]) == set(range(1, settings.queue.collectors + 1))
    for _, mine in op.groupby("collector"):
        assert mine["order"].tolist() == list(range(1, len(mine) + 1))
    sizes = op["collector"].value_counts()
    assert sizes.max() - sizes.min() <= 1
    # collector 1 and collector 4 both hold rows from the top of the ranking
    keyed = op.reset_index().rename(columns={"index": "position"})
    firsts = keyed.groupby("collector")["position"].min()
    assert firsts.max() < settings.queue.collectors


def test_the_contact_factor_stays_inside_its_band(queue):
    result, settings = queue
    factor = result.full["contact_factor"]
    assert factor.min() >= settings.queue.contact_floor - 1e-9 and factor.max() <= 1.0 + 1e-9
    assert result.summary["contact_factor"]["min"] == pytest.approx(factor.min())


def test_the_queue_can_be_ordered_by_the_ranker_instead(demo, trained):
    result, _ = trained
    settings = Settings().with_overrides({"ranking": {"strategy": "lambdarank"}})
    ranked = build_queue(demo["raw"], demo["prep"], result.bundle, settings, demo["as_of"]).full
    workable = ranked[ranked["contactable"]]
    assert workable["ranking_score"].is_monotonic_decreasing
    assert np.allclose(workable["ranking_score"], workable["score_lambdarank"])


def test_unknown_or_unavailable_strategies_fail_clearly(demo, trained):
    result, _ = trained
    bad = Settings().with_overrides({"ranking": {"strategy": "magic"}})
    with pytest.raises(ConfigError):
        build_queue(demo["raw"], demo["prep"], result.bundle, bad, demo["as_of"])
    import copy

    bundle = copy.copy(result.bundle)
    bundle.ranker = None
    ranker = Settings().with_overrides({"ranking": {"strategy": "lambdarank"}})
    with pytest.raises(DataError, match="does not contain"):
        build_queue(demo["raw"], demo["prep"], bundle, ranker, demo["as_of"])


def test_no_open_debtors_is_an_error_not_an_empty_file(demo, trained):
    result, _ = trained
    with pytest.raises(DataError, match="no debtor with an open balance"):
        build_queue(demo["raw"], demo["prep"], result.bundle, Settings(), pd.Timestamp("2001-01-01"))


def test_the_queue_is_written_as_three_files(queue, tmp_path):
    result, _ = queue
    files = write_queue(result, tmp_path / "q", separator=";")
    assert [f.name for f in files] == ["queue_full.csv", "queue_operational.csv",
                                       "queue_summary.json"]
    back = pd.read_csv(files[1], sep=";", dtype=str)
    assert len(back) == len(result.operational)
    assert back["debtor_id"].tolist() == result.operational["debtor_id"].tolist()
