import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import PhoneSettings, QueueSettings, Settings
from collection_queue_ranker.features.phones import (
    attach_contact_factor,
    contact_factor,
    phone_quality,
    top_phones_wide,
)
from world import NaT, events_df, make_raw, phones_df, ts

AS_OF = ts("2024-06-01")
P = PhoneSettings(smoothing_strength=2.0, recency_tiers=((100, 1.0), (300, 0.5)),
                  recency_floor=0.25, automated_weight=0.5, invalid_kill_count=2,
                  prior_mobile=0.5, prior_messaging=0.0, prior_priority=0.0, prior_fresh=0.0)
AGENTS = pd.DataFrame({"agent_id": [1, 2], "login": ["agent_01", "autodialer_01"]})
RESULTS = pd.DataFrame({
    "result_id": [1, 2, 3, 4], "label": "x", "answered": [0, 1, 0, 0], "right_party": [0, 1, 0, 0],
    "category": ["no_answer", "answered", "invalid_number", "message"]})


def _quality(events, phones, settings=P):
    raw = make_raw(events=events, phones=phones, agents=AGENTS, results=RESULTS)
    return phone_quality(raw.phones, raw.events, AS_OF, settings).set_index("phone_id")


def _dial(n, result, phone, day="2024-05-20", agent=1, start=0):
    return [(start + i, 1, result, ts(day), agent, phone) for i in range(n)]


def test_a_dialled_phone_scores_the_smoothed_answer_rate_times_recency():
    # phone 1: 4 human attempts, 2 answered -> 0.5 raw; phone 2: 4 attempts, none answered
    events = events_df(*_dial(2, 2, 1), *_dial(2, 1, 1, start=10), *_dial(4, 1, 2, start=20))
    q = _quality(events, phones_df((1, 1), (2, 1)))
    base_rate = 2 / 8
    assert q.loc[1, "score"] == pytest.approx(100 * (2 + 2 * base_rate) / (4 + 2), abs=0.06)
    assert q.loc[2, "score"] == pytest.approx(100 * (0 + 2 * base_rate) / (4 + 2), abs=0.06)
    assert q.loc[1, "score"] > q.loc[2, "score"]


def test_few_attempts_are_trusted_less_than_many():
    # phones 1 and 2 both answer every time; phone 3 drags the global rate down
    events = events_df(*_dial(1, 2, 1), *_dial(10, 2, 2, start=10), *_dial(40, 1, 3, start=100))
    q = _quality(events, phones_df((1, 1), (2, 1), (3, 1)))
    assert q.loc[2, "score"] > q.loc[1, "score"] > q.loc[3, "score"]


def test_automated_dialer_attempts_count_with_a_reduced_weight():
    human = _quality(events_df(*_dial(4, 1, 1), *_dial(1, 2, 1, start=9)), phones_df((1, 1)))
    robot = _quality(events_df(*_dial(4, 1, 1, agent=2), *_dial(1, 2, 1, start=9)),
                     phones_df((1, 1)))
    assert robot.loc[1, "attempts"] == pytest.approx(4 * 0.5 + 1)
    assert robot.loc[1, "score"] > human.loc[1, "score"]  # the dialer's misses weigh less


def test_messages_are_not_attempts_to_reach_a_human():
    q = _quality(events_df(*_dial(5, 4, 1)), phones_df((1, 1, 1, 0, 0, 0, NaT, NaT)))
    assert q.loc[1, "attempts"] == 0  # falls to the prior, not to a zero-rate history


def test_recency_decays_in_tiers_and_floors():
    def score(day):
        events = events_df(*_dial(2, 2, 1, day=day), *_dial(2, 1, 1, day=day, start=9))
        return _quality(events, phones_df((1, 1))).loc[1, "score"]

    fresh, mid, old = score("2024-05-20"), score("2024-02-01"), score("2022-01-01")
    assert fresh == pytest.approx(2 * mid, rel=0.01) and old == pytest.approx(fresh / 4, rel=0.01)


def test_a_dead_number_never_answered_is_killed():
    events = events_df(*_dial(2, 3, 1), *_dial(3, 1, 2), *_dial(2, 3, 2, start=30),
                       *_dial(1, 2, 2, start=40))
    q = _quality(events, phones_df((1, 1), (2, 1)))
    assert q.loc[1, "score"] == 0.0          # two dead-number results, no answer: condemned
    assert q.loc[2, "score"] > 0.0           # same dead results but it has answered once


def test_a_phone_never_dialled_gets_a_prior_and_invalid_ones_get_zero():
    phones = phones_df((1, 1, 1, 1, 0, 0, NaT, NaT), (2, 1, 1, 0, 0, 0, NaT, NaT),
                       (3, 1, 0, 1, 0, 0, NaT, NaT))
    q = _quality(events_df(*_dial(4, 1, 9), *_dial(1, 2, 9, start=9)), phones)
    assert q.loc[3, "score"] == 0.0
    assert q.loc[1, "score"] == pytest.approx(q.loc[2, "score"] * 1.5, abs=0.2)  # mobile boost


def test_quality_is_point_in_time():
    events = events_df(*_dial(3, 2, 1, day="2024-05-20"), *_dial(3, 1, 1, day="2024-07-01", start=9))
    raw = make_raw(events=events, phones=phones_df((1, 1, 1, 0, 0, 0, ts("2024-07-15"), NaT),
                                                   (2, 1)), agents=AGENTS, results=RESULTS)
    q = phone_quality(raw.phones, raw.events, AS_OF, P).set_index("phone_id")
    assert 1 not in q.index                      # created after the as-of date
    assert q.loc[2, "voice_human"] == 0           # nothing known about phone 2 yet
    later = phone_quality(raw.phones, raw.events, ts("2024-08-01"), P).set_index("phone_id")
    assert later.loc[1, "voice_human"] == 6


def test_contact_factor_combines_the_best_phones_into_a_reach_probability():
    quality = pd.DataFrame({
        "product_line": "auto", "debtor_id": [1, 1, 1, 1, 2], "phone_id": [1, 2, 3, 4, 5],
        "score": [50.0, 50.0, 10.0, 90.0, 0.0], "is_mobile": 0, "updated_at": NaT})
    phones, queue = PhoneSettings(top_phones=2), QueueSettings(contact_floor=0.4)
    out = contact_factor(quality, phones, queue).set_index("debtor_id")
    reach = 1 - (1 - 0.9) * (1 - 0.5)   # the two best phones: 90 and 50
    assert out.loc[1, "contact_factor"] == pytest.approx(0.4 + 0.6 * reach)
    assert out.loc[1, "usable_phones"] == 4 and out.loc[1, "best_phone_score"] == 90.0
    assert 2 not in out.index              # a zero-score phone is not usable


def test_debtors_without_a_usable_phone_get_the_floor_not_zero():
    frame = pd.DataFrame({"product_line": "auto", "debtor_id": [1, 2]})
    quality = pd.DataFrame({"product_line": "auto", "debtor_id": [1], "phone_id": [1],
                            "score": [40.0], "is_mobile": 0, "updated_at": NaT})
    out = attach_contact_factor(frame, quality, PhoneSettings(), QueueSettings(contact_floor=0.3))
    assert out.set_index("debtor_id")["usable_phones"].to_dict() == {1: 1, 2: 0}
    assert out.set_index("debtor_id").loc[2, "contact_factor"] == 0.3
    empty = attach_contact_factor(frame, quality.iloc[0:0], PhoneSettings(),
                                  QueueSettings(contact_floor=0.3))
    assert (empty["contact_factor"] == 0.3).all() and (empty["usable_phones"] == 0).all()


def test_top_phones_are_ranked_with_ties_broken_by_mobile_then_recency():
    quality = pd.DataFrame({
        "product_line": "auto", "debtor_id": 1, "phone_id": [10, 11, 12, 13],
        "score": [30.0, 30.0, 80.0, 0.0], "is_mobile": [0, 1, 0, 1],
        "updated_at": ts("2024-01-01")})
    wide = top_phones_wide(quality, 3).iloc[0]
    assert [wide["phone_1_id"], wide["phone_2_id"], wide["phone_3_id"]] == [12, 11, 10]
    assert wide["phone_1_score"] == 80.0
    only_one = top_phones_wide(quality[quality["phone_id"] == 12], 3).iloc[0]
    assert np.isnan(only_one["phone_2_id"])
    assert top_phones_wide(quality.iloc[0:0], 3).empty


def test_default_settings_run_on_empty_inputs():
    raw = make_raw()
    assert phone_quality(raw.phones, raw.events, AS_OF, Settings().phones).empty
