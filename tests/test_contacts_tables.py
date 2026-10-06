import pandas as pd
import pytest

from collection_queue_ranker.config import AgentSettings
from collection_queue_ranker.contacts import (
    ANSWERED,
    CATEGORIES,
    VOICE,
    annotate_events,
    automated_agents,
    derive_category,
    draft_categories,
    draft_category,
    inconsistent_categories,
    like_to_regex,
    matches_any,
    resolve_categories,
)
from collection_queue_ranker.contract import CONTRACT, check_table
from collection_queue_ranker.errors import ContractError, DataError
from collection_queue_ranker.tables import RawTables, normalise_frame
from world import NaT, debts_df, events_df, make_raw, results_df, ts


def test_vocabulary_is_closed_and_consistent():
    assert set(VOICE) | {"message", "other"} == set(CATEGORIES)
    assert ANSWERED <= VOICE


def test_category_is_derived_from_the_two_flags():
    assert derive_category(0, 0) == "no_answer"
    assert derive_category(1, 0) == "wrong_party"
    assert derive_category(1, 1) == "answered"
    out = resolve_categories(results_df().assign(product_line="auto"))
    assert out.set_index("result_id")["category"].to_dict() == {
        1: "no_answer", 2: "wrong_party", 3: "answered"}


def test_an_explicit_category_wins_and_is_validated():
    frame = results_df().assign(category=[None, "voicemail", "promise"])
    out = resolve_categories(frame).set_index("result_id")["category"]
    assert out.to_dict() == {1: "no_answer", 2: "voicemail", 3: "promise"}
    with pytest.raises(ContractError, match="unknown contact category"):
        resolve_categories(frame.assign(category=["bogus", None, None]))


def test_inconsistent_categories_flag_contradictions_with_the_answered_flag():
    frame = results_df().assign(category=["answered", "no_answer", "promise"])
    bad = inconsistent_categories(frame)
    assert sorted(bad["result_id"]) == [1, 2]


@pytest.mark.parametrize("label,expected", [
    ("Automated message sent", "message"),
    ("SMS dispatched", "message"),
    ("Promise to pay recorded", "promise"),
    ("Wrong party answered", "wrong_party"),
    ("Number disconnected", "invalid_number"),
    ("Left on voicemail", "voicemail"),
    ("Line busy", "busy"),
    ("No answer after ringing", "no_answer"),
    ("Spoke with debtor", "answered"),
    ("Café résumé", "other"),
    (None, "other"),
])
def test_draft_rules_are_ordered_and_tolerant(label, expected):
    assert draft_category(label) == expected


def test_draft_categories_adds_a_column_without_touching_input():
    frame = results_df()
    out = draft_categories(frame)
    assert "category" not in frame and out["category"].tolist() == ["no_answer", "wrong_party",
                                                                      "answered"]


def test_like_patterns_are_case_insensitive_and_anchored():
    assert like_to_regex("auto%").match("AutoDialer_01")
    assert not like_to_regex("auto%").match("my_auto")
    assert like_to_regex("a_c").match("abc") and not like_to_regex("a_c").match("abbc")
    assert like_to_regex("a.c").match("a.c") and not like_to_regex("a.c").match("abc")
    assert matches_any("system_02", ["x%", "system%"]) and not matches_any(None, ["%"])


def test_automated_agents_are_matched_on_login():
    agents = pd.DataFrame({"product_line": "auto", "agent_id": [1, 2, 3],
                           "login": ["agent_01", "autodialer_01", "system_9"]})
    found = automated_agents(agents, AgentSettings())
    assert sorted(found["agent_id"]) == [2, 3]
    assert automated_agents(agents.iloc[0:0], AgentSettings()).empty


def test_events_get_category_and_automation_flag():
    agents = pd.DataFrame({"agent_id": [1, 2], "login": ["agent_01", "autodialer_01"]})
    events = events_df((1, 10, 3, ts("2024-01-01"), 1), (2, 10, 1, ts("2024-01-02"), 2),
                       (3, 10, 99, ts("2024-01-03"), None))
    raw = make_raw(events=events, agents=agents)
    e = raw.events.set_index("event_id")
    assert e["category"].to_dict() == {1: "answered", 2: "no_answer", 3: "other"}
    assert e["is_automated"].to_dict() == {1: False, 2: True, 3: False}


def test_events_without_any_agents_table_are_all_human():
    raw = make_raw(events=events_df((1, 10, 3, ts("2024-01-01"), 1)))
    assert raw.events["is_automated"].tolist() == [False]
    assert annotate_events(raw.events.iloc[0:0], raw.results, raw.agents,
                           AgentSettings()).empty


def test_normalise_adds_optional_columns_and_parses_types():
    spec = CONTRACT["debtor_phones"]
    raw = pd.DataFrame({"PHONE_ID": [1, 2], "Debtor_ID": [5, 5], "is_mobile": [None, 1],
                        "created_at": ["2024-01-05", "bad"]})
    out = normalise_frame(spec, raw)
    assert out["product_line"].tolist() == ["auto", "auto"]
    assert out["is_valid"].tolist() == [1, 1]  # NULL validity means valid
    assert out["is_mobile"].tolist() == [0, 1]
    assert out["created_at"].iloc[0] == ts("2024-01-05") and pd.isna(out["created_at"].iloc[1])
    assert out["updated_at"].isna().all()


def test_normalise_keeps_unknown_columns_after_contract_ones():
    spec = CONTRACT["debtors"]
    out = normalise_frame(spec, pd.DataFrame({"debtor_id": [1], "extra_flag": [7]}))
    assert out.columns.tolist() == ["product_line", "debtor_id", "is_company", "extra_flag"]


def test_raw_tables_fill_row_ids_and_report_the_latest_timestamp():
    debts = debts_df((1, 10, ts("2024-01-01"), ts("2023-06-01"), 100.0))
    raw = RawTables.from_frames({"debts": debts})
    assert raw.agreements.empty and raw.events.empty and raw.product_lines == ["auto"]
    assert raw.latest_timestamp() == ts("2024-01-01")
    with pytest.raises(DataError):
        RawTables.from_frames({}).latest_timestamp()


def test_row_ids_default_to_input_order():
    from world import agreements_df

    rows = agreements_df(("a", 1, 0, 10.0, 0.0, ts("2024-01-01"), NaT, "ACTIVE"))
    raw = RawTables.from_frames({"agreements": rows.drop(columns=["row_id"])})
    assert raw.agreements["row_id"].tolist() == [0]


def test_check_table_distinguishes_required_from_optional():
    ok = check_table(CONTRACT["agents"], ["agent_id", "login"])
    assert ok.ok and ok.missing_optional == ("display_name",)
    bad = check_table(CONTRACT["agents"], ["agent_id"])
    assert not bad.ok and bad.missing_required == ("login",)
    assert not check_table(CONTRACT["debts"], []).ok
    assert check_table(CONTRACT["debtors"], []).ok  # optional table
