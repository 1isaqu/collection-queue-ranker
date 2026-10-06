import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.diagnostics import (
    answer_rate_by_month,
    concentration,
    contact_lift,
    dialled_below_best,
    model_vs_phone_matrix,
    phone_choice_report,
    render_diagnostics,
    run_diagnostics,
    stage_losses,
)
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.experiments import (
    break_experiment_frame,
    label_target_experiment,
    unresolved_experiment,
)
from collection_queue_ranker.features.phones import phone_quality
from collection_queue_ranker.labels import prepare_agreements
from world import agreements_df, events_df, make_raw, phones_df, results_df, ts

AS_OF = ts("2024-06-01")
RESULTS = results_df().assign(category=["no_answer", "wrong_party", "answered"])


def _world():
    phones = phones_df((1, 1, 1), (2, 1, 1), (3, 2, 0), (4, 2, 1))
    events = events_df(
        (1, 1, 3, ts("2024-04-02"), 1, 2), (2, 1, 1, ts("2024-04-20"), 1, 2),
        (3, 1, 1, ts("2024-05-03"), 1, 1), (4, 2, 1, ts("2024-05-04"), 1, 3),
        (5, 2, 1, ts("2024-05-05"), 1, 3), (6, 2, 2, ts("2024-05-09"), 1, 4),
        (7, 3, 3, ts("2024-05-11"), 1, None),
    )
    agreements = agreements_df(("a1", 1, 0, 100.0, 100.0, ts("2024-04-04"), ts("2024-04-20"), "PAID"))
    raw = make_raw(events=events, phones=phones, agreements=agreements, results=RESULTS)
    return raw, prepare_agreements(raw, Settings())


def test_answer_rate_by_month_counts_voice_attempts_only():
    raw, _ = _world()
    table = answer_rate_by_month(raw.events).set_index("month")
    assert table.loc["2024-04", "voice_attempts"] == 2 and table.loc["2024-04", "answered"] == 1
    assert table.loc["2024-05", "answer_rate"] == pytest.approx(1 / 5, abs=1e-3)
    assert answer_rate_by_month(raw.events.iloc[0:0]).empty


def test_stage_losses_show_each_step_of_the_funnel():
    raw, prep = _world()
    text = stage_losses(raw.events, prep)
    assert "dialled by voice: 3" in text and "answered: 2" in text and "agreement: 1" in text
    assert "agreement paid: 1" in text and "step conversion 66.7%" in text


def test_contact_lift_warns_that_it_is_correlational():
    raw, prep = _world()
    text = contact_lift(raw.events, prep)
    assert "correlational" in text and "answered at least once: 2" in text
    assert "no voice attempts" in contact_lift(raw.events.iloc[0:0], prep)


def test_phone_choice_report_compares_the_first_listed_phone_with_the_best():
    raw, _ = _world()
    quality = phone_quality(raw.phones, raw.events, AS_OF, Settings().phones)
    text = phone_choice_report(quality, kill_count=3)
    assert "Debtors with phones: 2" in text and "a better phone existed" in text
    assert phone_choice_report(quality.iloc[0:0], 3) == "no phones"


def test_calls_on_worse_phones_are_counted_with_a_circularity_caveat():
    raw, _ = _world()
    quality = phone_quality(raw.phones, raw.events, AS_OF, Settings().phones)
    text = dialled_below_best(raw.events, quality, margin=0.0)
    assert "circular" in text and "Attempts with a known phone: 6" in text
    assert dialled_below_best(raw.events.iloc[0:0], quality) == "no dialled phones"


def test_concentration_needs_enough_debtors_and_otherwise_breaks_down_by_size_and_age():
    raw, _ = _world()
    quality = phone_quality(raw.phones, raw.events, AS_OF, Settings().phones)
    assert concentration(quality, raw.debts, AS_OF) in ("no phones", "too few debtors for a breakdown")
    many = pd.DataFrame({"product_line": "auto", "debtor_id": range(40), "phone_id": range(40),
                         "score": np.linspace(0, 60, 40), "created_at": ts("2022-01-01")})
    debts = pd.DataFrame({"product_line": "auto", "debtor_id": range(40),
                          "balance": np.arange(1, 41) * 10.0})
    text = concentration(many, debts, AS_OF)
    assert "by debt_quartile" in text and "by phone_age" in text


def test_the_matrix_flags_top_ranked_debtors_with_weak_phones():
    queue = pd.DataFrame({"ranking_score": np.arange(50.0),
                          "best_phone_score": np.where(np.arange(50) > 40, 5.0, 40.0)})
    text = model_vs_phone_matrix(queue)
    assert "Q5_high" in text and "high score, bad phone" in text
    assert "share of the top quintile" in text


def test_the_full_diagnostic_report_runs_on_the_synthetic_world(demo, demo_settings, tmp_path):
    sections = run_diagnostics(demo["raw"], demo["prep"], demo_settings, demo["as_of"])
    assert [t.split(".")[0] for t, _ in sections] == ["1", "2", "3", "4", "5", "6"]
    text = render_diagnostics(sections, demo["as_of"])
    assert text.startswith("# Funnel diagnostics") and "correlational" in text


def test_the_matrix_section_is_added_when_a_queue_is_given(demo, demo_settings, trained):
    from collection_queue_ranker.worklist.build import build_queue

    result, _ = trained
    queue = build_queue(demo["raw"], demo["prep"], result.bundle, demo_settings, demo["as_of"]).full
    sections = run_diagnostics(demo["raw"], demo["prep"], demo_settings, demo["as_of"], queue)
    assert sections[-1][0].startswith("7.") and "Ranking quintile" in sections[-1][1]


# ---- experiments --------------------------------------------------------------------------------


def test_label_target_experiment_judges_every_target_by_captured_value(trained, demo_settings):
    result, _ = trained
    table = label_target_experiment(result.split, demo_settings)
    assert set(table["target"]) == {"closed", "paid_any", "paid_material"}
    assert (table["train_positive_rate"] > 0).all()
    shares = [c for c in table.columns if c.startswith("value_share_at_")]
    assert len(shares) == len(demo_settings.ranking.capacities)
    assert table[shares].apply(lambda col: col.between(0, 1)).all().all()
    # a wider capacity can only capture more value
    ordered = table[shares].to_numpy()
    assert (np.diff(ordered, axis=1) >= -1e-9).all()


def test_label_target_experiment_needs_a_target_with_both_classes(trained, demo_settings):
    from collection_queue_ranker.split import TimeSplit

    result, _ = trained
    flat = result.split.dev.assign(closed=0, realized_value=0.0)
    with pytest.raises(DataError, match="both classes"):
        label_target_experiment(
            TimeSplit(flat, result.split.holdout, result.split.holdout_start, 0), demo_settings)


def test_unresolved_agreements_get_three_treatments_judged_on_resolved_rows(demo, demo_settings):
    frame = break_experiment_frame(demo["raw"], demo["prep"], demo_settings, demo["as_of"])
    assert {"resolved", "n_paid", "n_active", "broke", "anchor_at"} <= set(frame.columns)
    table = unresolved_experiment(frame, demo_settings).set_index("variant")
    assert set(table.index) == {"unresolved_as_failure", "unresolved_excluded",
                                "unresolved_as_success"}
    assert table["judge_rows"].nunique() == 1                        # one common judge set
    assert table.loc["unresolved_excluded", "train_rows"] <= table.loc[
        "unresolved_as_failure", "train_rows"]
    assert table["auc_on_resolved"].between(0, 1).all()
    assert table.loc["unresolved_as_failure", "train_failure_rate"] >= table.loc[
        "unresolved_as_success", "train_failure_rate"]


def test_the_break_experiment_needs_mature_agreements():
    raw = make_raw()
    with pytest.raises(DataError, match="no mature agreements"):
        break_experiment_frame(raw, prepare_agreements(raw, Settings()), Settings(), AS_OF)
