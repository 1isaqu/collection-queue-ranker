import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import EvaluationSettings
from collection_queue_ranker.evaluation.backtest import (
    robustness_by_queue,
    run_backtest,
    top_value_per_queue,
    verdict,
)
from collection_queue_ranker.evaluation.metrics import (
    auc,
    auc_with_ci,
    bootstrap_ci,
    calibration_summary,
    cluster_resampler,
    discrimination_verdict,
    gain_table,
    lift_at,
    ndcg_at,
    order_desc,
    precision_at,
    recall_at,
    top_count,
    value_capture,
)

Y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
SCORE = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])


def test_top_count_floors_but_never_below_one():
    assert top_count(10, 0.25) == 2 and top_count(10, 0.01) == 1 and top_count(0, 0.5) == 0


def test_order_is_stable_for_ties():
    assert order_desc(np.array([1.0, 2.0, 2.0, 0.0])).tolist() == [1, 2, 0, 3]


def test_precision_recall_and_lift_by_hand():
    assert precision_at(Y, SCORE, 0.3) == pytest.approx(2 / 3)     # top 3: hits at ranks 1 and 3
    assert recall_at(Y, SCORE, 0.3) == pytest.approx(2 / 3)        # 2 of the 3 positives
    assert lift_at(Y, SCORE, 0.3) == pytest.approx((2 / 3) / 0.3)
    assert np.isnan(recall_at(np.zeros(5), np.arange(5.0), 0.5))
    assert np.isnan(lift_at(np.zeros(5), np.arange(5.0), 0.5))


def test_auc_matches_the_pairwise_definition_and_is_nan_for_one_class():
    pos, neg = SCORE[Y == 1], SCORE[Y == 0]
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    assert auc(Y, SCORE) == pytest.approx(np.mean(pairs))
    assert np.isnan(auc(np.ones(4), np.arange(4.0)))


def test_gain_table_is_cumulative_and_ends_at_one():
    table = gain_table(Y, SCORE, n_bins=5)
    assert table["captured_share"].is_monotonic_increasing
    assert table["captured_share"].iloc[-1] == 1.0 and table["population_share"].iloc[-1] == 1.0
    assert table["positives_captured"].tolist() == [1, 2, 2, 3, 3]
    assert table["lift"].iloc[0] == pytest.approx(0.3333 / 0.2, rel=1e-3)


def test_value_capture_is_the_share_of_value_in_the_worked_rows():
    value = np.array([100.0, 0.0, 50.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert value_capture(value, SCORE, 0.3) == pytest.approx(150 / 200)
    assert np.isnan(value_capture(np.zeros(4), np.arange(4.0), 0.5))


def test_ndcg_is_one_for_a_perfect_order_and_lower_otherwise():
    grades = np.array([3, 0, 1, 0, 2])
    perfect = np.array([5.0, 1.0, 3.0, 0.0, 4.0])
    assert ndcg_at(grades, perfect, 1.0) == pytest.approx(1.0)
    assert ndcg_at(grades, -perfect, 1.0) < 1.0
    assert np.isnan(ndcg_at(np.zeros(3), np.arange(3.0), 1.0))


def test_the_cluster_resampler_keeps_whole_groups_together():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    draw = cluster_resampler(groups, np.random.default_rng(0))
    for _ in range(25):
        rows = draw()
        counts = pd.Series(groups[rows]).value_counts()
        assert all(counts.get(g, 0) % n == 0 for g, n in {"a": 3, "b": 1, "c": 2}.items())
        assert len(rows) in range(3, 19)


def test_bootstrap_interval_is_deterministic_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.3).astype(int)
    score = y * 0.6 + rng.random(400)
    a = auc_with_ci(y, score, n_boot=60, seed=4)
    b = auc_with_ci(y, score, n_boot=60, seed=4)
    assert a == b
    value, low, high = a
    assert low <= value <= high and low > 0.5


def test_resampling_by_group_widens_the_interval_when_rows_are_not_independent():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(40), 10)
    y = np.repeat((rng.random(40) < 0.4).astype(int), 10)       # outcome fixed within a group
    score = y * 0.5 + rng.random(400)
    plain = auc_with_ci(y, score, n_boot=80, seed=1)
    clustered = auc_with_ci(y, score, n_boot=80, seed=1, groups=groups)
    assert (clustered[2] - clustered[1]) > (plain[2] - plain[1])


def test_bootstrap_ignores_degenerate_samples():
    low, high = bootstrap_ci(5, lambda rows: float("nan"), n_boot=5, level=0.9, seed=0)
    assert np.isnan(low) and np.isnan(high)


def test_discrimination_verdict_reads_the_interval_against_chance():
    assert "not distinguishable" in discrimination_verdict(0.53, 0.45, 0.61)
    assert "weak but real" in discrimination_verdict(0.62, 0.55, 0.69)
    assert "worse than random" in discrimination_verdict(0.38, 0.31, 0.45)
    assert "not estimable" in discrimination_verdict(float("nan"), float("nan"), float("nan"))


def test_calibration_summary_reports_means():
    out = calibration_summary(np.array([1, 0, 0, 0]), np.array([0.5, 0.25, 0.25, 0.0]))
    assert out == {"mean_predicted": 0.25, "observed_rate": 0.25}


# ---- backtest ---------------------------------------------------------------------------------


def test_capacity_is_applied_inside_each_queue_then_summed():
    value = np.array([10.0, 1.0, 1.0, 1.0, 100.0, 5.0, 5.0, 5.0])
    score = np.array([4, 3, 2, 1, 4, 3, 2, 1.0])
    queues = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert top_value_per_queue(value, score, queues, 0.5) == 10 + 1 + 100 + 5
    assert top_value_per_queue(value, score, queues, 0.01) == 10 + 100    # at least one per queue
    pooled_top = np.sort(value)[::-1][:4].sum()                           # what pooling would give
    assert pooled_top != top_value_per_queue(value, score, queues, 0.5)
    assert top_value_per_queue(np.array([]), np.array([]), np.array([], dtype=int), 0.5) == 0.0


def _backtest_frame(seed=0, queues=4, per_queue=200):
    rng = np.random.default_rng(seed)
    n = queues * per_queue
    balance = rng.lognormal(7, 0.8, n)
    p = rng.beta(1, 12, n)
    paid = rng.random(n) < p
    frame = pd.DataFrame({
        "snapshot_at": np.repeat(pd.date_range("2024-01-01", periods=queues, freq="MS"), per_queue),
        "product_line": "auto", "debtor_id": np.arange(n),
        "realized_value": np.where(paid, balance * 0.4, 0.0),
    })
    frame["score_largest_balance"] = balance
    frame["score_random"] = rng.random(n)
    frame["score_ev"] = balance * p                 # the generating process: should win
    frame["score_prob"] = p
    return frame


def test_a_strategy_that_knows_the_process_beats_the_balance_rule_and_random():
    frame = _backtest_frame()
    settings = EvaluationSettings(n_bootstrap=20)
    result = run_backtest(frame, (0.1, 0.3), settings)
    table = result.table.set_index(["strategy", "capacity"])
    assert result.queues == 4 and result.rows == 800
    assert table.loc[("largest_balance", 0.1), "uplift_vs_balance_pct"] == 0.0
    assert table.loc[("random", 0.3), "captured_share"] < table.loc[("ev", 0.3), "captured_share"]
    assert result.uplift("ev", 0.3) > 0
    row = table.loc[("ev", 0.3)]
    assert row["ci_low_pct"] <= row["uplift_vs_balance_pct"] <= row["ci_high_pct"]
    assert np.isnan(table.loc[("largest_balance", 0.3), "ci_low_pct"])
    assert np.isnan(result.uplift("missing", 0.3))


def test_bootstrap_can_be_switched_off():
    result = run_backtest(_backtest_frame(), (0.2,), EvaluationSettings(n_bootstrap=0))
    assert result.table["ci_low_pct"].isna().all()


def test_the_verdict_follows_the_configured_thresholds():
    s = EvaluationSettings(uplift_proof_pct=8.0, uplift_floor_pct=3.0)
    assert "evidence of value" in verdict(12.0, s)
    assert "grey zone" in verdict(5.0, s)
    assert "not worth" in verdict(1.0, s)
    assert "not estimable" in verdict(float("nan"), s)


def test_robustness_flags_queues_below_the_alert_level():
    good = pd.DataFrame({"snapshot_at": "2024-01-01", "closed": [1, 0, 1, 0],
                         "p": [0.9, 0.2, 0.8, 0.1]})
    bad = pd.DataFrame({"snapshot_at": "2024-02-01", "closed": [1, 0, 1, 0],
                        "p": [0.1, 0.9, 0.2, 0.8]})
    flat = pd.DataFrame({"snapshot_at": "2024-03-01", "closed": [0, 0, 0, 0], "p": [0.1] * 4})
    out = robustness_by_queue(pd.concat([good, bad, flat]), "p", min_auc=0.6)
    assert out["auc"].iloc[:2].tolist() == [1.0, 0.0] and np.isnan(out["auc"].iloc[2])
    assert out["alert"].tolist() == [False, True, False]  # an undefined AUC is not an alert
