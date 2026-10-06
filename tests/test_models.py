import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.encoding import UNKNOWN, CategoryEncoder, FeatureSpace
from collection_queue_ranker.errors import ConfigError, DataError
from collection_queue_ranker.models import (
    ClassifierHead,
    ModelBundle,
    RankerHead,
    break_space,
    close_space,
    lgbm_params,
    relevance_grades,
)
from collection_queue_ranker.scoring import (
    STRATEGIES,
    expected_value,
    require_strategy,
    scored_frame,
    strategy_scores,
)

SMALL = {"n_estimators": 30, "num_leaves": 4, "max_depth": 3, "min_child_samples": 5,
         "learning_rate": 0.2}


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    kind = rng.choice(["a", "b", "c"], n)
    p = 1 / (1 + np.exp(-(1.5 * x + (kind == "a") - 1.0)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x": x, "kind": kind}), y


def test_encoder_learns_categories_on_fit_and_flags_the_unseen():
    enc = CategoryEncoder(["k"]).fit(pd.DataFrame({"k": ["b", "a", "b", None]}))
    assert enc.categories == {"k": ["a", "b"]}
    out = enc.transform(pd.DataFrame({"k": ["a", "b", "zzz", None]}))
    assert out["k"].tolist() == [0, 1, UNKNOWN, UNKNOWN]
    assert CategoryEncoder.from_dict(enc.to_dict()).categories == enc.categories
    with pytest.raises(DataError, match="before fit"):
        CategoryEncoder(["k"]).transform(pd.DataFrame({"k": ["a"]}))


def test_numeric_ids_and_text_ids_encode_the_same_way():
    enc = CategoryEncoder(["k"]).fit(pd.DataFrame({"k": [10, 20]}))
    assert enc.transform(pd.DataFrame({"k": ["20", "10"]}))["k"].tolist() == [1, 0]


def test_feature_space_orders_columns_and_reports_what_is_missing():
    frame, _ = _toy(20)
    space = FeatureSpace(["x"], ["kind"]).fit(frame)
    assert space.matrix(frame).columns.tolist() == ["x", "kind"]
    with pytest.raises(DataError, match="kind"):
        space.matrix(frame[["x"]])
    with pytest.raises(DataError, match="before fit"):
        FeatureSpace(["x"], ["kind"]).matrix(frame)


def test_a_head_learns_a_signal_and_survives_unseen_categories():
    frame, y = _toy()
    head = ClassifierHead("toy", FeatureSpace(["x"], ["kind"]), SMALL, seed=1).fit(frame, y)
    proba = head.predict(frame)
    assert proba.min() >= 0 and proba.max() <= 1
    assert proba[y == 1].mean() > proba[y == 0].mean() + 0.1
    novel = frame.head(5).assign(kind="never seen before")
    assert np.isfinite(head.predict(novel)).all()
    importance = head.importance()
    assert importance.iloc[0]["feature"] == "x" and importance["gain_share"].sum() == pytest.approx(1)


def test_a_head_is_reproducible_for_a_fixed_seed():
    frame, y = _toy()
    space = lambda: FeatureSpace(["x"], ["kind"])  # noqa: E731
    a = ClassifierHead("t", space(), SMALL, seed=3).fit(frame, y).predict(frame)
    b = ClassifierHead("t", space(), SMALL, seed=3).fit(frame, y).predict(frame)
    assert np.array_equal(a, b)


def test_heads_refuse_one_class_and_unfitted_use():
    frame, _ = _toy(30)
    with pytest.raises(DataError, match="single class"):
        ClassifierHead("t", FeatureSpace(["x"], ["kind"]), SMALL, 0).fit(frame, np.zeros(30))
    head = ClassifierHead("t", FeatureSpace(["x"], ["kind"]), SMALL, 0)
    with pytest.raises(DataError, match="not fitted"):
        head.predict(frame)
    with pytest.raises(DataError, match="not fitted"):
        head.importance()


def test_reproducibility_parameters_are_added_to_user_parameters():
    params = lgbm_params({"max_depth": 3}, seed=9)
    assert params["max_depth"] == 3 and params["random_state"] == 9
    assert params["deterministic"] is True and params["verbose"] == -1


def test_the_break_head_cannot_see_contact_response_features():
    s = Settings()
    assert not set(break_space(s).names) & {"n_answered", "answer_rate", "days_since_answered"}
    assert {"n_answered", "answer_rate"} <= set(close_space(s).names)
    extra = s.with_overrides({"features": {"extra_numeric": ["external_score"],
                                           "extra_categorical": ["external_rating"]}})
    assert "external_score" in close_space(extra).numeric
    assert "external_rating" in close_space(extra).categorical


# ---- relevance grades and the ranker ------------------------------------------------------------


def test_value_grades_split_positive_values_into_tiers_and_zero_stays_zero():
    frame = pd.DataFrame({"closed": [0, 1, 1, 1, 1, 1, 1], "realized_value":
                          [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    grades, edges = relevance_grades(frame, "value_grades", 4)
    assert grades[0] == 0 and grades.max() == 3 and len(edges) == 2
    assert np.all(np.diff(grades[1:]) >= 0)                         # more value, never a lower grade
    again, same = relevance_grades(frame.assign(realized_value=[0, 1, 25, 25, 25, 25, 1e6]),
                                   "value_grades", 4, edges)
    assert np.array_equal(same, edges) and again[-1] == 3 and again[1] == 1


def test_binary_relevance_and_unknown_modes():
    frame = pd.DataFrame({"closed": [0, 1], "realized_value": [0.0, 5.0]})
    assert relevance_grades(frame, "closed", 4)[0].tolist() == [0, 1]
    with pytest.raises(DataError, match="unknown relevance"):
        relevance_grades(frame, "bogus", 4)


def test_the_ranker_orders_better_rows_first_within_each_queue():
    rng = np.random.default_rng(1)
    n = 600
    frame = pd.DataFrame({"x": rng.normal(size=n), "kind": rng.choice(["a", "b"], n),
                          "snapshot_at": np.repeat(pd.date_range("2024-01-01", periods=6, freq="MS"),
                                                   n // 6)})
    value = np.where(rng.random(n) < 1 / (1 + np.exp(-(2 * frame["x"] - 2))), 100.0, 0.0)
    frame["realized_value"], frame["closed"] = value, (value > 0).astype(int)
    ranker = RankerHead(FeatureSpace(["x"], ["kind"]), SMALL, seed=2, n_grades=3).fit(frame)
    score = ranker.predict(frame)
    assert pd.Series(score).corr(frame["x"], method="spearman") > 0.5
    assert ranker.importance().iloc[0]["feature"] == "x"
    with pytest.raises(DataError, match="no row has a positive"):
        RankerHead(FeatureSpace(["x"], ["kind"]), SMALL, 0).fit(
            frame.assign(realized_value=0.0, closed=0))
    with pytest.raises(DataError, match="not fitted"):
        RankerHead(FeatureSpace(["x"], ["kind"]), SMALL, 0).predict(frame)


# ---- scoring and the bundle ---------------------------------------------------------------------


def test_expected_value_is_balance_times_close_times_survival():
    ev = expected_value(np.array([1000.0, 200.0]), np.array([0.1, 0.5]), np.array([0.4, 0.0]))
    assert ev.tolist() == pytest.approx([60.0, 100.0])


def test_unknown_strategies_are_rejected_with_the_choices():
    assert require_strategy("ev") == "ev"
    with pytest.raises(ConfigError) as err:
        require_strategy("magic")
    assert "ev_contact" in str(err.value)


def test_a_trained_bundle_scores_every_strategy(trained, demo):
    result, _ = trained
    holdout = result.split.holdout.head(50)
    scores = strategy_scores(holdout, result.bundle)
    assert set(scores) == set(STRATEGIES)
    assert (scores["ev_contact"] <= scores["ev"] + 1e-9).all()      # the factor is at most 1
    assert np.array_equal(scores["largest_balance"], holdout["balance"].to_numpy())
    assert not np.array_equal(scores["random"], strategy_scores(holdout, result.bundle, 1)["random"])
    frame = scored_frame(holdout, result.bundle)
    assert {"p_close", "p_break", "score_ev", "score_lambdarank"} <= set(frame.columns)
    assert frame["p_close"].between(0, 1).all()
    assert holdout.shape[1] < frame.shape[1]


def test_an_external_score_column_adds_a_baseline_strategy(trained):
    result, _ = trained
    holdout = result.split.holdout.head(30).assign(external_score=np.arange(30.0))
    scores = strategy_scores(holdout, result.bundle)
    assert "external_score" in scores and scores["external_score"][-1] == 29.0


def test_bundle_round_trips_and_rejects_foreign_files(trained, tmp_path):
    result, out = trained
    loaded = ModelBundle.load(out / "bundle.joblib")
    holdout = result.split.holdout.head(40)
    assert np.allclose(loaded.close.predict(holdout), result.bundle.close.predict(holdout))
    assert loaded.meta["holdout_start"] == result.bundle.meta["holdout_start"]
    assert "settings" in loaded.meta and "created_at" in loaded.meta
    with pytest.raises(DataError, match="not found"):
        ModelBundle.load(tmp_path / "absent.joblib")
    import joblib

    joblib.dump({"something": "else"}, tmp_path / "other.joblib")
    with pytest.raises(DataError, match="not a bundle"):
        ModelBundle.load(tmp_path / "other.joblib")
