import json

import pytest

from collection_queue_ranker.config import Settings, load_settings, settings_from_dict
from collection_queue_ranker.env import load_env_file, parse_env_text
from collection_queue_ranker.errors import ConfigError


def test_defaults_are_self_consistent():
    s = Settings()
    assert s.labels.horizon_days > 0
    assert s.ranking.tier_p1_share + s.ranking.tier_p2_share < 1
    assert set(s.statuses.valid) == set(s.statuses.paid + s.statuses.broken + s.statuses.active)
    assert s.model.close_params["max_depth"] == 4


def test_partial_override_keeps_other_defaults():
    s = Settings().with_overrides({"labels": {"horizon_days": 30}})
    assert s.labels.horizon_days == 30
    assert s.labels.legacy_tolerance_days == Settings().labels.legacy_tolerance_days


def test_override_merges_free_form_parameter_dicts():
    s = Settings().with_overrides({"model": {"close_params": {"max_depth": 2}}})
    assert s.model.close_params["max_depth"] == 2
    assert s.model.close_params["learning_rate"] == Settings().model.close_params["learning_rate"]


def test_lists_become_tuples_and_ints_become_floats():
    s = settings_from_dict({
        "phones": {"recency_tiers": [[10, 1], [20, 0.5]], "automated_weight": 1},
        "ranking": {"capacities": [0.1, 0.2]},
    })
    assert s.phones.recency_tiers == ((10, 1.0), (20, 0.5))
    assert isinstance(s.phones.automated_weight, float)
    assert s.ranking.capacities == (0.1, 0.2)


def test_unknown_key_is_rejected_with_the_valid_names():
    with pytest.raises(ConfigError) as err:
        settings_from_dict({"labels": {"horizon": 3}})
    assert "horizon" in str(err.value) and "horizon_days" in str(err.value)


def test_wrong_type_is_rejected_with_the_path():
    with pytest.raises(ConfigError) as err:
        settings_from_dict({"labels": {"horizon_days": "sixty"}})
    assert "labels.horizon_days" in str(err.value)


def test_load_settings_from_file_and_errors(tmp_path):
    file = tmp_path / "s.json"
    file.write_text(json.dumps({"queue": {"top_n": 10}}), encoding="utf-8")
    assert load_settings(file).queue.top_n == 10
    assert load_settings(None) == Settings()
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.json")
    file.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_settings(file)


def test_settings_round_trip_through_dict():
    s = Settings().with_overrides({"queue": {"collectors": 3}})
    assert settings_from_dict(s.to_dict()) == s


def test_env_text_parsing_strips_quotes_and_comments():
    text = "# note\nA=1\nB = 'two words'\n\nC=\"3\"\nbroken line\n"
    assert parse_env_text(text) == {"A": "1", "B": "two words", "C": "3"}


def test_env_file_never_overrides_existing_variables(tmp_path, monkeypatch):
    file = tmp_path / ".env"
    file.write_text("CQR_TEST_ONE=from-file\nCQR_TEST_TWO=from-file\n", encoding="utf-8")
    monkeypatch.setenv("CQR_TEST_ONE", "from-shell")
    monkeypatch.delenv("CQR_TEST_TWO", raising=False)
    applied = load_env_file(file)
    assert applied == {"CQR_TEST_TWO": "from-file"}
    monkeypatch.delenv("CQR_TEST_TWO", raising=False)
    assert load_env_file(tmp_path / "absent.env") == {}
