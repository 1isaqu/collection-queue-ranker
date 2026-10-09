import os

import pytest

from collection_queue_ranker.config import TrackingSettings
from collection_queue_ranker.tracking import (
    MlflowTracker,
    NullTracker,
    flatten,
    make_tracker,
    tracking_uri,
)


def test_flatten_turns_nested_settings_into_short_strings():
    flat = flatten({"a": {"b": 1, "c": {"d": (1, 2)}}, "long": "x" * 400})
    assert flat["a.b"] == "1" and flat["a.c.d"] == "(1, 2)"
    assert len(flat["long"]) == 250


def test_null_tracker_has_the_full_interface_and_does_nothing(tmp_path):
    tracker = NullTracker()
    with tracker.run("r", {"a": 1}, {"t": "x"}) as active:
        active.log_metrics({"m": 1.0})
        active.log_artifact(tmp_path / "nothing")
    assert tracker.run_id is None


def test_directory_uris_become_file_uris_and_schemes_pass_through(tmp_path):
    assert tracking_uri(str(tmp_path / "runs")).startswith("file:")
    assert tracking_uri("sqlite:///x.db") == "sqlite:///x.db"
    assert tracking_uri("http://tracking.example.invalid") == "http://tracking.example.invalid"


def test_make_tracker_honours_the_flag_over_the_setting():
    on = TrackingSettings(enabled=True)
    assert isinstance(make_tracker(on, enabled=False), NullTracker)
    assert isinstance(make_tracker(TrackingSettings(enabled=False)), NullTracker)


def _clear_opt_in(monkeypatch):
    """Start from a clean environment and have pytest restore it afterwards."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "placeholder")
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE")


def test_a_real_run_lands_in_a_local_store(tmp_path, monkeypatch):
    _clear_opt_in(monkeypatch)
    store = tmp_path / "mlruns"
    tracker = make_tracker(TrackingSettings(uri=str(store), experiment="unit-test"))
    assert isinstance(tracker, MlflowTracker)
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"   # MLflow 3 needs this for a directory
    report = tmp_path / "report.md"
    report.write_text("hello", encoding="utf-8")
    with tracker.run("r1", {"model": {"depth": 3}}, {"stage": "test"}) as active:
        active.log_metrics({"auc": 0.7, "bad": float("nan"), "none": None})
        active.log_artifact(report, "reports")
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracker.uri)
    run = client.get_run(tracker.run_id)
    assert run.data.metrics == {"auc": 0.7}                # NaN and None are dropped, not logged
    assert run.data.params == {"model.depth": "3"} and run.data.tags["stage"] == "test"
    assert [a.path for a in client.list_artifacts(tracker.run_id, "reports")] == ["reports/report.md"]
    assert store.exists() and mlflow.active_run() is None   # the run was closed


def test_a_database_uri_is_left_alone(tmp_path, monkeypatch):
    _clear_opt_in(monkeypatch)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MlflowTracker(TrackingSettings(uri=uri, experiment="db"))
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ
    with tracker.run("r") as active:
        active.log_metrics({"x": 1.0})
    assert tracker.run_id


def test_an_existing_file_store_opt_in_is_not_overridden(tmp_path, monkeypatch):
    from mlflow.exceptions import MlflowException

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    with pytest.raises(MlflowException):
        tracker = MlflowTracker(TrackingSettings(uri=str(tmp_path / "r"), experiment="x"))
        with tracker.run("r"):
            pass
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"
