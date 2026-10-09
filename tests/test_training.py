import json

import numpy as np
import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import ConfigError, DataError
from collection_queue_ranker.evaluation.report import (
    jsonable,
    render_markdown,
    write_report,
)
from collection_queue_ranker.training import (
    check_inputs,
    select_close_params,
    train,
)


class RecordingTracker:
    """Stands in for MLflow: records what training asks it to log."""

    run_id = "fake"

    def __init__(self):
        self.runs, self.metrics, self.artifacts = [], {}, []

    def run(self, name, params=None, tags=None):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            self.runs.append((name, params, tags))
            yield self

        return ctx()

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_artifact(self, path, subdir=None):
        self.artifacts.append((str(path), subdir))


def test_training_splits_at_a_date_boundary_and_reports_the_purge(trained, demo):
    result, _ = trained
    split = result.split
    assert split.dev["snapshot_at"].max() < split.holdout["snapshot_at"].min()
    horizon = pd.Timedelta(days=Settings().labels.horizon_days)
    assert (split.dev["snapshot_at"] + horizon <= split.holdout_start).all()
    assert len(split.dev) + len(split.holdout) + split.purged == len(demo["snapshots"])
    # the heads share one boundary, so the break head never learns from holdout-period outcomes
    assert result.break_split.holdout_start == split.holdout_start
    assert (result.break_split.dev["anchor_at"] < split.holdout_start).all()


def test_the_report_has_the_expected_sections_and_sane_numbers(trained):
    result, _ = trained
    report = result.report
    close = report["close_head"]
    assert close["rows"] == len(result.split.holdout) and 0 < close["base_rate"] < 1
    assert close["auc"] > 0.5 and close["auc_ci"][0] <= close["auc"] <= close["auc_ci"][1]
    assert set(close["precision_at"]) == {"0.05", "0.1", "0.2"}
    assert close["gain_table"]["captured_share"].is_monotonic_increasing
    rank = report["ranking"]
    assert rank["queues"] == result.split.holdout["snapshot_at"].nunique()
    assert {"ev_contact", "lambdarank", "largest_balance", "random"} <= set(
        rank["table"]["strategy"])
    assert "break_head" in report and report["robustness"]["alert"].dtype == bool
    assert report["data"]["dev_rows"] == len(result.split.dev)
    low, high = rank["uplift_interval_pct"]
    assert low <= rank["uplift_vs_balance_pct"] <= high


def test_the_headline_verdict_cannot_claim_evidence_when_the_interval_includes_zero(trained):
    rank = trained[0].report["ranking"]
    low = rank["uplift_interval_pct"][0]
    if low <= 0:
        assert "evidence of value" not in rank["verdict"]
    headline = next(line for line in render_markdown(trained[0].report).splitlines()
                    if line.startswith("- headline capacity"))
    assert "(interval " in headline and rank["verdict"] in headline


def test_the_model_beats_chance_and_the_demo_world_has_signal(trained):
    result, _ = trained
    table = result.report["ranking"]["table"].set_index(["strategy", "capacity"])
    # a world with real signal: value-aware strategies capture more than random at 30% capacity
    assert table.loc[("ev", 0.3), "captured_share"] > table.loc[("random", 0.3), "captured_share"]


def test_training_writes_the_bundle_and_every_report_file(trained):
    result, out = trained
    names = {p.name for p in out.iterdir()}
    assert {"bundle.joblib", "metrics.json", "report.md", "backtest.csv", "gain_table.csv",
            "robustness.csv", "selection.csv", "importance_close.csv", "importance_break.csv",
            "importance_ranker.csv"} <= names
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["close_head"]["auc"] == pytest.approx(result.report["close_head"]["auc"])
    assert "Holdout evaluation" in (out / "report.md").read_text(encoding="utf-8")


def test_selection_compares_the_configured_candidates_on_development_rows_only(trained, demo_settings):
    result, _ = trained
    table = result.selection
    assert len(table) == len(demo_settings.model.candidates) and table["folds"].min() >= 1
    assert result.selected_params["max_depth"] in {3, 5}
    best = table.sort_values("value_capture", ascending=False).iloc[0]
    assert result.selected_params == best["params"]


def test_selection_by_auc_is_available_and_validated(trained, demo_settings):
    result, _ = trained
    by_auc = demo_settings.with_overrides({"model": {"selection_metric": "auc"}})
    params, table = select_close_params(result.split.dev, by_auc)
    assert params == table.sort_values("auc", ascending=False).iloc[0]["params"]
    with pytest.raises(ConfigError, match="selection_metric"):
        select_close_params(result.split.dev, demo_settings.with_overrides(
            {"model": {"selection_metric": "luck"}}))


def test_without_candidates_the_default_parameters_are_used(trained, demo_settings):
    result, _ = trained
    plain = demo_settings.with_overrides({"model": {"candidates": []}})
    params, table = select_close_params(result.split.dev, plain)
    assert params == plain.model.close_params and len(table) == 1


def test_tracking_receives_metrics_params_and_the_report_files(demo, demo_settings, tmp_path):
    tracker = RecordingTracker()
    no_ranker = demo_settings.with_overrides({"model": {"train_ranker": False}})
    # what `cqr retrain` leaves in the directory: datasets, and a queue folder from an earlier run
    (tmp_path / "snapshots.csv").write_text("debtor_id\n", encoding="utf-8")
    (tmp_path / "queue").mkdir()
    (tmp_path / "queue" / "queue_full.csv").write_text("debtor_id\n", encoding="utf-8")
    result = train(demo["snapshots"], demo["break_df"], no_ranker, demo["as_of"],
                   tracker=tracker, out_dir=tmp_path)
    expected = {"metrics.json", "report.md", "backtest.csv", "gain_table.csv", "robustness.csv",
                "selection.csv", "importance_close.csv", "importance_break.csv"}
    assert sorted(p.name for p in result.files) == sorted({*expected, "bundle.joblib"})
    assert result.bundle.ranker is None
    name, params, tags = tracker.runs[0]
    assert name == "train" and tags["strategy"] == "ev_contact"
    assert params["settings"]["labels"]["horizon_days"] == no_ranker.labels.horizon_days
    assert {"close_auc", "uplift_vs_balance_pct", "break_auc"} <= set(tracker.metrics)
    logged = [path.split("\\")[-1].split("/")[-1] for path, _ in tracker.artifacts]
    assert sorted(logged) == sorted(expected)
    assert "lambdarank" not in set(result.report["ranking"]["table"]["strategy"])


def test_training_without_an_output_directory_writes_nothing(demo, demo_settings, tmp_path):
    result = train(demo["snapshots"], demo["break_df"], demo_settings, demo["as_of"])
    assert result.files == []


def test_inputs_with_a_constant_label_are_refused_before_any_fit(demo):
    snapshots, break_df = demo["snapshots"], demo["break_df"]
    with pytest.raises(DataError, match="close label"):
        check_inputs(snapshots.assign(closed=0), break_df)
    with pytest.raises(DataError, match="break-cohort"):
        check_inputs(snapshots, break_df.assign(broke=1))
    check_inputs(snapshots, break_df)


def test_the_report_renders_and_serialises(trained, tmp_path):
    result, _ = trained
    text = render_markdown(result.report)
    assert "## Close head" in text and "## Ranking under limited capacity" in text
    assert "ALERT" in text or "| 20" in text
    files = write_report(result.report, tmp_path / "again")
    assert len(files) == 5
    payload = jsonable({"a": np.float64("nan"), "b": np.int64(3), "t": pd.Timestamp("2024-01-02"),
                        "c": [np.bool_(True), float("inf")]})
    assert payload == {"a": None, "b": 3, "t": "2024-01-02T00:00:00", "c": [True, None]}
