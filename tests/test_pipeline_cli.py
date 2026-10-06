import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from collection_queue_ranker import synth
from collection_queue_ranker.cli import build_parser, main
from collection_queue_ranker.config import Settings
from collection_queue_ranker.db.sqlite import SqliteSource
from collection_queue_ranker.errors import DataError, PipelineError
from collection_queue_ranker.models import ModelBundle
from collection_queue_ranker.pipeline import read_datasets, retrain, write_datasets

OVERRIDES = {
    "evaluation": {"n_bootstrap": 20},
    "model": {"candidates": [{"max_depth": 3, "num_leaves": 8}]},
    "queue": {"top_n": 400, "collectors": 5},
}


def test_datasets_survive_a_round_trip_through_csv(demo, tmp_path):
    paths = write_datasets(tmp_path, demo["snapshots"], demo["break_df"], demo["as_of"])
    assert [p.name for p in paths] == ["snapshots.csv", "break.csv", "dataset_meta.json"]
    snapshots, break_df, as_of = read_datasets(tmp_path)
    assert as_of == demo["as_of"]
    pd.testing.assert_frame_equal(snapshots, demo["snapshots"], check_dtype=False, atol=1e-9)
    pd.testing.assert_frame_equal(break_df, demo["break_df"], check_dtype=False, atol=1e-9)
    assert snapshots["snapshot_at"].dtype.kind == "M" and snapshots["product_id"].dtype == "string"


def test_missing_dataset_files_point_at_the_command_that_builds_them(tmp_path):
    with pytest.raises(DataError, match="build-dataset"):
        read_datasets(tmp_path)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Two synthetic product lines, a settings file and one full retrain through the CLI."""
    root = tmp_path_factory.mktemp("cli")
    assert main(["synth", "--out", str(root / "demo"), "--debtors", "1500", "--seed", "21"]) == 0
    config = root / "settings.json"
    config.write_text(json.dumps({**OVERRIDES, "tracking": {"uri": str(root / "mlruns")}}),
                      encoding="utf-8")
    out = root / "artifacts"
    common = ["--config", str(config), "--artifacts", str(out), "--backend", "sqlite",
              "--source", f"auto={root / 'demo' / 'auto.db'}",
              "--source", f"consumer={root / 'demo' / 'consumer.db'}"]
    assert main(["retrain", *common]) == 0
    return {"root": root, "out": out, "config": config, "common": common}


def test_retrain_writes_datasets_bundle_reports_queue_and_tracks_the_run(world):
    out = world["out"]
    expected = {"snapshots.csv", "break.csv", "dataset_meta.json", "bundle.joblib", "metrics.json",
                "report.md", "queue"}
    assert expected <= {p.name for p in out.iterdir()}
    assert {p.name for p in (out / "queue").iterdir()} == {
        "queue_full.csv", "queue_operational.csv", "queue_summary.json"}
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=(world["root"] / "mlruns").as_uri())
    experiment = client.get_experiment_by_name("collection-queue-ranker")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1 and "close_auc" in runs[0].data.metrics
    assert runs[0].data.tags["stage"] == "train"


def test_both_product_lines_are_in_the_data_even_though_ids_collide(world):
    snapshots, _, _ = read_datasets(world["out"])
    assert set(snapshots["product_line"]) == {"auto", "consumer"}
    ids = snapshots.groupby("product_line")["debtor_id"].agg(set)
    assert ids["auto"] & ids["consumer"]   # the same debtor_id exists in both lines
    keys = snapshots.groupby(["product_line", "debtor_id"]).size()
    assert len(keys) > len(ids["auto"] | ids["consumer"])   # and they are kept apart
    full = pd.read_csv(world["out"] / "queue" / "queue_full.csv", dtype={"debtor_id": str})
    assert not full.duplicated(["product_line", "debtor_id"]).any()


def test_the_cli_backtest_reproduces_the_holdout_from_the_saved_bundle(world, capsys):
    assert main(["backtest", "--config", str(world["config"]), "--artifacts",
                 str(world["out"])]) == 0
    printed = capsys.readouterr().out
    assert "Ranking under limited capacity" in printed
    saved = json.loads((world["out"] / "metrics.json").read_text(encoding="utf-8"))
    again = json.loads((world["out"] / "backtest" / "metrics.json").read_text(encoding="utf-8"))
    assert again["close_head"]["auc"] == pytest.approx(saved["close_head"]["auc"], abs=1e-9)
    assert again["ranking"]["rows"] == saved["ranking"]["rows"]


def test_the_train_command_refits_from_the_saved_datasets(world, capsys):
    before = ModelBundle.load(world["out"] / "bundle.joblib").meta["created_at"]
    assert main(["train", "--config", str(world["config"]), "--artifacts", str(world["out"]),
                 "--no-tracking"]) == 0
    assert "selected close-head parameters" in capsys.readouterr().out
    assert ModelBundle.load(world["out"] / "bundle.joblib").meta["created_at"] >= before


def test_the_queue_command_rebuilds_the_files_from_the_bundle(world, capsys):
    assert main(["queue", *world["common"]]) == 0
    assert "operational_rows" in capsys.readouterr().out
    operational = pd.read_csv(world["out"] / "queue" / "queue_operational.csv", dtype=str)
    assert len(operational) <= 400 and set(operational["collector"]) <= {"1", "2", "3", "4", "5"}


def test_monitoring_reads_the_delivered_file_and_the_source(world, capsys):
    assert main(["monitor", *world["common"], "--since", "2024-12-01 00:00"]) == 0
    assert "Delivered queue:" in capsys.readouterr().out


def test_diagnose_writes_a_report_with_the_ranking_section(world, capsys):
    assert main(["diagnose", *world["common"]]) == 0
    text = (world["out"] / "diagnostics.md").read_text(encoding="utf-8")
    assert text.startswith("# Funnel diagnostics") and "7. Does the ranking prioritise" in text
    capsys.readouterr()


@pytest.mark.parametrize("name", ["label-target", "unresolved"])
def test_experiments_run_and_print_a_table(world, capsys, name):
    assert main(["experiment", name, *world["common"]]) == 0
    printed = capsys.readouterr().out
    assert ("paid_material" in printed) if name == "label-target" else ("unresolved_excluded" in printed)


def test_check_contract_passes_a_good_source_and_fails_a_bad_one(world, tmp_path, capsys):
    assert main(["check-contract", *world["common"]]) == 0
    assert "[consumer]" in capsys.readouterr().out
    bad = tmp_path / "bad.db"
    with closing(sqlite3.connect(bad)) as conn:
        conn.execute("CREATE TABLE debts (debt_id INTEGER)")
    code = main(["check-contract", "--backend", "sqlite", "--source", f"auto={bad}"])
    assert code == 1 and "FAIL" in capsys.readouterr().out


def test_draft_contact_map_proposes_categories_for_review(world, tmp_path, capsys):
    target = tmp_path / "map.csv"
    assert main(["draft-contact-map", *world["common"], "--out", str(target)]) == 0
    draft = pd.read_csv(target)
    assert {"product_line", "result_id", "label", "category"} <= set(draft.columns)
    assert draft.set_index("label").loc["Promise to pay recorded", "category"].iloc[0] == "promise"
    assert "review the `category` column" in capsys.readouterr().out


# ---- the retrain gate ---------------------------------------------------------------------------


def test_a_failed_stage_stops_the_chain_and_regenerates_nothing(tmp_path):
    bad = tmp_path / "bad.db"
    with closing(sqlite3.connect(bad)) as conn:
        conn.execute("CREATE TABLE debts (debt_id INTEGER)")
    out = tmp_path / "out"
    with pytest.raises(PipelineError) as err:
        retrain([SqliteSource("auto", bad)], Settings(), out)
    assert err.value.stage == "load" and "Later stages were not run" in str(err.value)
    assert not out.exists()   # no dataset, no bundle, no queue from a half-read source


def test_a_dataset_failure_leaves_an_older_queue_untouched(tmp_path):
    frames = synth.generate(80, seed=2, months=3)           # too short for the default horizon
    path = synth.write_sqlite(frames, tmp_path / "short.db")
    out = tmp_path / "out"
    (out / "queue").mkdir(parents=True)
    old = out / "queue" / "queue_operational.csv"
    old.write_text("kept", encoding="utf-8")
    with pytest.raises(PipelineError) as err:
        retrain([SqliteSource("auto", path)], Settings(), out)
    assert err.value.stage == "datasets"
    assert old.read_text(encoding="utf-8") == "kept"
    assert not (out / "bundle.joblib").exists()


def test_the_cli_turns_a_pipeline_failure_into_exit_code_one(tmp_path, capsys):
    bad = tmp_path / "bad.db"
    with closing(sqlite3.connect(bad)) as conn:
        conn.execute("CREATE TABLE debts (debt_id INTEGER)")
    code = main(["retrain", "--backend", "sqlite", "--source", f"auto={bad}",
                 "--artifacts", str(tmp_path / "o"), "--no-tracking"])
    assert code == 1
    assert "error: stage `load` failed" in capsys.readouterr().err


def test_argument_errors_are_reported_without_a_traceback(tmp_path, capsys):
    assert main(["check-contract", "--backend", "sqlite", "--source", "oops"]) == 1
    assert "expected LINE=TARGET" in capsys.readouterr().err
    assert main(["check-contract", "--backend", "sqlite"]) == 1
    assert "no sources configured" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_info:
        main(["not-a-command"])
    assert exit_info.value.code == 2


def test_every_command_is_documented_in_the_parser():
    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "command").choices
    assert set(choices) == {"synth", "check-contract", "draft-contact-map", "build-dataset",
                            "train", "backtest", "queue", "monitor", "diagnose", "experiment",
                            "retrain", "serve"}
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
