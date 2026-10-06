"""Command line: `cqr <command>`. Run `cqr --help` for the list.

Typical flow on the synthetic demo:

    cqr synth --out data/demo
    cqr retrain --backend sqlite --source auto=data/demo/auto.db --source consumer=data/demo/consumer.db

or step by step: build-dataset -> train -> backtest -> queue -> monitor.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from collection_queue_ranker import __version__
from collection_queue_ranker.config import Settings, load_settings
from collection_queue_ranker.contacts import draft_categories, inconsistent_categories
from collection_queue_ranker.contract import check_source, format_checks
from collection_queue_ranker.db.base import DataSource
from collection_queue_ranker.db.sources import make_sources
from collection_queue_ranker.env import load_env_file
from collection_queue_ranker.errors import ConfigError, CqrError, DataError
from collection_queue_ranker.models import ModelBundle
from collection_queue_ranker.pipeline import (
    build_datasets,
    load_inputs,
    read_datasets,
    retrain,
    summarise,
    write_datasets,
)

Handler = Callable[[argparse.Namespace], int]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=os.environ.get("CQR_CONFIG") or None,
                        help="JSON settings file (see examples/settings.example.json)")
    parser.add_argument("--artifacts", default="artifacts",
                        help="directory for datasets, bundle and reports (default: artifacts)")
    parser.add_argument("--backend", choices=["sqlite", "sqlserver"], default=None)
    parser.add_argument("--source", action="append", default=[], metavar="LINE=TARGET",
                        help="one product line, e.g. auto=data/demo/auto.db (repeatable)")
    parser.add_argument("--as-of", default=None, metavar="YYYY-MM-DD",
                        help="reference date; default: the day after the latest data")
    parser.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    parser.add_argument("-v", "--verbose", action="store_true")


def _settings(args: argparse.Namespace) -> Settings:
    load_env_file(".env")
    return load_settings(args.config or os.environ.get("CQR_CONFIG"))


def _sources(args: argparse.Namespace, settings: Settings) -> list[DataSource]:
    mapping: dict[str, str] = {}
    for item in args.source:
        line, sep, target = item.partition("=")
        if not sep or not line or not target:
            raise ConfigError(f"bad --source {item!r}; expected LINE=TARGET")
        mapping[line] = target
    return make_sources(settings, backend=args.backend, sources=mapping or None)


def _as_of(args: argparse.Namespace) -> pd.Timestamp | None:
    return pd.Timestamp(args.as_of) if args.as_of else None


def _tracker(args: argparse.Namespace, settings: Settings):
    from collection_queue_ranker.tracking import make_tracker

    return make_tracker(settings.tracking, enabled=False if args.no_tracking else None)


def cmd_synth(args: argparse.Namespace) -> int:
    from collection_queue_ranker.synth import build_demo

    paths = build_demo(args.out, args.debtors, args.seed, args.end)
    print("wrote:", ", ".join(str(p) for p in paths.values()))
    print("use:  " + " ".join(f"--source {line}={path}" for line, path in paths.items()))
    return 0


def cmd_check_contract(args: argparse.Namespace) -> int:
    settings = _settings(args)
    failed = False
    for source in _sources(args, settings):
        checks = check_source(source, settings.source.tables)
        print(f"[{source.name}]\n{format_checks(checks)}")
        failed |= not all(c.ok for c in checks)
    return 1 if failed else 0


def cmd_draft_contact_map(args: argparse.Namespace) -> int:
    settings = _settings(args)
    raw, _, _ = load_inputs(_sources(args, settings), settings, _as_of(args))
    proposed = draft_categories(raw.results.drop(columns=["category"]))
    bad = inconsistent_categories(proposed)
    columns = ["product_line", "result_id", "label", "answered", "right_party", "category"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    proposed[columns].to_csv(args.out, index=False)
    print(f"wrote {len(proposed)} draft rows to {args.out}; review the `category` column, then "
          f"load it into contact_results.category.\n{len(bad)} rows contradict their answered flag.")
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    settings = _settings(args)
    raw, prep, as_of = load_inputs(_sources(args, settings), settings, _as_of(args))
    snapshots, break_df = build_datasets(raw, prep, settings, as_of)
    paths = write_datasets(args.artifacts, snapshots, break_df, as_of)
    print(f"as of {as_of.date()}: {len(snapshots)} snapshot rows over "
          f"{snapshots['snapshot_at'].nunique()} cutoffs, {len(break_df)} break rows")
    print("wrote:", ", ".join(str(p) for p in paths))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from collection_queue_ranker.evaluation.report import render_markdown
    from collection_queue_ranker.training import train

    settings = _settings(args)
    snapshots, break_df, as_of = read_datasets(args.artifacts)
    result = train(snapshots, break_df, settings, as_of, tracker=_tracker(args, settings),
                   out_dir=args.artifacts)
    print(render_markdown(result.report))
    print("selected close-head parameters:", result.selected_params)
    print("wrote:", ", ".join(str(p) for p in result.files))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from collection_queue_ranker.evaluation.report import (
        build_report,
        render_markdown,
        write_report,
    )
    from collection_queue_ranker.scoring import scored_frame

    settings = _settings(args)
    bundle = ModelBundle.load(Path(args.artifacts) / "bundle.joblib")
    snapshots, break_df, _ = read_datasets(args.artifacts)
    start = pd.Timestamp(bundle.meta["holdout_start"])
    holdout = scored_frame(snapshots[snapshots["snapshot_at"] >= start].reset_index(drop=True),
                           bundle, settings.model.seed)
    break_hold = break_df[break_df["anchor_at"] >= start].reset_index(drop=True)
    if holdout.empty:
        raise DataError(f"no snapshot on or after the bundle's holdout start {start.date()}")
    report = build_report(holdout, bundle, settings, break_holdout=break_hold)
    write_report(report, Path(args.artifacts) / "backtest")
    print(render_markdown(report))
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    from collection_queue_ranker.worklist.build import build_queue, write_queue

    settings = _settings(args)
    bundle = ModelBundle.load(Path(args.artifacts) / "bundle.joblib")
    raw, prep, as_of = load_inputs(_sources(args, settings), settings, _as_of(args))
    result = build_queue(raw, prep, bundle, settings, as_of)
    paths = write_queue(result, Path(args.artifacts) / "queue", settings.queue.csv_separator)
    print(result.summary)
    print("wrote:", ", ".join(str(p) for p in paths))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from collection_queue_ranker.monitoring import monitor, render_monitor

    settings = _settings(args)
    queue_dir = Path(args.artifacts) / "queue"
    sep = settings.queue.csv_separator
    delivered = pd.read_csv(args.delivered or queue_dir / "queue_operational.csv", sep=sep,
                            dtype=str)
    predicted = pd.read_csv(queue_dir / "queue_full.csv", sep=sep,
                            dtype={"debtor_id": str, "product_line": str})
    raw, prep, _ = load_inputs(_sources(args, settings), settings, _as_of(args))
    since = pd.Timestamp(args.since)
    report = monitor(delivered, predicted, raw, prep, since, settings.queue.cooldown_days)
    print(render_monitor(report))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from collection_queue_ranker.diagnostics import render_diagnostics, run_diagnostics

    settings = _settings(args)
    raw, prep, as_of = load_inputs(_sources(args, settings), settings, _as_of(args))
    full = Path(args.artifacts) / "queue" / "queue_full.csv"
    queue = pd.read_csv(full, sep=settings.queue.csv_separator) if full.is_file() else None
    text = render_diagnostics(run_diagnostics(raw, prep, settings, as_of, queue), as_of)
    out = Path(args.artifacts) / "diagnostics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    from collection_queue_ranker.experiments import (
        break_experiment_frame,
        label_target_experiment,
        unresolved_experiment,
    )
    from collection_queue_ranker.split import time_split

    settings = _settings(args)
    if args.name == "label-target":
        snapshots, _, _ = read_datasets(args.artifacts)
        split = time_split(snapshots, "snapshot_at", settings.split.holdout_fraction,
                           settings.labels.horizon_days, purge=settings.split.purge)
        print(label_target_experiment(split, settings).to_string(index=False))
        return 0
    raw, prep, as_of = load_inputs(_sources(args, settings), settings, _as_of(args))
    frame = break_experiment_frame(raw, prep, settings, as_of)
    print(unresolved_experiment(frame, settings).to_string(index=False))
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    settings = _settings(args)
    result = retrain(_sources(args, settings), settings, args.artifacts, as_of=_as_of(args),
                     tracker=_tracker(args, settings))
    print(summarise(result))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigError(
            "serving needs uvicorn: pip install 'collection-queue-ranker[serve]'"
        ) from exc
    from collection_queue_ranker.serving.app import create_app

    app = create_app(Path(args.artifacts) / "bundle.joblib")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cqr", description="Rank a collections work queue by expected recovered value."
    )
    parser.add_argument("--version", action="version", version=f"cqr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Handler, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        _common(p)
        p.set_defaults(handler=handler)
        return p

    synth = add("synth", cmd_synth, "write a synthetic demo database per product line")
    synth.add_argument("--out", default="data/demo")
    synth.add_argument("--debtors", type=int, default=4000)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--end", default="2024-12-31")
    add("check-contract", cmd_check_contract, "check each source against the data contract")
    draft = add("draft-contact-map", cmd_draft_contact_map,
                "propose contact_results.category from the result labels")
    draft.add_argument("--out", default="artifacts/contact_map_draft.csv")
    add("build-dataset", cmd_build_dataset, "build snapshot and break datasets")
    add("train", cmd_train, "train heads and ranker, report on the holdout")
    add("backtest", cmd_backtest, "re-run the holdout backtest from a saved bundle")
    add("queue", cmd_queue, "build today's work queue from a saved bundle")
    mon = add("monitor", cmd_monitor, "funnel and ordering test for a delivered queue")
    mon.add_argument("--since", required=True, metavar="'YYYY-MM-DD HH:MM'")
    mon.add_argument("--delivered", default=None, help="delivered queue CSV (default: latest)")
    add("diagnose", cmd_diagnose, "funnel diagnostics: can we reach the people ranked first?")
    exp = add("experiment", cmd_experiment, "label experiments judged on a common set")
    exp.add_argument("name", choices=["label-target", "unresolved"])
    add("retrain", cmd_retrain, "load, build, train, judge and queue, stopping at the first failure")
    serve = add("serve", cmd_serve, "serve a saved bundle over HTTP (needs the serve extra)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # joblib probes the physical core count with a Windows command that may not exist; this
    # only silences that probe.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.handler(args)
    except CqrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
