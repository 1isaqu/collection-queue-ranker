"""Dataset files and the gated retrain pipeline.

Retraining is an ordered chain: load -> datasets -> train (which also writes the holdout report)
-> queue. The order matters and so does the gate after each step. The original retrain script had two
bugs that this structure rules out: it forgot to refit the models the queue actually loads (so
"retraining" left the queue on old weights), and it carried on after a failed extraction (so the
queue was rebuilt from the previous data and the run still exited with code 0). Here a stage that
raises stops the chain with a `PipelineError`, and the queue is only written from the bundle that
the same run just trained.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.db.base import DataSource
from collection_queue_ranker.db.loader import load_raw_tables
from collection_queue_ranker.errors import CqrError, DataError, PipelineError
from collection_queue_ranker.labels import Agreements, prepare_agreements
from collection_queue_ranker.snapshots import (
    Enricher,
    build_break_dataset,
    build_snapshots,
    infer_as_of,
)
from collection_queue_ranker.tables import RawTables
from collection_queue_ranker.tracking import Tracker
from collection_queue_ranker.training import TrainResult, train
from collection_queue_ranker.worklist.build import QueueResult, build_queue, write_queue

log = logging.getLogger(__name__)
T = TypeVar("T")

SNAPSHOTS_FILE, BREAK_FILE, META_FILE = "snapshots.csv", "break.csv", "dataset_meta.json"
_TEXT_COLUMNS = {"product_id": "string", "phase_id": "string", "product_line": "string"}


def write_datasets(
    out_dir: str | Path, snapshots: pd.DataFrame, break_df: pd.DataFrame, as_of: pd.Timestamp
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / SNAPSHOTS_FILE, out / BREAK_FILE, out / META_FILE]
    snapshots.to_csv(paths[0], index=False)
    break_df.to_csv(paths[1], index=False)
    meta = {"as_of": pd.Timestamp(as_of).isoformat(), "snapshot_rows": len(snapshots),
            "break_rows": len(break_df)}
    paths[2].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def read_datasets(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    folder = Path(directory)
    missing = [n for n in (SNAPSHOTS_FILE, BREAK_FILE, META_FILE) if not (folder / n).is_file()]
    if missing:
        raise DataError(f"{folder} lacks {missing}: run `cqr build-dataset` first")
    snapshots = pd.read_csv(folder / SNAPSHOTS_FILE, parse_dates=["snapshot_at"],
                            dtype=_TEXT_COLUMNS)
    break_df = pd.read_csv(folder / BREAK_FILE, parse_dates=["anchor_at"], dtype=_TEXT_COLUMNS)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    return snapshots, break_df, pd.Timestamp(meta["as_of"])


def load_inputs(
    sources: Sequence[DataSource], settings: Settings, as_of: pd.Timestamp | None = None
) -> tuple[RawTables, Agreements, pd.Timestamp]:
    """Read the contract tables and prepare agreements.

    Without `as_of` every debt is read and `as_of` becomes the day after the latest timestamp in the
    data, which suits an export. In production pass today's date: then `history_months` bounds the
    debts that are read.
    """
    raw = load_raw_tables(sources, settings, as_of)
    resolved = pd.Timestamp(as_of) if as_of is not None else infer_as_of(raw)
    return raw, prepare_agreements(raw, settings), resolved


def build_datasets(
    raw: RawTables, prep: Agreements, settings: Settings, as_of: pd.Timestamp,
    enricher: Enricher | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = build_snapshots(raw, prep, settings, as_of, enricher)
    return snapshots, build_break_dataset(raw, prep, settings, as_of)


def _stage(name: str, fn: Callable[[], T]) -> T:
    started = time.perf_counter()
    log.info("stage %s: start", name)
    try:
        result = fn()
    except CqrError as exc:
        raise PipelineError(name, exc) from exc
    except Exception as exc:  # unexpected: still stop the chain, keep the cause
        raise PipelineError(name, exc) from exc
    log.info("stage %s: done in %.1fs", name, time.perf_counter() - started)
    return result


@dataclass
class RetrainResult:
    train: TrainResult
    queue: QueueResult
    files: list[Path]
    as_of: pd.Timestamp


def retrain(
    sources: Sequence[DataSource],
    settings: Settings,
    out_dir: str | Path,
    *,
    as_of: pd.Timestamp | None = None,
    tracker: Tracker | None = None,
    enricher: Enricher | None = None,
) -> RetrainResult:
    """Load, build datasets, train and judge, then write the queue from the new bundle."""
    out = Path(out_dir)
    raw, prep, when = _stage("load", lambda: load_inputs(sources, settings, as_of))
    snapshots, break_df = _stage("datasets",
                                 lambda: build_datasets(raw, prep, settings, when, enricher))
    files = write_datasets(out, snapshots, break_df, when)
    result = _stage("train", lambda: train(snapshots, break_df, settings, when,
                                           tracker=tracker, out_dir=out))
    queue = _stage("queue", lambda: build_queue(raw, prep, result.bundle, settings, when))
    files += result.files + write_queue(queue, out / "queue", settings.queue.csv_separator)
    return RetrainResult(result, queue, files, when)


def summarise(result: RetrainResult) -> dict[str, Any]:
    """The few numbers worth printing after a retrain."""
    close = result.train.report["close_head"]
    rank = result.train.report["ranking"]
    return {
        "as_of": result.as_of.date().isoformat(), "close_auc": close["auc"],
        "uplift_vs_balance_pct": rank["uplift_vs_balance_pct"], "verdict": rank["verdict"],
        "queue": result.queue.summary,
    }
