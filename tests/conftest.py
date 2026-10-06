"""Shared fixtures. The `demo` world is a small synthetic database generated at test time."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from collection_queue_ranker import synth
from collection_queue_ranker.config import Settings
from collection_queue_ranker.labels import prepare_agreements
from collection_queue_ranker.pipeline import build_datasets
from collection_queue_ranker.snapshots import infer_as_of
from collection_queue_ranker.tables import RawTables
from collection_queue_ranker.training import train

DEMO_OVERRIDES = {
    "evaluation": {"n_bootstrap": 25},
    "model": {"candidates": [{"max_depth": 3, "num_leaves": 8}, {"max_depth": 5, "num_leaves": 20}]},
    "tracking": {"enabled": False},
}


@pytest.fixture(scope="session")
def demo_settings() -> Settings:
    return Settings().with_overrides(DEMO_OVERRIDES)


@pytest.fixture(scope="session")
def demo(demo_settings):
    """Raw tables, prepared agreements and datasets for a 1,800-debtor synthetic line."""
    frames = synth.generate(1800, seed=5, months=20)
    raw = RawTables.from_frames(frames, demo_settings, default_line="auto")
    prep = prepare_agreements(raw, demo_settings)
    as_of = infer_as_of(raw)
    snapshots, break_df = build_datasets(raw, prep, demo_settings, as_of)
    return {"frames": frames, "raw": raw, "prep": prep, "as_of": as_of,
            "snapshots": snapshots, "break_df": break_df}


@pytest.fixture(scope="session")
def trained(demo, demo_settings, tmp_path_factory):
    out = tmp_path_factory.mktemp("trained")
    result = train(demo["snapshots"], demo["break_df"], demo_settings, demo["as_of"], out_dir=out)
    return result, out


@pytest.fixture(scope="session")
def day() -> pd.Timestamp:
    return pd.Timestamp("2024-03-01")
