"""A small scoring API over a saved bundle (optional: `pip install 'collection-queue-ranker[serve]'`).

    GET  /health   liveness and whether a bundle is loaded
    GET  /model    what the loaded bundle is (version, dates, features, holdout metrics)
    POST /score    feature rows in, probabilities and a ranking out

`/score` takes rows that already hold the model features (the output of `features_at`, or the
columns of `queue_full.csv`). It does not look anything up in a database: building features needs the
whole history, which is the batch pipeline's job. Missing columns are reported by name with a 422.

This module deliberately does not use `from __future__ import annotations`: FastAPI resolves the
request model from the real annotation, and the model is defined inside `create_app` so that
FastAPI stays an optional dependency.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.models import ModelBundle
from collection_queue_ranker.scoring import scored_frame

MAX_ROWS = 5000


def _required_columns(bundle: ModelBundle) -> list[str]:
    names = [*bundle.close.space.names, *bundle.brk.space.names]
    if bundle.ranker is not None:
        names += bundle.ranker.space.names
    return sorted(set(names))


def score_rows(bundle: ModelBundle, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score feature rows and rank them by adjusted expected value (or the ranker, if present)."""
    frame = pd.DataFrame(rows)
    missing = [c for c in _required_columns(bundle) if c not in frame.columns]
    if missing:
        raise KeyError(missing)
    if "contact_factor" not in frame:
        frame["contact_factor"] = 1.0
    scored = scored_frame(frame, bundle)
    order = "score_lambdarank" if "score_lambdarank" in scored else "score_ev_contact"
    rank = scored[order].rank(method="first", ascending=False).astype(int)
    out = []
    for i in range(len(scored)):
        item = {
            "index": i,
            "p_close": float(scored["p_close"].iloc[i]),
            "p_break": float(scored["p_break"].iloc[i]),
            "expected_value": float(scored["score_ev"].iloc[i]),
            "adjusted_value": float(scored["score_ev_contact"].iloc[i]),
            "rank": int(rank.iloc[i]),
        }
        if "score_lambdarank" in scored:
            item["ranker_score"] = float(scored["score_lambdarank"].iloc[i])
        out.append(item)
    return out


def create_app(bundle_path: str | Path | None = None, bundle: ModelBundle | None = None) -> Any:
    """Build the FastAPI app. Pass a path to a saved bundle, or a bundle object."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "serving needs FastAPI: pip install 'collection-queue-ranker[serve]'"
        ) from exc

    loaded = bundle or (ModelBundle.load(bundle_path) if bundle_path else None)

    class ScoreRequest(BaseModel):
        rows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_ROWS)

    app = FastAPI(title="collection-queue-ranker", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": loaded is not None}

    @app.get("/model")
    def model() -> dict[str, Any]:
        if loaded is None:
            raise HTTPException(503, "no model bundle loaded (run `cqr train`, then restart)")
        meta = {k: v for k, v in loaded.meta.items() if k != "settings"}
        meta["features"] = {
            "close": loaded.close.space.names, "break": loaded.brk.space.names,
            "ranker": loaded.ranker.space.names if loaded.ranker else [],
        }
        return meta

    @app.post("/score")
    def score(request: ScoreRequest) -> dict[str, Any]:
        if loaded is None:
            raise HTTPException(503, "no model bundle loaded (run `cqr train`, then restart)")
        try:
            return {"scores": score_rows(loaded, request.rows)}
        except KeyError as exc:
            raise HTTPException(422, f"missing feature column(s): {exc.args[0]}") from exc

    return app

