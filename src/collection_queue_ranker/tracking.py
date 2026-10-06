"""Experiment tracking with MLflow, using a local store that never leaves your machine.

The default store is a directory (`mlruns/`, gitignored). MLflow 3 treats the plain file store as
legacy and refuses it unless `MLFLOW_ALLOW_FILE_STORE=true` is set; this module sets that variable
for you (without overriding a value you chose) when the tracking URI is a directory. If you prefer a
database store, set `tracking.uri` to something like `sqlite:///mlruns/mlflow.db`; nothing else changes.

`NullTracker` has the same interface and does nothing, so tracking can be switched off (`--no-tracking`)
without touching the training code, and MLflow is imported only when tracking is on.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from collection_queue_ranker.config import TrackingSettings


class Tracker(Protocol):
    run_id: str | None

    def run(
        self, name: str, params: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Any: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None: ...


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested settings into `a.b.c` -> short string, the shape MLflow params need."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:250]
    return flat


class NullTracker:
    """Does nothing. Used when tracking is disabled."""

    run_id: str | None = None

    @contextmanager
    def run(
        self, name: str, params: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[NullTracker]:
        yield self

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        return None

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        return None


def tracking_uri(uri: str) -> str:
    """A directory name becomes a `file:` URI; URIs with a scheme pass through unchanged."""
    if "://" in uri or uri.startswith(("sqlite:", "file:")):
        return uri
    return Path(uri).resolve().as_uri()


class MlflowTracker:
    def __init__(self, settings: TrackingSettings) -> None:
        uri = tracking_uri(settings.uri)
        if uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        self._mlflow = mlflow
        self.uri = uri
        self.run_id: str | None = None
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(settings.experiment)

    @contextmanager
    def run(
        self, name: str, params: Mapping[str, Any] | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[MlflowTracker]:
        with self._mlflow.start_run(run_name=name) as active:
            self.run_id = active.info.run_id
            if params:
                self._mlflow.log_params(flatten(params))
            if tags:
                self._mlflow.set_tags(dict(tags))
            yield self

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(v)}
        if clean:
            self._mlflow.log_metrics(clean)

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=subdir)


def make_tracker(settings: TrackingSettings, enabled: bool | None = None) -> Tracker:
    """`enabled` (from a CLI flag) wins over `settings.enabled`."""
    on = settings.enabled if enabled is None else enabled
    return MlflowTracker(settings) if on else NullTracker()
