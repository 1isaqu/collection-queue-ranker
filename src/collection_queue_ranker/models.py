"""LightGBM heads, the LambdaRank strategy and the saved bundle.

Two classifier heads feed the expected-value ordering:

* **close**: will this debtor sign an agreement within the horizon?
* **break**: if they sign, will the agreement fail?

Collapsing them into one "will it be paid" probability and then multiplying by (1 - P(break)) would
count non-breakage twice, which is why they are separate. They are trained *without* class weights:
the probabilities are multiplied by a balance, so they need to be calibrated, and reweighting a
rare class inflates every probability.

The **ranker** is a LightGBM LambdaRank model. Each snapshot is one query (one queue), and the graded
relevance of a row is its realised value tier, so the model optimises the order of the top of the
queue directly instead of a probability.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from collection_queue_ranker import __version__
from collection_queue_ranker.config import ModelSettings, Settings
from collection_queue_ranker.encoding import FeatureSpace
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.base import BASE_CATEGORICAL, BASE_NUMERIC
from collection_queue_ranker.features.contact import CONTACT_FEATURES

BUNDLE_FORMAT = 1


def close_space(settings: Settings) -> FeatureSpace:
    f = settings.features
    return FeatureSpace(
        numeric=[*BASE_NUMERIC, *CONTACT_FEATURES, *f.extra_numeric],
        categorical=[*BASE_CATEGORICAL, *f.extra_categorical],
    )


def break_space(settings: Settings) -> FeatureSpace:
    """The break head sees only what is known before any call: no contact response features."""
    return FeatureSpace(numeric=list(BASE_NUMERIC), categorical=list(BASE_CATEGORICAL))


def ranker_space(settings: Settings) -> FeatureSpace:
    return close_space(settings)


def lgbm_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    """User parameters plus the settings that make a fit reproducible and quiet."""
    return {
        **params, "random_state": seed, "verbose": -1, "deterministic": True,
        "force_row_wise": True,
    }


def _fit_kwargs(space: FeatureSpace) -> dict[str, Any]:
    return {"categorical_feature": list(space.categorical)} if space.categorical else {}


def _importance(model: Any, names: list[str]) -> pd.DataFrame:
    gain = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
    total = gain.sum() or 1.0
    table = pd.DataFrame({"feature": names, "gain_share": gain / total})
    return table.sort_values("gain_share", ascending=False).reset_index(drop=True)


@dataclass
class ClassifierHead:
    name: str
    space: FeatureSpace
    params: dict[str, Any]
    seed: int
    model: lgb.LGBMClassifier | None = None

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> ClassifierHead:
        y = np.asarray(target).astype(int)
        if len(np.unique(y)) < 2:
            raise DataError(f"head `{self.name}`: the training target has a single class")
        X = self.space.fit(frame).matrix(frame)
        self.model = lgb.LGBMClassifier(**lgbm_params(self.params, self.seed))
        self.model.fit(X, y, **_fit_kwargs(self.space))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise DataError(f"head `{self.name}` is not fitted")
        return self.model.predict_proba(self.space.matrix(frame))[:, 1]

    def importance(self) -> pd.DataFrame:
        if self.model is None:
            raise DataError(f"head `{self.name}` is not fitted")
        return _importance(self.model, self.space.names)


def relevance_grades(
    frame: pd.DataFrame, relevance: str, n_grades: int, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Integer relevance per row and the value edges used.

    `closed`: 1 when an agreement started, else 0. `value_grades`: 0 when nothing was received,
    otherwise 1..n_grades-1 by tier of the received value (tier edges are quantiles of the positive
    values in the training frame and are reused unchanged when grading anything else).
    """
    if relevance == "closed":
        return frame["closed"].to_numpy().astype(int), np.array([])
    if relevance != "value_grades":
        raise DataError(f"unknown relevance {relevance!r}; use 'closed' or 'value_grades'")
    value = frame["realized_value"].to_numpy(dtype=float)
    positive = value[value > 0]
    if edges is None:
        n_edges = max(n_grades - 2, 0)
        probs = np.arange(1, n_edges + 1) / (n_edges + 1)
        edges = np.quantile(positive, probs) if len(positive) and n_edges else np.array([])
    grades = np.where(value > 0, 1 + np.digitize(value, edges), 0)
    return grades.astype(int), np.asarray(edges)


@dataclass
class RankerHead:
    space: FeatureSpace
    params: dict[str, Any]
    seed: int
    relevance: str = "value_grades"
    n_grades: int = 4
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    model: lgb.LGBMRanker | None = None

    @classmethod
    def from_settings(cls, space: FeatureSpace, ms: ModelSettings) -> RankerHead:
        return cls(space, dict(ms.ranker_params), ms.seed, ms.relevance, ms.n_grades)

    def fit(self, frame: pd.DataFrame, group_col: str = "snapshot_at") -> RankerHead:
        ordered = frame.sort_values(group_col, kind="stable").reset_index(drop=True)
        grades, self.edges = relevance_grades(ordered, self.relevance, self.n_grades)
        if grades.max() == 0:
            raise DataError("ranker: no row has a positive relevance grade")
        groups = ordered.groupby(group_col, sort=True).size().to_numpy()
        X = self.space.fit(ordered).matrix(ordered)
        top = int(grades.max()) + 1
        params = lgbm_params(self.params, self.seed)
        self.model = lgb.LGBMRanker(
            objective="lambdarank", label_gain=[2**i - 1 for i in range(top)], **params
        )
        self.model.fit(X, grades, group=groups, **_fit_kwargs(self.space))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise DataError("ranker is not fitted")
        return np.asarray(self.model.predict(self.space.matrix(frame)), dtype=float)

    def importance(self) -> pd.DataFrame:
        if self.model is None:
            raise DataError("ranker is not fitted")
        return _importance(self.model, self.space.names)


@dataclass
class ModelBundle:
    """Everything needed to score: both heads, the optional ranker, and how they were made.

    Bundles are pickled with joblib. Only load bundles you created yourself: unpickling runs code.
    """

    close: ClassifierHead
    brk: ClassifierHead
    ranker: RankerHead | None
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": BUNDLE_FORMAT, "bundle": self}
        joblib.dump(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        file = Path(path)
        if not file.is_file():
            raise DataError(f"model bundle not found: {file} (run `cqr train` first)")
        payload = joblib.load(file)
        if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
            raise DataError(f"{file} is not a bundle written by this version")
        return payload["bundle"]


def new_meta(settings: Settings, as_of: pd.Timestamp, holdout_start: pd.Timestamp) -> dict[str, Any]:
    return {
        "package_version": __version__,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "as_of": pd.Timestamp(as_of).isoformat(),
        "holdout_start": pd.Timestamp(holdout_start).isoformat(),
        "settings": settings.to_dict(),
    }
