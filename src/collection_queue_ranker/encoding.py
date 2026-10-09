"""Feature matrices for LightGBM.

Categorical columns become integer codes learned on the training rows only. A value never seen in
training (or a missing one) gets code -1, which LightGBM reads as "missing" for a categorical
feature, so scoring new data never crashes and never invents a category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from collection_queue_ranker.errors import DataError

UNKNOWN = -1


@dataclass
class CategoryEncoder:
    columns: list[str]
    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> CategoryEncoder:
        for col in self.columns:
            values = frame[col].astype("string").dropna().unique().tolist()
            self.categories[col] = sorted(str(v) for v in values)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.categories and self.columns:
            raise DataError("CategoryEncoder.transform called before fit")
        out = pd.DataFrame(index=frame.index)
        for col in self.columns:
            index = {v: i for i, v in enumerate(self.categories[col])}
            codes = frame[col].astype("string").map(index)
            out[col] = codes.fillna(UNKNOWN).astype(int)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "categories": self.categories}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryEncoder:
        return cls(list(data["columns"]), {k: list(v) for k, v in data["categories"].items()})


@dataclass
class FeatureSpace:
    """The feature names of one model and the encoder for its categoricals."""

    numeric: list[str]
    categorical: list[str]
    encoder: CategoryEncoder | None = None

    @property
    def names(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def fit(self, frame: pd.DataFrame) -> FeatureSpace:
        self._require(frame)
        self.encoder = CategoryEncoder(list(self.categorical)).fit(frame)
        return self

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric columns as floats followed by the integer-coded categoricals."""
        self._require(frame)
        if self.encoder is None:
            raise DataError("FeatureSpace.matrix called before fit")
        numeric = frame[self.numeric].apply(pd.to_numeric, errors="coerce").astype(float)
        return pd.concat([numeric, self.encoder.transform(frame)], axis=1)

    def _require(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.names if c not in frame.columns]
        if missing:
            raise DataError(f"input is missing feature column(s) {missing}")
