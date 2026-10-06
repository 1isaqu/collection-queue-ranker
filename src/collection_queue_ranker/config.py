"""Settings tree.

Every number here is a plainly invented default. None of them was calibrated on real data: the
README says so, and the point of keeping them in one place is that you calibrate them on your own
data (horizon, cooldown, tier shares, phone-score weights and so on).

Settings are frozen dataclasses. A JSON file overrides any subset (see
`examples/settings.example.json`); unknown keys are rejected so a typo cannot silently do nothing.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collection_queue_ranker.errors import ConfigError


@dataclass(frozen=True)
class StatusSettings:
    """Agreement status labels (from `agreement_statuses.label`) by meaning."""

    paid: tuple[str, ...] = ("PAID", "PAID_MANUAL")
    broken: tuple[str, ...] = ("BROKEN", "AUTO_BROKEN")
    active: tuple[str, ...] = ("ACTIVE",)

    @property
    def valid(self) -> tuple[str, ...]:
        """Statuses that count as a real agreement. Pending exceptions and rejections do not."""
        return self.paid + self.broken + self.active


@dataclass(frozen=True)
class LabelSettings:
    horizon_days: int = 60
    legacy_tolerance_days: int = 45
    break_observation_days: int = 50


@dataclass(frozen=True)
class SnapshotSettings:
    frequency: str = "MS"
    first_cutoff: str | None = None
    warmup_days: int = 45
    history_months: int = 30
    min_balance: float = 0.0


@dataclass(frozen=True)
class FeatureSettings:
    balance_column: str = "balance"
    extra_numeric: tuple[str, ...] = ()
    extra_categorical: tuple[str, ...] = ()


@dataclass(frozen=True)
class SplitSettings:
    holdout_fraction: float = 0.3
    cv_folds: int = 3
    purge: bool = True


def _close_params() -> dict[str, Any]:
    return {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }


def _break_params() -> dict[str, Any]:
    return {**_close_params(), "max_depth": 5, "num_leaves": 20, "min_child_samples": 30}


def _ranker_params() -> dict[str, Any]:
    return {**_close_params(), "n_estimators": 150}


def _candidates() -> tuple[dict[str, Any], ...]:
    return (
        {"max_depth": 3, "num_leaves": 8},
        {"max_depth": 4, "num_leaves": 15},
        {"max_depth": 6, "num_leaves": 31},
    )


@dataclass(frozen=True)
class ModelSettings:
    seed: int = 17
    close_params: dict[str, Any] = field(default_factory=_close_params)
    break_params: dict[str, Any] = field(default_factory=_break_params)
    ranker_params: dict[str, Any] = field(default_factory=_ranker_params)
    candidates: tuple[dict[str, Any], ...] = field(default_factory=_candidates)
    selection_metric: str = "value_capture"
    selection_capacity: float = 0.2
    relevance: str = "value_grades"
    n_grades: int = 4
    train_ranker: bool = True


@dataclass(frozen=True)
class RankingSettings:
    capacities: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)
    tier_p1_share: float = 0.15
    tier_p2_share: float = 0.25
    strategy: str = "ev_contact"


@dataclass(frozen=True)
class AgentSettings:
    """Agent patterns use SQL LIKE syntax and are matched case-insensitively on `agents.login`."""

    automated_patterns: tuple[str, ...] = ("autodialer%", "system%")
    excluded_patterns: tuple[str, ...] = ("test%",)


@dataclass(frozen=True)
class PhoneSettings:
    smoothing_strength: float = 4.0
    recency_tiers: tuple[tuple[int, float], ...] = ((120, 1.0), (300, 0.75))
    recency_floor: float = 0.5
    automated_weight: float = 0.6
    invalid_kill_count: int = 3
    prior_mobile: float = 0.35
    prior_messaging: float = 0.25
    prior_priority: float = 0.15
    prior_fresh: float = 0.15
    fresh_days: int = 365
    top_phones: int = 3


@dataclass(frozen=True)
class QueueSettings:
    cooldown_days: int = 21
    top_n: int = 2000
    collectors: int = 8
    contact_floor: float = 0.4
    exclude_existing_agreements: bool = True
    csv_separator: str = ","


@dataclass(frozen=True)
class EvaluationSettings:
    n_bootstrap: int = 200
    ci_level: float = 0.95
    min_auc_alert: float = 0.60
    uplift_proof_pct: float = 8.0
    uplift_floor_pct: float = 3.0
    precision_ks: tuple[float, ...] = (0.05, 0.10, 0.20)
    seed: int = 7


@dataclass(frozen=True)
class TrackingSettings:
    enabled: bool = True
    uri: str = "mlruns"
    experiment: str = "collection-queue-ranker"


@dataclass(frozen=True)
class SourceSettings:
    """Where the contract tables live. Connection secrets are read from the environment only."""

    backend: str = "sqlite"
    sources: dict[str, str] = field(default_factory=dict)
    tables: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    statuses: StatusSettings = field(default_factory=StatusSettings)
    labels: LabelSettings = field(default_factory=LabelSettings)
    snapshots: SnapshotSettings = field(default_factory=SnapshotSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    ranking: RankingSettings = field(default_factory=RankingSettings)
    agents: AgentSettings = field(default_factory=AgentSettings)
    phones: PhoneSettings = field(default_factory=PhoneSettings)
    queue: QueueSettings = field(default_factory=QueueSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    source: SourceSettings = field(default_factory=SourceSettings)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def with_overrides(self, data: dict[str, Any]) -> Settings:
        """Return a copy with the nested `data` merged over the current values."""
        merged = _deep_merge(self.to_dict(), data)
        return settings_from_dict(merged)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge objects key by key (this also lets a file override one LightGBM parameter)."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if origin in (typing.Union, types.UnionType):
        non_null = [a for a in args if a is not type(None)]
        if value is None:
            return None
        return _coerce(non_null[0], value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value, strict=False)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
        return dict(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
    return value


def _build(cls: Any, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'settings'}: expected an object")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{path or 'settings'}: unknown key(s) {unknown}; known: {sorted(names)}")
    kwargs = {k: _coerce(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
    return cls(**kwargs)


def settings_from_dict(data: dict[str, Any]) -> Settings:
    """Build `Settings` from a nested dict. Missing keys keep their defaults."""
    return _build(Settings, data, "")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a JSON file, or return the defaults when `path` is empty."""
    if not path:
        return Settings()
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"settings file not found: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{file}: invalid JSON ({exc})") from exc
    return Settings().with_overrides(data)
