"""Turn model outputs into orderings.

Expected value of working a debtor:

    EV = balance x P(close) x (1 - P(break))

and the **adjusted** value multiplies by the contact factor (how reachable the debtor is). A
probability alone is a poor ordering under limited capacity: it ranks a small, likely payer above a
large, slightly less likely one. Ordering by value is what beats the "largest balance first" rule
that a manual operation tends to use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.models import ModelBundle

STRATEGIES: tuple[str, ...] = (
    "ev_contact", "ev", "prob", "lambdarank", "largest_balance", "random",
)
DESCRIPTIONS = {
    "ev_contact": "balance x P(close) x (1 - P(break)) x contact factor",
    "ev": "balance x P(close) x (1 - P(break))",
    "prob": "P(close) x (1 - P(break)), ignoring the balance",
    "lambdarank": "LightGBM LambdaRank score trained on graded realised value",
    "largest_balance": "baseline: biggest balance first",
    "random": "baseline: no prioritisation",
}


def expected_value(
    balance: np.ndarray, p_close: np.ndarray, p_break: np.ndarray
) -> np.ndarray:
    return np.asarray(balance, dtype=float) * p_close * (1.0 - p_break)


def require_strategy(name: str) -> str:
    if name not in STRATEGIES:
        raise ConfigError(f"unknown ranking strategy {name!r}; choose one of {STRATEGIES}")
    return name


def head_probabilities(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """P(close) and P(break) for every row (the break head sees only base features)."""
    return pd.DataFrame(
        {"p_close": bundle.close.predict(frame), "p_break": bundle.brk.predict(frame)},
        index=frame.index,
    )


def _scores(
    frame: pd.DataFrame, probs: pd.DataFrame, bundle: ModelBundle, seed: int
) -> dict[str, np.ndarray]:
    p_close, p_break = probs["p_close"].to_numpy(), probs["p_break"].to_numpy()
    balance = frame["balance"].to_numpy(dtype=float)
    ev = expected_value(balance, p_close, p_break)
    scores = {
        "ev_contact": ev * frame["contact_factor"].to_numpy(dtype=float),
        "ev": ev,
        "prob": p_close * (1.0 - p_break),
        "largest_balance": balance,
        "random": np.random.default_rng(seed).random(len(frame)),
    }
    if bundle.ranker is not None:
        scores["lambdarank"] = bundle.ranker.predict(frame)
    if "external_score" in frame:  # baseline: simply buy the external score and work it top-down
        scores["external_score"] = frame["external_score"].fillna(-1.0).to_numpy(dtype=float)
    return scores


def strategy_scores(
    frame: pd.DataFrame, bundle: ModelBundle, seed: int = 0
) -> dict[str, np.ndarray]:
    """Score `frame` under every strategy. A strategy the bundle cannot run is left out."""
    return _scores(frame, head_probabilities(frame, bundle), bundle, seed)


def scored_frame(frame: pd.DataFrame, bundle: ModelBundle, seed: int = 0) -> pd.DataFrame:
    """`frame` plus `p_close`, `p_break` and one `score_<strategy>` column per strategy."""
    out = frame.copy()
    probs = head_probabilities(frame, bundle)
    out["p_close"], out["p_break"] = probs["p_close"], probs["p_break"]
    for name, values in _scores(frame, probs, bundle, seed).items():
        out[f"score_{name}"] = values
    return out
