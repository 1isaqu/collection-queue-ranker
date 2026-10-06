"""Contact-response features measured before an anchor date.

Only features about how the debtor **responded** are used: did anyone answer, was it the wrong
party, is the number dead, how recently did someone answer. Features about how **hard we tried**
(attempt counts, attempts in the last N days) are deliberately left out even though they score a
higher AUC. They measure who the operation already chose to work, so a model using them would learn
to prioritise whoever is already prioritised: a feedback loop, not a signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from collection_queue_ranker.contacts import ANSWERED, VOICE
from collection_queue_ranker.tables import KEYS

CONTACT_FEATURES = [
    "n_answered", "n_wrong_party", "n_invalid_number", "answer_rate", "days_since_answered",
]
_COUNTS = ["n_answered", "n_wrong_party", "n_invalid_number"]


def contact_features(events: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Per anchored debtor: response features from events strictly before the anchor.

    Debtors with no earlier event are absent from the result; the caller fills counts with zero and
    leaves the rate and recency as missing ("never answered" is not the median).
    """
    columns = [*KEYS, *CONTACT_FEATURES]
    if events.empty or anchors.empty:
        return pd.DataFrame(columns=columns)
    e = events[[*KEYS, "category", "occurred_at"]].merge(anchors, on=KEYS, how="inner")
    e = e[e["occurred_at"] < e["anchor_at"]]
    if e.empty:
        return pd.DataFrame(columns=columns)
    category = e["category"].astype("object")
    e = e.assign(
        voice=category.isin(VOICE), answered=category.isin(ANSWERED),
        wrong=category.eq("wrong_party"), dead=category.eq("invalid_number"),
    )
    e["answered_at"] = e["occurred_at"].where(e["answered"])
    grouped = e.groupby(KEYS, as_index=False).agg(
        voice=("voice", "sum"), n_answered=("answered", "sum"),
        n_wrong_party=("wrong", "sum"), n_invalid_number=("dead", "sum"),
        last_answered=("answered_at", "max"), anchor_at=("anchor_at", "first"),
    )
    grouped["answer_rate"] = grouped["n_answered"] / grouped["voice"].replace(0, np.nan)
    grouped["days_since_answered"] = (grouped["anchor_at"] - grouped["last_answered"]).dt.days
    return grouped[columns]


def attach_contact(base: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Left-join contact features onto `base` (which carries `anchor_at`)."""
    found = contact_features(events, base[[*KEYS, "anchor_at"]])
    if found.empty:
        out = base.copy()
        for col in CONTACT_FEATURES:
            out[col] = 0 if col in _COUNTS else np.nan
        return out
    out = base.merge(found, on=KEYS, how="left")
    for col in _COUNTS:
        out[col] = out[col].fillna(0).astype(int)
    out["answer_rate"] = out["answer_rate"].astype(float)
    out["days_since_answered"] = out["days_since_answered"].astype(float)
    return out
