"""Phone quality and the contact factor.

A phone gets a 0-100 score from its own dial history:

* attempts are voice attempts only (automated messages are not an attempt to reach a human), and
  attempts by an automated dialer count with a reduced weight, because a dialer sweeps the whole
  base to hold daily volume and answers far less often than a person: counting its attempts in full
  would punish every phone it happened to touch;
* the answer rate is smoothed toward the global rate (Bayesian shrinkage), so a phone dialled twice
  is not trusted as much as one dialled thirty times;
* the score decays with the age of the last attempt, in tiers;
* a phone reported as a dead number several times and never answered is killed (score 0);
* a phone that was never dialled gets a prior around the global rate, raised by good attributes
  (mobile, messaging-capable, priority, recently updated) and zero when flagged invalid.

Every weight lives in `PhoneSettings`; the defaults are invented and must be calibrated on your data.
The score is computed from events strictly before `as_of`, so it can be evaluated point-in-time.

The **contact factor** of a debtor turns the best few phones into one multiplier for the
expected-value ordering: reach probability = 1 - prod(1 - score_i / 100), mapped to
[floor, 1]. It reorders the queue by reachability but never zeroes a debtor, because the model
itself is uncertain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from collection_queue_ranker.config import PhoneSettings, QueueSettings
from collection_queue_ranker.contacts import ANSWERED, VOICE
from collection_queue_ranker.tables import KEYS

PHONE_KEYS = ["product_line", "phone_id"]
FALLBACK_RATE = 0.2


def _history(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-phone dial history from events before `as_of`."""
    e = events[events["phone_id"].notna() & (events["occurred_at"] < as_of)]
    if e.empty:
        return pd.DataFrame(columns=[*PHONE_KEYS, "voice_human", "voice_auto", "contacts",
                                     "dead", "last_attempt"])
    category = e["category"].astype("object")
    voice = category.isin(VOICE)
    e = e.assign(
        voice_human=voice & ~e["is_automated"], voice_auto=voice & e["is_automated"],
        contacts=category.isin(ANSWERED), dead=category.eq("invalid_number"),
    )
    return e.groupby(PHONE_KEYS, as_index=False).agg(
        voice_human=("voice_human", "sum"), voice_auto=("voice_auto", "sum"),
        contacts=("contacts", "sum"), dead=("dead", "sum"), last_attempt=("occurred_at", "max"),
    )


def _recency(days: pd.Series, settings: PhoneSettings) -> pd.Series:
    factor = pd.Series(settings.recency_floor, index=days.index, dtype=float)
    for limit, value in sorted(settings.recency_tiers, reverse=True):
        factor[days <= limit] = value
    return factor


def phone_quality(
    phones: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    settings: PhoneSettings,
) -> pd.DataFrame:
    """Score every phone known at `as_of`. Adds `score` (0-100) and the history columns."""
    known = phones[phones["created_at"].isna() | (phones["created_at"] < as_of)]
    if known.empty:
        return known.assign(score=pd.Series(dtype=float))
    hist = _history(events, as_of)
    df = known.merge(hist, on=PHONE_KEYS, how="left") if not hist.empty else known.copy()
    for col in ("voice_human", "voice_auto", "contacts", "dead"):
        df[col] = df[col].fillna(0) if col in df else 0.0
    if "last_attempt" not in df:
        df["last_attempt"] = pd.NaT
    df["attempts"] = df["voice_human"] + settings.automated_weight * df["voice_auto"]
    base_rate = _global_rate(df)
    df["score"] = _score(df, base_rate, as_of, settings)
    return df


def _global_rate(df: pd.DataFrame) -> float:
    dialled = df[df["attempts"] > 0]
    total = dialled["attempts"].sum()
    return float(dialled["contacts"].sum() / total) if total > 0 else FALLBACK_RATE


def _score(
    df: pd.DataFrame, base_rate: float, as_of: pd.Timestamp, s: PhoneSettings
) -> pd.Series:
    smoothed = (df["contacts"] + s.smoothing_strength * base_rate) / (
        df["attempts"] + s.smoothing_strength
    )
    days = (as_of - df["last_attempt"]).dt.days
    history = 100 * smoothed * _recency(days, s)
    killed = (df["dead"] >= s.invalid_kill_count) & (df["contacts"] == 0)
    history = history.where(~killed, 0.0)
    fresh = ((as_of - df["updated_at"]).dt.days <= s.fresh_days).astype(float)
    boost = (1.0 + s.prior_mobile * df["is_mobile"] + s.prior_messaging * df["has_messaging"]
             + s.prior_priority * df["is_priority"] + s.prior_fresh * fresh)
    prior = (100 * base_rate * boost).where(df["is_valid"] == 1, 0.0)
    score = history.where(df["attempts"] > 0, prior)
    return pd.Series(np.clip(score, 0, 100).round(1), index=df.index)


def _usable_sorted(quality: pd.DataFrame) -> pd.DataFrame:
    """Phones with a positive score, best first (ties: mobile, then most recently updated)."""
    usable = quality[quality["score"] > 0]
    return usable.sort_values(
        ["score", "is_mobile", "updated_at"], ascending=[False, False, False], kind="stable"
    )


def contact_factor(
    quality: pd.DataFrame, phones_settings: PhoneSettings, queue_settings: QueueSettings
) -> pd.DataFrame:
    """Per debtor: `contact_factor`, `usable_phones` and `best_phone_score`."""
    columns = [*KEYS, "contact_factor", "usable_phones", "best_phone_score"]
    usable = _usable_sorted(quality) if not quality.empty else quality
    if usable.empty:
        return pd.DataFrame(columns=columns)
    top = usable.groupby(KEYS).head(phones_settings.top_phones)
    miss = (1 - top["score"] / 100).groupby([top[k] for k in KEYS]).prod()
    summary = usable.groupby(KEYS).agg(
        usable_phones=("score", "size"), best_phone_score=("score", "max")
    )
    out = summary.join((1 - miss).rename("reach")).reset_index()
    floor = queue_settings.contact_floor
    out["contact_factor"] = floor + (1 - floor) * out["reach"]
    return out[columns]


def attach_contact_factor(
    frame: pd.DataFrame,
    quality: pd.DataFrame,
    phones_settings: PhoneSettings,
    queue_settings: QueueSettings,
) -> pd.DataFrame:
    """Add `contact_factor`, `usable_phones` and `best_phone_score` to a debtor frame."""
    factors = contact_factor(quality, phones_settings, queue_settings)
    out = frame.copy()
    if factors.empty:
        out["contact_factor"] = queue_settings.contact_floor
        out["usable_phones"] = 0
        out["best_phone_score"] = np.nan
        return out
    out = out.merge(factors, on=KEYS, how="left")
    out["contact_factor"] = out["contact_factor"].fillna(queue_settings.contact_floor)
    out["usable_phones"] = out["usable_phones"].fillna(0).astype(int)
    return out


def top_phones_wide(quality: pd.DataFrame, k: int) -> pd.DataFrame:
    """Best `k` phones per debtor as `phone_i_id` / `phone_i_score` columns (opaque ids only)."""
    columns = [*KEYS, *[f"phone_{i}_{c}" for i in range(1, k + 1) for c in ("id", "score")]]
    if quality.empty:
        return pd.DataFrame(columns=columns)
    top = _usable_sorted(quality).groupby(KEYS).head(k).copy()
    top["rank"] = top.groupby(KEYS).cumcount() + 1
    wide = top.pivot(index=KEYS, columns="rank", values=["phone_id", "score"])
    wide.columns = [f"phone_{r}_{'id' if v == 'phone_id' else 'score'}" for v, r in wide.columns]
    return wide.reindex(columns=columns[len(KEYS):]).reset_index()
