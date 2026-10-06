"""Build today's work queue from open debts and a saved model bundle.

Steps, in order:

1. the working population as of `as_of`, from the same `features_at` used in training (so scoring
   cannot drift from training); debtors who already hold a valid agreement are left to the
   agreement flow, matching how the model was trained;
2. score the two heads and the chosen strategy;
3. **cooldown**: a debtor a *human* attempted within `cooldown_days` is already being worked and is
   left out. An automated dialer sweeps the whole base to hold volume, so counting its attempts here
   would remove people nobody has actually spoken to;
4. tiers by rank share (P1 / P2 / P3) and a **contactability gate**: a debtor with no usable phone is
   not workable by voice today, so it goes to the end under its own label (find the person first);
5. the operational cut: contactable debtors only, top N, dealt round-robin to collectors.

The operational file carries no scores on purpose. Next to a debtor, a probability becomes a promise
in the caller's mouth; the order of the rows already holds all the prioritisation. Phones appear as
opaque ids: the numbers themselves stay in your system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.phones import phone_quality, top_phones_wide
from collection_queue_ranker.labels import Agreements
from collection_queue_ranker.models import ModelBundle
from collection_queue_ranker.scoring import require_strategy, scored_frame
from collection_queue_ranker.snapshots import features_at
from collection_queue_ranker.tables import KEYS, RawTables
from collection_queue_ranker.util import text_id
from collection_queue_ranker.worklist.assign import assign_round_robin

NEVER, COLD, RECENT = "never_contacted", "cold", "recent"
NO_CONTACT = "NO_CONTACT"
PHONE_SLOTS = 3


@dataclass
class QueueResult:
    full: pd.DataFrame
    operational: pd.DataFrame
    summary: dict[str, Any]


def last_human_attempt(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Most recent attempt by a non-automated agent before `as_of`, per debtor."""
    human = events[~events["is_automated"] & (events["occurred_at"] < as_of)]
    if human.empty:
        return pd.DataFrame(columns=[*KEYS, "last_human_at"])
    return (human.groupby(KEYS, as_index=False)["occurred_at"].max()
            .rename(columns={"occurred_at": "last_human_at"}))


def contact_status(
    frame: pd.DataFrame, events: pd.DataFrame, as_of: pd.Timestamp, cooldown_days: int
) -> pd.DataFrame:
    """Add `last_human_at`, `days_since_contact` and `contact_status` (never / cold / recent)."""
    last = last_human_attempt(events, as_of)
    out = frame.merge(last, on=KEYS, how="left") if not last.empty else frame.assign(
        last_human_at=pd.NaT
    )
    out["days_since_contact"] = (as_of - out["last_human_at"]).dt.days
    status = np.select(
        [out["last_human_at"].isna(), out["days_since_contact"] < cooldown_days],
        [NEVER, RECENT], default=COLD,
    )
    out["contact_status"] = status
    return out


def assign_tiers(score: pd.Series, p1_share: float, p2_share: float) -> pd.Series:
    """P1 for the top `p1_share` of rows by score, P2 for the next `p2_share`, P3 for the rest."""
    share = score.rank(method="first", ascending=False) / max(len(score), 1)
    return pd.Series(
        np.select([share <= p1_share, share <= p1_share + p2_share], ["P1", "P2"], default="P3"),
        index=score.index,
    )


def _ranked(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    strategy = require_strategy(settings.ranking.strategy)
    column = f"score_{strategy}"
    if column not in frame:
        raise DataError(f"strategy `{strategy}` needs a model that this bundle does not contain")
    out = frame.copy()
    out["ranking_score"] = out[column]
    out["expected_value"] = out["score_ev"]
    out["adjusted_value"] = out["score_ev_contact"]
    out["contactable"] = out["usable_phones"] > 0
    out["tier"] = NO_CONTACT
    workable = out["contactable"]
    out.loc[workable, "tier"] = assign_tiers(
        out.loc[workable, "ranking_score"], settings.ranking.tier_p1_share,
        settings.ranking.tier_p2_share,
    )
    out = out.sort_values(
        ["contactable", "ranking_score", "product_line", "debtor_id"],
        ascending=[False, False, True, True], kind="stable",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _operational(full: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    work = full[full["contactable"]].head(settings.queue.top_n)
    columns = [
        "product_line", "debtor_id", *[f"phone_{i}_id" for i in range(1, PHONE_SLOTS + 1)],
        "balance", "days_overdue_max", "contact_status", "days_since_contact",
    ]
    out = work[columns].rename(columns={
        "days_overdue_max": "days_overdue", "contact_status": "last_contact_status",
        "days_since_contact": "days_since_last_contact",
    }).reset_index(drop=True)
    collector, order = assign_round_robin(len(out), settings.queue.collectors)
    out.insert(0, "collector", collector)
    out.insert(1, "order", order)
    for col in ("debtor_id", *[f"phone_{i}_id" for i in range(1, PHONE_SLOTS + 1)]):
        out[col] = out[col].map(text_id)
    out["days_since_last_contact"] = out["days_since_last_contact"].astype("Int64")
    out["balance"] = out["balance"].round(2)
    return out


def build_queue(
    raw: RawTables,
    prep: Agreements,
    bundle: ModelBundle,
    settings: Settings,
    as_of: pd.Timestamp,
) -> QueueResult:
    """Score the working population at `as_of` and return the full and operational queues."""
    as_of = pd.Timestamp(as_of)
    frame = features_at(
        raw, prep, as_of, settings,
        exclude_existing_agreements=settings.queue.exclude_existing_agreements,
    )
    if frame.empty:
        raise DataError(f"no debtor with an open balance at {as_of.date()}")
    scored = scored_frame(frame, bundle, settings.model.seed)
    scored = contact_status(scored, raw.events, as_of, settings.queue.cooldown_days)
    recent = int((scored["contact_status"] == RECENT).sum())
    active = scored[scored["contact_status"] != RECENT].reset_index(drop=True)
    if active.empty:
        raise DataError("every debtor is inside the cooldown window: nothing to queue")
    quality = phone_quality(raw.phones, raw.events, as_of, settings.phones)
    phones = top_phones_wide(quality, PHONE_SLOTS)
    active = active.merge(phones, on=KEYS, how="left") if not phones.empty else active.assign(
        **{f"phone_{i}_{c}": np.nan for i in range(1, PHONE_SLOTS + 1) for c in ("id", "score")}
    )
    full = _ranked(active, settings)
    operational = _operational(full, settings)
    return QueueResult(full, operational, _summary(scored, full, operational, recent, as_of))


def _summary(
    scored: pd.DataFrame, full: pd.DataFrame, operational: pd.DataFrame, recent: int,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    factor = full["contact_factor"]
    return {
        "as_of": as_of.date().isoformat(), "open_population": len(scored),
        "excluded_recent_human_attempt": recent, "active_base": len(full),
        "tiers": {k: int(v) for k, v in full["tier"].value_counts().items()},
        "contactable": int(full["contactable"].sum()), "operational_rows": len(operational),
        "contact_factor": {"min": float(factor.min()), "median": float(factor.median()),
                           "max": float(factor.max())},
    }


def write_queue(result: QueueResult, directory: str | Path, separator: str = ",") -> list[Path]:
    """Write `queue_full.csv`, `queue_operational.csv` and `queue_summary.json`."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    full, operational = out / "queue_full.csv", out / "queue_operational.csv"
    result.full.to_csv(full, index=False, sep=separator)
    result.operational.to_csv(operational, index=False, sep=separator)
    summary = out / "queue_summary.json"
    summary.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    return [full, operational, summary]
