"""Did the delivered queue get worked, and what did it yield?

Adoption and funnel of a delivered queue, measured straight from the contact and agreement tables
(nobody has to tabulate anything by hand):

    delivered -> attempted -> answered -> promised -> agreement -> value

Three design rules:

1. **Bind dates as parameters.** A date spliced into SQL text is parsed under the server's date
   format; with an unlucky setting a day below 13 silently swaps day and month and the window is wrong
   without any error. Every date in this package is a bound parameter.
2. **Do not trust a column nobody fills.** A "promise" flag that exists in the schema but is never set
   reads as zero promises. Promises are read from the contact-result category instead.
3. **The answered map is a bias, not a measurement.** If the result catalogue under-counts answers,
   the answer rate under-states reality by the same factor everywhere. Compare bands against each other;
   do not quote the absolute number.

The delivered queue is disjoint from recent work by construction (the cooldown). So "queue versus the
rest of the operation" would compare cold debtors with hot ones. The fair comparison group is debtors
who were **equally cold** when the window opened, were worked in it, and were *not* on the list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from collection_queue_ranker.contacts import ANSWERED
from collection_queue_ranker.labels import AGREEMENT_KEYS, Agreements
from collection_queue_ranker.tables import RawTables
from collection_queue_ranker.util import debtor_key

MIN_ATTEMPTED = 30


def category_counts(events: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Per debtor key: attempts and one count column per contact category, from `since` on."""
    window = events[events["occurred_at"] >= since]
    if window.empty:
        return pd.DataFrame(columns=["key", "attempts"])
    window = window.assign(key=debtor_key(window))
    counts = window.pivot_table(index="key", columns="category", values="event_id",
                                aggfunc="count", fill_value=0)
    counts["attempts"] = window.groupby("key").size()
    return counts.reset_index()


def agreements_since(prep: Agreements, since: pd.Timestamp) -> pd.DataFrame:
    """Valid agreements issued from `since` on: key, agreement id and total installment amount."""
    first = prep.first[prep.first["issued_at"] >= since]
    if first.empty:
        return pd.DataFrame(columns=["key", "agreement_id", "agreed_value"])
    value = (prep.installments.groupby(AGREEMENT_KEYS)["amount"].sum()
             .rename("agreed_value").reset_index())
    out = first.merge(value, on=AGREEMENT_KEYS, how="left")
    return out.assign(key=debtor_key(out))[["key", "agreement_id", "agreed_value"]]


def _has(counts: pd.DataFrame, categories: frozenset[str]) -> pd.Series:
    present = [c for c in categories if c in counts.columns]
    if not present:
        return pd.Series(False, index=counts.index)
    return counts[present].sum(axis=1) > 0


def funnel(delivered_keys: set[str], counts: pd.DataFrame, agreements: pd.DataFrame) -> dict:
    """Funnel restricted to the debtors on the delivered queue."""
    c = counts[counts["key"].isin(delivered_keys)] if len(counts) else counts
    a = agreements[agreements["key"].isin(delivered_keys)]
    return {
        "delivered": len(delivered_keys),
        "attempted": len(c),
        "answered": int(_has(c, ANSWERED).sum()) if len(c) else 0,
        "promised": int(_has(c, frozenset({"promise"})).sum()) if len(c) else 0,
        "agreements": int(a["agreement_id"].nunique()),
        "debtors_with_agreement": int(a["key"].nunique()),
        "agreed_value": float(a["agreed_value"].sum()),
    }


def by_predicted_band(
    delivered: pd.DataFrame,
    counts: pd.DataFrame,
    agreements: pd.DataFrame,
    column: str,
    bands: int = 4,
) -> pd.DataFrame:
    """Observed rates per band of a predicted column (Q1 = lowest predicted, Qn = highest).

    This is the ordering test: the absolute level may be off, what matters is whether a better
    predicted band yields more than a worse one.
    """
    frame = delivered.assign(key=debtor_key(delivered))
    rank = frame[column].rank(method="first")
    frame["band"] = pd.qcut(rank, bands, labels=[f"Q{i}" for i in range(1, bands + 1)])
    attempted = set(counts["key"]) if len(counts) else set()
    answered = set(counts.loc[_has(counts, ANSWERED), "key"]) if len(counts) else set()
    agreed = set(agreements["key"])
    frame["attempted"] = frame["key"].isin(attempted)
    frame["answered"] = frame["key"].isin(answered)
    frame["agreed"] = frame["key"].isin(agreed)
    grouped = frame.groupby("band", observed=True).agg(
        rows=("key", "size"), predicted_median=(column, "median"),
        attempted=("attempted", "sum"), answered=("answered", "sum"), agreed=("agreed", "sum"),
    )
    grouped["answer_rate_of_attempted"] = (
        grouped["answered"] / grouped["attempted"].replace(0, np.nan)).round(3)
    grouped["agreement_rate_of_attempted"] = (
        grouped["agreed"] / grouped["attempted"].replace(0, np.nan)).round(3)
    return grouped.reset_index()


def cold_baseline(
    delivered_keys: set[str],
    counts: pd.DataFrame,
    agreements: pd.DataFrame,
    last_before: pd.DataFrame,
    since: pd.Timestamp,
    cooldown_days: int,
) -> dict[str, Any]:
    """Debtors equally cold at `since`, worked in the window, that were not on the list."""
    if last_before.empty or counts.empty:
        return {"n": 0, "note": "no cold debtor outside the list was worked in the window"}
    cold = last_before[(since - last_before["last_attempt"]).dt.days >= cooldown_days]
    outside = counts[~counts["key"].isin(delivered_keys) & counts["key"].isin(cold["key"])]
    if outside.empty:
        return {"n": 0, "note": "no cold debtor outside the list was worked in the window"}
    return {
        "n": len(outside),
        "answer_rate": round(float(_has(outside, ANSWERED).mean()), 3),
        "agreement_rate": round(float(outside["key"].isin(set(agreements["key"])).mean()), 3),
    }


def last_attempt_before(events: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """Last attempt (any agent) before `since`, per debtor key."""
    earlier = events[events["occurred_at"] < since]
    if earlier.empty:
        return pd.DataFrame(columns=["key", "last_attempt"])
    earlier = earlier.assign(key=debtor_key(earlier))
    return (earlier.groupby("key", as_index=False)["occurred_at"].max()
            .rename(columns={"occurred_at": "last_attempt"}))


@dataclass
class MonitorReport:
    since: pd.Timestamp
    funnel: dict[str, Any]
    enough_volume: bool
    bands: dict[str, pd.DataFrame]
    baseline: dict[str, Any]


def monitor(
    delivered: pd.DataFrame,
    predicted: pd.DataFrame,
    raw: RawTables,
    prep: Agreements,
    since: pd.Timestamp,
    cooldown_days: int,
    band_columns: tuple[str, ...] = ("best_phone_score", "ranking_score"),
) -> MonitorReport:
    """Funnel, ordering test and cold baseline for a delivered queue.

    `delivered` is the operational queue; `predicted` is the full queue, joined on the debtor key
    to bring the predicted columns the ordering test bands on.
    """
    since = pd.Timestamp(since)
    keys = set(debtor_key(delivered))
    counts = category_counts(raw.events, since)
    agreements = agreements_since(prep, since)
    result = funnel(keys, counts, agreements)
    enough = result["attempted"] >= MIN_ATTEMPTED
    bands: dict[str, pd.DataFrame] = {}
    baseline: dict[str, Any] = {}
    if enough:
        joined = delivered.assign(key=debtor_key(delivered)).merge(
            predicted.assign(key=debtor_key(predicted))[["key", *band_columns]],
            on="key", how="left",
        )
        for column in band_columns:
            if joined[column].notna().sum() >= 4:
                bands[column] = by_predicted_band(joined, counts, agreements, column)
        baseline = cold_baseline(
            keys, counts, agreements, last_attempt_before(raw.events, since), since, cooldown_days
        )
    return MonitorReport(since, result, enough, bands, baseline)


def render_monitor(report: MonitorReport) -> str:
    f = report.funnel
    delivered = max(f["delivered"], 1)
    lines = [
        f"Delivered queue: {f['delivered']} debtors; window opens {report.since:%Y-%m-%d %H:%M}",
        "", "Funnel (queue debtors only)",
        f"  delivered ......... {f['delivered']:>7}",
        f"  attempted ......... {f['attempted']:>7}  ({f['attempted'] / delivered:.1%} of queue)",
        f"  answered .......... {f['answered']:>7}",
        f"  promised .......... {f['promised']:>7}",
        f"  agreements ........ {f['agreements']:>7}  ({f['debtors_with_agreement']} debtors)",
        f"  agreed value ...... {f['agreed_value']:>12,.2f}",
    ]
    if not report.enough_volume:
        lines += ["", f"Not enough volume to read rates (fewer than {MIN_ATTEMPTED} attempted "
                      "debtors). Any rate here would be sampling noise; run again after a full "
                      "working day with the list in hand."]
        return "\n".join(lines) + "\n"
    for column, table in report.bands.items():
        lines += ["", f"Ordering test by `{column}` (Q1 = lowest predicted, highest = best)",
                  table.to_string(index=False)]
    lines += ["", f"Cold baseline (equally cold, worked, not on the list): {report.baseline}",
              "", "Read answer rates as a comparison between bands, not as an absolute level: "
              "the result map may under-count answers."]
    return "\n".join(lines) + "\n"


def delivered_overlap(first: pd.DataFrame, second: pd.DataFrame) -> float:
    """Share of `first` also present in `second`: how much two deliveries of a queue agree."""
    a, b = set(debtor_key(first)), set(debtor_key(second))
    return len(a & b) / len(a) if a else float("nan")

