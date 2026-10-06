"""Debt-level features measured at an anchor date.

Everything is computed from rows dated strictly before the anchor, and a debt only counts while it
is open at the anchor (entered before it, not closed before it). The same function serves training
snapshots, the break cohort and the live queue, so the definition cannot drift between them.
"""

from __future__ import annotations

import pandas as pd

from collection_queue_ranker.tables import KEYS

BASE_NUMERIC = [
    "balance", "days_overdue_max", "days_overdue_min", "overdue_spread", "is_company",
    "phone_count", "email_count",
]
BASE_CATEGORICAL = ["product_line", "product_id", "phase_id"]
# Computed alongside the features but never fed to a model.
CONTEXT_COLUMNS = ["tenure_days"]


def make_anchors(debts: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Every debtor with at least one debt open at `cutoff`, paired with the cutoff."""
    open_now = (debts["entered_at"] < cutoff) & (
        debts["closed_at"].isna() | (debts["closed_at"] >= cutoff)
    )
    anchors = debts.loc[open_now, KEYS].drop_duplicates()
    return anchors.assign(anchor_at=pd.Timestamp(cutoff))


def _count_before(
    table: pd.DataFrame, anchors: pd.DataFrame, name: str
) -> pd.DataFrame:
    """Rows per debtor created before the anchor (rows without a creation date always count)."""
    if table.empty:
        return pd.DataFrame(columns=[*KEYS, name])
    joined = table[[*KEYS, "created_at"]].merge(anchors, on=KEYS, how="inner")
    known = joined["created_at"].isna() | (joined["created_at"] < joined["anchor_at"])
    return joined[known].groupby(KEYS).size().rename(name).reset_index()


def _open_debts(debts: pd.DataFrame, anchors: pd.DataFrame, balance_column: str) -> pd.DataFrame:
    d = debts.merge(anchors, on=KEYS, how="inner")
    is_open = (d["entered_at"] < d["anchor_at"]) & (
        d["closed_at"].isna() | (d["closed_at"] >= d["anchor_at"])
    )
    d = d[is_open].copy()
    d["_balance"] = pd.to_numeric(d[balance_column], errors="coerce")
    for col in ("product_id", "phase_id"):
        d[col] = d[col].astype("string")
    return d


def base_features(
    debts: pd.DataFrame,
    debtors: pd.DataFrame,
    phones: pd.DataFrame,
    emails: pd.DataFrame,
    anchors: pd.DataFrame,
    balance_column: str = "balance",
) -> pd.DataFrame:
    """One row per anchored debtor that still has an open debt.

    Phone and e-mail counts honour `created_at` when it is filled. Without it they are the count
    at extraction time, which is not point-in-time (documented limitation).
    """
    d = _open_debts(debts, anchors, balance_column)
    grouped = d.groupby(KEYS, as_index=False).agg(
        anchor_at=("anchor_at", "first"), balance=("_balance", "sum"),
        due_min=("due_at", "min"), due_max=("due_at", "max"), entered_min=("entered_at", "min"),
        product_id=("product_id", "max"), phase_id=("phase_id", "max"),
    )
    grouped["days_overdue_max"] = (grouped["anchor_at"] - grouped["due_min"]).dt.days
    grouped["days_overdue_min"] = (grouped["anchor_at"] - grouped["due_max"]).dt.days
    grouped["overdue_spread"] = grouped["days_overdue_max"] - grouped["days_overdue_min"]
    grouped["tenure_days"] = (grouped["anchor_at"] - grouped["entered_min"]).dt.days
    grouped = grouped.drop(columns=["due_min", "due_max", "entered_min"])
    grouped = _attach_company(grouped, debtors)
    for table, name in ((phones, "phone_count"), (emails, "email_count")):
        counts = _count_before(table, anchors, name)
        if counts.empty:
            grouped[name] = 0
            continue
        grouped = grouped.merge(counts, on=KEYS, how="left")
        grouped[name] = grouped[name].fillna(0).astype(int)
    for col in ("product_id", "phase_id"):
        grouped[col] = grouped[col].fillna("NA")
    return grouped


def _attach_company(frame: pd.DataFrame, debtors: pd.DataFrame) -> pd.DataFrame:
    if debtors.empty:
        return frame.assign(is_company=0)
    merged = frame.merge(
        debtors[[*KEYS, "is_company"]].drop_duplicates(KEYS), on=KEYS, how="left"
    )
    merged["is_company"] = merged["is_company"].fillna(0).astype(int)
    return merged
