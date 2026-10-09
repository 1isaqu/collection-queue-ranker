"""What counts as an agreement, and the outcomes that become labels.

The rules below are stated against the generic contract:

* A **valid agreement** is the first installment (`installment_no = 0`) of an agreement whose status
  is one of the configured paid / broken / active labels. Pending exceptions and rejections are not
  agreements.
* Installment rows can be re-written. Per (agreement, portfolio, installment) the latest row wins,
  after the status filter, so a re-write with an out-of-scope status cannot hide a valid row.
* An agreement issued long before the debt it settles entered the portfolio is **legacy history**
  that arrived with the portfolio, not something the operation produced. It is dropped.
* An agreement that is still active is **right-censored**: it may yet be paid or broken, so it is not
  a negative example for the break head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings, StatusSettings
from collection_queue_ranker.tables import KEYS, RawTables

AGREEMENT_KEYS = ["product_line", "agreement_id", "portfolio_id"]
INSTALLMENT_KEYS = [*AGREEMENT_KEYS, "installment_no"]


def valid_installments(raw: RawTables, statuses: StatusSettings) -> pd.DataFrame:
    """Installment rows with a valid status, one per installment (latest re-write wins)."""
    labels = raw.statuses[["product_line", "status_id", "label"]].rename(
        columns={"label": "status"}
    )
    inst = raw.agreements.merge(labels, on=["product_line", "status_id"], how="left")
    inst = inst[inst["status"].isin(statuses.valid)]
    inst = inst.sort_values(["issued_at", "row_id"], kind="stable")
    return inst.drop_duplicates(INSTALLMENT_KEYS, keep="last").reset_index(drop=True)


def _outcome(status: pd.Series, statuses: StatusSettings) -> pd.Series:
    out = pd.Series("active", index=status.index, dtype="object")
    out[status.isin(statuses.paid)] = "paid"
    out[status.isin(statuses.broken)] = "broken"
    return out


def _drop_legacy(first: pd.DataFrame, raw: RawTables, tolerance_days: int) -> pd.DataFrame:
    """Drop agreements issued more than `tolerance_days` before every debt they settle.

    An agreement with no recorded debt link cannot be judged and is kept.
    """
    if raw.agreement_debts.empty or first.empty:
        return first
    debts = raw.debts[["product_line", "debt_id", "entered_at"]]
    links = raw.agreement_debts.merge(debts, on=["product_line", "debt_id"], how="inner")
    links = links.dropna(subset=["entered_at"])
    joined = first[[*AGREEMENT_KEYS, "issued_at"]].merge(
        links[[*AGREEMENT_KEYS, "entered_at"]], on=AGREEMENT_KEYS, how="inner"
    )
    cutoff = joined["entered_at"] - pd.Timedelta(days=tolerance_days)
    joined["fresh"] = joined["issued_at"] >= cutoff
    verdict = joined.groupby(AGREEMENT_KEYS)["fresh"].any()
    legacy = verdict[~verdict].index
    if len(legacy) == 0:
        return first
    marked = first.set_index(AGREEMENT_KEYS).index.isin(legacy)
    return first[~marked].reset_index(drop=True)


@dataclass
class Agreements:
    """Valid agreements, prepared once and reused for every cutoff."""

    first: pd.DataFrame
    installments: pd.DataFrame


def prepare_agreements(raw: RawTables, settings: Settings) -> Agreements:
    """Valid first installments (with outcome) and every installment of those agreements."""
    inst = valid_installments(raw, settings.statuses)
    first = inst[inst["installment_no"] == 0].copy()
    first["outcome"] = _outcome(first["status"], settings.statuses)
    first = _drop_legacy(first, raw, settings.labels.legacy_tolerance_days)
    first = first[[*AGREEMENT_KEYS, "debtor_id", "issued_at", "status", "outcome"]]
    issue = first[[*AGREEMENT_KEYS, "issued_at"]].rename(columns={"issued_at": "first_issued_at"})
    paid = inst.merge(issue, on=AGREEMENT_KEYS, how="inner")
    return Agreements(first=first.reset_index(drop=True), installments=paid)


def agreements_before(prep: Agreements, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Debtors who already hold a valid agreement issued before `cutoff`."""
    done = prep.first[prep.first["issued_at"] < cutoff]
    return done[KEYS].drop_duplicates()


def window_labels(
    prep: Agreements, cutoff: pd.Timestamp, horizon_days: int
) -> pd.DataFrame:
    """Per debtor: did a valid agreement start in [cutoff, cutoff + horizon), and what was paid.

    `realized_value` is the money received inside the same window from agreements that started in
    it. It is used only to judge and to grade, never as a feature.
    """
    end = cutoff + pd.Timedelta(days=horizon_days)
    started = prep.first[(prep.first["issued_at"] >= cutoff) & (prep.first["issued_at"] < end)]
    closed = started[KEYS].drop_duplicates().assign(closed=1)
    inst = prep.installments
    in_window = (
        (inst["first_issued_at"] >= cutoff) & (inst["first_issued_at"] < end)
        & (inst["paid_at"] >= cutoff) & (inst["paid_at"] < end) & (inst["paid_amount"] > 0)
    )
    value = (
        inst[in_window].groupby(KEYS, as_index=False)["paid_amount"].sum()
        .rename(columns={"paid_amount": "realized_value"})
    )
    return closed.merge(value, on=KEYS, how="outer")


def break_cohort(prep: Agreements, as_of: pd.Timestamp, settings: Settings) -> pd.DataFrame:
    """Debtors with a valid agreement and the outcome of their agreements.

    `anchor_at` is the first valid agreement date: features are measured then. `broke` is 1 when the
    debtor has no paid agreement. It is only meaningful for `resolved` rows (something was paid or
    broken). `mature` rows are old enough that the pay/break outcome had time to appear.
    """
    first = prep.first
    flags = first.assign(
        paid=first["outcome"].eq("paid"), broken=first["outcome"].eq("broken"),
        active=first["outcome"].eq("active"),
    )
    cohort = flags.groupby(KEYS, as_index=False).agg(
        anchor_at=("issued_at", "min"), n_paid=("paid", "sum"),
        n_broken=("broken", "sum"), n_active=("active", "sum"),
    )
    cohort["resolved"] = (cohort["n_paid"] > 0) | (cohort["n_broken"] > 0)
    limit = pd.Timestamp(as_of) - pd.Timedelta(days=settings.labels.break_observation_days)
    cohort["mature"] = cohort["anchor_at"] <= limit
    cohort["broke"] = np.where(cohort["n_paid"] == 0, 1, 0)
    return cohort
