"""Helpers that build tiny, invented contract frames for tests. No realistic rows, no files."""

from __future__ import annotations

import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.tables import RawTables

ts = pd.Timestamp
NaT = pd.NaT

STATUS_IDS = {
    "ACTIVE": 1, "PAID": 2, "PAID_MANUAL": 3, "BROKEN": 4, "AUTO_BROKEN": 5,
    "PENDING_EXCEPTION": 6, "REJECTED": 7,
}


def statuses_df() -> pd.DataFrame:
    return pd.DataFrame({"status_id": list(STATUS_IDS.values()), "label": list(STATUS_IDS)})


def results_df() -> pd.DataFrame:
    """answered / right_party flags only: categories are derived (the contract minimum)."""
    return pd.DataFrame(
        [(1, "no answer", 0, 0), (2, "wrong party", 1, 0), (3, "spoke", 1, 1)],
        columns=["result_id", "label", "answered", "right_party"],
    )


def debts_df(*rows: tuple) -> pd.DataFrame:
    """(debt_id, debtor_id, entered_at, due_at, balance[, closed_at[, product_id[, phase_id]]])."""
    out = [list(r) + [None] * (8 - len(r)) for r in rows]
    frame = pd.DataFrame(out, columns=[
        "debt_id", "debtor_id", "entered_at", "due_at", "balance", "closed_at", "product_id",
        "phase_id"])
    frame["closed_at"] = pd.to_datetime(frame["closed_at"])
    frame["product_id"] = frame["product_id"].fillna("p1")
    frame["phase_id"] = frame["phase_id"].fillna("f1")
    return frame


def events_df(*rows: tuple) -> pd.DataFrame:
    """(event_id, debtor_id, result_id, occurred_at[, agent_id[, phone_id]])."""
    padded = [list(r) + [None] * (6 - len(r)) for r in rows]
    return pd.DataFrame(padded, columns=[
        "event_id", "debtor_id", "result_id", "occurred_at", "agent_id", "phone_id"])


def phones_df(*rows: tuple) -> pd.DataFrame:
    """(phone_id, debtor_id[, is_valid, is_mobile, has_messaging, is_priority, created, updated])."""
    defaults = [1, 0, 0, 0, NaT, NaT]
    padded = [list(r) + defaults[len(r) - 2:] for r in rows]
    return pd.DataFrame(padded, columns=[
        "phone_id", "debtor_id", "is_valid", "is_mobile", "has_messaging", "is_priority",
        "created_at", "updated_at"])


def agreements_df(*rows: tuple) -> pd.DataFrame:
    """(agreement_id, debtor_id, installment_no, amount, paid_amount, issued_at, paid_at, status)."""
    out = [
        (a, 1, d, k, amt, paid, issued, paid_at, STATUS_IDS[status], i)
        for i, (a, d, k, amt, paid, issued, paid_at, status) in enumerate(rows)
    ]
    return pd.DataFrame(out, columns=[
        "agreement_id", "portfolio_id", "debtor_id", "installment_no", "amount", "paid_amount",
        "issued_at", "paid_at", "status_id", "row_id"])


def make_raw(settings: Settings | None = None, **frames: pd.DataFrame) -> RawTables:
    """Build `RawTables` from frames keyed by contract table name; the rest stay empty."""
    names = {
        "debts": "debts", "events": "contact_events", "phones": "debtor_phones",
        "agreements": "agreements", "debtors": "debtors", "emails": "debtor_emails",
        "links": "agreement_debts", "results": "contact_results", "agents": "agents",
        "statuses": "agreement_statuses",
    }
    mapped = {names[k]: v for k, v in frames.items()}
    mapped.setdefault("agreement_statuses", statuses_df())
    mapped.setdefault("contact_results", results_df())
    return RawTables.from_frames(mapped, settings or Settings(), default_line="auto")
