"""In-memory form of the contract tables, with the type clean-up done once."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.contacts import annotate_events
from collection_queue_ranker.contract import CONTRACT, TableSpec
from collection_queue_ranker.errors import DataError

KEYS = ["product_line", "debtor_id"]

_NUMERIC = (
    "balance", "original_balance", "amount", "paid_amount", "installment_no", "days_overdue",
)


def normalise_frame(spec: TableSpec, df: pd.DataFrame, default_line: str = "auto") -> pd.DataFrame:
    """Lower-case names, add absent optional columns, parse dates, coerce flags and numbers."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "product_line" not in out:
        out["product_line"] = default_line
    flags = dict(spec.flags)
    for col in spec.columns:
        if col in out:
            continue
        if col in flags:
            out[col] = flags[col]
        elif col in spec.datetimes:
            out[col] = pd.NaT
        else:
            out[col] = np.nan
    for col in spec.datetimes:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col, default in flags.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(int)
    for col in _NUMERIC:
        if col in out and col in spec.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    extras = [c for c in out.columns if c not in spec.columns and c != "product_line"]
    return out[["product_line", *spec.columns, *extras]].reset_index(drop=True)


@dataclass
class RawTables:
    debtors: pd.DataFrame
    debts: pd.DataFrame
    phones: pd.DataFrame
    emails: pd.DataFrame
    events: pd.DataFrame
    results: pd.DataFrame
    agreements: pd.DataFrame
    agreement_debts: pd.DataFrame
    statuses: pd.DataFrame
    agents: pd.DataFrame

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        settings: Settings | None = None,
        default_line: str = "auto",
    ) -> RawTables:
        """Normalise frames keyed by contract table name. Absent tables become empty frames."""
        settings = settings or Settings()
        norm = {
            name: normalise_frame(
                spec, frames.get(name, pd.DataFrame(columns=list(spec.columns))), default_line
            )
            for name, spec in CONTRACT.items()
        }
        agr = norm["agreements"]
        if agr["row_id"].isna().all():
            agr["row_id"] = np.arange(len(agr))
        events = annotate_events(
            norm["contact_events"], norm["contact_results"], norm["agents"], settings.agents
        )
        return cls(
            debtors=norm["debtors"], debts=norm["debts"], phones=norm["debtor_phones"],
            emails=norm["debtor_emails"], events=events, results=norm["contact_results"],
            agreements=agr, agreement_debts=norm["agreement_debts"],
            statuses=norm["agreement_statuses"], agents=norm["agents"],
        )

    @property
    def product_lines(self) -> list[str]:
        return sorted(self.debts["product_line"].dropna().unique().tolist())

    def latest_timestamp(self) -> pd.Timestamp:
        """Latest date seen in any time column; the default `as_of` is the next day."""
        stamps = [
            self.debts["entered_at"].max(),
            self.events["occurred_at"].max(),
            self.agreements["issued_at"].max(),
            self.agreements["paid_at"].max(),
        ]
        valid = [s for s in stamps if pd.notna(s)]
        if not valid:
            raise DataError("no timestamps found in the data")
        return max(valid).normalize()
