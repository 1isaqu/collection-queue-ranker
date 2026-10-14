"""The public "Default of Credit Card Clients" panel, shaped as the datasets the ranker trains on.

The data is a six-month panel of 30,000 credit-card clients of a Taiwanese bank, April to September
2005: per client and month a repayment status, the bill statement and the amount paid, plus the
credit limit. It was donated by I-Cheng Yeh to the UCI Machine Learning Repository under CC BY 4.0
(https://doi.org/10.24432/C55S3H). This repository does not contain the data.

The panel has no contact history, no agreements and no installments, so it cannot go through the
contract tables. This module writes the files that `cqr build-dataset` would write (`snapshots.csv`,
`break.csv`, `dataset_meta.json`) and `cqr train` and `cqr backtest` then run on them unchanged. What
it exercises and what it cannot is in docs/real-data-evaluation.md. The mapping, in short:

* a debtor is a client in arrears at a monthly cutoff: repayment status of 1 or more and a positive
  bill. Clients can appear at several cutoffs, as repeat debtors do;
* the cutoff is the first day of the month after the statement, so features use that statement and
  the months before it, and the label window is the next month;
* the close label is "paid something in the next month" and the realised value is the amount paid.
  That is a stand-in for "signed an agreement", not the same event;
* the break label is "paid in the next month, then paid nothing in the month after", the nearest
  thing to a missed second installment;
* the contact, phone and e-mail features take the values the package gives a debtor with no
  contact history. Age, sex, education and marital status are not used.

Nothing here runs unless you call it: the library opens no connection by itself, and the tests use
small inline panels.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.base import BASE_CATEGORICAL, BASE_NUMERIC

DATA_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
DATA_SHA256 = "45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0"
DATA_FILE = "default_of_credit_card_clients.csv"
DATASET_DOI = "10.24432/C55S3H"
CITATION = (
    "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. "
    f"https://doi.org/{DATASET_DOI}"
)

PRODUCT_LINE = "card"
MONTHS = ("2005-04", "2005-05", "2005-06", "2005-07", "2005-08", "2005-09")
# A status of 1 or more is a payment delay. -2 (no consumption), -1 (paid duly) and 0 are not.
ARREARS_FROM = 1
DAYS_PER_DELAY = 30

# Column order is oldest month first, the reverse of the order in the source file.
_STATUS = ("X11", "X10", "X9", "X8", "X7", "X6")
_BILL = ("X17", "X16", "X15", "X14", "X13", "X12")
_PAID = ("X23", "X22", "X21", "X20", "X19", "X18")
_RAW_COLUMNS = ("ID", *(f"X{i}" for i in range(1, 24)), "Y")

EXTRA_NUMERIC = (
    "credit_limit", "utilisation", "paid_last_month", "paid_share_of_previous_bill",
    "paid_two_months_ago", "months_paid_recent", "months_in_arrears", "worst_delay_history",
    "bill_growth", "delay_change",
)

# What a JSON settings file would hold to run the stages on this panel. Everything else keeps the
# package default. Monthly cutoffs make the label window one month and the break window two; the
# development side has three cutoffs, so two walk-forward folds is the most it can make. The
# bootstrap is larger than the default only to tighten the published intervals.
SETTINGS_OVERRIDES: dict[str, object] = {
    "labels": {"horizon_days": 30, "break_observation_days": 60},
    "features": {"extra_numeric": list(EXTRA_NUMERIC)},
    "split": {"cv_folds": 2},
    "evaluation": {"n_bootstrap": 1000},
}


@dataclass(frozen=True)
class Panel:
    """The panel as arrays, one row per client and one column per month (oldest first)."""

    client_id: np.ndarray
    limit: np.ndarray
    status: np.ndarray
    bill: np.ndarray
    paid: np.ndarray
    default_next_month: np.ndarray

    def __len__(self) -> int:
        return len(self.client_id)


@dataclass(frozen=True)
class DataQuality:
    """What was read and what the population rules left out, so every exclusion can be reported."""

    clients: int
    negative_bill_cells: int
    status_one_by_month: tuple[int, ...]
    status_two_or_more_by_month: tuple[int, ...]
    late_by_cutoff: tuple[int, ...]
    late_without_positive_bill: tuple[int, ...]
    snapshot_rows: int
    snapshot_clients: int
    clients_at_several_cutoffs: int
    break_rows: int


@dataclass(frozen=True)
class CardDatasets:
    snapshots: pd.DataFrame
    break_df: pd.DataFrame
    as_of: pd.Timestamp
    quality: DataQuality


def card_settings(**overrides: object) -> Settings:
    """The package defaults plus `SETTINGS_OVERRIDES`; `overrides` are merged on top."""
    merged = Settings().with_overrides(SETTINGS_OVERRIDES)
    return merged.with_overrides(dict(overrides)) if overrides else merged


def check_hash(payload: bytes, expected: str) -> None:
    """Refuse bytes that are not the file the results were produced from."""
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise DataError(
            f"the downloaded file has sha256 {actual}, expected {expected}: it is not the file "
            "this evaluation was written for, so no number in the docs applies to it"
        )


def fetch(directory: str | Path = "data/uci_credit_card", timeout: float = 120.0) -> Path:
    """Download the CSV from the UCI repository unless it is already there, and verify its hash."""
    target = Path(directory) / DATA_FILE
    if target.is_file():
        try:
            check_hash(target.read_bytes(), DATA_SHA256)
            return target
        except DataError:
            pass  # a different file under our name: download again
    with urllib.request.urlopen(DATA_URL, timeout=timeout) as response:
        payload = response.read()
    check_hash(payload, DATA_SHA256)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def panel_from_frame(raw: pd.DataFrame) -> Panel:
    """Validate the source columns and arrange them as a `Panel`."""
    missing = [c for c in _RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"the credit-card panel lacks the column(s) {missing}")
    frame = raw[list(_RAW_COLUMNS)]
    numbers = frame.apply(pd.to_numeric, errors="coerce")
    if numbers.isna().any().any():
        raise DataError("the credit-card panel has a missing or non-numeric cell")
    if numbers["ID"].duplicated().any():
        raise DataError("the credit-card panel has a repeated client ID")
    if (numbers[list(_PAID)] < 0).any().any():
        raise DataError("the credit-card panel has a negative payment")
    status = numbers[list(_STATUS)].to_numpy()
    if status.min() < -2 or status.max() > 9:
        raise DataError("a repayment status is outside the documented range of -2 to 9")
    if not set(numbers["Y"].unique()) <= {0, 1}:
        raise DataError("the default flag has a value other than 0 and 1")
    return Panel(
        client_id=numbers["ID"].to_numpy(dtype=np.int64),
        limit=numbers["X1"].to_numpy(dtype=float),
        status=status.astype(int),
        bill=numbers[list(_BILL)].to_numpy(dtype=float),
        paid=numbers[list(_PAID)].to_numpy(dtype=float),
        default_next_month=numbers["Y"].to_numpy(dtype=int),
    )


def read_clients(path: str | Path) -> Panel:
    """Read the CSV written by `fetch`."""
    return panel_from_frame(pd.read_csv(path))


def cutoff_dates() -> pd.DatetimeIndex:
    """First day of the month after each statement month that has a following month."""
    return pd.DatetimeIndex(
        [pd.Timestamp(f"{m}-01") + pd.offsets.MonthBegin(1) for m in MONTHS[:-1]]
    )


def as_of_date() -> pd.Timestamp:
    """The day after the last month of the panel."""
    return pd.Timestamp(f"{MONTHS[-1]}-01") + pd.offsets.MonthBegin(1)


def _months_in_arrears(status: np.ndarray, t: int) -> np.ndarray:
    """Consecutive months up to and including month `t` with a payment delay."""
    run = np.zeros(len(status), dtype=int)
    alive = np.ones(len(status), dtype=bool)
    for u in range(t, -1, -1):
        alive &= status[:, u] >= ARREARS_FROM
        run += alive
    return run


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """`numerator / denominator` where the denominator is positive, NaN elsewhere."""
    out = np.full(len(numerator), np.nan)
    ok = denominator > 0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def _features(panel: Panel, rows: np.ndarray, t: int, contact_floor: float) -> pd.DataFrame:
    """Everything known at the end of month `t` about the clients in `rows`, with no label."""
    status, bill, paid = panel.status[rows], panel.bill[rows], panel.paid[rows]
    balance = bill[:, t]
    first = max(0, t - 2)
    nan = np.full(len(rows), np.nan)
    previous = t >= 1
    frame = pd.DataFrame({
        "product_line": PRODUCT_LINE,
        "debtor_id": panel.client_id[rows],
        "balance": balance,
        "days_overdue_max": DAYS_PER_DELAY * status[:, t],
        "days_overdue_min": DAYS_PER_DELAY * status[:, t],
        "overdue_spread": 0,
        "is_company": 0,
        "phone_count": 0,
        "email_count": 0,
        "product_id": "NA",
        "phase_id": "NA",
        "n_answered": 0,
        "n_wrong_party": 0,
        "n_invalid_number": 0,
        "answer_rate": np.nan,
        "days_since_answered": np.nan,
        "contact_factor": contact_floor,
        "credit_limit": panel.limit[rows],
        "utilisation": _ratio(balance, panel.limit[rows]),
        "paid_last_month": paid[:, t],
        "paid_share_of_previous_bill": _ratio(paid[:, t], bill[:, t - 1]) if previous else nan,
        "paid_two_months_ago": paid[:, t - 1] if previous else nan,
        "months_paid_recent": (paid[:, first:t + 1] > 0).sum(axis=1),
        "months_in_arrears": _months_in_arrears(status, t),
        "worst_delay_history": status[:, : t + 1].max(axis=1),
        "bill_growth": (
            _ratio(balance - bill[:, t - 1], bill[:, t - 1]) if previous else nan
        ),
        "delay_change": (status[:, t] - status[:, t - 1]).astype(float) if previous else nan,
    })
    return frame


def build_datasets(
    panel: Panel,
    settings: Settings | None = None,
    *,
    min_status: int = ARREARS_FROM,
    min_months_in_arrears: int = 1,
) -> CardDatasets:
    """Snapshots, break cohort and the `as_of` date, in the shape `cqr train` reads.

    One snapshot per statement month except the last: the last month has no following month to
    label. A client is in a snapshot when its repayment status in that month is `min_status` or
    more, it has been in arrears for `min_months_in_arrears` consecutive months or more, and its
    bill is above `snapshots.min_balance`. The defaults are the documented meaning of a payment
    delay; stricter values are for sensitivity checks.
    """
    settings = settings or card_settings()
    cutoffs = cutoff_dates()
    floor = settings.queue.contact_floor
    min_balance = settings.snapshots.min_balance
    frames: list[pd.DataFrame] = []
    late_counts: list[int] = []
    no_bill_counts: list[int] = []
    anchored: set[int] = set()
    break_frames: list[pd.DataFrame] = []
    last_break_month = len(MONTHS) - 3
    for t, cutoff in enumerate(cutoffs):
        late = (panel.status[:, t] >= min_status) & (
            _months_in_arrears(panel.status, t) >= min_months_in_arrears
        )
        rows = np.flatnonzero(late & (panel.bill[:, t] > min_balance))
        late_counts.append(int(late.sum()))
        no_bill_counts.append(int(late.sum()) - len(rows))
        if len(rows) == 0:
            continue
        frame = _features(panel, rows, t, floor)
        frame["snapshot_at"] = cutoff
        frame["realized_value"] = panel.paid[rows, t + 1]
        frame["closed"] = (frame["realized_value"] > 0).astype(int)
        frames.append(frame)
        if t <= last_break_month:
            cohort = frame[(frame["closed"] == 1) & ~frame["debtor_id"].isin(anchored)]
            anchored.update(cohort["debtor_id"].tolist())
            followed = panel.paid[rows[cohort.index.to_numpy()], t + 2]
            break_part = cohort[["debtor_id", *BASE_NUMERIC, *BASE_CATEGORICAL]]
            break_frames.append(
                break_part.assign(anchor_at=cutoff, broke=(followed == 0).astype(int))
            )
    if not frames:
        raise DataError("no client is in arrears with a positive bill at any cutoff")
    snapshots = pd.concat(frames, ignore_index=True)
    break_df = (
        pd.concat(break_frames, ignore_index=True) if break_frames
        else pd.DataFrame(columns=["anchor_at", "broke"])
    )
    counts = snapshots["debtor_id"].value_counts()
    quality = DataQuality(
        clients=len(panel),
        negative_bill_cells=int((panel.bill < 0).sum()),
        status_one_by_month=tuple(int(x) for x in (panel.status == 1).sum(axis=0)),
        status_two_or_more_by_month=tuple(int(x) for x in (panel.status >= 2).sum(axis=0)),
        late_by_cutoff=tuple(late_counts),
        late_without_positive_bill=tuple(no_bill_counts),
        snapshot_rows=len(snapshots),
        snapshot_clients=int(counts.size),
        clients_at_several_cutoffs=int((counts > 1).sum()),
        break_rows=len(break_df),
    )
    return CardDatasets(snapshots, break_df, as_of_date(), quality)
