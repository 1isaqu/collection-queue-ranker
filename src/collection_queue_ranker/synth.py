"""Synthetic data in the shape of the data contract. Every value is invented.

It exists so the whole pipeline (dataset, training, backtest, queue, monitoring) runs end to end
without a database. The generator has a hidden "willingness" per debtor and a quality per phone.
Agreements are triggered by answered calls, willing debtors keep their promises more often, and the
operation dials willing debtors more often. That last link is on purpose: it is the feedback loop
that makes attempt-count features look predictive without being usable, and the reason they are not
features here.

The numbers it produces say nothing about any real operation.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from collection_queue_ranker.contract import CONTRACT

DEFAULT_END = "2024-12-31"
NUM_AGENTS = 10
AUTOMATED_AGENT_ID = 0
TEST_AGENT_ID = NUM_AGENTS + 1

STATUSES = pd.DataFrame({
    "status_id": range(1, 8),
    "label": ["ACTIVE", "PAID", "PAID_MANUAL", "BROKEN", "AUTO_BROKEN",
              "PENDING_EXCEPTION", "REJECTED"],
})
_STATUS_ID = dict(zip(STATUSES["label"], STATUSES["status_id"], strict=True))

# (result_id, label, answered, right_party, category)
RESULTS = pd.DataFrame(
    [
        (1, "No answer after ringing", 0, 0, "no_answer"),
        (2, "Line busy", 0, 0, "busy"),
        (3, "Answering machine", 0, 0, "voicemail"),
        (4, "Number disconnected", 0, 0, "invalid_number"),
        (5, "Number does not exist", 0, 0, "invalid_number"),
        (6, "Automated message sent", 0, 0, "message"),
        (7, "Wrong party answered", 1, 0, "wrong_party"),
        (8, "Spoke with debtor", 1, 1, "answered"),
        (9, "Debtor asked for a callback", 1, 1, "answered"),
        (10, "Debtor hung up", 1, 1, "answered"),
        (11, "Promise to pay recorded", 1, 1, "promise"),
        (12, "Negotiation started", 1, 1, "promise"),
        (13, "Unclassified system result", 0, 0, "other"),
    ],
    columns=["result_id", "label", "answered", "right_party", "category"],
)
_BY_CATEGORY = {c: g["result_id"].tolist() for c, g in RESULTS.groupby("category")}


_PRODUCT_EFFECT = {1: 0.5, 2: 0.4, 3: 0.0, 4: -0.2, 5: -0.5}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _agents() -> pd.DataFrame:
    rows = [(AUTOMATED_AGENT_ID, "Auto Dialer 01", "autodialer_01")]
    rows += [(i, f"Agent {i:02d}", f"agent_{i:02d}") for i in range(1, NUM_AGENTS + 1)]
    rows.append((TEST_AGENT_ID, "Test User", "test_user"))
    return pd.DataFrame(rows, columns=["agent_id", "display_name", "login"])


class _Line:
    """Generates one product line. Debtor ids restart at 1 in every line on purpose."""

    def __init__(self, n: int, seed: int, end: pd.Timestamp, months: int) -> None:
        self.n, self.end = n, end
        self.start = end - pd.DateOffset(months=months)
        self.rng = np.random.default_rng(seed)
        self.phones: list[tuple] = []
        self.emails: list[tuple] = []
        self.debts: list[tuple] = []
        self.events: dict[str, list[np.ndarray]] = {
            "debtor_id": [], "result_id": [], "agent_id": [], "phone_id": [], "occurred_at": []
        }
        self.agreements: list[tuple] = []
        self.links: list[tuple] = []
        self.phone_id = self.email_id = self.debt_id = 0
        self.agreement_id = self.row_id = 0
        self.closed_at: dict[int, pd.Timestamp] = {}

    @staticmethod
    def _day(base: pd.Timestamp, days: float) -> pd.Timestamp:
        """`base` shifted by whole days (bases are midnight, so the result stays at midnight)."""
        return base + pd.Timedelta(days=int(days))

    def _make_debts(
        self, d: int
    ) -> tuple[float, pd.Timestamp, pd.Timestamp, list[int], int]:
        r = self.rng
        span = max((self.end - self.start).days - 30, 1)
        entered = self._day(self.start, r.integers(0, span))
        product = int(r.integers(1, 6))
        total, ids, due_min = 0.0, [], None
        for k in range(int(r.integers(1, 4))):
            self.debt_id += 1
            balance = float(np.round(r.lognormal(7.0, 0.9) * (1 + 0.1 * (product == 3)), 2))
            due = self._day(entered, -r.integers(20, 600))
            when = self._day(entered, 0 if k == 0 else r.integers(0, 60))
            self.debts.append((self.debt_id, d, when, due, balance, balance, int(r.integers(1, 4)),
                               product, int(r.integers(1, 13)), f"C-{self.debt_id:06d}"))
            total += balance
            ids.append(self.debt_id)
            due_min = due if due_min is None else min(due_min, due)
        return total, entered, due_min, ids, product

    def _make_phones(self, d: int, entered: pd.Timestamp) -> list[tuple[int, float, int]]:
        r, made = self.rng, []
        for _ in range(1 + int(min(r.poisson(1.2), 4))):
            self.phone_id += 1
            valid = int(r.random() < 0.85)
            quality = float(r.uniform(0.04, 0.6)) * valid
            created = self._day(entered, r.integers(-5, 40))
            updated = min(self._day(created, r.integers(0, 400)), self.end)
            mobile = int(r.random() < 0.6)
            self.phones.append((self.phone_id, d, valid, mobile, int(mobile and r.random() < 0.55),
                                int(r.random() < 0.1), created, updated))
            made.append((self.phone_id, quality, valid))
        for _ in range(int(r.poisson(0.8))):
            self.email_id += 1
            self.emails.append((self.email_id, d, self._day(entered, r.integers(-5, 40))))
        return made

    def _categories(
        self, auto: np.ndarray, quality: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Vectorised outcome of each attempt: depends on the phone and on who dialled."""
        r, n = self.rng, len(auto)
        u_msg, u_ans, u_sub = r.random(n), r.random(n), r.random(n)
        message = auto & (u_msg < 0.25)
        answered = ~message & (valid == 1) & (u_ans < quality * np.where(auto, 0.5, 1.0))
        conditions = [
            message,
            answered & (u_sub < 0.6), answered & (u_sub >= 0.6) & (u_sub < 0.8), answered,
            (valid == 0) & (u_sub < 0.7), valid == 0,
            u_sub < 0.62, u_sub < 0.82, u_sub < 0.92,
        ]
        names = ["message", "answered", "promise", "wrong_party", "invalid_number", "no_answer",
                 "no_answer", "voicemail", "busy"]
        return np.select(conditions, names, default="invalid_number").astype(object)

    def _make_events(
        self, d: int, entered: pd.Timestamp, will: float, phones: list[tuple[int, float, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one debtor's attempts. Returns (timestamps, categories) for the agreement step."""
        r = self.rng
        days = max((self.end - entered).days, 1)
        count = min(int(r.poisson(0.05 * np.exp(0.4 * will) * days)), 60)
        if count == 0:
            return np.array([], dtype="datetime64[ns]"), np.array([], dtype=object)
        secs = np.sort(np.floor(r.uniform(3600, days * 86400, count))).astype("int64")
        stamps = entered.to_datetime64().astype("datetime64[s]") + secs.astype("timedelta64[s]")
        auto = r.random(count) < 0.55
        agents = np.where(auto, AUTOMATED_AGENT_ID, r.integers(1, NUM_AGENTS + 1, count))
        agents = np.where(r.random(count) < 0.01, TEST_AGENT_ID, agents)
        pick = r.integers(0, len(phones), count)
        table = np.array(phones, dtype=float)
        categories = self._categories(auto, table[pick, 1], table[pick, 2].astype(int))
        offsets = r.integers(0, 1000, count)
        results = np.empty(count, dtype=int)
        for category, ids in _BY_CATEGORY.items():
            mask = categories == category
            results[mask] = np.array(ids)[offsets[mask] % len(ids)]
        for key, values in (
            ("debtor_id", np.full(count, d)), ("result_id", results), ("agent_id", agents),
            ("phone_id", table[pick, 0].astype(int)), ("occurred_at", stamps),
        ):
            self.events[key].append(values)
        return stamps, categories

    def _agreement_chance(self, will: float, total: float, overdue: float, category: str) -> float:
        logit = -2.1 + 0.9 * will - 0.25 * np.log(max(total, 1.0) / 1100.0) - 0.002 * (
            overdue - 200)
        return _sigmoid(logit + (0.7 if category == "promise" else 0.0))

    def _add_agreement(
        self, d: int, issued: pd.Timestamp, will: float, total: float, debts: list[int]
    ) -> None:
        r = self.rng
        self.agreement_id += 1
        plan = int(r.choice([1, 3, 6, 12], p=[0.25, 0.35, 0.25, 0.15]))
        amount = round(total * float(r.uniform(0.35, 0.8)) / plan, 2)
        verdict = r.random()
        pending = "PENDING_EXCEPTION" if verdict < 0.03 else "REJECTED" if verdict < 0.05 else None
        rows, state, last_paid = self._installments(issued, will, total, plan, amount, pending)
        if r.random() < 0.03:  # an older, out-of-scope draft of the same installments
            self._write_rows(d, issued - pd.Timedelta(days=1), amount, rows,
                             "PENDING_EXCEPTION", blank=True)
        self._write_rows(d, issued, amount, rows, pending or state)
        for debt in debts:
            self.links.append((self.agreement_id, 1, debt))
            if state in ("PAID", "PAID_MANUAL") and last_paid is not None and not pending:
                self.closed_at[debt] = last_paid

    def _write_rows(
        self, d: int, issued: pd.Timestamp, amount: float, rows: list[tuple], status: str,
        blank: bool = False,
    ) -> None:
        for k, (due, paid_amount, paid_at) in enumerate(rows):
            self.row_id += 1
            self.agreements.append((
                self.agreement_id, 1, d, k, amount, 0.0 if blank else paid_amount, issued,
                pd.NaT if blank else paid_at, due, _STATUS_ID[status],
                int(self.rng.integers(1, NUM_AGENTS + 1)), self.row_id))

    def _installments(
        self, issued: pd.Timestamp, will: float, total: float, plan: int, amount: float,
        pending: str | None,
    ) -> tuple[list[tuple], str, pd.Timestamp | None]:
        r = self.rng
        rows, broken, last_paid = [], False, None
        for k in range(plan):
            due = self._day(issued, int(r.integers(5, 16)) + 30 * k)
            paid_at, paid = pd.NaT, 0.0
            if not broken and not pending and due <= self.end:
                p_pay = _sigmoid((0.4 if k == 0 else 1.5) + 0.9 * will - 0.12 * k
                                 - 0.25 * np.log(max(total, 1.0) / 1100.0))
                if r.random() < p_pay:
                    when = self._day(due, int(r.integers(-2, 7)))
                    if when <= self.end:
                        paid_at, paid, last_paid = when, amount, when
                else:
                    broken = True
            rows.append((due, paid, paid_at))
        if pending:
            return rows, pending, None
        if broken:
            return rows, "AUTO_BROKEN" if r.random() < 0.4 else "BROKEN", last_paid
        if all(row[1] > 0 for row in rows):
            return rows, "PAID_MANUAL" if r.random() < 0.1 else "PAID", last_paid
        return rows, "ACTIVE", last_paid

    def _agreements_for(
        self, d: int, will: float, total: float, entered: pd.Timestamp, due_min: pd.Timestamp,
        debts: list[int], events: tuple[np.ndarray, np.ndarray],
    ) -> None:
        r = self.rng
        stamps, categories = events
        for i in np.flatnonzero(np.isin(categories, ["answered", "promise"])):
            ts = pd.Timestamp(stamps[i])
            overdue = float((ts - due_min).days)
            if r.random() < self._agreement_chance(will, total, overdue, categories[i]):
                issued = min(self._day(ts.normalize(), int(r.integers(0, 6))), self.end)
                self._add_agreement(d, issued, will, total, debts)
                return
        if r.random() < 0.03:  # legacy: issued long before the debts entered the portfolio
            self._add_agreement(d, self._day(entered, -r.integers(200, 400)), will, total, debts)

    def build(self) -> dict[str, pd.DataFrame]:
        r = self.rng
        company = (r.random(self.n) < 0.08).astype(int)
        for d in range(1, self.n + 1):
            total, entered, due_min, debt_ids, product = self._make_debts(d)
            # willingness has an observable part (product, company) and a hidden part
            will = float(r.normal()) + _PRODUCT_EFFECT[product] - 0.4 * company[d - 1]
            phones = self._make_phones(d, entered)
            events = self._make_events(d, entered, will, phones)
            self._agreements_for(d, will, total, entered, due_min, debt_ids, events)
        return self._frames(company)

    def _event_frame(self) -> pd.DataFrame:
        if not self.events["debtor_id"]:
            return pd.DataFrame(columns=["event_id", "debtor_id", "result_id", "agent_id",
                                         "phone_id", "occurred_at"])
        frame = pd.DataFrame({k: np.concatenate(v) for k, v in self.events.items()})
        frame.insert(0, "event_id", np.arange(1, len(frame) + 1))
        return frame

    def _frames(self, company: np.ndarray) -> dict[str, pd.DataFrame]:
        debts = pd.DataFrame(self.debts, columns=[
            "debt_id", "debtor_id", "entered_at", "due_at", "balance", "original_balance",
            "portfolio_id", "product_id", "phase_id", "contract_no"])
        debts["closed_at"] = debts["debt_id"].map(self.closed_at)
        agreements = pd.DataFrame(self.agreements, columns=[
            "agreement_id", "portfolio_id", "debtor_id", "installment_no", "amount",
            "paid_amount", "issued_at", "paid_at", "due_at", "status_id", "agent_id", "row_id"])
        return {
            "debtors": pd.DataFrame({"debtor_id": range(1, self.n + 1), "is_company": company}),
            "debts": debts,
            "debtor_phones": pd.DataFrame(self.phones, columns=[
                "phone_id", "debtor_id", "is_valid", "is_mobile", "has_messaging", "is_priority",
                "created_at", "updated_at"]),
            "debtor_emails": pd.DataFrame(self.emails, columns=["email_id", "debtor_id",
                                                                "created_at"]),
            "contact_events": self._event_frame(),
            "contact_results": RESULTS.copy(),
            "agreements": agreements,
            "agreement_debts": pd.DataFrame(self.links, columns=["agreement_id", "portfolio_id",
                                                                 "debt_id"]),
            "agreement_statuses": STATUSES.copy(),
            "agents": _agents(),
        }


def generate(
    n_debtors: int = 4000, seed: int = 7, end: str = DEFAULT_END, months: int = 24,
    line_offset: int = 0,
) -> dict[str, pd.DataFrame]:
    """Contract tables for one product line. `line_offset` makes lines differ for the same seed."""
    return _Line(n_debtors, seed + 1000 * line_offset, pd.Timestamp(end), months).build()


def write_sqlite(frames: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Write the frames as SQLite tables named after the contract. Dates are ISO text."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    with closing(sqlite3.connect(target)) as conn:
        for name, frame in frames.items():
            out = frame.copy()
            for col in CONTRACT[name].datetimes:
                if col in out:
                    out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
            out.to_sql(name, conn, index=False)
        conn.commit()
    return target


def build_demo(
    out_dir: str | Path, n_debtors: int = 4000, seed: int = 7, end: str = DEFAULT_END
) -> dict[str, Path]:
    """Write `auto.db` and `consumer.db` under `out_dir`; the second line is smaller."""
    out = Path(out_dir)
    sizes = {"auto": n_debtors, "consumer": max(n_debtors * 2 // 3, 50)}
    paths = {}
    for offset, (line, n) in enumerate(sizes.items()):
        frames = generate(n, seed, end, line_offset=offset)
        paths[line] = write_sqlite(frames, out / f"{line}.db")
    return paths
