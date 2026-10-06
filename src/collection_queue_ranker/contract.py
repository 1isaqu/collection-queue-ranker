"""The data contract: which tables and columns the package expects.

You map your own database to these names (usually with views). `check_source` compares a live
source against the contract and reports what is missing, so a wrong mapping fails early with a
readable list instead of deep inside feature building. See `docs/data-contract.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from collection_queue_ranker.errors import ContractError


@dataclass(frozen=True)
class TableSpec:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    datetimes: tuple[str, ...] = ()
    # bit columns and the value a NULL means
    flags: tuple[tuple[str, int], ...] = ()
    # a table the package can run without (its features degrade gracefully)
    optional_table: bool = False

    @property
    def columns(self) -> tuple[str, ...]:
        return self.required + self.optional


CONTRACT: dict[str, TableSpec] = {
    "debtors": TableSpec(
        "debtors", ("debtor_id",), ("is_company",), flags=(("is_company", 0),),
        optional_table=True,
    ),
    "debts": TableSpec(
        "debts",
        ("debt_id", "debtor_id", "entered_at", "due_at", "balance"),
        (
            "portfolio_id", "contract_no", "days_overdue", "product_id", "phase_id",
            "closed_at", "original_balance",
        ),
        datetimes=("entered_at", "due_at", "closed_at"),
    ),
    "debtor_phones": TableSpec(
        "debtor_phones",
        ("phone_id", "debtor_id"),
        ("is_valid", "is_mobile", "has_messaging", "is_priority", "created_at", "updated_at"),
        datetimes=("created_at", "updated_at"),
        flags=(
            ("is_valid", 1), ("is_mobile", 0), ("has_messaging", 0), ("is_priority", 0),
        ),
    ),
    "debtor_emails": TableSpec(
        "debtor_emails", ("email_id", "debtor_id"), ("created_at",), datetimes=("created_at",),
        optional_table=True,
    ),
    "contact_events": TableSpec(
        "contact_events",
        ("event_id", "debtor_id", "result_id", "occurred_at"),
        ("agent_id", "phone_id"),
        datetimes=("occurred_at",),
    ),
    "contact_results": TableSpec(
        "contact_results",
        ("result_id", "answered", "right_party"),
        ("label", "category"),
        flags=(("answered", 0), ("right_party", 0)),
    ),
    "agreements": TableSpec(
        "agreements",
        (
            "agreement_id", "portfolio_id", "debtor_id", "installment_no", "amount",
            "paid_amount", "issued_at", "paid_at", "status_id",
        ),
        ("due_at", "agent_id", "row_id"),
        datetimes=("issued_at", "paid_at", "due_at"),
    ),
    "agreement_debts": TableSpec(
        "agreement_debts", ("agreement_id", "portfolio_id", "debt_id"), optional_table=True
    ),
    "agreement_statuses": TableSpec("agreement_statuses", ("status_id", "label")),
    "agents": TableSpec(
        "agents", ("agent_id", "login"), ("display_name",), optional_table=True
    ),
}


class ColumnSource(Protocol):
    """The part of a data source `check_source` needs."""

    name: str

    def table_columns(self, table: str) -> list[str]: ...


@dataclass(frozen=True)
class TableCheck:
    table: str
    present: bool
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]

    optional_table: bool = False

    @property
    def ok(self) -> bool:
        if not self.present:
            return self.optional_table
        return not self.missing_required


def check_table(spec: TableSpec, found_columns: Iterable[str]) -> TableCheck:
    found = {c.lower() for c in found_columns}
    return TableCheck(
        table=spec.name,
        present=bool(found),
        missing_required=tuple(c for c in spec.required if c not in found),
        missing_optional=tuple(c for c in spec.optional if c not in found),
        optional_table=spec.optional_table,
    )


def read_available(
    source: ColumnSource, table_map: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Ask `source` which columns each contract table has (physical names from `table_map`)."""
    table_map = table_map or {}
    return {name: source.table_columns(table_map.get(name, name)) for name in CONTRACT}


def check_available(available: dict[str, list[str]]) -> list[TableCheck]:
    return [check_table(spec, available.get(name, ())) for name, spec in CONTRACT.items()]


def check_source(
    source: ColumnSource, table_map: dict[str, str] | None = None
) -> list[TableCheck]:
    """Check every contract table against `source`. Does not raise; see `require_ok`."""
    return check_available(read_available(source, table_map))


def require_ok(checks: Sequence[TableCheck], source_name: str) -> None:
    """Raise `ContractError` listing every table or required column that is missing."""
    problems = []
    for check in checks:
        if check.ok:
            continue
        if not check.present:
            problems.append(f"table `{check.table}` not found")
        else:
            problems.append(f"`{check.table}` lacks {list(check.missing_required)}")
    if problems:
        raise ContractError(f"source `{source_name}` does not satisfy the contract: "
                            + "; ".join(problems))


def format_checks(checks: Sequence[TableCheck]) -> str:
    """Plain-text report for `cqr check-contract`."""
    lines = []
    for check in checks:
        if not check.present:
            tag = "absent  " if check.optional_table else "MISSING "
            lines.append(f"{tag} {check.table}")
        elif check.missing_required:
            lines.append(f"FAIL     {check.table}: missing required {list(check.missing_required)}")
        else:
            note = f" (optional absent: {list(check.missing_optional)})" if (
                check.missing_optional) else ""
            lines.append(f"ok       {check.table}{note}")
    return "\n".join(lines)
