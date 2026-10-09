# Data contract

The package reads ten tables with fixed names and column names. You map your own database to them,
usually with views, so nothing in the code knows your real schema. `cqr check-contract` compares a live
source with this page and lists every missing table or required column.

Conventions:

- One data source is one product line (`auto`, `consumer`, ...). Every source has the same tables.
  The package tags rows with the line name, and keys are always `(product_line, id)`, so the same
  `debtor_id` can exist in two lines without colliding.
- Ids are opaque. They can be integers or text. There is no column for a name, a document number, a
  phone number or an e-mail address, and the package never asks for one. Phones are referenced by
  `phone_id`; the numbers stay in your system and you join them back to the queue yourself.
- Dates are timestamps without a time zone. A NULL date means "not set".
- Flags are `0`/`1` (or bit). A NULL flag takes the default given below.
- Everything is read with plain `SELECT` statements. The package refuses anything else (see
  `sqlguard.py`) and binds every date as a parameter.

If your tables have other names, set `source.tables` in the settings file, for example
`{"debts": "dbo.v_collection_debts"}`. Identifiers must be plain (letters, digits, `_`, up to three dotted
parts); anything stranger is refused instead of escaped.

Column sections below are checked against `contract.py` by `tests/test_docs.py`, so this page cannot drift.

## `debtors` (optional table)

| Column | Need | Meaning |
|---|---|---|
| `debtor_id` | required | Opaque id of the person or company that owes. |
| `is_company` | optional | `1` for a company, `0` (default) for a person. |

## `debts`

| Column | Need | Meaning |
|---|---|---|
| `debt_id` | required | Debt id, unique within a product line. |
| `debtor_id` | required | Owner of the debt. |
| `entered_at` | required | When the debt entered the portfolio. Snapshots only see debts entered before their cutoff. |
| `due_at` | required | Original due date. Days overdue are computed from it at each cutoff. |
| `balance` | required | Amount owed. See "Balances and point-in-time" below. |
| `portfolio_id` | optional | Portfolio the debt belongs to. |
| `contract_no` | optional | Contract number, informational only. |
| `days_overdue` | optional | Ignored: it is a value as of extraction. Overdue days are recomputed from `due_at`. |
| `product_id` | optional | Product (categorical feature). |
| `phase_id` | optional | Collection phase (categorical feature). |
| `closed_at` | optional | When the debt was closed or written off. A debt counts as open at a date when it entered before it and was not closed before it. |
| `original_balance` | optional | Balance at entry. Select it with `features.balance_column` when `balance` moves after entry. |

## `debtor_phones`

| Column | Need | Meaning |
|---|---|---|
| `phone_id` | required | Phone id. Used in contact events and in the queue output. |
| `debtor_id` | required | Owner. |
| `is_valid` | optional | `1` valid (default when NULL), `0` known bad. |
| `is_mobile` | optional | `1` mobile line. |
| `has_messaging` | optional | `1` can receive messages. |
| `is_priority` | optional | `1` flagged as the preferred line. |
| `created_at` | optional | When the phone was registered. When filled, phone counts are point-in-time. |
| `updated_at` | optional | Last update. A recent update raises the prior of a never-dialled phone. |

## `debtor_emails` (optional table)

| Column | Need | Meaning |
|---|---|---|
| `email_id` | required | E-mail record id (the address itself is never needed). |
| `debtor_id` | required | Owner. |
| `created_at` | optional | When registered. |

## `contact_events`

One row per outreach attempt.

| Column | Need | Meaning |
|---|---|---|
| `event_id` | required | Event id. |
| `debtor_id` | required | Debtor the attempt was about. |
| `result_id` | required | Outcome, joins `contact_results`. |
| `occurred_at` | required | When the attempt happened. |
| `agent_id` | optional | Who made it. Joins `agents` for the automated-dialer and excluded-agent rules. |
| `phone_id` | optional | Which phone was dialled. Needed for phone quality. |

## `contact_results`

The catalogue of attempt outcomes.

| Column | Need | Meaning |
|---|---|---|
| `result_id` | required | Outcome id. |
| `answered` | required | `1` when a person answered. |
| `right_party` | required | `1` when it was the person you wanted. Answered and right party is a right-party contact. |
| `label` | optional | Human-readable outcome. Used only to draft categories. |
| `category` | optional | One of `answered`, `promise`, `wrong_party`, `no_answer`, `voicemail`, `busy`, `invalid_number`, `message`, `other`. |

When `category` is empty it is derived from the flags: not answered is `no_answer`; answered and right
party is `answered`; answered but not the right party is `wrong_party`. The flags cannot express
`message` (an automated message is not an attempt to reach a human), `invalid_number` or `promise`, so
give a `category` when you can: `cqr draft-contact-map` proposes one from the labels for you to review.

## `agreements`

One row per installment of an agreement.

| Column | Need | Meaning |
|---|---|---|
| `agreement_id` | required | Agreement id. Not unique alone: pair it with `portfolio_id`. |
| `portfolio_id` | required | Part of the agreement key. |
| `debtor_id` | required | Who signed. |
| `installment_no` | required | `0` is the first installment (the down payment). Only installment 0 defines an agreement. |
| `amount` | required | Installment amount. |
| `paid_amount` | required | Amount received, `0` or NULL when unpaid. |
| `issued_at` | required | When the agreement was issued. |
| `paid_at` | required | When the installment was paid, NULL when unpaid. |
| `status_id` | required | Joins `agreement_statuses`. |
| `due_at` | optional | Installment due date. |
| `agent_id` | optional | Agent who made the agreement (for the excluded-agent rule). |
| `row_id` | optional | Tie-breaker for rewrites. Defaults to the input order. |

## `agreement_debts` (optional table)

| Column | Need | Meaning |
|---|---|---|
| `agreement_id` | required | Agreement. |
| `portfolio_id` | required | Part of the agreement key. |
| `debt_id` | required | A debt the agreement settles. |

Used only for the legacy rule. Without it no agreement is treated as legacy.

## `agreement_statuses`

| Column | Need | Meaning |
|---|---|---|
| `status_id` | required | Status id. |
| `label` | required | Label, matched against `statuses.paid`, `statuses.broken` and `statuses.active`. |

The defaults are `PAID`, `PAID_MANUAL` (paid), `BROKEN`, `AUTO_BROKEN` (broken) and `ACTIVE`. Any other
label, such as `PENDING_EXCEPTION` or `REJECTED`, is not an agreement: it neither counts as one nor
removes a debtor from the queue. Map your own labels in the settings; nothing refers to numeric ids.

## `agents` (optional table)

| Column | Need | Meaning |
|---|---|---|
| `agent_id` | required | Agent id. |
| `login` | required | Login, matched case-insensitively against LIKE patterns. |
| `display_name` | optional | Not used. |

`agents.automated_patterns` (default `autodialer%`, `system%`) marks an automated dialer: its attempts do
not count toward the cooldown and weigh less in the phone score. `agents.excluded_patterns` (default
`test%`) drops events and agreements made by matching logins, in SQL.

## Rules the package applies to this data

A valid agreement is installment 0 of an agreement with a configured paid, broken or active status.
Rewrites are resolved per (agreement, portfolio, installment): the latest row wins, after the status
filter. An agreement issued more than `labels.legacy_tolerance_days` before every debt it settles is
history that arrived with the portfolio, and is dropped. The population at a cutoff is the debtors with a
positive balance over their open debts and no valid agreement issued before the cutoff. The excluded-agent
rule and read-only access are enforced in SQL.

## Balances and point-in-time

A snapshot at a past cutoff must not know the future. `debts.balance` is read as stored. If your system
sets the balance to zero when a debtor pays, using it for past cutoffs removes the debtors who later paid
(only the ones who never paid survive) and inflates every metric. Either keep a balance that does not move
after entry and select it with `features.balance_column`, or supply `closed_at` so that debts are open or
closed as of each cutoff. Likewise `is_valid` and `updated_at` on phones are values as of extraction.

## Example: mapping a different schema with views

```sql
-- SQL Server flavour; the names on the left are the contract, the right side is your schema.
CREATE VIEW dbo.debts AS
SELECT d.id AS debt_id, d.owner_id AS debtor_id, d.loaded_on AS entered_at,
       d.due_date AS due_at, d.open_amount AS balance, d.product AS product_id,
       d.phase AS phase_id, d.closed_on AS closed_at
FROM your_schema.receivables AS d;
```

This repository has not been run against a live SQL Server. The adapter builds its connection string and
statements from the contract, and tests check that text and the parameter passing with a fake connection;
SQLite is exercised for real.
