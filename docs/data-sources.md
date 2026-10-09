# Data sources

This repository contains no data and no fitted model. It ships code that reads a database you
provide. This page lists every place a number or a row comes from, grouped into tiers by how far
you can trust and reproduce it. Last reviewed on 2026-10-08.

| Tier | What it is | In this repository |
|---|---|---|
| 0 | Invented data from a seeded generator, and invented rows written inside the tests | Yes: `src/collection_queue_ranker/synth.py` and `tests/` |
| 1 | A public dataset downloaded at run time | None used |
| 2 | Your own database mapped to the data contract | Never committed |

## Tier 0: invented data

`cqr synth` runs the generator in `src/collection_queue_ranker/synth.py`. It writes one SQLite
database per product line in the shape described in [data-contract.md](data-contract.md): debtors,
phones, debts, contact events, agreements with installments and rewrites, and legacy agreements.
It is seeded (`--seed`, default 7), so the same arguments give the same rows. The README quick
start and the end-to-end tests run on it.

Every value is invented. The constants in the generator are arbitrary: the balance distribution,
the range of phone quality, how many phones and emails a debtor has, the plan lengths and their
probabilities, the coefficients that tie a hidden willingness to calls, agreements and payments,
the agent count, the contact-result catalogue and the status labels. None comes from a public
source. None was fitted to, tuned against or compared with results from a real system, and the
output is not meant to resemble any real operation.

The generator plants two things on purpose, so that the pipeline has something to find and
something to avoid:

* a signal: willing debtors answer, sign and pay more often, and
* a feedback loop: the simulated operation dials willing debtors more often, which makes attempt
  counts look predictive. The package keeps attempt counts out of the features for this reason.

Metrics computed on this data describe the generator. They show that the pipeline runs and that
the leakage rules hold. They say nothing about how well a ranker would do on real portfolios.

The unit tests use a second kind of tier 0 data: small frames written inline in the test modules
(`tests/world.py` and the test files themselves), with made-up ids, amounts, dates and logins such
as `agent_01`. There are no fixture files. `git ls-files tests` lists only Python modules.
CI runs on tier 0 only. No test downloads anything or opens a network connection.

## Tier 1: public datasets

None is used. If you add one, document its source, its license, the date you checked the terms,
and what it cannot test (a credit table without contact history, for example, cannot exercise the response
features or the queue rules).

## Tier 2: your own data

Point the package at a database you own, as described in the
[README](../README.md#use-it-on-your-own-data). The rules:

* Provide the tables or views listed in [data-contract.md](data-contract.md). Map your own names
  with `source.tables` in a settings file.
* Use a login that can only read. Put the host, user and password in a `.env` file or in the
  process environment. `.env` is ignored by git and `.env.example` holds placeholders only.
* Run `cqr check-contract` first. It lists missing tables and required columns. It does not check
  that the values are sensible, so compare one debtor and one agreement by hand against your
  source system before anyone relies on a queue.
* The contract has no name, document number, phone number, e-mail address or street address.
  Phones and e-mails appear as opaque ids and flags. Keep it that way in your views.
* `data/`, `artifacts/`, `mlruns/` and the usual data file extensions are ignored by git. The
  files the pipeline writes (snapshots, the saved bundle, the queue files) derive from your data,
  so treat them as your data and do not commit them.

## Where the constants come from

Every default in `src/collection_queue_ranker/config.py` is arbitrary and was picked so the demo
runs: the label horizon, the cooldown, the tier shares, the phone-score weights, the diagnostics
and monitoring cut-offs, the verdict thresholds and the LightGBM parameters. None comes from a
public source and none was calibrated on data. The right values depend on how your operation
records its work. Calibrate them on your own history, and do not read a number the package
returns with the defaults as a finding.
