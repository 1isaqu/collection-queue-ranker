# collection-queue-ranker

A LightGBM ranker that orders a collections work queue. It builds point-in-time snapshots of the
debtor base, trains on a leakage-safe time split, tracks runs with MLflow, and judges the result by the
money that lands in the top of each queue, not only by AUC. The package reads a small relational
contract, so you point it at your own database and it produces a ranked list for your collectors.

This is an independent portfolio project. It is not affiliated with any company and ships no real data:
the demo runs on a generator that invents every value.

## What it does

A collections team can only work part of its queue each period. The question is which part. The usual
answer is "largest balance first", and a model has to beat that to be worth anything. This repository
answers the question end to end:

1. It turns your tables into snapshots: one row per debtor per cutoff date, with features computed only
   from rows dated before the cutoff and a label read from the window after it.
2. It fits two LightGBM classifiers (will the debtor sign an agreement, will the agreement break) and a
   LambdaRank model graded by the money actually received.
3. It scores the holdout once and reports AUC with bootstrap intervals, precision and lift at capacity,
   and a backtest that gives the team a fixed capacity per queue and sums what each ordering captures.
4. It builds today's queue: expected value, a cooldown for debtors a human just worked, a gate on
   phone reachability, P1/P2/P3 tiers, and a list per collector.

Three choices carry most of the weight:

- The training and the live queue share one feature function, so a feature cannot mean one thing in the
  backtest and another in production.
- Features describe how a debtor responded to contact, never how hard the operation tried. Attempt
  counts look predictive because the operation already works the people it expects to pay, and a model
  that learns them ranks whoever is already ranked.
- Model selection, the backtest and the verdict are all about value under limited capacity. A parameter
  set is chosen by the share of realised value it finds in the top of each queue. The headline uplift
  over the balance rule only counts as evidence when its debtor-level bootstrap interval stays above zero.

## Architecture

```
 your database (SQL Server, SQLite, or views over either)
        |   contract tables, read-only SELECTs, dates bound as parameters
        v
 db/ adapters and loader  -->  RawTables
        |
        v   labels.py: valid agreements, latest rewrite wins, legacy dropped
 snapshots.py   features_at(T)  +  window labels      -->  snapshots.csv   (one row per debtor and T)
                break cohort at the first agreement   -->  break.csv
        |
        v
 training.py   date split with purge  ->  parameter selection by walk-forward value capture
               close head, break head, LambdaRank ranker  ->  bundle.joblib  (+ one MLflow run)
        |
        v
 evaluation/   AUC with cluster bootstrap, precision and lift, gain table,
               per-queue backtest against two baselines, robustness per queue  ->  report.md
        |
        v
 worklist/     score today's population -> cooldown -> tiers -> reachability gate -> collector lists
        |
        v
 monitoring.py, diagnostics.py, experiments.py   (what happened to a delivered queue, and why)
```

`docs/architecture.md` has the module map and the places where leakage is closed.

## Quick start

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cqr synth --out data/demo
cqr retrain --backend sqlite \
    --source auto=data/demo/auto.db --source consumer=data/demo/consumer.db
```

`cqr synth` writes two SQLite databases in the data contract shape: 4000 invented debtors in `auto` and
two thirds of that in `consumer` (`--debtors` changes the first number). `cqr retrain` loads them, builds
the datasets, trains, writes the holdout report and builds the queue, and stops at the first stage that
fails. It prints a short summary (reference date, close-head AUC, headline uplift with its verdict,
queue counts) and takes several minutes. Everything lands in `artifacts/` (the MLflow store goes to
`mlruns/`; both are ignored by git):

```
artifacts/snapshots.csv, break.csv, dataset_meta.json     the datasets
artifacts/bundle.joblib                                   the fitted models
artifacts/report.md, metrics.json, backtest.csv, ...      the holdout evaluation
artifacts/queue/queue_full.csv                            every debtor, ranked, with scores
artifacts/queue/queue_operational.csv                     the lists for collectors, no scores
```

Read `artifacts/report.md` first, then `docs/evaluation.md` for what each line means. The demo numbers
come from a generator with a planted signal. They show the pipeline runs, nothing more.

The same flow in steps, each reading what the previous one wrote:

```
cqr build-dataset --backend sqlite --source auto=data/demo/auto.db
cqr train
cqr backtest
cqr queue --backend sqlite --source auto=data/demo/auto.db
```

Other commands: `check-contract`, `draft-contact-map`, `monitor`, `diagnose`, `experiment`, `serve`.
`cqr --help` lists them and `cqr <command> --help` shows the options.

## Use it on your own data

1. Map your database to the tables in `docs/data-contract.md`. Views are the usual way; nothing in the
   code knows your real schema. If your tables have other names, set `source.tables` in a settings file.
2. Install the SQL Server driver extra if you need it: `pip install -e ".[sqlserver]"`. Copy
   `.env.example` to `.env` and fill the connection variables. Secrets are read from the environment
   only. Use a login that can only read.
3. Run `cqr check-contract --backend sqlserver --source auto=YOUR_DATABASE`. It lists every missing table
   or required column.
4. Optional: run `cqr draft-contact-map --out artifacts/contact_map_draft.csv`. It proposes a
   `category` for each contact result from its label (answered, wrong party, voicemail, and so on).
   Review it and load it into `contact_results.category`. Without it the category is derived from the
   `answered` and `right_party` flags, which cannot express automated messages or dead numbers.
5. Copy `examples/settings.example.json`, set your status labels, horizon and agent patterns, and pass it
   with `--config`. Unknown keys are rejected.
6. Run `cqr retrain --config my.json --backend sqlserver --source auto=YOUR_DATABASE --as-of YYYY-MM-DD`.
   Without `--as-of` the reference date is the day after the latest timestamp in the data, which suits an
   export; in production pass today's date as `--as-of` so `snapshots.history_months` bounds what is read.

One product line is one source. `--source` is repeatable, and every key in the package is
`(product_line, id)`, so two lines can reuse the same ids.

## Scoring service

An optional FastAPI app serves a saved bundle:

```
pip install -e ".[serve]"
cqr serve --artifacts artifacts --port 8000
```

`GET /health` reports liveness, `GET /model` returns the bundle's version, dates, feature lists and
holdout metrics, and `POST /score` takes `{"rows": [...]}` (up to 5000 rows that already hold the model
features, such as the columns of `queue_full.csv`) and returns `p_close`, `p_break`, the expected value,
and a rank. A missing feature column is reported by name with a 422. It does not look anything up in a
database.

## Configuration

Settings are frozen dataclasses in `src/collection_queue_ranker/config.py`. Every value below is an
invented default that makes the demo run. None was calibrated on real data. Calibrate them on yours.
Set any of them in a JSON settings file (`--config` or `CQR_CONFIG`); a file may override a single
LightGBM parameter inside `model.close_params`.

| Setting | Default | Meaning |
|---|---|---|
| `statuses.paid` | `["PAID", "PAID_MANUAL"]` | `agreement_statuses.label` values that mean an agreement was paid |
| `statuses.broken` | `["BROKEN", "AUTO_BROKEN"]` | Labels that mean it failed |
| `statuses.active` | `["ACTIVE"]` | Labels that mean it is still running (not yet an outcome) |
| `labels.horizon_days` | `60` | Length of the label window after a cutoff |
| `labels.legacy_tolerance_days` | `45` | An agreement issued earlier than this before its debts entered is legacy |
| `labels.break_observation_days` | `50` | An agreement must be this old before its outcome trains the break head |
| `snapshots.frequency` | `"MS"` | pandas offset alias for cutoff spacing (month start) |
| `snapshots.first_cutoff` | `null` | First cutoff date; by default earliest data plus `warmup_days` |
| `snapshots.warmup_days` | `45` | Days of history before the first cutoff |
| `snapshots.history_months` | `30` | How far back debts are read when `--as-of` is given |
| `snapshots.min_balance` | `0.0` | Debtors at or below this balance are not ranked |
| `features.balance_column` | `"balance"` | Which debt balance column feeds the features |
| `features.extra_numeric` | `[]` | Extra numeric columns your enricher adds to snapshots |
| `features.extra_categorical` | `[]` | Extra categorical columns |
| `split.holdout_fraction` | `0.3` | Share of rows kept for the final report, cut on a date boundary |
| `split.cv_folds` | `3` | Walk-forward folds for parameter selection |
| `split.purge` | `true` | Drop development rows whose label window reaches the holdout |
| `model.seed` | `17` | Seed for LightGBM and the random baseline |
| `model.candidates` | `[{"max_depth": 3, "num_leaves": 8}, {"max_depth": 4, "num_leaves": 15}, {"max_depth": 6, "num_leaves": 31}]` | Close-head parameter overrides compared by cross-validation |
| `model.selection_metric` | `"value_capture"` | `value_capture` or `auc` |
| `model.selection_capacity` | `0.2` | Capacity at which value capture is measured |
| `model.relevance` | `"value_grades"` | Ranker target: `value_grades` or `closed` |
| `model.n_grades` | `4` | Number of relevance grades for `value_grades` |
| `model.train_ranker` | `true` | Train the LambdaRank model as well |
| `ranking.capacities` | `[0.05, 0.1, 0.2, 0.3, 0.5]` | Shares of each queue the backtest works |
| `ranking.tier_p1_share` | `0.15` | Share of workable debtors in tier P1 |
| `ranking.tier_p2_share` | `0.25` | Share in tier P2 (the rest are P3) |
| `ranking.strategy` | `"ev_contact"` | Production ordering: `ev_contact`, `ev`, `prob`, `lambdarank`, `largest_balance` or `random` |
| `agents.automated_patterns` | `["autodialer%", "system%"]` | LIKE patterns on `agents.login` for automated dialers |
| `agents.excluded_patterns` | `["test%"]` | Agents whose events and agreements are dropped in SQL |
| `phones.smoothing_strength` | `4.0` | Weight of the global answer rate when scoring a phone |
| `phones.recency_tiers` | `[[120, 1.0], [300, 0.75]]` | Pairs of (up to this many days since the last attempt, score multiplier) |
| `phones.recency_floor` | `0.5` | Multiplier when the last attempt is older than every tier |
| `phones.automated_weight` | `0.6` | Weight of an automated attempt relative to a human one |
| `phones.invalid_kill_count` | `3` | Dead-number reports, with no answer ever, that set a phone score to 0 |
| `phones.prior_mobile` | `0.35` | Prior uplift for a never-dialled mobile phone |
| `phones.prior_messaging` | `0.25` | Prior uplift for a messaging-capable phone |
| `phones.prior_priority` | `0.15` | Prior uplift for a priority phone |
| `phones.prior_fresh` | `0.15` | Prior uplift for a recently updated phone |
| `phones.fresh_days` | `365` | "Recently updated" window |
| `phones.top_phones` | `3` | Phones combined into the contact factor |
| `queue.cooldown_days` | `21` | Days after a human attempt before a debtor is queued again |
| `queue.top_n` | `2000` | Rows in the operational file |
| `queue.collectors` | `8` | Number of collector lists |
| `queue.contact_floor` | `0.4` | Lowest contact factor |
| `queue.exclude_existing_agreements` | `true` | Leave debtors with a valid agreement out of the queue |
| `queue.csv_separator` | `","` | Separator of the queue files |
| `evaluation.n_bootstrap` | `200` | Bootstrap draws for intervals |
| `evaluation.ci_level` | `0.95` | Interval level |
| `evaluation.min_auc_alert` | `0.6` | Per-queue AUC below this is flagged |
| `evaluation.uplift_proof_pct` | `8.0` | Uplift needed for "evidence of value" |
| `evaluation.uplift_floor_pct` | `3.0` | Uplift below this is "little or no gain" |
| `evaluation.precision_ks` | `[0.05, 0.1, 0.2]` | Capacities for precision, recall and lift |
| `evaluation.seed` | `7` | Seed for the bootstrap |
| `tracking.enabled` | `true` | Log to MLflow |
| `tracking.uri` | `"mlruns"` | Directory or MLflow URI of the store |
| `tracking.experiment` | `"collection-queue-ranker"` | MLflow experiment name |
| `source.backend` | `"sqlite"` | `sqlite` or `sqlserver` |

`model.close_params`, `model.break_params` and `model.ranker_params` hold the LightGBM parameters of
each model. `source.sources` maps product lines to targets and `source.tables` maps contract names to
your own tables or views; both default to empty.

Environment variables (see `.env.example`): `CQR_BACKEND`, `CQR_SOURCES` (`auto=target,consumer=target`),
`CQR_CONFIG`, and the `CQR_DB_*` connection variables for SQL Server. For the backend and the sources,
command-line flags beat the environment, which beats the settings file. A `.env` file in the working
directory is read at start and never overrides a variable that is already set.

## Tests

```
pip install -e ".[dev]"
pytest
ruff check .
```

The tests use small frames written inline in the test code and a generator that builds a synthetic
database at test time. They cover the leakage rules (rows planted at and after a cutoff, purge, shared
split boundary), label rules (rewrites, legacy agreements, censoring), the metrics and the cluster
bootstrap, the SQL guard, query construction for both dialects, the adapters, the queue rules, the CLI
end to end, MLflow tracking in a temporary store, and the documentation itself (the data contract page
and the configuration table above are checked against the code). The suite takes several minutes
because the end-to-end tests train real models.

## Project layout

```
src/collection_queue_ranker/
  config.py contract.py tables.py contacts.py labels.py snapshots.py split.py
  encoding.py models.py scoring.py training.py tracking.py pipeline.py cli.py
  features/    base.py (debt features), contact.py (response features), phones.py (quality, factor)
  db/          base.py, sqlite.py, sqlserver.py, queries.py, sources.py, loader.py
  evaluation/  metrics.py, backtest.py, report.py
  worklist/    build.py (queue), assign.py (collector lists)
  serving/     app.py (optional FastAPI scoring service)
  sqlguard.py  monitoring.py  diagnostics.py  experiments.py  enrichment.py  synth.py
tests/         grouped by topic, plus world.py (small hand-built tables)
docs/          architecture, data-contract, decisions, evaluation, glossary
examples/      settings.example.json
```

## Key design decisions

The reasoning for each is in `docs/decisions.md`. The short version:

- Snapshots by cutoff date, not a static table, with the last cutoff placed so every label window is
  observed.
- One `features_at` function for training and scoring.
- Response features in, effort features out.
- Two classifier heads multiplied into an expected value, plus LambdaRank graded by realised value.
- Parameters chosen by value capture in walk-forward folds, not by AUC.
- A split on whole dates with a purge, one boundary shared by both heads, and a holdout scored once.
- A backtest per queue with a debtor-level cluster bootstrap, and a verdict that reads the interval.
- Cooldown on human attempts only, a phone-quality gate, and collector lists dealt round-robin.
- Read-only SQL enforced in code, dates always bound as parameters, no personal identifiers in the
  contract, and external enrichment as an interface with no client.

## Limitations

- The SQL Server adapter has not been run against a live server. Tests cover the connection string, the
  introspection statement, parameter passing and the read-only guard with a fake connection, and they
  check the SQL Server flavour of the queries as text. SQLite is exercised for real, including the
  full pipeline.
- Every default is invented. The horizon, cooldown, tier shares, phone weights, thresholds and
  hyperparameters need calibration on your data before any number means something.
- The synthetic data has a planted signal and a planted feedback loop. Its metrics describe the
  generator, not any real operation.
- The backtest is a replay under a fixed capacity. It cannot say whether your team would reach the
  people it is told to work. A controlled pilot is the test that settles it.
- Monthly cutoffs with a long horizon give overlapping label windows, so queues are not independent. The
  debtor bootstrap handles repeated debtors, not shared periods.
- The saved bundle is trained on development rows only, so the newest dates are not in the model. Refit
  on everything yourself if you accept having nothing left to judge it with.
- Phone-quality scores are computed from the same call history they are later checked against, so the
  diagnostics treat them as a reachability hint, not as ground truth.
- The scoring API takes feature rows. It does not build features, which needs the whole history and is
  the batch pipeline's job.
- `bundle.joblib` is a pickle. Load only bundles you made yourself.
- An external enrichment value is only valid on a past snapshot if it was known at that cutoff.
  Using today's score on old snapshots leaks the future.
- It has been run on Windows with Python 3.14. CI runs it on Linux.

## License

MIT. See `LICENSE`.
