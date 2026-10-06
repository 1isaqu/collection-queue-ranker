# Architecture

## Data flow

```
 your database (SQL Server, SQLite, a view layer over either)
        |  contract tables, read-only SELECTs, dates bound as parameters
        v
 db/ (adapters, queries, loader) --> RawTables  (tables.py, contacts.py)
        |
        v  labels.py: valid agreements, latest rewrite, legacy dropped, outcomes
   Agreements (first installments + all installments of valid agreements)
        |
        +--> snapshots.py: for every cutoff T
        |        features_at(T):  open debts, contact response, phone quality, contact factor
        |        window_labels:   closed in [T, T+H), realised value in the same window
        |    => snapshots.csv     one row per (debtor, T)
        |
        +--> break cohort: one row per debtor with a resolved, mature agreement,
                 base features at the first agreement date  => break.csv
        v
 training.py
   split.py            whole-date holdout boundary shared by both heads, purge, walk-forward folds
   close head          LightGBM classifier, parameters picked by walk-forward value capture
   break head          LightGBM classifier on base features only
   ranker              LightGBM LambdaRank, one query per snapshot, graded by realised value
   => bundle.joblib, metrics.json, report.md, backtest.csv, ...   (+ one MLflow run)
        v
 evaluation/            AUC with a cluster bootstrap, precision/recall/lift, gain table,
                        monetised backtest per queue, robustness per queue
        v
 worklist/build.py      today's population -> scores -> cooldown -> tiers -> gate -> operational cut
   => queue_full.csv, queue_operational.csv, queue_summary.json
        v
 monitoring.py          adoption funnel and ordering test of a delivered queue
 diagnostics.py         can the operation reach the people ranked first?
 experiments.py         label experiments with a common judge
```

`pipeline.retrain` runs load, datasets, train and queue in that order and stops at the first failing
stage with a `PipelineError`; the queue is only written from the bundle that the same run just trained.

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Frozen settings dataclasses, JSON overrides, unknown keys rejected |
| `contract.py` | Table and column specification, `check_source` |
| `sqlguard.py` | Token-based read-only guard |
| `db/` | `DataSource` protocol, SQLite and SQL Server adapters, query builders, loader |
| `tables.py` | `RawTables`: normalised frames (types, flags, defaults, annotated events) |
| `contacts.py` | Contact categories, draft classifier, automated-agent matching |
| `labels.py` | Valid agreements, outcomes, window labels, break cohort |
| `features/` | `base.py` debt features, `contact.py` response features, `phones.py` quality and factor |
| `snapshots.py` | Cutoff dates, `features_at`, snapshot and break datasets |
| `split.py` | Time split with purge, walk-forward folds |
| `encoding.py`, `models.py` | Categorical encoding, LightGBM heads, ranker, bundle |
| `scoring.py` | Expected value, strategies |
| `training.py` | Selection, fitting, holdout report, persistence |
| `evaluation/` | Metrics, backtest, report |
| `worklist/` | Queue building and collector assignment |
| `monitoring.py`, `diagnostics.py`, `experiments.py` | Post-delivery and analysis tools |
| `enrichment.py` | Optional provider interface with a TTL cache |
| `tracking.py` | MLflow wrapper and a no-op tracker |
| `serving/app.py` | Optional FastAPI scoring service |
| `pipeline.py`, `cli.py` | Dataset files, gated retrain, command line |
| `synth.py` | Synthetic data in the contract shape |

## One code path for training and scoring

`snapshots.features_at(raw, prep, cutoff, settings)` builds every feature, and it is the only function that
does. Training calls it for each past cutoff; the queue calls it for "now". A feature cannot mean one
thing in training and another in production, which is the most common way a ranker that backtests well
disappoints once deployed. The break head uses `base_features` with the first agreement date as the
anchor, the same function again.

## Where leakage is closed

| Risk | Where it is handled |
|---|---|
| A feature reading rows after the cutoff | `base_features` and `contact_features` filter on the anchor; tests plant rows at and after the cutoff |
| Attempt counts that describe the operation, not the debtor | not computed as features (`features/contact.py`); a test asserts they are absent |
| A label window leaking into the holdout | `split.time_split` purges development rows whose window reaches the boundary |
| Rows of one date straddling the boundary | the boundary is a date, never a row count |
| Outcomes still pending | active agreements are excluded from the break head; the last cutoff is `as_of - horizon` |
| Balances that move after the outcome | documented in the data contract, `closed_at` and `features.balance_column` |
| Score fitted on holdout | selection uses development folds only; the holdout is read once for the report |

## Storage

Nothing is persisted except under `--artifacts` (default `artifacts/`) and `mlruns/`; both are ignored by
git, as are `data/` and every csv, parquet, pickle and joblib file. The bundle is a joblib pickle: load only
bundles you made yourself.
