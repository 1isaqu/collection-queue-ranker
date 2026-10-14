# Reading the evaluation

`cqr train` and `cqr backtest` score the holdout once and write the files below. `cqr train` also prints
the report. Every number in it comes from rows the models never saw: the boundary between development
and holdout is a date (see `docs/decisions.md`).

| File | Content |
|---|---|
| `report.md` | The report in text |
| `metrics.json` | The same numbers, machine readable (NaN and infinity become `null`) |
| `backtest.csv` | One row per strategy and capacity |
| `gain_table.csv` | Close-head capture by population decile |
| `robustness.csv` | Close-head AUC per queue |
| `selection.csv` | Cross-validation score of every parameter candidate |
| `importance_<head>.csv` | Share of split gain per feature, for `close`, `break` and `ranker` |

`cqr backtest` rebuilds the first five files in `<artifacts>/backtest/` from a saved bundle and the
dataset files, without refitting anything.

## Close head: does the debtor sign an agreement within the horizon

| Line | Meaning |
|---|---|
| rows, positives, base rate | Size of the holdout and share of snapshots with a started agreement |
| AUC and interval | ROC AUC with a percentile bootstrap interval. The bootstrap resamples debtors, not rows, because a debtor appears in many snapshots |
| verdict | The interval read against 0.5: covers chance, above chance, or below chance |
| calibration | Mean predicted probability next to the observed rate |
| precision, recall, lift at capacity | Computed at each fraction in `evaluation.precision_ks` |

Precision at capacity c is the share of positives among the top `max(1, floor(n x c))` rows by
probability. Recall at the same point is the share of all positives found there (the gain). Lift is
precision divided by the base rate. `gain_table.csv` repeats the idea over ten equal bins.

The calibration line matters because the probabilities are multiplied by a balance. A model that says
the base rate for everybody is perfectly calibrated and cannot order anything, so read calibration
together with the AUC and its interval.

## Break head: does an agreement fail

Same layout, on the break-cohort rows at or after the holdout boundary. Only agreements with a known
outcome that are older than `labels.break_observation_days` are in it. This holdout is usually much
smaller than the snapshot holdout; its interval will be wide, and the verdict line is there to say so.

## Ranking under limited capacity

- Each holdout snapshot is a queue. In every queue the top `max(1, floor(n x capacity))` rows by a
  strategy's score are "worked", and the amount actually received inside the label window is summed
  across queues. The judge does not depend on how any model defines its target.
- `captured share` is that sum over the total received in the holdout.
- `vs balance rule %` is the percentage gain over "largest balance first". `vs random %` is the gain over
  random order. The interval next to the first one is a debtor-level cluster bootstrap with
  `evaluation.n_bootstrap` draws at `evaluation.ci_level`.
- The headline capacity is the entry of `ranking.capacities` closest to 20%. The headline strategy is
  `ranking.strategy`.
- Mean NDCG at the headline capacity uses gain `2^grade - 1` and a log2 discount, per queue, averaged. Grades
  follow the ranker's `model.relevance` setting, or the closed flag when the bundle has no ranker.

Strategies in the table:

| Strategy | Order |
|---|---|
| `ev_contact` | balance x P(close) x (1 - P(break)) x contact factor |
| `ev` | balance x P(close) x (1 - P(break)) |
| `prob` | P(close) x (1 - P(break)), ignoring the balance |
| `lambdarank` | LightGBM LambdaRank score (when the bundle has a ranker) |
| `largest_balance` | baseline: biggest balance first |
| `random` | baseline: seeded random order |
| `external_score` | only when the data has that column: highest external score first |

### Verdict

The verdict is a rule on the headline uplift, with thresholds in `evaluation`:

| Condition | Verdict |
|---|---|
| uplift at or above `uplift_proof_pct` and lower interval end above zero | evidence of value, worth a controlled pilot |
| uplift at or above `uplift_proof_pct` and lower interval end at or below zero | grey zone: clears the threshold, but the interval includes zero |
| uplift at or above `uplift_proof_pct` and no interval (`evaluation.n_bootstrap` is 0, or no draw was usable) | grey zone: clears the threshold, but there is no interval to read |
| uplift below `uplift_floor_pct` | little or no gain over the balance rule |
| anything between | grey zone, a business decision, ideally decided by a pilot |
| uplift not computable | not estimable |

The thresholds are policy. The defaults are invented; set them to the gain that would justify changing
how your team works.

## Robustness

`robustness.csv` holds the close-head AUC of every holdout queue and flags queues below
`evaluation.min_auc_alert`. A single weak queue can be noise on a small queue. A run of them is drift.

## What the numbers cannot tell you

- The backtest replays history under a fixed capacity. It assumes the operation would have reached the
  people it was told to work and that working a debtor earlier or later does not change the outcome.
  A controlled pilot is the only test of that.
- Realised value is limited to the label window and to agreements that started inside it. Money from an
  older agreement is not credited to the model.
- With few queues the interval is wide. With many cutoffs close together the queues overlap in time and
  are not independent either; the debtor bootstrap handles repeated debtors, not shared periods.
- Numbers from the synthetic demo are an artefact of the generator. They show that the pipeline runs,
  not how well it ranks anything real. The one run on public data, with its own limits, is in
  [real-data-evaluation.md](real-data-evaluation.md).

## Experiment tracking

With tracking on, `train` opens one MLflow run named `train`. It logs the whole settings tree and the
selected close-head parameters as parameters, the headline metrics, and the files in the table at the
top of this page as artifacts under `reports/`. The bundle, the dataset files and the queue files are
not logged.

| Metric | Meaning |
|---|---|
| `close_auc`, `close_auc_ci_low`, `close_auc_ci_high` | Close-head AUC and interval |
| `close_precision_at_<fraction>` | Precision at each configured capacity |
| `uplift_vs_balance_pct` | Headline uplift over the balance rule |
| `break_auc` | Break-head AUC, when the break holdout is not empty |

To browse the runs, point the MLflow UI at the same store. With MLflow 3 and a plain directory store,
set `MLFLOW_ALLOW_FILE_STORE=true` in the shell first, as the tracker does for you.

## Label experiments

`cqr experiment label-target` trains the close head on three targets (an agreement started, anything
was paid, a material amount was paid) and judges all three by the realised value captured in the top of
each queue, which is the one measure that does not change with the target. `cqr experiment unresolved`
trains the break head three ways (active agreements as failures, excluded, or as successes) and judges
them by AUC on holdout rows whose outcome is resolved.
