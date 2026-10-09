# Key decisions

Each entry says what the code does and why. Where a number appears it is a default you are meant to
change; none of them was calibrated on real data.

## Data and labels

### One row per debtor and cutoff, not one row per debtor

A static "ever paid" table cannot say what was known when. `snapshots.py` builds one row per
(debtor, cutoff T): features from rows dated before T, label from the window `[T, T + horizon)`. The
last cutoff is `as_of - horizon`, so every label window is fully observed. A debtor lives in many
snapshots, so rows are not independent, and the confidence intervals in the report resample debtors
instead of rows.

### Training and scoring share one function

`snapshots.features_at(raw, prep, cutoff, settings)` is the only place features are computed. Training
calls it for each past cutoff and the work queue calls it for today. The break head uses the same
`base_features` with the first agreement date as the anchor. A feature cannot be defined one way in
the backtest and another in production.

### Response features are in, effort features are out

The model sees how a debtor responded: answered attempts, wrong-party attempts, dead numbers, answer
rate, days since the last answer. It never sees how hard the operation tried (attempt counts, attempts
in the last N days). Those score a higher AUC, but only because the operation already works the
debtors it expects to pay, so a model trained on them learns to rank whoever is already ranked. The
synthetic generator builds that loop on purpose, and `test_attempt_intensity_is_not_a_feature` fails
if an effort column enters the feature list.

### Pending outcomes are not failures

An agreement that is still active may yet be paid or broken. The break head trains only on debtors
with a resolved outcome whose first agreement is older than `labels.break_observation_days`. Treating
active agreements as failures would put a pending outcome into the same class as a real one.
`cqr experiment unresolved` compares the three treatments (failure, excluded, success), judged by AUC on
holdout rows whose outcome is known.

### What counts as an agreement

Only installment 0 defines an agreement, and only with a configured paid, broken or active status. A
pending exception or a rejection is not an agreement and does not take a debtor out of the queue.
Installment rows can be rewritten; per (agreement, portfolio, installment) the latest row wins, and the
status filter is applied before that choice so a later rewrite with an out-of-scope status cannot hide
a valid row. An agreement issued more than `labels.legacy_tolerance_days` before every debt it settles
is history that arrived with the portfolio and is dropped; an agreement with no debt link cannot be
judged and is kept.

### Balances must not move after the fact

If your system zeroes a balance when the debtor pays, a past snapshot built from it contains only the
debtors who never paid, and every metric looks better than it is. The contract asks for `closed_at`
(a debt is open at a date when it entered before it and was not closed before it) or a balance column
that does not move (`features.balance_column`). The same caveat applies to phone validity and update
dates when no creation date is supplied.

## Models

### Two heads, no class weights

`close` estimates whether a debtor signs an agreement within the horizon. `break` estimates whether an
agreement fails. Collapsing them into one "will it be paid" probability and multiplying by one minus the
break probability would count non-breakage twice. Neither head is class-weighted: the probabilities are
multiplied by a balance, so they have to be calibrated, and reweighting a rare class inflates every
probability. The break head sees only base features, because it is meant to describe debt and debtor
before any call.

### Order by expected value, not by probability

A probability ranks a small, likely payer above a large, slightly less likely one. Under limited
capacity the useful order is by value:

    EV = balance x P(close) x (1 - P(break))
    adjusted EV = EV x contact factor

The contact factor turns the best few phones of a debtor into a multiplier between `queue.contact_floor`
and 1. It reorders by reachability but never zeroes a debtor, because the model itself is uncertain.
The default production strategy is the adjusted EV (`ev_contact`).

### LambdaRank as a second strategy

`lambdarank` trains a LightGBM ranker with one query per snapshot. Relevance is graded by realised
value: 0 when nothing was received, otherwise `1 .. n_grades - 1` by tier of the amount received. The
tier edges are quantiles of the positive values in the training frame and are reused unchanged when the
holdout is graded. Gains are `2^grade - 1`. It is trained and judged next to the expected-value
strategies, not instead of them, so the report shows whether optimising the order directly pays off.

### Select by money, not by AUC

Two parameter sets can tie on AUC and differ in money. Candidates in `model.candidates` are compared by
walk-forward cross-validation on development rows, and the default metric is `value_capture`: the share
of realised value found in the top `model.selection_capacity` of each queue when ranking by
`balance x P(close)`. `model.selection_metric = "auc"` switches back to AUC. The selection table is
written to `selection.csv`.

## Evaluation

### The split is on dates

`split.time_split` puts the boundary between two distinct dates, so rows that share a date never
straddle it. It then purges development rows whose label window reaches the first holdout date. Both
heads use the same boundary, which means the break head never sees an outcome from the holdout period.
Walk-forward folds apply the same purge between each training block and the block after it. The holdout
is scored once, for the report. The saved bundle is the one trained on development rows; refitting on
everything would leave nothing to judge it by.

### The backtest asks the question a manager asks

Each snapshot is a queue and capacity is applied inside it (`max(1, floor(n x capacity))` rows), then
summed, because a team works one queue per period. The judge is the amount actually received inside the
label window, which does not depend on how any model defines its target. Every strategy is compared with
"largest balance first" and with random order. The uplift over the balance rule gets a debtor-level
cluster bootstrap interval.

### The verdict reads the interval

A point estimate above `evaluation.uplift_proof_pct` is called evidence only when the lower end of its
interval is above zero. Otherwise the report says the uplift clears the threshold but could be zero, or,
when no interval was computed, that it cannot be called evidence. The thresholds are policy, not
statistics, and they are configuration for that reason.

### AUC is reported with an interval and a calibration line

An AUC near 0.5 may be sampling noise, so each head's AUC comes with a bootstrap interval and a one-line
reading of it against chance. Mean predicted rate and observed rate are printed together with it,
because a model that predicts the base rate for everyone is perfectly calibrated and useless for ordering.
`evaluation.min_auc_alert` marks queues where the AUC falls below a floor, which is how drift shows up.

## The work queue

### Cooldown counts human attempts only

An automated dialer sweeps the whole base to hold volume. Counting its attempts would remove people
nobody has spoken to. Agent logins matching `agents.automated_patterns` are treated as automated.

### Phones are scored, and the queue is gated on them

A phone's score comes from its own history: voice attempts only, automated attempts down-weighted,
the answer rate shrunk toward the global rate, a recency tier, a hard zero for a number reported dead
repeatedly and never answered, and a prior for numbers never dialled. A debtor with no usable phone is
not workable by voice, so it goes to the end under the tier `NO_CONTACT`.

### The operational file is dealt like cards

Rows go to collectors round-robin. Blocks would give the first collector all the value and the last one
the tail. Dealing gives each collector an ordered list that represents the whole ranking, so when
everyone works from the top the worked set approximates the global top. The file carries no scores on
purpose. A collector who sees a probability may repeat it to the debtor as a promise, and the row order
already carries the priority.

## Engineering

### Read-only, in code

Every statement passes `sqlguard.assert_read_only`, which strips comments, string literals and quoted
identifiers and then checks whole-word tokens. It must start with `SELECT` or `WITH`, hold a single
statement and contain no write or administrative keyword. It is a seatbelt: connect with a login that
can only read. Dates are always bound parameters. A date spliced into text is parsed under the server's
date format and can silently swap day and month.

### A contract, not a schema

The package reads ten tables with fixed names. You map your database to them, usually with views, and
`cqr check-contract` lists what is missing. There is no column for a name, document number, phone number
or e-mail address anywhere. Phones are opaque ids and stay joinable in your own system.

### Enrichment is an interface

An external score is a provider with a `fetch(keys)` method plus a TTL cache. No vendor client ships.
A snapshot row describes the world at its cutoff, so an enrichment value must be the one known then;
today's score on a past snapshot leaks the future. When an `external_score` column exists the backtest
adds a strategy that works the highest score first, which shows whether the model adds anything over
simply buying the score.

### Retraining stops at the first failure

`pipeline.retrain` runs load, datasets, train and queue in order. A stage that raises ends the run with
a `PipelineError` naming the stage, and the queue is written only from the bundle that the same run just
trained. A retrain that quietly reuses old data or old weights and exits cleanly is worse than one that
fails.

### Tracking is local

MLflow writes to a directory (`mlruns/`, ignored by git). MLflow 3 treats the plain file store as
legacy, so the tracker sets `MLFLOW_ALLOW_FILE_STORE` for a directory URI unless you have set it. A
`sqlite:///...` URI works without code changes, and `NullTracker` turns tracking off.

### Defaults are invented

Horizon, cooldown, tier shares, phone weights, capacities, hyperparameters and verdict thresholds are
plain defaults chosen to make the demo run. Calibrate them on your own data. Unknown keys in a settings
file are rejected, so a typo cannot silently do nothing.
