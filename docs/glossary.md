# Glossary

Terms as this repository uses them, then the columns of the files it writes.

## Collections terms

| Term | Meaning here |
|---|---|
| Debtor | A person or company that owes money. Identified by an opaque `debtor_id` inside a product line. |
| Debt | One amount owed, with a due date and a balance. A debtor can have several. |
| Delinquency | A debt past its due date. `days_overdue_max` counts from the oldest open due date. |
| Product line | One data source, for example `auto` or `consumer`. Ids are unique only inside a line, so every key is `(product_line, id)`. |
| Portfolio | A group of debts handled under the same terms. Part of the agreement key (`portfolio_id`). |
| Agreement | A payment plan a debtor signs. Stored as one row per installment. |
| Installment | One scheduled payment of an agreement. Installment 0 is the first installment (the down payment) and is the row that defines the agreement. |
| Valid agreement | Installment 0 of an agreement whose status is one of the configured paid, broken or active labels. A pending exception or a rejection is not valid. |
| Broken agreement | An agreement the debtor stopped paying. An auto-broken agreement is one the system closed after a grace period. Both map to `statuses.broken`. |
| Rewrite | A later row for the same (agreement, portfolio, installment). The latest one wins. |
| Legacy agreement | An agreement issued long before the debt it settles entered the portfolio. It arrived with the portfolio and is dropped. |
| Outreach attempt | One contact event: a call, a message, a dialer attempt. One row in `contact_events`. |
| Answer | An attempt where a person answered (`contact_results.answered = 1`). |
| Right-party contact (CPC) | An attempt where the person answered and was the one wanted (`answered = 1` and `right_party = 1`). |
| Contact category | The closed vocabulary `answered`, `promise`, `wrong_party`, `no_answer`, `voicemail`, `busy`, `invalid_number`, `message`, `other`. |
| Voice attempt | Any category except `message` and `other`. Automated messages are not an attempt to reach a human. |
| Automated agent | An agent whose login matches `agents.automated_patterns`. Its attempts do not count toward the cooldown. |
| Collector | The person who works one operational list. |
| Collection queue | The list of debtors to work, ordered. |
| Capacity | The share of a queue the team can work in a period. |
| Cooldown | The number of days after a human attempt during which a debtor is left alone. |
| Tier | P1, P2 or P3 by rank share, or `NO_CONTACT` for a debtor with no usable phone. |
| Contact factor | A multiplier between `queue.contact_floor` and 1 for how likely the debtor's best phones are to reach someone. |
| Phone score | A 0 to 100 score for one phone from its own dial history. |

## Modelling terms

| Term | Meaning here |
|---|---|
| Snapshot | One row per (debtor, cutoff): features known before the cutoff and the label from the window after it. |
| Cutoff (T) | The date a snapshot is taken. Cutoffs repeat at `snapshots.frequency`. |
| Horizon | The label window length in days (`labels.horizon_days`). |
| Label window | `[T, T + horizon)`. |
| Realised value | Money received inside the label window from agreements that started in it. Used to judge and to grade, never as a feature. |
| As-of date | The date the data is read up to. Without `--as-of` it is the day after the latest timestamp. |
| Close head | Classifier for "a valid agreement starts within the horizon". |
| Break head | Classifier for "the agreement fails". |
| Expected value (EV) | `balance x P(close) x (1 - P(break))`. Adjusted EV multiplies by the contact factor. |
| LambdaRank | A LightGBM objective that optimises the order of a list directly. Here one list is one snapshot. |
| Relevance grade | The integer a ranker is trained on: 0 when nothing was received, otherwise a tier of the amount. |
| Holdout | The most recent dates, scored once for the report. |
| Purge | Dropping development rows whose label window reaches into the holdout. |
| Walk-forward folds | Expanding training blocks, each tested on the block that follows. |
| Value capture | Share of realised value found in the top of each queue. |
| Uplift | Percentage gain in captured value over a baseline ordering. |
| Cluster bootstrap | A bootstrap that redraws whole debtors, because one debtor appears in many snapshots. |
| Right-censoring | An outcome that has not happened yet (an active agreement). It is not a failure. |
| Bundle | The saved models plus metadata (`bundle.joblib`). |

## `snapshots.csv`

| Column | Meaning |
|---|---|
| `product_line`, `debtor_id` | Key. |
| `snapshot_at` | The cutoff. |
| `balance` | Sum of the balances of the debts open at the cutoff. |
| `product_id`, `phase_id` | Categorical, the largest value among the open debts (`NA` when absent). |
| `days_overdue_max`, `days_overdue_min`, `overdue_spread` | Days since the oldest and the latest due date among open debts, and their difference. |
| `tenure_days` | Days since the oldest open debt entered. Kept for context, not a model feature. |
| `is_company` | 1 for a company. |
| `phone_count`, `email_count` | Records known before the cutoff. |
| `n_answered`, `n_wrong_party`, `n_invalid_number` | Counts of attempts before the cutoff by outcome. A recorded promise counts as answered. |
| `answer_rate` | Answered voice attempts over voice attempts. Missing when there were none. |
| `days_since_answered` | Days since the last answered attempt. Missing when there never was one. |
| `contact_factor`, `usable_phones`, `best_phone_score` | Reachability from the phone scores at the cutoff. |
| `closed` | 1 when a valid agreement started in the window. |
| `realized_value` | Money received in the window from agreements that started in it. |

Columns added with `features.extra_numeric` or `features.extra_categorical` appear after these.

## `break.csv`

One row per debtor with a resolved, mature agreement.

| Column | Meaning |
|---|---|
| `anchor_at` | Date of the first valid agreement. Features are measured then. |
| base feature columns | As in `snapshots.csv`, measured at `anchor_at`. |
| `n_paid`, `n_broken`, `n_active` | Count of the debtor's valid agreements by outcome. |
| `resolved` | Something was paid or broken. |
| `mature` | The first agreement is older than `labels.break_observation_days`. |
| `broke` | 1 when none of the debtor's agreements was paid. |

## `queue/queue_full.csv`

Every debtor outside the cooldown, best first.

| Column | Meaning |
|---|---|
| `rank` | Position in the full order. Contactable debtors come first. |
| feature columns | The snapshot features as of the queue date. |
| `p_close`, `p_break` | Head probabilities. |
| `score_<strategy>` | The score of each strategy (see `docs/evaluation.md`). |
| `last_human_at`, `days_since_contact`, `contact_status` | Last attempt by a human agent. Status is `never_contacted` or `cold`. |
| `phone_1_id` to `phone_3_id`, `phone_<n>_score` | The three best usable phones, as ids. |
| `ranking_score` | The score of the configured `ranking.strategy`. |
| `expected_value`, `adjusted_value` | EV without and with the contact factor. |
| `contactable` | Has at least one usable phone. |
| `tier` | `P1`, `P2`, `P3` or `NO_CONTACT`. |

## `queue/queue_operational.csv`

The file that goes to the team. It holds no scores and no personal data.

| Column | Meaning |
|---|---|
| `collector` | Collector number, 1 to `queue.collectors`. |
| `order` | Position inside that collector's list. |
| `product_line`, `debtor_id` | Key, to join back to your own system. |
| `phone_1_id` to `phone_3_id` | Phone ids to dial, best first. The numbers stay in your system. |
| `balance` | Open balance, rounded to two decimals. |
| `days_overdue` | Days since the oldest open due date. |
| `last_contact_status`, `days_since_last_contact` | `never_contacted` or `cold`, and days since the last human attempt. |
