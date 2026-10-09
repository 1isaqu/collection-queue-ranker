"""Point-in-time datasets.

The close head is trained on **snapshots**: one row per (debtor, cutoff T) with

* features from rows dated before T only (debts open at T, contacts before T, phones known at T),
* a population of debtors with a positive balance and no valid agreement issued before T,
* a label looking forward from T: did a valid agreement start in [T, T + horizon)?

The row for "now" in production is built by the very same `features_at`, so a feature cannot mean
one thing in training and another when scoring. The break head uses one row per debtor, measured at
the date of the debtor's first valid agreement.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.features.base import base_features, make_anchors
from collection_queue_ranker.features.contact import attach_contact
from collection_queue_ranker.features.phones import attach_contact_factor, phone_quality
from collection_queue_ranker.labels import (
    Agreements,
    agreements_before,
    break_cohort,
    window_labels,
)
from collection_queue_ranker.tables import KEYS, RawTables

log = logging.getLogger(__name__)

Enricher = Callable[[pd.DataFrame], pd.DataFrame]


def infer_as_of(raw: RawTables) -> pd.Timestamp:
    """The day after the latest timestamp in the data."""
    return raw.latest_timestamp() + pd.Timedelta(days=1)


def cutoff_dates(raw: RawTables, settings: Settings, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    """Snapshot dates whose label window is fully observed: the last one is as_of - horizon."""
    snap = settings.snapshots
    if snap.first_cutoff:
        start = pd.Timestamp(snap.first_cutoff)
    else:
        seen = [raw.debts["entered_at"].min(), raw.events["occurred_at"].min()]
        earliest = min(s for s in seen if pd.notna(s))
        start = earliest.normalize() + pd.Timedelta(days=snap.warmup_days)
    end = pd.Timestamp(as_of) - pd.Timedelta(days=settings.labels.horizon_days)
    dates = pd.date_range(start, end, freq=snap.frequency)
    if len(dates) == 0:
        raise DataError(
            f"no snapshot dates between {start.date()} and {end.date()}: the data is too short "
            "for the configured horizon and warm-up"
        )
    return dates


def features_at(
    raw: RawTables,
    prep: Agreements,
    cutoff: pd.Timestamp,
    settings: Settings,
    *,
    exclude_existing_agreements: bool = True,
) -> pd.DataFrame:
    """Everything known about the working population at `cutoff`, with no label.

    Columns: keys, `snapshot_at`, base features, contact features, `contact_factor`,
    `usable_phones`, and the context column `tenure_days`.
    """
    cutoff = pd.Timestamp(cutoff)
    anchors = make_anchors(raw.debts, cutoff)
    base = base_features(
        raw.debts, raw.debtors, raw.phones, raw.emails, anchors, settings.features.balance_column
    )
    base = base[base["balance"] > settings.snapshots.min_balance]
    if exclude_existing_agreements:
        done = agreements_before(prep, cutoff)
        if not done.empty:
            marked = base.merge(done.assign(_done=True), on=KEYS, how="left")["_done"]
            base = base[marked.isna().to_numpy()]
    out = attach_contact(base.reset_index(drop=True), raw.events)
    quality = phone_quality(raw.phones, raw.events, cutoff, settings.phones)
    out = attach_contact_factor(out, quality, settings.phones, settings.queue)
    out["snapshot_at"] = cutoff
    return out.drop(columns=["anchor_at"])


def _label_snapshot(
    frame: pd.DataFrame, prep: Agreements, cutoff: pd.Timestamp, settings: Settings
) -> pd.DataFrame:
    labels = window_labels(prep, cutoff, settings.labels.horizon_days)
    out = frame.merge(labels, on=KEYS, how="left") if not labels.empty else frame.assign(
        closed=np.nan, realized_value=np.nan
    )
    out["closed"] = out["closed"].fillna(0).astype(int)
    out["realized_value"] = out["realized_value"].fillna(0.0)
    return out


def build_snapshots(
    raw: RawTables,
    prep: Agreements,
    settings: Settings,
    as_of: pd.Timestamp,
    enricher: Enricher | None = None,
) -> pd.DataFrame:
    """Stack labelled snapshots for every cutoff date.

    `enricher`, if given, receives each snapshot frame (it has `snapshot_at`) and returns it with
    extra columns, for example an external score. It must be point-in-time: use the value that was
    known at `snapshot_at`, never today's. List the added columns in `features.extra_numeric` /
    `features.extra_categorical` so the models pick them up.
    """
    frames = []
    for cutoff in cutoff_dates(raw, settings, as_of):
        frame = features_at(raw, prep, cutoff, settings)
        if frame.empty:
            continue
        frame = _label_snapshot(frame, prep, cutoff, settings)
        frames.append(enricher(frame) if enricher else frame)
        log.info("snapshot %s: %d rows, %.2f%% closed", cutoff.date(), len(frame),
                 100 * frame["closed"].mean())
    if not frames:
        raise DataError("no snapshot has any debtor in the working population")
    return pd.concat(frames, ignore_index=True)


def break_features(raw: RawTables, cohort: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Base features measured at each cohort debtor's `anchor_at`, joined with the cohort labels."""
    anchors = cohort[[*KEYS, "anchor_at"]]
    base = base_features(
        raw.debts, raw.debtors, raw.phones, raw.emails, anchors, settings.features.balance_column
    )
    base = base[base["balance"] > settings.snapshots.min_balance]
    return base.merge(cohort.drop(columns=["anchor_at"]), on=KEYS, how="inner")


def build_break_dataset(
    raw: RawTables, prep: Agreements, settings: Settings, as_of: pd.Timestamp
) -> pd.DataFrame:
    """One row per debtor whose agreements are resolved and old enough to trust.

    Active-only debtors are excluded: they may still pay or break, so labelling them would put a
    pending outcome in the same class as a failure (right-censoring).
    """
    cohort = break_cohort(prep, as_of, settings)
    cohort = cohort[cohort["resolved"] & cohort["mature"]]
    if cohort.empty:
        raise DataError("no resolved, mature agreements: cannot build the break dataset")
    return break_features(raw, cohort, settings).reset_index(drop=True)
