"""Funnel diagnostics: is the queue pointing at people we can actually reach?

A ranker can be right about *who is willing* and still be useless if the operation cannot reach
them. The funnel is

    P(recover) = P(valid phone) x P(answer | valid) x P(agree | answered) x P(pay | agreed)

and these functions measure each factor from the contract tables, then cross the model's ranking
with phone quality ("high score, bad phone"). They are pure functions over frames, so you can run them
on any slice. The report carries two caveats: the phone score is computed from the same history it
is judged against (circular), and "contact lifts recovery" is correlational, because people who
answer differ from people who do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from collection_queue_ranker.config import DiagnosticsSettings, Settings
from collection_queue_ranker.contacts import ANSWERED, VOICE
from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.features.phones import phone_quality
from collection_queue_ranker.labels import Agreements
from collection_queue_ranker.tables import KEYS, RawTables
from collection_queue_ranker.util import debtor_key

_DEFAULTS = DiagnosticsSettings()


def _checked_edges(name: str, edges: Sequence[float]) -> list[float]:
    values = [float(e) for e in edges]
    if not values or values[0] <= 0 or any(b <= a for a, b in pairwise(values)):
        raise ConfigError(f"diagnostics.{name} must be a non-empty, increasing list of positive numbers")
    return values


def _score_bands(edges: Sequence[float]) -> tuple[list[float], list[str]]:
    """Bin edges and labels for phone scores. Debtors with no phone fall into the first band."""
    cuts = _checked_edges("phone_score_edges", edges)
    bins = [-1.0, 0.0, *cuts, np.inf]
    labels = ["none", f"0-{cuts[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in pairwise(cuts)]
    return bins, [*labels, f"{cuts[-1]:g}+"]


def _age_bands(edges: Sequence[float]) -> tuple[list[float], list[str]]:
    """Bin edges and labels for phone age in years."""
    cuts = _checked_edges("phone_age_edges_years", edges)
    labels = [f"<{cuts[0]:g}y"]
    labels += [f"{lo:g}-{hi:g}y" for lo, hi in pairwise(cuts)]
    return [-1.0, *cuts, np.inf], [*labels, f">{cuts[-1]:g}y"]


def phone_choice_report(
    quality: pd.DataFrame, kill_count: int, weak_score: float = _DEFAULTS.weak_phone_score
) -> str:
    """How good is the arbitrary choice (lowest phone id) compared with the best phone?"""
    if quality.empty:
        return "no phones"
    first = quality.sort_values("phone_id", kind="stable").groupby(KEYS).first()
    best = quality.groupby(KEYS)["score"].max()
    condemned = (first["dead"] >= kill_count) & (first["contacts"] == 0)
    return "\n".join([
        f"Debtors with phones: {len(first)}",
        f"- first-listed phone flagged invalid: {(first['is_valid'] == 0).mean():.1%}",
        f"- first-listed phone condemned by history (dead number, never answered): "
        f"{condemned.mean():.1%}",
        f"- first-listed phone weak (score < {weak_score:g}): {(first['score'] < weak_score).mean():.1%}",
        f"- a better phone existed for the same debtor: "
        f"{(first['score'] < best.reindex(first.index)).mean():.1%}",
    ])


def dialled_below_best(
    events: pd.DataFrame, quality: pd.DataFrame, margin: float = _DEFAULTS.below_best_margin
) -> str:
    """Share of attempts made on a phone scoring `margin` points below the debtor's best phone."""
    dialled = events[events["phone_id"].notna()]
    if dialled.empty or quality.empty:
        return "no dialled phones"
    score = quality.set_index(["product_line", "phone_id"])["score"]
    best = quality.groupby(KEYS)["score"].max()
    keys = pd.MultiIndex.from_frame(dialled[["product_line", "phone_id"]])
    used = pd.Series(score.reindex(keys).to_numpy(), index=dialled.index)
    top = pd.Series(best.reindex(pd.MultiIndex.from_frame(dialled[KEYS])).to_numpy(),
                    index=dialled.index)
    share = float((used < top - margin).mean())
    return (f"Attempts with a known phone: {len(dialled)}\n- dialled a phone more than {margin:g} "
            f"points below the best available: {share:.1%}\n- caveat: the score comes from the same "
            "history, so this is circular; read it as 'how many calls went to phones we now know "
            "are worse'.")


def answer_rate_by_month(events: pd.DataFrame) -> pd.DataFrame:
    """Voice attempts and answered share per calendar month."""
    voice = events[events["category"].isin(VOICE)]
    if voice.empty:
        return pd.DataFrame(columns=["month", "voice_attempts", "answered", "answer_rate"])
    month = voice["occurred_at"].dt.to_period("M").astype(str)
    grouped = voice.assign(month=month, answered=voice["category"].isin(ANSWERED)).groupby(
        "month", as_index=False).agg(voice_attempts=("answered", "size"), answered=("answered", "sum"))
    grouped["answer_rate"] = (grouped["answered"] / grouped["voice_attempts"]).round(3)
    return grouped


def _debtor_flags(events: pd.DataFrame, prep: Agreements) -> pd.DataFrame:
    voice = events[events["category"].isin(VOICE)]
    keyed = voice.assign(
        key=debtor_key(voice), ans=voice["category"].isin(ANSWERED),
        prom=voice["category"].eq("promise"),
    )
    flags = keyed.groupby("key").agg(answered=("ans", "max"), promised=("prom", "max"))
    first = prep.first.assign(key=debtor_key(prep.first))
    flags["agreed"] = flags.index.isin(first["key"])
    paid = first[first["outcome"] == "paid"]["key"]
    flags["paid"] = flags.index.isin(paid)
    return flags


def contact_lift(events: pd.DataFrame, prep: Agreements) -> str:
    """Agreement rate for debtors who answered vs those who never did (correlational)."""
    flags = _debtor_flags(events, prep)
    if flags.empty:
        return "no voice attempts"
    with_answer, without = flags[flags["answered"]], flags[~flags["answered"]]
    rate_with = with_answer["agreed"].mean() if len(with_answer) else float("nan")
    rate_without = without["agreed"].mean() if len(without) else float("nan")
    lift = rate_with / rate_without if rate_without else float("inf")
    return (f"Debtors dialled by voice: {len(flags)}\n"
            f"- answered at least once: {len(with_answer)}, agreement rate {rate_with:.2%}\n"
            f"- never answered: {len(without)}, agreement rate {rate_without:.2%}\n"
            f"- lift of an answer: {lift:.1f}x\n"
            "- caveat: correlational. People who answer differ from people who do not.")


def stage_losses(events: pd.DataFrame, prep: Agreements) -> str:
    flags = _debtor_flags(events, prep)
    if flags.empty:
        return "no voice attempts"
    stages = [
        ("dialled by voice", len(flags)), ("answered", int(flags["answered"].sum())),
        ("promised", int(flags["promised"].sum())), ("agreement", int(flags["agreed"].sum())),
        ("agreement paid", int(flags["paid"].sum())),
    ]
    lines, previous = [], None
    for name, count in stages:
        step = f"  (step conversion {count / previous:.1%})" if previous else ""
        lines.append(f"{name}: {count}{step}")
        previous = count or 1
    return "\n".join(lines)


def concentration(
    quality: pd.DataFrame,
    debts: pd.DataFrame,
    as_of: pd.Timestamp,
    age_edges: Sequence[float] = _DEFAULTS.phone_age_edges_years,
) -> str:
    """Best phone score by debt-size quartile and phone age: concentrated or homogeneous?"""
    if quality.empty:
        return "no phones"
    best = quality.groupby(KEYS).agg(best_score=("score", "max"), created=("created_at", "min"))
    size = debts.groupby(KEYS)["balance"].sum().rename("balance")
    table = best.join(size, how="inner").reset_index()
    if len(table) < 8:
        return "too few debtors for a breakdown"
    table["debt_quartile"] = pd.qcut(table["balance"].rank(method="first"), 4,
                                     labels=["Q1_small", "Q2", "Q3", "Q4_large"])
    age = (as_of - table["created"]).dt.days / 365
    bins, labels = _age_bands(age_edges)
    table["phone_age"] = pd.cut(age, bins, labels=labels)
    parts = ["Mean best-phone score by dimension (a large spread means a concentrated problem):"]
    for dim in ("debt_quartile", "phone_age"):
        parts.append(f"\nby {dim}:\n" + table.groupby(dim, observed=True)["best_score"].agg(
            ["mean", "count"]).round(1).to_string())
    return "\n".join(parts)


def model_vs_phone_matrix(
    queue_full: pd.DataFrame,
    weak_score: float = _DEFAULTS.weak_phone_score,
    score_edges: Sequence[float] = _DEFAULTS.phone_score_edges,
) -> str:
    """Cross the ranking with phone quality: how much of the top is practically unreachable?"""
    frame = queue_full.copy()
    frame["quintile"] = pd.qcut(frame["ranking_score"].rank(method="first"), 5,
                                labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    best = frame["best_phone_score"].fillna(-0.5)
    bins, labels = _score_bands(score_edges)
    frame["phone_band"] = pd.cut(best, bins, labels=labels)
    matrix = pd.crosstab(frame["quintile"], frame["phone_band"])
    top = frame[frame["quintile"] == "Q5_high"]
    weak = float((top["best_phone_score"].fillna(0) < weak_score).mean())
    return (f"Ranking quintile x best phone score:\n{matrix.to_string()}\n\n"
            f"- share of the top quintile whose best phone scores below {weak_score:g}, "
            f"or that has none: {weak:.1%}\n"
            "- those are the 'high score, bad phone' debtors: money the model sees and the "
            "operation cannot reach by voice.")


def run_diagnostics(
    raw: RawTables, prep: Agreements, settings: Settings, as_of: pd.Timestamp,
    queue_full: pd.DataFrame | None = None,
) -> list[tuple[str, str]]:
    """Every section as (title, text). `queue_full` adds the model-versus-phone matrix."""
    cfg = settings.diagnostics
    quality = phone_quality(raw.phones, raw.events, as_of, settings.phones)
    months = answer_rate_by_month(raw.events)
    sections = [
        ("1. How good is an arbitrary phone choice?",
         phone_choice_report(quality, settings.phones.invalid_kill_count, cfg.weak_phone_score)),
        ("2. Were calls placed on the best available phone?",
         dialled_below_best(raw.events, quality, cfg.below_best_margin)),
        ("3. Answer rate by month",
         months.to_string(index=False) if len(months) else "no voice attempts"),
        ("4. Does an answer raise the agreement rate?", contact_lift(raw.events, prep)),
        ("5. Loss at each funnel stage", stage_losses(raw.events, prep)),
        ("6. Is the reachability problem concentrated?",
         concentration(quality, raw.debts, as_of, cfg.phone_age_edges_years)),
    ]
    if queue_full is not None and "best_phone_score" in queue_full:
        sections.append(("7. Does the ranking prioritise unreachable debtors?",
                         model_vs_phone_matrix(queue_full, cfg.weak_phone_score,
                                               cfg.phone_score_edges)))
    return sections


def render_diagnostics(sections: list[tuple[str, str]], as_of: pd.Timestamp) -> str:
    body = "\n\n".join(f"## {title}\n\n```\n{text}\n```" for title, text in sections)
    header = f"Reference date {pd.Timestamp(as_of):%Y-%m-%d}."
    funnel = "P(recover) = P(valid phone) x P(answer | valid) x P(agree | answered) x P(pay | agreed)."
    return "\n".join(["# Funnel diagnostics", "", header, funnel, "", body, ""])

