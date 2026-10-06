"""Contact outcomes: a small closed vocabulary on top of the result catalogue.

The contract only guarantees two flags per result (`answered`, `right_party`). The features and the
phone-quality score need a little more: was it a voice attempt at all, a wrong party, a dead number?
That is the optional `contact_results.category` column. When it is absent a category is derived from
the flags. `draft_categories` proposes categories from the result labels so a human can review them
once, which is much cheaper than building the map by hand.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd

from collection_queue_ranker.config import AgentSettings
from collection_queue_ranker.errors import ContractError

CATEGORIES: tuple[str, ...] = (
    "answered", "promise", "wrong_party", "no_answer", "voicemail", "busy",
    "invalid_number", "message", "other",
)
# An attempt that rang a human line. Automated messages and unclassified results are not voice.
VOICE: frozenset[str] = frozenset(CATEGORIES) - {"message", "other"}
# Somebody on the line who is the debtor (or acts for them).
ANSWERED: frozenset[str] = frozenset({"answered", "promise"})

_NEEDS_ANSWER = frozenset({"answered", "promise", "wrong_party"})
_NEEDS_NO_ANSWER = frozenset({"no_answer", "voicemail", "busy", "invalid_number"})


def derive_category(answered: int, right_party: int) -> str:
    """Category implied by the two contract flags alone."""
    if not answered:
        return "no_answer"
    return "answered" if right_party else "wrong_party"


def resolve_categories(results: pd.DataFrame) -> pd.DataFrame:
    """Return `results` with a complete, validated `category` column."""
    out = results.copy()
    if "category" not in out:
        out["category"] = np.nan
    given = out["category"].astype("string").str.strip().str.lower()
    unknown = sorted(set(given.dropna()) - set(CATEGORIES))
    if unknown:
        raise ContractError(f"unknown contact category value(s) {unknown}; use {CATEGORIES}")
    derived = [
        derive_category(a, r)
        for a, r in zip(out["answered"].fillna(0), out["right_party"].fillna(0), strict=True)
    ]
    out["category"] = given.where(given.notna(), pd.Series(derived, index=out.index))
    return out


def inconsistent_categories(results: pd.DataFrame) -> pd.DataFrame:
    """Rows whose category contradicts the `answered` flag (a review aid, not an error)."""
    res = resolve_categories(results)
    answered = res["answered"].fillna(0).astype(int) == 1
    bad = ((res["category"].isin(_NEEDS_ANSWER)) & ~answered) | (
        (res["category"].isin(_NEEDS_NO_ANSWER)) & answered
    )
    return res[bad]


# Ordered: the first rule that matches wins. A heuristic to start a review, not an authority.
_DRAFT_RULES: tuple[tuple[str, str], ...] = (
    ("message", r"\b(sms|text message|messaging|auto(mated)? ?message|voicebot|robocall|broadcast|"
                r"bulk|blast|dispatch(ed)?)\b"),
    ("promise", r"promis|agreement|negotiat|settle|payment plan"),
    ("wrong_party", r"wrong (party|person|number)|does not know|unknown (person|debtor)|"
                    r"not the (debtor|owner)|moved away"),
    ("invalid_number", r"invalid|does not exist|disconnected|out of service|fax|unallocated|"
                       r"number not in use"),
    ("voicemail", r"voice ?mail|answering machine|mailbox"),
    ("busy", r"\bbusy\b"),
    ("no_answer", r"no answer|not answer|unanswered|no response|rings? out|silent|dropped|"
                  r"poor line|bad line|failed"),
    ("answered", r"answered|spoke|contact made|call ?back|right party|owner|message left|"
                 r"hung up|requested|callback"),
)


def _normalise(text: object) -> str:
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def draft_category(label: object) -> str:
    text = _normalise(label)
    for category, pattern in _DRAFT_RULES:
        if re.search(pattern, text):
            return category
    return "other"


def draft_categories(results: pd.DataFrame) -> pd.DataFrame:
    """Propose a `category` for every result from its label. Review before using it."""
    out = results.copy()
    labels = out["label"] if "label" in out else pd.Series([""] * len(out), index=out.index)
    out["category"] = [draft_category(v) for v in labels]
    return out


def like_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a SQL LIKE pattern (`%` and `_`) into an anchored, case-insensitive regex."""
    parts = []
    for ch in pattern:
        parts.append(".*" if ch == "%" else "." if ch == "_" else re.escape(ch))
    return re.compile("^" + "".join(parts) + "$", re.IGNORECASE)


def matches_any(value: object, patterns: Iterable[str]) -> bool:
    if not isinstance(value, str):
        return False
    return any(like_to_regex(p).match(value.strip()) for p in patterns)


def automated_agents(agents: pd.DataFrame, settings: AgentSettings) -> pd.DataFrame:
    """Agents whose login matches an automated-dialer pattern: (product_line, agent_id)."""
    if agents.empty:
        return pd.DataFrame(columns=["product_line", "agent_id"])
    mask = agents["login"].map(lambda v: matches_any(v, settings.automated_patterns))
    return agents.loc[mask, ["product_line", "agent_id"]].drop_duplicates()


def annotate_events(
    events: pd.DataFrame,
    results: pd.DataFrame,
    agents: pd.DataFrame,
    settings: AgentSettings,
) -> pd.DataFrame:
    """Attach `category` and `is_automated` to contact events."""
    out = events.drop(columns=["category", "is_automated"], errors="ignore")
    if out.empty:
        return out.assign(category=pd.Series(dtype="object"),
                          is_automated=pd.Series(dtype="bool"))
    cats = resolve_categories(results)[["product_line", "result_id", "category"]]
    out = out.merge(cats.drop_duplicates(["product_line", "result_id"]),
                    on=["product_line", "result_id"], how="left")
    out["category"] = out["category"].astype("object").where(out["category"].notna(), "other")
    auto = automated_agents(agents, settings)
    if auto.empty:
        out["is_automated"] = False
    else:
        out = out.merge(auto.assign(is_automated=True), on=["product_line", "agent_id"],
                        how="left")
        out["is_automated"] = out["is_automated"].eq(True)
    return out
