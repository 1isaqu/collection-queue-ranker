"""Build, render and write the evaluation report for a holdout."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.evaluation.backtest import (
    robustness_by_queue,
    run_backtest,
    strategies_in,
    verdict,
)
from collection_queue_ranker.evaluation.metrics import (
    auc_with_ci,
    calibration_summary,
    discrimination_verdict,
    gain_table,
    lift_at,
    ndcg_at,
    precision_at,
    recall_at,
)
from collection_queue_ranker.models import ModelBundle, relevance_grades
from collection_queue_ranker.scoring import DESCRIPTIONS

HEADLINE_CAPACITY = 0.20


def jsonable(value: Any) -> Any:
    """Convert numpy / pandas values to plain JSON types (NaN and inf become null)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient="records"))
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def head_report(
    y: np.ndarray,
    proba: np.ndarray,
    settings: Settings,
    groups: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Discrimination, calibration and capacity metrics for one probability head."""
    ev = settings.evaluation
    value, low, high = auc_with_ci(
        y, proba, n_boot=ev.n_bootstrap, level=ev.ci_level, seed=ev.seed, groups=groups
    )
    ks = ev.precision_ks
    return {
        "rows": len(y), "positives": int(np.sum(y)),
        "base_rate": float(np.mean(y)) if len(y) else float("nan"),
        "auc": value, "auc_ci": [low, high],
        "verdict": discrimination_verdict(value, low, high),
        "precision_at": {str(k): precision_at(y, proba, k) for k in ks},
        "recall_at": {str(k): recall_at(y, proba, k) for k in ks},
        "lift_at": {str(k): lift_at(y, proba, k) for k in ks},
        "calibration": calibration_summary(y, proba),
        "gain_table": gain_table(y, proba),
    }


def _mean_ndcg(frame: pd.DataFrame, grades: np.ndarray, capacity: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in strategies_in(frame):
        per_queue = []
        for _, idx in frame.groupby("snapshot_at").indices.items():
            per_queue.append(ndcg_at(grades[idx], frame[f"score_{name}"].to_numpy()[idx], capacity))
        out[name] = float(np.nanmean(per_queue)) if per_queue else float("nan")
    return out


def ranking_report(
    holdout: pd.DataFrame, bundle: ModelBundle, settings: Settings
) -> dict[str, Any]:
    """Backtest of every strategy plus the headline decision."""
    capacities = settings.ranking.capacities
    result = run_backtest(holdout, capacities, settings.evaluation)
    headline = min(capacities, key=lambda c: abs(c - HEADLINE_CAPACITY))
    chosen = settings.ranking.strategy
    uplift = result.uplift(chosen, headline)
    if bundle.ranker is not None:
        grades, _ = relevance_grades(
            holdout, bundle.ranker.relevance, bundle.ranker.n_grades, bundle.ranker.edges
        )
    else:
        grades = holdout["closed"].to_numpy().astype(int)
    return {
        "headline_capacity": headline, "strategy": chosen,
        "strategy_description": DESCRIPTIONS.get(chosen, ""),
        "uplift_vs_balance_pct": uplift, "verdict": verdict(uplift, settings.evaluation),
        "total_realized_value": result.total_value, "queues": result.queues,
        "rows": result.rows, "table": result.table,
        "mean_ndcg_at_headline": _mean_ndcg(holdout, grades, headline),
    }


def build_report(
    holdout: pd.DataFrame,
    bundle: ModelBundle,
    settings: Settings,
    *,
    break_holdout: pd.DataFrame | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`holdout` is a scored frame (`scoring.scored_frame`)."""
    debtors = holdout["product_line"].astype(str) + "|" + holdout["debtor_id"].astype(str)
    report: dict[str, Any] = {
        "data": data or {},
        "close_head": head_report(
            holdout["closed"].to_numpy(), holdout["p_close"].to_numpy(), settings, debtors
        ),
        "ranking": ranking_report(holdout, bundle, settings),
        "robustness": robustness_by_queue(
            holdout, "p_close", min_auc=settings.evaluation.min_auc_alert
        ),
    }
    if break_holdout is not None and len(break_holdout):
        proba = bundle.brk.predict(break_holdout)
        report["break_head"] = head_report(break_holdout["broke"].to_numpy(), proba, settings)
    return report


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _head_section(title: str, head: dict[str, Any]) -> list[str]:
    low, high = head["auc_ci"]
    lines = [
        f"## {title}", "",
        f"- rows: {head['rows']}, positives: {head['positives']} "
        f"(base rate {_fmt(head['base_rate'], 4)})",
        f"- AUC {_fmt(head['auc'])} (interval {_fmt(low)} to {_fmt(high)}): {head['verdict']}",
        f"- calibration: mean predicted {_fmt(head['calibration']['mean_predicted'], 4)} "
        f"vs observed {_fmt(head['calibration']['observed_rate'], 4)}",
        "", "| capacity | precision | recall (gain) | lift |", "|---|---|---|---|",
    ]
    for k in head["precision_at"]:
        lines.append(
            f"| {k} | {_fmt(head['precision_at'][k])} | {_fmt(head['recall_at'][k])} "
            f"| {_fmt(head['lift_at'][k], 2)} |"
        )
    return [*lines, ""]


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable report. Numbers come from the holdout the models never saw."""
    lines = ["# Holdout evaluation", ""]
    data = report.get("data", {})
    if data:
        lines += ["## Data", ""] + [f"- {k}: {_fmt(v)}" for k, v in data.items()] + [""]
    lines += _head_section("Close head: P(agreement within horizon)", report["close_head"])
    if "break_head" in report:
        lines += _head_section("Break head: P(agreement breaks)", report["break_head"])
    rank = report["ranking"]
    lines += [
        "## Ranking under limited capacity", "",
        f"- production strategy: `{rank['strategy']}` ({rank['strategy_description']})",
        f"- headline capacity {rank['headline_capacity']:.0%}: uplift over the balance rule "
        f"{_fmt(rank['uplift_vs_balance_pct'], 1)}%, {rank['verdict']}",
        f"- queues: {rank['queues']}, rows: {rank['rows']}, "
        f"total realised value: {_fmt(rank['total_realized_value'], 2)}", "",
        "| strategy | capacity | captured share | vs balance rule % | interval % | vs random % |",
        "|---|---|---|---|---|---|",
    ]
    for row in rank["table"].to_dict(orient="records"):
        interval = (f"{_fmt(row['ci_low_pct'], 1)} to {_fmt(row['ci_high_pct'], 1)}"
                    if np.isfinite(row["ci_low_pct"]) else "n/a")
        lines.append(
            f"| {row['strategy']} | {row['capacity']:.0%} | {_fmt(row['captured_share'])} "
            f"| {_fmt(row['uplift_vs_balance_pct'], 1)} | {interval} "
            f"| {_fmt(row['uplift_vs_random_pct'], 1)} |"
        )
    lines += ["", "### Mean NDCG at the headline capacity", ""]
    lines += [f"- {k}: {_fmt(v)}" for k, v in rank["mean_ndcg_at_headline"].items()]
    rob = report["robustness"]
    lines += ["", "## Robustness: close-head AUC per queue", "",
              "| queue | rows | positives | auc | alert |", "|---|---|---|---|---|"]
    for row in rob.to_dict(orient="records"):
        lines.append(f"| {pd.Timestamp(row['snapshot_at']).date()} | {row['rows']} "
                     f"| {row['positives']} | {_fmt(row['auc'])} | {'ALERT' if row['alert'] else ''} |")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], directory: str | Path) -> list[Path]:
    """Write `metrics.json`, `report.md`, `backtest.csv`, `gain_table.csv`, `robustness.csv`."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "metrics.json": json.dumps(jsonable(report), indent=2),
        "report.md": render_markdown(report),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    report["ranking"]["table"].to_csv(out / "backtest.csv", index=False)
    report["close_head"]["gain_table"].to_csv(out / "gain_table.csv", index=False)
    report["robustness"].to_csv(out / "robustness.csv", index=False)
    return sorted(out.glob("*"))
