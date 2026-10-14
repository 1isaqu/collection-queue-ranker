"""Run the training and evaluation stages on the public credit-card panel, with the figures.

    pip install -e ".[dev,realdata]"
    python scripts/evaluate_uci_credit_card.py --data data --out artifacts/uci_credit_card \
        --figures docs/img

The first run downloads a 2.9 MB CSV from the UCI repository and checks its hash; `data/` and
`artifacts/` are ignored by git. The script writes the three dataset files that `cqr build-dataset`
would write, trains with the package's `train` (what `cqr train` calls) and writes the usual
reports to `--out`, then adds what the package does not report: simple reference orderings, a
perfect-foresight ceiling, a rolling-origin check, an ablation, several seeds and a stricter
population. Everything is written to `--out/summary.json`; `--figures` also draws four PNG files.
The run is deterministic on one platform and library set.
"""

import argparse
import json
import platform
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from collection_queue_ranker import uci_credit_card as card
from collection_queue_ranker.config import Settings
from collection_queue_ranker.evaluation.backtest import run_backtest, top_value_per_queue
from collection_queue_ranker.evaluation.metrics import auc, bootstrap_ci
from collection_queue_ranker.evaluation.report import jsonable
from collection_queue_ranker.models import ClassifierHead, close_space
from collection_queue_ranker.pipeline import write_datasets
from collection_queue_ranker.split import time_split
from collection_queue_ranker.training import TrainResult, train

HEADLINE = 0.20
COLORS = {
    "model": "#1f6f8b", "band": "#9ccbdc", "balance": "#b0762c", "payment": "#6aa5bf",
    "random": "#7f7f7f", "oracle": "#1b1b1b", "grid": "#d9d9d9", "keep": "#c9c9c9",
}  # fmt: skip
# The ten card features of `uci_credit_card.EXTRA_NUMERIC`, split by what they describe.
FEATURE_GROUPS = {
    "payments": ("paid_last_month", "paid_two_months_ago", "months_paid_recent",
                 "paid_share_of_previous_bill"),
    "bill": ("credit_limit", "utilisation", "bill_growth"),
    "delay": ("months_in_arrears", "worst_delay_history", "delay_change"),
}
LABELS = {
    "ev_contact": "balance x P(close) x (1 - P(break))",
    "prob": "P(close) x (1 - P(break))",
    "lambdarank": "LambdaRank",
    "balance_x_p_close": "balance x P(close)",
    "largest_balance": "largest balance first",
    "last_payment": "largest payment last month",
    "random": "random",
    "oracle": "perfect foresight",
}  # fmt: skip


def run_stages(
    data: card.CardDatasets, settings: Settings, out_dir: Path | None = None
) -> TrainResult:
    """What `cqr train` does: split, select, fit, score the holdout, report."""
    return train(data.snapshots, data.break_df, settings, data.as_of, out_dir=out_dir)


def with_reference_scores(holdout: pd.DataFrame) -> pd.DataFrame:
    """Add the orderings the package does not report.

    `last_payment` is the rule an analyst would try first: work whoever paid the most last month.
    `oracle` knows the amount paid, so it is the most any ordering can capture.
    """
    out = holdout.copy()
    out["score_last_payment"] = out["paid_last_month"]
    out["score_balance_x_p_close"] = out["balance"] * out["p_close"]
    out["score_oracle"] = out["realized_value"]
    return out


def head_to_head(
    frame: pd.DataFrame, candidate: str, reference: str, capacity: float, settings: Settings
) -> dict[str, float]:
    """Uplift of one ordering over another with the package's debtor-level cluster bootstrap."""
    value = frame["realized_value"].to_numpy(dtype=float)
    queues = frame["snapshot_at"].to_numpy()
    a = frame[f"score_{candidate}"].to_numpy(dtype=float)
    b = frame[f"score_{reference}"].to_numpy(dtype=float)
    debtors = (frame["product_line"].astype(str) + "|" + frame["debtor_id"].astype(str)).to_numpy()

    def statistic(rows: np.ndarray) -> float:
        codes = pd.factorize(queues[rows])[0]
        got = top_value_per_queue(value[rows], a[rows], codes, capacity)
        base = top_value_per_queue(value[rows], b[rows], codes, capacity)
        return 100.0 * (got - base) / base if base > 0 else float("nan")

    ev = settings.evaluation
    low, high = bootstrap_ci(
        len(frame), statistic, n_boot=ev.n_bootstrap, level=ev.ci_level, seed=ev.seed,
        groups=debtors,
    )
    return {"uplift_pct": statistic(np.arange(len(frame))), "ci_low_pct": low, "ci_high_pct": high}


def headline_row(report: dict, result: TrainResult, settings: Settings) -> dict:
    """The numbers every variant is compared on."""
    rank, close = report["ranking"], report["close_head"]
    table = rank["table"]
    at = table[np.isclose(table["capacity"], HEADLINE)].set_index("strategy")

    def pick(name: str) -> dict[str, float]:
        if name not in at.index:
            return {}
        row = at.loc[name]
        return {
            "captured_share": float(row["captured_share"]),
            "uplift_pct": float(row["uplift_vs_balance_pct"]),
            "ci_low_pct": float(row["ci_low_pct"]), "ci_high_pct": float(row["ci_high_pct"]),
        }

    frame = with_reference_scores(result.holdout)
    reference = {}
    for name in ("last_payment", "balance_x_p_close", "oracle"):
        reference[name] = float(
            top_value_per_queue(
                frame["realized_value"].to_numpy(dtype=float),
                frame[f"score_{name}"].to_numpy(dtype=float),
                pd.factorize(frame["snapshot_at"])[0], HEADLINE,
            ) / frame["realized_value"].sum()
        )
    return {
        "data": report["data"], "close_auc": close["auc"], "close_auc_ci": close["auc_ci"],
        "break_auc": report.get("break_head", {}).get("auc"),
        "headline_strategy": rank["strategy"], "verdict": rank["verdict"],
        "strategies": {n: pick(n) for n in ("ev_contact", "prob", "lambdarank", "largest_balance")},
        "captured_share_reference": reference,
    }


def rolling_origin(data: card.CardDatasets, settings: Settings, index: int) -> dict:
    """Train on every cutoff before cutoff `index` and judge on that one queue alone."""
    cutoff = card.cutoff_dates()[index]
    tuned = settings.with_overrides(
        {"split": {"holdout_fraction": 0.2, "cv_folds": min(2, index - 1)}}
    )
    part = card.CardDatasets(
        data.snapshots[data.snapshots["snapshot_at"] <= cutoff].reset_index(drop=True),
        data.break_df[data.break_df["anchor_at"] <= cutoff].reset_index(drop=True),
        data.as_of, data.quality,
    )
    result = run_stages(part, tuned)
    frame = with_reference_scores(result.holdout)
    row = headline_row(result.report, result, tuned)
    row["holdout_cutoff"] = cutoff.date().isoformat()
    row["development_cutoffs"] = int(result.split.dev["snapshot_at"].nunique())
    row["versus_last_payment"] = head_to_head(frame, "ev_contact", "last_payment", HEADLINE, tuned)
    return row


def horizon_check(
    panel: card.Panel, data: card.CardDatasets, settings: Settings, horizons: tuple[int, ...] = (1, 2, 3)
) -> list[dict]:
    """AUC of the close head when its label moves further from the cutoff.

    The label becomes "paid something `h` months after the cutoff". A signal that comes from a
    one-step leak would fall away at once; one that comes from how payments persist fades. There
    is no purge here, which can only flatter the longer horizons because their label windows
    overlap the test queue, so the fade is not an artefact of the check.
    """
    cutoffs = {c: i for i, c in enumerate(card.cutoff_dates())}
    month = data.snapshots["snapshot_at"].map(cutoffs).to_numpy()
    row_of = pd.Series(np.arange(len(panel)), index=panel.client_id)
    rows = row_of.reindex(data.snapshots["debtor_id"]).to_numpy()
    out = []
    for h in horizons:
        usable = month + h <= len(card.MONTHS) - 1
        frame = data.snapshots[usable].copy()
        frame["closed"] = (panel.paid[rows[usable], month[usable] + h] > 0).astype(int)
        split = time_split(frame, "snapshot_at", settings.split.holdout_fraction, 30 * h,
                           purge=False)
        head = ClassifierHead(
            "close", close_space(settings), dict(settings.model.close_params), settings.model.seed
        )
        head.fit(split.dev, split.dev["closed"].to_numpy())
        holdout = split.holdout
        out.append({
            "months_ahead": h, "development_cutoffs": int(split.dev["snapshot_at"].nunique()),
            "holdout_cutoffs": int(holdout["snapshot_at"].nunique()),
            "holdout_rows": len(holdout), "paying_share": float(holdout["closed"].mean()),
            "auc": auc(holdout["closed"].to_numpy(), head.predict(holdout)),
        })
    return out


def status_transitions(panel: card.Panel) -> dict[str, list[int]]:
    """Clients that move between delay codes from one month to the next, April to September.

    A delay counter would send a client from 1 to 2 to 3. In the file clients go from 0 straight
    to 2 and back to 0, and few go from 2 to 3.
    """
    before, after = panel.status[:, :-1], panel.status[:, 1:]
    return {
        "zero_to_two": [int(x) for x in ((before == 0) & (after == 2)).sum(axis=0)],
        "two_to_zero": [int(x) for x in ((before == 2) & (after == 0)).sum(axis=0)],
        "two_to_three": [int(x) for x in ((before == 2) & (after == 3)).sum(axis=0)],
    }


def draw_figures(
    panel: card.Panel, data: card.CardDatasets, result: TrainResult, extended: pd.DataFrame,
    rolling: list[dict], target: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "figure.dpi": 110,
        }
    )
    target.mkdir(parents=True, exist_ok=True)
    months = [pd.Timestamp(m + "-01").strftime("%b") for m in card.MONTHS]

    # 1. who is in the queue, and the delay codes the population rule rests on
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    by_cutoff = data.snapshots.groupby("snapshot_at").agg(
        rows=("closed", "size"), paid=("closed", "mean")
    )
    x = np.arange(len(by_cutoff))
    axes[0].bar(x, by_cutoff["rows"], color=COLORS["band"], width=0.6)
    axes[0].set_xticks(x, months[:-1])
    axes[0].set_ylabel("clients in arrears with a positive bill")
    axes[0].set_xlabel("statement month (2005)")
    twin = axes[0].twinx()
    twin.plot(x, 100 * by_cutoff["paid"], color=COLORS["model"], lw=2, marker="o")
    twin.set_ylim(0, 100)
    twin.set_ylabel("paid something the next month (%)", color=COLORS["model"])
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    axes[0].set_title("Queue size and share that pays next month")
    one = (panel.status == 1).sum(axis=0)
    two = (panel.status == 2).sum(axis=0)
    more = (panel.status >= 3).sum(axis=0)
    y = np.arange(len(months))
    axes[1].bar(y, two, color=COLORS["model"], width=0.6, label="code 2")
    axes[1].bar(y, more, bottom=two, color=COLORS["balance"], width=0.6, label="code 3 or more")
    axes[1].bar(y, one, bottom=two + more, color=COLORS["oracle"], width=0.6, label="code 1")
    axes[1].set_xticks(y, months)
    axes[1].set_ylabel("clients")
    axes[1].set_xlabel("repayment status in (2005)")
    axes[1].set_title("Delay codes: code 1 appears only at the end")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(target / "uci_credit_card_1_queue_and_codes.png", bbox_inches="tight")
    plt.close(fig)

    # 2. capture curves on the holdout
    frame = with_reference_scores(result.holdout)
    value = frame["realized_value"].to_numpy(dtype=float)
    codes = pd.factorize(frame["snapshot_at"])[0]
    grid = np.linspace(0.01, 1.0, 100)
    curves = {
        name: [
            top_value_per_queue(value, frame[f"score_{name}"].to_numpy(dtype=float), codes, c)
            / value.sum()
            for c in grid
        ]
        for name in ("oracle", "ev_contact", "prob", "lambdarank", "last_payment",
                     "largest_balance", "random")
    }
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    style = {
        "oracle": (COLORS["oracle"], ":", 1.6), "ev_contact": (COLORS["model"], "-", 2.4),
        "prob": (COLORS["band"], "-", 1.8), "lambdarank": (COLORS["payment"], "--", 1.6),
        "last_payment": (COLORS["keep"], "-", 1.8), "largest_balance": (COLORS["balance"], "-", 2.2),
        "random": (COLORS["random"], ":", 1.2),
    }  # fmt: skip
    for name, ys in curves.items():
        color, ls, lw = style[name]
        ax.plot(100 * grid, 100 * np.array(ys), color=color, ls=ls, lw=lw, label=LABELS[name])
    ax.axvline(100 * HEADLINE, color=COLORS["grid"], lw=1.2)
    ax.set_xlabel("share of each queue worked (%)")
    ax.set_ylabel("share of the money paid next month that lands in it (%)")
    ax.set_title("Holdout: value captured under limited capacity")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(target / "uci_credit_card_2_capture_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 3. uplift over the balance rule, by capacity and by month
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    capacities = sorted(extended["capacity"].unique())
    shown = ("ev_contact", "prob", "lambdarank", "last_payment")
    width = 0.2
    for i, name in enumerate(shown):
        part = extended[extended["strategy"] == name].set_index("capacity").loc[capacities]
        pos = np.arange(len(capacities)) + (i - 1.5) * width
        low = part["uplift_vs_balance_pct"] - part["ci_low_pct"]
        high = part["ci_high_pct"] - part["uplift_vs_balance_pct"]
        axes[0].bar(
            pos, part["uplift_vs_balance_pct"], width, yerr=[low, high], capsize=2,
            color=style[name][0], label=LABELS[name], error_kw={"lw": 1},
        )
    axes[0].axhline(0, color=COLORS["oracle"], lw=1)
    axes[0].set_xticks(np.arange(len(capacities)), [f"{int(c * 100)}%" for c in capacities])
    axes[0].set_xlabel("share of each queue worked")
    axes[0].set_ylabel("uplift over the largest-balance rule (%)")
    axes[0].set_title("Holdout (two queues), 95% debtor bootstrap")
    axes[0].set_ylim(-80, 66)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    names = [pd.Timestamp(r["holdout_cutoff"]).strftime("%b %d") for r in rolling]
    for i, (key, label) in enumerate(
        (("ev_contact", LABELS["ev_contact"]), ("prob", LABELS["prob"]))
    ):
        point = [r["strategies"][key]["uplift_pct"] for r in rolling]
        low = [p - r["strategies"][key]["ci_low_pct"] for p, r in zip(point, rolling, strict=True)]
        high = [r["strategies"][key]["ci_high_pct"] - p for p, r in zip(point, rolling, strict=True)]
        pos = np.arange(len(rolling)) + (i - 0.5) * 0.35
        axes[1].bar(pos, point, 0.35, yerr=[low, high], capsize=2, color=style[key][0],
                    label=label, error_kw={"lw": 1})
    axes[1].axhline(0, color=COLORS["oracle"], lw=1)
    axes[1].set_xticks(np.arange(len(rolling)), names)
    axes[1].set_xlabel("queue judged (cutoff in 2005), model trained on every earlier cutoff")
    axes[1].set_ylabel("uplift at 20% capacity (%)")
    axes[1].set_title("Rolling origin: one queue at a time")
    axes[1].set_ylim(0, 46)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(target / "uci_credit_card_3_uplift_by_capacity_and_month.png", bbox_inches="tight")
    plt.close(fig)

    # 4. calibration of the close head and what it uses
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    holdout = result.holdout
    bins = pd.qcut(holdout["p_close"].rank(method="first"), 10, labels=False)
    reliability = holdout.groupby(bins).agg(predicted=("p_close", "mean"), observed=("closed", "mean"))
    axes[0].plot([0, 1], [0, 1], color=COLORS["random"], ls=":", lw=1.2)
    axes[0].plot(reliability["predicted"], reliability["observed"], color=COLORS["model"], lw=2,
                 marker="o")
    axes[0].set_xlabel("mean predicted P(close), ten equal groups")
    axes[0].set_ylabel("observed share that paid")
    axes[0].set_title("Close head on the holdout")
    importance = result.bundle.close.importance().head(10).iloc[::-1]
    extras = set(card.EXTRA_NUMERIC)
    axes[1].barh(
        importance["feature"], 100 * importance["gain_share"],
        color=[COLORS["model"] if f in extras else COLORS["balance"] for f in importance["feature"]],
    )
    axes[1].set_xlabel("share of split gain (%)")
    axes[1].set_title("Close head: blue = card features, orange = contract features")
    fig.tight_layout()
    fig.savefig(target / "uci_credit_card_4_calibration_and_importance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="folder that receives uci_credit_card/")
    parser.add_argument("--out", default="artifacts/uci_credit_card", help="folder for the results")
    parser.add_argument("--figures", help="folder for the four PNG figures (needs matplotlib)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    started = time.time()
    csv = card.fetch(Path(args.data) / "uci_credit_card")
    panel = card.read_clients(csv)
    settings = card.card_settings()
    data = card.build_datasets(panel, settings)
    out = Path(args.out)
    write_datasets(out, data.snapshots, data.break_df, data.as_of)
    (out / "settings.json").write_text(json.dumps(card.SETTINGS_OVERRIDES, indent=2), "utf-8")
    print("quality:", json.dumps(asdict(data.quality)))

    result = run_stages(data, settings, out)
    report = result.report
    primary = headline_row(report, result, settings)
    frame = with_reference_scores(result.holdout)
    extended = run_backtest(frame, settings.ranking.capacities, settings.evaluation).table
    extended.to_csv(out / "backtest_extended.csv", index=False)
    primary["versus_last_payment"] = {
        name: head_to_head(frame, name, "last_payment", HEADLINE, settings)
        for name in ("ev_contact", "prob", "lambdarank")
    }
    primary["selected_params"] = result.selected_params
    primary["importance_close"] = result.bundle.close.importance().head(12).to_dict("records")
    primary["importance_break"] = result.bundle.brk.importance().head(4).to_dict("records")
    primary["split"] = {
        "holdout_start": result.split.holdout_start.date().isoformat(),
        "purged_rows": result.split.purged,
    }
    print("primary done", round(time.time() - started), "s")

    rolling = [rolling_origin(data, settings, i) for i in (2, 3, 4)]
    print("rolling origin done", round(time.time() - started), "s")

    base_only = settings.with_overrides({"features": {"extra_numeric": []}})
    ablation = run_stages(data, base_only)
    ablation_row = headline_row(ablation.report, ablation, base_only)
    print("ablation done", round(time.time() - started), "s")

    groups = []
    for name, columns in FEATURE_GROUPS.items():
        only = settings.with_overrides(
            {"features": {"extra_numeric": list(columns)}, "evaluation": {"n_bootstrap": 0}}
        )
        run = run_stages(data, only)
        row = headline_row(run.report, run, only)
        groups.append({"group": name, "features": list(columns), "close_auc": row["close_auc"],
                       **{n: row["strategies"][n]["uplift_pct"]
                          for n in ("ev_contact", "prob", "lambdarank")}})
    horizons = horizon_check(panel, data, settings)
    print("feature groups and horizon check done", round(time.time() - started), "s")

    seeds = []
    for seed in (17, 18, 19, 20, 21):
        tuned = settings.with_overrides({"model": {"seed": seed}, "evaluation": {"n_bootstrap": 0}})
        run = run_stages(data, tuned)
        row = headline_row(run.report, run, tuned)
        seeds.append({"seed": seed, "close_auc": row["close_auc"],
                      **{n: row["strategies"][n]["uplift_pct"]
                         for n in ("ev_contact", "prob", "lambdarank")}})
    print("seeds done", round(time.time() - started), "s")

    strict_settings = settings.with_overrides({"split": {"cv_folds": 1}})
    strict_data = card.build_datasets(
        panel, strict_settings, min_status=2, min_months_in_arrears=2
    )
    strict = run_stages(strict_data, strict_settings)
    strict_row = headline_row(strict.report, strict, strict_settings)
    strict_row["quality"] = asdict(strict_data.quality)
    strict_row["base_rates"] = {
        "close": float(strict_data.snapshots["closed"].mean()),
        "break": float(strict_data.break_df["broke"].mean()),
    }
    print("persistent arrears done", round(time.time() - started), "s")

    summary = {
        "quality": asdict(data.quality),
        "base_rates": {
            "close": float(data.snapshots["closed"].mean()),
            "break": float(data.break_df["broke"].mean()),
            "mean_paid_next_month": float(data.snapshots["realized_value"].mean()),
            "median_paid_next_month": float(data.snapshots["realized_value"].median()),
            "paid_next_month_if_paid_last_month": float(
                data.snapshots.loc[data.snapshots["paid_last_month"] > 0, "closed"].mean()
            ),
            "paid_next_month_if_not_paid_last_month": float(
                data.snapshots.loc[data.snapshots["paid_last_month"] == 0, "closed"].mean()
            ),
            "share_paid_last_month": float((data.snapshots["paid_last_month"] > 0).mean()),
            "mean_balance": float(data.snapshots["balance"].mean()),
        },
        "status_transitions": status_transitions(panel),
        "primary": primary,
        "extended_backtest": extended.to_dict("records"),
        "rolling_origin": rolling,
        "ablation_base_features_only": ablation_row,
        "feature_groups_with_base": groups,
        "horizon_check": horizons,
        "seeds": seeds,
        "persistent_arrears": strict_row,
        "csv_sha256": card.DATA_SHA256,
        "seconds": round(time.time() - started, 1),
        "versions": {
            "python": platform.python_version(),
            **{n: version(n) for n in ("numpy", "pandas", "scikit-learn", "lightgbm")},
        },
    }
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    print(json.dumps(jsonable({k: summary[k] for k in ("base_rates", "horizon_check")}), indent=1))
    if args.figures:
        draw_figures(panel, data, result, extended, rolling, Path(args.figures))
        print("figures written to", args.figures)
    print("total", round(time.time() - started), "s; summary in", out / "summary.json")


if __name__ == "__main__":
    main()
