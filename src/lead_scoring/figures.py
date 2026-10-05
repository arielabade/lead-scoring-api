"""Figures for the README, built from the committed report files.

Nothing here refits a model. Everything reads ``reports/``, so the charts and
the numbers quoted around them cannot drift apart: change the pipeline, the
reports change, the charts follow.

Run with ``python -m lead_scoring.figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brandviz as bv

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "assets" / "figures"


def leakage_ladder(ladder: pd.DataFrame) -> Path:
    """What the same model scores under five evaluation setups.

    Horizontal because the labels are sentences, and sorted best-looking
    first, which is the order a reader meets these numbers in the wild: the
    flattering one is the published one.
    """
    ladder = ladder.sort_values("test_auc", ascending=True).reset_index(drop=True)
    colors = [bv.COBALT if row.deployable else bv.SLATE for row in ladder.itertuples()]

    fig, ax = bv.panel(
        12.4, 5.4,
        title="The same model, scored five ways",
        subtitle="Only the highlighted rung can be deployed. The top rung is the number this dataset is usually published at.",
    )
    positions = np.arange(len(ladder), dtype=float)
    ax.set_ylim(-0.7, len(ladder) - 0.3)
    ax.set_xlim(0.5, 1.0)
    fig.canvas.draw()

    bv.bars(ax, positions, ladder["test_auc"], colors, horizontal=True, width=0.58,
            base=0.5, labels=ladder["test_auc"], label_fmt="{:.3f}", label_pad=0.012)
    ax.set_yticks(positions, ladder["setup"])
    ax.set_xlabel("Test ROC AUC")
    shipped = ladder[ladder["deployable"]].iloc[0]
    best = ladder.iloc[-1]
    bv.annotate(
        ax,
        f"Everything between these two bars is evaluation, not model:\n"
        f"{(best['test_auc'] - shipped['test_auc']) * 100:.1f} AUC points bought by a leak and a shuffle",
        xy=(shipped["test_auc"], float(shipped.name) + 0.42),
        xytext=(0.695, len(ladder) - 1.55), color=bv.IVORY,
    )
    bv.clean(ax, axis="x", spines=("top", "right", "bottom"))
    return bv.save(fig, FIGURES / "leakage_ladder.svg")


def capacity_curve(capacity: pd.DataFrame, curve: pd.DataFrame) -> Path:
    """What the ranking is worth to a team that cannot call everyone.

    The diagonal is the honest baseline: calling the list in a random order
    reaches exactly the share of conversions you have capacity for. The gap
    between the curve and the diagonal is the entire value of the model.
    """
    fig, ax = bv.panel(
        12.4, 5.6,
        title="The model does not decide who to skip, it decides who goes first",
        subtitle="Every lead clears the 5% break-even probability, so the ranking is the product, not the cutoff",
    )
    share = np.concatenate([[0.0], curve["share_called"].to_numpy(), [1.0]])
    captured = np.concatenate(
        [[0.0], (curve["conversions"] / curve["conversions"].max()).to_numpy(), [1.0]]
    )

    ax.plot([0, 1], [0, 1], color=bv.SLATE, linewidth=2.0,
            linestyle=(0, (5, 4)), zorder=2, label="Random call order")
    ax.fill_between(share, share, captured, color=bv.COBALT, alpha=0.14, zorder=1)
    ax.plot(share, captured, color=bv.COBALT, linewidth=2.4, zorder=3,
            label="Called in score order")

    focus = capacity[np.isclose(capacity["share_called"], 0.3)]
    if not focus.empty:
        row = focus.iloc[0]
        ax.plot([row["share_called"]], [row["share_of_conversions"]], "o", markersize=10,
                markerfacecolor=bv.COBALT, markeredgecolor=bv.CARBON,
                markeredgewidth=2.0, zorder=5)
        bv.annotate(
            ax,
            f"{row['share_called']:.0%} of call capacity\nreaches {row['share_of_conversions']:.0%} of conversions\n"
            f"({row['lift_vs_random_order']:.2f}x random order)",
            xy=(row["share_called"] + 0.012, row["share_of_conversions"]),
            xytext=(0.40, 0.33), color=bv.IVORY,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of the list called")
    ax.set_ylabel("Share of all conversions reached")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(loc="lower right")
    bv.clean(ax, axis="both", spines=("top", "right"))
    return bv.save(fig, FIGURES / "capacity_curve.svg")


def decile_lift(deciles: pd.DataFrame) -> Path:
    """Conversion rate by score decile, against the base rate.

    A monotonic staircase is what a usable ranking looks like. This one is not
    quite monotonic, and the chart says so rather than smoothing it.
    """
    fig, ax = bv.panel(
        12.4, 5.0,
        title="Conversion rate by score decile",
        subtitle="Ranking quality, decile by decile, on the held-out chronological test period",
    )
    positions = np.arange(len(deciles), dtype=float)
    base_rate = (deciles["conversions"].sum() / deciles["leads"].sum())
    top = deciles["conversion_rate"].max()

    ax.set_xlim(-0.6, len(deciles) - 0.4)
    ax.set_ylim(0, top * 1.28)
    fig.canvas.draw()
    colors = [bv.COBALT if decile <= 3 else bv.SLATE for decile in deciles["decile"]]
    bv.bars(ax, positions, deciles["conversion_rate"], colors,
            labels=[f"{value:.0%}" for value in deciles["conversion_rate"]])
    ax.set_xticks(positions, [str(d) for d in deciles["decile"]])
    ax.set_xlabel("Score decile (1 = highest scored)")
    ax.set_ylabel("Conversion rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    bv.reference_line(ax, base_rate, f"base rate {base_rate:.0%}", where=0.985)
    bv.clean(ax)
    return bv.save(fig, FIGURES / "decile_lift.svg")


def headline(ladder: pd.DataFrame, capacity: pd.DataFrame, metrics: dict):
    """The three numbers the README leads with."""
    shipped = ladder[ladder["deployable"]].iloc[0]
    published = ladder["test_auc"].max()
    focus = capacity[np.isclose(capacity["share_called"], 0.3)].iloc[0]
    fig, _ = bv.kpi_strip([
        (f"{shipped['test_auc']:.3f}",
         f"Test AUC, down from {published:.3f} once the post-call\nfeature and the random split are removed"),
        (f"{focus['lift_vs_random_order']:.2f}x",
         f"Lift on call ordering at {focus['share_called']:.0%} capacity,\nwhat survives honest evaluation"),
        (f"{focus['share_of_conversions']:.0%}",
         f"Of all conversions reached by\nthe first {focus['share_called']:.0%} of the list"),
    ])
    return bv.save(fig, FIGURES / "headline.svg")


def build_all() -> list[Path]:
    ladder = pd.read_csv(REPORTS / "leakage_ladder.csv")
    capacity = pd.read_csv(REPORTS / "capacity_value.csv")
    curve = pd.read_csv(REPORTS / "expected_value_curve.csv")
    deciles = pd.read_csv(REPORTS / "deciles.csv")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    return [
        headline(ladder, capacity, metrics),
        leakage_ladder(ladder),
        capacity_curve(capacity, curve),
        decile_lift(deciles),
    ]


if __name__ == "__main__":
    for path in build_all():
        print(path.relative_to(ROOT))
