"""Evaluation, in the units the business decides in.

AUC is reported because it is expected, but it does not answer the question the
call centre asks, which is "who do we call today, given we can make N calls".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import ECONOMICS


def ranking_metrics(actual: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(actual, scores)),
        "pr_auc": float(average_precision_score(actual, scores)),
        # Brier is included because a score that drives a value calculation has
        # to be a probability, not just a ranking.
        "brier": float(brier_score_loss(actual, scores)),
        "base_rate": float(actual.mean()),
    }


def decile_table(actual: np.ndarray, scores: np.ndarray, deciles: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual, "score": scores})
    frame["rank"] = frame["score"].rank(method="first", ascending=False)
    frame["decile"] = pd.qcut(frame["rank"], deciles, labels=range(1, deciles + 1))
    table = (
        frame.groupby("decile", observed=True)
        .agg(leads=("actual", "size"), conversions=("actual", "sum"), mean_score=("score", "mean"))
        .reset_index()
    )
    table["conversion_rate"] = table["conversions"] / table["leads"]
    table["lift"] = table["conversion_rate"] / frame["actual"].mean()
    table["cumulative_conversions"] = table["conversions"].cumsum()
    table["share_of_conversions"] = table["cumulative_conversions"] / frame["actual"].sum()
    return table


def expected_value_curve(actual: np.ndarray, scores: np.ndarray, steps: int = 100) -> pd.DataFrame:
    """Profit as a function of how deep into the ranked list the team calls.

    Each row answers: if we called the top X% of leads, what would we have
    earned. The maximum of this curve is the operating point, and it is
    determined by the cost of a call, not by a 0.5 probability cutoff.
    """
    order = np.argsort(-scores)
    ordered_actual = actual[order]
    conversions = np.cumsum(ordered_actual)
    calls = np.arange(1, len(actual) + 1)

    value = conversions * ECONOMICS.value_per_conversion - calls * ECONOMICS.cost_per_call
    indices = np.unique(np.linspace(0, len(actual) - 1, steps).astype(int))
    return pd.DataFrame(
        {
            "calls": calls[indices],
            "share_called": calls[indices] / len(actual),
            "conversions": conversions[indices],
            "expected_value": value[indices],
        }
    )


def optimal_threshold(actual: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """The score cutoff that maximises profit, and what it implies.

    Not 0.5. The break-even conversion probability is cost / value: below it, a
    call loses money in expectation however confident the model is.
    """
    order = np.argsort(-scores)
    ordered_actual = actual[order]
    ordered_scores = scores[order]
    conversions = np.cumsum(ordered_actual)
    calls = np.arange(1, len(actual) + 1)
    value = conversions * ECONOMICS.value_per_conversion - calls * ECONOMICS.cost_per_call

    best = int(np.argmax(value))
    break_even = ECONOMICS.cost_per_call / ECONOMICS.value_per_conversion
    return {
        "threshold": float(ordered_scores[best]),
        "calls": int(calls[best]),
        "share_called": float(calls[best] / len(actual)),
        "conversions": int(conversions[best]),
        "expected_value": float(value[best]),
        "break_even_probability": float(break_even),
        "value_calling_everyone": float(value[-1]),
    }


def capacity_value(actual: np.ndarray, scores: np.ndarray,
                   capacities: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)) -> pd.DataFrame:
    """What ranking is worth when capacity, not profitability, is the constraint.

    When the break-even probability sits far below the base rate, the model does
    not decide who to skip — every lead is worth calling in expectation. What it
    decides is who gets called FIRST with a fixed number of agents, which is the
    constraint a call centre actually operates under.
    """
    order = np.argsort(-scores)
    ordered = actual[order]
    total_conversions = ordered.sum()

    rows = []
    for share in capacities:
        calls = int(len(ordered) * share)
        captured = int(ordered[:calls].sum())
        rows.append(
            {
                "share_called": share,
                "calls": calls,
                "conversions_captured": captured,
                "share_of_conversions": captured / total_conversions,
                # Random dialling captures its own share of the list, by definition.
                "lift_vs_random_order": (captured / total_conversions) / share,
                "net_value": captured * ECONOMICS.value_per_conversion - calls * ECONOMICS.cost_per_call,
            }
        )
    return pd.DataFrame(rows)
