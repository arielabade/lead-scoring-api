import numpy as np

from lead_scoring.config import ECONOMICS
from lead_scoring.evaluate import capacity_value, decile_table, optimal_threshold, ranking_metrics


def _perfect(n=1000):
    rng = np.random.default_rng(0)
    actual = rng.binomial(1, 0.3, n)
    return actual, actual.astype(float)  # a model that knows the answer


def test_perfect_model_scores_auc_one():
    actual, scores = _perfect()
    assert ranking_metrics(actual, scores)["roc_auc"] == 1.0


def test_break_even_is_cost_over_value():
    actual, scores = _perfect()
    expected = ECONOMICS.cost_per_call / ECONOMICS.value_per_conversion
    assert optimal_threshold(actual, scores)["break_even_probability"] == expected


def test_capacity_lift_is_one_for_a_random_scorer():
    rng = np.random.default_rng(1)
    actual = rng.binomial(1, 0.3, 5000)
    noise = rng.random(5000)
    table = capacity_value(actual, noise)
    # Random ordering captures roughly its own share of conversions.
    assert abs(table["lift_vs_random_order"].mean() - 1.0) < 0.1


def test_capacity_lift_beats_one_for_a_perfect_scorer():
    actual, scores = _perfect(5000)
    table = capacity_value(actual, scores)
    assert (table["lift_vs_random_order"] > 1.5).all()


def test_deciles_cover_every_lead():
    actual, scores = _perfect()
    table = decile_table(actual, scores)
    assert table["leads"].sum() == len(actual)
    assert abs(table["share_of_conversions"].iloc[-1] - 1.0) < 1e-9
