"""Leakage rules are the whole point of this project, so they are tested."""

import pandas as pd

from lead_scoring.config import LEAKED_COLUMNS, MACRO_COLUMNS, NUMERIC
from lead_scoring.data import chronological_split, load


def test_duration_is_dropped_by_default():
    """`duration` is known only after the call. It must never reach the model."""
    assert "duration" not in load().columns


def test_duration_can_be_kept_only_explicitly():
    assert "duration" in load(drop_leakage=False).columns


def test_macro_columns_are_not_model_features():
    """They are identical for every lead on a given day and cannot rank leads."""
    for column in MACRO_COLUMNS:
        assert column not in NUMERIC


def test_target_is_binary_and_renamed():
    frame = load()
    assert "y" not in frame.columns
    assert set(frame["converted"].unique()) <= {0, 1}


def test_split_preserves_file_order():
    """The file is ordered by campaign date, so order is the time signal."""
    frame = load()
    train, validation, test = chronological_split(frame)
    assert len(train) + len(validation) + len(test) == len(frame)
    # Index ranges must not interleave.
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()


def test_split_periods_have_different_base_rates():
    """Documents the shift rather than hiding it.

    Conversion runs 4.8% in training and 30.8% in test: the campaign spans the
    2008 crisis. Any evaluation that reports a single base rate is lying.
    """
    train, _, test = chronological_split(load())
    assert test["converted"].mean() > train["converted"].mean() * 3
