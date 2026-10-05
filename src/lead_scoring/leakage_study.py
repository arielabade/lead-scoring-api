"""The leakage ladder: what this dataset's AUC is worth under each setup.

The README's central claim is that the number this dataset is usually reported
at describes an evaluation, not a model. That claim is only worth making if it
is reproducible, so it is computed here rather than quoted.

Five configurations, each fitting the same LightGBM with the same
hyperparameters, varying only two things: which features are allowed, and how
the data is split.

======================  ==================================================
`duration`              Seconds the call lasted. Known only once the call is
                        over, so a model that uses it answers "did this call
                        go well", not "should we make this call". It is the
                        single largest source of optimism on this dataset.
macro columns           euribor3m and friends: identical for every lead on a
                        given day, so useless for ranking leads against each
                        other, and under a time split a date in disguise.
random vs time split    The file is ordered by campaign date and spans the
                        2008 crisis. A random split lets training see the
                        economy of the test period.
======================  ==================================================

Run with ``python -m lead_scoring.leakage_study``. It writes
``reports/leakage_ladder.csv``, which is what the README figure reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL, MACRO_COLUMNS, NUMERIC, RANDOM_SEED
from .data import chronological_split, load

REPORTS = Path(__file__).resolve().parents[2] / "reports"

#: Each rung: a label, the extra columns allowed in, and the split to use.
LADDER = (
    ("Random split, call duration kept", ("duration",) + MACRO_COLUMNS, "random"),
    ("Random split, macro features kept", MACRO_COLUMNS, "random"),
    ("Random split, deployable features", (), "random"),
    ("Time split, macro features kept", MACRO_COLUMNS, "chronological"),
    ("Time split, deployable features", (), "chronological"),
)

#: The rung the service actually ships, named so the figure can mark it.
SHIPPED = "Time split, deployable features"


def _prepare(frame: pd.DataFrame, extra: tuple[str, ...]) -> pd.DataFrame:
    columns = list(CATEGORICAL) + list(NUMERIC) + [c for c in extra if c in frame.columns]
    prepared = frame[columns].copy()
    for column in CATEGORICAL:
        prepared[column] = prepared[column].astype("category")
    return prepared


def _split(frame: pd.DataFrame, how: str):
    if how == "chronological":
        return chronological_split(frame)
    # A random split of the same proportions, so only the ordering differs.
    rest, test = train_test_split(frame, test_size=0.2, random_state=RANDOM_SEED,
                                  shuffle=True, stratify=frame["converted"])
    train, validation = train_test_split(rest, test_size=0.2, random_state=RANDOM_SEED,
                                         shuffle=True, stratify=rest["converted"])
    return train, validation, test


def _fit_and_score(frame: pd.DataFrame, extra: tuple[str, ...], how: str) -> float:
    train, validation, test = _split(frame, how)
    x_train, x_validation, x_test = (_prepare(part, extra) for part in (train, validation, test))

    model = LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(
        x_train, train["converted"].to_numpy(),
        eval_set=[(x_validation, validation["converted"].to_numpy())],
        # Same early-stopping metric as train.py, so a rung of this ladder and
        # the shipped model are the same procedure with different inputs.
        eval_metric="average_precision",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    return float(roc_auc_score(test["converted"], model.predict_proba(x_test)[:, 1]))


def run() -> pd.DataFrame:
    """Fit every rung and return the ladder, best-looking first."""
    # Loaded with everything still attached; each rung selects what it may use.
    frame = load(drop_leakage=False, drop_macro=False)

    rows = []
    for label, extra, how in LADDER:
        auc = _fit_and_score(frame, extra, how)
        rows.append({
            "setup": label,
            "split": how,
            "uses_duration": "duration" in extra,
            "uses_macro": bool(set(extra) & set(MACRO_COLUMNS)),
            "test_auc": round(auc, 4),
            "deployable": label == SHIPPED,
        })
        print(f"{label:38s} AUC {auc:.3f}")

    ladder = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    ladder.to_csv(REPORTS / "leakage_ladder.csv", index=False)
    return ladder


if __name__ == "__main__":
    run()
    print(f"\nwrote {REPORTS / 'leakage_ladder.csv'}")
