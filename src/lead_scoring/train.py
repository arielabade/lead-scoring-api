"""Train, calibrate, evaluate and register the scoring model.

Tracked with MLflow so that a run can be compared against the one before it
rather than against memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .config import CATEGORICAL, NUMERIC, RANDOM_SEED
from .data import chronological_split, load
from .evaluate import (
    capacity_value, decile_table, expected_value_curve, optimal_threshold, ranking_metrics,
)
from .features import FEATURES, category_map, prepare

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def _logistic_baseline() -> object:
    """A regularised linear baseline.

    Present so that the gradient-boosted model has to earn its complexity. A
    lead-scoring model that cannot beat logistic regression should not be the
    thing carrying an API, a container and a CI pipeline.
    """
    encoder = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
            ("numeric", StandardScaler(), list(NUMERIC)),
        ]
    )
    return make_pipeline(encoder, LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))


def train(track: bool = True) -> dict:
    frame = load(drop_leakage=True)
    train_frame, validation_frame, test_frame = chronological_split(frame)

    x_train, y_train = prepare(train_frame), train_frame["converted"].to_numpy()
    x_validation, y_validation = prepare(validation_frame), validation_frame["converted"].to_numpy()
    x_test, y_test = prepare(test_frame), test_frame["converted"].to_numpy()

    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="average_precision",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    # Calibrate on the validation split, not on training: the scores drive a
    # value calculation, so they have to behave like probabilities, and a model
    # calibrated on data it was fitted on is calibrated to its own overconfidence.
    # FrozenEstimator keeps the booster fixed while the isotonic map is fitted.
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(x_validation, y_validation)

    baseline = _logistic_baseline().fit(train_frame[FEATURES], y_train)
    naive = DummyClassifier(strategy="prior").fit(x_train, y_train)

    scores = {
        "lightgbm_calibrated": calibrated.predict_proba(x_test)[:, 1],
        "lightgbm_raw": model.predict_proba(x_test)[:, 1],
        "logistic": baseline.predict_proba(test_frame[FEATURES])[:, 1],
        "prior": naive.predict_proba(x_test)[:, 1],
    }
    metrics = {name: ranking_metrics(y_test, value) for name, value in scores.items()}

    chosen = scores["lightgbm_calibrated"]
    decision = optimal_threshold(y_test, chosen)
    deciles = decile_table(y_test, chosen)
    curve = expected_value_curve(y_test, chosen)
    capacity = capacity_value(y_test, chosen)

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    deciles.to_csv(REPORTS / "deciles.csv", index=False)
    curve.to_csv(REPORTS / "expected_value_curve.csv", index=False)
    capacity.to_csv(REPORTS / "capacity_value.csv", index=False)

    summary = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "metrics": metrics,
        "decision": decision,
        "capacity": capacity.to_dict(orient="records"),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))

    import joblib

    joblib.dump(
        {
            "model": calibrated,
            "categories": category_map(x_train),
            "columns": list(x_train.columns),
            "threshold": decision["threshold"],
        },
        MODELS / "lead_scoring.joblib",
    )

    if track:
        # SQLite rather than the ./mlruns file store: MLflow put the
        # filesystem backend into maintenance mode, and a local database is
        # the smallest thing that still supports the model registry.
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("lead-scoring")
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": "LGBMClassifier + isotonic calibration",
                    "leakage_columns_dropped": "duration",
                    "split": "chronological",
                    "best_iteration": summary["best_iteration"],
                }
            )
            for name, values in metrics.items():
                mlflow.log_metrics({f"{name}_{key}": value for key, value in values.items()})
            mlflow.log_metrics(
                {f"decision_{k}": v for k, v in decision.items() if isinstance(v, (int, float))}
            )
            mlflow.log_artifact(str(REPORTS / "metrics.json"))
            mlflow.lightgbm.log_model(model, name="lightgbm")

    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
