"""Lead scoring service.

Scores a lead with what is known BEFORE anyone picks up the phone, and returns
a call/skip decision plus a priority band for queueing.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lead_scoring.config import ECONOMICS  # noqa: E402
from lead_scoring.features import align_categories  # noqa: E402
from lead_scoring.schema import (  # noqa: E402
    BatchRequest, BatchResponse, Health, Lead, ScoredLead,
)

MODEL_PATH = ROOT / "models" / "lead_scoring.joblib"
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the artifact once at startup, not per request."""
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        state["model"] = bundle["model"]
        state["reference"] = bundle["feature_frame"]
        # Per-lead decision theory: call when the expected gain clears the cost
        # of the call, i.e. p * value >= cost, i.e. p >= cost / value.
        #
        # The profit-maximising cutoff found on the test period is 0.0 ("call
        # everyone") because that period's base rate, 30.8%, sits far above the
        # 5% break-even. That is a correct answer about one period, not a
        # per-lead rule, so the service gates on break-even and lets the
        # profit-max cutoff stay a reporting artefact.
        state["threshold"] = ECONOMICS.cost_per_call / ECONOMICS.value_per_conversion
        state["profit_max_threshold"] = bundle["threshold"]
    yield
    state.clear()


app = FastAPI(
    title="Lead scoring",
    description="Scores marketing leads on conversion probability, before the call.",
    version="1.0.0",
    lifespan=lifespan,
)


def _score(leads: list[Lead]) -> list[float]:
    if "model" not in state:
        raise HTTPException(503, "model artifact not loaded; run `python -m lead_scoring.train`")
    frame = pd.DataFrame([lead.model_dump() for lead in leads])
    for column in state["reference"].columns:
        if column in frame.columns and str(state["reference"][column].dtype) == "category":
            frame[column] = frame[column].astype("category")
    aligned = align_categories(frame, state["reference"])
    return state["model"].predict_proba(aligned)[:, 1].tolist()


def _band(probability: float) -> str:
    """Queue band.

    Cut at multiples of the break-even probability rather than at round numbers,
    so the bands mean something in money.
    """
    break_even = ECONOMICS.cost_per_call / ECONOMICS.value_per_conversion
    if probability >= break_even * 4:
        return "high"
    if probability >= break_even * 2:
        return "medium"
    return "low"


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok",
        model_loaded="model" in state,
        threshold=state.get("threshold"),
    )


@app.post("/score", response_model=ScoredLead)
def score(lead: Lead) -> ScoredLead:
    probability = _score([lead])[0]
    return ScoredLead(
        probability=probability,
        call=probability >= state["threshold"],
        priority=_band(probability),
    )


@app.post("/score/batch", response_model=BatchResponse)
def score_batch(request: BatchRequest) -> BatchResponse:
    """Score many leads and return them in the order they were sent.

    Ranking is left to the caller: the dialler owns capacity, this service owns
    the probability.
    """
    probabilities = _score(request.leads)
    return BatchResponse(
        scored=[
            ScoredLead(probability=p, call=p >= state["threshold"], priority=_band(p))
            for p in probabilities
        ],
        threshold=state["threshold"],
    )
