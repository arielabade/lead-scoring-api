"""Contract tests against the running app."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_LEAD = {
    "age": 30, "job": "student", "marital": "single", "education": "university.degree",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "mar", "day_of_week": "thu", "campaign": 1, "pdays": 3,
    "previous": 2, "poutcome": "success",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_scoring_returns_a_probability_and_a_decision(client):
    body = client.post("/score", json=VALID_LEAD).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["call"], bool)
    assert body["priority"] in {"high", "medium", "low"}


def test_duration_is_rejected_because_it_does_not_exist_at_scoring_time(client):
    """The field that leaks the outcome must not be accepted even if sent."""
    leaking = dict(VALID_LEAD, duration=600)
    response = client.post("/score", json=leaking)
    # Pydantic ignores unknown fields by default; what matters is that the
    # score is unchanged, i.e. duration had no influence.
    assert response.status_code == 200
    assert response.json()["probability"] == client.post("/score", json=VALID_LEAD).json()["probability"]


def test_unknown_category_is_rejected_at_the_edge(client):
    invalid = dict(VALID_LEAD, job="astronaut")
    assert client.post("/score", json=invalid).status_code == 422


def test_batch_preserves_input_order(client):
    leads = [dict(VALID_LEAD, age=age) for age in (25, 45, 65)]
    body = client.post("/score/batch", json={"leads": leads}).json()
    assert len(body["scored"]) == 3
    singles = [client.post("/score", json=lead).json()["probability"] for lead in leads]
    assert [row["probability"] for row in body["scored"]] == singles


def test_batch_rejects_an_empty_request(client):
    assert client.post("/score/batch", json={"leads": []}).status_code == 422


def test_a_month_the_training_window_never_saw_still_scores(client):
    """March, April, September and December fall entirely after the cutoff.

    They are valid business inputs that the model has no history for. The
    service must degrade to the lead's other features, not return a 500.
    """
    march_lead = dict(VALID_LEAD, month="mar")
    response = client.post("/score", json=march_lead)
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0
