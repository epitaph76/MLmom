import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mlmom.contracts import ApiError, ResearchJob, ResearchRequest
from mlmom.research_mock import create_app


@pytest.fixture
def client():
    with TestClient(create_app(delay_ms=30)) as client:
        yield client


def wait(client, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/v1/research/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return ResearchJob.model_validate(job)
        time.sleep(0.01)
    pytest.fail("Mock job did not finish")


@pytest.mark.parametrize("scenario,status,cards", [("success", "succeeded", 1), ("empty", "succeeded", 0), ("failed", "failed", None)])
def test_job_lifecycle(client, scenario, status, cards):
    assert client.get("/health").json()["mock"] is True
    response = client.post("/v1/research/jobs", json={"request_id": "one", "task": "ML task"}, headers={"x-mock-scenario": scenario})
    assert response.status_code == 202
    initial = ResearchJob.model_validate(response.json())
    assert initial.status == "queued"
    final = wait(client, initial.job_id)
    assert final.status == status
    assert final.mock is True
    if cards is not None:
        assert len(final.result.cards) == cards
    else:
        assert final.error.code == "MOCK_FAILURE"
    assert client.post(f"/v1/research/jobs/{initial.job_id}/cancel").json() == final.model_dump(mode="json")


def test_idempotency_and_conflict(client):
    body = {"request_id": "one", "task": "ML task"}
    first = client.post("/v1/research/jobs", json=body).json()
    second = client.post("/v1/research/jobs", json=body).json()
    assert first["job_id"] == second["job_id"]
    assert client.post("/v1/research/jobs", json=body, headers={"x-mock-scenario": "failed"}).json()["job_id"] == first["job_id"]
    conflict = client.post("/v1/research/jobs", json={**body, "task": "different"})
    assert conflict.status_code == 409
    assert ApiError.model_validate(conflict.json()).code == "REQUEST_ID_CONFLICT"


def test_cancel_is_terminal_and_idempotent():
    with TestClient(create_app(delay_ms=100)) as client:
        job = client.post("/v1/research/jobs", json={"request_id": "cancel", "task": "task"}).json()
        url = f"/v1/research/jobs/{job['job_id']}"
        first = client.post(url + "/cancel").json()
        assert first["status"] == "cancelled"
        assert client.post(url + "/cancel").json() == first
        time.sleep(0.15)
        assert client.get(url).json() == first


def test_error_envelope(client):
    for method, url in [("get", "/v1/research/jobs/missing"), ("post", "/v1/research/jobs/missing/cancel")]:
        response = getattr(client, method)(url)
        assert response.status_code == 404
        assert ApiError.model_validate(response.json()).mock is True
    for body in [{}, {"request_id": 42, "task": "task"}, {"request_id": "bad", "task": ""}]:
        response = client.post("/v1/research/jobs", json=body)
        assert response.status_code == 422
        assert ApiError.model_validate(response.json()).code == "VALIDATION_ERROR"


def test_examples_match_contract_and_python():
    root = Path(__file__).resolve().parents[1] / "contracts"
    doc = json.loads((root / "research.openapi.json").read_text(encoding="utf-8"))
    schemas = json.loads(json.dumps(doc["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    for path in (root / "examples").glob("*.json"):
        model = ResearchRequest if path.stem == "request" else ApiError if path.stem == "error" else ResearchJob
        value = json.loads(path.read_text(encoding="utf-8"))
        model.model_validate(value)
        Draft202012Validator({"$defs": schemas, "$ref": f"#/$defs/{model.__name__}"}).validate(value)


def test_invalid_state_and_references_rejected():
    root = Path(__file__).resolve().parents[1] / "contracts/examples"
    value = json.loads((root / "succeeded.json").read_text())
    value["result"]["hypotheses"][0]["card_ids"] = ["missing"]
    with pytest.raises(ValidationError):
        ResearchJob.model_validate(value)
    with pytest.raises(ValidationError):
        ResearchJob(job_id="x", request_id="y", status="succeeded")
