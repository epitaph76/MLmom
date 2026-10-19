import asyncio
import time

from fastapi.testclient import TestClient

from mlmom.contracts import ResearchJob
from mlmom.research_service import create_app


def wait_for_terminal(client: TestClient, job_id: str) -> ResearchJob:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = ResearchJob.model_validate(
            client.get(f"/v1/research/jobs/{job_id}").json()
        )
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Research job did not finish")


def test_service_runs_graph_and_preserves_contract():
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["mock"] is True
        response = client.post(
            "/v1/research/jobs",
            json={
                "request_id": "real-service-success",
                "task": "Find an imbalanced classification baseline",
                "task_type": "classification",
                "metric": "f1",
            },
        )
        assert response.status_code == 202
        initial = ResearchJob.model_validate(response.json())
        assert initial.status == "queued"

        final = wait_for_terminal(client, initial.job_id)
        assert final.status == "succeeded"
        assert final.mock is True
        assert len(final.result.cards) == 4
        assert final.result.cards[0].url.host == "www.kaggle.com"
        assert final.result.hypotheses[0].proposed_experiment["source_priority"] == [
            "kaggle",
            "github",
            "other",
        ]


def test_service_request_id_is_idempotent_and_conflicts_are_rejected():
    with TestClient(create_app()) as client:
        body = {"request_id": "same-request", "task": "ML task"}
        first = client.post("/v1/research/jobs", json=body)
        second = client.post("/v1/research/jobs", json=body)
        assert first.json()["job_id"] == second.json()["job_id"]

        conflict = client.post(
            "/v1/research/jobs", json={**body, "task": "Different task"}
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "REQUEST_ID_CONFLICT"
        assert conflict.json()["mock"] is True


class FailingGraph:
    async def ainvoke(self, state):
        raise RuntimeError("Sensitive internal failure")


def test_service_turns_graph_exceptions_into_safe_job_errors():
    with TestClient(create_app(graph=FailingGraph())) as client:
        initial = client.post(
            "/v1/research/jobs",
            json={"request_id": "failure", "task": "ML task"},
        ).json()
        final = wait_for_terminal(client, initial["job_id"])

        assert final.status == "failed"
        assert final.error.code == "RESEARCH_WORKFLOW_FAILED"
        assert "Sensitive" not in final.error.message
        assert final.error.mock is True


class SlowGraph:
    async def ainvoke(self, state):
        await asyncio.sleep(10)


def test_service_cancellation_is_terminal_and_idempotent():
    with TestClient(create_app(graph=SlowGraph())) as client:
        initial = client.post(
            "/v1/research/jobs",
            json={"request_id": "cancel", "task": "ML task"},
        ).json()
        url = f"/v1/research/jobs/{initial['job_id']}"
        cancelled = client.post(url + "/cancel").json()

        assert cancelled["status"] == "cancelled"
        assert client.post(url + "/cancel").json() == cancelled
        time.sleep(0.05)
        assert client.get(url).json() == cancelled
