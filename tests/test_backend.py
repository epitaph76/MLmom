import pytest
from fastapi.testclient import TestClient

from mlmom.backend import app
from mlmom.contracts import ApiError


def test_backend_capabilities_and_empty_history():
    with TestClient(app) as client:
        assert client.get("/health").json()["mock"] is False
        caps = client.get("/v1/capabilities").json()
        assert caps["stage"] == "infrastructure"
        assert set(caps["unavailable"]) == {"profile_data", "train", "evaluate"}
        assert client.get("/v1/history").json() == {"experiments": []}


@pytest.mark.parametrize("path,body", [("profile", {"dataset_ref": "future.csv"}), ("train", {"dataset_ref": "future.csv", "recipe": {}}), ("evaluate", {"experiment_id": "future"})])
def test_unimplemented_operations(path, body):
    with TestClient(app) as client:
        response = client.post(f"/v1/{path}", json=body)
        assert response.status_code == 501
        assert ApiError.model_validate(response.json()).code == "NOT_IMPLEMENTED"
        assert client.post(f"/v1/{path}", json={}).status_code == 422
