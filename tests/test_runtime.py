from fastapi.testclient import TestClient

from gpu_inference_pipeline.app import app
from gpu_inference_pipeline.runtime import InferenceRuntime


def test_runtime_fallback_predicts_batch() -> None:
    runtime = InferenceRuntime()
    result = runtime.predict([[0.1, 0.2, 0.3, 0.4], [0.9, 0.1, 0.0, 0.5]])

    assert result.runtime == "numpy-fallback"
    assert len(result.outputs) == 2
    assert len(result.outputs[0]) == 2
    assert result.latency_ms >= 0.0


def test_api_predict() -> None:
    client = TestClient(app)
    response = client.post("/predict", json={"inputs": [[0.1, 0.2, 0.3, 0.4]]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime"] == "numpy-fallback"
    assert payload["batch_size"] == 1
    assert len(payload["outputs"][0]) == 2


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
