from __future__ import annotations

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from gpu_inference_pipeline.runtime import InferenceRuntime

app = FastAPI(title="GPU Inference Pipeline", version="0.1.0")
runtime = InferenceRuntime()

REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


class PredictRequest(BaseModel):
    inputs: list[list[float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    outputs: list[list[float]]
    runtime: str
    batch_size: int
    latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": runtime.name}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    REQUEST_COUNT.inc()
    try:
        result = runtime.predict(request.inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    REQUEST_LATENCY.observe(result.latency_ms / 1000.0)
    return PredictResponse(
        outputs=result.outputs,
        runtime=result.runtime,
        batch_size=len(request.inputs),
        latency_ms=result.latency_ms,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
