# GPU Inference Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![ONNX](https://img.shields.io/badge/Runtime-ONNX%20Runtime-orange)
![Docker](https://img.shields.io/badge/Deploy-Docker-blueviolet)
![Metrics](https://img.shields.io/badge/Metrics-Prometheus-lightgrey)

Production-style AI inference service for model execution, batching, streaming-shaped inputs, latency benchmarking, observability, and containerized deployment.

The repo is intentionally small and reviewable: it implements the service layer around a model rather than only a notebook or training script. It can run with a deterministic NumPy fallback for smoke tests, or load an ONNX model through ONNX Runtime when `MODEL_PATH` is provided.

## Why This Exists

Production inference systems depend on the path from model artifact to reliable service:

- request validation and preprocessing
- runtime provider selection
- batch inference
- latency and throughput measurement
- metrics and health endpoints
- containerized deployment
- CI checks that keep the service runnable

## Architecture

```text
client / benchmark
  -> FastAPI service
    -> request validation
    -> inference runtime
      -> ONNX Runtime when MODEL_PATH is set
      -> deterministic NumPy fallback otherwise
    -> metrics and structured response
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn gpu_inference_pipeline.app:app --reload
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/metrics`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs":[[0.1,0.2,0.3,0.4],[0.9,0.1,0.0,0.5]]}'
```

## ONNX Runtime Mode

```bash
export MODEL_PATH=/path/to/model.onnx
export ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
uvicorn gpu_inference_pipeline.app:app
```

If CUDA providers are unavailable, ONNX Runtime will fall back to available providers depending on the installed package and runtime environment.

## Benchmark

```bash
python scripts/benchmark.py --url http://127.0.0.1:8000/predict --requests 200 --batch-size 16 --features 4
```

The script reports throughput and latency percentiles.

## Docker

```bash
docker build -t gpu-inference-pipeline .
docker run --rm -p 8000:8000 gpu-inference-pipeline
```

## Roadmap

- Add a sample exported ONNX model and model-card metadata.
- Add dynamic batching queue for concurrent clients.
- Add streaming input examples for time-series or RF/IQ windows.
- Add TensorRT execution notes for GPU deployment.
- Add OpenTelemetry traces and structured JSON logging.
