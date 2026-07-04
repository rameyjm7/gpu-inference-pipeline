.PHONY: install test lint run benchmark docker-build

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

run:
	uvicorn gpu_inference_pipeline.app:app --reload

benchmark:
	python scripts/benchmark.py --url http://127.0.0.1:8000/predict

docker-build:
	docker build -t gpu-inference-pipeline .
