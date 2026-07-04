FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "gpu_inference_pipeline.app:app", "--host", "0.0.0.0", "--port", "8000"]
