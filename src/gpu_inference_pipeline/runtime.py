from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    outputs: list[list[float]]
    runtime: str
    latency_ms: float


class InferenceRuntime:
    """Small ONNX-ready runtime with a deterministic fallback for smoke tests."""

    def __init__(self, model_path: str | None = None, providers: list[str] | None = None) -> None:
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.providers = providers or _providers_from_env()
        self.session: Any | None = None
        self.input_name: str | None = None

        if self.model_path:
            self._load_onnx_session(self.model_path)

    @property
    def name(self) -> str:
        if self.session is not None:
            return "onnxruntime"
        return "numpy-fallback"

    def predict(self, inputs: list[list[float]]) -> InferenceResult:
        batch = np.asarray(inputs, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError("inputs must be a 2D array shaped [batch, features]")

        start = time.perf_counter()
        if self.session is not None and self.input_name is not None:
            outputs = self.session.run(None, {self.input_name: batch})[0]
        else:
            outputs = self._fallback_predict(batch)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return InferenceResult(
            outputs=np.asarray(outputs, dtype=np.float32).tolist(),
            runtime=self.name,
            latency_ms=latency_ms,
        )

    def _load_onnx_session(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "MODEL_PATH was provided, but onnxruntime is not installed. "
                "Install with `pip install -e .[onnx]` or `pip install -e .[gpu]`."
            ) from exc

        available = set(ort.get_available_providers())
        selected = [provider for provider in self.providers if provider in available]
        if not selected:
            selected = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=selected)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _fallback_predict(batch: np.ndarray) -> np.ndarray:
        weights = np.linspace(0.25, 1.0, batch.shape[1], dtype=np.float32)
        logits = batch @ weights
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - probabilities, probabilities], axis=1)


def _providers_from_env() -> list[str]:
    raw = os.getenv("ONNX_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
    return [item.strip() for item in raw.split(",") if item.strip()]
