"""Model comparison and ablation study utilities.

ModelComparator evaluates multiple DocumentClassifierV2 checkpoints (or
instances) on a shared test DataLoader and logs every metric to MLflow.

Metrics collected per model:
  - accuracy, f1, auc_roc, precision, recall  (classification)
  - avg_inference_ms                           (latency, CPU, batch=1)
  - model_size_mb                              (checkpoint file size)
  - total_params                               (parameter count)

Usage:
    comparator = ModelComparator(test_loader)
    comparator.add_model("efficientnet_b0", model_a)
    comparator.add_model("resnet18", model_b)
    comparator.add_model("mobilenet_v3_small", model_c)
    df = comparator.run(mlflow_run_name="ablation_v1")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

import mlflow
from src.models.architectures import DocumentClassifierV2


class ModelComparator:
    """Evaluates a set of models on the same test DataLoader.

    Args:
        test_loader: DataLoader for the held-out test split.
        device: Torch device string ('cpu', 'cuda', 'mps').
        mlflow_tracking_uri: MLflow backend URI.
        n_warmup: Number of warmup forward passes before timing latency.
        n_latency_reps: Number of single-image passes for latency averaging.
    """

    def __init__(
        self,
        test_loader: DataLoader,
        device: str = "cpu",
        mlflow_tracking_uri: str = "sqlite:///mlflow.db",
        n_warmup: int = 5,
        n_latency_reps: int = 50,
    ) -> None:
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.n_warmup = n_warmup
        self.n_latency_reps = n_latency_reps
        self._entries: list[dict[str, Any]] = []

    def add_model(
        self,
        name: str,
        model: DocumentClassifierV2,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        """Register a model for comparison.

        Args:
            name: Display name (used as MLflow run tag and DataFrame row label).
            model: Loaded model instance (eval mode preferred; forced inside run()).
            checkpoint_path: Optional .pt file used to measure model_size_mb.
        """
        self._entries.append(
            {
                "name": name,
                "model": model,
                "checkpoint_path": Path(checkpoint_path) if checkpoint_path else None,
            }
        )

    def run(
        self,
        mlflow_run_name: str = "model_comparison",
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Evaluate all registered models and return a list of result dicts.

        Each dict has keys: name, accuracy, f1, auc_roc, precision, recall,
        avg_inference_ms, model_size_mb, total_params.

        All results are also logged as MLflow child runs under mlflow_run_name.
        """
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        results: list[dict[str, Any]] = []

        with mlflow.start_run(run_name=mlflow_run_name):
            for entry in self._entries:
                metrics = self._evaluate_one(entry, threshold)
                results.append(metrics)
                self._log_to_mlflow(metrics, parent_run=mlflow.active_run())

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_one(self, entry: dict[str, Any], threshold: float) -> dict[str, Any]:
        name: str = entry["name"]
        model: DocumentClassifierV2 = entry["model"]
        ckpt_path: Path | None = entry["checkpoint_path"]

        model.eval()
        model.to(self.device)

        all_probs, all_labels = self._collect_predictions(model)
        preds = (all_probs >= threshold).astype(int)

        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            auc = 0.5

        model_size_mb = (
            ckpt_path.stat().st_size / 1e6 if ckpt_path and ckpt_path.exists() else float("nan")
        )

        return {
            "name": name,
            "accuracy": float(accuracy_score(all_labels, preds)),
            "f1": float(f1_score(all_labels, preds, zero_division=0)),
            "auc_roc": auc,
            "precision": float(precision_score(all_labels, preds, zero_division=0)),
            "recall": float(recall_score(all_labels, preds, zero_division=0)),
            "avg_inference_ms": self._measure_latency(model),
            "model_size_mb": model_size_mb,
            "total_params": model.total_params(),
        }

    @torch.no_grad()
    def _collect_predictions(self, model: DocumentClassifierV2) -> tuple[np.ndarray, np.ndarray]:
        prob_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        for images, labels in self.test_loader:
            images = images.to(self.device)
            probs = model(images)
            prob_chunks.append(probs.cpu().numpy())
            label_chunks.append(labels.numpy())
        all_probs: np.ndarray = np.concatenate(prob_chunks)
        all_labels: np.ndarray = np.concatenate(label_chunks)
        return all_probs, all_labels

    def _measure_latency(self, model: DocumentClassifierV2) -> float:
        """Average inference time (ms) over n_latency_reps single-image passes on CPU."""
        cpu_model = model.to(torch.device("cpu"))
        cpu_model.eval()
        dummy = torch.zeros(1, 3, 224, 224)

        with torch.no_grad():
            for _ in range(self.n_warmup):
                cpu_model(dummy)

        times: list[float] = []
        with torch.no_grad():
            for _ in range(self.n_latency_reps):
                t0 = time.perf_counter()
                cpu_model(dummy)
                times.append((time.perf_counter() - t0) * 1e3)

        model.to(self.device)
        return float(np.mean(times))

    def _log_to_mlflow(self, metrics: dict[str, Any], parent_run: Any) -> None:
        with mlflow.start_run(
            run_name=metrics["name"],
            nested=True,
        ):
            mlflow.log_params(
                {
                    "backbone": metrics["name"],
                    "total_params": metrics["total_params"],
                }
            )
            mlflow.log_metrics(
                {
                    k: v
                    for k, v in metrics.items()
                    if k not in ("name", "total_params") and not np.isnan(float(v or 0))
                }
            )
