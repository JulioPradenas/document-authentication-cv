"""Holdout evaluation for DocumentClassifier.

ModelEvaluator computes the full set of classification metrics on a DataLoader
and returns structured results ready for plotting and reporting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.models.classifier import DocumentClassifier


class ModelEvaluator:
    """Runs inference on a DataLoader and computes classification metrics.

    Args:
        model: Loaded DocumentClassifier in eval mode.
        device: 'cpu', 'cuda', or 'mps'.
    """

    def __init__(self, model: DocumentClassifier, device: str = "cpu") -> None:
        self.model  = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> dict:
        """Run inference on loader and return full metrics dict.

        Returns:
            {
                probs:      np.ndarray (N,) — raw forgery probabilities
                labels:     np.ndarray (N,) int — ground truth
                preds:      np.ndarray (N,) int — binary predictions at threshold
                accuracy:   float
                precision:  float
                recall:     float
                f1:         float
                auc_roc:    float
                auc_pr:     float
                threshold:  float
                confusion_matrix: np.ndarray (2, 2)
                roc_curve:  (fpr, tpr, thresholds)
                pr_curve:   (precision, recall, thresholds)
                per_class:  {authentic: {precision, recall, f1},
                             forged:    {precision, recall, f1}}
            }
        """
        probs, labels = self._collect_predictions(loader)
        preds = (probs >= threshold).astype(int)
        return self._compute_metrics(probs, labels, preds, threshold)

    def find_optimal_threshold(
        self,
        loader: DataLoader,
        metric: str = "f1",
    ) -> tuple[float, dict]:
        """Search for the threshold that maximises the given metric on loader.

        Args:
            metric: 'f1', 'precision', 'recall', or 'accuracy'.

        Returns:
            (best_threshold, metrics_at_best_threshold)
        """
        probs, labels = self._collect_predictions(loader)

        thresholds = np.linspace(0.1, 0.9, 81)
        best_th    = 0.5
        best_score = -1.0

        for th in thresholds:
            preds = (probs >= th).astype(int)
            if metric == "f1":
                score = float(f1_score(labels, preds, zero_division=0))
            elif metric == "precision":
                score = float(precision_score(labels, preds, zero_division=0))
            elif metric == "recall":
                score = float(recall_score(labels, preds, zero_division=0))
            else:
                score = float(accuracy_score(labels, preds))

            if score > best_score:
                best_score = score
                best_th    = float(th)

        best_preds   = (probs >= best_th).astype(int)
        best_metrics = self._compute_metrics(probs, labels, best_preds, best_th)
        return best_th, best_metrics

    def per_forgery_type_metrics(
        self,
        loader: DataLoader,
        forgery_types: list[str],
        threshold: float = 0.5,
    ) -> dict[str, dict]:
        """Compute metrics broken down by forgery type.

        Args:
            loader: DataLoader where each sample also returns a type index.
            forgery_types: List of type names matching type indices.

        Returns:
            {type_name: {precision, recall, f1, n_samples, n_correct}}
        """
        probs, labels = self._collect_predictions(loader)
        preds = (probs >= threshold).astype(int)

        # Infer type from ordering: first N//len(types) samples are type 0, etc.
        n = len(labels)
        k = len(forgery_types)
        type_indices = np.array([i * k // n for i in range(n)])

        results = {}
        for t_idx, t_name in enumerate(forgery_types):
            mask = type_indices == t_idx
            if mask.sum() == 0:
                continue
            t_labels = labels[mask]
            t_preds  = preds[mask]
            results[t_name] = {
                "precision":  float(precision_score(t_labels, t_preds, zero_division=0)),
                "recall":     float(recall_score(t_labels, t_preds, zero_division=0)),
                "f1":         float(f1_score(t_labels, t_preds, zero_division=0)),
                "n_samples":  int(mask.sum()),
                "n_correct":  int((t_labels == t_preds).sum()),
            }
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_predictions(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        all_probs  = []
        all_labels = []
        for images, labels in loader:
            images = images.to(self.device)
            probs  = self.model(images).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
        return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)

    def _compute_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        preds: np.ndarray,
        threshold: float,
    ) -> dict:
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        try:
            auc_roc = float(roc_auc_score(labels, probs))
            fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        except ValueError:
            auc_roc = 0.5
            fpr = tpr = roc_thresholds = np.array([0.0, 1.0])

        try:
            auc_pr = float(average_precision_score(labels, probs))
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(labels, probs)
        except ValueError:
            auc_pr = 0.0
            prec_curve = rec_curve = pr_thresholds = np.array([0.0, 1.0])

        return {
            "probs":            probs,
            "labels":           labels,
            "preds":            preds,
            "threshold":        threshold,
            "accuracy":         float(accuracy_score(labels, preds)),
            "precision":        float(precision_score(labels, preds, zero_division=0)),
            "recall":           float(recall_score(labels, preds, zero_division=0)),
            "f1":               float(f1_score(labels, preds, zero_division=0)),
            "auc_roc":          auc_roc,
            "auc_pr":           auc_pr,
            "confusion_matrix": cm,
            "roc_curve":        (fpr, tpr, roc_thresholds),
            "pr_curve":         (prec_curve, rec_curve, pr_thresholds),
            "per_class": {
                "authentic": {
                    "precision": float(precision_score(1-labels, 1-preds, zero_division=0)),
                    "recall":    float(recall_score(1-labels, 1-preds, zero_division=0)),
                    "f1":        float(f1_score(1-labels, 1-preds, zero_division=0)),
                },
                "forged": {
                    "precision": float(precision_score(labels, preds, zero_division=0)),
                    "recall":    float(recall_score(labels, preds, zero_division=0)),
                    "f1":        float(f1_score(labels, preds, zero_division=0)),
                },
            },
        }
