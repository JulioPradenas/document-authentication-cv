"""Tests for src/models/evaluator.py."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("sklearn", reason="scikit-learn not installed")

from src.models.classifier import DocumentClassifier
from src.models.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(n: int = 20, seed: int = 0) -> DataLoader:
    rng    = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, 224, 224, generator=rng)
    labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)


def make_evaluator(seed: int = 0) -> ModelEvaluator:
    model = DocumentClassifier(pretrained=False)
    model.eval()
    return ModelEvaluator(model, device="cpu")


# ---------------------------------------------------------------------------
# 1. evaluate() — output structure
# ---------------------------------------------------------------------------

class TestEvaluateOutput:
    def test_required_keys_present(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader)
        required = {"probs", "labels", "preds", "threshold", "accuracy",
                    "precision", "recall", "f1", "auc_roc", "auc_pr",
                    "confusion_matrix", "roc_curve", "pr_curve", "per_class"}
        assert required <= result.keys()

    def test_arrays_correct_length(self):
        n      = 20
        ev     = make_evaluator()
        loader = make_loader(n=n)
        result = ev.evaluate(loader)
        assert len(result["probs"])  == n
        assert len(result["labels"]) == n
        assert len(result["preds"])  == n

    def test_metrics_in_valid_range(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader)
        for key in ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"):
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    def test_confusion_matrix_shape(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader)
        assert result["confusion_matrix"].shape == (2, 2)

    def test_confusion_matrix_sum_equals_n(self):
        n      = 20
        ev     = make_evaluator()
        loader = make_loader(n=n)
        result = ev.evaluate(loader)
        assert result["confusion_matrix"].sum() == n

    def test_per_class_structure(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader)
        for cls in ("authentic", "forged"):
            assert {"precision", "recall", "f1"} <= result["per_class"][cls].keys()

    def test_roc_curve_starts_at_zero_ends_at_one(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader)
        fpr, tpr, _ = result["roc_curve"]
        assert fpr[0] == pytest.approx(0.0, abs=0.01)
        assert tpr[-1] == pytest.approx(1.0, abs=0.01)

    def test_threshold_stored(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader, threshold=0.7)
        assert result["threshold"] == pytest.approx(0.7)

    def test_preds_consistent_with_threshold(self):
        ev     = make_evaluator()
        loader = make_loader()
        result = ev.evaluate(loader, threshold=0.5)
        expected = (result["probs"] >= 0.5).astype(int)
        np.testing.assert_array_equal(result["preds"], expected)


# ---------------------------------------------------------------------------
# 2. find_optimal_threshold
# ---------------------------------------------------------------------------

class TestOptimalThreshold:
    def test_returns_threshold_in_range(self):
        ev     = make_evaluator()
        loader = make_loader()
        th, _  = ev.find_optimal_threshold(loader, metric="f1")
        assert 0.0 <= th <= 1.0

    def test_returns_metrics_dict(self):
        ev     = make_evaluator()
        loader = make_loader()
        _, metrics = ev.find_optimal_threshold(loader, metric="f1")
        assert "f1" in metrics and "auc_roc" in metrics

    @pytest.mark.parametrize("metric", ["f1", "precision", "recall", "accuracy"])
    def test_all_metric_modes(self, metric):
        ev     = make_evaluator()
        loader = make_loader()
        th, _  = ev.find_optimal_threshold(loader, metric=metric)
        assert 0.0 <= th <= 1.0


# ---------------------------------------------------------------------------
# 3. Perfect predictor — sanity check
# ---------------------------------------------------------------------------

def test_perfect_predictor_metrics():
    """A classifier that always predicts the correct class should have F1=1."""
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probs  = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    preds  = (probs >= 0.5).astype(int)

    ev = make_evaluator()
    result = ev._compute_metrics(probs, labels, preds, threshold=0.5)
    assert result["f1"]       == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["auc_roc"]  == pytest.approx(1.0)


def test_all_same_label_no_crash():
    """Edge case: all labels are 0 — should not raise, just return defaults."""
    n      = 8
    images = torch.rand(n, 3, 224, 224)
    labels = torch.zeros(n)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    ev     = make_evaluator()
    result = ev.evaluate(loader)   # should not raise
    assert "f1" in result
