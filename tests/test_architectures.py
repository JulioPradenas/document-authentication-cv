"""Tests for src/models/architectures.py and src/models/comparator.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision", reason="torchvision not installed")

from src.models.architectures import DocumentClassifierV2, build_classifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_loader(n: int = 40, seed: int = 0) -> DataLoader:
    torch.manual_seed(seed)
    images = torch.randn(n, 3, 224, 224)
    labels = torch.randint(0, 2, (n,)).float()
    return DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)


BACKBONES = ["efficientnet_b0", "resnet18", "mobilenet_v3_small"]


# ---------------------------------------------------------------------------
# 1. DocumentClassifierV2
# ---------------------------------------------------------------------------


class TestDocumentClassifierV2:
    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_forward_output_shape(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4,)

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_output_in_zero_one(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_total_params_positive(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        assert model.total_params() > 0

    def test_efficientnet_has_fewer_params_than_resnet18(self):
        eff = build_classifier("efficientnet_b0", pretrained=False)
        res = build_classifier("resnet18", pretrained=False)
        assert eff.total_params() < res.total_params()

    def test_mobilenet_has_fewest_params(self):
        mob = build_classifier("mobilenet_v3_small", pretrained=False)
        eff = build_classifier("efficientnet_b0", pretrained=False)
        assert mob.total_params() < eff.total_params()

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_freeze_reduces_trainable(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        total = model.total_params()
        model.freeze_backbone()
        assert model.trainable_params() < total

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_unfreeze_all_restores(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        total = model.total_params()
        model.freeze_backbone()
        model.unfreeze_all()
        assert model.trainable_params() == total

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_predict_proba_no_grad(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            proba = model.predict_proba(x)
        assert proba.shape == (2,)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            DocumentClassifierV2(backbone="vgg16")  # type: ignore[arg-type]

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_save_and_load_roundtrip(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_out = model(x).item()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "model.pt"
            model.save(ckpt, metadata={"backbone": backbone})
            loaded = DocumentClassifierV2.load(ckpt)
            loaded.eval()
            with torch.no_grad():
                loaded_out = loaded(x).item()

        assert abs(original_out - loaded_out) < 1e-5

    @pytest.mark.parametrize("backbone", BACKBONES)
    def test_load_restores_backbone_name(self, backbone):
        model = build_classifier(backbone=backbone, pretrained=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "model.pt"
            model.save(ckpt)
            loaded = DocumentClassifierV2.load(ckpt)
        assert loaded.backbone_name == backbone


# ---------------------------------------------------------------------------
# 2. ModelComparator
# ---------------------------------------------------------------------------

from src.models.comparator import ModelComparator  # noqa: E402


class TestModelComparator:
    @pytest.fixture()
    def comparator(self, tmp_path):
        loader = make_loader(n=32)
        return ModelComparator(
            test_loader=loader,
            device="cpu",
            mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            n_warmup=1,
            n_latency_reps=3,
        )

    def test_run_returns_correct_number_of_results(self, comparator):
        for name in ["efficientnet_b0", "resnet18"]:
            comparator.add_model(name, build_classifier(name, pretrained=False))
        results = comparator.run(mlflow_run_name="test_run")
        assert len(results) == 2

    def test_result_has_required_keys(self, comparator):
        comparator.add_model("resnet18", build_classifier("resnet18", pretrained=False))
        results = comparator.run(mlflow_run_name="test_keys")
        r = results[0]
        for key in (
            "name",
            "accuracy",
            "f1",
            "auc_roc",
            "precision",
            "recall",
            "avg_inference_ms",
            "model_size_mb",
            "total_params",
        ):
            assert key in r, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, comparator):
        comparator.add_model(
            "mobilenet_v3_small", build_classifier("mobilenet_v3_small", pretrained=False)
        )
        results = comparator.run(mlflow_run_name="test_range")
        r = results[0]
        for metric in ("accuracy", "f1", "precision", "recall"):
            assert 0.0 <= r[metric] <= 1.0, f"{metric}={r[metric]} out of [0,1]"
        assert r["auc_roc"] >= 0.0
        assert r["avg_inference_ms"] > 0.0
        assert r["total_params"] > 0

    def test_names_preserved(self, comparator):
        for name in ["resnet18", "mobilenet_v3_small"]:
            comparator.add_model(name, build_classifier(name, pretrained=False))
        results = comparator.run(mlflow_run_name="test_names")
        names = [r["name"] for r in results]
        assert names == ["resnet18", "mobilenet_v3_small"]

    def test_model_size_nan_when_no_checkpoint(self, comparator):
        comparator.add_model("resnet18", build_classifier("resnet18", pretrained=False))
        results = comparator.run(mlflow_run_name="test_size_nan")
        assert np.isnan(results[0]["model_size_mb"])

    def test_model_size_populated_when_checkpoint_given(self, comparator, tmp_path):
        model = build_classifier("resnet18", pretrained=False)
        ckpt = tmp_path / "resnet18.pt"
        model.save(ckpt)
        comparator.add_model("resnet18", model, checkpoint_path=ckpt)
        results = comparator.run(mlflow_run_name="test_size_populated")
        assert results[0]["model_size_mb"] > 0.0
