"""Tests for src/models/classifier.py and src/models/trainer.py.

Uses a tiny dummy model (no pretrained weights) and synthetic data — no dataset
or GPU required. CI-safe.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from src.models.classifier import DocumentClassifier
from src.models.trainer import EarlyStopping, Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model(pretrained: bool = False) -> DocumentClassifier:
    return DocumentClassifier(pretrained=pretrained)


def make_batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.rand(n, 3, 224, 224)
    labels = torch.randint(0, 2, (n,)).float()
    return images, labels


def make_tiny_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    images = torch.rand(n, 3, 224, 224)
    labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# 1. DocumentClassifier — architecture and forward pass
# ---------------------------------------------------------------------------


class TestDocumentClassifier:
    def test_forward_output_shape(self):
        model = make_model()
        images, _ = make_batch(4)
        with torch.no_grad():
            out = model(images)
        assert out.shape == (4,)

    def test_output_in_zero_one(self):
        model = make_model()
        images, _ = make_batch(8)
        with torch.no_grad():
            out = model(images)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_output_dtype_float32(self):
        model = make_model()
        images, _ = make_batch(2)
        with torch.no_grad():
            out = model(images)
        assert out.dtype == torch.float32

    def test_single_image_forward(self):
        model = make_model()
        img = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            out = model(img)
        assert out.shape == (1,)

    def test_total_params_positive(self):
        model = make_model()
        assert model.total_params() > 0

    def test_predict_proba_no_grad(self):
        model = make_model()
        images, _ = make_batch(4)
        probs = model.predict_proba(images)
        assert probs.shape == (4,)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0


# ---------------------------------------------------------------------------
# 2. freeze / unfreeze
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    def test_freeze_backbone_reduces_trainable(self):
        model = make_model()
        model.unfreeze_all()
        total = model.trainable_params()
        model.freeze_backbone()
        frozen = model.trainable_params()
        assert frozen < total

    def test_freeze_backbone_head_still_trainable(self):
        model = make_model()
        model.freeze_backbone()
        # Classifier parameters should require grad
        for param in model.backbone.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_last_n_increases_trainable(self):
        model = make_model()
        model.freeze_backbone()
        frozen = model.trainable_params()
        model.unfreeze_last_n_layers(2)
        unfrozen = model.trainable_params()
        assert unfrozen > frozen

    def test_unfreeze_all_restores_full_params(self):
        model = make_model()
        model.freeze_backbone()
        model.unfreeze_all()
        for param in model.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# 3. Checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_creates_file(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_load_restores_weights(self):
        model = make_model()
        model.eval()  # disable dropout for deterministic comparison
        images, _ = make_batch(2)
        with torch.no_grad():
            original_out = model(images).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path, metadata={"val_f1": 0.95})
            loaded = DocumentClassifier.load(path)
            with torch.no_grad():
                loaded_out = loaded(images).numpy()

        np.testing.assert_allclose(original_out, loaded_out, atol=1e-6)

    def test_load_model_is_eval_mode(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)
            loaded = DocumentClassifier.load(path)
        assert not loaded.training

    def test_save_creates_parent_dirs(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "model.pt"
            model.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# 4. EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_no_stop_when_improving(self):
        stopper = EarlyStopping(patience=3)
        for score in [0.5, 0.6, 0.7, 0.8]:
            stopped = stopper.step(score)
        assert not stopped

    def test_stops_after_patience(self):
        stopper = EarlyStopping(patience=3)
        stopper.step(0.8)
        for _ in range(3):
            stopped = stopper.step(0.79)
        assert stopped

    def test_resets_wait_on_improvement(self):
        stopper = EarlyStopping(patience=2)
        stopper.step(0.8)
        stopper.step(0.79)  # wait=1
        stopper.step(0.81)  # improvement → wait=0
        stopped = stopper.step(0.80)  # wait=1
        assert not stopped


# ---------------------------------------------------------------------------
# 5. Trainer — one-epoch smoke test (tiny data, no MLflow remote)
# ---------------------------------------------------------------------------


class TestTrainer:
    def _make_trainer(self, tmpdir: Path) -> Trainer:
        model = make_model()
        loader = make_tiny_loader(n=8, batch_size=4)
        cfg = TrainerConfig(
            phase_a_epochs=1,
            phase_b_epochs=1,
            early_stopping_patience=2,
            checkpoint_dir=tmpdir,
            checkpoint_name="test_model.pt",
            mlflow_tracking_uri=f"sqlite:///{tmpdir}/mlflow.db",
            mlflow_run_name="test_run",
            device="cpu",
        )
        return Trainer(model, loader, loader, cfg)

    def test_run_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(Path(tmpdir))
            ckpt_path = trainer.run()
            assert ckpt_path.exists()

    def test_evaluate_returns_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(Path(tmpdir))
            loader = make_tiny_loader()
            metrics = trainer._evaluate(loader)
        assert {"loss", "f1", "auc", "accuracy"} <= metrics.keys()

    def test_evaluate_metrics_in_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._make_trainer(Path(tmpdir))
            loader = make_tiny_loader()
            metrics = trainer._evaluate(loader)
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] >= 0.0
