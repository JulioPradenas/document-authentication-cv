"""Tests for src/models/registry.py (MLflow Model Registry wrapper)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlflow", reason="mlflow not installed")
pytest.importorskip("torch", reason="torch not installed")

from src.models.classifier import DocumentClassifier
from src.models.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def checkpoint(tmp_path):
    """A small untrained checkpoint saved to a temp path."""
    model = DocumentClassifier(pretrained=False)
    path = tmp_path / "model.pt"
    model.save(path, metadata={"note": "test"})
    return path


@pytest.fixture()
def registry(tmp_path):
    """Isolated registry backed by a per-test SQLite store."""
    db = tmp_path / "mlflow.db"
    return ModelRegistry(
        model_name="test-authenticator",
        tracking_uri=f"sqlite:///{db}",
    )


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_version_1(self, registry, checkpoint):
        version = registry.register(checkpoint, metrics={"val_f1": 0.9})
        assert version == 1

    def test_register_increments_version(self, registry, checkpoint):
        v1 = registry.register(checkpoint)
        v2 = registry.register(checkpoint)
        assert v2 == v1 + 1

    def test_register_missing_checkpoint_raises(self, registry, tmp_path):
        with pytest.raises(FileNotFoundError):
            registry.register(tmp_path / "does_not_exist.pt")

    def test_register_with_description(self, registry, checkpoint):
        version = registry.register(checkpoint, description="champion v1")
        versions = registry.list_versions()
        match = next(v for v in versions if v["version"] == version)
        assert match["description"] == "champion v1"

    def test_register_logs_metrics(self, registry, checkpoint):
        version = registry.register(checkpoint, metrics={"val_f1": 0.94, "val_auc": 0.97})
        versions = registry.list_versions()
        match = next(v for v in versions if v["version"] == version)
        assert match["metrics"]["val_f1"] == pytest.approx(0.94)
        assert match["metrics"]["val_auc"] == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# 2. Promotion / aliases
# ---------------------------------------------------------------------------


class TestPromote:
    def test_promote_then_resolve(self, registry, checkpoint):
        version = registry.register(checkpoint)
        registry.promote(version, alias="production")
        assert registry.get_version_by_alias("production") == version

    def test_promote_moves_alias(self, registry, checkpoint):
        v1 = registry.register(checkpoint)
        v2 = registry.register(checkpoint)
        registry.promote(v1, alias="production")
        registry.promote(v2, alias="production")  # reassign
        assert registry.get_version_by_alias("production") == v2

    def test_staging_and_production_coexist(self, registry, checkpoint):
        v1 = registry.register(checkpoint)
        v2 = registry.register(checkpoint)
        registry.promote(v1, alias="production")
        registry.promote(v2, alias="staging")
        assert registry.get_version_by_alias("production") == v1
        assert registry.get_version_by_alias("staging") == v2

    def test_demote_removes_alias(self, registry, checkpoint):
        version = registry.register(checkpoint)
        registry.promote(version, alias="staging")
        registry.demote("staging")
        with pytest.raises(Exception):
            registry.get_version_by_alias("staging")

    def test_aliases_in_list_versions(self, registry, checkpoint):
        version = registry.register(checkpoint)
        registry.promote(version, alias="production")
        versions = registry.list_versions()
        match = next(v for v in versions if v["version"] == version)
        assert "production" in match["aliases"]


# ---------------------------------------------------------------------------
# 3. Loading
# ---------------------------------------------------------------------------


class TestLoad:
    def test_download_checkpoint_by_alias(self, registry, checkpoint, tmp_path):
        version = registry.register(checkpoint)
        registry.promote(version, alias="production")
        local = registry.download_checkpoint(tmp_path / "dl", alias="production")
        assert local.exists()
        assert local.suffix == ".pt"

    def test_download_checkpoint_by_version(self, registry, checkpoint, tmp_path):
        version = registry.register(checkpoint)
        local = registry.download_checkpoint(tmp_path / "dl", version=version)
        assert local.exists()

    def test_download_requires_alias_or_version(self, registry, tmp_path):
        with pytest.raises(ValueError):
            registry.download_checkpoint(tmp_path / "dl", alias=None, version=None)

    def test_load_model_returns_classifier(self, registry, checkpoint, tmp_path):
        version = registry.register(checkpoint)
        registry.promote(version, alias="production")
        model = registry.load_model(alias="production", device="cpu", dest_dir=tmp_path / "dl")
        assert isinstance(model, DocumentClassifier)

    def test_loaded_model_matches_original_output(self, registry, checkpoint, tmp_path):
        import torch

        original = DocumentClassifier.load(checkpoint, device="cpu")
        original.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            expected = original(x)

        version = registry.register(checkpoint)
        registry.promote(version, alias="production")
        loaded = registry.load_model(alias="production", dest_dir=tmp_path / "dl")
        with torch.no_grad():
            actual = loaded(x)

        assert torch.allclose(expected, actual, atol=1e-5)

    def test_list_versions_empty_for_new_model(self, tmp_path):
        db = tmp_path / "mlflow.db"
        reg = ModelRegistry(model_name="empty-model", tracking_uri=f"sqlite:///{db}")
        with pytest.raises(Exception):
            # No registered model yet → get_registered_model raises
            reg.list_versions()
