"""MLflow Model Registry wrapper for document-authentication checkpoints.

Provides a thin, typed API over MLflow's registry so the champion model can be
versioned and promoted through deployment stages, and loaded by stage at
serving time instead of from a hardcoded checkpoint path.

Stages are modeled with MLflow **aliases** (`staging`, `production`) — the
modern replacement for the deprecated stage transitions, available since
MLflow 2.3. A version can hold multiple aliases; promoting moves the alias.

Usage:
    registry = ModelRegistry(model_name="document-authenticator")
    version = registry.register(
        "models/saved/efficientnet_b0_best.pt",
        metrics={"val_f1": 0.94, "val_auc": 0.97},
        description="EfficientNet-B0, two-phase fine-tune",
    )
    registry.promote(version, alias="production")
    model = registry.load_model(alias="production")   # DocumentClassifier
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.models.classifier import DocumentClassifier

DEFAULT_MODEL_NAME = "document-authenticator"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class ModelRegistry:
    """Register, promote and load document-authentication models via MLflow.

    Args:
        model_name: Registered model name in the MLflow registry.
        tracking_uri: MLflow backend store URI.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        tracking_uri: str = DEFAULT_TRACKING_URI,
    ) -> None:
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    # ------------------------------------------------------------------
    # Registration & promotion
    # ------------------------------------------------------------------

    def register(
        self,
        checkpoint_path: Path | str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        description: str | None = None,
        run_name: str = "register_model",
    ) -> int:
        """Log a checkpoint as a run artifact and register a new model version.

        Args:
            checkpoint_path: Path to the .pt checkpoint to register.
            metrics: Optional metrics to log on the backing run (e.g. val_f1).
            params: Optional params to log on the backing run.
            description: Optional human-readable description for the version.
            run_name: Name for the backing MLflow run.

        Returns:
            The new integer model version number.
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        with mlflow.start_run(run_name=run_name) as run:
            if metrics:
                mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            mlflow.log_artifact(str(ckpt), artifact_path="model")
            run_id = run.info.run_id
            source = mlflow.get_artifact_uri("model")

        # Ensure the registered model exists, then create a version pointing at
        # the logged artifact. create_model_version (vs register_model) keeps
        # working with plain .pt artifacts under MLflow 3.x.
        try:
            self.client.create_registered_model(self.model_name)
        except MlflowException:
            pass  # already exists

        mv = self.client.create_model_version(
            name=self.model_name,
            source=source,
            run_id=run_id,
            description=description,
        )
        return int(mv.version)

    def promote(self, version: int, alias: str = "production") -> None:
        """Assign a deployment alias ('staging' or 'production') to a version.

        Moving an alias that already points elsewhere reassigns it atomically.
        """
        self.client.set_registered_model_alias(
            name=self.model_name, alias=alias, version=str(version)
        )

    def demote(self, alias: str) -> None:
        """Remove a deployment alias (e.g. retiring a production model)."""
        self.client.delete_registered_model_alias(name=self.model_name, alias=alias)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_version_by_alias(self, alias: str) -> int:
        """Resolve a deployment alias to its model version number."""
        mv = self.client.get_model_version_by_alias(self.model_name, alias)
        return int(mv.version)

    def list_versions(self) -> list[dict[str, Any]]:
        """List all registered versions with their aliases and metrics.

        Returns:
            List of dicts: {version, aliases, run_id, description, metrics}.
        """
        registered = self.client.get_registered_model(self.model_name)
        alias_map: dict[str, list[str]] = {}
        for alias, ver in registered.aliases.items():
            alias_map.setdefault(str(ver), []).append(alias)

        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        out: list[dict[str, Any]] = []
        for mv in sorted(versions, key=lambda m: int(m.version)):
            run_metrics: dict[str, float] = {}
            if mv.run_id:
                run = self.client.get_run(mv.run_id)
                run_metrics = dict(run.data.metrics)
            out.append(
                {
                    "version": int(mv.version),
                    "aliases": alias_map.get(str(mv.version), []),
                    "run_id": mv.run_id,
                    "description": mv.description or "",
                    "metrics": run_metrics,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def download_checkpoint(
        self,
        dest_dir: Path | str,
        alias: str | None = "production",
        version: int | None = None,
    ) -> Path:
        """Download the .pt checkpoint for an alias or explicit version.

        Args:
            dest_dir: Local directory to download artifacts into.
            alias: Deployment alias to resolve (ignored if version is given).
            version: Explicit version number (takes precedence over alias).

        Returns:
            Path to the downloaded .pt file.
        """
        if version is not None:
            uri = f"models:/{self.model_name}/{version}"
        elif alias is not None:
            uri = f"models:/{self.model_name}@{alias}"
        else:
            raise ValueError("Provide either alias or version")

        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dest_dir))
        checkpoints = list(Path(local_dir).rglob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .pt checkpoint found in registered model {uri}")
        return checkpoints[0]

    def load_model(
        self,
        alias: str | None = "production",
        version: int | None = None,
        device: str = "cpu",
        dest_dir: Path | str | None = None,
    ) -> DocumentClassifier:
        """Download and load a registered model as a DocumentClassifier.

        Args:
            alias: Deployment alias to load (default 'production').
            version: Explicit version (takes precedence over alias).
            device: Torch device for the loaded model.
            dest_dir: Optional download directory (defaults to a temp dir).

        Returns:
            A loaded DocumentClassifier in eval mode.
        """
        import tempfile

        target = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp())
        ckpt = self.download_checkpoint(target, alias=alias, version=version)
        return DocumentClassifier.load(ckpt, device=device)
