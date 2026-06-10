"""Two-phase training loop for DocumentClassifier.

Phase A — Feature extraction (freeze backbone, train head only).
Phase B — Full fine-tuning (unfreeze last 2 backbone layers + head, lower LR).

Early stopping monitors validation F1. Best checkpoint saved to models/saved/.
All metrics logged to MLflow.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import mlflow
from src.models.classifier import DocumentClassifier


@dataclass
class TrainerConfig:
    # Phase A
    phase_a_epochs: int = 5
    phase_a_lr: float = 1e-3

    # Phase B
    phase_b_epochs: int = 15
    phase_b_lr: float = 1e-4
    phase_b_unfreeze_n: int = 2

    # Common
    early_stopping_patience: int = 5
    checkpoint_dir: Path = Path("models/saved")
    checkpoint_name: str = "efficientnet_b0_best.pt"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_run_name: str = "efficientnet_b0_finetune"
    device: str = "cpu"
    pos_weight: float | None = None  # BCEWithLogitsLoss weight for class imbalance


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.wait = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: DocumentClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or TrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.criterion = nn.BCELoss()  # model already has Sigmoid

        self._best_val_f1 = -1.0
        self._best_ckpt_path = Path(self.cfg.checkpoint_dir) / self.cfg.checkpoint_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Execute both phases and return path to best checkpoint."""
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)

        with mlflow.start_run(run_name=self.cfg.mlflow_run_name):
            mlflow.log_params(
                {
                    "backbone": "efficientnet_b0",
                    "phase_a_epochs": self.cfg.phase_a_epochs,
                    "phase_a_lr": self.cfg.phase_a_lr,
                    "phase_b_epochs": self.cfg.phase_b_epochs,
                    "phase_b_lr": self.cfg.phase_b_lr,
                    "device": self.cfg.device,
                }
            )

            history = self._train_phase_a()
            history.extend(self._train_phase_b())

            mlflow.log_metric("best_val_f1", self._best_val_f1)

        return self._best_ckpt_path

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _train_phase_a(self) -> list[dict]:
        self.model.freeze_backbone()
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.phase_a_lr,
        )
        print(f"\n--- Phase A: feature extraction ({self.cfg.phase_a_epochs} epochs) ---")
        print(f"    Trainable params: {self.model.trainable_params():,}")
        return self._run_epochs(
            optimizer, scheduler=None, n_epochs=self.cfg.phase_a_epochs, phase="A"
        )

    def _train_phase_b(self) -> list[dict]:
        self.model.unfreeze_last_n_layers(self.cfg.phase_b_unfreeze_n)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.phase_b_lr,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.phase_b_epochs)
        stopper = EarlyStopping(patience=self.cfg.early_stopping_patience)

        print(f"\n--- Phase B: fine-tuning ({self.cfg.phase_b_epochs} epochs) ---")
        print(f"    Trainable params: {self.model.trainable_params():,}")
        return self._run_epochs(
            optimizer, scheduler, self.cfg.phase_b_epochs, phase="B", early_stopper=stopper
        )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epochs(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        n_epochs: int,
        phase: str,
        early_stopper: EarlyStopping | None = None,
    ) -> list[dict]:
        history = []
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_one_epoch(optimizer)
            val_metrics = self._evaluate(self.val_loader)
            val_f1 = val_metrics["f1"]
            val_loss = val_metrics["loss"]

            if scheduler is not None:
                scheduler.step()

            step = epoch if phase == "A" else self.cfg.phase_a_epochs + epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_auc": val_metrics["auc"],
                },
                step=step,
            )

            print(
                f"  [{phase}] Epoch {epoch:>2}/{n_epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_f1={val_f1:.4f}  val_auc={val_metrics['auc']:.4f}"
            )

            if val_f1 > self._best_val_f1:
                self._best_val_f1 = val_f1
                self.model.save(
                    self._best_ckpt_path, metadata={"epoch": step, "val_f1": val_f1, "phase": phase}
                )
                print(f"       ✓ Checkpoint saved (val_f1={val_f1:.4f})")

            history.append(
                {"phase": phase, "epoch": epoch, "train_loss": train_loss, **val_metrics}
            )

            if early_stopper is not None and early_stopper.step(val_f1):
                print(
                    f"  Early stopping at epoch {epoch} (patience={self.cfg.early_stopping_patience})"
                )
                break

        return history

    def _train_one_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            probs = self.model(images)
            loss = self.criterion(probs, labels)
            total_loss += loss.item() * len(labels)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds = (all_probs >= 0.5).astype(int)

        # Guard for edge cases where all labels are one class (small dataset)
        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            auc = 0.5

        return {
            "loss": total_loss / max(len(loader.dataset), 1),
            "f1": float(f1_score(all_labels, preds, zero_division=0)),
            "auc": auc,
            "accuracy": float((preds == all_labels).mean()),
        }
