#!/usr/bin/env python3
"""Train the demo checkpoint (T1 path) on the synthetic training set.

Runs the project's two-phase Trainer on data/train/ and saves the best
checkpoint to models/saved/efficientnet_b0_best.pt — the exact path the API
and dashboard load from. Epochs are kept short because the synthetic task is
trivially separable; early stopping handles the rest.

Usage:
    uv run python scripts/build_training_data.py --n-per-class 200
    uv run python scripts/train_demo.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.loader import create_dataloaders
from src.models.classifier import DocumentClassifier
from src.models.evaluator import ModelEvaluator
from src.models.trainer import Trainer, TrainerConfig


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  data: {args.data_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=0,  # DocumentPreprocessor holds an unpicklable cv2.CLAHE
    )
    print(
        f"Splits — train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = DocumentClassifier(pretrained=True)
    cfg = TrainerConfig(
        phase_a_epochs=2,
        phase_b_epochs=5,
        early_stopping_patience=2,
        device=device,
        mlflow_run_name="efficientnet_b0_demo_t1",
    )
    ckpt_path = Trainer(model, train_loader, val_loader, cfg).run()
    print(f"\nBest checkpoint: {ckpt_path}")

    # Holdout evaluation on the test split
    best = DocumentClassifier.load(ckpt_path, device=device)
    metrics = ModelEvaluator(best, device=device).evaluate(test_loader)
    print(
        f"Test — acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
        f"auc_roc={metrics['auc_roc']:.3f} (n={len(metrics['labels'])})"
    )


if __name__ == "__main__":
    main()
