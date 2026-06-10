"""EfficientNet-B0 fine-tuned classifier for document authentication.

DocumentClassifier wraps torchvision EfficientNet-B0 with a custom binary head.
Provides freeze/unfreeze helpers for two-phase training and checkpoint I/O.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models


class DocumentClassifier(nn.Module):
    """EfficientNet-B0 with a custom binary classification head.

    Output: scalar probability of forgery (0=authentic, 1=forged).
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace default classifier (1280 → 1000) with binary head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)   # (B,)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Phase A: only train the classifier head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        """Phase B: unfreeze the last n blocks of the feature extractor + head."""
        # Unfreeze head unconditionally
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        # features is a Sequential; unfreeze the last n children
        layers = list(self.backbone.features.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return forgery probability for a batch. Model must be in eval mode."""
        self.eval()
        return self(x)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DocumentClassifier":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model
