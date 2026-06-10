"""Backbone factory for DocumentClassifier variants.

Supported backbones:
  - efficientnet_b0  : 5.3M params, 224×224  (default, production model)
  - resnet18         : 11.2M params, 224×224  (baseline)
  - mobilenet_v3_small: 2.5M params, 224×224  (edge/mobile)

All variants share the same interface as DocumentClassifier:
  - forward(x) → scalar probability in [0,1]
  - freeze_backbone() / unfreeze_last_n_layers(n) / unfreeze_all()
  - save(path) / load(path)
  - total_params() / trainable_params()
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models

Backbone = Literal["efficientnet_b0", "resnet18", "mobilenet_v3_small"]

_HEAD_IN_FEATURES: dict[str, int] = {
    "efficientnet_b0": 1280,
    "resnet18": 512,
    "mobilenet_v3_small": 576,
}


def _build_head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


class DocumentClassifierV2(nn.Module):
    """Binary document classifier supporting multiple torchvision backbones.

    Args:
        backbone: Architecture name — 'efficientnet_b0', 'resnet18', or 'mobilenet_v3_small'.
        pretrained: Use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        backbone: Backbone = "efficientnet_b0",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        weights_flag = "IMAGENET1K_V1" if pretrained else None

        if backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights_flag)
            base.classifier = _build_head(_HEAD_IN_FEATURES[backbone])
            self.backbone = base

        elif backbone == "resnet18":
            base = models.resnet18(weights=weights_flag)
            base.fc = _build_head(_HEAD_IN_FEATURES[backbone])
            self.backbone = base

        elif backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=weights_flag)
            in_feat = base.classifier[-1].in_features
            base.classifier[-1] = _build_head(in_feat)
            self.backbone = base

        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out.squeeze(-1).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _feature_layers(self) -> list[nn.Module]:
        if self.backbone_name == "efficientnet_b0":
            return list(self.backbone.features.children())
        if self.backbone_name == "resnet18":
            return [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
        # mobilenet_v3_small
        return list(self.backbone.features.children())

    def _head_params(self):
        if self.backbone_name == "efficientnet_b0":
            return self.backbone.classifier.parameters()
        if self.backbone_name == "resnet18":
            return self.backbone.fc.parameters()
        return self.backbone.classifier.parameters()

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self._head_params():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        for param in self._head_params():
            param.requires_grad = True
        for layer in self._feature_layers()[-n:]:
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
        self.eval()
        return self(x)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "backbone": self.backbone_name,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> DocumentClassifierV2:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        backbone: Backbone = ckpt.get("backbone", "efficientnet_b0")
        model = cls(backbone=backbone, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def build_classifier(
    backbone: Backbone = "efficientnet_b0",
    pretrained: bool = True,
) -> DocumentClassifierV2:
    """Factory shortcut: build and return a DocumentClassifierV2."""
    return DocumentClassifierV2(backbone=backbone, pretrained=pretrained)
