"""Dataset loader for document authentication.

DocumentDataset: PyTorch Dataset backed by image paths + labels.
create_dataloaders: builds train/val/test splits with Albumentations augmentation.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.pipeline import DocumentPreprocessor, PreprocessorConfig

# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------


def _train_transform(size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomRotate90(p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.125),
                hole_width_range=(0.05, 0.125),
                fill=0,
                p=0.2,
            ),
            ToTensorV2(),
        ]
    )


def _val_transform() -> A.Compose:
    return A.Compose([ToTensorV2()])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DocumentDataset(Dataset):
    """Loads document images from disk and applies preprocessing + augmentation.

    Args:
        paths: List of image file paths.
        labels: Corresponding binary labels (0=authentic, 1=forged).
        preprocessor: DocumentPreprocessor instance. Applied before augmentation.
        augment: Albumentations Compose pipeline. None = no augmentation.
    """

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        preprocessor: DocumentPreprocessor,
        augment: A.Compose | None = None,
    ) -> None:
        assert len(paths) == len(labels), "paths and labels must have equal length"
        self.paths = paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img: np.ndarray = cv2.imread(str(self.paths[idx]))  # type: ignore[assignment]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocessing (perspective, denoise, CLAHE, resize, normalize)
        processed = self.preprocessor.process(img)  # float32 HxWx3

        # Albumentations expects uint8 or float32; ToTensorV2 handles both
        if self.augment is not None:
            result = self.augment(image=processed)
            tensor = result["image"].float()
        else:
            tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float()

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    preprocessor_cfg: PreprocessorConfig | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from a directory with authentic/ and forged/ subdirs.

    Directory layout expected:
        data_dir/
          authentic/*.png (or .jpg)
          forged/*.png (or .jpg)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    authentic_paths = sorted((data_dir / "authentic").glob("*.png")) + sorted(
        (data_dir / "authentic").glob("*.jpg")
    )
    forged_paths = sorted((data_dir / "forged").glob("*.png")) + sorted(
        (data_dir / "forged").glob("*.jpg")
    )

    all_paths = authentic_paths + forged_paths
    all_labels = [0] * len(authentic_paths) + [1] * len(forged_paths)

    # Deterministic shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_paths))
    all_paths = [all_paths[i] for i in idx]
    all_labels = [all_labels[i] for i in idx]

    n = len(all_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_paths = all_paths[:n_train]
    train_labels = all_labels[:n_train]
    val_paths = all_paths[n_train : n_train + n_val]
    val_labels = all_labels[n_train : n_train + n_val]
    test_paths = all_paths[n_train + n_val :]
    test_labels = all_labels[n_train + n_val :]

    cfg = preprocessor_cfg or PreprocessorConfig()
    preprocessor = DocumentPreprocessor(cfg)

    train_ds = DocumentDataset(train_paths, train_labels, preprocessor, augment=_train_transform())
    val_ds = DocumentDataset(val_paths, val_labels, preprocessor)
    test_ds = DocumentDataset(test_paths, test_labels, preprocessor)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader
