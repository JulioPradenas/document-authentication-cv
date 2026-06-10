"""Tests for src/data/augmentation.py.

Requires OpenCV — skipped entirely if cv2 is not installed.
Run: pytest tests/test_augmentation.py -v
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")

from src.data.augmentation import (  # noqa: E402
    ForgeryConfig,
    ForgeryType,
    SyntheticForgeryGenerator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


def make_generator(seed: int = 42) -> SyntheticForgeryGenerator:
    return SyntheticForgeryGenerator(seed=seed)


ALL_TYPES = list(ForgeryType)
ALL_INTENSITIES = ["mild", "medium", "strong"]


# ---------------------------------------------------------------------------
# 1. apply() returns a copy, not the same object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
def test_apply_returns_copy(forgery_type: ForgeryType) -> None:
    img = make_test_image()
    gen = make_generator()
    config = ForgeryConfig(forgery_type=forgery_type, intensity="medium", seed=42)
    result = gen.apply(img, config)
    assert result is not img, "apply() must return a copy, not the input object"


# ---------------------------------------------------------------------------
# 2. Output shape matches input shape for all 4 types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
def test_apply_same_shape(forgery_type: ForgeryType) -> None:
    img = make_test_image()
    gen = make_generator()
    config = ForgeryConfig(forgery_type=forgery_type, intensity="medium", seed=42)
    result = gen.apply(img, config)
    assert result.shape == img.shape, (
        f"Shape mismatch for {forgery_type}: {result.shape} != {img.shape}"
    )


# ---------------------------------------------------------------------------
# 3. Output differs from input (perturbation was actually applied)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
def test_apply_modifies_image(forgery_type: ForgeryType) -> None:
    img = make_test_image()
    gen = make_generator()
    config = ForgeryConfig(forgery_type=forgery_type, intensity="strong", seed=42)
    result = gen.apply(img, config)
    assert not np.array_equal(result, img), f"apply() with {forgery_type} did not modify the image"


# ---------------------------------------------------------------------------
# 4. generate_batch returns correct number of items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_images,n_per_image", [(1, 1), (3, 2), (4, 4)])
def test_generate_batch_count(n_images: int, n_per_image: int) -> None:
    images = [make_test_image(seed=i) for i in range(n_images)]
    gen = make_generator()
    batch = gen.generate_batch(images, n_per_image=n_per_image)
    assert len(batch) == n_images * n_per_image, (
        f"Expected {n_images * n_per_image} items, got {len(batch)}"
    )


def test_generate_batch_returns_tuples() -> None:
    images = [make_test_image()]
    gen = make_generator()
    batch = gen.generate_batch(images, n_per_image=2)
    for forged, config in batch:
        assert isinstance(forged, np.ndarray)
        assert isinstance(config, ForgeryConfig)
        assert forged.shape == images[0].shape


# ---------------------------------------------------------------------------
# 5. Reproducibility — same seed + same input → identical output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
def test_reproducibility(forgery_type: ForgeryType) -> None:
    img = make_test_image(seed=7)
    config = ForgeryConfig(forgery_type=forgery_type, intensity="medium", seed=99)

    gen1 = make_generator(seed=99)
    result1 = gen1.apply(img, config)

    gen2 = make_generator(seed=99)
    result2 = gen2.apply(img, config)

    assert np.array_equal(result1, result2), f"Results not reproducible for {forgery_type}"


# ---------------------------------------------------------------------------
# 6. All 4 ForgeryType values apply without error across all intensities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
@pytest.mark.parametrize("intensity", ALL_INTENSITIES)
def test_all_forgery_types(forgery_type: ForgeryType, intensity: str) -> None:
    img = make_test_image()
    gen = make_generator()
    config = ForgeryConfig(forgery_type=forgery_type, intensity=intensity, seed=42)
    result = gen.apply(img, config)
    assert result.dtype == np.uint8
    assert result.shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# 7. Splicing with a reference image works correctly
# ---------------------------------------------------------------------------


def test_splicing_with_reference_image() -> None:
    img = make_test_image(seed=0)
    ref = make_test_image(seed=5)
    gen = make_generator()
    config = ForgeryConfig(forgery_type=ForgeryType.SPLICING, intensity="medium", seed=42)
    result = gen.apply(img, config, reference_image=ref)
    assert result.shape == img.shape
    assert not np.array_equal(result, img)


# ---------------------------------------------------------------------------
# 8. Input image is never mutated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgery_type", ALL_TYPES)
def test_input_not_mutated(forgery_type: ForgeryType) -> None:
    img = make_test_image()
    original = img.copy()
    gen = make_generator()
    config = ForgeryConfig(forgery_type=forgery_type, intensity="strong", seed=42)
    gen.apply(img, config)
    assert np.array_equal(img, original), f"apply() mutated the input image for {forgery_type}"
