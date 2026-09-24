"""Pruebas numéricas de los controles locales de preprocesamiento."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.control_preprocessing import crop_union_roi, percentile_normalize_rgb


def test_percentile_normalize_rgb_uses_original_channel_percentiles() -> None:
    image = np.zeros((10, 2, 3), dtype=np.uint8)
    image[:, 0, 0] = np.arange(10, dtype=np.uint8)
    image[:, 1, 0] = 100
    image[:, :, 1] = 7
    image[:, :, 2] = np.arange(20, dtype=np.uint8).reshape(10, 2)

    normalized = percentile_normalize_rgb(image)

    # p1/p99 of red are 0.09/99.01, from all 20 source pixels, not a resized image.
    assert normalized[0, 0].tolist() == [0, 0, 0]
    assert normalized[9, 1, 0] == 255
    assert normalized[:, :, 1].tolist() == [[0, 0]] * 10
    assert normalized[0, 1, 2] == 11
    assert normalized[9, 1, 2] == 255
    assert normalized.dtype == np.uint8


def test_percentile_normalize_rgb_clips_outliers_and_returns_new_array() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    image[0, :2, 0] = 0
    image[-1, -2:, 0] = 200
    image[:, :, 2] = 42

    normalized = percentile_normalize_rgb(image)

    assert normalized[0, 0, 0] == 0
    assert normalized[1, 0, 0] == 127
    assert normalized[-1, -1, 0] == 255
    assert np.all(normalized[:, :, 1] == 0)
    assert np.all(normalized[:, :, 2] == 0)
    assert not np.shares_memory(normalized, image)


def test_percentile_normalize_rgb_rejects_non_rgb_or_non_uint8_input() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        percentile_normalize_rgb(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        percentile_normalize_rgb(np.zeros((2, 2, 3), dtype=np.float32))


def test_crop_union_roi_uses_inclusive_box_and_ceil_margin_per_dimension() -> None:
    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    mask_a = np.zeros((12, 20), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    mask_a[4:6, 5:8] = 1
    mask_b[6:8, 8:10] = 255

    cropped = crop_union_roi(image, [mask_a, mask_b])

    # Union bbox y=4..7 (h=4, margin=1), x=5..9 (w=5, margin=1).
    assert cropped.shape == (6, 7, 3)
    np.testing.assert_array_equal(cropped, image[3:9, 4:11])


def test_crop_union_roi_clips_margin_at_image_edges() -> None:
    image = np.arange(5 * 8 * 3, dtype=np.uint8).reshape(5, 8, 3)
    mask = np.zeros((5, 8), dtype=np.uint8)
    mask[0:2, 6:8] = 1

    cropped = crop_union_roi(image, [mask])

    # Height bbox 2 => margin 1; width bbox 2 => margin 1, clipped to frame.
    assert cropped.shape == (3, 3, 3)
    np.testing.assert_array_equal(cropped, image[0:3, 5:8])


@pytest.mark.parametrize(
    ("masks", "message"),
    [([], "máscara"), (None, "máscara"), ([np.zeros((3, 4), dtype=np.uint8)], "vacía")],
)
def test_crop_union_roi_rejects_absent_or_empty_masks(masks, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        crop_union_roi(np.zeros((3, 4, 3), dtype=np.uint8), masks)


def test_crop_union_roi_rejects_misaligned_masks() -> None:
    with pytest.raises(ValueError, match="dimensiones"):
        crop_union_roi(
            np.zeros((3, 4, 3), dtype=np.uint8),
            [np.zeros((4, 4), dtype=np.uint8)],
        )
