from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classes import UNKNOWN, WHITE, YELLOW


@dataclass(frozen=True)
class ColorDecision:
    cls: str
    score: float
    white_fraction: float
    yellow_fraction: float
    valid_pixels: int


def _safe_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape[:2] != shape:
        import cv2

        mask = cv2.resize(mask.astype("uint8"), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def classify_lane_color(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_value: int = 70,
    white_sat_max: int = 85,
    white_value_min: int = 155,
    yellow_hue_min: int = 14,
    yellow_hue_max: int = 45,
    yellow_sat_min: int = 45,
    yellow_value_min: int = 90,
    min_color_fraction: float = 0.04,
) -> ColorDecision:
    """Classify a detected lane-line region as white or yellow.

    The detector should provide a tight mask/box. The thresholds are intentionally
    configurable because strong sunlight and camera exposure can shift colors.
    OpenCV HSV hue is in [0, 179].
    """
    if image_bgr.size == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    import cv2

    mask_bool = _safe_mask(mask, image_bgr.shape[:2])
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    valid = mask_bool & (v >= min_value)
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    white_pixels = valid & (s <= white_sat_max) & (v >= white_value_min)
    yellow_pixels = (
        valid
        & (h >= yellow_hue_min)
        & (h <= yellow_hue_max)
        & (s >= yellow_sat_min)
        & (v >= yellow_value_min)
    )

    white_fraction = float(white_pixels.sum() / valid_pixels)
    yellow_fraction = float(yellow_pixels.sum() / valid_pixels)

    if yellow_fraction >= min_color_fraction and yellow_fraction >= white_fraction * 0.65:
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)
    if white_fraction >= min_color_fraction:
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)

    if yellow_fraction > white_fraction:
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)
    if white_fraction > 0:
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)
    return ColorDecision(UNKNOWN, 0.0, white_fraction, yellow_fraction, valid_pixels)
