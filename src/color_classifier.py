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


def _image_stats(image_bgr: np.ndarray) -> dict:
    """Compute global image statistics for adaptive thresholding.

    Focuses on the road region (bottom 50% of image) since that's where lanes live.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    road_region = image_bgr[h // 2:, :]
    if road_region.size == 0:
        road_region = image_bgr

    hsv = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    v_sorted = np.sort(v.ravel())
    n = len(v_sorted)
    # Use median and percentile to be robust against outliers
    median_v = float(np.median(v))
    p10_v = float(v_sorted[int(n * 0.10)])  # dark area
    p90_v = float(v_sorted[int(n * 0.90)])  # bright area
    mean_s = float(s.mean())
    std_v = float(v.std())
    # Dynamic range: how much contrast is in the road region
    dyn_range = p90_v - p10_v

    return {
        "median_v": median_v,
        "p10_v": p10_v,
        "p90_v": p90_v,
        "mean_s": mean_s,
        "std_v": std_v,
        "dyn_range": dyn_range,
    }


def _region_stats(image_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Compute HSV statistics for the detected lane region and its local surround."""
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Lane region stats
    lane_v = v[mask].astype(np.float32)
    lane_s = s[mask].astype(np.float32)

    if lane_v.size == 0:
        return {"lane_median_v": 0, "lane_median_s": 0, "contrast_vs_surround": 0}

    lane_median_v = float(np.median(lane_v))
    lane_median_s = float(np.median(lane_s))

    # Surround region (dilated mask minus original mask)
    kernel = np.ones((15, 15), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    surround = dilated & ~mask
    if surround.sum() > 0:
        surround_v = v[surround].astype(np.float32)
        surround_median_v = float(np.median(surround_v))
        contrast = lane_median_v - surround_median_v
    else:
        contrast = 0

    return {
        "lane_median_v": lane_median_v,
        "lane_median_s": lane_median_s,
        "contrast_vs_surround": contrast,
    }


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
    """Classify a detected lane-line region as white or yellow (fixed thresholds).

    Use adaptive_classify_lane_color for per-image threshold tuning.
    """
    return _classify_impl(
        image_bgr, mask,
        min_value=min_value, white_sat_max=white_sat_max,
        white_value_min=white_value_min, yellow_hue_min=yellow_hue_min,
        yellow_hue_max=yellow_hue_max, yellow_sat_min=yellow_sat_min,
        yellow_value_min=yellow_value_min, min_color_fraction=min_color_fraction,
    )


def adaptive_classify_lane_color(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_color_fraction: float = 0.04,
) -> ColorDecision:
    """Classify lane color with thresholds adapted to image lighting conditions.

    Key adaptations:
    - Bright scenes (high median V): raise white_value_min (need brighter to count as white)
    - Dark scenes (low median V): lower all value thresholds
    - Low saturation scenes (overcast): lower yellow_sat_min to not miss yellow
    - High contrast (sun+shadow): use more permissive range, rely more on relative contrast
    """
    if image_bgr.size == 0:
        return ColorDecision(UNKNOWN, 0.0, 0.0, 0.0, 0)

    stats = _image_stats(image_bgr)
    median_v = stats["median_v"]
    mean_s = stats["mean_s"]
    dyn_range = stats["dyn_range"]

    # -- Adapt thresholds to scene brightness --
    # Brightness factor: 1.0 at median_v=128, <1 for dark, >1 for bright
    brightness = np.clip(median_v / 128.0, 0.5, 1.8)

    # In bright scenes: lines need higher V to be "white" or "yellow"
    # In dark scenes: tolerante lower V
    base_white_v = int(np.clip(140 + 30 * (brightness - 1.0), 110, 190))
    base_yellow_v = int(np.clip(75 + 25 * (brightness - 1.0), 50, 115))

    # Saturation factor: in low-saturation scenes, be more permissive
    sat_factor = np.clip(mean_s / 45.0, 0.5, 1.5)  # 1.0 at mean_s=45

    base_yellow_s = int(np.clip(40 / max(sat_factor, 0.6), 20, 70))
    base_white_s_max = int(np.clip(90 * sat_factor, 55, 130))

    # Min value: very dark pixels are noise, adjust by scene
    base_min_v = int(np.clip(50 + 0.15 * dyn_range, 35, 85))

    # In low-contrast scenes (dyn_range < 40), be more lenient
    if dyn_range < 40:
        base_min_v = max(30, base_min_v - 15)
        base_white_v = max(100, base_white_v - 20)

    # -- Region-level refinement --
    mask_bool = _safe_mask(mask, image_bgr.shape[:2])
    region_stats = _region_stats(image_bgr, mask_bool)
    lane_v = region_stats["lane_median_v"]
    contrast = region_stats["contrast_vs_surround"]
    lane_s = region_stats["lane_median_s"]

    # If lane is significantly brighter than surround, it's likely white
    # even if absolute V is not extremely high
    if contrast > 30 and lane_v > 100:
        base_white_v = min(base_white_v, int(lane_v - 5))

    # If lane has very high saturation, it's likely yellow
    if lane_s > 80:
        base_yellow_s = max(25, base_yellow_s - 15)

    return _classify_impl(
        image_bgr, mask,
        min_value=base_min_v, white_sat_max=base_white_s_max,
        white_value_min=base_white_v, yellow_hue_min=14,
        yellow_hue_max=48, yellow_sat_min=base_yellow_s,
        yellow_value_min=base_yellow_v, min_color_fraction=min_color_fraction,
    )


def _classify_impl(
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    min_value: int,
    white_sat_max: int,
    white_value_min: int,
    yellow_hue_min: int,
    yellow_hue_max: int,
    yellow_sat_min: int,
    yellow_value_min: int,
    min_color_fraction: float,
) -> ColorDecision:
    """Core classification logic. Works with any thresholds."""
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

    # Decision logic: prefer yellow when it's clearly dominant,
    # otherwise pick the stronger signal
    if yellow_fraction >= min_color_fraction and yellow_fraction > white_fraction * 1.2:
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)
    if white_fraction >= min_color_fraction:
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)

    if yellow_fraction > white_fraction:
        return ColorDecision(YELLOW, yellow_fraction, white_fraction, yellow_fraction, valid_pixels)
    if white_fraction > 0:
        return ColorDecision(WHITE, white_fraction, white_fraction, yellow_fraction, valid_pixels)
    return ColorDecision(UNKNOWN, 0.0, white_fraction, yellow_fraction, valid_pixels)
