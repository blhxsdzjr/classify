from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .classes import WHITE, YELLOW
from .classical_lane import color_masks
from .geometry import fit_line_from_points


@dataclass(frozen=True)
class RowAnchorParams:
    num_rows: int = 28
    roi_top_ratio: float = 0.38
    bottom_margin: int = 10
    band_height: int = 7
    smooth_width: int = 31
    peak_min_distance: int = 34
    peak_rel_threshold: float = 0.32
    peak_abs_threshold: float = 0.10
    max_peaks_per_row: int = 8
    max_match_distance: float = 80.0
    max_missed_rows: int = 4
    min_points: int = 5
    min_vertical_span_ratio: float = 0.16
    curve_degree: int = 2
    curve_samples: int = 80


@dataclass
class RowPeak:
    row_idx: int
    x: float
    y: float
    score: float
    white_score: float
    yellow_score: float


@dataclass
class RowTrack:
    points: list[list[float]] = field(default_factory=list)
    score: float = 0.0
    white_score: float = 0.0
    yellow_score: float = 0.0
    missed: int = 0

    def add(self, peak: RowPeak) -> None:
        self.points.append([float(peak.x), float(peak.y)])
        self.score += float(peak.score)
        self.white_score += float(peak.white_score)
        self.yellow_score += float(peak.yellow_score)
        self.missed = 0

    def predict_x(self, y: float) -> float:
        if not self.points:
            return 0.0
        if len(self.points) < 3:
            return float(self.points[-1][0])
        pts = np.asarray(self.points[-5:], dtype=np.float32)
        try:
            slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], deg=1)
            return float(slope * y + intercept)
        except np.linalg.LinAlgError:
            return float(self.points[-1][0])

    def avg_score(self) -> float:
        return self.score / max(len(self.points), 1)

    def cls(self) -> str:
        return YELLOW if self.yellow_score > self.white_score else WHITE


def row_anchors(height: int, params: RowAnchorParams) -> list[int]:
    top = int(height * params.roi_top_ratio)
    bottom = height - params.bottom_margin
    if params.num_rows <= 1:
        return [bottom]
    return [round(top + i * (bottom - top) / (params.num_rows - 1)) for i in range(params.num_rows)]


def roi_x_bounds(width: int, height: int, y: float, params: RowAnchorParams) -> tuple[int, int]:
    top = height * params.roi_top_ratio
    bottom = max(float(height - params.bottom_margin), top + 1.0)
    t = np.clip((y - top) / (bottom - top), 0.0, 1.0)
    left = (0.23 * (1.0 - t) + 0.04 * t) * width
    right = (0.77 * (1.0 - t) + 0.96 * t) * width
    return int(max(0, left)), int(min(width - 1, right))


def smooth_1d(values: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width) | 1)
    kernel = np.hanning(width).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1e-6)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def build_response_maps(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    white, yellow, combined = color_masks(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 135)
    response = (
        0.45 * (white.astype(np.float32) / 255.0)
        + 0.45 * (yellow.astype(np.float32) / 255.0)
        + 0.20 * (edges.astype(np.float32) / 255.0)
        + 0.10 * (combined.astype(np.float32) / 255.0)
    )
    return response, white.astype(np.float32) / 255.0, yellow.astype(np.float32) / 255.0


def local_peak_centroid(row: np.ndarray, x: int, radius: int = 7) -> float:
    left = max(0, x - radius)
    right = min(len(row), x + radius + 1)
    xs = np.arange(left, right, dtype=np.float32)
    weights = row[left:right].astype(np.float32)
    if float(weights.sum()) <= 1e-6:
        return float(x)
    return float((xs * weights).sum() / weights.sum())


def find_row_peaks(
    response: np.ndarray,
    white_map: np.ndarray,
    yellow_map: np.ndarray,
    params: RowAnchorParams,
) -> list[list[RowPeak]]:
    height, width = response.shape
    all_peaks: list[list[RowPeak]] = []
    for row_idx, y in enumerate(row_anchors(height, params)):
        y1 = max(0, y - params.band_height // 2)
        y2 = min(height, y + params.band_height // 2 + 1)
        row = response[y1:y2].mean(axis=0)
        white_row = white_map[y1:y2].mean(axis=0)
        yellow_row = yellow_map[y1:y2].mean(axis=0)
        left, right = roi_x_bounds(width, height, y, params)
        masked = np.zeros_like(row)
        masked[left : right + 1] = row[left : right + 1]
        smoothed = smooth_1d(masked, params.smooth_width)
        row_max = float(smoothed.max())
        threshold = max(params.peak_abs_threshold, row_max * params.peak_rel_threshold)
        candidates = []
        for x in range(max(1, left), min(width - 1, right)):
            if smoothed[x] < threshold:
                continue
            if smoothed[x] >= smoothed[x - 1] and smoothed[x] >= smoothed[x + 1]:
                cx = local_peak_centroid(smoothed, x)
                wx = int(np.clip(round(cx), 0, width - 1))
                candidates.append(
                    RowPeak(
                        row_idx=row_idx,
                        x=cx,
                        y=float(y),
                        score=float(smoothed[x]),
                        white_score=float(smooth_1d(white_row, params.smooth_width)[wx]),
                        yellow_score=float(smooth_1d(yellow_row, params.smooth_width)[wx]),
                    )
                )
        candidates.sort(key=lambda peak: peak.score, reverse=True)
        filtered: list[RowPeak] = []
        for peak in candidates:
            if all(abs(peak.x - kept.x) >= params.peak_min_distance for kept in filtered):
                filtered.append(peak)
            if len(filtered) >= params.max_peaks_per_row:
                break
        all_peaks.append(filtered)
    return all_peaks


def track_peaks(peaks_by_row: list[list[RowPeak]], params: RowAnchorParams) -> list[RowTrack]:
    active: list[RowTrack] = []
    finished: list[RowTrack] = []

    # Start from the bottom of the image and go upward, which follows lane perspective.
    for peaks in reversed(peaks_by_row):
        assigned_peaks: set[int] = set()
        assignments: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(active):
            if not peaks:
                continue
            expected = track.predict_x(peaks[0].y)
            for peak_idx, peak in enumerate(peaks):
                dist = abs(peak.x - expected)
                if dist <= params.max_match_distance:
                    assignments.append((dist, track_idx, peak_idx))
        assignments.sort(key=lambda item: item[0])

        used_tracks: set[int] = set()
        for _, track_idx, peak_idx in assignments:
            if track_idx in used_tracks or peak_idx in assigned_peaks:
                continue
            active[track_idx].add(peaks[peak_idx])
            used_tracks.add(track_idx)
            assigned_peaks.add(peak_idx)

        for idx, track in enumerate(active):
            if idx not in used_tracks:
                track.missed += 1

        still_active = []
        for track in active:
            if track.missed > params.max_missed_rows:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active

        for peak_idx, peak in enumerate(peaks):
            if peak_idx in assigned_peaks:
                continue
            track = RowTrack()
            track.add(peak)
            active.append(track)

    finished.extend(active)
    return finished


def track_to_prediction(track: RowTrack, image_shape: tuple[int, int], source: str) -> dict[str, Any] | None:
    height, width = image_shape
    points = sorted(track.points, key=lambda pt: pt[1])
    if len(points) < 2:
        return None
    curve_points = smooth_curve_points(points, image_shape, RowAnchorParams())
    fitted = fit_line_from_points(np.asarray(curve_points if curve_points else points, dtype=np.float32))
    if fitted is None:
        return None
    angle, endpoints, bbox = fitted
    cls = track.cls()
    color_total = max(track.white_score + track.yellow_score, 1e-6)
    return {
        "class": cls,
        "conf": float(np.clip(track.avg_score(), 0.0, 1.0)),
        "angle_deg": float(angle),
        "endpoints": endpoints,
        "bbox": bbox,
        "source_class": source,
        "color_score": float((track.yellow_score if cls == YELLOW else track.white_score) / color_total),
        "white_score": float(track.white_score / color_total),
        "yellow_score": float(track.yellow_score / color_total),
        "row_points": points,
        "curve_points": curve_points,
    }


def smooth_curve_points(
    points: list[list[float]],
    image_shape: tuple[int, int],
    params: RowAnchorParams,
) -> list[list[float]]:
    if len(points) < 3:
        return points
    height, width = image_shape
    pts = np.asarray(sorted(points, key=lambda pt: pt[1]), dtype=np.float32)
    degree = min(params.curve_degree, len(pts) - 1)
    try:
        coeff = np.polyfit(pts[:, 1], pts[:, 0], deg=degree)
    except np.linalg.LinAlgError:
        return points
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    ys = np.linspace(y_min, y_max, max(params.curve_samples, len(points)))
    xs = np.polyval(coeff, ys)
    curve = []
    for x, y in zip(xs, ys):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        curve.append([float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))])
    return curve if len(curve) >= 2 else points


def filter_tracks(tracks: list[RowTrack], image_shape: tuple[int, int], params: RowAnchorParams) -> list[RowTrack]:
    height, _ = image_shape
    kept = []
    for track in tracks:
        if len(track.points) < params.min_points:
            continue
        ys = [pt[1] for pt in track.points]
        if max(ys) - min(ys) < height * params.min_vertical_span_ratio:
            continue
        kept.append(track)
    kept.sort(key=lambda tr: (len(tr.points), tr.avg_score()), reverse=True)
    return kept


def detect_row_anchor_lanes(image_bgr: np.ndarray, params: RowAnchorParams | None = None) -> list[dict[str, Any]]:
    params = params or RowAnchorParams()
    response, white_map, yellow_map = build_response_maps(image_bgr)
    peaks_by_row = find_row_peaks(response, white_map, yellow_map, params)
    tracks = filter_tracks(track_peaks(peaks_by_row, params), image_bgr.shape[:2], params)
    predictions = []
    for track in tracks:
        pred = track_to_prediction(track, image_bgr.shape[:2], "row_anchor_response")
        if pred is not None:
            predictions.append(pred)
    return predictions


def apply_count_constraints(instances: list[dict[str, Any]], counts: dict[str, int]) -> list[dict[str, Any]]:
    selected = []
    used: set[int] = set()
    for cls in sorted([WHITE, YELLOW], key=lambda name: (counts.get(name, 0), name)):
        need = max(0, int(counts.get(cls, 0)))
        score_key = "white_score" if cls == WHITE else "yellow_score"
        ranked = [
            (float(inst.get(score_key, 0.0)), float(inst.get("conf", 0.0)), idx, inst)
            for idx, inst in enumerate(instances)
            if idx not in used
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx, inst in ranked[:need]:
            fixed = dict(inst)
            fixed["class"] = cls
            selected.append(fixed)
            used.add(idx)
    selected.sort(key=lambda inst: float(inst.get("conf", 0.0)), reverse=True)
    return selected
