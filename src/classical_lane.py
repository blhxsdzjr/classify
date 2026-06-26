from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .classes import UNKNOWN, WHITE, YELLOW, normalize_class_name
from .geometry import LineInstance


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_NAMES = [
    "white_fraction",
    "yellow_fraction",
    "mean_hue_yellowness",
    "mean_saturation",
    "mean_value",
    "lab_b_yellow",
    "relative_length",
    "verticalness",
]


@dataclass
class DetectorParams:
    roi_top_ratio: float = 0.38
    canny_low: int = 45
    canny_high: int = 140
    hough_threshold: int = 28
    min_line_length: int = 45
    max_line_gap: int = 28
    line_width: int = 12
    min_angle_from_horizontal: float = 12.0
    merge_angle_threshold: float = 8.0
    merge_distance_threshold: float = 45.0
    min_candidate_score: float = 0.03


@dataclass
class LaneCandidate:
    angle_deg: float
    endpoints: list[list[float]]
    bbox: list[float]
    length: float
    features: list[float]
    white_fraction: float
    yellow_fraction: float
    candidate_score: float

    def to_instance(
        self,
        cls: str,
        *,
        conf: float,
        color_score: float,
        extra: dict[str, Any] | None = None,
    ) -> LineInstance:
        inst = LineInstance(
            cls=cls,
            conf=conf,
            angle_deg=self.angle_deg,
            endpoints=self.endpoints,
            bbox=self.bbox,
            source_class="classical_line",
            color_score=color_score,
            white_fraction=self.white_fraction,
            yellow_fraction=self.yellow_fraction,
        )
        data = inst.to_dict()
        if extra:
            data.update(extra)
        return LineInstance.from_dict(data)


def image_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def segment_angle(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def segment_length(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    return math.hypot(x2 - x1, y2 - y1)


def segment_center(seg: np.ndarray) -> np.ndarray:
    return np.array([(seg[0] + seg[2]) * 0.5, (seg[1] + seg[3]) * 0.5], dtype=np.float32)


def point_line_distance(point: np.ndarray, seg: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    dx, dy = x2 - x1, y2 - y1
    denom = math.hypot(dx, dy)
    if denom < 1e-6:
        return float(np.linalg.norm(point - np.array([x1, y1], dtype=np.float32)))
    return abs(dy * point[0] - dx * point[1] + x2 * y1 - y2 * x1) / denom


def roi_mask(shape: tuple[int, int], roi_top_ratio: float) -> np.ndarray:
    import cv2

    h, w = shape
    top = int(h * roi_top_ratio)
    polygon = np.array(
        [
            [int(w * 0.05), h - 1],
            [int(w * 0.95), h - 1],
            [int(w * 0.78), top],
            [int(w * 0.22), top],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def color_masks(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = ((s <= 95) & (v >= 145)).astype(np.uint8) * 255
    yellow = ((h >= 12) & (h <= 45) & (s >= 45) & (v >= 85)).astype(np.uint8) * 255
    combined = cv2.bitwise_or(white, yellow)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return white, yellow, combined


def build_edge_map(image_bgr: np.ndarray, params: DetectorParams) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_gray = cv2.Canny(blurred, params.canny_low, params.canny_high)
    _, _, combined_color = color_masks(image_bgr)
    edges_color = cv2.Canny(combined_color, 30, 100)
    edges = cv2.bitwise_or(edges_gray, edges_color)
    edges = cv2.bitwise_and(edges, roi_mask(edges.shape, params.roi_top_ratio))
    return edges


def detect_raw_segments(image_bgr: np.ndarray, params: DetectorParams) -> list[np.ndarray]:
    import cv2

    edges = build_edge_map(image_bgr, params)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params.hough_threshold,
        minLineLength=params.min_line_length,
        maxLineGap=params.max_line_gap,
    )
    if lines is None:
        return []

    h, w = edges.shape
    segments: list[np.ndarray] = []
    for raw in lines[:, 0, :].astype(np.float32):
        length = segment_length(raw)
        if length < params.min_line_length:
            continue
        angle = segment_angle(raw)
        if angle < params.min_angle_from_horizontal or angle > 180.0 - params.min_angle_from_horizontal:
            continue
        cx, cy = segment_center(raw)
        if cy < h * params.roi_top_ratio or cx < 0 or cx >= w:
            continue
        segments.append(raw)
    return segments


def fit_segment_group(group: list[np.ndarray]) -> np.ndarray:
    import cv2

    pts = np.vstack([seg.reshape(2, 2) for seg in group]).astype(np.float32)
    if len(pts) < 2:
        return group[0]
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    origin = np.array([float(cx), float(cy)], dtype=np.float32)
    t = (pts - origin) @ direction
    p1 = origin + direction * float(np.min(t))
    p2 = origin + direction * float(np.max(t))
    return np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32)


def merge_segments(segments: list[np.ndarray], params: DetectorParams) -> list[np.ndarray]:
    if len(segments) < 2:
        return segments

    ordered = sorted(segments, key=segment_length, reverse=True)
    used = [False] * len(ordered)
    merged: list[np.ndarray] = []

    for idx, base in enumerate(ordered):
        if used[idx]:
            continue
        used[idx] = True
        group = [base]
        base_angle = segment_angle(base)
        base_center = segment_center(base)

        for other_idx, other in enumerate(ordered[idx + 1 :], start=idx + 1):
            if used[other_idx]:
                continue
            other_angle = segment_angle(other)
            other_center = segment_center(other)
            if angle_diff_deg(base_angle, other_angle) > params.merge_angle_threshold:
                continue
            distance = min(
                float(np.linalg.norm(base_center - other_center)),
                point_line_distance(other_center, base),
                point_line_distance(base_center, other),
            )
            if distance <= params.merge_distance_threshold:
                used[other_idx] = True
                group.append(other)

        merged.append(fit_segment_group(group))
    return merged


def line_mask(shape: tuple[int, int], endpoints: list[list[float]], width: int) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    p1 = tuple(int(round(v)) for v in endpoints[0])
    p2 = tuple(int(round(v)) for v in endpoints[1])
    cv2.line(mask, p1, p2, 255, max(1, int(width)), cv2.LINE_AA)
    return mask.astype(bool)


def segment_bbox(seg: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in seg]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def extract_features(
    image_bgr: np.ndarray,
    endpoints: list[list[float]],
    length: float,
    angle_deg: float,
    params: DetectorParams,
) -> tuple[list[float], float, float, float]:
    import cv2

    mask = line_mask(image_bgr.shape[:2], endpoints, params.line_width)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    pixels = hsv[mask]
    lab_pixels = lab[mask]
    if len(pixels) == 0:
        return [0.0] * len(FEATURE_NAMES), 0.0, 0.0, 0.0

    h = pixels[:, 0].astype(np.float32)
    s = pixels[:, 1].astype(np.float32)
    v = pixels[:, 2].astype(np.float32)
    lab_b = lab_pixels[:, 2].astype(np.float32)

    valid = v >= 45
    if not np.any(valid):
        valid = np.ones_like(v, dtype=bool)
    h, s, v, lab_b = h[valid], s[valid], v[valid], lab_b[valid]

    white_mask = (s <= 95) & (v >= 145)
    yellow_mask = (h >= 12) & (h <= 45) & (s >= 45) & (v >= 85)
    white_fraction = float(np.mean(white_mask)) if len(white_mask) else 0.0
    yellow_fraction = float(np.mean(yellow_mask)) if len(yellow_mask) else 0.0

    hue_yellowness = np.maximum(0.0, 1.0 - np.abs(h - 28.0) / 32.0)
    verticalness = abs(math.sin(math.radians(angle_deg)))
    diag = math.hypot(image_bgr.shape[1], image_bgr.shape[0])
    relative_length = min(1.0, length / max(diag * 0.45, 1.0))
    lab_b_yellow = float(np.clip((np.mean(lab_b) - 128.0) / 60.0, -1.0, 1.0))

    features = [
        white_fraction,
        yellow_fraction,
        float(np.mean(hue_yellowness)),
        float(np.mean(s) / 255.0),
        float(np.mean(v) / 255.0),
        lab_b_yellow,
        relative_length,
        verticalness,
    ]
    candidate_score = max(white_fraction, yellow_fraction, 0.10) * (0.35 + relative_length) * (0.35 + verticalness)
    return features, white_fraction, yellow_fraction, float(candidate_score)


def detect_lane_candidates(image_bgr: np.ndarray, params: DetectorParams | None = None) -> list[LaneCandidate]:
    params = params or DetectorParams()
    raw_segments = detect_raw_segments(image_bgr, params)
    merged = merge_segments(raw_segments, params)
    candidates: list[LaneCandidate] = []
    h, w = image_bgr.shape[:2]

    for seg in merged:
        length = segment_length(seg)
        if length < params.min_line_length:
            continue
        angle = segment_angle(seg)
        endpoints = [
            [float(np.clip(seg[0], 0, w - 1)), float(np.clip(seg[1], 0, h - 1))],
            [float(np.clip(seg[2], 0, w - 1)), float(np.clip(seg[3], 0, h - 1))],
        ]
        features, white_fraction, yellow_fraction, candidate_score = extract_features(
            image_bgr, endpoints, length, angle, params
        )
        if candidate_score < params.min_candidate_score:
            continue
        candidates.append(
            LaneCandidate(
                angle_deg=angle,
                endpoints=endpoints,
                bbox=segment_bbox(seg),
                length=length,
                features=features,
                white_fraction=white_fraction,
                yellow_fraction=yellow_fraction,
                candidate_score=candidate_score,
            )
        )

    candidates.sort(key=lambda cand: cand.candidate_score, reverse=True)
    return candidates


def fallback_color_scores(features: Iterable[float]) -> dict[str, float]:
    values = list(features)
    white_fraction = values[0] if len(values) > 0 else 0.0
    yellow_fraction = values[1] if len(values) > 1 else 0.0
    hue_yellowness = values[2] if len(values) > 2 else 0.0
    saturation = values[3] if len(values) > 3 else 0.0
    value = values[4] if len(values) > 4 else 0.0
    lab_b = values[5] if len(values) > 5 else 0.0
    white_score = white_fraction + 0.35 * max(0.0, value - 0.45) + 0.25 * max(0.0, 0.45 - saturation)
    yellow_score = yellow_fraction + 0.35 * hue_yellowness * max(0.0, saturation) + 0.25 * max(0.0, lab_b)
    return {WHITE: float(white_score), YELLOW: float(yellow_score)}


def load_color_model(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_color_scores(features: list[float], model: dict[str, Any] | None) -> dict[str, float]:
    if not model:
        return fallback_color_scores(features)
    x = np.asarray(features, dtype=np.float32)
    scores: dict[str, float] = {}
    for cls in (WHITE, YELLOW):
        item = model.get("classes", {}).get(cls)
        if not item:
            scores[cls] = fallback_color_scores(features)[cls]
            continue
        mean = np.asarray(item["mean"], dtype=np.float32)
        std = np.asarray(item["std"], dtype=np.float32)
        std = np.maximum(std, 0.05)
        dist = float(np.mean(((x - mean) / std) ** 2))
        prior = float(item.get("prior", 0.5))
        scores[cls] = -dist + math.log(max(prior, 1e-3))
    return scores


def classify_candidate(candidate: LaneCandidate, model: dict[str, Any] | None) -> tuple[str, dict[str, float]]:
    model_scores = model_color_scores(candidate.features, model)
    fallback_scores = fallback_color_scores(candidate.features)

    # Prefer fallback (heuristic) when model is uncertain or strongly disagrees.
    # The model's prior-heavy log-probability scores can misclassify yellow as white
    # due to the 9:1 class imbalance.
    fallback_cls = YELLOW if fallback_scores[YELLOW] > fallback_scores[WHITE] else WHITE
    model_cls = YELLOW if model_scores[YELLOW] > model_scores[WHITE] else WHITE

    if fallback_cls == model_cls:
        cls = model_cls
    elif fallback_cls == YELLOW and fallback_scores[YELLOW] > fallback_scores[WHITE] + 0.3:
        # Fallback strongly prefers yellow — override model
        cls = YELLOW
    elif fallback_cls == WHITE and fallback_scores[WHITE] > fallback_scores[YELLOW] + 0.3:
        cls = WHITE
    else:
        cls = model_cls

    return cls, model_scores


def select_with_count_constraints(
    candidates: list[LaneCandidate],
    target_counts: dict[str, int],
    model: dict[str, Any] | None,
) -> list[tuple[LaneCandidate, str, dict[str, float]]]:
    scored = [(candidate, model_color_scores(candidate.features, model)) for candidate in candidates]
    selected: list[tuple[LaneCandidate, str, dict[str, float]]] = []
    used: set[int] = set()

    for cls in sorted((WHITE, YELLOW), key=lambda name: (target_counts.get(name, 0), name)):
        need = max(0, int(target_counts.get(cls, 0)))
        ranked = [
            (scores[cls], candidate.candidate_score, idx, candidate, scores)
            for idx, (candidate, scores) in enumerate(scored)
            if idx not in used
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx, candidate, scores in ranked[:need]:
            used.add(idx)
            selected.append((candidate, cls, scores))

    selected.sort(key=lambda item: item[0].candidate_score, reverse=True)
    return selected


def candidate_to_prediction(candidate: LaneCandidate, cls: str, scores: dict[str, float]) -> dict[str, Any]:
    conf = float(candidate.candidate_score)
    color_score = float(scores.get(cls, 0.0))
    return {
        "class": cls,
        "conf": conf,
        "angle_deg": float(candidate.angle_deg),
        "endpoints": candidate.endpoints,
        "bbox": candidate.bbox,
        "source_class": "classical_line",
        "color_score": color_score,
        "white_fraction": float(candidate.white_fraction),
        "yellow_fraction": float(candidate.yellow_fraction),
        "white_score": float(scores.get(WHITE, 0.0)),
        "yellow_score": float(scores.get(YELLOW, 0.0)),
    }


def write_counts_csv(images: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "车道线数", "白线数", "黄线数"])
        for filename in sorted(images, key=lambda name: (Path(name).stem.zfill(8), name)):
            instances = images[filename].get("instances", [])
            white = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == WHITE)
            yellow = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == YELLOW)
            writer.writerow([filename, white + yellow, white, yellow])


def draw_predictions(
    image_bgr: np.ndarray,
    instances: list[dict[str, Any]],
    *,
    clean: bool = True,
    line_color: tuple[int, int, int] = (255, 170, 0),
    line_thickness: int = 6,
) -> np.ndarray:
    import cv2

    vis = image_bgr.copy()
    for inst in instances:
        cls = normalize_class_name(inst.get("class", UNKNOWN))
        color = line_color if clean else ((0, 220, 255) if cls == YELLOW else (245, 245, 245))
        poly_points = inst.get("curve_points") or inst.get("row_points")
        if poly_points is not None and len(poly_points) >= 2:
            pts = np.asarray(poly_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
            if not clean:
                for pt in pts[:, 0, :]:
                    cv2.circle(vis, tuple(int(v) for v in pt), 3, color, -1, cv2.LINE_AA)
            x1, y1 = pts[:, 0, :].min(axis=0)
            x2, y2 = pts[:, 0, :].max(axis=0)
        else:
            p1 = tuple(int(round(v)) for v in inst["endpoints"][0])
            p2 = tuple(int(round(v)) for v in inst["endpoints"][1])
            cv2.line(vis, p1, p2, color, line_thickness, cv2.LINE_AA)
            x1, y1, x2, y2 = [int(round(v)) for v in inst["bbox"]]
        if not clean:
            label = f"{cls} {float(inst.get('conf', 0.0)):.2f}"
            cv2.putText(vis, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis
