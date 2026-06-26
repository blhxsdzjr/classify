from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .classes import class_id_to_name, is_eval_class


@dataclass
class LineInstance:
    cls: str
    conf: float
    angle_deg: float
    endpoints: list[list[float]]
    bbox: list[float]
    source_class: str | None = None
    color_score: float | None = None
    white_fraction: float | None = None
    yellow_fraction: float | None = None

    def to_dict(self) -> dict:
        data = {
            "class": self.cls,
            "conf": float(self.conf),
            "angle_deg": float(self.angle_deg),
            "endpoints": self.endpoints,
            "bbox": self.bbox,
        }
        if self.source_class is not None:
            data["source_class"] = self.source_class
        if self.color_score is not None:
            data["color_score"] = float(self.color_score)
        if self.white_fraction is not None:
            data["white_fraction"] = float(self.white_fraction)
        if self.yellow_fraction is not None:
            data["yellow_fraction"] = float(self.yellow_fraction)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "LineInstance":
        return cls(
            cls=str(data["class"]),
            conf=float(data.get("conf", 1.0)),
            angle_deg=float(data["angle_deg"]),
            endpoints=[[float(v) for v in pt] for pt in data["endpoints"]],
            bbox=[float(v) for v in data["bbox"]],
            source_class=data.get("source_class"),
            color_score=data.get("color_score"),
            white_fraction=data.get("white_fraction"),
            yellow_fraction=data.get("yellow_fraction"),
        )


def angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def bbox_iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def center_distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def points_from_mask(mask: np.ndarray, max_points: int = 20000) -> np.ndarray:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return points


def fit_line_from_points(points: np.ndarray) -> tuple[float, list[list[float]], list[float]] | None:
    if points is None or len(points) < 2:
        return None

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    bbox = [float(x1), float(y1), float(x2), float(y2)]

    if len(pts) == 2:
        p0, p1 = pts
        vx, vy = p1 - p0
        if abs(float(vx)) + abs(float(vy)) < 1e-6:
            return None
        angle = math.degrees(math.atan2(float(vy), float(vx))) % 180.0
        return angle, [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]], bbox

    origin = np.mean(pts, axis=0)
    centered = pts - origin
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0].astype(np.float32)
    vx, vy = float(direction[0]), float(direction[1])
    if abs(vx) + abs(vy) < 1e-6:
        return None
    t = centered @ direction
    p_start = origin + direction * float(np.min(t))
    p_end = origin + direction * float(np.max(t))
    angle = math.degrees(math.atan2(float(vy), float(vx))) % 180.0
    endpoints = [
        [float(p_start[0]), float(p_start[1])],
        [float(p_end[0]), float(p_end[1])],
    ]
    return angle, endpoints, bbox


def line_from_bbox_xyxy(bbox: Iterable[float]) -> tuple[float, list[list[float]], list[float]]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if height >= width:
        cx = (x1 + x2) * 0.5
        angle = 90.0
        endpoints = [[cx, y1], [cx, y2]]
    else:
        cy = (y1 + y2) * 0.5
        angle = 0.0
        endpoints = [[x1, cy], [x2, cy]]
    return angle, endpoints, [x1, y1, x2, y2]


def polygon_to_points(values: list[float], image_width: int, image_height: int) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    coords[:, 0] *= float(image_width)
    coords[:, 1] *= float(image_height)
    return coords


def bbox_normalized_to_xyxy(values: list[float], image_width: int, image_height: int) -> list[float]:
    xc, yc, bw, bh = values
    xc *= image_width
    yc *= image_height
    bw *= image_width
    bh *= image_height
    return [
        float(xc - bw / 2.0),
        float(yc - bh / 2.0),
        float(xc + bw / 2.0),
        float(yc + bh / 2.0),
    ]


def read_line_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    names: Mapping[int, str] | None,
    *,
    keep_only_eval_classes: bool = True,
) -> list[LineInstance]:
    instances: list[LineInstance] = []
    if not label_path.exists():
        return instances

    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: invalid normalized label line: {raw!r}") from exc

        class_name = class_id_to_name(class_id, names)
        if keep_only_eval_classes and not is_eval_class(class_name):
            continue

        if len(values) == 4:
            bbox = bbox_normalized_to_xyxy(values, image_width, image_height)
            angle, endpoints, bbox = line_from_bbox_xyxy(bbox)
        elif len(values) >= 4 and len(values) % 2 == 0:
            points = polygon_to_points(values, image_width, image_height)
            fitted = fit_line_from_points(points)
            if fitted is None:
                continue
            angle, endpoints, bbox = fitted
        else:
            raise ValueError(
                f"{label_path}:{line_no}: expected bbox or polygon values, got {len(values)} values"
            )

        instances.append(
            LineInstance(
                cls=class_name,
                conf=1.0,
                angle_deg=angle,
                endpoints=endpoints,
                bbox=bbox,
                source_class=class_name,
            )
        )
    return instances


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for ext in extensions:
        direct = image_dir / f"{stem}{ext}"
        if direct.exists():
            return direct
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions and path.stem == stem:
            return path
    return None
