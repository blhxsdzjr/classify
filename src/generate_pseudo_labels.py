"""Generate YOLO segmentation pseudo-labels using Canny edge + Hough line detection.

Since the zip files contain only images (no YOLO labels), this script creates
weak labels for the lane_line class so we can bootstrap YOLO training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for lane_line training.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/train")
    parser.add_argument("--label-dir", default="datasets/local_colm/labels/train")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-thr", type=int, default=40)
    parser.add_argument("--min-line-len", type=int, default=80)
    parser.add_argument("--max-line-gap", type=int, default=30)
    parser.add_argument("--line-width", type=int, default=12, help="Half-width in pixels for polygon around line.")
    parser.add_argument("--roi-fraction", type=float, default=0.55,
                        help="Only process the bottom fraction of the image (road area).")
    return parser.parse_args()


def detect_lane_lines(image_bgr: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    """Detect lane-like line segments and return them as (x1,y1,x2,y2) arrays."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)

    h, w = edges.shape
    roi_top = int(h * (1 - args.roi_fraction))
    edges[:roi_top, :] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=args.hough_thr,
        minLineLength=args.min_line_len,
        maxLineGap=args.max_line_gap,
    )

    if lines is None:
        return []

    segments = [line[0].astype(np.float32) for line in lines]
    # Filter near-horizontal lines (lane lines are mostly vertical-ish)
    filtered = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = math.degrees(math.atan2(dx, dy)) if dy > 1 else 90
        if angle < 45:  # Keep lines that are more vertical than horizontal
            filtered.append(seg)

    return filtered


def line_to_polygon(x1: float, y1: float, x2: float, y2: float,
                    width: float, h: int, w: int) -> list[tuple[float, float]]:
    """Convert a line segment to a thin quadrilateral (polygon) around it."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return []
    nx = -dy / length * width
    ny = dx / length * width

    pts = [
        (x1 + nx, y1 + ny),
        (x1 - nx, y1 - ny),
        (x2 - nx, y2 - ny),
        (x2 + nx, y2 + ny),
    ]

    # Extend endpoints slightly
    ex = dx / length * width * 0.5
    ey = dy / length * width * 0.5
    pts[0] = (pts[0][0] - ex, pts[0][1] - ey)
    pts[1] = (pts[1][0] - ex, pts[1][1] - ey)
    pts[2] = (pts[2][0] + ex, pts[2][1] + ey)
    pts[3] = (pts[3][0] + ex, pts[3][1] + ey)

    return pts


def polygon_to_yolo_line(polygon: list[tuple[float, float]], w: int, h: int) -> str:
    """Convert polygon pixel coords to YOLO segmentation format string."""
    norm = []
    for px, py in polygon:
        norm.append(f"{float(px) / w:.6f}")
        norm.append(f"{float(py) / h:.6f}")
    return "0 " + " ".join(norm)


def merge_nearby_lines(segments: list[np.ndarray], image_shape: tuple[int, int],
                        angle_thr: float = 10.0, dist_thr: float = 80.0) -> list[np.ndarray]:
    """Merge lines that are close and have similar angles."""
    if len(segments) < 2:
        return segments

    h, w = image_shape

    def line_angle(seg):
        dx = seg[2] - seg[0]
        dy = seg[3] - seg[1]
        return math.degrees(math.atan2(dx, dy))

    def line_center(seg):
        return np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])

    merged = []
    used = [False] * len(segments)

    for i, seg_i in enumerate(segments):
        if used[i]:
            continue
        angle_i = line_angle(seg_i)
        center_i = line_center(seg_i)
        group = [seg_i]
        used[i] = True

        for j, seg_j in enumerate(segments):
            if used[j]:
                continue
            angle_j = line_angle(seg_j)
            center_j = line_center(seg_j)
            if abs(angle_i - angle_j) < angle_thr and np.linalg.norm(center_i - center_j) < dist_thr:
                group.append(seg_j)
                used[j] = True

        if len(group) == 1:
            merged.append(group[0])
        else:
            all_pts = np.vstack([g.reshape(2, 2) for g in group])
            # Fit line through all points
            vx, vy, cx, cy = cv2.fitLine(all_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            # Project points onto line to find extremes
            proj = []
            for pt in all_pts:
                t = (pt[0] - cx) * vx + (pt[1] - cy) * vy
                proj.append(t)
            t_min, t_max = min(proj), max(proj)
            p1 = np.array([cx + t_min * vx[0], cy + t_min * vy[0]])
            p2 = np.array([cx + t_max * vx[0], cy + t_max * vy[0]])
            merged.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    return merged


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in image_dir.rglob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    total_labels = 0
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  SKIP cannot read: {img_path.name}")
            continue
        h, w = image.shape[:2]

        segments = detect_lane_lines(image, args)
        if not segments:
            label_path = label_dir / f"{img_path.stem}.txt"
            label_path.write_text("", encoding="utf-8")
            continue

        merged = merge_nearby_lines(segments, (h, w))

        yolo_lines = []
        for seg in merged:
            x1, y1, x2, y2 = seg
            poly = line_to_polygon(x1, y1, x2, y2, args.line_width, h, w)
            if poly:
                yolo_lines.append(polygon_to_yolo_line(poly, w, h))

        label_path = label_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        total_labels += len(yolo_lines)

        if len(image_files) <= 10 or len(yolo_lines) > 0:
            print(f"  {img_path.name}: {len(merged)} segments -> {len(yolo_lines)} labels")

    print(f"\nTotal: {total_labels} labels across {len(image_files)} images in {label_dir}")


if __name__ == "__main__":
    main()
