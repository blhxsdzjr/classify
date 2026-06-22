"""Pure CV lane line detection + color classification — no YOLO, no training.

Pipeline:
  1. Canny edge detection on road ROI (bottom half of image)
  2. Probabilistic Hough transform → line segments
  3. Merge colinear nearby segments into full lane lines
  4. Color classification (HSV or learned ML classifier)
  5. Output per-image white/yellow counts

Evaluate against 结果统计.xlsx count-level GT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# --- Parameters ---
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 80         # higher = fewer but more confident lines
HOUGH_MIN_LINE_LEN = 100     # minimum line length in pixels
HOUGH_MAX_LINE_GAP = 20
ROI_FRACTION = 0.55
MERGE_ANGLE_THR = 6.0        # stricter angle for merging
MERGE_DIST_THR = 60.0        # closer lines to merge
LINE_HALF_WIDTH = 12
MIN_LINE_LEN = 80
TOP_EDGE_MARGIN = 30


def extract_line_candidates(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Canny + HoughLinesP → rank by length×brightness → merge → keep top K.

    Returns line segments sorted by saliency (longer, brighter lines first).
    """
    h, w = image_bgr.shape[:2]
    roi_top = int(h * (1 - ROI_FRACTION))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges[:roi_top, :] = 0  # mask sky

    hough_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=30,
        minLineLength=40, maxLineGap=30,
    )
    if hough_lines is None:
        return []

    # Score and filter
    scored = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0].astype(np.float32)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 35:
            continue
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        angle = math.degrees(math.atan2(dx, max(dy, 1)))
        if angle > 50:
            continue

        num_samples = max(5, int(length / 5))
        xs = np.linspace(x1, x2, num_samples).astype(int)
        ys = np.linspace(y1, y2, num_samples).astype(int)
        xs = np.clip(xs, 0, w - 1); ys = np.clip(ys, 0, h - 1)
        brightness = float(np.mean(gray[ys, xs]))
        score = length * (brightness / 255.0)
        scored.append((score, np.array([x1, y1, x2, y2], dtype=np.float32)))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Keep top 8 lines (typical road has 3-5 lanes, 8 is generous)
    top_segs = [seg for _, seg in scored[:8]]

    if len(top_segs) > 1:
        top_segs = merge_segments(top_segs)

    return top_segs


def detect_lines(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Thin wrapper — contour-based line detection replaces Canny+Hough."""
    return extract_line_candidates(image_bgr)


def line_angle_deg(seg: np.ndarray) -> float:
    dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    return math.degrees(math.atan2(dx, max(abs(dy), 1)))


def line_center(seg: np.ndarray) -> np.ndarray:
    return np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])


def merge_segments(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Merge colinear nearby segments into longer lines."""
    if len(segments) < 2:
        return segments

    used = [False] * len(segments)
    merged = []

    for i, seg_i in enumerate(segments):
        if used[i]:
            continue
        group = [seg_i]
        used[i] = True
        angle_i = line_angle_deg(seg_i)
        center_i = line_center(seg_i)

        for j, seg_j in enumerate(segments):
            if used[j]:
                continue
            angle_j = line_angle_deg(seg_j)
            center_j = line_center(seg_j)
            if (abs(angle_i - angle_j) < MERGE_ANGLE_THR and
                    np.linalg.norm(center_i - center_j) < MERGE_DIST_THR):
                group.append(seg_j)
                used[j] = True

        if len(group) == 1:
            merged.append(group[0])
        else:
            # Fit a single line through all points in the group
            pts = np.vstack([g.reshape(2, 2) for g in group])
            xs, ys = pts[:, 0], pts[:, 1]
            # Use total least squares (PCA) for line fitting
            mean = np.array([xs.mean(), ys.mean()])
            centered = pts - mean
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]  # principal component
            # Project all points onto the line, find extremes
            proj = centered @ direction
            t_min, t_max = proj.min(), proj.max()
            p1 = mean + t_min * direction
            p2 = mean + t_max * direction
            merged.append(np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32))

    return merged


def filter_lines(segments: list[np.ndarray], img_h: int) -> list[np.ndarray]:
    """Remove short lines and lines too close to ROI top edge."""
    roi_top = int(img_h * (1 - ROI_FRACTION))
    valid = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_LINE_LEN:
            continue
        # At least one endpoint should be well into the road area
        if min(y1, y2) < roi_top + TOP_EDGE_MARGIN and max(y1, y2) < roi_top + 2 * TOP_EDGE_MARGIN:
            continue
        valid.append(seg)
    return valid


def extract_line_region(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Extract a binary mask covering a band around the line segment."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return np.zeros((h, w), dtype=bool)

    # Normal direction
    nx, ny = -dy / length, dx / length

    # Four corners of the band
    hw = LINE_HALF_WIDTH
    corners = np.array([
        [x1 + nx * hw, y1 + ny * hw],
        [x1 - nx * hw, y1 - ny * hw],
        [x2 - nx * hw, y2 - ny * hw],
        [x2 + nx * hw, y2 + ny * hw],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners], 1)
    return mask.astype(bool)


def classify_color(image_bgr: np.ndarray, mask: np.ndarray,
                   use_ml: bool = False) -> tuple[str, float, float, float]:
    """Classify a line region as white or yellow.

    Returns (class, score, white_fraction, yellow_fraction).
    """
    if use_ml:
        from .color_classifier import learned_classify_lane_color
        x_idx = np.where(mask.any(axis=0))[0]
        y_idx = np.where(mask.any(axis=1))[0]
        if len(x_idx) == 0 or len(y_idx) == 0:
            return "unknown", 0, 0, 0
        bbox = np.array([x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()])
        decision = learned_classify_lane_color(image_bgr, mask, bbox)
        return decision.cls, decision.score, decision.white_fraction, decision.yellow_fraction

    from .color_classifier import classify_lane_color as hsv_classify
    # Balanced thresholds for CV line detection
    decision = hsv_classify(
        image_bgr, mask,
        min_value=60,
        white_sat_max=80,
        white_value_min=140,
        yellow_hue_min=14,
        yellow_hue_max=48,
        yellow_sat_min=40,
        yellow_value_min=80,
        min_color_fraction=0.05,
    )
    return decision.cls, decision.score, decision.white_fraction, decision.yellow_fraction


def process_image(image_path: Path, use_ml: bool = False,
                  save_vis: str | None = None) -> dict:
    """Full pipeline on one image: detect → merge → filter → classify."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Cannot read {image_path}"}
    h, w = image.shape[:2]

    segments = detect_lines(image)
    merged = merge_segments(segments)
    filtered = filter_lines(merged, h)

    instances = []
    vis = image.copy() if save_vis else None

    for seg in filtered:
        mask = extract_line_region(image, seg)
        cls, score, wf, yf = classify_color(image, mask, use_ml)

        if cls == "unknown":
            continue

        x1, y1, x2, y2 = seg
        angle = line_angle_deg(seg)
        instances.append({
            "class": cls,
            "score": round(score, 4),
            "white_fraction": round(wf, 4),
            "yellow_fraction": round(yf, 4),
            "angle_deg": round(angle, 2),
            "endpoints": [[round(x1, 1), round(y1, 1)], [round(x2, 1), round(y2, 1)]],
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

        if vis is not None:
            color = (0, 220, 0) if cls == "white_lane" else (0, 210, 255)
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.line(vis, p1, p2, color, 2)
            label = f"{cls} {score:.2f}"
            cv2.putText(vis, label, (int(x1), max(20, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if vis is not None:
        out_dir = Path(save_vis)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / image_path.name), vis)

    return {
        "file": image_path.name,
        "width": w, "height": h,
        "raw_segments": len(segments),
        "merged_lines": len(merged),
        "final_lines": len(instances),
        "instances": instances,
    }


def main():
    parser = argparse.ArgumentParser(description="Pure CV lane line detection + color classification.")
    parser.add_argument("--source", default="datasets/local_colm/images/test",
                        help="Image directory.")
    parser.add_argument("--out", default="predictions_cv.json",
                        help="Output JSON path.")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx",
                        help="Ground truth count spreadsheet for evaluation.")
    parser.add_argument("--use-ml", action="store_true", default=True,
                        help="Use learned ML color classifier instead of HSV (default).")
    parser.add_argument("--no-ml", action="store_true",
                        help="Use HSV color classifier instead of ML.")
    parser.add_argument("--save-vis", default=None,
                        help="Optional directory for visualization images.")
    args = parser.parse_args()

    image_dir = Path(args.source)
    image_files = sorted(p for p in image_dir.rglob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    results = {}
    for img_path in image_files:
        res = process_image(img_path, use_ml=args.use_ml, save_vis=args.save_vis)
        results[img_path.name] = res
        n = res.get("final_lines", 0)
        w = sum(1 for i in res.get("instances", []) if i["class"] == "white_lane")
        y = sum(1 for i in res.get("instances", []) if i["class"] == "yellow_lane")
        print(f"  {img_path.name}: {n} lines ({w} white, {y} yellow)")

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {"meta": {"method": "cv_hough", "use_ml": args.use_ml}, "images": convert(results)}
    with open(args.out, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.out}")

    # Evaluate against GT if xlsx provided
    gt_path = Path(args.gt_xlsx)
    if gt_path.exists():
        from .evaluate_lane_metrics import evaluate_count_only
        import argparse as ap
        eval_args = ap.Namespace(
            pred=args.out, gt_counts=None, gt_xlsx=str(gt_path),
            count_only=True, conf_thr=0, out="metrics_cv.json",
        )
        evaluate_count_only(eval_args, Path(args.out))
    else:
        print("No GT xlsx found, skipping evaluation.")


if __name__ == "__main__":
    main()
