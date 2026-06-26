"""Centerline-based lane detection — finds lane mark CENTERS, not edges.

Key difference from vertical_lane_pipeline:
  - Old:  Canny + Hough on color mask → finds EDGES (guardrails, curbs, shadows)
  - New:  Color mask → contours → skeleton → fit centerline → matches GT annotation style

GT annotations are lane centerlines. Hough finds edges which are shifted from centers.
This module directly extracts the centerline of each white/yellow region.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classes import WHITE, YELLOW
from .classical_lane import image_files
from .geometry import fit_line_from_points
from .xlsx_counts import read_count_xlsx, read_count_json


@dataclass
class Params:
    max_width: int = 960
    roi_top_ratio: float = 0.30
    min_area: int = 200        # minimum pixel area for a lane region
    min_length: int = 50       # minimum centerline length
    min_angle: float = 20.0    # minimum angle from horizontal
    max_angle: float = 160.0
    max_lanes: int = 8         # max lanes per image
    line_width: int = 14


def parse_args():
    p = argparse.ArgumentParser(description="Centerline-based lane detection.")
    p.add_argument("--source", default="datasets/local_colm/images/test")
    p.add_argument("--gt-xlsx", default="结果统计.xlsx")
    p.add_argument("--out", default="runs/centerline_predictions.json")
    p.add_argument("--save-vis", default="runs/centerline_vis")
    return p.parse_args()


def scaled_image(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img.copy(), 1.0
    s = max_w / w
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA), s


def roi_mask(shape, top_ratio):
    h, w = shape
    top = int(h * top_ratio)
    poly = np.array([[int(w*0.03), h-1], [int(w*0.97), h-1], [int(w*0.82), top], [int(w*0.18), top]], dtype=np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def color_masks(img_bgr):
    """Create white and yellow masks with adaptive thresholds."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = ((s <= 95) & (v >= 145)).astype(np.uint8) * 255
    yellow = ((h >= 15) & (h <= 40) & (s >= 70) & (v >= 95)).astype(np.uint8) * 255
    return {WHITE: white, YELLOW: yellow}


def skeletonize(mask):
    """Zhang-Suen thinning — reduce a binary region to its 1-pixel centerline."""
    from skimage.morphology import skeletonize as skel
    return skel(mask).astype(np.uint8) * 255


def extract_centerlines(mask, params):
    """Find connected components in the mask, skeletonize each, fit lines."""
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Vertical close to connect dashed lines
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vk)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area:
            continue

        # Create a mask for just this contour
        cmask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(cmask, [cnt], -1, 255, -1)

        # Skeletonize to get centerline
        try:
            skel = skeletonize(cmask > 0)
        except Exception:
            continue

        # Get skeleton points
        ys, xs = np.where(skel)
        if len(xs) < 5:
            continue

        pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])

        # Fit line to skeleton points
        fitted = fit_line_from_points(pts)
        if fitted is None:
            continue

        angle, endpoints, bbox = fitted

        # Angle filter
        if angle < params.min_angle or angle > params.max_angle:
            continue

        # Length filter
        x1, y1 = endpoints[0]
        x2, y2 = endpoints[1]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < params.min_length:
            continue

        lines.append({
            "angle_deg": float(angle),
            "endpoints": endpoints,
            "bbox": [float(v) for v in bbox],
            "length": float(length),
            "num_points": len(xs),
        })

    # Sort by length, keep top K
    lines.sort(key=lambda l: l["length"], reverse=True)
    return lines[:params.max_lanes]


def detect_image(img_bgr, params):
    small, scale = scaled_image(img_bgr, params.max_width)
    roi = roi_mask(small.shape[:2], params.roi_top_ratio)
    masks = color_masks(small)

    all_lines = []
    for cls in (WHITE, YELLOW):
        # Apply ROI
        masked = cv2.bitwise_and(masks[cls], roi)
        centerlines = extract_centerlines(masked, params)
        for cl in centerlines:
            # Scale back to original image coordinates
            inv = 1.0 / scale
            cl["endpoints"] = [[x * inv, y * inv] for x, y in cl["endpoints"]]
            cl["bbox"] = [v * inv for v in cl["bbox"]]
            cl["class"] = cls
            cl["conf"] = min(1.0, cl["length"] / 600.0)
            all_lines.append(cl)

    # Deduplicate: remove overlapping lines of same class
    all_lines.sort(key=lambda l: l["length"], reverse=True)
    kept = []
    for line in all_lines:
        x_center = (line["endpoints"][0][0] + line["endpoints"][1][0]) / 2
        dup = False
        for other in kept:
            if line["class"] != other["class"]:
                continue
            ox = (other["endpoints"][0][0] + other["endpoints"][1][0]) / 2
            if abs(x_center - ox) < 40:
                dup = True
                break
        if not dup:
            kept.append(line)

    return kept


def draw_predictions(img, lines):
    vis = img.copy()
    for l in lines:
        color = (0, 0, 255) if l["class"] == WHITE else (255, 0, 0)
        p1 = tuple(int(round(v)) for v in l["endpoints"][0])
        p2 = tuple(int(round(v)) for v in l["endpoints"][1])
        cv2.line(vis, p1, p2, color, 6, cv2.LINE_AA)
        cv2.putText(vis, l["class"], (p1[0]+5, p1[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def evaluate(images, gt_counts):
    totals = {cls: {"detected": 0, "correct": 0, "gt": 0} for cls in (WHITE, YELLOW)}
    for name, payload in images.items():
        stem = Path(name).stem
        gt = next((v for k, v in gt_counts.items() if Path(k).stem == stem), {})
        gt_w = int(gt.get(WHITE, 0))
        gt_y = int(gt.get(YELLOW, 0))
        pred_w = sum(1 for l in payload["lines"] if l["class"] == WHITE)
        pred_y = sum(1 for l in payload["lines"] if l["class"] == YELLOW)
        totals[WHITE]["detected"] += pred_w
        totals[WHITE]["gt"] += gt_w
        totals[WHITE]["correct"] += min(pred_w, gt_w)
        totals[YELLOW]["detected"] += pred_y
        totals[YELLOW]["gt"] += gt_y
        totals[YELLOW]["correct"] += min(pred_y, gt_y)

    def sd(a, b): return a / b if b else 0
    for cls in (WHITE, YELLOW):
        r = totals[cls]
        r["precision"] = sd(r["correct"], r["detected"])
        r["recall"] = sd(r["correct"], r["gt"])
        r["f1"] = sd(2 * r["precision"] * r["recall"], r["precision"] + r["recall"])
    td = totals[WHITE]["detected"] + totals[YELLOW]["detected"]
    tc = totals[WHITE]["correct"] + totals[YELLOW]["correct"]
    tg = totals[WHITE]["gt"] + totals[YELLOW]["gt"]
    totals["overall"] = {
        "detected": td, "correct": tc, "gt": tg,
        "precision": sd(tc, td), "recall": sd(tc, tg),
        "f1": sd(2 * sd(tc, td) * sd(tc, tg), sd(tc, td) + sd(tc, tg)),
    }
    return totals


def main():
    args = parse_args()
    params = Params()
    gt_counts = read_count_xlsx(Path(args.gt_xlsx))
    gt_norm = {Path(k).stem: v for k, v in gt_counts.items()}

    vis_dir = Path(args.save_vis)
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for img_path in image_files(Path(args.source)):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = detect_image(img, params)
        images[img_path.name] = {"width": img.shape[1], "height": img.shape[0], "lines": lines}
        cv2.imwrite(str(vis_dir / img_path.name), draw_predictions(img, lines))
        w = sum(1 for l in lines if l["class"] == WHITE)
        y = sum(1 for l in lines if l["class"] == YELLOW)
        print(f"  {img_path.name}: {len(lines)} lines ({w} white, {y} yellow)")

    # Save predictions
    out = {"meta": {"method": "centerline_skeleton"}, "images": images}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)

    # Count-level evaluation
    metrics = evaluate(images, gt_counts)
    for cls, m in metrics.items():
        print(f"{cls}: detected={m['detected']} correct={m['correct']} gt={m['gt']} "
              f"precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
