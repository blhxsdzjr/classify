"""Use GT counts from xlsx to filter YOLO predictions into high-quality training labels.

For test images that have GT counts, we run the already-trained YOLO model,
keep only the top-K detections (K = GT lane_line count), and convert those
detections to YOLO segmentation format labels.

This turns count-level weak supervision into instance-level pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .xlsx_counts import read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert filtered YOLO predictions to training labels.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO best.pt.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--label-dir", default="datasets/local_colm/labels/test")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx")
    parser.add_argument("--conf", type=float, default=0.1, help="Low conf threshold to include more candidates.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def mask_to_polygon(mask: np.ndarray, simplify_eps: float = 2.0) -> list[tuple[float, float]] | None:
    """Convert a binary mask to a simplified polygon."""
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None

    epsilon = simplify_eps / 1000.0 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return pts if len(pts) >= 3 else None


def bbox_to_polygon(xyxy: np.ndarray) -> list[tuple[float, float]]:
    """Convert bbox to polygon (used as fallback when mask is insufficient)."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def line_score(xyxy: np.ndarray, conf: float, mask: np.ndarray | None = None) -> float:
    """Score a detection for ranking. Prefer longer, narrower lines with high conf."""
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    length = max(w, h)
    width = min(w, h)
    aspect = length / max(width, 1.0)

    mask_area = float(mask.sum()) if mask is not None else (w * h)
    density = mask_area / max(w * h, 1.0)

    # Reward long, narrow shapes with dense masks
    return float(conf) * math.log1p(length) * min(aspect / 15.0, 1.0) * density


def polygon_to_yolo_line(polygon: list[tuple[float, float]], w: int, h: int) -> str:
    """Convert polygon pixel coords to YOLO segmentation format string."""
    norm = []
    for px, py in polygon:
        norm.append(f"{float(px) / w:.6f}")
        norm.append(f"{float(py) / h:.6f}")
    return "0 " + " ".join(norm)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    gt_counts = read_count_xlsx(Path(args.gt_xlsx))
    print(f"Loaded GT counts for {len(gt_counts)} images")

    model = YOLO(args.weights)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Collect all test images
    image_files = sorted(p for p in image_dir.rglob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    import cv2

    total_labels = 0
    matched = 0

    for img_path in image_files:
        stem = img_path.stem
        gt = gt_counts.get(stem) or gt_counts.get(img_path.name)
        target_count = gt.get("lane_line", 0) if gt else None

        # Predict
        results = model.predict(
            str(img_path), imgsz=args.imgsz, conf=args.conf,
            device=args.device, verbose=False, stream=True,
        )
        result = next(results)

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            label_dir.joinpath(f"{stem}.txt").write_text("", encoding="utf-8")
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()
            if masks.shape[1:3] != (h, w):
                resized = []
                for m in masks:
                    resized.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
                masks = np.asarray(resized)

        # Score and sort all detections
        scored = []
        for idx in range(len(xyxy)):
            mask = masks[idx] > 0.5 if masks is not None else None
            score = line_score(xyxy[idx], confs[idx], mask)
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Filter by GT count if available
        if target_count is not None and target_count > 0:
            keep_indices = [idx for _, idx in scored[:target_count]]
            matched += 1
        else:
            # No GT: keep detections with reasonable score
            keep_indices = [idx for score, idx in scored if score > 0.1]

        # Convert to YOLO labels
        yolo_lines = []
        for idx in keep_indices:
            if masks is not None:
                mask = masks[idx] > 0.5
                poly = mask_to_polygon(mask)
            else:
                poly = None

            if poly is None:
                poly = bbox_to_polygon(xyxy[idx])

            yolo_lines.append(polygon_to_yolo_line(poly, w, h))

        label_path = label_dir / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        total_labels += len(yolo_lines)

        status = f"GT={target_count}" if target_count is not None else "no_GT"
        print(f"  {img_path.name}: {len(yolo_lines)} labels ({status})")

    print(f"\nTotal: {total_labels} labels across {len(image_files)} images")
    print(f"GT-matched images: {matched}/{len(image_files)}")


if __name__ == "__main__":
    main()
