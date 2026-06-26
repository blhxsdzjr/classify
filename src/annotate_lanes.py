#!/usr/bin/env python3
"""OpenCV manual lane-line annotation tool.

Draw lane lines with mouse, label as white/yellow, save as YOLO segmentation format.
Each line = a polygon (list of clicked points + the line segment).

Controls:
  LEFT CLICK       — add point to current line
  RIGHT CLICK      — finish current line
  w / y            — set current line class to white_lane / yellow_lane
  1 / 2            — same as w / y
  BACKSPACE / DEL  — delete last point
  d                — delete last finished line
  n / p            — next / previous image
  s                — save annotations for current image
  q / ESC          — quit (auto-saves current image)

Display:
  white_lane lines  — green
  yellow_lane lines — blue/cyan
  current line      — red dots + yellow line preview
  top-left          — image name, current class, line count
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASS_COLORS = {"white_lane": (0, 255, 0), "yellow_lane": (255, 180, 0)}
CURRENT_COLOR = (0, 0, 255)  # red dots for active line
PREVIEW_COLOR = (0, 255, 255)  # yellow line preview


def parse_args():
    p = argparse.ArgumentParser(description="Manual lane-line annotation with OpenCV.")
    p.add_argument("--image-dir", default="datasets/local_colm/images/test")
    p.add_argument("--label-dir", default="datasets/local_colm/labels/test")
    p.add_argument("--start", type=int, default=0, help="Start image index (0-based).")
    return p.parse_args()


def yolo_polygon_str(points, w, h):
    """Convert pixel points to YOLO segmentation format string."""
    if len(points) < 2:
        return None
    # YOLO format: class_id x1 y1 x2 y2 ... xn yn  (normalized)
    norm = []
    for px, py in points:
        norm.append(f"{px / w:.6f}")
        norm.append(f"{py / h:.6f}")
    # Use last two points as the main line endpoints for the label
    return " ".join(norm)


def draw_annotations(img, lines):
    """Draw all saved lines on the image."""
    vis = img.copy()
    for cls, pts in lines:
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        # Draw line segments
        for i in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), color, 3, cv2.LINE_AA)
        # Draw endpoints as circles
        for pt in pts[:1] + pts[-1:]:
            cv2.circle(vis, tuple(pt), 5, color, -1, cv2.LINE_AA)
        # Class label near first point
        cv2.putText(vis, cls, (pts[0][0] + 8, pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def save_labels(label_dir, stem, lines, w, h):
    """Save lines as YOLO segmentation format .txt file."""
    label_dir.mkdir(parents=True, exist_ok=True)
    yolo_lines = []
    for cls, pts in lines:
        cls_id = 0 if cls == "white_lane" else 1
        poly_str = yolo_polygon_str(pts, w, h)
        if poly_str:
            yolo_lines.append(f"{cls_id} {poly_str}")
    label_path = label_dir / f"{stem}.txt"
    label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def get_image_files(image_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts)
    return files


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    files = get_image_files(image_dir)

    if not files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(files)} images.")
    print("Controls: LEFT=add point | RIGHT=finish line | w/y=white/yellow | d=delete line")
    print("          n/p=next/prev | s=save | DEL=delete point | q=quit")
    print()

    cv2.namedWindow("Lane Annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Annotation", 1280, 720)

    idx = max(0, min(args.start, len(files) - 1))
    current_cls = "white_lane"
    current_points = []  # active line being drawn
    saved_lines = []     # finished lines: [(cls, [(x,y),...]), ...]

    def load_image(index):
        nonlocal current_points, saved_lines
        img = cv2.imread(str(files[index]))
        if img is None:
            return None, None, None
        stem = files[index].stem
        # Load existing labels if any
        label_path = label_dir / f"{stem}.txt"
        saved_lines = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(float(parts[0]))
                cls = "white_lane" if cls_id == 0 else "yellow_lane"
                coords = [float(x) for x in parts[1:]]
                # Group into (x,y) pairs, convert from normalized to pixel
                h, w = img.shape[:2]
                pts = [(int(coords[i] * w), int(coords[i + 1] * h)) for i in range(0, len(coords), 2)]
                if len(pts) >= 2:
                    saved_lines.append((cls, pts))
        current_points = []
        return img, stem, img.shape[:2]

    img, stem, (h, w) = load_image(idx)
    if img is None:
        print(f"Cannot load {files[idx]}")
        return

    while True:
        vis = draw_annotations(img, saved_lines)

        # Draw current line in progress
        for i, pt in enumerate(current_points):
            cv2.circle(vis, pt, 4, CURRENT_COLOR, -1, cv2.LINE_AA)
        if len(current_points) >= 2:
            for i in range(len(current_points) - 1):
                cv2.line(vis, current_points[i], current_points[i + 1], PREVIEW_COLOR, 2, cv2.LINE_AA)

        # Status text
        info_lines = [
            f"Image {idx + 1}/{len(files)}: {stem}.jpg",
            f"Class: {current_cls} (w=white, y=yellow)",
            f"Lines: {len(saved_lines)} | Points: {len(current_points)}",
            f"n/p=nav | s=save | q=quit | d=del line | DEL=del pt",
        ]
        for i, text in enumerate(info_lines):
            cv2.putText(vis, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Lane Annotation", vis)
        key = cv2.waitKey(1) & 0xFF

        # --- Mouse callback ---
        def on_mouse(event, x, y, flags, param):
            nonlocal current_points, current_cls, saved_lines
            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current_points) >= 2:
                    saved_lines.append((current_cls, current_points.copy()))
                    current_points = []

        cv2.setMouseCallback("Lane Annotation", on_mouse)

        # --- Keyboard ---
        if key == ord("q") or key == 27:  # q or ESC
            save_labels(label_dir, stem, saved_lines, w, h)
            print(f"Saved {stem}.jpg ({len(saved_lines)} lines). Quit.")
            break

        elif key == ord("s"):
            save_labels(label_dir, stem, saved_lines, w, h)
            print(f"Saved {stem}.jpg ({len(saved_lines)} lines)")

        elif key == ord("w") or key == ord("1"):
            current_cls = "white_lane"
        elif key == ord("y") or key == ord("2"):
            current_cls = "yellow_lane"

        elif key == ord("d"):
            if saved_lines:
                removed = saved_lines.pop()
                print(f"Deleted line: {removed[0]} ({len(removed[1])} pts)")

        elif key == 8 or key == 127:  # BACKSPACE or DEL
            if current_points:
                current_points.pop()

        elif key == ord("n"):
            save_labels(label_dir, stem, saved_lines, w, h)
            idx = (idx + 1) % len(files)
            img, stem, (h, w) = load_image(idx)
            print(f"→ {stem}.jpg ({len(saved_lines)} lines loaded)")

        elif key == ord("p"):
            save_labels(label_dir, stem, saved_lines, w, h)
            idx = (idx - 1) % len(files)
            img, stem, (h, w) = load_image(idx)
            print(f"← {stem}.jpg ({len(saved_lines)} lines loaded)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
