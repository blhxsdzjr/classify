#!/usr/bin/env python3
"""Matplotlib-based manual lane-line annotation tool (works over SSH X11 forwarding).

Draw lane lines with mouse clicks. Each lane = click start + end points.

Controls:
  LEFT CLICK     — add point (first click = line start, second click = line end)
  w / y key      — switch to white_lane / yellow_lane class
  d key          — delete last line
  n / p key      — next / previous image
  s key          — save current image annotations
  q / ESC        — quit (auto-saves)

Lines saved as YOLO segmentation format in --label-dir.
X11 forwarding required: ssh -X user@host
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # Try TkAgg first (most common over X11)
import matplotlib.pyplot as plt
import numpy as np

CLASS_COLORS = {"white_lane": "lime", "yellow_lane": "gold"}


def parse_args():
    p = argparse.ArgumentParser(description="Manual lane-line annotation (matplotlib).")
    p.add_argument("--image-dir", default="datasets/local_colm/images/test")
    p.add_argument("--label-dir", default="datasets/local_colm/labels/test")
    p.add_argument("--start", type=int, default=0)
    return p.parse_args()


def yolo_polygon_str(pts, w, h):
    """Pixel points → YOLO segmentation format string."""
    if len(pts) < 2:
        return None
    norm = []
    for px, py in pts:
        norm.append(f"{px / w:.6f}")
        norm.append(f"{py / h:.6f}")
    return " ".join(norm)


def save_labels(label_dir, stem, lines, w, h):
    label_dir.mkdir(parents=True, exist_ok=True)
    yolo_lines = []
    for cls, pts in lines:
        cls_id = 0 if cls == "white_lane" else 1
        s = yolo_polygon_str(pts, w, h)
        if s:
            yolo_lines.append(f"{cls_id} {s}")
    (label_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")


def get_files(d):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in exts)


class Annotator:
    def __init__(self, image_dir, label_dir, start=0):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.files = get_files(image_dir)
        self.idx = max(0, min(start, len(self.files) - 1))
        self.current_cls = "white_lane"
        self.current_pt = None   # first point of current line (waiting for second)
        self.saved_lines = []    # [(cls, [(x1,y1),(x2,y2),...]), ...]

        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Lane Annotation")
        self.load_image()
        self.setup_callbacks()
        self.redraw()

    def load_image(self):
        path = str(self.files[self.idx])
        self.img = plt.imread(path)
        # BGR→RGB if needed (cv2 images are BGR, but plt.imread loads correctly)
        self.h, self.w = self.img.shape[:2]
        self.stem = self.files[self.idx].stem

        # Load existing labels
        label_path = self.label_dir / f"{self.stem}.txt"
        self.saved_lines = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls_id = int(float(parts[0]))
                cls = "white_lane" if cls_id == 0 else "yellow_lane"
                coords = [float(x) for x in parts[1:]]
                pts = [(int(coords[i] * self.w), int(coords[i + 1] * self.h))
                       for i in range(0, len(coords), 2)]
                if len(pts) >= 2:
                    self.saved_lines.append((cls, pts))
        self.current_pt = None
        print(f"Loaded {self.stem}.jpg ({len(self.saved_lines)} existing lines)")

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.img)

        # Draw saved lines
        for cls, pts in self.saved_lines:
            color = CLASS_COLORS.get(cls, "white")
            xs, ys = zip(*pts)
            self.ax.plot(xs, ys, color=color, linewidth=3, marker="o", markersize=5)
            self.ax.text(xs[0] + 8, ys[0] - 5, cls, color=color, fontsize=9, weight="bold")

        # Draw current first point
        if self.current_pt is not None:
            self.ax.plot(self.current_pt[0], self.current_pt[1], "ro", markersize=8)

        # Title
        title = (f"[{self.idx + 1}/{len(self.files)}] {self.stem}.jpg | "
                 f"Class: {self.current_cls} | Lines: {len(self.saved_lines)} | "
                 f"n/p=nav w/y=class s=save d=del q=quit")
        self.ax.set_title(title, fontsize=11)
        self.ax.axis("off")
        self.fig.canvas.draw()

    def setup_callbacks(self):
        def on_click(event):
            if event.inaxes != self.ax or event.xdata is None:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            x = max(0, min(self.w - 1, x))
            y = max(0, min(self.h - 1, y))

            if self.current_pt is None:
                # First click: set line start
                self.current_pt = (x, y)
            else:
                # Second click: finish line
                pts = [self.current_pt, (x, y)]
                self.saved_lines.append((self.current_cls, pts))
                self.current_pt = None
            self.redraw()

        def on_key(event):
            if event.key in ("w", "1"):
                self.current_cls = "white_lane"
            elif event.key in ("y", "2"):
                self.current_cls = "yellow_lane"
            elif event.key == "d":
                if self.saved_lines:
                    self.saved_lines.pop()
                elif self.current_pt:
                    self.current_pt = None
            elif event.key == "n":
                save_labels(self.label_dir, self.stem, self.saved_lines, self.w, self.h)
                self.idx = (self.idx + 1) % len(self.files)
                self.load_image()
            elif event.key == "p":
                save_labels(self.label_dir, self.stem, self.saved_lines, self.w, self.h)
                self.idx = (self.idx - 1) % len(self.files)
                self.load_image()
            elif event.key == "s":
                save_labels(self.label_dir, self.stem, self.saved_lines, self.w, self.h)
                print(f"  Saved {self.stem}.jpg: {len(self.saved_lines)} lines")
            elif event.key in ("q", "escape"):
                save_labels(self.label_dir, self.stem, self.saved_lines, self.w, self.h)
                print(f"Saved. Quit.")
                plt.close()
                return
            elif event.key == "backspace":
                if self.current_pt:
                    self.current_pt = None
            self.redraw()

        self.fig.canvas.mpl_connect("button_press_event", on_click)
        self.fig.canvas.mpl_connect("key_press_event", on_key)


def main():
    args = parse_args()
    if not get_files(args.image_dir):
        print(f"No images in {args.image_dir}")
        return

    print(f"Found {len(get_files(args.image_dir))} images.")
    print("LEFT click = start/end line | w/y = class | d = delete | n/p = nav | s = save | q = quit")
    print("NOTE: needs X11 forwarding: ssh -X user@host")
    print()

    Annotator(args.image_dir, args.label_dir, args.start)
    plt.show()


if __name__ == "__main__":
    main()
