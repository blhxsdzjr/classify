from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WHITE_ID = 0
YELLOW_ID = 1
CLASS_NAMES = {
    WHITE_ID: "white_lane",
    YELLOW_ID: "yellow_lane",
}
DRAW_COLORS = {
    WHITE_ID: (0, 0, 255),
    YELLOW_ID: (255, 0, 0),
}


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class LaneLine:
    cls_id: int
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class AnnotatorState:
    image: np.ndarray
    image_path: Path
    label_path: Path
    lanes: list[LaneLine]
    current_points: list[tuple[float, float]] = field(default_factory=list)
    current_cls: int = WHITE_ID
    display_scale: float = 1.0
    dirty: bool = False
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV lane-line annotation tool.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--label-dir", default="datasets/local_colm/labels/test")
    parser.add_argument("--start", default=None, help="Optional image filename/stem to start from.")
    parser.add_argument("--max-window-width", type=int, default=1600)
    parser.add_argument("--max-window-height", type=int, default=900)
    return parser.parse_args()


def image_files(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name,
    )


def label_path_for(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    rel = image_path.relative_to(image_dir)
    return (label_dir / rel).with_suffix(".txt")


def load_labels(label_path: Path, width: int, height: int) -> list[LaneLine]:
    if not label_path.exists():
        return []
    lanes: list[LaneLine] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 5 or len(parts[1:]) % 2:
            raise ValueError(f"{label_path}:{line_no}: expected class_id and at least two normalized points")
        cls_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        points = []
        for x_norm, y_norm in zip(values[0::2], values[1::2]):
            points.append((x_norm * width, y_norm * height))
        lanes.append(LaneLine(cls_id=cls_id, points=points))
    return lanes


def save_labels(label_path: Path, lanes: list[LaneLine], width: int, height: int) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for lane in lanes:
        if len(lane.points) < 2:
            continue
        values = []
        for x, y in lane.points:
            values.append(f"{np.clip(x / width, 0.0, 1.0):.6f}")
            values.append(f"{np.clip(y / height, 0.0, 1.0):.6f}")
        lines.append(f"{lane.cls_id} " + " ".join(values))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def window_scale(image: np.ndarray, max_width: int, max_height: int) -> float:
    height, width = image.shape[:2]
    return min(1.0, max_width / max(width, 1), max_height / max(height, 1))


def to_display(point: tuple[float, float], scale: float) -> tuple[int, int]:
    x, y = point
    return int(round(x * scale)), int(round(y * scale))


def from_display(x: int, y: int, scale: float, image: np.ndarray) -> tuple[float, float]:
    height, width = image.shape[:2]
    return float(np.clip(x / scale, 0, width - 1)), float(np.clip(y / scale, 0, height - 1))


def draw_polyline(canvas: np.ndarray, points: list[tuple[float, float]], scale: float, color: tuple[int, int, int], closed: bool = False) -> None:
    if not points:
        return
    draw_points = np.asarray([to_display(pt, scale) for pt in points], dtype=np.int32).reshape(-1, 1, 2)
    for pt in draw_points.reshape(-1, 2):
        cv2.circle(canvas, tuple(int(v) for v in pt), 5, color, -1, cv2.LINE_AA)
    if len(points) >= 2:
        cv2.polylines(canvas, [draw_points], isClosed=closed, color=color, thickness=4, lineType=cv2.LINE_AA)


def render(state: AnnotatorState, index: int, total: int) -> np.ndarray:
    scale = state.display_scale
    if scale != 1.0:
        canvas = cv2.resize(state.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        canvas = state.image.copy()

    overlay = canvas.copy()
    for lane_idx, lane in enumerate(state.lanes, start=1):
        color = DRAW_COLORS.get(lane.cls_id, (0, 255, 255))
        draw_polyline(overlay, lane.points, scale, color)
        if lane.points:
            x, y = to_display(lane.points[0], scale)
            cv2.putText(overlay, str(lane_idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    draw_polyline(canvas, state.current_points, scale, DRAW_COLORS[state.current_cls])
    header = (
        f"{index + 1}/{total} {state.image_path.name} | "
        f"class={CLASS_NAMES[state.current_cls]} | saved={not state.dirty}"
    )
    help_text = "L-click add point | Enter/R-click finish | w/y class | u undo point | z undo lane | s save | n/p nav | d clear | q quit"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 64), (20, 20, 20), -1)
    cv2.putText(canvas, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, help_text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    if state.message:
        cv2.rectangle(canvas, (0, canvas.shape[0] - 34), (canvas.shape[1], canvas.shape[0]), (20, 20, 20), -1)
        cv2.putText(canvas, state.message, (12, canvas.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def finish_current_line(state: AnnotatorState) -> None:
    if len(state.current_points) < 2:
        state.message = "Need at least two points to finish a lane."
        return
    state.lanes.append(LaneLine(cls_id=state.current_cls, points=state.current_points[:]))
    state.current_points.clear()
    state.dirty = True
    state.message = "Lane added."


def load_state(image_path: Path, image_dir: Path, label_dir: Path, args: argparse.Namespace) -> AnnotatorState:
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    label_path = label_path_for(image_path, image_dir, label_dir)
    lanes = load_labels(label_path, width, height)
    return AnnotatorState(
        image=image,
        image_path=image_path,
        label_path=label_path,
        lanes=lanes,
        display_scale=window_scale(image, args.max_window_width, args.max_window_height),
        dirty=False,
        message=f"Loaded {len(lanes)} lanes from {label_path.name if label_path.exists() else 'new label'}.",
    )


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    label_dir = Path(args.label_dir).resolve()
    images = image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")

    index = 0
    if args.start:
        start = Path(args.start).stem
        for idx, image_path in enumerate(images):
            if image_path.stem == start or image_path.name == args.start:
                index = idx
                break

    window = "manual_lane_annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state = load_state(images[index], image_dir, label_dir, args)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal state
        if event == cv2.EVENT_LBUTTONDOWN:
            state.current_points.append(from_display(x, y, state.display_scale, state.image))
            state.message = f"Point added ({len(state.current_points)} in current lane)."
        elif event == cv2.EVENT_RBUTTONDOWN:
            finish_current_line(state)

    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, render(state, index, len(images)))
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):
            if state.dirty:
                height, width = state.image.shape[:2]
                save_labels(state.label_path, state.lanes, width, height)
            break
        if key in (13, 10):
            finish_current_line(state)
        elif key == ord("w"):
            state.current_cls = WHITE_ID
            state.message = "Current class: white_lane."
        elif key == ord("y"):
            state.current_cls = YELLOW_ID
            state.message = "Current class: yellow_lane."
        elif key == ord("u"):
            if state.current_points:
                state.current_points.pop()
                state.message = "Undid one point."
        elif key == ord("z"):
            if state.current_points:
                state.current_points.clear()
                state.message = "Cleared current unfinished lane."
            elif state.lanes:
                state.lanes.pop()
                state.dirty = True
                state.message = "Removed last saved lane in this image."
        elif key == ord("d"):
            state.lanes.clear()
            state.current_points.clear()
            state.dirty = True
            state.message = "Cleared all labels in this image."
        elif key == ord("s"):
            height, width = state.image.shape[:2]
            save_labels(state.label_path, state.lanes, width, height)
            state.dirty = False
            state.message = f"Saved {len(state.lanes)} lanes to {state.label_path}."
        elif key in (ord("n"), ord("p")):
            if state.dirty:
                height, width = state.image.shape[:2]
                save_labels(state.label_path, state.lanes, width, height)
            step = 1 if key == ord("n") else -1
            index = (index + step) % len(images)
            state = load_state(images[index], image_dir, label_dir, args)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
