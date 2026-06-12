from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .classes import EVAL_CLASSES, LANE_LINE, UNKNOWN, class_id_to_name, is_eval_class, normalize_class_name
from .color_classifier import classify_lane_color
from .geometry import LineInstance, fit_line_from_points, line_from_bbox_xyxy, points_from_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO lane inference and classify white/yellow lanes.")
    parser.add_argument("--weights", required=True, help="YOLO weights, e.g. runs/segment/colm_lane/weights/best.pt")
    parser.add_argument("--source", required=True, help="Image, directory, or glob accepted by Ultralytics.")
    parser.add_argument("--out", default="predictions.json", help="Output JSON path.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--class-mode",
        choices=("auto", "model", "hsv"),
        default="hsv",
        help="hsv is recommended for the one-class lane_line model; auto uses model class if it is white/yellow.",
    )
    parser.add_argument("--hsv-refine", action="store_true", help="Use HSV color result even for white/yellow model classes.")
    parser.add_argument("--keep-unknown", action="store_true", help="Keep detections whose color cannot be decided.")
    parser.add_argument("--save-vis", default=None, help="Optional directory for visualization images.")
    parser.add_argument("--counts-out", default=None, help="Optional CSV with filename/lane/white/yellow counts.")
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def image_files_from_source(source: str) -> list[Path] | None:
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return None


def mask_for_box(shape: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def decide_class(
    image_bgr: np.ndarray,
    region_mask: np.ndarray,
    model_class: str,
    class_mode: str,
    hsv_refine: bool,
) -> tuple[str, dict]:
    normalized_model_class = normalize_class_name(model_class)
    need_hsv = class_mode == "hsv" or hsv_refine or (
        class_mode == "auto" and normalized_model_class not in EVAL_CLASSES
    )
    if class_mode == "model" and normalized_model_class in EVAL_CLASSES:
        return normalized_model_class, {}
    if class_mode == "model":
        return normalized_model_class, {}

    if need_hsv:
        decision = classify_lane_color(image_bgr, region_mask)
        return decision.cls, {
            "color_score": decision.score,
            "white_fraction": decision.white_fraction,
            "yellow_fraction": decision.yellow_fraction,
        }

    return normalized_model_class, {}


def draw_instance(image: np.ndarray, inst: LineInstance) -> None:
    import cv2

    color = (240, 240, 240) if inst.cls == "white_lane" else (0, 220, 255)
    if inst.cls == UNKNOWN:
        color = (160, 160, 160)
    p0 = tuple(int(round(v)) for v in inst.endpoints[0])
    p1 = tuple(int(round(v)) for v in inst.endpoints[1])
    x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    cv2.line(image, p0, p1, color, 2)
    label = f"{inst.cls} {inst.conf:.2f} {inst.angle_deg:.1f}"
    cv2.putText(image, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def write_counts_csv(images: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "车道线数", "白线数", "黄线数"])
        for filename in sorted(images):
            instances = images[filename].get("instances", [])
            white = sum(1 for inst in instances if inst.get("class") == "white_lane")
            yellow = sum(1 for inst in instances if inst.get("class") == "yellow_lane")
            writer.writerow([filename, white + yellow, white, yellow])


def predict_one_result(result, model_names: dict[int, str], args: argparse.Namespace) -> tuple[str, dict]:
    import cv2

    image_path = Path(result.path)
    image_bgr = result.orig_img
    height, width = image_bgr.shape[:2]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image_path.name, {"width": width, "height": height, "instances": []}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    masks = None
    if result.masks is not None and result.masks.data is not None:
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:3] != (height, width):
            resized = []
            for mask in masks:
                resized.append(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
            masks = np.asarray(resized)

    instances: list[LineInstance] = []
    vis = image_bgr.copy() if args.save_vis else None

    for idx in range(len(xyxy)):
        region_mask = masks[idx] > 0.5 if masks is not None else mask_for_box((height, width), xyxy[idx])
        points = points_from_mask(region_mask)
        fitted = fit_line_from_points(points)
        if fitted is None:
            angle, endpoints, bbox = line_from_bbox_xyxy(xyxy[idx])
        else:
            angle, endpoints, bbox = fitted

        source_class = class_id_to_name(int(class_ids[idx]), model_names)
        cls, color_info = decide_class(image_bgr, region_mask, source_class, args.class_mode, args.hsv_refine)
        if not args.keep_unknown and (cls == UNKNOWN or (not is_eval_class(cls) and cls != LANE_LINE)):
            continue
        if cls == LANE_LINE and not args.keep_unknown:
            continue

        inst = LineInstance(
            cls=cls,
            conf=float(confs[idx]),
            angle_deg=angle,
            endpoints=endpoints,
            bbox=bbox,
            source_class=source_class,
            color_score=color_info.get("color_score"),
            white_fraction=color_info.get("white_fraction"),
            yellow_fraction=color_info.get("yellow_fraction"),
        )
        instances.append(inst)
        if vis is not None:
            draw_instance(vis, inst)

    if vis is not None:
        save_dir = Path(args.save_vis)
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / image_path.name), vis)

    return image_path.name, {
        "width": width,
        "height": height,
        "instances": [inst.to_dict() for inst in instances],
    }


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model_names = {int(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {
        i: str(v) for i, v in enumerate(model.names)
    }

    predict_kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    files = image_files_from_source(args.source)
    total = len(files) if files is not None else None
    images: dict[str, dict] = {}

    for result in tqdm(model.predict(**predict_kwargs), total=total, desc="Predicting"):
        key, payload = predict_one_result(result, model_names, args)
        images[key] = payload

    output = {
        "meta": {
            "weights": args.weights,
            "source": args.source,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "class_mode": args.class_mode,
            "hsv_refine": args.hsv_refine,
            "model_names": model_names,
        },
        "images": images,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions to {out_path}")
    if args.counts_out:
        counts_path = Path(args.counts_out)
        write_counts_csv(images, counts_path)
        print(f"Saved count CSV to {counts_path}")


if __name__ == "__main__":
    main()
