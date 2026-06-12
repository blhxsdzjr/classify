from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .classes import EVAL_CLASSES, names_from_yaml, normalize_class_name
from .geometry import (
    LineInstance,
    angle_diff_deg,
    bbox_iou_xyxy,
    center_distance,
    find_image_by_stem,
    read_yolo_label_file,
)
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate white/yellow lane detections with an angle threshold.")
    parser.add_argument("--pred", required=True, help="predictions.json from src.predict_yolo_lane")
    parser.add_argument("--data", default=None, help="Optional YOLO data yaml. Used to resolve names/images/labels.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-dir", default=None, help="Test image directory if --data is not enough.")
    parser.add_argument("--gt-label-dir", default=None, help="Ground-truth YOLO label directory.")
    parser.add_argument("--gt-counts", default=None, help="GT count JSON from src.prepare_local_dataset.")
    parser.add_argument("--gt-xlsx", default=None, help="Course count spreadsheet, e.g. 结果统计.xlsx.")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Use count-only GT. This cannot verify the 15-degree angle rule.",
    )
    parser.add_argument("--angle-thr", type=float, default=15.0, help="Max angle difference in degrees.")
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--max-center-dist-ratio", type=float, default=0.08)
    parser.add_argument("--min-bbox-iou", type=float, default=0.01)
    parser.add_argument(
        "--ignore-distance",
        action="store_true",
        help="Only require class and angle. Use this if the course rule is strictly angle-only.",
    )
    parser.add_argument("--out", default="metrics.json")
    return parser.parse_args()


def resolve_from_data_yaml(data_yaml: Path, split: str) -> tuple[Path | None, Path | None, dict[int, str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = names_from_yaml(data.get("names"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_value = data.get(split)
    image_dir = None
    label_dir = None
    if split_value is not None:
        image_dir = Path(split_value)
        if not image_dir.is_absolute():
            image_dir = (root / image_dir).resolve()
        parts = list(image_dir.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            label_dir = Path(*parts)
    return image_dir, label_dir, names


def load_predictions(pred_path: Path, conf_thr: float) -> dict[str, list[LineInstance]]:
    raw = json.loads(pred_path.read_text(encoding="utf-8"))
    if "images" not in raw:
        raise ValueError(f"{pred_path} must contain an 'images' object.")
    by_stem: dict[str, list[LineInstance]] = {}
    for key, payload in raw["images"].items():
        stem = Path(key).stem
        instances = []
        for item in payload.get("instances", []):
            inst = LineInstance.from_dict(item)
            inst.cls = normalize_class_name(inst.cls)
            if inst.conf >= conf_thr and inst.cls in EVAL_CLASSES:
                instances.append(inst)
        by_stem[stem] = instances
    return by_stem


def load_ground_truth(
    label_dir: Path,
    image_dir: Path,
    names: dict[int, str],
) -> tuple[dict[str, list[LineInstance]], dict[str, tuple[int, int]]]:
    if not label_dir.exists():
        raise FileNotFoundError(f"GT label dir not found: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    import cv2

    gt_by_stem: dict[str, list[LineInstance]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for label_path in sorted(label_dir.rglob("*.txt")):
        image_path = find_image_by_stem(image_dir, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image found for label stem {label_path.stem!r} under {image_dir}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        gt_by_stem[label_path.stem] = read_yolo_label_file(label_path, width, height, names)
        sizes[label_path.stem] = (width, height)
    return gt_by_stem, sizes


def can_match(
    pred: LineInstance,
    gt: LineInstance,
    image_diag: float,
    args: argparse.Namespace,
) -> tuple[bool, float, dict[str, float]]:
    angle_diff = angle_diff_deg(pred.angle_deg, gt.angle_deg)
    iou = bbox_iou_xyxy(pred.bbox, gt.bbox)
    dist = center_distance(pred.bbox, gt.bbox)
    max_dist = args.max_center_dist_ratio * image_diag

    angle_ok = angle_diff <= args.angle_thr
    distance_ok = args.ignore_distance or dist <= max_dist or iou >= args.min_bbox_iou
    ok = pred.cls == gt.cls and angle_ok and distance_ok

    norm_dist = dist / max(max_dist, 1e-6)
    score = angle_diff + 3.0 * norm_dist - iou
    details = {"angle_diff": angle_diff, "bbox_iou": iou, "center_dist": dist}
    return ok, score, details


def match_image(
    preds: list[LineInstance],
    gts: list[LineInstance],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    width, height = image_size
    image_diag = math.hypot(width, height)
    matches: list[dict[str, Any]] = []
    used_gt: set[int] = set()

    for pred_idx, pred in sorted(enumerate(preds), key=lambda item: item[1].conf, reverse=True):
        best: tuple[float, int, dict[str, float]] | None = None
        for gt_idx, gt in enumerate(gts):
            if gt_idx in used_gt:
                continue
            ok, score, details = can_match(pred, gt, image_diag, args)
            if not ok:
                continue
            if best is None or score < best[0]:
                best = (score, gt_idx, details)
        if best is not None:
            _, gt_idx, details = best
            used_gt.add(gt_idx)
            matches.append(
                {
                    "pred_index": pred_idx,
                    "gt_index": gt_idx,
                    "class": pred.cls,
                    **details,
                }
            )
    return matches


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def print_table(metrics: dict[str, dict[str, float]]) -> None:
    header = f"{'class':<14}{'detected':>10}{'correct':>10}{'gt':>10}{'precision':>12}{'recall':>10}{'f1':>10}"
    print(header)
    print("-" * len(header))
    for cls, row in metrics.items():
        print(
            f"{cls:<14}"
            f"{int(row['detected']):>10}"
            f"{int(row['correct']):>10}"
            f"{int(row['gt']):>10}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
        )


def empty_counts() -> dict[str, int]:
    return {cls: 0 for cls in EVAL_CLASSES}


def load_gt_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        raise ValueError("Count-only evaluation needs --gt-counts or --gt-xlsx.")
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def evaluate_count_only(args: argparse.Namespace, pred_path: Path) -> None:
    preds_by_stem = load_predictions(pred_path, args.conf_thr)
    gt_counts_by_stem = load_gt_counts(args)

    counts = {
        cls: {"detected": 0, "correct": 0, "gt": 0}
        for cls in EVAL_CLASSES
    }
    image_reports: dict[str, Any] = {}

    all_stems = sorted(set(gt_counts_by_stem) | set(preds_by_stem))
    for stem in all_stems:
        preds = preds_by_stem.get(stem, [])
        pred_counts = empty_counts()
        for pred in preds:
            if pred.cls in pred_counts:
                pred_counts[pred.cls] += 1
        gt_raw = gt_counts_by_stem.get(stem, {})
        gt_counts = {cls: int(gt_raw.get(cls, 0)) for cls in EVAL_CLASSES}
        correct_counts = {
            cls: min(pred_counts[cls], gt_counts[cls])
            for cls in EVAL_CLASSES
        }

        for cls in EVAL_CLASSES:
            counts[cls]["detected"] += pred_counts[cls]
            counts[cls]["gt"] += gt_counts[cls]
            counts[cls]["correct"] += correct_counts[cls]

        image_reports[stem] = {
            "pred_counts": pred_counts,
            "gt_counts": gt_counts,
            "correct_counts": correct_counts,
        }

    metrics: dict[str, dict[str, float]] = {}
    total = {"detected": 0, "correct": 0, "gt": 0}
    for cls, row in counts.items():
        precision = safe_div(row["correct"], row["detected"])
        recall = safe_div(row["correct"], row["gt"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics[cls] = {**row, "precision": precision, "recall": recall, "f1": f1}
        for key in total:
            total[key] += row[key]

    precision = safe_div(total["correct"], total["detected"])
    recall = safe_div(total["correct"], total["gt"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    metrics["overall"] = {**total, "precision": precision, "recall": recall, "f1": f1}

    output = {
        "settings": {
            "mode": "count_only",
            "conf_thr": args.conf_thr,
            "note": "Count-only GT cannot verify the 15-degree angle rule.",
        },
        "metrics": metrics,
        "images": image_reports,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WARNING: count-only GT cannot verify the 15-degree angle rule.")
    print_table(metrics)
    print(f"Saved metrics to {out_path}")


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)

    if args.count_only or args.gt_counts or args.gt_xlsx:
        evaluate_count_only(args, pred_path)
        return

    image_dir = Path(args.image_dir).resolve() if args.image_dir else None
    label_dir = Path(args.gt_label_dir).resolve() if args.gt_label_dir else None
    names: dict[int, str] = {}

    if args.data:
        data_image_dir, data_label_dir, names = resolve_from_data_yaml(Path(args.data), args.split)
        image_dir = image_dir or data_image_dir
        label_dir = label_dir or data_label_dir

    if image_dir is None or label_dir is None:
        raise ValueError("Please provide --data, or both --image-dir and --gt-label-dir.")

    preds_by_stem = load_predictions(pred_path, args.conf_thr)
    gt_by_stem, sizes = load_ground_truth(label_dir, image_dir, names)

    counts = {
        cls: {"detected": 0, "correct": 0, "gt": 0}
        for cls in EVAL_CLASSES
    }
    image_reports: dict[str, Any] = {}

    all_stems = sorted(set(gt_by_stem) | set(preds_by_stem))
    for stem in all_stems:
        preds = preds_by_stem.get(stem, [])
        gts = gt_by_stem.get(stem, [])
        size = sizes.get(stem)
        if size is None:
            # A prediction without GT is counted as detected but cannot be correct.
            size = (1, 1)

        for cls in EVAL_CLASSES:
            counts[cls]["detected"] += sum(1 for pred in preds if pred.cls == cls)
            counts[cls]["gt"] += sum(1 for gt in gts if gt.cls == cls)

        matches = match_image(preds, gts, size, args)
        for match in matches:
            counts[match["class"]]["correct"] += 1

        image_reports[stem] = {
            "pred_count": len(preds),
            "gt_count": len(gts),
            "matches": matches,
        }

    metrics: dict[str, dict[str, float]] = {}
    total = {"detected": 0, "correct": 0, "gt": 0}
    for cls, row in counts.items():
        precision = safe_div(row["correct"], row["detected"])
        recall = safe_div(row["correct"], row["gt"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics[cls] = {
            **row,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for key in total:
            total[key] += row[key]

    precision = safe_div(total["correct"], total["detected"])
    recall = safe_div(total["correct"], total["gt"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    metrics["overall"] = {
        **total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    output = {
        "settings": {
            "angle_thr": args.angle_thr,
            "conf_thr": args.conf_thr,
            "max_center_dist_ratio": args.max_center_dist_ratio,
            "min_bbox_iou": args.min_bbox_iou,
            "ignore_distance": args.ignore_distance,
        },
        "metrics": metrics,
        "images": image_reports,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(metrics)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
