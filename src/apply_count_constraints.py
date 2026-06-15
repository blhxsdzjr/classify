from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .classes import WHITE, YELLOW, normalize_class_name
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use white/yellow count annotations to constrain prediction results."
    )
    parser.add_argument("--pred", required=True, help="Input predictions.json from src.predict_yolo_lane.")
    parser.add_argument("--out", default="predictions_count_constrained.json")
    parser.add_argument("--counts-out", default="prediction_counts_constrained.csv")
    parser.add_argument("--gt-xlsx", default=None, help="Count spreadsheet, e.g. 结果统计.xlsx.")
    parser.add_argument("--gt-counts", default=None, help="Count JSON from src.prepare_local_dataset.")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep predictions for images that have no count GT. Default drops only unmatched extras on matched images.",
    )
    parser.add_argument(
        "--class-order",
        choices=("rare-first", "yellow-first", "white-first"),
        default="rare-first",
        help="Order used when assigning classes to candidates.",
    )
    return parser.parse_args()


def load_gt_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        raise ValueError("Please provide --gt-xlsx or --gt-counts.")
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def color_scores(instance: dict[str, Any]) -> dict[str, float]:
    """Score how likely a candidate is white/yellow.

    The prediction script writes white_fraction/yellow_fraction when HSV mode is
    used. If those fields are absent, fall back to the existing predicted class.
    """
    conf = safe_float(instance.get("conf"), 1.0)
    white_fraction = safe_float(instance.get("white_fraction"), 0.0)
    yellow_fraction = safe_float(instance.get("yellow_fraction"), 0.0)
    cls = normalize_class_name(instance.get("class", ""))

    if white_fraction == 0.0 and yellow_fraction == 0.0:
        white_fraction = 1.0 if cls == WHITE else 0.0
        yellow_fraction = 1.0 if cls == YELLOW else 0.0

    white_bonus = 0.15 if cls == WHITE else 0.0
    yellow_bonus = 0.15 if cls == YELLOW else 0.0
    white_score = conf * max(0.0, white_fraction + white_bonus - 0.05 * yellow_fraction)
    yellow_score = conf * max(0.0, yellow_fraction + yellow_bonus - 0.05 * white_fraction)

    return {
        WHITE: white_score,
        YELLOW: yellow_score,
        "candidate": conf * (max(white_fraction, yellow_fraction) + 0.05),
    }


def ordered_classes(target_counts: dict[str, int], order: str) -> list[str]:
    if order == "yellow-first":
        return [YELLOW, WHITE]
    if order == "white-first":
        return [WHITE, YELLOW]
    return sorted([WHITE, YELLOW], key=lambda cls: (target_counts.get(cls, 0), cls))


def constrain_instances(
    instances: list[dict[str, Any]],
    target_counts: dict[str, int],
    *,
    min_conf: float,
    class_order: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        (idx, inst, color_scores(inst))
        for idx, inst in enumerate(instances)
        if safe_float(inst.get("conf"), 1.0) >= min_conf
    ]

    selected_indices: set[int] = set()
    selected: list[dict[str, Any]] = []
    assignments: dict[int, str] = {}

    for cls in ordered_classes(target_counts, class_order):
        need = max(0, int(target_counts.get(cls, 0)))
        if need == 0:
            continue
        available = [
            (scores[cls], scores["candidate"], idx, inst)
            for idx, inst, scores in candidates
            if idx not in selected_indices
        ]
        available.sort(reverse=True, key=lambda item: (item[0], item[1], safe_float(item[3].get("conf"), 0.0)))
        for _, _, idx, inst in available[:need]:
            selected_indices.add(idx)
            assignments[idx] = cls
            constrained = deepcopy(inst)
            constrained["class"] = cls
            constrained["count_constraint"] = {
                "target_class": cls,
                "source_class": normalize_class_name(inst.get("class", "")),
                "white_score": color_scores(inst)[WHITE],
                "yellow_score": color_scores(inst)[YELLOW],
            }
            selected.append(constrained)

    selected.sort(key=lambda inst: safe_float(inst.get("conf"), 0.0), reverse=True)
    pred_counts = {
        WHITE: sum(1 for inst in selected if normalize_class_name(inst.get("class", "")) == WHITE),
        YELLOW: sum(1 for inst in selected if normalize_class_name(inst.get("class", "")) == YELLOW),
    }
    report = {
        "candidate_count": len(candidates),
        "kept_count": len(selected),
        "dropped_count": max(0, len(candidates) - len(selected)),
        "target_counts": {
            WHITE: int(target_counts.get(WHITE, 0)),
            YELLOW: int(target_counts.get(YELLOW, 0)),
        },
        "pred_counts": pred_counts,
        "shortage": {
            WHITE: max(0, int(target_counts.get(WHITE, 0)) - pred_counts[WHITE]),
            YELLOW: max(0, int(target_counts.get(YELLOW, 0)) - pred_counts[YELLOW]),
        },
        "assignments": assignments,
    }
    return selected, report


def write_counts_csv(images: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "车道线数", "白线数", "黄线数"])
        for filename in sorted(images, key=lambda name: (Path(name).stem.zfill(8), name)):
            instances = images[filename].get("instances", [])
            white = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == WHITE)
            yellow = sum(1 for inst in instances if normalize_class_name(inst.get("class", "")) == YELLOW)
            writer.writerow([filename, white + yellow, white, yellow])


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_counts = load_gt_counts(args)

    output = deepcopy(predictions)
    output.setdefault("meta", {})
    output["meta"]["count_constraints"] = {
        "enabled": True,
        "gt_xlsx": args.gt_xlsx,
        "gt_counts": args.gt_counts,
        "min_conf": args.min_conf,
        "class_order": args.class_order,
        "note": "This uses count-level GT as a constraint; do not report it as unconstrained model performance.",
    }

    reports: dict[str, Any] = {}
    constrained_images: dict[str, dict[str, Any]] = {}
    matched = 0

    for filename, payload in predictions.get("images", {}).items():
        stem = Path(filename).stem
        target = gt_counts.get(stem)
        if target is None:
            if args.keep_unmatched:
                constrained_images[filename] = payload
            reports[stem] = {"matched_gt": False}
            continue

        matched += 1
        instances = payload.get("instances", [])
        selected, report = constrain_instances(
            instances,
            target,
            min_conf=args.min_conf,
            class_order=args.class_order,
        )
        new_payload = deepcopy(payload)
        new_payload["instances"] = selected
        constrained_images[filename] = new_payload
        reports[stem] = {"matched_gt": True, **report}

    output["images"] = constrained_images
    output["constraint_report"] = reports

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(constrained_images, Path(args.counts_out))

    print(f"Loaded predictions: {len(predictions.get('images', {}))} images")
    print(f"Matched count GT: {matched}/{len(predictions.get('images', {}))} images")
    print(f"Saved constrained predictions to {out_path}")
    print(f"Saved constrained counts to {args.counts_out}")
    print("NOTE: constrained results use count-level GT and should be described as weakly supervised/post-processed.")


if __name__ == "__main__":
    main()
