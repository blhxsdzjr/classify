from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .classes import WHITE, YELLOW
from .classical_lane import (
    DetectorParams,
    candidate_to_prediction,
    classify_candidate,
    detect_lane_candidates,
    draw_predictions,
    image_files,
    load_color_model,
    select_with_count_constraints,
    write_counts_csv,
)
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect lane lines without YOLO, then classify white/yellow color.")
    parser.add_argument("--source", default="datasets/local_colm/images/test")
    parser.add_argument("--color-model", default="models/color_model.json")
    parser.add_argument("--out", default="predictions_classical.json")
    parser.add_argument("--counts-out", default="prediction_counts_classical.csv")
    parser.add_argument("--save-vis", default="runs/classical_vis")
    parser.add_argument("--gt-xlsx", default=None)
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--use-count-constraints", action="store_true")
    parser.add_argument("--roi-top-ratio", type=float, default=0.38)
    parser.add_argument("--min-line-length", type=int, default=45)
    parser.add_argument("--line-width", type=int, default=12)
    parser.add_argument("--min-candidate-score", type=float, default=0.03)
    return parser.parse_args()


def load_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        return {}
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def unconstrained_predictions(candidates, model) -> list[dict[str, Any]]:
    predictions = []
    for candidate in candidates:
        cls, scores = classify_candidate(candidate, model)
        predictions.append(candidate_to_prediction(candidate, cls, scores))
    return predictions


def constrained_predictions(candidates, counts: dict[str, int], model) -> list[dict[str, Any]]:
    selected = select_with_count_constraints(candidates, counts, model)
    return [
        candidate_to_prediction(candidate, cls, scores)
        for candidate, cls, scores in selected
    ]


def main() -> None:
    args = parse_args()
    import cv2

    model_path = Path(args.color_model) if args.color_model else None
    model = load_color_model(model_path)
    counts_by_stem = load_counts(args)
    params = DetectorParams(
        roi_top_ratio=args.roi_top_ratio,
        min_line_length=args.min_line_length,
        line_width=args.line_width,
        min_candidate_score=args.min_candidate_score,
    )

    files = image_files(Path(args.source))
    images: dict[str, dict[str, Any]] = {}
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(files, desc="Classical lane detection"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP cannot read: {image_path}")
            continue
        candidates = detect_lane_candidates(image, params)
        counts = counts_by_stem.get(image_path.stem)
        if args.use_count_constraints and counts is not None:
            instances = constrained_predictions(candidates, counts, model)
        else:
            instances = unconstrained_predictions(candidates, model)

        images[image_path.name] = {
            "width": image.shape[1],
            "height": image.shape[0],
            "instances": instances,
            "candidate_count": len(candidates),
            "count_targets": {
                WHITE: int(counts.get(WHITE, 0)),
                YELLOW: int(counts.get(YELLOW, 0)),
            } if counts else None,
        }
        if vis_dir:
            cv2.imwrite(str(vis_dir / image_path.name), draw_predictions(image, instances))

    output = {
        "meta": {
            "method": "classical_hough_line_detection_plus_color_learning",
            "source": str(args.source),
            "color_model": str(model_path) if model_path else None,
            "use_count_constraints": args.use_count_constraints,
            "gt_xlsx": args.gt_xlsx,
            "gt_counts": args.gt_counts,
            "notes": [
                "No YOLO or neural object detector is used.",
                "Lines are detected by classical edge/Hough processing.",
                "Colors are learned from count-level white/yellow labels when a model is provided.",
            ],
        },
        "images": images,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(images, Path(args.counts_out))
    print(f"Saved predictions to {out_path}")
    print(f"Saved counts to {args.counts_out}")
    if args.use_count_constraints:
        print("Count constraints were applied. Report this as count-supervised post-processing.")


if __name__ == "__main__":
    main()
