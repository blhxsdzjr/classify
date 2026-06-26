from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .classical_lane import draw_predictions, image_files, write_counts_csv
from .row_anchor_detector import RowAnchorParams, apply_count_constraints, detect_row_anchor_lanes
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFLD-style row-anchor response detector without model training.")
    parser.add_argument("--source", default="datasets/local_colm/images/test")
    parser.add_argument("--out", default="predictions_row_anchor.json")
    parser.add_argument("--counts-out", default="prediction_counts_row_anchor.csv")
    parser.add_argument("--save-vis", default="runs/row_anchor_vis")
    parser.add_argument("--gt-xlsx", default=None)
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--use-count-constraints", action="store_true")
    parser.add_argument("--num-rows", type=int, default=28)
    parser.add_argument("--roi-top-ratio", type=float, default=0.38)
    parser.add_argument("--peak-rel-threshold", type=float, default=0.32)
    parser.add_argument("--max-match-distance", type=float, default=80.0)
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--debug-vis", action="store_true", help="Draw labels/points instead of clean blue curves.")
    parser.add_argument("--line-thickness", type=int, default=6)
    return parser.parse_args()


def load_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        return {}
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def main() -> None:
    args = parse_args()
    import cv2

    params = RowAnchorParams(
        num_rows=args.num_rows,
        roi_top_ratio=args.roi_top_ratio,
        peak_rel_threshold=args.peak_rel_threshold,
        max_match_distance=args.max_match_distance,
        min_points=args.min_points,
    )
    counts_by_stem = load_counts(args)
    images = {}
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_files(Path(args.source)), desc="row-anchor detect"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        instances = detect_row_anchor_lanes(image, params)
        counts = counts_by_stem.get(image_path.stem)
        if args.use_count_constraints and counts is not None:
            instances = apply_count_constraints(instances, counts)
        images[image_path.name] = {
            "width": image.shape[1],
            "height": image.shape[0],
            "instances": instances,
        }
        if vis_dir:
            cv2.imwrite(
                str(vis_dir / image_path.name),
                draw_predictions(image, instances, clean=not args.debug_vis, line_thickness=args.line_thickness),
            )

    output = {
        "meta": {
            "method": "ufld_style_row_anchor_response_tracking",
            "source": args.source,
            "use_count_constraints": args.use_count_constraints,
            "note": "No YOLO and no learned detector. Row anchors are linked by continuity constraints.",
        },
        "images": images,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(images, Path(args.counts_out))
    print(f"Saved predictions to {out_path}")
    print(f"Saved counts to {args.counts_out}")


if __name__ == "__main__":
    main()
