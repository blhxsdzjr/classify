from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .classes import WHITE, YELLOW
from .classical_lane import (
    DetectorParams,
    detect_lane_candidates,
    fallback_color_scores,
    image_files,
    select_with_count_constraints,
)
from .ufld_model import UFLDConfig
from .xlsx_counts import read_count_json, read_count_xlsx


COLOR_TO_ID = {WHITE: 0, YELLOW: 1, "none": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UFLD-style row-grid labels from weak line candidates.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx")
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--out-dir", default="datasets/local_colm/ufld_labels/test")
    parser.add_argument("--index-out", default="datasets/local_colm/ufld_test_index.json")
    parser.add_argument("--image-width", type=int, default=800)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--num-rows", type=int, default=18)
    parser.add_argument("--num-grids", type=int, default=100)
    parser.add_argument("--max-lanes", type=int, default=6)
    parser.add_argument("--roi-top-ratio", type=float, default=0.38)
    parser.add_argument("--min-line-length", type=int, default=45)
    parser.add_argument("--line-width", type=int, default=12)
    return parser.parse_args()


def load_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    else:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def line_x_at_y(endpoints: list[list[float]], y: float) -> float | None:
    (x1, y1), (x2, y2) = endpoints
    if abs(y2 - y1) < 1e-6:
        return None
    lo, hi = sorted([y1, y2])
    if y < lo or y > hi:
        return None
    t = (y - y1) / (y2 - y1)
    return float(x1 + t * (x2 - x1))


def grid_label_from_x(x: float, image_width: int, num_grids: int) -> int:
    grid = int(round(x / max(image_width - 1, 1) * (num_grids - 1)))
    return int(np.clip(grid, 0, num_grids - 1))


def choose_candidates(image_bgr: np.ndarray, counts: dict[str, int], params: DetectorParams):
    candidates = detect_lane_candidates(image_bgr, params)
    # Use count supervision to choose exactly white/yellow target counts.
    return select_with_count_constraints(candidates, counts, model=None)


def make_label(
    image_bgr: np.ndarray,
    counts: dict[str, int],
    cfg: UFLDConfig,
    params: DetectorParams,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    selected = choose_candidates(image_bgr, counts, params)
    selected = sorted(selected, key=lambda item: max(pt[0] for pt in item[0].endpoints))[: cfg.max_lanes]
    selected = sorted(selected, key=lambda item: np.mean([pt[0] for pt in item[0].endpoints]))

    row_targets = np.full((cfg.max_lanes, cfg.num_rows), cfg.no_lane_index, dtype=np.int64)
    color_targets = np.full((cfg.max_lanes,), COLOR_TO_ID["none"], dtype=np.int64)
    exists = np.zeros((cfg.max_lanes,), dtype=np.float32)
    row_anchors_original = [row / cfg.image_height * h for row in cfg.row_anchors()]

    meta_lanes = []
    for lane_idx, (candidate, cls, scores) in enumerate(selected):
        if lane_idx >= cfg.max_lanes:
            break
        exists[lane_idx] = 1.0
        color_targets[lane_idx] = COLOR_TO_ID[cls]
        valid_rows = 0
        for row_idx, y in enumerate(row_anchors_original):
            x = line_x_at_y(candidate.endpoints, y)
            if x is None:
                continue
            row_targets[lane_idx, row_idx] = grid_label_from_x(x, w, cfg.num_grids)
            valid_rows += 1
        meta_lanes.append(
            {
                "class": cls,
                "valid_rows": valid_rows,
                "candidate_score": candidate.candidate_score,
                "endpoints": candidate.endpoints,
                "white_score": scores.get(WHITE, 0.0),
                "yellow_score": scores.get(YELLOW, 0.0),
            }
        )

    return {
        "row_targets": row_targets,
        "color_targets": color_targets,
        "exists": exists,
        "meta": {
            "image_width": w,
            "image_height": h,
            "row_anchors_original": row_anchors_original,
            "lanes": meta_lanes,
        },
    }


def save_label(path: Path, label: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_targets=label["row_targets"],
        color_targets=label["color_targets"],
        exists=label["exists"],
        meta=json.dumps(label["meta"], ensure_ascii=False),
    )


def main() -> None:
    args = parse_args()
    import cv2

    cfg = UFLDConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        num_rows=args.num_rows,
        num_grids=args.num_grids,
        max_lanes=args.max_lanes,
    )
    params = DetectorParams(
        roi_top_ratio=args.roi_top_ratio,
        min_line_length=args.min_line_length,
        line_width=args.line_width,
    )
    counts_by_stem = load_counts(args)
    out_dir = Path(args.out_dir)
    index_rows = []

    files = image_files(Path(args.image_dir))
    for image_path in tqdm(files, desc="Generating UFLD labels"):
        counts = counts_by_stem.get(image_path.stem)
        if counts is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label = make_label(image, counts, cfg, params)
        label_path = out_dir / f"{image_path.stem}.npz"
        save_label(label_path, label)
        index_rows.append({"image": str(image_path), "label": str(label_path)})

    index_path = Path(args.index_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"config": cfg.__dict__, "items": index_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(index_rows)} labels to {out_dir}")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
