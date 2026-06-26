from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .classes import WHITE, YELLOW
from .classical_lane import (
    DetectorParams,
    FEATURE_NAMES,
    detect_lane_candidates,
    fallback_color_scores,
    image_files,
)
from .xlsx_counts import read_count_json, read_count_xlsx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a white/yellow color model from count-level labels.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx")
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--out", default="models/color_model.json")
    parser.add_argument("--samples-out", default="models/color_samples.csv")
    parser.add_argument("--max-candidates-factor", type=float, default=2.5)
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


def weak_assign_labels(candidates, counts: dict[str, int], max_factor: float) -> list[tuple[str, object]]:
    total_need = int(counts.get(WHITE, 0)) + int(counts.get(YELLOW, 0))
    if total_need <= 0 or not candidates:
        return []

    max_candidates = max(total_need, int(round(total_need * max_factor)))
    candidates = candidates[:max_candidates]
    scored = [(candidate, fallback_color_scores(candidate.features)) for candidate in candidates]
    assigned: list[tuple[str, object]] = []
    used: set[int] = set()

    for cls in sorted((WHITE, YELLOW), key=lambda name: (counts.get(name, 0), name)):
        need = max(0, int(counts.get(cls, 0)))
        ranked = [
            (scores[cls], candidate.candidate_score, idx, candidate)
            for idx, (candidate, scores) in enumerate(scored)
            if idx not in used
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx, candidate in ranked[:need]:
            used.add(idx)
            assigned.append((cls, candidate))
    return assigned


def class_stats(samples: list[list[float]]) -> tuple[list[float], list[float]]:
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) == 0:
        return [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std = np.maximum(std, 0.05)
    return mean.astype(float).tolist(), std.astype(float).tolist()


def main() -> None:
    args = parse_args()
    import cv2

    counts_by_stem = load_counts(args)
    params = DetectorParams(
        roi_top_ratio=args.roi_top_ratio,
        min_line_length=args.min_line_length,
        line_width=args.line_width,
    )

    samples = {WHITE: [], YELLOW: []}
    rows: list[list[object]] = []

    files = image_files(Path(args.image_dir))
    for image_path in tqdm(files, desc="Learning color"):
        counts = counts_by_stem.get(image_path.stem)
        if counts is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        candidates = detect_lane_candidates(image, params)
        assigned = weak_assign_labels(candidates, counts, args.max_candidates_factor)
        for cls, candidate in assigned:
            samples[cls].append(candidate.features)
            rows.append([image_path.name, cls, candidate.candidate_score, *candidate.features])

    total = sum(len(items) for items in samples.values())
    priors = {
        cls: (len(items) / total if total else 0.5)
        for cls, items in samples.items()
    }
    model = {
        "method": "classical_line_plus_color_prototypes",
        "feature_names": FEATURE_NAMES,
        "classes": {},
        "sample_counts": {cls: len(items) for cls, items in samples.items()},
    }
    for cls, cls_samples in samples.items():
        mean, std = class_stats(cls_samples)
        model["classes"][cls] = {
            "mean": mean,
            "std": std,
            "prior": priors[cls],
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    samples_path = Path(args.samples_out)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["filename", "label", "candidate_score", *FEATURE_NAMES])
        writer.writerows(rows)

    print(f"Saved color model to {out_path}")
    print(f"Saved weak samples to {samples_path}")
    print(f"Sample counts: {model['sample_counts']}")
    if total == 0:
        print("WARNING: no samples were learned. Check image directory and count labels.")


if __name__ == "__main__":
    main()
