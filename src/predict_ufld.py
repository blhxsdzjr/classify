from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .classes import WHITE, YELLOW
from .classical_lane import draw_predictions, image_files, write_counts_csv
from .geometry import fit_line_from_points
from .ufld_model import UFLDConfig, build_model
from .xlsx_counts import read_count_json, read_count_xlsx


ID_TO_COLOR = {0: WHITE, 1: YELLOW, 2: "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict lanes with the UFLD-inspired row-anchor model.")
    parser.add_argument("--weights", default="models/ufld_tiny.pt")
    parser.add_argument("--source", default="datasets/local_colm/images/test")
    parser.add_argument("--out", default="predictions_ufld.json")
    parser.add_argument("--counts-out", default="prediction_counts_ufld.csv")
    parser.add_argument("--save-vis", default="runs/ufld_vis")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf-thr", type=float, default=0.35)
    parser.add_argument("--min-valid-rows", type=int, default=3)
    parser.add_argument("--gt-xlsx", default=None)
    parser.add_argument("--gt-counts", default=None)
    parser.add_argument("--use-count-constraints", action="store_true")
    return parser.parse_args()


def normalize_device(device: str) -> str:
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


def load_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    if args.gt_counts:
        raw = read_count_json(Path(args.gt_counts))
    elif args.gt_xlsx:
        raw = read_count_xlsx(Path(args.gt_xlsx))
    else:
        return {}
    return {Path(filename).stem: counts for filename, counts in raw.items()}


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    cfg = UFLDConfig(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def preprocess(image_bgr: np.ndarray, cfg: UFLDConfig, device: str) -> torch.Tensor:
    import cv2

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)


def lane_points_from_logits(
    grid_logits: torch.Tensor,
    cfg: UFLDConfig,
    original_shape: tuple[int, int],
    lane_idx: int,
    conf_thr: float,
) -> tuple[list[list[float]], float]:
    h, w = original_shape
    probs = F.softmax(grid_logits[lane_idx], dim=-1)
    confs, labels = torch.max(probs, dim=-1)
    points = []
    used_confs = []
    for row_idx, (conf, label) in enumerate(zip(confs.cpu().numpy(), labels.cpu().numpy())):
        if int(label) == cfg.no_lane_index or float(conf) < conf_thr:
            continue
        row_y = cfg.row_anchors()[row_idx] / cfg.image_height * h
        x = int(label) / max(cfg.num_grids - 1, 1) * (w - 1)
        points.append([float(x), float(row_y)])
        used_confs.append(float(conf))
    return points, (float(np.mean(used_confs)) if used_confs else 0.0)


def prediction_from_points(
    points: list[list[float]],
    cls: str,
    conf: float,
    color_scores: dict[str, float],
) -> dict[str, Any] | None:
    fitted = fit_line_from_points(np.asarray(points, dtype=np.float32))
    if fitted is None:
        return None
    angle, endpoints, bbox = fitted
    return {
        "class": cls,
        "conf": conf,
        "angle_deg": float(angle),
        "endpoints": endpoints,
        "bbox": bbox,
        "source_class": "ufld_row_anchor",
        "color_score": float(color_scores.get(cls, 0.0)),
        "white_score": float(color_scores.get(WHITE, 0.0)),
        "yellow_score": float(color_scores.get(YELLOW, 0.0)),
    }


def apply_count_constraints(instances: list[dict[str, Any]], counts: dict[str, int]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    used: set[int] = set()
    for cls in sorted([WHITE, YELLOW], key=lambda name: (counts.get(name, 0), name)):
        need = max(0, int(counts.get(cls, 0)))
        score_key = "white_score" if cls == WHITE else "yellow_score"
        ranked = [
            (float(inst.get(score_key, 0.0)), float(inst.get("conf", 0.0)), idx, inst)
            for idx, inst in enumerate(instances)
            if idx not in used
        ]
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, idx, inst in ranked[:need]:
            fixed = dict(inst)
            fixed["class"] = cls
            kept.append(fixed)
            used.add(idx)
    kept.sort(key=lambda inst: float(inst.get("conf", 0.0)), reverse=True)
    return kept


@torch.no_grad()
def predict_one(model, cfg: UFLDConfig, image_bgr: np.ndarray, args) -> list[dict[str, Any]]:
    output = model(preprocess(image_bgr, cfg, args.device))
    grid_logits = output["grid_logits"][0]
    color_probs = F.softmax(output["color_logits"][0], dim=-1).cpu().numpy()
    instances = []
    for lane_idx in range(cfg.max_lanes):
        points, line_conf = lane_points_from_logits(
            grid_logits,
            cfg,
            image_bgr.shape[:2],
            lane_idx,
            args.conf_thr,
        )
        if len(points) < args.min_valid_rows:
            continue
        color_id = int(np.argmax(color_probs[lane_idx]))
        cls = ID_TO_COLOR.get(color_id, "none")
        if cls == "none":
            continue
        scores = {WHITE: float(color_probs[lane_idx][0]), YELLOW: float(color_probs[lane_idx][1])}
        inst = prediction_from_points(points, cls, line_conf, scores)
        if inst is not None:
            inst["row_points"] = points
            instances.append(inst)
    return instances


def main() -> None:
    args = parse_args()
    args.device = normalize_device(args.device)
    import cv2

    model, cfg = load_model(Path(args.weights), args.device)
    counts_by_stem = load_counts(args)
    images: dict[str, dict[str, Any]] = {}
    vis_dir = Path(args.save_vis) if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_files(Path(args.source)), desc="UFLD predict"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        instances = predict_one(model, cfg, image, args)
        counts = counts_by_stem.get(image_path.stem)
        if args.use_count_constraints and counts is not None:
            instances = apply_count_constraints(instances, counts)
        images[image_path.name] = {
            "width": image.shape[1],
            "height": image.shape[0],
            "instances": instances,
        }
        if vis_dir:
            cv2.imwrite(str(vis_dir / image_path.name), draw_predictions(image, instances))

    output = {
        "meta": {
            "method": "ufld_inspired_row_anchor",
            "reference": "https://github.com/cfzd/Ultra-Fast-Lane-Detection",
            "weights": args.weights,
            "source": args.source,
            "use_count_constraints": args.use_count_constraints,
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
