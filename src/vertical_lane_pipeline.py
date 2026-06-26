from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import cv2
import numpy as np
import torch

from .classes import WHITE, YELLOW
from .classical_lane import image_files
from .geometry import fit_line_from_points
from .xlsx_counts import read_count_xlsx


@dataclass(frozen=True)
class VerticalLaneParams:
    max_width: int = 960
    roi_top_ratio: float = 0.30
    min_angle_from_horizontal: float = 35.0
    max_angle_from_horizontal: float = 145.0
    min_vertical_span_ratio: float = 0.16
    min_color_fraction: float = 0.18
    hough_threshold: int = 30
    min_line_length: int = 45
    max_line_gap: int = 125
    cluster_angle_deg: float = 18.0
    cluster_x_gap: float = 105.0
    min_group_segments: int = 1
    line_width: int = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect near-vertical white/yellow lane markings and count them.")
    parser.add_argument("--source", default="datasets/local_colm/images/test")
    parser.add_argument("--gt-xlsx", default="结果统计.xlsx")
    parser.add_argument("--out", default="runs/vertical_lane_predictions.json")
    parser.add_argument("--counts-out", default="runs/vertical_lane_counts.csv")
    parser.add_argument("--metrics-out", default="runs/vertical_lane_metrics.json")
    parser.add_argument("--report-xlsx", default="runs/vertical_lane_report.xlsx")
    parser.add_argument("--save-vis", default="runs/vertical_lane_vis")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--conf-thr", type=float, default=0.08)
    parser.add_argument("--color-model", default=None, help="Path to learned color classifier .pkl (logistic regression).")
    return parser.parse_args()


def scaled_image(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy(), 1.0
    scale = max_width / float(width)
    resized = cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def roi_mask(shape: tuple[int, int], top_ratio: float) -> np.ndarray:
    height, width = shape
    top = int(height * top_ratio)
    polygon = np.array(
        [
            [int(width * 0.03), height - 1],
            [int(width * 0.97), height - 1],
            [int(width * 0.82), top],
            [int(width * 0.18), top],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def color_masks(image_bgr: np.ndarray, params: VerticalLaneParams) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white = ((s <= 95) & (v >= 145)).astype(np.uint8) * 255
    yellow = ((h >= 15) & (h <= 40) & (s >= 70) & (v >= 95)).astype(np.uint8) * 255

    road_roi = roi_mask(image_bgr.shape[:2], params.roi_top_ratio)
    kernel = np.ones((3, 3), np.uint8)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 19))
    out = {}
    for cls, mask in ((WHITE, white), (YELLOW, yellow)):
        mask = cv2.bitwise_and(mask, road_roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel)
        out[cls] = mask
    return out


def angle_deg(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def angle_diff(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def line_length(seg: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    return math.hypot(x2 - x1, y2 - y1)


def line_mask(shape: tuple[int, int], seg: np.ndarray, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    p1 = (int(round(seg[0])), int(round(seg[1])))
    p2 = (int(round(seg[2])), int(round(seg[3])))
    cv2.line(mask, p1, p2, 255, max(1, int(width)), cv2.LINE_AA)
    return mask.astype(bool)


def x_at_y(seg: np.ndarray, y: float) -> float:
    x1, y1, x2, y2 = [float(v) for v in seg]
    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) * 0.5
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


def detect_segments(mask: np.ndarray, cls: str, params: VerticalLaneParams) -> list[np.ndarray]:
    edges = cv2.Canny(mask, 35, 110)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=params.hough_threshold,
        minLineLength=params.min_line_length,
        maxLineGap=params.max_line_gap,
    )
    if lines is None:
        return []
    height = mask.shape[0]
    segments: list[np.ndarray] = []
    for raw in lines[:, 0, :].astype(np.float32):
        angle = angle_deg(raw)
        if angle < params.min_angle_from_horizontal or angle > params.max_angle_from_horizontal:
            continue
        vertical_span = abs(float(raw[3] - raw[1]))
        if vertical_span < height * params.min_vertical_span_ratio:
            continue
        sample = line_mask(mask.shape, raw, params.line_width)
        color_fraction = float(np.mean(mask[sample] > 0)) if np.any(sample) else 0.0
        min_color_fraction = params.min_color_fraction + (0.07 if cls == YELLOW else 0.0)
        if color_fraction < min_color_fraction:
            continue
        segments.append(raw)
    segments.sort(key=line_length, reverse=True)
    return segments


def segment_distance(a: np.ndarray, b: np.ndarray, height: int) -> float:
    y_values = [height * 0.42, height * 0.62, height * 0.86]
    distances = [abs(x_at_y(a, y) - x_at_y(b, y)) for y in y_values]
    return float(min(distances))


def cluster_segments(segments: list[np.ndarray], image_shape: tuple[int, int], params: VerticalLaneParams) -> list[list[np.ndarray]]:
    height, _ = image_shape
    groups: list[list[np.ndarray]] = []
    group_refs: list[np.ndarray] = []
    for seg in segments:
        seg_angle = angle_deg(seg)
        best_idx = None
        best_dist = float("inf")
        for idx, ref in enumerate(group_refs):
            if angle_diff(seg_angle, angle_deg(ref)) > params.cluster_angle_deg:
                continue
            dist = segment_distance(seg, ref, height)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None and best_dist <= params.cluster_x_gap:
            groups[best_idx].append(seg)
            group_refs[best_idx] = fit_group(groups[best_idx])
        else:
            groups.append([seg])
            group_refs.append(seg.copy())
    return [group for group in groups if len(group) >= params.min_group_segments]


def fit_group(group: list[np.ndarray]) -> np.ndarray:
    pts = np.vstack([seg.reshape(2, 2) for seg in group]).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([float(vx), float(vy)], dtype=np.float32)
    origin = np.array([float(cx), float(cy)], dtype=np.float32)
    t = (pts - origin) @ direction
    p1 = origin + direction * float(np.min(t))
    p2 = origin + direction * float(np.max(t))
    return np.array([p1[0], p1[1], p2[0], p2[1]], dtype=np.float32)


def group_to_prediction(
    group: list[np.ndarray],
    cls: str,
    image_shape: tuple[int, int],
    scale: float,
    params: VerticalLaneParams,
) -> dict[str, Any] | None:
    fitted = fit_group(group)
    points = np.vstack([seg.reshape(2, 2) for seg in group]).astype(np.float32)
    line = fit_line_from_points(points)
    if line is None:
        return None
    angle, endpoints, bbox = line
    length = line_length(fitted)
    height, width = image_shape
    y_values = points[:, 1]
    if float(np.max(y_values)) < height * 0.55:
        return None
    vertical_span = float(np.max(y_values) - np.min(y_values))
    if vertical_span < height * params.min_vertical_span_ratio * 1.15:
        return None
    verticalness = abs(math.sin(math.radians(angle)))
    if verticalness < 0.66:
        return None
    conf = min(
        1.0,
        (vertical_span / max(height * 0.42, 1.0))
        * (0.45 + 0.12 * min(len(group), 5))
        * (0.55 + 0.45 * verticalness),
    )
    inv_scale = 1.0 / scale
    endpoints = [[float(x * inv_scale), float(y * inv_scale)] for x, y in endpoints]
    bbox = [float(v * inv_scale) for v in bbox]
    row_points = [[float(x * inv_scale), float(y * inv_scale)] for x, y in points.tolist()]
    return {
        "class": cls,
        "conf": float(conf),
        "angle_deg": float(angle),
        "endpoints": endpoints,
        "bbox": bbox,
        "source_class": "vertical_color_hough_group",
        "segments": len(group),
        "row_points": row_points,
    }


def suppress_duplicates(instances: list[dict[str, Any]], distance_thr: float = 42.0) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for inst in sorted(instances, key=lambda item: item["conf"], reverse=True):
        x1 = (inst["endpoints"][0][0] + inst["endpoints"][1][0]) * 0.5
        duplicate = False
        for other in kept:
            if inst["class"] != other["class"]:
                continue
            x2 = (other["endpoints"][0][0] + other["endpoints"][1][0]) * 0.5
            if abs(x1 - x2) <= distance_thr and angle_diff(inst["angle_deg"], other["angle_deg"]) <= 12.0:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def limit_lane_counts(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limits = {WHITE: 5, YELLOW: 1}
    selected: list[dict[str, Any]] = []
    for cls in (WHITE, YELLOW):
        ranked = [inst for inst in instances if inst.get("class") == cls]
        ranked.sort(key=lambda item: (float(item.get("conf", 0.0)), int(item.get("segments", 0))), reverse=True)
        selected.extend(ranked[: limits[cls]])
    selected.sort(key=lambda item: (item["class"], item["endpoints"][0][0] + item["endpoints"][1][0]))
    return selected


def _load_color_model(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


class _ColorMLP(torch.nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 2),
        )
    def forward(self, x):
        return self.net(x)


def _classify_with_model(fitted_line: np.ndarray, small_bgr: np.ndarray, model_data: dict, line_width: int) -> tuple[str, float, float]:
    """Use learned classifier (LR or MLP) to predict white/yellow from line-region features."""
    x1, y1, x2, y2 = [float(v) for v in fitted_line]
    x1, y1 = np.clip(x1, 0, small_bgr.shape[1]-1), np.clip(y1, 0, small_bgr.shape[0]-1)
    x2, y2 = np.clip(x2, 0, small_bgr.shape[1]-1), np.clip(y2, 0, small_bgr.shape[0]-1)

    mask = np.zeros(small_bgr.shape[:2], dtype=np.uint8)
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, max(1, line_width), cv2.LINE_AA)
    mask_bool = mask.astype(bool)
    if mask_bool.sum() < 10:
        return WHITE, 0.5, 0.5

    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    h, s, v = hsv[:,:,0][mask_bool].astype(np.float32), hsv[:,:,1][mask_bool].astype(np.float32), hsv[:,:,2][mask_bool].astype(np.float32)
    lab_b = lab[:,:,2][mask_bool].astype(np.float32)

    length = float(np.hypot(x2-x1, y2-y1))
    angle = math.degrees(math.atan2(y2-y1, x2-x1)) % 180.0
    diag = math.hypot(small_bgr.shape[1], small_bgr.shape[0])

    feats = np.array([
        float(((s <= 95) & (v >= 145)).mean()),
        float(((h >= 15) & (h <= 40) & (s >= 70) & (v >= 95)).mean()),
        float(np.mean(np.maximum(0.0, 1.0 - np.abs(h - 28.0) / 32.0))),
        float(np.mean(s) / 255.0),
        float(np.mean(v) / 255.0),
        float(np.clip((np.mean(lab_b) - 128.0) / 60.0, -1.0, 1.0)),
        float(min(1.0, length / max(diag * 0.45, 1.0))),
        float(abs(math.sin(math.radians(angle)))),
    ], dtype=np.float32).reshape(1, -1)

    X = model_data["scaler"].transform(feats)
    X_t = torch.tensor(X, dtype=torch.float32)

    if "in_dim" in model_data:
        # MLP model
        in_dim = model_data["in_dim"]
        model = _ColorMLP(in_dim)
        model.load_state_dict(model_data["model"])
        model.eval()
        with torch.no_grad():
            logits = model(X_t)[0]
        white_prob = float(torch.softmax(logits, 0)[0])
        yellow_prob = float(torch.softmax(logits, 0)[1])
    else:
        # Logistic regression
        proba = model_data["model"].predict_proba(X)[0]
        white_prob = float(proba[0])
        yellow_prob = float(proba[1])

    cls = YELLOW if yellow_prob > white_prob else WHITE
    return cls, white_prob, yellow_prob


def detect_image(image_bgr: np.ndarray, params: VerticalLaneParams,
                 color_model=None) -> list[dict[str, Any]]:
    small, scale = scaled_image(image_bgr, params.max_width)
    masks = color_masks(small, params)
    instances: list[dict[str, Any]] = []

    # Always use HSV masks for detection (proven line quality)
    for cls in (WHITE, YELLOW):
        segments = detect_segments(masks[cls], cls, params)
        groups = cluster_segments(segments, small.shape[:2], params)
        for group in groups:
            pred = group_to_prediction(group, cls, small.shape[:2], scale, params)
            if pred is not None:
                instances.append(pred)

    # Post-process: for YELLOW lines only, use LR model to suppress false positives.
    # White lines are left untouched (HSV white precision is already 65%).
    # Yellow suffers from road-surface false positives; LR provides a second opinion.
    if color_model is not None:
        filtered = []
        for inst in instances:
            ep = inst["endpoints"]
            fitted = np.array([ep[0][0], ep[0][1], ep[1][0], ep[1][1]], dtype=np.float32)
            cls, wp, yp = _classify_with_model(fitted, small, color_model, params.line_width)
            inst["white_score"] = float(wp)
            inst["yellow_score"] = float(yp)
            inst["color_source"] = "logistic_regression_filter"
            # For yellow detections: keep only if LR also says yellow (dual confirmation)
            if inst["class"] == YELLOW and cls != YELLOW:
                continue  # Drop — LR rejects this yellow
            filtered.append(inst)
        instances = filtered

    return limit_lane_counts(suppress_duplicates(instances))


def draw_predictions(image: np.ndarray, instances: list[dict[str, Any]], conf_thr: float) -> np.ndarray:
    canvas = image.copy()
    for inst in instances:
        if float(inst.get("conf", 0.0)) < conf_thr:
            continue
        color = (0, 0, 255) if inst["class"] == WHITE else (255, 0, 0)
        p1 = tuple(int(round(v)) for v in inst["endpoints"][0])
        p2 = tuple(int(round(v)) for v in inst["endpoints"][1])
        cv2.line(canvas, p1, p2, color, 7, cv2.LINE_AA)
    return canvas


def empty_counts() -> dict[str, int]:
    return {WHITE: 0, YELLOW: 0}


def summarize(images: dict[str, dict[str, Any]], gt_counts: dict[str, dict[str, int]], conf_thr: float) -> dict[str, Any]:
    totals = {cls: {"detected": 0, "correct": 0, "gt": 0} for cls in (WHITE, YELLOW)}
    per_image: dict[str, Any] = {}
    stems = sorted({Path(k).stem for k in images} | {Path(k).stem for k in gt_counts}, key=lambda s: int(s) if s.isdigit() else s)
    for stem in stems:
        pred_counts = empty_counts()
        payload = next((v for k, v in images.items() if Path(k).stem == stem), {"instances": []})
        for inst in payload.get("instances", []):
            if float(inst.get("conf", 0.0)) >= conf_thr and inst.get("class") in pred_counts:
                pred_counts[inst["class"]] += 1
        gt_raw = next((v for k, v in gt_counts.items() if Path(k).stem == stem), {})
        gt = {WHITE: int(gt_raw.get(WHITE, 0)), YELLOW: int(gt_raw.get(YELLOW, 0))}
        correct = {cls: min(pred_counts[cls], gt[cls]) for cls in (WHITE, YELLOW)}
        per_image[stem] = {"pred_counts": pred_counts, "gt_counts": gt, "correct_counts": correct}
        for cls in (WHITE, YELLOW):
            totals[cls]["detected"] += pred_counts[cls]
            totals[cls]["gt"] += gt[cls]
            totals[cls]["correct"] += correct[cls]

    metrics: dict[str, dict[str, float]] = {}
    overall = {"detected": 0, "correct": 0, "gt": 0}
    for cls, row in totals.items():
        precision = row["correct"] / row["detected"] if row["detected"] else 0.0
        recall = row["correct"] / row["gt"] if row["gt"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[cls] = {**row, "precision": precision, "recall": recall, "f1": f1}
        for key in overall:
            overall[key] += row[key]
    precision = overall["correct"] / overall["detected"] if overall["detected"] else 0.0
    recall = overall["correct"] / overall["gt"] if overall["gt"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics["overall"] = {**overall, "precision": precision, "recall": recall, "f1": f1}
    return {"metrics": metrics, "images": per_image}


def write_counts_csv(images: dict[str, dict[str, Any]], path: Path, conf_thr: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["文件名", "检测车道线数", "检测白线数", "检测黄线数"])
        for name in sorted(images, key=lambda s: int(Path(s).stem) if Path(s).stem.isdigit() else s):
            counts = empty_counts()
            for inst in images[name]["instances"]:
                if float(inst.get("conf", 0.0)) >= conf_thr and inst.get("class") in counts:
                    counts[inst["class"]] += 1
            writer.writerow([name, counts[WHITE] + counts[YELLOW], counts[WHITE], counts[YELLOW]])


def cell_ref(row: int, col: int) -> str:
    letters = ""
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row}"


def xlsx_cell(row: int, col: int, value: Any) -> str:
    ref = cell_ref(row, col)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def write_report_xlsx(report: dict[str, Any], path: Path) -> None:
    headers = [
        "文件名",
        "GT白线数",
        "检测白线数",
        "正确白线数",
        "GT黄线数",
        "检测黄线数",
        "正确黄线数",
    ]
    rows: list[list[Any]] = [headers]
    for stem, item in sorted(report["images"].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        rows.append(
            [
                f"{stem}.jpg",
                item["gt_counts"][WHITE],
                item["pred_counts"][WHITE],
                item["correct_counts"][WHITE],
                item["gt_counts"][YELLOW],
                item["pred_counts"][YELLOW],
                item["correct_counts"][YELLOW],
            ]
        )
    rows.append([])
    rows.append(["类别", "检测数", "正确数", "GT数", "准确率", "召回率", "F1"])
    for cls, label in ((WHITE, "白线"), (YELLOW, "黄线"), ("overall", "总体")):
        metric = report["metrics"][cls]
        rows.append(
            [
                label,
                metric["detected"],
                metric["correct"],
                metric["gt"],
                round(metric["precision"], 6),
                round(metric["recall"], 6),
                round(metric["f1"], 6),
            ]
        )

    sheet_rows = []
    for row_idx, values in enumerate(rows, start=1):
        cells = "".join(xlsx_cell(row_idx, col_idx, value) for col_idx, value in enumerate(values, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(sheet_rows)
        + "</sheetData></worksheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="统计结果" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def main() -> None:
    args = parse_args()
    params = VerticalLaneParams(max_width=args.max_width)
    source = Path(args.source)
    vis_dir = Path(args.save_vis)
    vis_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, dict[str, Any]] = {}

    color_model = None
    if args.color_model:
        color_model = _load_color_model(args.color_model)
        print(f"Loaded color model from {args.color_model}")

    for image_path in image_files(source):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        instances = detect_image(image, params, color_model=color_model)
        images[image_path.name] = {"width": image.shape[1], "height": image.shape[0], "instances": instances}
        cv2.imwrite(str(vis_dir / image_path.name), draw_predictions(image, instances, args.conf_thr))

    output = {
        "meta": {
            "method": "vertical_color_hough_group",
            "note": "White/yellow HSV/Lab masks + Hough line segments + near-vertical grouping for dashed lane markings.",
            "visualization": "white_lane drawn in red, yellow_lane drawn in blue",
        },
        "images": images,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counts_csv(images, Path(args.counts_out), args.conf_thr)

    gt_counts = read_count_xlsx(Path(args.gt_xlsx))
    report = summarize(images, gt_counts, args.conf_thr)
    report["settings"] = {
        "conf_thr": args.conf_thr,
        "note": "The provided xlsx has count GT only, so correct counts are computed as min(pred_count, gt_count) per class/image; line-level 15-degree matching needs per-line GT geometry.",
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_xlsx(report, Path(args.report_xlsx))

    for cls, row in report["metrics"].items():
        print(
            f"{cls}: detected={row['detected']} correct={row['correct']} gt={row['gt']} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
        )
    print(f"Saved predictions to {out_path}")
    print(f"Saved counts to {args.counts_out}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved report xlsx to {args.report_xlsx}")
    print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
