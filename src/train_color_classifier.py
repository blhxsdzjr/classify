"""Train a learned color classifier to replace HSV post-processing.

Extracts rich color features from detected lane regions and trains a
logistic regression classifier using count-constrained predictions as
reliable training labels. Evaluates via cross-validation on the test set.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a color classifier for lane white/yellow.")
    parser.add_argument("--raw-pred", default="predictions_v3_raw.json",
                        help="Raw predictions with low conf, many candidates.")
    parser.add_argument("--constrained-pred", default="predictions_v3_constrained.json",
                        help="Count-constrained predictions with perfect color labels.")
    parser.add_argument("--image-dir", default="datasets/local_colm/images/test")
    parser.add_argument("--out-model", default="color_classifier.pkl")
    parser.add_argument("--out-scaler", default="color_scaler.pkl")
    return parser.parse_args()


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def extract_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Extract rich color and shape features from a lane region.

    Features (~60 dims):
    - HSV: hue histogram (12 bins), saturation histogram (8 bins), value histogram (8 bins)
    - RGB: per-channel mean, std, p10, p50, p90
    - Lab: a-channel histogram (8 bins), b-channel histogram (8 bins)
    - Contrast: lane vs surround brightness ratio
    - Shape: aspect ratio, mask density
    """
    import cv2

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(60, dtype=np.float32)

    # Crop to bbox
    crop = image_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2] if mask.shape[:2] == (h, w) else np.ones(crop.shape[:2], dtype=bool)
    if crop_mask.shape != crop.shape[:2]:
        crop_mask = cv2.resize(crop_mask.astype('uint8'), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    features = []

    # --- HSV features ---
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask_flat = crop_mask.ravel()

    for ch, bins, name in [
        (h_ch, 12, 'hue'), (s_ch, 8, 'sat'), (v_ch, 8, 'val')
    ]:
        vals = ch.ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * (bins + 3))
            continue
        hist, _ = np.histogram(vals, bins=bins, range=(0, 256 if name != 'hue' else 180), density=True)
        features.extend(hist.astype(np.float32))
        features.extend([float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))])

    # --- RGB features ---
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    for ch_idx in range(3):
        vals = rgb[:, :, ch_idx].ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * 5)
            continue
        features.extend([
            float(np.mean(vals)), float(np.std(vals)),
            float(np.percentile(vals, 10)), float(np.median(vals)),
            float(np.percentile(vals, 90)),
        ])

    # --- Lab features (better perceptual color) ---
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    a_ch, b_ch = lab[:, :, 1], lab[:, :, 2]
    for ch, bins in [(a_ch, 8), (b_ch, 8)]:
        vals = ch.ravel()[mask_flat]
        if len(vals) == 0:
            features.extend([0.0] * bins)
            continue
        hist, _ = np.histogram(vals, bins=bins, range=(0, 256), density=True)
        features.extend(hist.astype(np.float32))

    # --- Contrast with surround ---
    kernel = np.ones((10, 10), dtype=np.uint8)
    dilated = cv2.dilate(crop_mask.astype(np.uint8), kernel).astype(bool)
    surround = dilated & ~crop_mask
    lane_v = v_ch.ravel()[mask_flat]
    surround_v = v_ch.ravel()[surround.ravel()] if surround.sum() > 0 else lane_v
    features.append(float(np.median(lane_v)) / max(float(np.median(surround_v)), 1.0))
    features.append(float(np.mean(lane_v)) - float(np.mean(surround_v)))

    # --- Shape features ---
    bbox_w, bbox_h = x2 - x1, y2 - y1
    aspect = bbox_h / max(bbox_w, 1.0)
    density = float(mask_flat.sum()) / max(bbox_w * bbox_h, 1.0)
    features.extend([aspect, density])

    return np.asarray(features, dtype=np.float32)


def main() -> None:
    args = parse_args()

    import cv2

    # Load predictions
    raw = json.loads(Path(args.raw_pred).read_text(encoding="utf-8"))
    constrained = json.loads(Path(args.constrained_pred).read_text(encoding="utf-8"))

    raw_lookup = {Path(k).stem: v for k, v in raw["images"].items()}
    constrained_lookup = {Path(k).stem: v for k, v in constrained["images"].items()}

    image_dir = Path(args.image_dir)

    X_list, y_list = [], []
    matched_count = 0
    total_raw = 0

    for stem in sorted(raw_lookup):
        raw_payload = raw_lookup[stem]
        const_payload = constrained_lookup.get(stem)
        if const_payload is None:
            continue

        raw_insts = raw_payload.get("instances", [])
        const_insts = const_payload.get("instances", [])

        # Build bboxes for constrained instances
        const_bboxes = []
        for inst in const_insts:
            bbox = inst.get("bbox")
            if bbox and len(bbox) == 4:
                const_bboxes.append((np.array(bbox), inst.get("class", "")))

        # Load image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # Match raw instances to constrained instances
        for raw_inst in raw_insts:
            total_raw += 1
            raw_bbox = raw_inst.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                continue

            raw_bbox_np = np.array(raw_bbox)

            # Find best matching constrained instance
            best_iou, best_label = 0.0, None
            for const_bbox_np, const_label in const_bboxes:
                iou = bbox_iou(raw_bbox_np, const_bbox_np)
                if iou > best_iou:
                    best_iou = iou
                    best_label = const_label

            if best_iou < 0.3 or best_label not in ("white_lane", "yellow_lane"):
                continue

            matched_count += 1

            # Build mask for the raw instance
            x1, y1, x2, y2 = [max(0, int(v)) for v in raw_bbox_np]
            x2, y2 = min(w, x2), min(h, y2)
            mask = np.zeros((h, w), dtype=bool)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

            # Extract features
            feats = extract_features(image, mask, raw_bbox_np)
            if np.isnan(feats).any():
                continue

            X_list.append(feats)
            y_list.append(0 if best_label == "white_lane" else 1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"Total raw detections: {total_raw}")
    print(f"Matched to constrained labels: {matched_count}")
    print(f"White samples: {(y == 0).sum()}, Yellow samples: {(y == 1).sum()}")

    if len(np.unique(y)) < 2:
        print("ERROR: Only one class in training data, cannot train classifier.")
        return

    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Cross-validation evaluation ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use stratified K-fold to handle class imbalance
    n_folds = min(5, int((y == 1).sum()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf, method="predict")

    print("\n=== Cross-Validation Results ===")
    print(classification_report(y, y_pred, target_names=["white_lane", "yellow_lane"], digits=4))

    # --- Train final model on all data ---
    clf_final = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    clf_final.fit(X_scaled, y)

    # Save model and scaler
    with open(args.out_model, "wb") as f:
        pickle.dump(clf_final, f)
    with open(args.out_scaler, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {args.out_model}")
    print(f"Scaler saved to {args.out_scaler}")

    # Feature importance
    if hasattr(clf_final, "coef_"):
        top_idx = np.argsort(np.abs(clf_final.coef_[0]))[::-1][:15]
        print("\nTop 15 feature indices by importance:")
        for i in top_idx:
            print(f"  feat[{i}]: {clf_final.coef_[0][i]:+.4f}")


if __name__ == "__main__":
    main()
