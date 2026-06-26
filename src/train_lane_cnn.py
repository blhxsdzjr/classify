"""Train a row-anchor lane detection CNN on GT labels.

Model: Tiny CNN backbone → grid head (x-position per row) + color head (white/yellow/none)
Input:  800×288 RGB image
Output: per-lane per-row grid cell + per-lane color class

GT labels: 37 images, 112 manually annotated lane centerlines
Training: 30 train / 7 val split, heavy augmentation to compensate for small dataset
"""

from __future__ import annotations

import argparse, json, math, pickle, random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .geometry import fit_line_from_points

# ── config ──────────────────────────────────────────────────
IMAGE_W, IMAGE_H = 800, 288
NUM_ROWS = 18
NUM_GRIDS = 100
MAX_LANES = 6
NO_LANE = NUM_GRIDS  # special "no lane" class index
COLOR_CLASSES = 3  # white=0, yellow=1, none=2
ROI_TOP = 0.30     # top 30% of image is sky, ignore for row anchors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="datasets/local_colm/images/test")
    p.add_argument("--gt-dir", default="datasets/local_colm/labels/test")
    p.add_argument("--out", default="models/lane_cnn.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── dataset ─────────────────────────────────────────────────
class LaneDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = cv2.imread(item["image"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                # Flip GT x-coordinates too
                item = dict(item)
                item["lanes"] = [
                    {**l, "x_norm": [1.0 - x for x in l["x_norm"]]}
                    for l in item["lanes"]
                ]

        h, w = img.shape[:2]
        img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5

        # Build row targets and color targets
        row_targets = np.full((MAX_LANES, NUM_ROWS), NO_LANE, dtype=np.int64)
        color_targets = np.full((MAX_LANES,), 2, dtype=np.int64)  # none
        exists = np.zeros((MAX_LANES,), dtype=np.float32)

        row_ys = np.linspace(int(IMAGE_H * ROI_TOP), IMAGE_H - 8, NUM_ROWS)

        for li, lane in enumerate(item.get("lanes", [])[:MAX_LANES]):
            exists[li] = 1.0
            color_targets[li] = 0 if lane["class"] == "white_lane" else 1

            # Interpolate GT x at each row anchor y
            xs_norm = lane["x_norm"]  # normalized [0, 1]
            # GT is stored as endpoints; interpolate along the line
            x1, y1, x2, y2 = lane["endpoints_norm"]

            for ri, ry in enumerate(row_ys):
                ry_norm = ry / IMAGE_H
                # Check if row y is within the GT line vertical span
                ly1, ly2 = y1 * h / IMAGE_H, y2 * h / IMAGE_H
                if not (min(ly1, ly2) <= ry * h / IMAGE_H <= max(ly1, ly2)):
                    continue
                if abs(y2 - y1) < 1e-6:
                    continue
                t = (ry_norm - y1) / (y2 - y1)
                x_norm = x1 + t * (x2 - x1)
                grid = int(round(np.clip(x_norm, 0.0, 1.0) * (NUM_GRIDS - 1)))
                row_targets[li, ri] = grid

        return {
            "image": img_t,
            "row_targets": torch.from_numpy(row_targets),
            "color_targets": torch.from_numpy(color_targets),
            "exists": torch.from_numpy(exists),
        }


# ── model ───────────────────────────────────────────────────
class TinyLaneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: 800×288 → 25×9 feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 12)),
        )
        self.feat_dim = 128 * 4 * 12
        self.hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        # Grid head: per-lane per-row grid classification
        self.grid_head = nn.Linear(512, MAX_LANES * NUM_ROWS * (NUM_GRIDS + 1))
        # Color head: per-lane color classification
        self.color_head = nn.Linear(512, MAX_LANES * COLOR_CLASSES)

    def forward(self, x):
        feat = self.hidden(self.backbone(x))
        grid = self.grid_head(feat).view(-1, MAX_LANES, NUM_ROWS, NUM_GRIDS + 1)
        color = self.color_head(feat).view(-1, MAX_LANES, COLOR_CLASSES)
        return {"grid_logits": grid, "color_logits": color}


# ── losses ──────────────────────────────────────────────────
def structure_loss(grid_logits, targets, no_lane_idx):
    """Smoothness: adjacent rows should have similar x-positions."""
    probs = F.softmax(grid_logits[..., :no_lane_idx], dim=-1)
    grid_range = torch.arange(no_lane_idx, device=grid_logits.device, dtype=probs.dtype)
    expected = (probs * grid_range).sum(dim=-1)  # [B, L, R]
    valid = targets != no_lane_idx
    if valid.sum() < 2:
        return grid_logits.new_tensor(0.0)
    # First-order difference between adjacent rows
    diff = expected[:, :, 1:] - expected[:, :, :-1]
    adj_valid = valid[:, :, 1:] & valid[:, :, :-1]
    if adj_valid.sum() == 0:
        return grid_logits.new_tensor(0.0)
    return diff[adj_valid].abs().mean() / max(float(no_lane_idx), 1.0)


# ── prediction helpers ─────────────────────────────────────
def row_anchors():
    start = int(IMAGE_H * ROI_TOP)
    end = IMAGE_H - 8
    return [round(start + i * (end - start) / (NUM_ROWS - 1)) for i in range(NUM_ROWS)]


def predict_lanes(model, image_bgr, device, conf_thr=0.3):
    """Detect lanes in an image. Returns list of {class, endpoints, conf}."""
    h, w = image_bgr.shape[:2]
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    t = t.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(t)

    grid_probs = F.softmax(out["grid_logits"][0], dim=-1)
    color_probs = F.softmax(out["color_logits"][0], dim=-1)

    rows_y = row_anchors()
    results = []

    for li in range(MAX_LANES):
        color_id = color_probs[li].argmax().item()
        if color_id == 2:  # none
            continue
        cls = "white_lane" if color_id == 0 else "yellow_lane"

        confs, labels = grid_probs[li].max(dim=-1)
        points = []
        for ri, (conf, label) in enumerate(zip(confs, labels)):
            if label.item() == NO_LANE or conf.item() < conf_thr:
                continue
            x = label.item() / (NUM_GRIDS - 1) * (w - 1)
            y = rows_y[ri] / IMAGE_H * h
            points.append([float(x), float(y)])

        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.float32)
        fitted = fit_line_from_points(pts)
        if fitted is None:
            continue
        angle, endpoints, bbox = fitted

        results.append({
            "class": cls,
            "conf": float(confs[labels != NO_LANE].mean().item()) if (labels != NO_LANE).any() else 0.5,
            "angle_deg": float(angle),
            "endpoints": endpoints,
            "bbox": [float(v) for v in bbox],
            "row_points": points,
        })

    return results


# ── training ────────────────────────────────────────────────
def train(args):
    device = f"cuda:{args.device}" if args.device.isdigit() and torch.cuda.is_available() else args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load GT labels
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.image_dir)
    items = []

    for label_path in sorted(gt_dir.glob("*.txt")):
        stem = label_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                img_path = str(p)
                break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        lanes = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            # Normalized coords
            xs_norm = [coords[i] for i in range(0, len(coords), 2)]
            ys_norm = [coords[i+1] for i in range(0, len(coords), 2)]
            lanes.append({
                "class": "white_lane" if cls_id == 0 else "yellow_lane",
                "x_norm": xs_norm,
                "y_norm": ys_norm,
                "endpoints_norm": (xs_norm[0], ys_norm[0], xs_norm[-1], ys_norm[-1]),
            })

        items.append({"image": img_path, "lanes": lanes})

    print(f"Loaded {len(items)} images, {sum(len(it['lanes']) for it in items)} GT lanes")

    # Split
    random.shuffle(items)
    n_train = int(len(items) * 0.8)
    train_ds = LaneDataset(items[:n_train], augment=True)
    val_ds = LaneDataset(items[n_train:], augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = TinyLaneCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val = float("inf")
    best_state = None

    # Class weights for grid: downweight no-lane (very frequent)
    grid_weight = torch.ones(NUM_GRIDS + 1, device=device)
    grid_weight[NO_LANE] = 0.2

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            img = batch["image"].to(device)
            row_tgt = batch["row_targets"].to(device)
            color_tgt = batch["color_targets"].to(device)

            out = model(img)
            grid_logits = out["grid_logits"]
            color_logits = out["color_logits"]

            # Grid loss — only compute on lanes that exist
            exists = batch["exists"].to(device).bool()  # [B, L]
            mask = exists.unsqueeze(-1).expand(-1, -1, NUM_ROWS).reshape(-1)
            if mask.sum() > 0:
                grid_loss = F.cross_entropy(
                    grid_logits.reshape(-1, NUM_GRIDS + 1)[mask],
                    row_tgt.reshape(-1)[mask],
                )
            else:
                grid_loss = grid_logits.new_tensor(0.0)

            # Color loss — only on lanes that exist
            if exists.sum() > 0:
                color_loss = F.cross_entropy(
                    color_logits.reshape(-1, COLOR_CLASSES)[exists.reshape(-1)],
                    color_tgt.reshape(-1)[exists.reshape(-1)],
                )
            else:
                color_loss = grid_logits.new_tensor(0.0)

            # Structure smoothness
            struct_loss = structure_loss(grid_logits, row_tgt, NO_LANE)

            loss = grid_loss + 0.5 * color_loss + 0.1 * struct_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(device)
                out = model(img)
                exists = batch["exists"].to(device).bool()
                mask = exists.unsqueeze(-1).expand(-1, -1, NUM_ROWS).reshape(-1)
                gl = F.cross_entropy(out["grid_logits"].reshape(-1, NUM_GRIDS + 1)[mask],
                                     batch["row_targets"].to(device).reshape(-1)[mask]) if mask.sum() > 0 else out["grid_logits"].new_tensor(0.0)
                cl = F.cross_entropy(out["color_logits"].reshape(-1, COLOR_CLASSES)[exists.reshape(-1)],
                                     batch["color_targets"].to(device).reshape(-1)[exists.reshape(-1)]) if exists.sum() > 0 else out["grid_logits"].new_tensor(0.0)
                val_losses.append((gl + 0.5 * cl).item())

        train_l = np.mean(train_losses)
        val_l = np.mean(val_losses)
        print(f"  epoch={epoch:03d}  train_loss={train_l:.4f}  val_loss={val_l:.4f}")

        if val_l < best_val:
            best_val = val_l
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "best_val_loss": best_val,
        "config": {"image_w": IMAGE_W, "image_h": IMAGE_H, "num_rows": NUM_ROWS,
                    "num_grids": NUM_GRIDS, "max_lanes": MAX_LANES, "roi_top": ROI_TOP},
    }, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    train(parse_args())
