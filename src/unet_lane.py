"""U-Net semantic segmentation for lane-line detection.

Replaces the failed HSV+Hough pipeline. GT line annotations → masks →
U-Net training → connected components → fitLine → evaluation.

Single-file implementation for neural network course final project.
"""

from __future__ import annotations

import argparse, json, math, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .geometry import fit_line_from_points

# ── constants ──────────────────────────────────────────────
IMG_SIZE = 512          # resize shorter side to this
NUM_CLASSES = 3         # 0=bg, 1=white_lane, 2=yellow_lane
LINE_THICKNESS = 10     # cv2.line thickness for mask rendering
BATCH_SIZE = 4
EPOCHS = 80
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════
# 1.  Mask generation from GT line annotations
# ═══════════════════════════════════════════════════════════════
def make_masks(image_dir, gt_dir, out_dir):
    """Convert per-image GT line files into 3-class segmentation masks."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    masks = {}

    for label_path in sorted(Path(gt_dir).glob("*.txt")):
        stem = label_path.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = Path(image_dir) / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Render GT lines onto a 3-channel mask image
        mask = np.zeros((h, w), dtype=np.uint8)  # 0=bg
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            pts = [(int(coords[i] * w), int(coords[i + 1] * h))
                   for i in range(0, len(coords), 2)]
            if len(pts) >= 2:
                cv2.line(mask, pts[0], pts[-1], cls_id + 1, LINE_THICKNESS, cv2.LINE_AA)

        save_path = out_path / f"{stem}.png"
        cv2.imwrite(str(save_path), mask)
        masks[stem] = str(save_path)

    print(f"Generated {len(masks)} masks → {out_dir}")
    return masks


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset with augmentation
# ═══════════════════════════════════════════════════════════════
class LaneMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, stems, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.stems = stems
        self.augment = augment

    def __len__(self):
        return len(self.stems)

    @staticmethod
    def _resize(img, mask, size):
        h, w = img.shape[:2]
        scale = size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Random crop or center crop to square
        if new_w > size:
            x0 = random.randint(0, new_w - size) if random.random() < 0.5 else (new_w - size) // 2
            img = img[:, x0:x0 + size]
            mask = mask[:, x0:x0 + size]
        if new_h > size:
            y0 = random.randint(0, new_h - size) if random.random() < 0.5 else (new_h - size) // 2
            img = img[y0:y0 + size]
            mask = mask[y0:y0 + size]
        # Pad if needed
        if img.shape[0] < size or img.shape[1] < size:
            pad_h = max(0, size - img.shape[0])
            pad_w = max(0, size - img.shape[1])
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return img[:size, :size], mask[:size, :size]

    def __getitem__(self, idx):
        stem = self.stems[idx]

        # Find image
        img = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = self.image_dir / f"{stem}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                break
        if img is None:
            raise FileNotFoundError(f"No image for {stem}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        # Augmentation
        if self.augment:
            # Random brightness/contrast
            if random.random() < 0.5:
                alpha = 0.8 + random.random() * 0.4
                beta = random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            # Random blur
            if random.random() < 0.3:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            # Horizontal flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Small rotation
            if random.random() < 0.5:
                angle = random.uniform(-8, 8)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Resize to fixed size
        img, mask = self._resize(img, mask, IMG_SIZE)

        # To tensor
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {"image": img_t, "mask": mask_t, "stem": stem}


# ═══════════════════════════════════════════════════════════════
# 3.  U-Net model
# ═══════════════════════════════════════════════════════════════
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            in_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.decoders.append(DoubleConv(f * 2, f))  # concat skip

        # Output
        self.out_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)


# ═══════════════════════════════════════════════════════════════
# 4.  Losses
# ═══════════════════════════════════════════════════════════════
def dice_loss(pred, target, eps=1e-6):
    """Soft Dice loss for multi-class."""
    pred_soft = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


# ═══════════════════════════════════════════════════════════════
# 5.  Training
# ═══════════════════════════════════════════════════════════════
def compute_pixel_metrics(pred, target):
    """Pixel-level accuracy, mIoU, dice."""
    pred_cls = pred.argmax(1)
    # Accuracy
    acc = (pred_cls == target).float().mean().item()

    # Per-class IoU
    ious = []
    dices = []
    for c in range(NUM_CLASSES):
        pred_c = (pred_cls == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        ious.append(((inter + 1e-6) / (union + 1e-6)).item())
        dices.append(((2 * inter + 1e-6) / (pred_c.sum() + target_c.sum() + 1e-6)).item())

    return acc, ious, dices


def train(args):
    # ── Generate masks ──
    mask_dir = "datasets/local_colm/masks"
    make_masks(args.image_dir, args.gt_dir, mask_dir)

    # ── Split ──
    stems = sorted(Path(mask_dir).glob("*.png"))
    stems = [p.stem for p in stems]
    random.seed(42)
    random.shuffle(stems)
    n_train = max(1, int(len(stems) * 0.75))
    train_stems = stems[:n_train]
    val_stems = stems[n_train:]
    print(f"Train: {len(train_stems)}, Val: {len(val_stems)}")

    train_ds = LaneMaskDataset(args.image_dir, mask_dir, train_stems, augment=True)
    val_ds = LaneMaskDataset(args.image_dir, mask_dir, val_stems, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ──
    model = UNet(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    # Class weights (bg is dominant)
    class_weight = torch.tensor([0.3, 1.5, 2.0], device=DEVICE)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            img = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = model(img)
            ce_loss = F.cross_entropy(logits, mask, weight=class_weight)
            d_loss = dice_loss(logits, mask)
            loss = ce_loss + 0.5 * d_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        sched.step()

        # ── Validate ──
        model.eval()
        val_losses = []
        all_acc, all_ious, all_dices = [], [], []
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                logits = model(img)
                ce = F.cross_entropy(logits, mask, weight=class_weight)
                dl = dice_loss(logits, mask)
                val_losses.append((ce + 0.5 * dl).item())
                acc, ious, dices = compute_pixel_metrics(logits, mask)
                all_acc.append(acc)
                all_ious.append(ious)
                all_dices.append(dices)

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        miou = np.mean([np.mean(io) for io in all_ious])
        mdice = np.mean([np.mean(di) for di in all_dices])
        print(f"  epoch={epoch:03d}  train_loss={tl:.4f}  val_loss={vl:.4f}  mIoU={miou:.4f}  mDice={mdice:.4f}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ── Save ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state or model.state_dict(), "best_val_loss": best_val}, out_path)
    print(f"Saved {out_path}  (best val_loss={best_val:.4f})")


# ═══════════════════════════════════════════════════════════════
# 6.  Inference + line fitting
# ═══════════════════════════════════════════════════════════════
def predict_mask(model, image_bgr):
    """Run U-Net inference on a full image, return class mask (H×W)."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = IMG_SIZE / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop/pad to square
    if new_w > IMG_SIZE:
        x0 = (new_w - IMG_SIZE) // 2
        img_r = img_r[:, x0:x0 + IMG_SIZE]
    if new_h > IMG_SIZE:
        y0 = (new_h - IMG_SIZE) // 2
        img_r = img_r[y0:y0 + IMG_SIZE]
    if img_r.shape[0] < IMG_SIZE or img_r.shape[1] < IMG_SIZE:
        pad_h = max(0, IMG_SIZE - img_r.shape[0])
        pad_w = max(0, IMG_SIZE - img_r.shape[1])
        img_r = cv2.copyMakeBorder(img_r, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    img_r = img_r[:IMG_SIZE, :IMG_SIZE]

    # Normalize
    t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
    t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(t)
    pred = logits[0].argmax(0).cpu().numpy().astype(np.uint8)

    # Resize back to original
    pred = cv2.resize(pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    # Reverse crop/pad to match resized image
    pred_full = np.zeros((new_h, new_w), dtype=np.uint8)
    y0 = max(0, (new_h - IMG_SIZE) // 2)
    x0 = max(0, (new_w - IMG_SIZE) // 2)
    pred_full[y0:y0 + min(IMG_SIZE, new_h - y0), x0:x0 + min(IMG_SIZE, new_w - x0)] = \
        pred[:min(IMG_SIZE, new_h - y0), :min(IMG_SIZE, new_w - x0)]

    # Resize to original
    pred_orig = cv2.resize(pred_full, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_orig


def mask_to_lines(mask, min_area=100, min_length=60):
    """Extract lane lines from segmentation mask via connected components + fitLine."""
    results = []
    for cls_id, cls_name in [(1, "white_lane"), (2, "yellow_lane")]:
        binary = (mask == cls_id).astype(np.uint8)
        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            # Get component mask
            comp_mask = (labels == i)
            ys, xs = np.where(comp_mask)
            if len(xs) < 10:
                continue

            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            fitted = fit_line_from_points(pts)
            if fitted is None:
                continue

            angle, endpoints, bbox = fitted
            length = math.hypot(endpoints[1][0] - endpoints[0][0], endpoints[1][1] - endpoints[0][1])
            if length < min_length:
                continue

            results.append({
                "class": cls_name,
                "conf": min(1.0, area / 1000.0),
                "angle_deg": float(angle),
                "endpoints": endpoints,
                "bbox": [float(v) for v in bbox],
                "area": int(area),
            })

    return results


def predict_all(model, image_dir, out_path, vis_dir=None):
    """Run U-Net inference on all images, output JSON + visualizations."""
    results = {}
    for img_path in tqdm(sorted(Path(image_dir).rglob("*.jpg")), desc="UNet predict"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        mask = predict_mask(model, img)
        lines = mask_to_lines(mask)

        results[img_path.name] = {
            "width": img.shape[1], "height": img.shape[0],
            "instances": lines,
        }

        if vis_dir:
            vis = img.copy()
            for l in lines:
                color = (0, 0, 255) if l["class"] == "white_lane" else (255, 0, 0)
                p1 = tuple(int(round(v)) for v in l["endpoints"][0])
                p2 = tuple(int(round(v)) for v in l["endpoints"][1])
                cv2.line(vis, p1, p2, color, 6, cv2.LINE_AA)
            (Path(vis_dir) / img_path.name).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(vis_dir) / img_path.name), vis)

    json.dump({"meta": {"method": "unet_semantic_segmentation"}, "images": results},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved predictions → {out_path}")


# ═══════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="U-Net lane segmentation")
    sub = p.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train")
    t.add_argument("--image-dir", default="datasets/local_colm/images/test")
    t.add_argument("--gt-dir", default="datasets/local_colm/labels/test")
    t.add_argument("--out", default="models/unet_lane.pt")

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--weights", default="models/unet_lane.pt")
    pr.add_argument("--source", default="datasets/local_colm/images/test")
    pr.add_argument("--out", default="runs/unet_predictions.json")
    pr.add_argument("--save-vis", default="runs/unet_vis")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
        model = UNet(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        predict_all(model, args.source, args.out, args.save_vis)
    else:
        print("Usage: python -m src.unet_lane train|predict")


if __name__ == "__main__":
    main()
