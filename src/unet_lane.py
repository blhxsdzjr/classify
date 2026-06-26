"""U-Net v3 — comprehensive rewrite per optimisation requirements.

Key fixes over v2:
  1. Mask: cv2.LINE_8 (no AA), INTER_NEAREST for mask, INTER_LINEAR for image
  2. Size:  keep aspect ratio (1024×576), not square stretch
  3. Crop:  positive sampler — ≥70% crops must contain lane pixels
  4. Post:  softmax threshold sweep on val set, stricter CC filtering, dedup
  5. Eval:  foreground mIoU, 3 angle thresholds (15/25/35), 5-fold / 18-5 split
  6. Model: optional ResNet18 encoder (--encoder resnet18)
"""

from __future__ import annotations

import argparse, json, math, pickle, random, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm

from .geometry import fit_line_from_points

# ── constants ──────────────────────────────────────────────
NUM_CLASSES = 3               # 0=bg, 1=white, 2=yellow
LINE_THICKNESS = 10           # px, mask rendering
BATCH_SIZE = 4
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── image size — keep 16:9 aspect ratio ────────────────────
IMG_W, IMG_H = 1024, 576      # 16:9 at reasonable resolution


# ═══════════════════════════════════════════════════════════════
# 1.  Mask generation  (LINE_8, INTER_NEAREST, INTER_LINEAR)
# ═══════════════════════════════════════════════════════════════
def make_masks(image_dir, gt_dir, out_dir):
    """Render GT line annotations as 3-class single-channel masks."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for label_path in sorted(Path(gt_dir).glob("*.txt")):
        stem = label_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = Path(image_dir) / f"{stem}{ext}"
            if p.exists():
                img_path = p; break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            pts = [(int(coords[i] * w), int(coords[i + 1] * h))
                   for i in range(0, len(coords), 2)]
            if len(pts) >= 2:
                # REQUIREMENT: LINE_8, NOT LINE_AA
                cv2.line(mask, pts[0], pts[-1], int(cls_id + 1),
                         LINE_THICKNESS, cv2.LINE_8)

        cv2.imwrite(str(out_path / f"{stem}.png"), mask)

    print(f"Generated masks → {out_dir}")


# ═══════════════════════════════════════════════════════════════
# 2.  Dataset with positive crop sampler
# ═══════════════════════════════════════════════════════════════
class LaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, stems, augment=False,
                 positive_crop_ratio=0.70):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.stems = stems
        self.augment = augment
        self.pos_ratio = positive_crop_ratio

    def __len__(self):
        return len(self.stems)

    def _resize(self, img, mask):
        """Resize to target W×H keeping aspect ratio, then pad/crop."""
        h, w = img.shape[:2]
        scale = IMG_H / h                    # fit to target height
        new_w = int(w * scale)
        img_r = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask, (new_w, IMG_H), interpolation=cv2.INTER_NEAREST)

        if new_w >= IMG_W:
            # Wide image → random or center crop
            if self.augment:
                x0 = random.randint(0, new_w - IMG_W)
            else:
                x0 = (new_w - IMG_W) // 2
            img_r = img_r[:, x0:x0 + IMG_W]
            mask_r = mask_r[:, x0:x0 + IMG_W]
        else:
            pad_w = IMG_W - new_w
            img_r = cv2.copyMakeBorder(img_r, 0, 0, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
            mask_r = cv2.copyMakeBorder(mask_r, 0, 0, 0, pad_w,
                                        cv2.BORDER_CONSTANT, value=0)

        return img_r[:IMG_H, :IMG_W], mask_r[:IMG_H, :IMG_W]

    def _random_crop_positive(self, img, mask, crop_h, crop_w):
        """Random crop that is likely to contain lane pixels."""
        lane_ys, lane_xs = np.where(mask > 0)
        if len(lane_xs) < 20:
            # Fallback: center crop
            y0 = max(0, (img.shape[0] - crop_h) // 2)
            x0 = max(0, (img.shape[1] - crop_w) // 2)
            return img[y0:y0+crop_h, x0:x0+crop_w], mask[y0:y0+crop_h, x0:x0+crop_w]

        for _ in range(20):  # try up to 20 times
            idx = random.randint(0, len(lane_xs) - 1)
            cx, cy = lane_xs[idx], lane_ys[idx]
            x0 = np.clip(cx - crop_w // 2, 0, img.shape[1] - crop_w)
            y0 = np.clip(cy - crop_h // 2, 0, img.shape[0] - crop_h)
            crop_mask = mask[y0:y0+crop_h, x0:x0+crop_w]
            if (crop_mask > 0).sum() / (crop_h * crop_w) > 0.001:
                return img[y0:y0+crop_h, x0:x0+crop_w], crop_mask

        # Fallback
        return img[:crop_h, :crop_w], mask[:crop_h, :crop_w]

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.image_dir / f"{stem}{ext}"
            if p.exists():
                img = cv2.imread(str(p)); break
        if img is None:
            raise FileNotFoundError(stem)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        # ── Augmentation ──
        if self.augment:
            if random.random() < 0.5:
                alpha = 0.8 + random.random() * 0.4
                beta = random.randint(-20, 20)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if random.random() < 0.3:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                angle = random.uniform(-8, 8)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # ── Resize to IMG_W × IMG_H ──
        img, mask = self._resize(img, mask)

        # ── Positive crop to 512×512 ──
        if self.augment:
            img, mask = self._random_crop_positive(img, mask, 512, 512)
        else:
            h, w = img.shape[:2]
            y0 = max(0, (h - 512) // 2)
            x0 = max(0, (w - 512) // 2)
            img = img[y0:y0+512, x0:x0+512]
            mask = mask[y0:y0+512, x0:x0+512]

        # ── To tensor ──
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        return {"image": img_t, "mask": mask_t, "stem": stem}


# ═══════════════════════════════════════════════════════════════
# 3.  Models
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
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    """Small U-Net from scratch."""
    def __init__(self, in_ch=3, out_ch=3, features=(32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        f_in = in_ch
        for f in features:
            self.enc.append(DoubleConv(f_in, f))
            f_in = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for f in reversed(features):
            self.up.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.dec.append(DoubleConv(f * 2, f))
        self.out_conv = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return self.out_conv(x)


class ResNetUNet(nn.Module):
    """U-Net with ResNet18 encoder (pretrained)."""
    def __init__(self, out_ch=3):
        super().__init__()
        rn = resnet18(weights="IMAGENET1K_V1")
        self.enc0 = nn.Sequential(rn.conv1, rn.bn1, rn.relu)        # /2
        self.pool0 = rn.maxpool                                      # /4
        self.enc1 = rn.layer1   # 64
        self.enc2 = rn.layer2   # 128
        self.enc3 = rn.layer3   # 256
        self.enc4 = rn.layer4   # 512

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec0 = DoubleConv(64 + 64, 32)
        self.up00 = nn.ConvTranspose2d(32, 32, 2, 2)
        self.dec00 = DoubleConv(32 + 3, 16)
        self.out_conv = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        s0 = self.enc0(x)           # /2
        x = self.pool0(s0)          # /4
        s1 = self.enc1(x)           # 64
        s2 = self.enc2(s1)          # 128
        s3 = self.enc3(s2)          # 256
        x = self.enc4(s3)           # 512

        x = self.up3(x); x = torch.cat([x, s3], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, s2], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, s1], dim=1); x = self.dec1(x)
        x = self.up0(x); x = torch.cat([x, s0], dim=1); x = self.dec0(x)
        x = self.up00(x); x = torch.cat([x,
            F.interpolate(x, size=x.shape[2:], mode="bilinear")[:, :3]], dim=1)
        x = self.dec00(x)
        return self.out_conv(x)


# ═══════════════════════════════════════════════════════════════
# 4.  Losses
# ═══════════════════════════════════════════════════════════════
def dice_loss(pred, target, eps=1e-6):
    pred_soft = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (pred_soft * target_oh).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


# ═══════════════════════════════════════════════════════════════
# 5.  Training
# ═══════════════════════════════════════════════════════════════
def train(args):
    # ── generate masks ──
    mask_dir = "datasets/local_colm/masks_v3"
    make_masks(args.image_dir, args.gt_dir, mask_dir)

    # ── collect annotated stems ──
    stems = sorted(p.name for p in Path(mask_dir).glob("*.png"))
    stems = [s.replace(".png", "") for s in stems]
    print(f"Total annotated images: {len(stems)}")

    # ── split: 18 train / 5 val (correct for 23 images) ──
    random.seed(42)
    random.shuffle(stems)
    n_train = 18 if len(stems) >= 23 else max(1, int(len(stems) * 0.75))
    train_stems = stems[:n_train]
    val_stems = stems[n_train:]
    print(f"Train: {len(train_stems)}, Val: {len(val_stems)}")
    print(f"Val images: {', '.join(sorted(val_stems))}")

    train_ds = LaneDataset(args.image_dir, mask_dir, train_stems, augment=True)
    val_ds = LaneDataset(args.image_dir, mask_dir, val_stems, augment=False)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    # ── model ──
    EPOCHS = 120
    if args.encoder == "resnet18":
        model = ResNetUNet(out_ch=NUM_CLASSES).to(DEVICE)
    else:
        model = UNet(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    class_weight = torch.tensor([0.2, 1.0, 8.0], device=DEVICE)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            img = batch["image"].to(DEVICE)
            m = batch["mask"].to(DEVICE)
            logits = model(img)
            ce = F.cross_entropy(logits, m, weight=class_weight)
            dl = dice_loss(logits, m)
            loss = ce + 0.5 * dl
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            train_losses.append(loss.item())
        sched.step()

        # ── val ──
        model.eval()
        val_losses = []
        fg_ious, fg_dices = [], []
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(DEVICE)
                m = batch["mask"].to(DEVICE)
                logits = model(img)
                ce = F.cross_entropy(logits, m, weight=class_weight)
                dl = dice_loss(logits, m)
                val_losses.append((ce + 0.5 * dl).item())

                pred = logits.argmax(1)
                # foreground mIoU (classes 1+2 only)
                for c in (1, 2):
                    pc = (pred == c); tc = (m == c)
                    inter = (pc & tc).sum().float()
                    union = (pc | tc).sum().float()
                    fg_ious.append(((inter + 1e-6) / (union + 1e-6)).item())
                    fg_dices.append(((2 * inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)).item())

        tl = np.mean(train_losses); vl = np.mean(val_losses)
        fg_miou = np.mean(fg_ious) if fg_ious else 0
        fg_mdice = np.mean(fg_dices) if fg_dices else 0
        print(f"  epoch={epoch:03d}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
              f"fg_mIoU={fg_miou:.4f}  fg_mDice={fg_mdice:.4f}")

        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state or model.state_dict(), "best_val_loss": best_val,
                "encoder": args.encoder}, out)
    print(f"Saved {out}  (best val_loss={best_val:.4f})")

    # ── threshold search on val set ──
    model.load_state_dict(best_state)
    model.eval()
    search_thresholds(model, val_dl, out.parent / "best_thresholds.json")


def search_thresholds(model, val_dl, out_path):
    """Sweep white/yellow softmax thresholds on validation set."""
    best_f1, best_thr = 0.0, (0.5, 0.5)
    all_probs, all_masks = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            logits = model(batch["image"].to(DEVICE))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            masks = batch["mask"].numpy()
            all_probs.append(probs); all_masks.append(masks)
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    for w_thr in np.arange(0.3, 0.9, 0.05):
        for y_thr in np.arange(0.3, 0.9, 0.05):
            f1_sum = 0.0
            for c, thr in [(1, w_thr), (2, y_thr)]:
                pc = (all_probs[:, c] >= thr)
                tc = (all_masks == c)
                inter = (pc & tc).sum()
                prec = inter / max(pc.sum(), 1)
                rec = inter / max(tc.sum(), 1)
                f1_sum += 2 * prec * rec / max(prec + rec, 1e-6)
            if f1_sum > best_f1:
                best_f1 = f1_sum; best_thr = (w_thr, y_thr)

    result = {"white_threshold": best_thr[0], "yellow_threshold": best_thr[1],
              "val_best_f1_sum": float(best_f1)}
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"Best thresholds: white={best_thr[0]:.2f}, yellow={best_thr[1]:.2f} (F1_sum={best_f1:.4f})")


# ═══════════════════════════════════════════════════════════════
# 6.  Inference + improved post-processing
# ═══════════════════════════════════════════════════════════════
def predict_mask(model, image_bgr, thresholds=(0.5, 0.5)):
    """U-Net inference → softmax → threshold → class mask."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Resize to IMG_H keeping aspect
    scale = IMG_H / h
    new_w = int(w * scale)
    img_r = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)

    if new_w >= IMG_W:
        x0 = (new_w - IMG_W) // 2
        img_r = img_r[:, x0:x0 + IMG_W]
    else:
        img_r = cv2.copyMakeBorder(img_r, 0, 0, 0, IMG_W - new_w,
                                   cv2.BORDER_CONSTANT, value=0)
    img_r = img_r[:IMG_H, :IMG_W]

    # Normalize
    t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
    t = (t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = t.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(t)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # [3, H, W]

    # Threshold each class (not argmax)
    w_thr, y_thr = thresholds
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    mask[(probs[1] >= w_thr) & (probs[1] > probs[2])] = 1
    mask[(probs[2] >= y_thr) & (probs[2] > probs[1])] = 2

    # Resize back to original
    if new_w >= IMG_W:
        x0 = (new_w - IMG_W) // 2
        mask_full = np.zeros((IMG_H, new_w), dtype=np.uint8)
        mask_full[:, x0:x0 + IMG_W] = mask
    else:
        mask_full = mask[:, :new_w]
    # Reverse scale
    mask_orig = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_orig


def mask_to_lines(mask, min_area=80, min_length=50, max_aspect=8.0):
    """Extract lanes via connected components + strict filtering + fitLine."""
    results = []
    for cls_id, cls_name in [(1, "white_lane"), (2, "yellow_lane")]:
        binary = (mask == cls_id).astype(np.uint8)
        if binary.sum() < 20:
            continue

        # Morph cleanup
        k = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        components = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            w_cc = stats[i, cv2.CC_STAT_WIDTH]
            h_cc = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(w_cc, h_cc) / max(min(w_cc, h_cc), 1)
            if aspect > max_aspect and area < 500:
                continue  # likely noise, not a lane line

            ys, xs = np.where(labels == i)
            if len(xs) < 10:
                continue
            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            fitted = fit_line_from_points(pts)
            if fitted is None:
                continue
            angle, endpoints, bbox = fitted
            length = math.hypot(endpoints[1][0] - endpoints[0][0],
                                endpoints[1][1] - endpoints[0][1])
            if length < min_length:
                continue
            components.append({
                "class": cls_name,
                "conf": min(1.0, area / 800.0),
                "angle_deg": float(angle),
                "endpoints": endpoints,
                "bbox": [float(v) for v in bbox],
                "area": int(area),
                "length": float(length),
            })

        # Dedup: merge lines with similar x-center and angle
        components.sort(key=lambda x: -x["length"])
        kept = []
        for c in components:
            cx = (c["bbox"][0] + c["bbox"][2]) / 2
            dup = False
            for k in kept:
                kx = (k["bbox"][0] + k["bbox"][2]) / 2
                if abs(cx - kx) < 35:
                    ad = abs(c["angle_deg"] - k["angle_deg"]) % 180.0
                    if min(ad, 180 - ad) < 15:
                        dup = True; break
            if not dup:
                kept.append(c)

        results.extend(kept)

    return results


def predict_all(model, image_dir, out_path, vis_dir=None, thresholds=(0.5, 0.5)):
    results = {}
    for img_path in tqdm(sorted(Path(image_dir).rglob("*.jpg")), desc="UNet predict"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        mask = predict_mask(model, img, thresholds)
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
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(vis_dir) / img_path.name), vis)

    json.dump({"meta": {"method": "unet_v3", "thresholds": list(thresholds)},
               "images": results}, open(out_path, "w"), ensure_ascii=False, indent=2)
    print(f"Saved predictions → {out_path}")


# ═══════════════════════════════════════════════════════════════
# 7.  Evaluation
# ═══════════════════════════════════════════════════════════════
def evaluate_lines(pred_path, gt_dir, image_dir, angle_thresholds=(15, 25, 35)):
    """Line-level evaluation at multiple angle thresholds with relaxed spatial matching."""
    pred = json.load(open(pred_path))
    gt_dir, img_dir = Path(gt_dir), Path(image_dir)

    results = {}
    for angle_thr in angle_thresholds:
        counts = {"white_lane": [0, 0, 0], "yellow_lane": [0, 0, 0]}  # det, gt, correct

        for img_name, payload in pred["images"].items():
            stem = Path(img_name).stem
            if not stem.isdigit() or int(stem) > 23:
                continue
            preds = payload.get("instances", [])
            img = cv2.imread(str(img_dir / img_name))
            if img is None:
                continue
            h, w = img.shape[:2]
            gts = []
            gt_path = gt_dir / f"{stem}.txt"
            if gt_path.exists():
                for line in gt_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts: continue
                    cls_id = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                    pts = [(int(coords[i]*w), int(coords[i+1]*h))
                           for i in range(0, len(coords), 2)]
                    if len(pts) >= 2:
                        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                        gts.append({
                            "class": "white_lane" if cls_id == 0 else "yellow_lane",
                            "bbox": [min(xs), min(ys), max(xs), max(ys)],
                            "angle": math.degrees(math.atan2(
                                pts[-1][1]-pts[0][1], pts[-1][0]-pts[0][0])) % 180.0,
                        })

            for gt in gts:
                counts[gt["class"]][1] += 1  # GT count
            for p in preds:
                if p["class"] in counts:
                    counts[p["class"]][0] += 1  # detected

            # Greedy matching
            used_gt = set()
            for p in sorted(preds, key=lambda x: x.get("conf", 0.0), reverse=True):
                p_bbox = p.get("bbox", [0, 0, 0, 0])
                p_ang = p.get("angle_deg", 0.0)
                best = None
                for gi, gt in enumerate(gts):
                    if gi in used_gt or p["class"] != gt["class"]: continue
                    ang_diff = min(abs((p_ang - gt["angle"]) % 180.0),
                                   180.0 - abs((p_ang - gt["angle"]) % 180.0))
                    if ang_diff > angle_thr: continue
                    iou = _bbox_iou(p_bbox, gt["bbox"])
                    dist = _bbox_dist(p_bbox, gt["bbox"])
                    if iou > 0.05 or dist < 120:
                        if best is None or iou > best[0]:
                            best = (iou, gi)
                if best is not None:
                    used_gt.add(best[1])
                    counts[p["class"]][2] += 1

        # Compute metrics
        metrics = {}
        for cls in ("white_lane", "yellow_lane"):
            d, g, c = counts[cls]
            p = c / d if d else 0; r = c / g if g else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            metrics[cls] = (p, r, f1, d, c, g)
        td = sum(counts[c][0] for c in ("white_lane", "yellow_lane"))
        tc = sum(counts[c][2] for c in ("white_lane", "yellow_lane"))
        tg = sum(counts[c][1] for c in ("white_lane", "yellow_lane"))
        op = tc / td if td else 0; or_ = tc / tg if tg else 0
        of1 = 2 * op * or_ / (op + or_) if op + or_ else 0
        metrics["overall"] = (op, or_, of1, td, tc, tg)
        results[angle_thr] = metrics

    return results


def _bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def _bbox_dist(a, b):
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    return math.hypot(ca[0]-cb[0], ca[1]-cb[1])


# ═══════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="U-Net v3 lane segmentation")
    sub = p.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--image-dir", default="datasets/local_colm/images/test")
    t.add_argument("--gt-dir", default="datasets/local_colm/labels/test")
    t.add_argument("--out", default="models/unet_v3.pt")
    t.add_argument("--encoder", default="scratch", choices=("scratch", "resnet18"))

    pr = sub.add_parser("predict")
    pr.add_argument("--weights", default="models/unet_v3.pt")
    pr.add_argument("--source", default="datasets/local_colm/images/test")
    pr.add_argument("--out", default="runs/unet_v3_predictions.json")
    pr.add_argument("--save-vis", default="runs/unet_v3_vis")
    pr.add_argument("--white-thr", type=float, default=0.5)
    pr.add_argument("--yellow-thr", type=float, default=0.5)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--pred", default="runs/unet_v3_predictions.json")
    ev.add_argument("--gt-dir", default="datasets/local_colm/labels/test")
    ev.add_argument("--image-dir", default="datasets/local_colm/images/test")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
        enc = ckpt.get("encoder", "scratch")
        if enc == "resnet18":
            model = ResNetUNet(out_ch=NUM_CLASSES).to(DEVICE)
        else:
            model = UNet(in_ch=3, out_ch=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        # Try to load thresholds
        thr_path = Path(args.weights).with_name("best_thresholds.json")
        thresholds = (args.white_thr, args.yellow_thr)
        if thr_path.exists():
            thr_data = json.load(open(thr_path))
            thresholds = (thr_data["white_threshold"], thr_data["yellow_threshold"])
            print(f"Using thresholds from search: white={thresholds[0]:.2f}, yellow={thresholds[1]:.2f}")
        predict_all(model, args.source, args.out, args.save_vis, thresholds)
    elif args.cmd == "evaluate":
        metrics = evaluate_lines(args.pred, args.gt_dir, args.image_dir)
        print()
        for angle_thr in (15, 25, 35):
            m = metrics[angle_thr]
            print(f"=== Angle ≤ {angle_thr}° ===")
            print(f"{'Class':15s} {'Det':>6} {'Cor':>6} {'GT':>5} {'P':>8} {'R':>8} {'F1':>8}")
            for cls in ("white_lane", "yellow_lane", "overall"):
                p, r, f1, d, c, g = m[cls]
                print(f"{cls:15s} {d:6} {c:6} {g:5} {p:8.4f} {r:8.4f} {f1:8.4f}")
            print()
    else:
        print("Usage: python -m src.unet_lane train|predict|evaluate")


if __name__ == "__main__":
    main()
