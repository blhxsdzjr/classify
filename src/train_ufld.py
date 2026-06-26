from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from .ufld_model import UFLDConfig, build_model, structure_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UFLD-inspired row-anchor lane detector.")
    parser.add_argument("--index", default="datasets/local_colm/ufld_test_index.json")
    parser.add_argument("--out", default="models/ufld_tiny.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--structure-weight", type=float, default=0.15)
    parser.add_argument("--color-weight", type=float, default=0.5)
    return parser.parse_args()


def normalize_device(device: str) -> str:
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    return device


class UFLDDataset(Dataset):
    def __init__(self, index_path: Path, cfg: UFLDConfig) -> None:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        self.items = raw["items"]
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import cv2

        item = self.items[idx]
        image = cv2.imread(item["image"])
        if image is None:
            raise ValueError(f"Cannot read image: {item['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.cfg.image_width, self.cfg.image_height), interpolation=cv2.INTER_AREA)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - 0.5) / 0.5

        label = np.load(item["label"], allow_pickle=False)
        return {
            "image": image_t,
            "row_targets": torch.from_numpy(label["row_targets"].astype(np.int64)),
            "color_targets": torch.from_numpy(label["color_targets"].astype(np.int64)),
            "exists": torch.from_numpy(label["exists"].astype(np.float32)),
        }


def load_config(index_path: Path) -> UFLDConfig:
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    cfg_raw = raw["config"]
    return UFLDConfig(**{k: cfg_raw[k] for k in UFLDConfig.__dataclass_fields__ if k in cfg_raw})


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    val_len = max(1, int(round(len(dataset) * val_ratio))) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    if val_len == 0:
        return dataset, dataset
    return random_split(dataset, [train_len, val_len], generator=generator)


def batch_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: UFLDConfig, args) -> torch.Tensor:
    grid_logits = outputs["grid_logits"]
    color_logits = outputs["color_logits"]
    row_targets = batch["row_targets"]
    color_targets = batch["color_targets"]

    # Downweight no-lane class because it is frequent.
    weights = torch.ones(cfg.num_grids + 1, device=grid_logits.device)
    weights[cfg.no_lane_index] = 0.25
    grid_loss = F.cross_entropy(
        grid_logits.reshape(-1, cfg.num_grids + 1),
        row_targets.reshape(-1),
        weight=weights,
    )
    color_loss = F.cross_entropy(
        color_logits.reshape(-1, cfg.color_classes),
        color_targets.reshape(-1),
    )
    smooth_loss = structure_loss(grid_logits, row_targets, cfg.no_lane_index)
    return grid_loss + args.color_weight * color_loss + args.structure_weight * smooth_loss


@torch.no_grad()
def evaluate(model, loader, cfg: UFLDConfig, args, device: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(float(batch_loss(model(batch["image"]), batch, cfg, args).item()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    args = parse_args()
    args.device = normalize_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    index_path = Path(args.index)
    cfg = load_config(index_path)
    dataset = UFLDDataset(index_path, cfg)
    if len(dataset) == 0:
        raise ValueError("No training samples. Run src.generate_ufld_labels first.")
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    device = args.device
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = batch_loss(model(batch["image"]), batch, cfg, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = evaluate(model, val_loader, cfg, args, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.__dict__,
            "state_dict": best_state or model.state_dict(),
            "best_val_loss": best_val,
        },
        out_path,
    )
    print(f"Saved UFLD-style model to {out_path}")


if __name__ == "__main__":
    main()
