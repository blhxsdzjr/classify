from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class UFLDConfig:
    image_width: int = 800
    image_height: int = 288
    num_rows: int = 18
    num_grids: int = 100
    max_lanes: int = 6
    color_classes: int = 3  # white, yellow, none

    @property
    def no_lane_index(self) -> int:
        return self.num_grids

    def row_anchors(self) -> list[int]:
        # Concentrate rows on road area, same spirit as UFLD row anchors.
        start = int(self.image_height * 0.42)
        end = self.image_height - 8
        if self.num_rows <= 1:
            return [end]
        return [round(start + i * (end - start) / (self.num_rows - 1)) for i in range(self.num_rows)]


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUFLD(nn.Module):
    """Small row-anchor lane detector inspired by Ultra Fast Lane Detection.

    It predicts, for each lane slot and row anchor, one horizontal grid cell or
    a special no-lane class. A separate head predicts white/yellow/none color.
    This is intentionally compact so it can train on a small class project
    dataset without a large external backbone.
    """

    def __init__(self, cfg: UFLDConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = nn.Sequential(
            ConvBNAct(3, 24, stride=2),
            ConvBNAct(24, 32, stride=2),
            ConvBNAct(32, 48, stride=2),
            ConvBNAct(48, 64, stride=2),
            ConvBNAct(64, 96, stride=2),
            ConvBNAct(96, 128, stride=2),
            nn.AdaptiveAvgPool2d((4, 10)),
        )
        feat_dim = 128 * 4 * 10
        hidden = 512
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.grid_classifier = nn.Linear(
            hidden,
            cfg.max_lanes * cfg.num_rows * (cfg.num_grids + 1),
        )
        self.color_classifier = nn.Linear(hidden, cfg.max_lanes * cfg.color_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.head(self.backbone(x))
        grid_logits = self.grid_classifier(feat).view(
            -1,
            self.cfg.max_lanes,
            self.cfg.num_rows,
            self.cfg.num_grids + 1,
        )
        color_logits = self.color_classifier(feat).view(
            -1,
            self.cfg.max_lanes,
            self.cfg.color_classes,
        )
        return {"grid_logits": grid_logits, "color_logits": color_logits}


def structure_loss(grid_logits: torch.Tensor, targets: torch.Tensor, no_lane_index: int) -> torch.Tensor:
    probs = F.softmax(grid_logits[..., :no_lane_index], dim=-1)
    grid = torch.arange(no_lane_index, device=grid_logits.device, dtype=probs.dtype)
    expected = (probs * grid).sum(dim=-1)
    valid = targets != no_lane_index
    if expected.shape[2] < 2:
        return expected.new_tensor(0.0)
    adjacent_valid = valid[:, :, 1:] & valid[:, :, :-1]
    if not torch.any(adjacent_valid):
        return expected.new_tensor(0.0)
    diffs = torch.abs(expected[:, :, 1:] - expected[:, :, :-1])
    return diffs[adjacent_valid].mean() / max(float(no_lane_index), 1.0)


def build_model(cfg: UFLDConfig) -> TinyUFLD:
    return TinyUFLD(cfg)
