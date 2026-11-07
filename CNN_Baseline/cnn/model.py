"""Compact ResNet written from scratch (no torchvision weights, no pretrained anything).

    stem   7x7 conv /2 + BN + ReLU + 3x3 max-pool /2        224 -> 56
    stage1 BasicBlock x2, width w1, stride 1                  56
    stage2 BasicBlock x2, width w2, stride 2                  28
    stage3 BasicBlock x2, width w3, stride 2                  14
    stage4 BasicBlock x2, width w4, stride 2                   7
    head   global average pool + Linear(w4, 200)

widths (39, 78, 156, 312) -> 4.22 M parameters (matched to the ViT);  (64, 128, 256, 512) -> 11.28 M (ResNet-18 layout).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CNNConfig
from .transformers import DropPath, PlainViT, PVTv2B0


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, drop_path: float = 0.0):
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(self.drop_path(out) + idt)


class CompactResNet(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        w = cfg.widths
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 7, 2, 3, bias=False), nn.BatchNorm2d(w[0]),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        stages, cin = [], w[0]
        rates = iter(torch.linspace(0, cfg.resnet_drop_path, sum(cfg.blocks)).tolist())
        for i, (c, n) in enumerate(zip(w, cfg.blocks)):
            blocks = []
            for j in range(n):
                blocks.append(BasicBlock(cin, c, stride=(2 if (i > 0 and j == 0) else 1), drop_path=next(rates)))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.resnet_dropout) if cfg.resnet_dropout > 0 else nn.Identity()
        self.head = nn.Linear(w[-1], cfg.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if cfg.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.head(self.dropout(torch.flatten(self.pool(x), 1)))


def build_model(cfg: CNNConfig) -> nn.Module:
    if cfg.arch == "resnet":
        return CompactResNet(cfg)
    return {"plain_vit": PlainViT, "pvtv2_b0": PVTv2B0}[cfg.arch](cfg)
