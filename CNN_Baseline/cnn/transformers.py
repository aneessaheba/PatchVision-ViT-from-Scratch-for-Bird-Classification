"""Outside reference models for the study, written from scratch with no pretrained weights.

PlainViT   the textbook ViT from the project proposal: 16x16 patches, 6 blocks, 6 heads, 384 dimensions, class token.
PVTv2B0    Pyramid Vision Transformer v2, B0 (Wang et al., Computational Visual Media 2022): embedding sizes 32, 64, 160,
           256; 2 blocks per stage; 1, 2, 5, 8 heads; sr ratios 8, 4, 2, 1; MLP ratios 8, 8, 4, 4.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CNNConfig


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y, y, y, need_weights=False)[0])
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PlainViT(nn.Module):
    def __init__(self, cfg: CNNConfig, patch: int = 16, dim: int = 384, depth: int = 6, heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        grid = cfg.img_size // patch
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        rates = torch.linspace(0, cfg.drop_path, depth).tolist()
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads, mlp_ratio, rate) for rate in rates)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, cfg.num_classes)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


class PVTAttention(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio) if sr_ratio > 1 else None
        self.norm = nn.LayerNorm(dim) if sr_ratio > 1 else None

    def forward(self, x, h, w):
        b, n, d = x.shape
        q = self.q(x).reshape(b, n, self.heads, -1).permute(0, 2, 1, 3)
        reduced = x
        if self.sr is not None:
            reduced = self.norm(self.sr(x.transpose(1, 2).reshape(b, d, h, w)).flatten(2).transpose(1, 2))
        k, v = self.kv(reduced).reshape(b, -1, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, d))


class ConvMlp(nn.Module):
    def __init__(self, dim: int, ratio: int):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, n, d = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(b, d, h, w)).flatten(2).transpose(1, 2)
        return self.fc2(self.act(x))


class PVTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, mlp_ratio: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PVTAttention(dim, heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ConvMlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class PVTv2B0(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        dims, depths, heads = (32, 64, 160, 256), (2, 2, 2, 2), (1, 2, 5, 8)
        sr_ratios, mlp_ratios = (8, 4, 2, 1), (8, 8, 4, 4)
        rates = iter(torch.linspace(0, cfg.drop_path, sum(depths)).tolist())
        self.embeds, self.stages, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        previous = 3
        for stage, dim in enumerate(dims):
            kernel, stride = (7, 4) if stage == 0 else (3, 2)
            self.embeds.append(nn.Sequential(nn.Conv2d(previous, dim, kernel, stride, kernel // 2)))
            self.stages.append(nn.ModuleList(
                PVTBlock(dim, heads[stage], sr_ratios[stage], mlp_ratios[stage], next(rates))
                for _ in range(depths[stage])))
            self.norms.append(nn.ModuleList([nn.LayerNorm(dim), nn.LayerNorm(dim)]))
            previous = dim
        self.head = nn.Linear(dims[-1], cfg.num_classes)
        self.apply(init_weights)

    def forward(self, x):
        for embed, blocks, (embed_norm, stage_norm) in zip(self.embeds, self.stages, self.norms):
            x = embed(x)
            b, d, h, w = x.shape
            tokens = embed_norm(x.flatten(2).transpose(1, 2))
            for block in blocks:
                tokens = block(tokens, h, w)
            tokens = stage_norm(tokens)
            x = tokens.transpose(1, 2).reshape(b, d, h, w)
        return self.head(tokens.mean(dim=1))
