"""AdamW parameter groups and the per-step warm-up + cosine learning-rate schedule."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TrainConfig

_NO_DECAY_KEYS = ("pos_embed", "cls_token", "gate_slope", "log_tau", "phi", ".U")


def build_optimizer(model: nn.Module, tc: TrainConfig) -> torch.optim.Optimizer:
    """No weight decay on norms / biases / scalars / position embeddings / head-mixing (standard practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in _NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=tc.betas)


def build_scheduler(optimizer, tc: TrainConfig, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-STEP: linear warm-up for `warmup_epochs`, then cosine decay from lr to min_lr."""
    total = tc.epochs * steps_per_epoch
    warm = max(1, tc.warmup_epochs * steps_per_epoch)
    floor = tc.min_lr / tc.lr

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
