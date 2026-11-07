"""Metrics: top-k, macro P/R/F1, confusion matrix, calibration (ECE), attention-in-box, efficiency."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_parameters


def box_to_patch_mask(mask: torch.Tensor, grid: int) -> torch.Tensor:
    """(B,1,H,W) bird mask in [0,1] -> (B, grid*grid) fraction of each patch covered by the box."""
    return F.adaptive_avg_pool2d(mask.float(), grid).flatten(1)


def calibration_bins(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> Tuple[float, List[dict]]:
    """Expected calibration error (%) and the per-bin table used for the reliability diagram."""
    edges = torch.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c, frac = correct[m].float().mean().item(), conf[m].mean().item(), m.float().mean().item()
            ece += frac * abs(acc - c)
            rows.append({"bin_lo": lo.item(), "bin_hi": hi.item(), "confidence": c, "accuracy": acc, "fraction": frac})
    return 100.0 * ece, rows


class MetricTracker:
    """Accumulates loss / top-1 / top-5 / confusion matrix (/ calibration) over an epoch (hard labels only)."""

    def __init__(self, num_classes: int, track_calibration: bool = False):
        self.nc, self.track_cal = num_classes, track_calibration
        self.reset()

    def reset(self):
        self.loss_sum, self.n, self.top1, self.top5 = 0.0, 0, 0, 0
        self.conf = torch.zeros(self.nc, self.nc, dtype=torch.long)
        self.mass_sum, self.mass_n, self.area_sum = 0.0, 0, 0.0
        self.gate_mean_sum, self.gate_sp_sum, self.gate_n = 0.0, 0.0, 0
        self.confs: List[torch.Tensor] = []
        self.corrects: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, logits, targets, loss: float, mass=None, valid=None, area=None, gate=None):
        b = targets.shape[0]
        k = min(5, logits.shape[1])
        pred_top = logits.topk(k, dim=1).indices
        self.top1 += (pred_top[:, 0] == targets).sum().item()
        self.top5 += (pred_top == targets.unsqueeze(1)).any(1).sum().item()
        self.loss_sum += float(loss) * b
        self.n += b
        idx = targets.cpu() * self.nc + pred_top[:, 0].cpu()
        self.conf += torch.bincount(idx, minlength=self.nc * self.nc).view(self.nc, self.nc)
        if self.track_cal:
            self.confs.append(torch.softmax(logits.float(), -1).max(1).values.cpu())
            self.corrects.append((pred_top[:, 0] == targets).cpu())
        if mass is not None and valid is not None and valid.sum() > 0:
            self.mass_sum += (mass * valid).sum().item()
            self.mass_n += int(valid.sum().item())
            if area is not None:
                self.area_sum += (area * valid).sum().item()
        if gate is not None:
            self.gate_mean_sum += float(gate[0]) * b
            self.gate_sp_sum += float(gate[1]) * b
            self.gate_n += b

    def compute(self) -> Dict[str, float]:
        conf = self.conf.double()
        tp, support, pred_cnt = conf.diag(), conf.sum(1), conf.sum(0)
        prec = torch.where(pred_cnt > 0, tp / pred_cnt.clamp_min(1), torch.zeros_like(tp))
        rec = torch.where(support > 0, tp / support.clamp_min(1), torch.zeros_like(tp))
        f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp_min(1e-12), torch.zeros_like(tp))
        present = support > 0
        out = {
            "loss": self.loss_sum / max(1, self.n),
            "acc1": 100.0 * self.top1 / max(1, self.n),
            "acc5": 100.0 * self.top5 / max(1, self.n),
            "macro_precision": 100.0 * prec[present].mean().item(),
            "macro_recall": 100.0 * rec[present].mean().item(),
            "macro_f1": 100.0 * f1[present].mean().item(),
            "weighted_f1": 100.0 * (f1 * support).sum().item() / max(1.0, support.sum().item()),
            "n": self.n,
        }
        if self.mass_n:
            out["attn_in_box"] = self.mass_sum / self.mass_n
            out["box_area_frac"] = self.area_sum / self.mass_n
        if self.gate_n:
            out["gate_mean"] = self.gate_mean_sum / self.gate_n
            out["gate_sparsity"] = self.gate_sp_sum / self.gate_n
        if self.track_cal and self.confs:
            ece, _ = calibration_bins(torch.cat(self.confs), torch.cat(self.corrects))
            out["ece"] = ece
        return out

    def calibration_table(self) -> Tuple[float, List[dict]]:
        return calibration_bins(torch.cat(self.confs), torch.cat(self.corrects))

    def per_class_report(self, class_names: Optional[Sequence[str]] = None) -> List[dict]:
        conf = self.conf.double()
        tp, support, pred_cnt = conf.diag(), conf.sum(1), conf.sum(0)
        rows = []
        for c in range(self.nc):
            p = (tp[c] / pred_cnt[c]).item() if pred_cnt[c] > 0 else 0.0
            r = (tp[c] / support[c]).item() if support[c] > 0 else 0.0
            f = 2 * p * r / (p + r) if p + r > 0 else 0.0
            rows.append({"class_id": c, "class_name": class_names[c] if class_names else str(c),
                         "precision": 100 * p, "recall": 100 * r, "f1": 100 * f, "support": int(support[c])})
        return rows


def species_groups(class_names: Sequence[str]) -> Dict[str, List[int]]:
    """Look-alike groups from docs/PROTOCOL.md: species whose names end in the same word, at least 2 per group."""
    groups: Dict[str, List[int]] = {}
    for index, name in enumerate(class_names):
        groups.setdefault(name.split()[-1], []).append(index)
    return {word: members for word, members in groups.items() if len(members) >= 2}


def look_alike_metrics(confusion, class_names: Sequence[str]) -> Dict[str, float]:
    """From a confusion matrix (rows = true species): accuracy on species in look-alike groups, and the share of their
    images predicted as a different species in the same group."""
    conf = torch.as_tensor(confusion).double()
    total = correct = within = 0.0
    for members in species_groups(class_names).values():
        block = conf[members][:, members]
        total += conf[members].sum().item()
        correct += block.diag().sum().item()
        within += (block.sum() - block.diag().sum()).item()
    return {"within_group_accuracy": 100.0 * correct / max(total, 1.0),
            "within_group_confusion": 100.0 * within / max(total, 1.0)}


@torch.no_grad()
def part_localization(model: nn.Module, loader, device: torch.device, grid: int, amp: tuple) -> Dict[str, float]:
    """Distance from pooling queries to CUB part points, as a share of the image width.
    assigned: query i to part i, which is what part guidance trains. best: for each part, whichever query is closest
    on average, chosen on these same images, so it gives models without part guidance their best case."""
    use_amp, amp_dtype = amp
    model.eval()
    centers = (torch.arange(grid, device=device).float() + 0.5) / grid
    ys, xs = torch.meshgrid(centers, centers, indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    sums, counts = None, None
    for x, _m, _y, _id, points in loader:
        x, points = x.to(device), points.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            _, aux = model(x)
        assign = aux["pool_assign"].float()
        cx, cy = (assign * xs).sum(-1), (assign * ys).sum(-1)                  # (B, t)
        visible = points[..., 2]                                               # (B, P)
        error = torch.sqrt((cx[:, None] - points[..., 0, None]) ** 2 + (cy[:, None] - points[..., 1, None]) ** 2)
        batch_sums = (error * visible[..., None]).sum(0)                       # (P, t)
        sums = batch_sums if sums is None else sums + batch_sums
        counts = visible.sum(0) if counts is None else counts + visible.sum(0)
    mean = sums / counts.clamp_min(1.0)[:, None]
    seen = counts > 0
    num_parts = mean.shape[0]
    return {"assigned": mean[torch.arange(num_parts), torch.arange(num_parts)][seen].mean().item(),
            "best": mean.min(1).values[seen].mean().item()}


@torch.no_grad()
def measure_efficiency(model: nn.Module, device: torch.device, img_size: int = 224,
                       batch_sizes: Sequence[int] = (1, 64), iters: int = 20) -> Dict[str, float]:
    """Parameters, FLOPs of one forward pass (batch 1), inference latency and throughput."""
    was_training = model.training
    model.eval()
    out: Dict[str, float] = {"params_M": count_parameters(model)[0] / 1e6}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, 3, img_size, img_size, device=device)
        with FlopCounterMode(display=False) as fc:
            model(x)
        out["GFLOPs_per_image"] = fc.get_total_flops() / 1e9
    except Exception:
        out["GFLOPs_per_image"] = float("nan")
    for bs in batch_sizes:
        x = torch.randn(bs, 3, img_size, img_size, device=device)
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        out[f"latency_ms_bs{bs}"] = dt * 1000
        out[f"throughput_img_s_bs{bs}"] = bs / dt
    model.train(was_training)
    return out
