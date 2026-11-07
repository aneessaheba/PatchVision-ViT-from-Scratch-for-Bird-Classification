"""
Trainer: the training / validation / checkpoint / logging / test loop.

Everything is written to  <results>/<experiment>/seed<seed>/ :

    config.json  env.json                       exact configuration + software/hardware record
    train_log.txt                               every step and every epoch, human readable
    steps.csv    history.csv                    every step / every epoch as tables
    curves.png                                  refreshed every `plot_every` epochs
    attention_snapshots/epoch_XXX.png           where the model looks, over training
    val_reports/epoch_XXX_{per_class.csv,confusion.npy}
    weights/epoch_XXX.pt                        weights-only checkpoints every `weights_every` epochs
    best_model.pt  last.pt                      best (by validation) and resumable state
    val_metrics.json                            summary of a finished run, validation numbers only

and only when run_test=True:

    test_metrics.json  test_per_class.csv  test_confusion_matrix.{npy,png}  test_predictions.csv
    test_reliability.{png,csv}  test_attention_gallery.png
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, PathsConfig
from .data import MixupCutmix, build_loaders, flip_part_points, load_data_info
from .losses import BBoxAttentionLoss, TrainingLoss, smooth_one_hot
from .metrics import MetricTracker, box_to_patch_mask, measure_efficiency, part_localization
from .models import TOKENIZER_STRIDE, build_model
from .optim import build_optimizer, build_scheduler
from .regularization import EarlyStopping, ModelEMA
from .utils import Logger, count_parameters, env_info, fmt_time, get_device, save_json, seed_everything
from . import visualize as viz


def _fmt(d: dict, k: str, spec: str = ".2f") -> str:
    return format(d[k], spec) if k in d and d[k] == d[k] else "n/a"


# ----------------------------------------------------------------------------------------------
# Evaluation (also used for the final test)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, grid: int, num_classes: int, amp: tuple, collect: bool = False):
    """Returns (metrics dict, MetricTracker, predictions DataFrame or None). Hard labels, no augmentation."""
    use_amp, amp_dtype = amp
    model.eval()
    tracker, attn_loss = MetricTracker(num_classes, track_calibration=True), BBoxAttentionLoss()
    rows = []
    t0 = time.time()
    for x, m, y, iid in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(x)
        logits = logits.float()
        pmask = box_to_patch_mask(m, grid)
        _, mass, valid = attn_loss(aux["attn_maps"], pmask)
        tracker.update(logits, y, F.cross_entropy(logits, y).item(), mass=mass, valid=valid,
                       area=pmask.sum(1) / pmask.shape[1],
                       gate=(aux["gate_mean"].item(), aux["gate_sparsity"].item()))
        if collect:
            conf = torch.softmax(logits, -1).max(1).values
            pred = logits.topk(1, dim=1).indices.squeeze(1)      # same choice as MetricTracker when scores tie
            rows.append(pd.DataFrame({"image_id": iid.numpy(), "label": y.cpu().numpy(), "pred": pred.cpu().numpy(),
                                      "confidence": conf.cpu().numpy()}))
    res = tracker.compute()
    res["time_s"] = time.time() - t0
    return res, tracker, (pd.concat(rows, ignore_index=True) if collect else None)


class Trainer:
    def __init__(self, exp: ExperimentConfig, paths: PathsConfig, device: Optional[torch.device] = None,
                 num_workers: int = 4, step_print_every: int = 1, resume: bool = True, quiet: bool = False,
                 run_test: bool = False):
        self.exp = copy.deepcopy(exp)
        self.paths, self.resume, self.quiet, self.run_test = paths, resume, quiet, run_test
        self.device = device or get_device()
        self.num_workers, self.step_print_every = num_workers, step_print_every
        self.run_dir = paths.results / exp.name / f"seed{exp.seed}"

    # ------------------------------------------------------------------ public
    def run(self) -> dict:
        done = self.run_dir / "test_metrics.json"
        if self.resume and done.exists():
            print(f"[{self.exp.name} | seed {self.exp.seed}] already finished -> loading {done}")
            return json.loads(done.read_text())["row"]
        val_done = self.run_dir / "val_metrics.json"
        if self.resume and val_done.exists():
            row = json.loads(val_done.read_text())["row"]
            if not self.run_test:
                print(f"[{self.exp.name} | seed {self.exp.seed}] already trained -> loading {val_done}")
                return row
            for name in ("last.pt", "best_model.pt"):
                if not (self.run_dir / name).exists():
                    raise FileNotFoundError(f"{self.run_dir / name} is needed to run the test without retraining")
            self._setup()
            self.epochs_run, self.train_time_s = row["epochs_run"], row["train_time_s"]
            self.stopped_early = row["stopped_early"]
            return self._final_test()
        self._setup()
        self._train_loop()
        return self._final_test() if self.run_test else self._finish_validation_only()

    # ------------------------------------------------------------------ setup
    def _setup(self):
        exp, rd, dev = self.exp, self.run_dir, self.device
        tc, mc = exp.train, exp.model
        rd.mkdir(parents=True, exist_ok=True)
        resuming = self.resume and (rd / "last.pt").exists()
        if not resuming:
            for f in ("train_log.txt", "steps.csv", "history.csv"):
                (rd / f).unlink(missing_ok=True)
        self.log = Logger(rd / "train_log.txt", self.quiet)
        seed_everything(exp.seed)

        self.info = load_data_info(self.paths)
        if self.info.bboxes is None and (tc.use_bbox_loss or tc.use_box_crop):
            raise FileNotFoundError(f"{self.paths.cub / 'bounding_boxes.txt'} not found - needed by use_bbox_loss / "
                                    f"use_box_crop (set both to False to train without boxes)")
        mc.img_size, mc.num_classes = self.info.img_size, self.info.num_classes
        self.grid = mc.img_size // TOKENIZER_STRIDE
        self.ds, self.dl = build_loaders(self.info, tc, exp.seed, self.num_workers, dev.type == "cuda")

        self.model = build_model(mc).to(dev)
        self.opt = build_optimizer(self.model, tc)
        self.sched = build_scheduler(self.opt, tc, len(self.dl["train"]))
        self.use_amp = tc.amp and dev.type == "cuda"
        self.amp_dtype = (torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
                          else torch.float16)
        self.amp = (self.use_amp, self.amp_dtype)
        self.scaler = torch.amp.GradScaler("cuda") if (self.use_amp and self.amp_dtype == torch.float16) else None
        self.criterion = TrainingLoss(tc, self.grid)
        mx = self.info.mixcfg
        self.mixer = (MixupCutmix(self.info.num_classes, mx["mixup_alpha"], mx["cutmix_alpha"], mx["p_cutmix"],
                                  tc.label_smoothing) if tc.use_mixup_cutmix else None)
        self.ema = ModelEMA(self.model, tc.ema_decay) if tc.use_ema else None
        self.stopper = EarlyStopping(tc.early_stop_patience, tc.early_stop_start,
                                     "min" if tc.best_metric == "loss" else "max")
        self.history, self.best, self.best_epoch, self.start_epoch = [], None, 0, 1
        self.global_step, self.elapsed_prev, self.stopped_early = 0, 0.0, False
        self.n_params = count_parameters(self.model)[0]

        if resuming:
            ck = torch.load(rd / "last.pt", map_location=dev, weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["optimizer"])
            self.sched.load_state_dict(ck["scheduler"])
            if self.scaler is not None and ck.get("scaler"):
                self.scaler.load_state_dict(ck["scaler"])
            if self.ema is not None and ck.get("ema"):
                self.ema.load_state_dict(ck["ema"])
            self.stopper.load_state_dict(ck["stopper"])
            self.history, self.best, self.best_epoch = ck["history"], ck["best"], ck["best_epoch"]
            self.start_epoch, self.global_step, self.elapsed_prev = ck["epoch"] + 1, ck["global_step"], ck["elapsed_s"]
            self.log(f"### RESUMED from last.pt at epoch {self.start_epoch} "
                     f"(best {tc.best_metric} {self.best} @ epoch {self.best_epoch})")
        else:
            save_json(exp.to_dict(), rd / "config.json")
            save_json(env_info(), rd / "env.json")
            L = self.log
            L("=" * 130)
            L(f"RUN {exp.name} | seed {exp.seed} | device {dev} | AMP {self.use_amp} ({self.amp_dtype})")
            L(f"model  {self.n_params / 1e6:.2f} M params | {mc}")
            L(f"train  {tc}")
            L(f"data   splits {self.info.splits} | classes {self.info.num_classes} | "
              f"steps/epoch {len(self.dl['train'])} | total steps {len(self.dl['train']) * tc.epochs:,}")
            L("=" * 130)

    @property
    def eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def _is_better(self, new, best) -> bool:
        if best is None:
            return True
        return new < best if self.exp.train.best_metric == "loss" else new > best

    # ------------------------------------------------------------------ one training epoch
    def _train_epoch(self, epoch: int, step_rows: list) -> dict:
        tc, dev, log = self.exp.train, self.device, self.log
        model, loader = self.model, self.dl["train"]
        model.train()
        tracker = MetricTracker(self.info.num_classes)
        n_steps, t_epoch = len(loader), time.time()
        sums = {k: 0.0 for k in ("total", "ce", "attn", "cons_logit", "cons_attn", "part", "part_error")}
        gnorm_sum = data_time = 0.0
        skipped = 0
        lr_first = self.opt.param_groups[0]["lr"]
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        two = tc.use_consistency
        t_prev = time.time()
        for step, batch in enumerate(loader, 1):
            x, m, y = batch[0], batch[1], batch[2]
            t_data = time.time() - t_prev
            t0 = time.time()
            x, m, y = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            points = batch[4].to(dev, non_blocking=True) if tc.use_part_guidance else None
            if self.mixer is not None:
                x, soft, m, mixinfo = self.mixer(x, y, m, points=points)
            else:
                soft, mixinfo = smooth_one_hot(y, self.info.num_classes, tc.label_smoothing), {"type": "none", "lam": 1.0}
                if points is not None:
                    mixinfo.update(points=points, point_weight=x.new_ones(x.shape[0]))
            part_points, part_weights = mixinfo.pop("points", None), mixinfo.pop("point_weight", None)
            hard = soft.argmax(1)                                 # dominant label (approximate train accuracy)
            B = x.shape[0]
            if two:                                               # R3: second view = horizontal flip
                x_in, m_in = torch.cat([x, x.flip(-1)]), torch.cat([m, m.flip(-1)])
                if part_points is not None:
                    part_points = torch.cat([part_points, flip_part_points(part_points, self.ds["train"].parts.flip_order)])
                    part_weights = torch.cat([part_weights, part_weights])
            else:
                x_in, m_in = x, m
            pmask = box_to_patch_mask(m_in, self.grid)

            lr_now = self.opt.param_groups[0]["lr"]
            with torch.autocast(dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits, aux = model(x_in)
            out = self.criterion(logits.float(), soft, aux, pmask, two_views=two, part_points=part_points,
                                 part_weights=part_weights)
            if not torch.isfinite(out["total"]):
                skipped += 1
                log(f"  !! step {step}: non-finite loss ({out['total'].item()}) -> step skipped")
                self.opt.zero_grad(set_to_none=True)
                t_prev = time.time()
                continue
            self.opt.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(out["total"]).backward()
                self.scaler.unscale_(self.opt)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                out["total"].backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
                self.opt.step()
            self.sched.step()
            if self.ema is not None:
                self.ema.update(model)
            self.global_step += 1

            step_time = time.time() - t0
            lg = logits[:B].detach().float()
            pred5 = lg.topk(min(5, lg.shape[1]), 1).indices
            a1 = (pred5[:, 0] == hard).float().mean().item() * 100
            a5 = (pred5 == hard[:, None]).any(1).float().mean().item() * 100
            valid = out["valid"]
            area_s = pmask[:B].sum(1) / pmask.shape[1]
            mass = (out["mass"] * valid).sum().item() / max(1.0, valid.sum().item())
            area = (area_s * valid).sum().item() / max(1.0, valid.sum().item())
            gm, gs = aux["gate_mean"].item(), aux["gate_sparsity"].item()
            vals = {k: out[k].item() for k in sums}
            tracker.update(lg, hard, vals["total"], mass=out["mass"], valid=valid, area=area_s, gate=(gm, gs))
            for k in sums:
                sums[k] += vals[k]
            gnorm_sum += gnorm
            data_time += t_data
            ips = B / max(1e-9, step_time + t_data)
            eta_epoch = (time.time() - t_epoch) / step * (n_steps - step)
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
            line = (f"[E{epoch:03d}/{tc.epochs} | S{step:03d}/{n_steps} | g{self.global_step:06d}] "
                    f"loss {vals['total']:.4f} (ce {vals['ce']:.4f} | bbox {vals['attn']:.4f} | "
                    f"cons {vals['cons_logit']:.4f}/{vals['cons_attn']:.4f}) | acc1 {a1:5.1f}% acc5 {a5:5.1f}% | "
                    f"in-box {mass:.3f} (area {area:.3f}) | {mixinfo['type']}(lam={mixinfo['lam']:.2f}) | "
                    f"lr {lr_now:.3e} | gnorm {gnorm:6.3f} | gate {gm:.2f}/{gs:.2f} | "
                    f"{ips:5.0f} img/s (data wait {t_data * 1000:4.0f} ms) | mem {mem:.2f} GB | "
                    f"step {step_time:.2f}s | epoch ETA {fmt_time(eta_epoch)}")
            log(line, show=(step % self.step_print_every == 0 or step == n_steps))
            step_rows.append({"epoch": epoch, "step": step, "global_step": self.global_step, **vals, "acc1": a1,
                              "acc5": a5, "attn_in_box": mass, "box_area": area, "lr": lr_now, "gnorm": gnorm,
                              "gate_mean": gm, "gate_sparsity": gs, "img_per_s": ips, "data_wait_ms": t_data * 1000,
                              "mem_gb": mem, "mix": mixinfo["type"], "lam": mixinfo["lam"]})
            t_prev = time.time()
        res = tracker.compute()
        n = max(1, n_steps - skipped)
        res.update({k: sums[k] / n for k in ("ce", "attn", "cons_logit", "cons_attn", "part", "part_error")})
        res.update({"gnorm": gnorm_sum / n, "lr_first": lr_first, "lr_last": self.opt.param_groups[0]["lr"],
                    "time_s": time.time() - t_epoch, "data_wait_frac": data_time / max(1e-9, time.time() - t_epoch),
                    "skipped": skipped})
        return res

    # ------------------------------------------------------------------ main loop
    def _train_loop(self):
        tc, rd, log = self.exp.train, self.run_dir, self.log
        ev = lambda model, split, collect=False: evaluate(model, self.dl[split], self.device, self.grid,
                                                          self.info.num_classes, self.amp, collect)
        snap_idx = np.linspace(0, len(self.ds["val"]) - 1, min(16, len(self.ds["val"]))).astype(int).tolist()
        t_run, epoch_times, step_rows = time.time(), [], []
        epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, tc.epochs + 1):
            log(f"\n{'-' * 60} EPOCH {epoch:03d}/{tc.epochs} {'-' * 60}")
            tr = self._train_epoch(epoch, step_rows)
            va, va_trk, _ = ev(self.eval_model, "val")
            va_raw = ev(self.model, "val")[0] if self.ema is not None else None
            clean = None
            if epoch == 1 or epoch % tc.clean_eval_every == 0 or epoch == tc.epochs:
                clean = ev(self.eval_model, "train_clean")[0]
            epoch_times.append(tr["time_s"] + va["time_s"])
            eta = np.mean(epoch_times[-10:]) * (tc.epochs - epoch)

            score = va[tc.best_metric]
            improved = self._is_better(score, self.best)
            if improved:
                self.best, self.best_epoch = score, epoch
                torch.save({"exp": self.exp.to_dict(), "epoch": epoch, "state": self.eval_model.state_dict(),
                            "uses_ema": self.ema is not None, "val": va}, rd / "best_model.pt")
                np.save(rd / "best_val_confusion.npy", va_trk.conf.numpy())
            stop = self.stopper.step(score, epoch)
            parts = {}
            if tc.track_part_localization and (improved or epoch % tc.snapshot_every == 0 or stop or epoch == tc.epochs):
                found = part_localization(self.eval_model, self.dl["val_parts"], self.device, self.grid, self.amp)
                parts = {"val_part_error_assigned": found["assigned"], "val_part_error_best": found["best"]}
                log(f"  PARTS  distance to part points: assigned queries {found['assigned']:.4f} | "
                    f"best queries {found['best']:.4f} (share of image width)")

            rec = {"epoch": epoch, **parts, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
                   "valraw_acc1": va_raw["acc1"] if va_raw else np.nan, "valraw_loss": va_raw["loss"] if va_raw else np.nan,
                   "clean_acc1": clean["acc1"] if clean else np.nan, "clean_loss": clean["loss"] if clean else np.nan,
                   "gap_acc1": tr["acc1"] - va["acc1"], "is_best": improved}
            self.history.append(rec)

            log("")
            log(f"  TRAIN  loss {tr['loss']:.4f} (ce {tr['ce']:.4f} | bbox {tr['attn']:.4f} | cons "
                f"{tr['cons_logit']:.4f}/{tr['cons_attn']:.4f}) | top-1 {tr['acc1']:.2f}% top-5 {tr['acc5']:.2f}% | "
                f"macro F1 {tr['macro_f1']:.2f}   (vs dominant mixup/cutmix label: approximate)")
            log(f"  VAL    loss {va['loss']:.4f} | top-1 {va['acc1']:.2f}% top-5 {va['acc5']:.2f}% | macro P/R/F1 "
                f"{va['macro_precision']:.2f}/{va['macro_recall']:.2f}/{va['macro_f1']:.2f} | ECE {_fmt(va, 'ece')}% "
                f"(n={va['n']}, {'EMA' if self.ema else 'raw'} weights)")
            if va_raw:
                log(f"  VAL raw weights (no EMA): top-1 {va_raw['acc1']:.2f}% loss {va_raw['loss']:.4f}")
            if clean:
                log(f"  CLEAN  train images without augmentation: top-1 {clean['acc1']:.2f}% loss {clean['loss']:.4f} "
                    f"-> TRUE overfitting gap {clean['acc1'] - va['acc1']:+.2f} pts")
            log(f"  ATTN   share inside bird box: train {_fmt(tr, 'attn_in_box', '.3f')} | val {_fmt(va, 'attn_in_box', '.3f')} "
                f"(box = {_fmt(va, 'box_area_frac', '.3f')} of image = chance) | gate mean {_fmt(va, 'gate_mean', '.3f')} "
                f"sparsity {_fmt(va, 'gate_sparsity', '.3f')}")
            log(f"  OPT    lr {tr['lr_first']:.3e} -> {tr['lr_last']:.3e} | grad-norm {tr['gnorm']:.3f} | "
                f"skipped {tr['skipped']}")
            log(f"  TIME   {fmt_time(tr['time_s'])} train + {fmt_time(va['time_s'])} val | data-wait "
                f"{tr['data_wait_frac'] * 100:.0f}% | run {fmt_time(self.elapsed_prev + time.time() - t_run)} | ETA {fmt_time(eta)}")
            log(f"  BEST   val {tc.best_metric} {self.best:.4f} @ epoch {self.best_epoch}"
                + ("   <== NEW BEST" if improved else f"   (no improvement for {self.stopper.wait} epochs)"))

            pd.DataFrame(self.history).to_csv(rd / "history.csv", index=False)
            if step_rows:
                pd.DataFrame(step_rows).to_csv(rd / "steps.csv", mode="a", index=False,
                                               header=not (rd / "steps.csv").exists())
                step_rows.clear()

            last = epoch == tc.epochs or stop
            if epoch == 1 or epoch % tc.snapshot_every == 0 or last:
                self._snapshot(epoch, va_trk, snap_idx)
            if tc.weights_every and epoch % tc.weights_every == 0:
                (rd / "weights").mkdir(exist_ok=True)
                torch.save(self.eval_model.state_dict(), rd / "weights" / f"epoch_{epoch:03d}.pt")
            if epoch % tc.plot_every == 0 or last:
                viz.plot_curves(pd.DataFrame(self.history), rd / "curves.png", f"{self.exp.name} (seed {self.exp.seed})")
            torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                        "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict() if self.scaler else None,
                        "ema": self.ema.state_dict() if self.ema else None, "stopper": self.stopper.state_dict(),
                        "history": self.history, "best": self.best, "best_epoch": self.best_epoch,
                        "global_step": self.global_step, "elapsed_s": self.elapsed_prev + time.time() - t_run},
                       rd / "last.pt")
            if stop:
                self.stopped_early = True
                log(f"\n*** EARLY STOP: no validation improvement for {self.stopper.wait} epochs "
                    f"(best epoch {self.best_epoch}) ***")
                break
        self.epochs_run = epoch
        self.train_time_s = self.elapsed_prev + time.time() - t_run

    def _snapshot(self, epoch: int, va_trk: MetricTracker, idx: list):
        rd = self.run_dir
        viz.attention_overlays(self.eval_model, self.ds["val"], idx, self.device, self.grid, self.info.mean,
                               self.info.std, rd / "attention_snapshots" / f"epoch_{epoch:03d}.png",
                               title=f"{self.exp.name} | validation attention | epoch {epoch}")
        (rd / "val_reports").mkdir(exist_ok=True)
        pd.DataFrame(va_trk.per_class_report(self.ds["val"].class_names)).to_csv(
            rd / "val_reports" / f"epoch_{epoch:03d}_per_class.csv", index=False)
        np.save(rd / "val_reports" / f"epoch_{epoch:03d}_confusion.npy", va_trk.conf.numpy())

    # ------------------------------------------------------------------ summary
    def _validation_row(self) -> dict:
        exp, tc = self.exp, self.exp.train
        hist = pd.DataFrame(self.history)
        best_row = hist[hist["epoch"] == self.best_epoch].iloc[0]
        clean_rows = hist.dropna(subset=["clean_acc1"])
        near = clean_rows.iloc[(clean_rows["epoch"] - self.best_epoch).abs().argsort()[:1]]
        gap = float(near["clean_acc1"].iloc[0] - near["val_acc1"].iloc[0]) if len(near) else float("nan")
        return {"experiment": exp.name, "seed": exp.seed, "best_epoch": self.best_epoch, "epochs_run": self.epochs_run,
                "stopped_early": self.stopped_early, "train_time_s": self.train_time_s,
                "sec_per_epoch": self.train_time_s / max(1, self.epochs_run),
                f"best_val_{tc.best_metric}": self.best, "val_acc1_at_best": float(best_row["val_acc1"]),
                "overfit_gap_clean_minus_val": gap}

    def _finish_validation_only(self) -> dict:
        row = {**self._validation_row(), "params_M": self.n_params / 1e6, "run_dir": str(self.run_dir)}
        save_json({"row": row, "config": self.exp.to_dict(), "env": env_info()}, self.run_dir / "val_metrics.json")
        self.log(f"\nTRAINING DONE. Best epoch {self.best_epoch}, val {self.exp.train.best_metric} {self.best:.4f}. "
                 f"Test set not evaluated.\nSaved everything to {self.run_dir}")
        return row

    # ------------------------------------------------------------------ test
    def _final_test(self) -> dict:
        exp, tc, rd, log = self.exp, self.exp.train, self.run_dir, self.log
        log("\n" + "=" * 130 + f"\nTRAINING DONE. Loading best model (epoch {self.best_epoch}, val {tc.best_metric} "
            f"{self.best:.4f}) for the TEST set\n" + "=" * 130)
        ck = torch.load(rd / "best_model.pt", map_location=self.device, weights_only=False)
        model = self.eval_model
        model.load_state_dict(ck["state"])
        te, trk, preds = evaluate(model, self.dl["test"], self.device, self.grid, self.info.num_classes, self.amp, True)
        eff = measure_efficiency(model, self.device, self.info.img_size, batch_sizes=(1, tc.batch_size),
                                 iters=20 if self.device.type == "cuda" else 3)
        names = self.ds["test"].class_names
        log(f"TEST   loss {te['loss']:.4f} | top-1 {te['acc1']:.2f}% top-5 {te['acc5']:.2f}% | macro P/R/F1 "
            f"{te['macro_precision']:.2f}/{te['macro_recall']:.2f}/{te['macro_f1']:.2f} | weighted F1 {te['weighted_f1']:.2f} "
            f"| ECE {_fmt(te, 'ece')}% | n={te['n']}")
        log(f"TEST   attention inside bird box {_fmt(te, 'attn_in_box', '.3f')} (chance {_fmt(te, 'box_area_frac', '.3f')}) "
            f"| gate sparsity {_fmt(te, 'gate_sparsity', '.3f')}")
        log("EFFICIENCY " + " | ".join(f"{k} {v:.3f}" for k, v in eff.items()))

        report = pd.DataFrame(trk.per_class_report(names))
        report.to_csv(rd / "test_per_class.csv", index=False)
        preds.to_csv(rd / "test_predictions.csv", index=False)
        conf = trk.conf.clone()
        np.save(rd / "test_confusion_matrix.npy", conf.numpy())
        viz.plot_confusion(conf.numpy(), rd / "test_confusion_matrix.png", f"{exp.name} test confusion matrix")
        ece, bins = trk.calibration_table()
        pd.DataFrame(bins).to_csv(rd / "test_reliability.csv", index=False)
        viz.plot_reliability(bins, ece, rd / "test_reliability.png", f"{exp.name} test")
        gallery_idx = np.linspace(0, len(self.ds["test"]) - 1, 16).astype(int).tolist()
        viz.attention_overlays(model, self.ds["test"], gallery_idx, self.device, self.grid, self.info.mean,
                               self.info.std, rd / "test_attention_gallery.png",
                               title=f"{exp.name} | test attention (green box = ground truth, used only in training)")
        hist = pd.DataFrame(self.history)
        viz.plot_curves(hist, rd / "curves.png", f"{exp.name} (seed {exp.seed})")

        log("worst 10 classes (F1):\n" + report.sort_values("f1").head(10).to_string(index=False))
        log("best 10 classes (F1):\n" + report.sort_values("f1", ascending=False).head(10).to_string(index=False))
        off = conf.clone()
        off.fill_diagonal_(0)
        top = torch.topk(off.flatten(), min(15, off.numel()))
        nc = self.info.num_classes
        pairs = [(names[i // nc], names[i % nc], int(v)) for v, i in zip(top.values, top.indices) if v > 0]
        log("most frequent confusions (true -> predicted : count):\n" + "\n".join(f"  {a} -> {b} : {n}" for a, b, n in pairs))

        row = {**self._validation_row(),
               "test_acc1": te["acc1"], "test_acc5": te["acc5"], "test_macro_f1": te["macro_f1"],
               "test_macro_precision": te["macro_precision"], "test_macro_recall": te["macro_recall"],
               "test_weighted_f1": te["weighted_f1"], "test_loss": te["loss"], "test_ece": te.get("ece", float("nan")),
               "test_attn_in_box": te.get("attn_in_box", float("nan")), "params_M": eff["params_M"],
               "GFLOPs": eff["GFLOPs_per_image"], "latency_ms_bs1": eff["latency_ms_bs1"],
               f"throughput_img_s_bs{tc.batch_size}": eff[f"throughput_img_s_bs{tc.batch_size}"],
               "run_dir": str(rd)}
        save_json({"row": row, "test": te, "efficiency": eff, "config": exp.to_dict(), "env": env_info()},
                  rd / "test_metrics.json")
        log(f"\nSaved everything to {rd}")
        return row
