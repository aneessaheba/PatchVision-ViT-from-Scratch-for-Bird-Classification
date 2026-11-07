"""Self-supervised pretraining on the training images only: each image is its own class (instance discrimination).

A simplified version of the idea in Cao et al., "Training Vision Transformers with Only 2040 Images" (ECCV 2022), without
their multi-crop or low-resolution stage. It uses no species labels, boxes, part points, validation images, test images or
outside data. It writes results_protocol/<name>/seed<seed>/encoder.pt, which a later run starts from through
TrainConfig.pretrain_run. The classifier head is left out, because it only knows individual images.
"""
from __future__ import annotations

import time
from typing import Optional

import torch

from .config import ExperimentConfig, PathsConfig, get_experiment
from .data import MixupCutmix, build_loaders, load_data_info
from .losses import SoftTargetCrossEntropy
from .models import build_model
from .optim import build_optimizer, build_scheduler
from .regularization import ModelEMA
from .utils import Logger, env_info, get_device, save_json, seed_everything

PRETRAIN_RUN = "pretrain_instance"


def pretrain_experiment(seed: int = 42, epochs: int = 300) -> ExperimentConfig:
    return get_experiment(
        "full", seed, {"drop_path": 0.3},
        {"aug_tier": "auto", "use_bbox_loss": False, "use_box_crop": False, "use_consistency": False,
         "epochs": epochs}, run_name=PRETRAIN_RUN)


def run_instance_pretraining(exp: ExperimentConfig, paths: PathsConfig, device: Optional[torch.device] = None,
                             num_workers: int = 4, quiet: bool = False):
    device = device or get_device()
    tc, mc = exp.train, exp.model
    run_dir = paths.results / exp.name / f"seed{exp.seed}"
    encoder_path = run_dir / "encoder.pt"
    if encoder_path.exists():
        print(f"[{exp.name} | seed {exp.seed}] already finished -> {encoder_path}")
        return encoder_path
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train_log.txt", quiet)
    seed_everything(exp.seed)

    info = load_data_info(paths)
    datasets, loaders = build_loaders(info, tc, exp.seed, num_workers, device.type == "cuda")
    image_ids = [int(r["image_id"]) for r in datasets["train"].records]
    instance_of = torch.full((max(image_ids) + 1,), -1, dtype=torch.long)
    instance_of[torch.tensor(image_ids)] = torch.arange(len(image_ids))
    mc.img_size, mc.num_classes = info.img_size, len(image_ids)

    model = build_model(mc).to(device)
    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, tc, len(loaders["train"]))
    ema = ModelEMA(model, tc.ema_decay)
    criterion = SoftTargetCrossEntropy()
    mix = info.mixcfg
    mixer = MixupCutmix(len(image_ids), mix["mixup_alpha"], mix["cutmix_alpha"], mix["p_cutmix"], tc.label_smoothing)
    use_amp = tc.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    start_epoch, last_path = 1, run_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        log(f"### RESUMED at epoch {start_epoch}")
    else:
        save_json({"experiment": exp.to_dict(), "env": env_info()}, run_dir / "config.json")
        log(f"PRETRAIN {exp.name} | seed {exp.seed} | {len(image_ids)} images as classes | {tc.epochs} epochs")

    for epoch in range(start_epoch, tc.epochs + 1):
        model.train()
        started, total, steps = time.time(), 0.0, 0
        for x, m, _species, ids in loaders["train"]:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            instances = instance_of[ids].to(device, non_blocking=True)
            x, soft, _, _ = mixer(x, instances, m)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x)
            loss = criterion(logits.float(), soft)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                log(f"  !! epoch {epoch}: non-finite loss, step skipped")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            total, steps = total + loss.item(), steps + 1
        log(f"[E{epoch:03d}/{tc.epochs}] loss {total / max(1, steps):.4f} | {time.time() - started:.1f}s")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "ema": ema.state_dict()}, last_path)

    encoder = {k: v for k, v in ema.module.state_dict().items() if not k.startswith("head.")}
    torch.save(encoder, encoder_path)
    log(f"Saved {encoder_path}")
    return encoder_path
