"""High-level helpers used by the notebook: dry run, run one experiment, run a suite, results table."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import torch

from .config import ExperimentConfig, PathsConfig, ablation_suite, get_experiment
from .data import build_loaders, load_data_info
from .engine import Trainer
from .losses import TrainingLoss, smooth_one_hot
from .metrics import box_to_patch_mask, measure_efficiency
from .models import build_model
from .utils import count_parameters, get_device, seed_everything

METRIC_COLS = [("test_acc1", "top-1 (%)"), ("test_acc5", "top-5 (%)"), ("test_macro_f1", "macro-F1 (%)"),
               ("test_ece", "ECE (%)"), ("test_attn_in_box", "attn in box"),
               ("overfit_gap_clean_minus_val", "train-val gap (pts)"), ("best_epoch", "best epoch"),
               ("params_M", "params (M)"), ("GFLOPs", "GFLOPs"), ("latency_ms_bs1", "latency bs1 (ms)"),
               ("sec_per_epoch", "s / epoch")]


def dry_run(exp: ExperimentConfig, paths: PathsConfig, device: Optional[torch.device] = None) -> dict:
    """Builds the model + loss and pushes ONE real batch through forward/backward (single view and the
    flip-consistency two-view path). Checks shapes, loss at initialisation (~ln(classes)) and gradient flow."""
    device = device or get_device()
    tc, mc = exp.train, exp.model
    info = load_data_info(paths, tc.img_size)
    mc.img_size, mc.num_classes = info.img_size, info.num_classes
    grid = mc.grid
    seed_everything(exp.seed)
    _, dl = build_loaders(info, tc, exp.seed, num_workers=0, pin_memory=False)
    model = build_model(mc).to(device)
    total_p = count_parameters(model)[0]
    eff = measure_efficiency(model, device, info.img_size, batch_sizes=(1,), iters=3)
    print(f"{exp.name}: {total_p / 1e6:.2f} M parameters | GFLOPs {eff['GFLOPs_per_image']:.2f} | "
          f"latency bs1 {eff['latency_ms_bs1']:.1f} ms")
    x, m, y, _ = next(iter(dl["train"]))
    x, m, y = x.to(device), m.to(device), y.to(device)
    print(f"batch: images {tuple(x.shape)} | mask max {m.max().item():.2f} (must be 1.0 when boxes are loaded) | "
          f"x mean/std {x.mean().item():.2f}/{x.std().item():.2f}")
    crit = TrainingLoss(tc, grid)
    soft = smooth_one_hot(y, info.num_classes, tc.label_smoothing)
    report = {"params_M": total_p / 1e6, **eff}
    for two in sorted({False, tc.use_consistency}):
        model.train()
        model.zero_grad(set_to_none=True)
        x_in, m_in = (torch.cat([x, x.flip(-1)]), torch.cat([m, m.flip(-1)])) if two else (x, m)
        logits, aux = model(x_in)
        out = crit(logits.float(), soft, aux, box_to_patch_mask(m_in, grid), two_views=two)
        out["total"].backward()
        bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        print(f"[{'2 views (flip)' if two else '1 view'}] logits {tuple(logits.shape)} | attention maps "
              f"{len(aux['attn_maps'])} x {tuple(aux['attn_maps'][0].shape)} | loss {out['total'].item():.4f} "
              f"(ce {out['ce'].item():.4f}, chance ln({info.num_classes}) = {math.log(info.num_classes):.4f}, "
              f"bbox {out['attn'].item():.4f}, cons {out['cons_logit'].item():.4f}/{out['cons_attn'].item():.4f}) "
              f"| params without finite grad: {bad if bad else 'none [OK]'}")
        assert not bad, f"parameters without a finite gradient: {bad}"
        report[f"loss_two_views={two}"] = out["total"].item()
    return report


def run_experiment(exp: ExperimentConfig, paths: PathsConfig, **trainer_kwargs) -> dict:
    return Trainer(exp, paths, **trainer_kwargs).run()


def run_suite(names: Sequence[str], seeds: Sequence[int], paths: PathsConfig,
              model_overrides: Optional[dict] = None, train_overrides: Optional[dict] = None,
              **trainer_kwargs) -> pd.DataFrame:
    """Runs every (experiment, seed) pair; after each run the aggregated table in results/ is refreshed.
    Finished runs are skipped automatically (resume=True), so the call can simply be repeated after an interruption."""
    rows = []
    for name in names:
        for seed in seeds:
            print(f"\n######################## {name} | seed {seed} ########################")
            exp = get_experiment(name, seed, model_overrides, train_overrides)
            rows.append(run_experiment(exp, paths, **trainer_kwargs))
            summarize_results(paths)
    return pd.DataFrame(rows)


def collect_runs(paths: PathsConfig) -> pd.DataFrame:
    rows = []
    for f in sorted(paths.results.glob("*/seed*/test_metrics.json")):
        rows.append(json.loads(f.read_text())["row"])
    return pd.DataFrame(rows)


def summarize_results(paths: PathsConfig, names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Writes results/all_runs.csv, results/ablation_table.csv and results/ablation_table.md
    (mean +- std over seeds, one row per experiment, ordered like the ablation suite)."""
    runs = collect_runs(paths)
    if runs.empty:
        print("no finished runs yet")
        return runs
    paths.results.mkdir(parents=True, exist_ok=True)
    runs.to_csv(paths.results / "all_runs.csv", index=False)
    order = {n: i for i, n in enumerate(ablation_suite())}
    if names is not None:
        runs = runs[runs["experiment"].isin(list(names))]
    rows = []
    base_acc = runs.loc[runs["experiment"] == "base", "test_acc1"].mean() if (runs["experiment"] == "base").any() else None
    for name, g in runs.groupby("experiment"):
        r = {"experiment": name, "seeds": len(g)}
        for col, label in METRIC_COLS:
            if col in g:
                mu, sd = g[col].mean(), (g[col].std(ddof=1) if len(g) > 1 else 0.0)
                r[label] = f"{mu:.2f} +/- {sd:.2f}" if len(g) > 1 else f"{mu:.2f}"
                r[label + " (mean)"] = mu
        if base_acc is not None:
            r["delta top-1 vs base"] = f"{g['test_acc1'].mean() - base_acc:+.2f}"
        rows.append(r)
    table = pd.DataFrame(rows)
    table["_o"] = table["experiment"].map(lambda n: order.get(n, 999))
    table = table.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    table.to_csv(paths.results / "ablation_table.csv", index=False)
    show_cols = ["experiment", "seeds"] + [l for _, l in METRIC_COLS if l in table] + \
                (["delta top-1 vs base"] if "delta top-1 vs base" in table else [])
    md = ["| " + " | ".join(show_cols) + " |", "|" + "|".join(["---"] * len(show_cols)) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in show_cols) + " |" for _, r in table.iterrows()]
    (paths.results / "ablation_table.md").write_text("\n".join(md), encoding="utf-8")
    return table[show_cols]


def load_for_inference(run_dir, device: Optional[torch.device] = None):
    """Loads the best model of a finished run (EMA weights when the run used EMA) in eval mode.
    Use this later to compare PatchVision-X with the CNN / ViT baselines on identical data."""
    device = device or get_device()
    ck = torch.load(Path(run_dir) / "best_model.pt", map_location=device, weights_only=False)
    exp = ExperimentConfig.from_dict(ck["exp"])
    model = build_model(exp.model).to(device)
    model.load_state_dict(ck["state"])
    return model.eval(), exp


# ----------------------------------------------------------------------------------------------
# Comparing runs (used by overfitting_trials.ipynb)
# ----------------------------------------------------------------------------------------------
def _nearest_clean_gap(h: pd.DataFrame, epoch: int):
    """clean-train accuracy and (clean - val) gap at the clean evaluation closest to `epoch`."""
    c = h.dropna(subset=["clean_acc1"])
    if c.empty:
        return float("nan"), float("nan")
    r = c.iloc[(c["epoch"] - epoch).abs().argsort()[:1]].iloc[0]
    return float(r["clean_acc1"]), float(r["clean_acc1"] - r["val_acc1"])


def compare_runs(paths: PathsConfig, names: Sequence[str], seed: int = 42, out_name: str = "overfitting_trials",
                 smooth: int = 10, show: bool = False) -> pd.DataFrame:
    """Table + plots comparing several finished (or running) runs, using VALIDATION numbers to judge.

    Columns (all from history.csv; test numbers are listed for the record but are NOT used to rank):
      best_val_acc1 / smoothed_val_acc1 (best `smooth`-epoch rolling mean, robust to the noisy 600-photo val set)
      min_val_loss  (the most sensitive overfitting signal: it turns upwards when memorizing starts)
      gap_at_best   (clean-train accuracy minus validation accuracy near the best epoch)
    Writes results/<out_name>/comparison.{csv,md,png}.
    """
    import matplotlib.pyplot as plt
    rows, hist = [], {}
    for n in names:
        rd = paths.results / n / f"seed{seed}"
        if not (rd / "history.csv").exists():
            continue
        h = pd.read_csv(rd / "history.csv")
        hist[n] = h
        best = h.loc[h["val_acc1"].idxmax()]
        lo = h.loc[h["val_loss"].idxmin()]
        clean_at_best, gap_at_best = _nearest_clean_gap(h, int(best["epoch"]))
        summary = next((rd / f for f in ("test_metrics.json", "val_metrics.json") if (rd / f).exists()), None)
        done = summary is not None
        row = json.loads(summary.read_text())["row"] if done else {}
        rows.append({
            "run": n, "finished": done, "epochs_run": int(h["epoch"].max()),
            "best_val_acc1": float(best["val_acc1"]), "best_val_epoch": int(best["epoch"]),
            "smoothed_val_acc1": float(h["val_acc1"].rolling(smooth, min_periods=smooth).mean().max()),
            "min_val_loss": float(lo["val_loss"]), "min_val_loss_epoch": int(lo["epoch"]),
            "clean_train_acc_at_best": clean_at_best, "gap_at_best": gap_at_best,
            "val_attn_in_box": float(h["val_attn_in_box"].iloc[-1]),
            "train_min": row.get("train_time_s", float("nan")) / 60, "sec_per_epoch": row.get("sec_per_epoch", float("nan")),
            "test_acc1 (record only)": row.get("test_acc1", float("nan")), "test_ece (record only)": row.get("test_ece", float("nan")),
        })
    table = pd.DataFrame(rows)
    out = paths.results / out_name
    out.mkdir(parents=True, exist_ok=True)
    if table.empty:
        return table
    table.to_csv(out / "comparison.csv", index=False)
    cols = list(table.columns)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in table.iterrows()]
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    for n, h in hist.items():
        ax[0].plot(h["epoch"], h["val_acc1"].rolling(5, min_periods=1).mean(), label=n)
        ax[1].plot(h["epoch"], h["val_loss"], label=n)
        c = h.dropna(subset=["clean_acc1"])
        ax[2].plot(c["epoch"], c["clean_acc1"] - c["val_acc1"], "o-", ms=3, label=n)
    ax[0].set_title("validation top-1 (%) (5-epoch smoothing)")
    ax[1].set_title("validation loss (lower is better)")
    ax[2].set_title("memorizing gap: clean-train minus validation (points)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return table
