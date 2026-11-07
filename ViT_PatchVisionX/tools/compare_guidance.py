"""Compares finished guidance runs using validation data only. Implements the decision score in docs/PROTOCOL.md.

Run from inside ViT_PatchVisionX/:
    python tools/compare_guidance.py g_box g_keypoints --seeds 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from patchvision.metrics import look_alike_metrics


def summarize_run(run_dir: Path, class_names) -> dict:
    history = pd.read_csv(run_dir / "history.csv")
    best_epoch = int(history.loc[history["val_acc1"].idxmax(), "epoch"])
    best = history[history["epoch"] == best_epoch].iloc[0]
    row = {"best_epoch": best_epoch, "score": float(history["val_acc1"].rolling(10, min_periods=10).mean().max()),
           "val_top1_at_best": float(best["val_acc1"]), "min_val_loss": float(history["val_loss"].min())}
    row.update(look_alike_metrics(np.load(run_dir / "best_val_confusion.npy"), class_names))
    for name in ("val_part_error_assigned", "val_part_error_best"):
        row[name] = float(best[name]) if name in history else float("nan")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--results", default=str(HERE / "results_protocol"))
    parser.add_argument("--val-csv", default=str(HERE.parent / "DataPreprocessing_Results" / "val.csv"))
    args = parser.parse_args()
    val = pd.read_csv(args.val_csv)
    class_names = val.drop_duplicates("label").sort_values("label")["class_name"].tolist()
    rows = []
    for name in args.runs:
        for seed in args.seeds:
            run_dir = Path(args.results) / name / f"seed{seed}"
            if not (run_dir / "val_metrics.json").exists():
                print(f"NOT FINISHED: {run_dir}")
                continue
            if (run_dir / "test_metrics.json").exists():
                print(f"Note: {run_dir} also has test results; they are not used here.")
            rows.append({"run": name, "seed": seed, **summarize_run(run_dir, class_names)})
    if not rows:
        return
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", None)
    print(table.round(3).to_string(index=False))
    means = table.drop(columns="seed").groupby("run", sort=False).mean(numeric_only=True).round(3)
    print("\nMeans over seeds:\n" + means.to_string())
    print("\nA change is adopted only if its score is at least 1.0 point higher. A single seed can only screen an idea.")


if __name__ == "__main__":
    main()
