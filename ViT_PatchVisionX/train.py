"""Train one experiment from the terminal (same code path as training.ipynb).

    python train.py                 -> experiment "full", seed 42
    python train.py base 42         -> experiment "base", seed 42
    python train.py base 42 --final-test   -> also evaluate the test set; reuses a finished training

Experiment names are the keys of patchvision.config.ablation_suite() or all_trials(). Without --final-test the test set is not read, so development runs stay on validation data.
Everything is written to ViT_PatchVisionX/results_protocol/<experiment>/seed<seed>/ , kept apart from the earlier runs
in results/ . Re-running resumes an interrupted run.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # ViT_PatchVisionX/
ROOT = HERE.parent                           # project root: shared CUB_200_2011/ and DataPreprocessing_Results/
sys.path.insert(0, str(HERE))

from patchvision.config import PathsConfig, all_trials, get_experiment, get_trial
from patchvision.experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one PatchVision experiment.")
    parser.add_argument("name", nargs="?", default="full")
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--final-test", action="store_true", help="evaluate the test set after training")
    args = parser.parse_args()
    paths = PathsConfig(root=str(ROOT), results_dir="ViT_PatchVisionX/results_protocol")
    exp = get_trial(args.name, args.seed) if args.name in all_trials() else get_experiment(args.name, seed=args.seed)
    run_experiment(exp, paths, num_workers=6, step_print_every=1, resume=True, run_test=args.final_test)
