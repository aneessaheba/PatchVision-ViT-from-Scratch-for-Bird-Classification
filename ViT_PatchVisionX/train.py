"""Train one experiment from the terminal (same code path as training.ipynb).

    python train.py                 -> experiment "full", seed 42
    python train.py base 42         -> experiment "base", seed 42

Experiment names are the keys of patchvision.config.ablation_suite() or patchvision.config.overfit_trials().
Everything is written to ViT_PatchVisionX/results/<experiment>/seed<seed>/ ; re-running resumes an interrupted run.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # ViT_PatchVisionX/
ROOT = HERE.parent                           # project root: shared CUB_200_2011/ and DataPreprocessing_Results/
sys.path.insert(0, str(HERE))

from patchvision.config import (PathsConfig, architecture_trials, combined_trials, get_experiment, get_trial,
                                overfit_trials)
from patchvision.experiment import run_experiment

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "full"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    paths = PathsConfig(root=str(ROOT), results_dir="ViT_PatchVisionX/results")
    exp = (get_trial(name, seed) if name in {**overfit_trials(), **combined_trials(), **architecture_trials()}
           else get_experiment(name, seed=seed))
    run_experiment(exp, paths, num_workers=6, step_print_every=1, resume=True)
