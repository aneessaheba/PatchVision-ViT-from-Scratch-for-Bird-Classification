"""Train one CNN experiment from the terminal (same code path as training.ipynb).

    python train.py                  -> experiment "cnn_small", seed 42
    python train.py cnn_large 42     -> experiment "cnn_large", seed 42
    python train.py cnn_large 42 --final-test   -> also evaluate the test set; reuses a finished training

Experiment names are the keys of cnn.config.CNN_SUITE.  Run it from inside CNN_Baseline/ .
Without --final-test the test set is not read, so development runs stay on validation data.
Everything is written to CNN_Baseline/results/<experiment>/seed<seed>/ ; re-running resumes an interrupted run.
The shared data (CUB_200_2011/, DataPreprocessing_Results/) is read from the repository root (one level up).
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # CNN_Baseline/
ROOT = HERE.parent                           # repository root: shared CUB_200_2011/ and DataPreprocessing_Results/
sys.path.insert(0, str(HERE))

from cnn.config import PathsConfig, get_experiment
from cnn.engine import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one CNN experiment.")
    parser.add_argument("name", nargs="?", default="cnn_small")
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--final-test", action="store_true", help="evaluate the test set after training")
    args = parser.parse_args()
    paths = PathsConfig(root=str(ROOT), results_dir="CNN_Baseline/results")
    run_experiment(get_experiment(args.name, seed=args.seed), paths, num_workers=6, step_print_every=1, resume=True,
                   run_test=args.final_test)
