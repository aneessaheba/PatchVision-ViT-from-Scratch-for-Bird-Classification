"""Self-supervised pretraining on the training images only. See patchvision/pretrain.py.

    python pretrain.py 42           -> seed 42, 300 epochs; re-running resumes
    python train.py g_box_pretrained 42   -> then trains from the saved weights
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from patchvision.config import PathsConfig
from patchvision.pretrain import pretrain_experiment, run_instance_pretraining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised pretraining on the training images.")
    parser.add_argument("seed", nargs="?", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    paths = PathsConfig(root=str(ROOT), results_dir="ViT_PatchVisionX/results_protocol")
    run_instance_pretraining(pretrain_experiment(args.seed, args.epochs), paths, num_workers=6)
