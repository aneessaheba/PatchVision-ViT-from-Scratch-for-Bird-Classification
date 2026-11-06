"""Checks that training does not touch the test set unless asked, for both the ViT and the CNN trainer.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ViT_PatchVisionX"))
sys.path.insert(0, str(REPO / "CNN_Baseline"))

from cnn.config import get_experiment as get_cnn_experiment
from cnn.engine import CNNTrainer
from patchvision.config import get_experiment
from patchvision.data import make_synthetic_dataset
from patchvision.engine import Trainer

TRAIN = dict(epochs=2, warmup_epochs=1, batch_size=8, weights_every=0, early_stop_patience=0,
             clean_eval_every=1, clean_eval_size=8)
CPU = torch.device("cpu")


def vit(paths, run_test=False):
    exp = get_experiment("full", 42, {"depth": 2}, TRAIN, run_name="vit_run")
    return Trainer(exp, paths, device=CPU, num_workers=0, quiet=True, run_test=run_test)


def cnn(paths, run_test=False):
    exp = get_cnn_experiment("cnn_small", 42, None, TRAIN, run_name="cnn_run")
    return CNNTrainer(exp, paths, device=CPU, num_workers=0, quiet=True, run_test=run_test)


@pytest.fixture(params=[vit, cnn], ids=["vit", "cnn"])
def make(request):
    return request.param


@pytest.fixture
def paths(tmp_path):
    return make_synthetic_dataset(tmp_path, n_classes=4, n_train=6, n_val=2, n_test=2, batch_size=8)


def written_test_files(run_dir):
    return sorted(p.name for p in run_dir.glob("test_*"))


def block_test_loader(monkeypatch, trainer):
    original = trainer._setup

    def setup():
        original()
        loader = trainer.dl["test"]
        monkeypatch.setattr(type(loader), "__iter__", guard(type(loader).__iter__, loader))

    trainer._setup = setup


def guard(original_iter, blocked):
    def __iter__(self):
        assert self is not blocked, "the test loader was read"
        return original_iter(self)

    return __iter__


def test_default_run_never_reads_the_test_set(make, paths, monkeypatch):
    trainer = make(paths)
    block_test_loader(monkeypatch, trainer)
    row = trainer.run()
    assert written_test_files(trainer.run_dir) == []
    assert (trainer.run_dir / "val_metrics.json").exists()
    assert row["epochs_run"] == 2 and row["best_epoch"] >= 1
    assert not any(key.startswith("test_") for key in row)


def test_finished_run_is_not_trained_again(make, paths):
    first = make(paths)
    first.run()
    log = (first.run_dir / "train_log.txt").read_text()
    assert make(paths).run()["epochs_run"] == 2
    assert (first.run_dir / "train_log.txt").read_text() == log


def test_final_test_reuses_the_trained_model(make, paths):
    first = make(paths)
    trained = first.run()
    log = (first.run_dir / "train_log.txt").read_text()
    best = torch.load(first.run_dir / "best_model.pt", weights_only=False)["state"]

    row = make(paths, run_test=True).run()

    assert "EPOCH" not in (first.run_dir / "train_log.txt").read_text()[len(log):]
    for key in ("best_epoch", "epochs_run", "stopped_early", "train_time_s"):
        assert row[key] == trained[key]
    assert row["test_acc1"] >= 0
    assert (first.run_dir / "test_metrics.json").exists()
    after = torch.load(first.run_dir / "best_model.pt", weights_only=False)["state"]
    assert all(torch.equal(best[k], after[k]) for k in best)
    assert make(paths).run()["test_acc1"] == row["test_acc1"]


def test_run_with_final_test_from_the_start(make, paths):
    trainer = make(paths, run_test=True)
    row = trainer.run()
    assert row["epochs_run"] == 2 and "test_acc1" in row
    assert not (trainer.run_dir / "val_metrics.json").exists()


def test_final_test_without_last_checkpoint_keeps_the_logs(make, paths):
    first = make(paths)
    first.run()
    log = (first.run_dir / "train_log.txt").read_text()
    (first.run_dir / "last.pt").unlink()
    with pytest.raises(FileNotFoundError):
        make(paths, run_test=True).run()
    assert (first.run_dir / "train_log.txt").read_text() == log
    assert json.loads((first.run_dir / "val_metrics.json").read_text())["row"]["epochs_run"] == 2
