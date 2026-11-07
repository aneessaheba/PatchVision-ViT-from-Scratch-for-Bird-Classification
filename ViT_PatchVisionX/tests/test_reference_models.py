"""Checks the plain ViT and PVTv2-B0 reference models, and that the ResNets are unchanged.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ViT_PatchVisionX"))
sys.path.insert(0, str(REPO / "CNN_Baseline"))

from cnn.config import CNNConfig, get_experiment
from cnn.engine import CNNTrainer
from cnn.model import CompactResNet, build_model
from patchvision.data import make_synthetic_dataset
from patchvision.optim import build_optimizer
from patchvision.config import TrainConfig


def count(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def test_plain_vit_size_matches_its_design():
    assert count(build_model(CNNConfig(arch="plain_vit"))) == pytest.approx(11.096, abs=0.001)


def test_pvtv2_b0_size_matches_the_timm_reference():
    model = build_model(CNNConfig(arch="pvtv2_b0", num_classes=1000))
    assert sum(p.numel() for p in model.parameters()) == 3666760


@pytest.mark.parametrize("arch", ["plain_vit", "pvtv2_b0"])
def test_forward_and_gradients(arch):
    torch.manual_seed(0)
    model = build_model(CNNConfig(arch=arch)).train()
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 200)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_plain_vit_class_token_and_position_are_not_decayed():
    model = build_model(CNNConfig(arch="plain_vit"))
    decay, no_decay = build_optimizer(model, TrainConfig()).param_groups
    assert any(p is model.cls_token for p in no_decay["params"])
    assert any(p is model.pos_embed for p in no_decay["params"])
    assert any(p is model.head.weight for p in decay["params"])


def test_resnet_is_unchanged_and_remains_the_default():
    torch.manual_seed(3)
    built = build_model(CNNConfig())
    torch.manual_seed(3)
    direct = CompactResNet(CNNConfig())
    assert isinstance(built, CompactResNet)
    assert all(torch.equal(a, b) for a, b in zip(built.state_dict().values(), direct.state_dict().values()))
    assert count(built) == pytest.approx(4.22, abs=0.01)


@pytest.mark.parametrize("name", ["plain_vit", "pvtv2_b0"])
def test_reference_models_train_end_to_end(tmp_path, name):
    paths = make_synthetic_dataset(tmp_path, n_classes=4, n_train=6, n_val=2, n_test=2, batch_size=8)
    train = dict(epochs=2, warmup_epochs=1, batch_size=8, weights_every=0, clean_eval_every=1, clean_eval_size=8)
    exp = get_experiment(name, 42, None, train, run_name=name)
    assert exp.train.use_bbox_loss is False and exp.train.use_consistency is False and exp.train.early_stop_patience == 0
    trainer = CNNTrainer(exp, paths, device=torch.device("cpu"), num_workers=0, quiet=True)
    assert trainer.run()["epochs_run"] == 2 and not list(trainer.run_dir.glob("test_*"))
