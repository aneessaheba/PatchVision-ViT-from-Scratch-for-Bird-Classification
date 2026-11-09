"""Checks the patch-size option: 16x16 stays exactly as before, 12x12 gives an 18x18 token grid on 216x216 images.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchvision.config import ExperimentConfig, ModelConfig, all_trials, get_experiment, get_trial
from patchvision.data import load_data_info, make_synthetic_dataset
from patchvision.engine import Trainer
from patchvision.models import PatchVisionX
from patchvision.utils import count_parameters

TRAIN = dict(epochs=1, warmup_epochs=1, batch_size=8, weights_every=0, clean_eval_every=1, clean_eval_size=8)


def test_default_patch_keeps_the_original_tokenizer():
    model = PatchVisionX(ModelConfig())
    conv = model.tokenizer.conv1
    assert (conv.kernel_size, conv.stride) == ((8, 8), (8, 8))
    assert model.grid == 14 and count_parameters(model)[0] == count_parameters(PatchVisionX(ModelConfig()))[0]


def test_configs_saved_before_the_option_load_as_16x16():
    saved = get_trial("g_box").to_dict()
    del saved["model"]["patch_size"], saved["train"]["img_size"]
    exp = ExperimentConfig.from_dict(json.loads(json.dumps(saved)))
    assert exp.model.patch_size == 16 and exp.train.img_size is None


def test_12x12_patches_give_324_tokens_pooled_to_81():
    model = PatchVisionX(ModelConfig(img_size=216, patch_size=12, depth=2)).eval()
    assert model.tokenizer.conv1.kernel_size == (6, 6)
    with torch.no_grad():
        logits, aux = model(torch.randn(2, 3, 216, 216))
    assert model.grid == 18 and logits.shape == (2, 200)
    assert aux["pool_assign"].shape == (2, 81, 324)


def test_sizes_that_do_not_fit_the_patch_are_refused():
    with pytest.raises(AssertionError):
        PatchVisionX(ModelConfig(img_size=224, patch_size=12))
    with pytest.raises(AssertionError):
        PatchVisionX(ModelConfig(img_size=225, patch_size=15))


def test_p12_trials_change_only_the_patch_and_image_size():
    trials = all_trials()
    for name in ("g_none", "g_box", "g_keypoints", "g_both"):
        fine, coarse = get_trial(f"{name}_p12").to_dict(), get_trial(name).to_dict()
        assert (fine["model"].pop("patch_size"), coarse["model"].pop("patch_size")) == (12, 16)
        assert (fine["train"].pop("img_size"), coarse["train"].pop("img_size")) == (216, None)
        assert fine["model"] == coarse["model"] and fine["train"] == coarse["train"]
        assert f"{name}_p12" in trials


@pytest.fixture
def paths(tmp_path):
    paths = make_synthetic_dataset(tmp_path, n_classes=4, n_train=6, n_val=2, n_test=2, batch_size=8)
    parts_dir = paths.cub / "parts"
    parts_dir.mkdir()
    (parts_dir / "parts.txt").write_text("1 beak\n2 left eye\n3 right eye\n")
    lines = []
    for i in range(1, 41):
        lines += [f"{i} 1 150 100 1", f"{i} 2 100 140 1", f"{i} 3 200 140 1"]
    (parts_dir / "part_locs.txt").write_text("\n".join(lines))
    return paths


def test_image_size_override_reaches_the_data(paths):
    assert load_data_info(paths).img_size == 224 and load_data_info(paths, 216).img_size == 216


@pytest.mark.parametrize("name", ["g_keypoints_p12", "g_box_p12"])
def test_p12_trials_train_end_to_end(paths, name):
    spec = all_trials()[name]
    exp = get_experiment(spec["base"], 42, {**spec["model"], "depth": 2}, {**spec["train"], **TRAIN}, run_name=name)
    trainer = Trainer(exp, paths, device=torch.device("cpu"), num_workers=0, quiet=True)
    trainer.run()
    assert trainer.grid == 18 and trainer.model.grid == 18
    assert next(iter(trainer.dl["train"]))[0].shape[-2:] == (216, 216)
    history = trainer.history[-1]
    assert (history["train_part"] > 0) == exp.train.use_part_guidance
    assert json.loads((trainer.run_dir / "config.json").read_text())["model"]["patch_size"] == 12
