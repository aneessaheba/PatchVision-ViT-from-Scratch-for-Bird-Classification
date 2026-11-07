"""Checks the part-guided pooling loss, its data handling, and a short end-to-end training run.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchvision.config import ModelConfig, all_trials, get_experiment, get_trial
from patchvision.data import (MixupCutmix, build_loaders, hide_erased_points, load_data_info, make_synthetic_dataset,
                              select_annotated_images)
from patchvision.engine import Trainer
from patchvision.losses import PartPoolingLoss
from patchvision.models import PatchVisionX

GRID = 14
CPU = torch.device("cpu")
TRAIN = dict(epochs=2, warmup_epochs=1, batch_size=8, weights_every=0, clean_eval_every=1, clean_eval_size=8)


def peaked_assign(points):
    loss = PartPoolingLoss(GRID)
    dist2 = (loss.xs - points[..., 0, None]) ** 2 + (loss.ys - points[..., 1, None]) ** 2
    return torch.softmax(-dist2 / (2 * loss.sigma ** 2), dim=-1)


def test_loss_is_lower_when_queries_sit_on_their_parts():
    points = torch.tensor([[[0.2, 0.2, 1.0], [0.8, 0.7, 1.0]]])
    criterion = PartPoolingLoss(GRID)
    right = peaked_assign(points)
    wrong = peaked_assign(points.flip(1))
    weights = torch.ones(1)
    right_loss, right_error = criterion(right, points, weights)
    wrong_loss, wrong_error = criterion(wrong, points, weights)
    assert right_loss < wrong_loss and right_error < 0.02 < wrong_error


def test_hidden_parts_and_extra_queries_do_not_count():
    points = torch.tensor([[[0.2, 0.2, 1.0], [0.8, 0.7, 0.0]]])
    criterion = PartPoolingLoss(GRID)
    assign = peaked_assign(points)
    moved = assign.clone()
    moved[0, 1] = torch.roll(moved[0, 1], 50)
    extra = torch.cat([assign, torch.full((1, 3, GRID * GRID), 1 / GRID ** 2)], dim=1)
    weights = torch.ones(1)
    base = criterion(assign, points, weights)[0]
    assert criterion(moved, points, weights)[0] == pytest.approx(base.item())
    assert criterion(extra, points, weights)[0] == pytest.approx(base.item())


def test_no_visible_parts_gives_zero_loss():
    points = torch.zeros(2, 3, 3)
    assign = torch.full((2, 3, GRID * GRID), 1 / GRID ** 2)
    loss, error = PartPoolingLoss(GRID)(assign, points, torch.ones(2))
    assert loss == 0 and error == 0


def test_erased_points_are_hidden_but_black_image_regions_are_not():
    image = torch.full((3, 100, 100), -2.0)
    image[:, 40:60, 40:60] = 0.0
    points = torch.tensor([[0.5, 0.5, 1.0], [0.1, 0.1, 1.0], [0.0, 0.0, 0.0]])
    out = hide_erased_points(points, image)
    assert out[0].tolist() == [0.0, 0.0, 0.0] and out[1].tolist() == pytest.approx([0.1, 0.1, 1.0])


def test_cutmix_hides_points_under_the_patch_and_mixup_uses_the_dominant_image():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.arange(1.0, 5.0)[:, None, None, None].expand(4, 3, 64, 64)    # each image is one distinct value
    y, mask = torch.arange(4), torch.zeros(4, 1, 64, 64)
    points = torch.rand(4, 40, 3)
    points[..., 2] = 1.0
    seen, hidden_total = set(), 0
    for _ in range(80):
        mixed, _, _, info = MixupCutmix(10, 0.2, 1.0, 0.5, 0.0)(x, y, mask, points=points)
        seen.add(info["type"])
        if info["type"] == "cutmix":
            pasted = mixed[:, 0] != x[:, 0]
            for image in range(4):
                for part in range(40):
                    px, py = int(points[image, part, 0] * 64), int(points[image, part, 1] * 64)
                    assert (info["points"][image, part, 2] == 0) == pasted[image, py, px]
            hidden_total += int((info["points"][..., 2] == 0).sum())
            assert (info["point_weight"] == 1).all()
        elif info["type"] == "mixup":
            expected = points if info["lam"] >= 0.5 else points.flip(0)
            assert torch.equal(info["points"], expected)
            assert info["point_weight"][0].item() == pytest.approx(max(info["lam"], 1 - info["lam"]))
    assert seen == {"cutmix", "mixup"} and hidden_total > 0


def test_without_points_the_mixer_adds_nothing():
    np.random.seed(0)
    x, y, mask = torch.rand(4, 3, 32, 32), torch.arange(4), torch.zeros(4, 1, 32, 32)
    _, _, _, info = MixupCutmix(10)(x, y, mask)
    assert set(info) == {"type", "lam"}


def test_keypoint_budget_selects_the_right_share_across_species():
    records = [{"image_id": i, "label": i % 200} for i in range(5400)]
    few = select_annotated_images(records, 0.005, seed=1)
    assert len(few) == 27 and len(set(few)) == 27 and len({i % 200 for i in few}) == 27
    tenth = select_annotated_images(records, 0.1, seed=1)
    assert len(tenth) == 540 and {i % 200 for i in tenth} == set(range(200))
    assert few == select_annotated_images(records, 0.005, seed=1)
    assert few != select_annotated_images(records, 0.005, seed=2)


def test_model_exposes_a_pooling_assignment_that_learns():
    torch.manual_seed(0)
    model = PatchVisionX(ModelConfig(depth=2)).train()
    criterion = PartPoolingLoss(GRID)
    images = torch.randn(4, 3, 224, 224)
    points = torch.tensor([[0.2, 0.3, 1.0], [0.7, 0.6, 1.0]]).expand(4, -1, -1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        _, aux = model(images)
        assert aux["pool_assign"].shape == (4, 49, 196)
        loss, _ = criterion(aux["pool_assign"], points, torch.ones(4))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.8


def test_guidance_trials_turn_the_right_signals_on():
    expected = {"g_none": (False, False, False), "g_box": (True, True, False),
                "g_keypoints": (False, False, True), "g_both": (True, True, True)}
    for name, (box_loss, box_crop, parts) in expected.items():
        cfg = get_trial(name)
        assert (cfg.train.use_bbox_loss, cfg.train.use_box_crop, cfg.train.use_part_guidance) == (box_loss, box_crop, parts)
        assert cfg.model.drop_path == 0.3 and cfg.train.aug_tier == "auto" and cfg.train.early_stop_patience == 0
    assert get_trial("g_keypoints_f0.005").train.part_fraction == 0.005
    assert get_experiment("full").train.use_part_guidance is False
    assert sum(name.startswith("g_keypoints_f") for name in all_trials()) == 4


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


@pytest.mark.parametrize("name", ["g_keypoints", "g_both", "g_keypoints_f0.1", "g_box", "g_none"])
def test_every_guidance_trial_trains_end_to_end(paths, name):
    spec = all_trials()[name]
    exp = get_experiment(spec["base"], 42, {**spec["model"], "depth": 2}, {**spec["train"], **TRAIN}, run_name=name)
    trainer = Trainer(exp, paths, device=CPU, num_workers=0, quiet=True)
    row = trainer.run()
    assert row["epochs_run"] == 2
    assert not list(trainer.run_dir.glob("test_*"))
    history = json.loads(json.dumps(trainer.history[-1], default=float))
    if exp.train.use_part_guidance:
        assert history["train_part"] > 0 and 0 < history["train_part_error"] < 1.5
    else:
        assert history["train_part"] == 0


def test_validation_and_test_loaders_do_not_receive_parts(paths):
    exp = get_trial("g_keypoints")
    info = load_data_info(paths)
    exp.train.batch_size = 8
    _, loaders = build_loaders(info, exp.train, 42, 0, False)
    assert len(next(iter(loaders["train"]))) == 5
    for split in ("val", "test", "train_clean"):
        assert len(next(iter(loaders[split]))) == 4


def test_budget_removes_keypoints_from_other_images(paths):
    exp = get_trial("g_keypoints_f0.1")
    info = load_data_info(paths)
    exp.train.batch_size = 8
    datasets, _ = build_loaders(info, exp.train, 42, 0, False)
    train_ids = {int(r["image_id"]) for r in datasets["train"].records}
    assert len(datasets["train"].parts.points) == 2 and set(datasets["train"].parts.points) <= train_ids
