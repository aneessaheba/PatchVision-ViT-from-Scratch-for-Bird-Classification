"""Checks that CUB part points stay on the correct pixels and labels through the data pipeline.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchvision.data import (CUBDataset, MaskAwareTrivialAugmentWide, PartAnnotations, PartAwareHorizontalFlip,
                              build_transforms, flip_part_points, load_part_annotations, part_points_from_map)

NAMES = ["beak", "left eye", "right eye", "tail"]
FLIP_ORDER = [0, 2, 1, 3]
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
WIDTH, HEIGHT = 400, 300
POINTS = np.array([[200, 60, 1], [120, 120, 1], [280, 120, 1], [200, 240, 0]], dtype=np.float32)
COLOURS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}


def make_dataset(tmp_path, transform, boxes=None, parts=None, **kwargs):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i, colour in COLOURS.items():
        x, y = POINTS[i, :2].astype(int)
        img[y - 8:y + 9, x - 8:x + 9] = colour
    (tmp_path / "images").mkdir(exist_ok=True)
    Image.fromarray(img).save(tmp_path / "images" / "a.png")
    pd.DataFrame([{"image_id": 1, "filepath": "images/a.png", "label": 0, "class_name": "bird"}]).to_csv(
        tmp_path / "split.csv", index=False)
    parts = parts or PartAnnotations(NAMES, FLIP_ORDER, {1: POINTS.copy()})
    return CUBDataset(tmp_path / "split.csv", transform, boxes, tmp_path, parts=parts, **kwargs)


def expected_scaled():
    out = np.zeros((4, 3))
    out[:3, 0], out[:3, 1], out[:3, 2] = POINTS[:3, 0] / WIDTH, POINTS[:3, 1] / HEIGHT, 1
    return out


def test_annotation_files_are_parsed(tmp_path):
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    (parts_dir / "parts.txt").write_text("1 beak\n2 left eye\n3 right eye\n")
    (parts_dir / "part_locs.txt").write_text("7 1 10.5 20.0 1\n7 2 0.0 0.0 0\n7 3 30.0 40.0 1\n")
    parts = load_part_annotations(tmp_path)
    assert parts.names == ["beak", "left eye", "right eye"]
    assert parts.flip_order == [0, 2, 1]
    assert parts.points[7].tolist() == [[10.5, 20.0, 1.0], [0.0, 0.0, 0.0], [30.0, 40.0, 1.0]]


def test_resize_keeps_points_and_hides_unlabelled_parts(tmp_path):
    transform = build_transforms("none", 224, MEAN, STD, train=False)
    _, _, _, _, points = make_dataset(tmp_path, transform)[0]
    assert points[3, 2] == 0
    assert np.abs(points.numpy()[:3] - expected_scaled()[:3]).max() < 2 / 224


def test_box_mask_is_still_returned(tmp_path):
    transform = build_transforms("none", 224, MEAN, STD, train=False)
    ds = make_dataset(tmp_path, transform, boxes={1: (100.0, 50.0, 200.0, 200.0)})
    _, box, *_ = ds[0]
    assert box.shape == (1, 224, 224) and set(box.unique().tolist()) == {0.0, 1.0}
    assert box.mean().item() == pytest.approx(200 * 200 / (WIDTH * HEIGHT), abs=0.02)


def test_mirror_moves_points_and_swaps_left_and_right(tmp_path):
    img = tv_tensors.Image(torch.zeros(3, 8, 8, dtype=torch.uint8))
    part_map = torch.zeros(2, 8, 8, dtype=torch.uint8)
    part_map[1, 3, 1] = 2      # left eye, on the left of the image
    part_map[1, 3, 6] = 3      # right eye, on the right of the image
    part_map[1, 1, 2] = 1      # beak has no partner
    flip = PartAwareHorizontalFlip(FLIP_ORDER, p=1.0)
    _, out = flip(img, tv_tensors.Mask(part_map))
    out = torch.as_tensor(out)[1]
    assert out[3, 6] == 3 and out[3, 1] == 2 and out[1, 5] == 1


def test_flip_helper_matches_a_mirrored_map():
    part_map = torch.zeros(20, 20)
    part_map[5, 4] = 2
    part_map[5, 15] = 3
    part_map[2, 10] = 1
    points = part_points_from_map(part_map, 4)
    mirrored = torch.flip(part_map, dims=[1])
    lookup = torch.tensor([0, 1, 3, 2, 4]).float()
    relabelled = lookup[mirrored.long()]
    assert torch.allclose(flip_part_points(points, FLIP_ORDER), part_points_from_map(relabelled, 4))


def test_invisible_parts_stay_at_zero_after_flip_helper():
    points = torch.tensor([[0.2, 0.2, 1.0], [0.0, 0.0, 0.0], [0.7, 0.3, 1.0], [0.0, 0.0, 0.0]])
    out = flip_part_points(points, FLIP_ORDER)
    assert out[2].tolist() == [0.0, 0.0, 0.0] and out[3].tolist() == [0.0, 0.0, 0.0]
    assert out[1].tolist() == pytest.approx([0.3, 0.3, 1.0])


def colour_centroid(image, channel):
    pixels = image[channel] > 0.5
    others = [c for c in range(3) if c != channel]
    pixels &= (image[others[0]] < 0.0) & (image[others[1]] < 0.0)
    ys, xs = torch.nonzero(pixels, as_tuple=True)
    if len(xs) < 5:
        return None
    return (xs.float().mean().item() + 0.5) / image.shape[2], (ys.float().mean().item() + 0.5) / image.shape[1]


@pytest.mark.parametrize("tier", ["basic"])
def test_points_follow_the_image_through_random_crops_and_mirrors(tmp_path, tier):
    torch.manual_seed(0)
    transform = build_transforms(tier, 224, MEAN, STD, train=True, part_flip_order=FLIP_ORDER)
    ds = make_dataset(tmp_path, transform)
    checked, swapped = 0, 0
    for _ in range(300):
        image, _, _, _, points = ds[0]
        beak = colour_centroid(image, 0)
        if beak is None or points[0, 2] == 0:
            continue
        assert abs(beak[0] - points[0, 0].item()) < 0.04 and abs(beak[1] - points[0, 1].item()) < 0.04
        left_dot, right_dot = colour_centroid(image, 1), colour_centroid(image, 2)
        if left_dot is None or right_dot is None or points[1, 2] == 0 or points[2, 2] == 0:
            continue
        checked += 1
        mirrored = left_dot[0] > right_dot[0]     # the green dot was on the left before any mirror
        swapped += int(mirrored)
        left_label, right_label = (2, 1) if mirrored else (1, 2)
        assert abs(left_dot[0] - points[left_label, 0].item()) < 0.04
        assert abs(right_dot[0] - points[right_label, 0].item()) < 0.04
    assert checked > 50 and 0 < swapped < checked


@pytest.mark.parametrize("op", ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"])
def test_points_follow_the_geometric_operations_of_auto_augmentation(op):
    aug = MaskAwareTrivialAugmentWide()
    fn, _ = aug._AUGMENTATION_SPACE[op]
    magnitude = float(fn(aug.num_magnitude_bins, 224, 224)[10])
    image = torch.zeros(3, 224, 224, dtype=torch.uint8)
    image[:, 100:121, 100:121] = 255
    part_map = torch.zeros(1, 224, 224, dtype=torch.uint8)
    part_map[:, 100:121, 100:121] = 1
    from torchvision.transforms import v2 as T
    out_i = aug._apply_image_or_video_transform(tv_tensors.Image(image), op, magnitude,
                                                interpolation=T.InterpolationMode.NEAREST, fill=aug._fill)
    out_m = aug._apply_image_or_video_transform(tv_tensors.Mask(part_map), op, magnitude,
                                                interpolation=T.InterpolationMode.NEAREST, fill=aug._fill)
    assert torch.equal(torch.as_tensor(out_i)[0] > 127, torch.as_tensor(out_m)[0] > 0)


def test_box_crop_shifts_points_and_hides_parts_outside_it(tmp_path):
    random.seed(0)
    transform = build_transforms("none", 224, MEAN, STD, train=False)
    ds = make_dataset(tmp_path, transform, boxes={1: (100.0, 40.0, 120.0, 100.0)}, train=True, box_crop_prob=1.0,
                      box_crop_margin=(0.0, 0.0))
    _, _, _, _, points = ds[0]
    assert abs(points[0, 0].item() - (200 - 100) / 120) < 0.03 and abs(points[0, 1].item() - (60 - 40) / 100) < 0.03
    assert abs(points[1, 0].item() - (120 - 100) / 120) < 0.03
    assert points[2, 2] == 0


def test_points_can_be_collated_into_a_batch(tmp_path):
    transform = build_transforms("none", 224, MEAN, STD, train=False)
    ds = make_dataset(tmp_path, transform)
    batch = next(iter(torch.utils.data.DataLoader([ds[0], ds[0]], batch_size=2)))
    assert batch[4].shape == (2, 4, 3)


@pytest.mark.parametrize("tier", ["basic", "strong", "auto"])
def test_part_aware_transform_produces_the_same_images(tmp_path, tier):
    def images(part_flip_order):
        torch.manual_seed(5)
        transform = build_transforms(tier, 224, MEAN, STD, train=True, part_flip_order=part_flip_order)
        out = []
        for _ in range(12):
            img = tv_tensors.Image(torch.randint(0, 255, (3, 240, 320), dtype=torch.uint8))
            channels = 2 if part_flip_order else 1
            out.append(torch.as_tensor(transform(img, tv_tensors.Mask(torch.zeros(channels, 240, 320, dtype=torch.uint8)))[0]))
        return out
    for a, b in zip(images(None), images(FLIP_ORDER)):
        assert torch.equal(a, b)


def test_without_parts_the_dataset_returns_four_items(tmp_path):
    ds = make_dataset(tmp_path, build_transforms("none", 224, MEAN, STD, train=False))
    ds.parts = None
    assert len(ds[0]) == 4
