"""
Data: everything that is read from DataPreprocessing_Results/ (splits, normalisation, augmentation tiers,
mixup/CutMix settings) plus the CUB images and bounding boxes.

* Bounding boxes are returned as a mask that receives the SAME geometric augmentation as the image.
  They are only used by training-time regularisers; the network never gets them as input.
* Validation / test use only the deterministic 'none' tier (Resize + Normalize), exactly as in preprocess_config.json.
"""
from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from .config import PathsConfig, TrainConfig
from .losses import smooth_one_hot
from .utils import seed_worker

# An erased patch is still part of the bird's box: RandomErasing leaving the mask alone is intended.
warnings.filterwarnings("ignore", message=".*RandomErasing.*passing through.*Mask.*")


# ----------------------------------------------------------------------------------------------
# What preprocessing decided
# ----------------------------------------------------------------------------------------------
@dataclass
class DataInfo:
    img_size: int
    batch_size: int
    mean: List[float]
    std: List[float]
    mixcfg: dict
    splits: dict
    tiers: dict
    num_classes: int
    train_csv: Path
    val_csv: Path
    test_csv: Path
    cub_root: Path
    bboxes: Optional[Dict[int, Tuple[float, float, float, float]]]


def load_bboxes(path: Optional[os.PathLike]) -> Optional[Dict[int, Tuple[float, float, float, float]]]:
    """CUB bounding_boxes.txt -> {image_id: (x, y, w, h)}; None if the file does not exist."""
    if path is None or not Path(path).exists():
        return None
    boxes = {}
    with open(path) as f:
        for line in f:
            i, x, y, w, h = line.split()
            boxes[int(i)] = (float(x), float(y), float(w), float(h))
    return boxes


@dataclass
class PartAnnotations:
    """CUB part keypoints. points: {image_id: (P, 3) array of x, y, visible} in original image pixels.
    flip_order[i] is the part that part i becomes when the image is mirrored (left eye <-> right eye)."""
    names: List[str]
    flip_order: List[int]
    points: Dict[int, np.ndarray]


def load_part_annotations(cub_root: os.PathLike) -> PartAnnotations:
    """Reads CUB's parts/parts.txt and parts/part_locs.txt. Used only as a training signal, never as model input."""
    parts_dir = Path(cub_root) / "parts"
    names = []
    for line in (parts_dir / "parts.txt").read_text().splitlines():
        if line.strip():
            names.append(line.split(maxsplit=1)[1].strip())
    index = {name: i for i, name in enumerate(names)}
    flip_order = []
    for name in names:
        if name.startswith("left "):
            flip_order.append(index["right " + name[len("left "):]])
        elif name.startswith("right "):
            flip_order.append(index["left " + name[len("right "):]])
        else:
            flip_order.append(index[name])
    points: Dict[int, np.ndarray] = {}
    for line in (parts_dir / "part_locs.txt").read_text().splitlines():
        if not line.strip():
            continue
        image_id, part_id, x, y, visible = line.split()
        row = points.setdefault(int(image_id), np.zeros((len(names), 3), dtype=np.float32))
        row[int(part_id) - 1] = (float(x), float(y), float(visible))
    return PartAnnotations(names, flip_order, points)


def paint_part_map(points: np.ndarray, offset: Tuple[int, int], width: int, height: int) -> torch.Tensor:
    """(1, H, W) uint8 map: a small disk with value part_index + 1 at each visible part. Going through the same
    geometric transforms as the image keeps every part on the correct pixels; later parts overwrite earlier ones."""
    part_map = torch.zeros(1, height, width, dtype=torch.uint8)
    radius = max(3, round(0.012 * max(width, height)))
    for i, (x, y, visible) in enumerate(points):
        cx, cy = x - offset[0], y - offset[1]
        if visible <= 0 or not (0 <= cx < width and 0 <= cy < height):
            continue
        x0, x1 = max(0, int(cx) - radius), min(width, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(height, int(cy) + radius + 1)
        ys, xs = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
        disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        part_map[0, y0:y1, x0:x1][disk] = i + 1
    return part_map


def part_points_from_map(part_map: torch.Tensor, num_parts: int) -> torch.Tensor:
    """(H, W) part map -> (P, 3) tensor of x, y in [0, 1] and visible. A part cropped out of the image is not visible."""
    height, width = part_map.shape
    out = torch.zeros(num_parts, 3)
    for i in range(num_parts):
        ys, xs = torch.nonzero(part_map == i + 1, as_tuple=True)
        if len(xs):
            out[i] = torch.stack([(xs.float().mean() + 0.5) / width, (ys.float().mean() + 0.5) / height,
                                  torch.tensor(1.0)])
    return out


def hide_erased_points(points: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """RandomErasing leaves the part map alone, so a part can sit on a blank patch. Erasing fills with exact zeros
    after normalisation, which real pixels do not produce in all channels, so those parts become not visible."""
    _, height, width = image.shape
    out = points.clone()
    for i, (x, y, visible) in enumerate(points.tolist()):
        if visible > 0 and (image[:, min(int(y * height), height - 1), min(int(x * width), width - 1)] == 0).all():
            out[i] = 0.0
    return out


def select_annotated_images(records: Sequence[dict], fraction: float, seed: int) -> List[int]:
    """Image ids whose keypoints may be used: `fraction` of the images, spread across species as evenly as possible."""
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[int]] = {}
    for r in records:
        by_class.setdefault(int(r["label"]), []).append(int(r["image_id"]))
    labels = sorted(by_class)
    rng.shuffle(labels)
    for label in labels:
        rng.shuffle(by_class[label])
    wanted = max(1, round(fraction * len(records)))
    chosen: List[int] = []
    depth = 0
    while len(chosen) < wanted:
        for label in labels:
            if depth < len(by_class[label]) and len(chosen) < wanted:
                chosen.append(by_class[label][depth])
        depth += 1
    return chosen


def flip_part_points(points: torch.Tensor, flip_order: Sequence[int]) -> torch.Tensor:
    """Part points of a mirrored image: (..., P, 3) -> mirrored x, with left and right parts swapped."""
    out = points[..., list(flip_order), :].clone()
    out[..., 0] = torch.where(out[..., 2] > 0, 1.0 - out[..., 0], out[..., 0])
    return out


def load_data_info(paths: PathsConfig) -> DataInfo:
    prep = paths.prep
    for name in ("preprocess_config.json", "norm_stats.json", "train.csv", "val.csv", "test.csv"):
        if not (prep / name).exists():
            raise FileNotFoundError(f"{prep / name} not found - check PathsConfig.root / prep_dir")
    cfg = json.loads((prep / "preprocess_config.json").read_text())
    stats = json.loads((prep / "norm_stats.json").read_text())
    assert np.allclose(stats["mean"], cfg["mean"]) and np.allclose(stats["std"], cfg["std"]), \
        "norm_stats.json and preprocess_config.json disagree on mean/std"
    num_classes = int(pd.read_csv(prep / "train.csv")["label"].nunique())
    return DataInfo(img_size=cfg["img_size"], batch_size=cfg["batch_size"], mean=stats["mean"], std=stats["std"],
                    mixcfg=cfg["mixup_cutmix"], splits=cfg.get("splits", {}), tiers=cfg.get("train_transforms", {}),
                    num_classes=num_classes, train_csv=prep / "train.csv", val_csv=prep / "val.csv",
                    test_csv=prep / "test.csv", cub_root=paths.cub, bboxes=load_bboxes(paths.cub / "bounding_boxes.txt"))


# ----------------------------------------------------------------------------------------------
# Augmentation tiers (as described in preprocess_config.json)
# ----------------------------------------------------------------------------------------------
class MaskAwareTrivialAugmentWide(T.TrivialAugmentWide):
    """TrivialAugmentWide that also moves the bird-box mask: the SAME sampled op/magnitude is applied to the
    mask when it is geometric (nearest interpolation); colour ops touch the image only. Uses a few private
    torchvision hooks; if a future torchvision removes them, use the 'strong' tier instead."""

    _GEOMETRIC = {"ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"}

    def forward(self, img, mask):
        h, w = T.functional.get_size(img)
        transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
        magnitudes = magnitudes_fn(self.num_magnitude_bins, h, w)
        if magnitudes is not None:
            magnitude = float(magnitudes[int(torch.randint(self.num_magnitude_bins, ()))])
            if signed and torch.rand(()) <= 0.5:
                magnitude *= -1
        else:
            magnitude = 0.0
        img = self._apply_image_or_video_transform(img, transform_id, magnitude,
                                                   interpolation=self.interpolation, fill=self._fill)
        if transform_id in self._GEOMETRIC:
            mask = self._apply_image_or_video_transform(mask, transform_id, magnitude,
                                                        interpolation=T.InterpolationMode.NEAREST, fill=self._fill)
        return img, mask


class PartAwareHorizontalFlip(torch.nn.Module):
    """Random mirror that also relabels left and right parts in channel 1 of the mask. Uses the same random draw
    as T.RandomHorizontalFlip, so the augmented images do not change."""

    def __init__(self, flip_order: Sequence[int], p: float = 0.5):
        super().__init__()
        self.p = p
        self.lookup = torch.tensor([0] + [i + 1 for i in flip_order], dtype=torch.uint8)

    def forward(self, img, mask):
        if torch.rand(1) >= self.p:
            return img, mask
        img, mask = T.functional.horizontal_flip(img), T.functional.horizontal_flip(mask)
        data = torch.as_tensor(mask).clone()
        data[1] = self.lookup[data[1].long()]
        return img, tv_tensors.Mask(data)


def build_transforms(tier: str, img_size: int, mean: Sequence[float], std: Sequence[float], train: bool,
                     part_flip_order: Optional[Sequence[int]] = None):
    size = (img_size, img_size)
    flip = PartAwareHorizontalFlip(part_flip_order) if part_flip_order is not None else T.RandomHorizontalFlip(0.5)
    to_float = T.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32, "others": None},
                         scale=True)
    norm = T.Normalize(mean=list(mean), std=list(std))
    if not train or tier == "none":
        return T.Compose([T.Resize(size, antialias=True), to_float, norm])
    if tier == "basic":
        ops = [T.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True), flip,
               to_float, norm]
    elif tier == "strong":
        ops = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), flip,
               T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
               to_float, norm, T.RandomErasing(p=0.25)]
    elif tier == "auto":
        ops = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), flip,
               MaskAwareTrivialAugmentWide(), to_float, norm, T.RandomErasing(p=0.25)]
    else:
        raise ValueError(f"unknown augmentation tier {tier!r}")
    return T.Compose(ops)


# ----------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------
class CUBDataset(Dataset):
    """
    Reads train/val/test.csv (columns: image_id, filepath, label, class_name).
    Returns (image[3,H,W] normalised, bird_mask[1,H,W] in [0,1], label, image_id).
    With `parts`, also returns part points: (P, 3) with x, y in [0, 1] and visible, after all augmentation. The
    transform must then be built with part_flip_order=parts.flip_order.

    The csv paths look like 'data/CUB_200_2011/images/...'; only the part after 'CUB_200_2011/' is used and it is
    joined to `cub_root`, so it does not matter where the dataset folder lives.

    R4 (training only): with probability `box_crop_prob` the image is first cropped to the bird box expanded by a
    random margin, so the model also sees the bird (and its parts) at a larger scale. Never used at val/test.
    """

    def __init__(self, csv_path, transform, bboxes, cub_root: os.PathLike, train: bool = False,
                 box_crop_prob: float = 0.0, box_crop_margin: Tuple[float, float] = (0.1, 0.4),
                 parts: Optional[PartAnnotations] = None):
        self.parts = parts
        df = pd.read_csv(csv_path)
        self.records = df.to_dict("records")
        self.transform, self.bboxes, self.cub_root = transform, bboxes, Path(cub_root)
        self.train, self.box_crop_prob, self.box_crop_margin = train, box_crop_prob, box_crop_margin
        self.class_names = df.drop_duplicates("label").sort_values("label")["class_name"].tolist()

    def __len__(self):
        return len(self.records)

    def image_path(self, i: int) -> Path:
        fp = str(self.records[i]["filepath"]).replace("\\", "/")
        key = "CUB_200_2011/"
        rel = fp.split(key, 1)[1] if key in fp else fp
        return self.cub_root / rel

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(self.image_path(i)).convert("RGB")
        w, h = img.size
        box = self.bboxes.get(int(r["image_id"])) if self.bboxes is not None else None
        offset = (0, 0)
        if box is not None:
            x, y, bw, bh = box
            if self.train and self.box_crop_prob > 0 and random.random() < self.box_crop_prob:
                m = random.uniform(*self.box_crop_margin)
                cx0, cy0 = max(0, int(x - m * bw)), max(0, int(y - m * bh))
                cx1, cy1 = min(w, int(x + bw + m * bw)), min(h, int(y + bh + m * bh))
                if cx1 - cx0 > 8 and cy1 - cy0 > 8:
                    img = img.crop((cx0, cy0, cx1, cy1))
                    x, y = x - cx0, y - cy0
                    offset = (cx0, cy0)
                    w, h = img.size
            mask = torch.zeros(1, h, w, dtype=torch.uint8)
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            mask[:, y0:y1, x0:x1] = 1
        else:
            mask = torch.zeros(1, h, w, dtype=torch.uint8)
        img_t = tv_tensors.Image(T.functional.pil_to_tensor(img))
        if self.parts is None:
            img_t, mask_t = self.transform(img_t, tv_tensors.Mask(mask))
            return img_t, torch.as_tensor(mask_t).float().clamp_(0, 1), int(r["label"]), int(r["image_id"])
        num_parts = len(self.parts.names)
        points = self.parts.points.get(int(r["image_id"]), np.zeros((num_parts, 3), dtype=np.float32))
        stacked = torch.cat([mask, paint_part_map(points, offset, w, h)])
        img_t, mask_t = self.transform(img_t, tv_tensors.Mask(stacked))
        mask_t = torch.as_tensor(mask_t).float()
        points = hide_erased_points(part_points_from_map(mask_t[1], num_parts), img_t)
        return img_t, mask_t[:1].clamp_(0, 1), int(r["label"]), int(r["image_id"]), points


class MixupCutmix:
    """Batch-level mixup / CutMix (probability p_cutmix of choosing CutMix over mixup). The bird-box masks are
    mixed in exactly the same way so the bbox attention loss stays consistent. Returns soft targets that already
    include label smoothing."""

    def __init__(self, num_classes: int, mixup_alpha=0.2, cutmix_alpha=1.0, p_cutmix=0.5,
                 label_smoothing=0.1, prob=1.0):
        self.nc, self.ma, self.ca, self.pc, self.ls, self.prob = (num_classes, mixup_alpha, cutmix_alpha,
                                                                   p_cutmix, label_smoothing, prob)

    def __call__(self, x, y, mask, points=None):
        """With part points, info also holds "points" and "point_weight". CutMix hides parts under the pasted patch.
        Mixup uses the parts of whichever image dominates, weighted by its share of the blend."""
        t1 = smooth_one_hot(y, self.nc, self.ls)
        info = {"type": "none", "lam": 1.0}
        if points is not None:
            info.update(points=points, point_weight=x.new_ones(x.shape[0]))
        if np.random.rand() > self.prob:
            return x, t1, mask, info
        x2, y2, m2 = x.flip(0), y.flip(0), mask.flip(0)
        if np.random.rand() < self.pc:                                   # CutMix
            lam = np.random.beta(self.ca, self.ca)
            H, W = x.shape[-2:]
            cut = math.sqrt(1.0 - lam)
            ch, cw = int(H * cut), int(W * cut)
            cy, cx = np.random.randint(H), np.random.randint(W)
            y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
            x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
            x, mask = x.clone(), mask.clone()
            x[..., y0:y1, x0:x1] = x2[..., y0:y1, x0:x1]
            mask[..., y0:y1, x0:x1] = m2[..., y0:y1, x0:x1]
            lam = 1.0 - ((y1 - y0) * (x1 - x0) / (H * W))
            info.update(type="cutmix", lam=lam)
            if points is not None:
                px, py = points[..., 0] * W, points[..., 1] * H
                covered = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
                info["points"] = points * (~covered).unsqueeze(-1).to(points.dtype)
        else:                                                            # Mixup
            lam = float(np.random.beta(self.ma, self.ma))
            x = lam * x + (1 - lam) * x2
            mask = lam * mask + (1 - lam) * m2
            info.update(type="mixup", lam=lam)
            if points is not None:
                info["points"] = points if lam >= 0.5 else points.flip(0)
                info["point_weight"] = x.new_full((x.shape[0],), max(lam, 1 - lam))
        t2 = smooth_one_hot(y2, self.nc, self.ls)
        return x, lam * t1 + (1 - lam) * t2, mask, info


# ----------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------
def build_loaders(info: DataInfo, tc: TrainConfig, seed: int, num_workers: int, pin_memory: bool):
    """Returns (datasets, loaders) with keys train / val / test / train_clean.
    train_clean = a fixed subset of the training images WITHOUT augmentation, used to measure real train accuracy."""
    parts = None
    if tc.use_part_guidance:
        parts = load_part_annotations(info.cub_root)
        if tc.part_fraction < 1.0:
            kept = select_annotated_images(pd.read_csv(info.train_csv).to_dict("records"), tc.part_fraction, seed)
            parts.points = {image_id: parts.points[image_id] for image_id in kept if image_id in parts.points}
    tr_tf = build_transforms(tc.aug_tier, info.img_size, info.mean, info.std, train=True,
                             part_flip_order=parts.flip_order if parts else None)
    ev_tf = build_transforms("none", info.img_size, info.mean, info.std, train=False)
    crop_p = tc.box_crop_prob if tc.use_box_crop else 0.0
    ds = {"train": CUBDataset(info.train_csv, tr_tf, info.bboxes, info.cub_root, train=True,
                              box_crop_prob=crop_p, box_crop_margin=tc.box_crop_margin, parts=parts),
          "val": CUBDataset(info.val_csv, ev_tf, info.bboxes, info.cub_root),
          "test": CUBDataset(info.test_csv, ev_tf, info.bboxes, info.cub_root)}
    clean_full = CUBDataset(info.train_csv, ev_tf, info.bboxes, info.cub_root)
    n_clean = min(tc.clean_eval_size, len(clean_full))
    ds["train_clean"] = Subset(clean_full, np.linspace(0, len(clean_full) - 1, n_clean).astype(int).tolist())
    ds["train_clean"].class_names = clean_full.class_names

    if tc.track_part_localization:
        ds["val_parts"] = CUBDataset(info.val_csv, ev_tf, info.bboxes, info.cub_root,
                                     parts=load_part_annotations(info.cub_root))
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker if num_workers else None)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    bs = tc.batch_size
    dl = {"train": DataLoader(ds["train"], bs, shuffle=True, drop_last=True, generator=g, **kw),
          "val": DataLoader(ds["val"], bs, shuffle=False, **kw),
          "test": DataLoader(ds["test"], bs, shuffle=False, **kw),
          "train_clean": DataLoader(ds["train_clean"], bs, shuffle=False, **kw)}
    if "val_parts" in ds:
        dl["val_parts"] = DataLoader(ds["val_parts"], bs, shuffle=False, **kw)
    return ds, dl


# ----------------------------------------------------------------------------------------------
# Synthetic CUB (smoke tests only): same folder layout as the real project
# ----------------------------------------------------------------------------------------------
def make_synthetic_dataset(out_root, n_classes=8, n_train=12, n_val=2, n_test=4, size=(320, 280), seed=0,
                           img_size=224, batch_size=16) -> PathsConfig:
    """Creates <out_root>/CUB_200_2011/{images,bounding_boxes.txt} and <out_root>/DataPreprocessing_Results/
    {train,val,test}.csv + json files, mirroring the real layout. Each class is a coloured ellipse ('bird')."""
    from PIL import ImageDraw
    rng = np.random.RandomState(seed)
    out_root = Path(out_root)
    cub, prep = out_root / "CUB_200_2011", out_root / "DataPreprocessing_Results"
    prep.mkdir(parents=True, exist_ok=True)
    palette = rng.randint(40, 255, size=(n_classes, 3))
    rows, boxes, image_id = {"train": [], "val": [], "test": []}, [], 1
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for c in range(n_classes):
            cname = f"{c + 1:03d}.Class_{c}"
            (cub / "images" / cname).mkdir(parents=True, exist_ok=True)
            for j in range(n):
                W, H = size
                bg = rng.randint(0, 255, size=(H, W, 3)).astype(np.uint8) // 3
                img = Image.fromarray(bg)
                bw, bh = rng.randint(W // 4, W // 2), rng.randint(H // 4, H // 2)
                bx, by = rng.randint(0, W - bw), rng.randint(0, H - bh)
                ImageDraw.Draw(img).ellipse([bx, by, bx + bw, by + bh], fill=tuple(int(v) for v in palette[c]))
                rel = f"images/{cname}/{split}_{j}.jpg"
                img.save(cub / rel, quality=90)
                rows[split].append({"image_id": image_id, "filepath": f"data/CUB_200_2011/{rel}", "label": c,
                                    "class_name": f"Class {c}"})
                boxes.append(f"{image_id} {bx} {by} {bw} {bh}")
                image_id += 1
    (cub / "bounding_boxes.txt").write_text("\n".join(boxes))
    for split in rows:
        pd.DataFrame(rows[split]).sample(frac=1, random_state=seed).to_csv(prep / f"{split}.csv", index=False)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    (prep / "norm_stats.json").write_text(json.dumps({"mean": mean, "std": std}))
    (prep / "preprocess_config.json").write_text(json.dumps({
        "seed": seed, "img_size": img_size, "batch_size": batch_size, "mean": mean, "std": std,
        "splits": {"train": n_train * n_classes, "val": n_val * n_classes, "test": n_test * n_classes},
        "mixup_cutmix": {"mixup_alpha": 0.2, "cutmix_alpha": 1.0, "p_cutmix": 0.5}}))
    return PathsConfig(root=str(out_root))
