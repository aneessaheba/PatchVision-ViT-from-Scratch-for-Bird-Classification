"""Checks CUB part points on real images. Does not train a model and does not read the test set.

Run from inside ViT_PatchVisionX/:  python tools/check_part_alignment.py

Prints how many annotated parts remain after painting and resizing, for the training split, and saves
results/_checks/part_alignment.png with parts drawn on augmented images. On that picture, each coloured label should
sit on the named part, and "left" and "right" should always mean the bird's own left and right.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from patchvision.config import PathsConfig
from patchvision.data import CUBDataset, build_transforms, load_bboxes, load_data_info, load_part_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(HERE.parent))
    parser.add_argument("--out", default=str(HERE / "results" / "_checks" / "part_alignment.png"))
    parser.add_argument("--tier", default="auto", choices=["basic", "strong", "auto"])
    args = parser.parse_args()

    info = load_data_info(PathsConfig(root=args.root))
    parts = load_part_annotations(info.cub_root)
    boxes = load_bboxes(info.cub_root / "bounding_boxes.txt")
    print("Parts:", parts.names)
    print("Mirror partners:", {parts.names[i]: parts.names[j] for i, j in enumerate(parts.flip_order) if i != j})

    plain = build_transforms("none", info.img_size, info.mean, info.std, train=False)
    ds = CUBDataset(info.train_csv, plain, boxes, info.cub_root, parts=parts)
    annotated, kept = torch.zeros(len(parts.names)), torch.zeros(len(parts.names))
    for i in range(len(ds)):
        points = ds[i][4]
        source = parts.points[int(ds.records[i]["image_id"])]
        annotated += torch.as_tensor(source[:, 2] > 0).float()
        kept += (points[:, 2] > 0).float()
    print("\nAnnotated parts that survive painting and resizing:")
    for name, a, k in zip(parts.names, annotated.tolist(), kept.tolist()):
        print(f"  {name:12s} {int(k):6d} of {int(a):6d} ({100 * k / max(a, 1):5.1f}%)")

    augmented = build_transforms(args.tier, info.img_size, info.mean, info.std, train=True,
                                 part_flip_order=parts.flip_order)
    ds = CUBDataset(info.train_csv, augmented, boxes, info.cub_root, train=True, box_crop_prob=0.3, parts=parts)
    mean, std = torch.tensor(info.mean)[:, None, None], torch.tensor(info.std)[:, None, None]
    picks = torch.linspace(0, len(ds) - 1, 12).long().tolist()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, index in zip(axes.flat, picks):
        image, _, _, _, points = ds[index]
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        for i, (x, y, visible) in enumerate(points.tolist()):
            if visible:
                ax.text(x * info.img_size, y * info.img_size, parts.names[i], color="yellow", fontsize=7,
                        ha="center", va="center", bbox=dict(facecolor="black", alpha=0.6, pad=0.5))
        ax.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
