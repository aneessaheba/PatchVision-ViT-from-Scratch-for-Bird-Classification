"""Experiment definitions for the CNN study.

Both CNNs use the SAME training recipe as the ViT 'full' run (data, augmentation, mixup/CutMix, label smoothing,
AdamW, warm-up + cosine, EMA, early stopping on validation).  The ViT-only parts are switched off because a CNN
has no attention maps and no token grid:  box-attention loss, flip-consistency loss, box-guided crop.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional, Tuple

from . import _vit  # noqa: F401  (adds ViT_PatchVisionX to sys.path)
from patchvision.config import PathsConfig, TrainConfig  # noqa: E402,F401


@dataclass
class CNNConfig:
    """Compact ResNet (BasicBlock). widths = channels of the 4 stages, blocks = BasicBlocks per stage."""
    img_size: int = 224
    num_classes: int = 200
    widths: Tuple[int, ...] = (39, 78, 156, 312)
    blocks: Tuple[int, ...] = (2, 2, 2, 2)
    zero_init_residual: bool = True      # last BatchNorm of every block starts at 0 (standard trick for stable training)
    arch: str = "resnet"                 # resnet | plain_vit | pvtv2_b0 (see cnn/transformers.py)
    drop_path: float = 0.1               # stochastic depth for plain_vit and pvtv2_b0


@dataclass
class CNNExperiment:
    name: str = "cnn_small"
    seed: int = 42
    model: CNNConfig = field(default_factory=CNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ViT-only components switched off; EMA stays on (it is generic, and 'full' uses it).
_CNN_TRAIN = {"use_bbox_loss": False, "use_consistency": False, "use_box_crop": False, "use_ema": True}

CNN_SUITE: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ~4.22 M parameters, matched to PatchVision-X (4.27 M): the fair CNN-vs-ViT comparison
    "cnn_small": {"model": {"widths": (39, 78, 156, 312)}, "train": dict(_CNN_TRAIN)},
    # ~11.28 M parameters = ResNet-18 layout trained from scratch: "what does more CNN capacity buy?"
    "cnn_large": {"model": {"widths": (64, 128, 256, 512)}, "train": dict(_CNN_TRAIN)},
    # outside reference models under the same recipe, with no annotations
    "plain_vit": {"model": {"arch": "plain_vit"}, "train": dict(_CNN_TRAIN)},
    "pvtv2_b0": {"model": {"arch": "pvtv2_b0"}, "train": dict(_CNN_TRAIN)},
}


def _build(cls, d: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})


def get_experiment(name: str, seed: int = 42, model_overrides: Optional[Dict[str, Any]] = None,
                   train_overrides: Optional[Dict[str, Any]] = None, run_name: Optional[str] = None) -> CNNExperiment:
    if name not in CNN_SUITE:
        raise KeyError(f"unknown experiment {name!r}. Available: {list(CNN_SUITE)}")
    spec = copy.deepcopy(CNN_SUITE[name])
    m = {**spec["model"], **(model_overrides or {})}
    t = {**spec["train"], **(train_overrides or {})}
    return CNNExperiment(name=run_name or name, seed=seed, model=_build(CNNConfig, m), train=_build(TrainConfig, t))
