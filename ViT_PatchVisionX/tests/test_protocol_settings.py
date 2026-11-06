"""Checks settings required by docs/PROTOCOL.md for both the ViT and the CNN.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ViT_PatchVisionX"))
sys.path.insert(0, str(REPO / "CNN_Baseline"))

from cnn.config import CNN_SUITE, get_experiment as get_cnn_experiment
from cnn.engine import evaluate as evaluate_cnn
from patchvision.config import ablation_suite, combined_trials, architecture_trials, get_experiment, get_trial
from patchvision.engine import evaluate as evaluate_vit

NO_AMP = (False, torch.bfloat16)


def test_early_stopping_is_off_for_every_model_we_will_run():
    configs = [get_experiment(name) for name in ablation_suite()]
    configs += [get_trial(name) for name in {**combined_trials(), **architecture_trials()}]
    configs += [get_cnn_experiment(name) for name in CNN_SUITE]
    assert all(cfg.train.early_stop_patience == 0 for cfg in configs)


def tied_batches():
    logits = torch.tensor([[2.0, 2.0, 0.0], [0.0, 3.0, 3.0], [5.0, 1.0, 5.0], [1.0, 1.0, 1.0]])
    labels = torch.tensor([0, 2, 2, 1])
    return logits, labels


class Fixed(torch.nn.Module):
    def __init__(self, logits, with_aux):
        super().__init__()
        self.logits, self.with_aux = logits, with_aux

    def forward(self, x):
        if not self.with_aux:
            return self.logits
        attention = [torch.full((x.shape[0], 4), 0.25)]
        return self.logits, {"attn_maps": attention, "gate_mean": torch.tensor(0.5), "gate_sparsity": torch.tensor(0.0)}


@pytest.mark.parametrize("name", ["vit", "cnn"])
def test_saved_predictions_match_reported_accuracy_when_scores_tie(name):
    logits, labels = tied_batches()
    n = len(labels)
    x, mask, ids = torch.zeros(n, 3, 8, 8), torch.ones(n, 1, 8, 8), torch.arange(n)
    loader = [(x, mask, labels, ids)]
    if name == "vit":
        res, _, preds = evaluate_vit(Fixed(logits, True), loader, torch.device("cpu"), 2, 3, NO_AMP, collect=True)
    else:
        res, _, preds = evaluate_cnn(Fixed(logits, False), loader, torch.device("cpu"), 3, NO_AMP, collect=True)
    assert 100 * (preds["pred"] == preds["label"]).mean() == pytest.approx(res["acc1"])
    assert preds["pred"].tolist() == logits.topk(1, dim=1).indices.squeeze(1).tolist()
