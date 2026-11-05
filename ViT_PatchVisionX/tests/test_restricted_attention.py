"""Checks that restrict_attention limits each head group to its own scale and is off by default.

Run from inside ViT_PatchVisionX/:  python -m pytest tests
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchvision.config import ModelConfig, get_trial
from patchvision.models import MultiScaleGraphAttention, PatchVisionX, grid_scale_masks

GRID = 14
SCALES = (1, 2, 4, 0)
HEADS_PER_GROUP = 2
HEAD_DIM = 32


def build(restrict, use_gate=False):
    torch.manual_seed(0)
    attn = MultiScaleGraphAttention(256, 8, GRID * GRID, SCALES, 0.0, 0.0, use_gate=use_gate,
                                    restrict_attention=restrict).eval()
    captured = {}
    attn.attn_drop.register_forward_hook(lambda _m, _i, out: captured.__setitem__("S", out.detach()))
    attn.proj.register_forward_pre_hook(lambda _m, inputs: captured.__setitem__("out", inputs[0].detach()))
    return attn, captured


def tokens():
    torch.manual_seed(1)
    return torch.randn(1, GRID * GRID, 256)


def test_default_attention_spans_all_tokens():
    attn, cap = build(restrict=False)
    masks = grid_scale_masks(GRID, SCALES)
    with torch.no_grad():
        attn(tokens(), masks)
    outside = cap["S"][0, 0] * (1 - masks[0])
    assert outside.sum() > 0


def test_restricted_attention_stays_inside_each_scale():
    attn, cap = build(restrict=True)
    masks = grid_scale_masks(GRID, SCALES)
    with torch.no_grad():
        attn(tokens(), masks)
    for g in range(len(SCALES)):
        heads = cap["S"][0, HEADS_PER_GROUP * g:HEADS_PER_GROUP * (g + 1)]
        assert (heads * (1 - masks[g])).abs().max() == 0
        assert torch.allclose(heads.sum(-1), torch.ones_like(heads.sum(-1)))


def test_far_token_cannot_reach_a_restricted_local_head():
    attn, cap = build(restrict=True)
    masks = grid_scale_masks(GRID, SCALES)
    x = tokens()
    with torch.no_grad():
        attn(x, masks)
        before = cap["out"].clone()
        far = x.clone()
        far[0, 0] += 1.0                     # token (0, 0) is far outside token 105's 3 x 3 reach
        attn(far, masks)
    slice_ = 2 * HEAD_DIM
    assert (before[0, 105, :slice_] - cap["out"][0, 105, :slice_]).abs().max() == 0
    assert (before[0, 105, -slice_:] - cap["out"][0, 105, -slice_:]).abs().max() > 0     # the global group does change


def test_phi_cannot_mix_head_groups_when_restricted():
    attn, _ = build(restrict=True)
    expected = torch.zeros(8, 8)
    for g in range(4):
        expected[2 * g:2 * g + 2, 2 * g:2 * g + 2] = 1
    assert torch.equal(attn.phi_mask, expected)


def test_flag_is_off_by_default_and_adds_no_saved_state():
    assert ModelConfig().restrict_attention is False
    plain, flagged = PatchVisionX(ModelConfig()), PatchVisionX(ModelConfig(restrict_attention=True))
    assert plain.state_dict().keys() == flagged.state_dict().keys()


def test_full_restricted_model_trains_through_pooling():
    cfg = get_trial("ovf5_restricted").model
    assert cfg.restrict_attention is True and cfg.drop_path == 0.3
    torch.manual_seed(0)
    model = PatchVisionX(cfg).train()
    logits, aux = model(torch.randn(2, 3, 224, 224))
    assert torch.isfinite(logits).all()
    (logits.sum() + sum(a.sum() for a in aux["attn_maps"])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
