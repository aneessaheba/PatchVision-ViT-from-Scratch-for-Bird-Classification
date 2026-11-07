"""
PatchVision-X  --  the model architecture (ONE model; components are switchable for ablations).

  conv tokenizer (GvT) -> [ LFE + multi-scale sparse graph-attention + MLP ] x n_pre
        -> graph pooling (196 -> 49 tokens) -> [ multi-scale sparse graph-attention + MLP ] x n_post
        -> LayerNorm -> global average pooling -> linear classifier

What comes from where
---------------------
GvT (Sci. Rep. 15, 2025, graph-based ViT with sparsity)
    * conv tokenizer with a group convolution (effective 16x16 patch -> 14x14 = 196 tokens)
    * attention on a token GRAPH: A = (A_I + I) * E  (8-neighbour grid prior x feature similarity),
      queries/keys by graph convolution, sparse talking-heads shrinkage gate, graph pooling.
MSCViT (arXiv 2501.06040)
    * Local Feature Extraction (LFE): depthwise-conv block that REPLACES the absolute position embedding and
      gives the transformer a local, translation-aware prior (key for tiny datasets).
    * Multi-scale attention: different heads work at different spatial scales (fine heads / coarse heads).
OURS (this project)
    * Multi-scale GRAPH attention: MSCViT's multi-scale idea is realised without breaking GvT's graph
      formulation. The heads are split into groups; group g restricts the graph to the s_g-hop
      neighbourhood of every token (3x3, 5x5, 9x9, ... and "0" = global). Fine heads learn local
      texture/part details, coarse heads learn body-level context. After pooling the hop powers are
      taken on the learned (soft) adjacency instead of the grid.
    * Every block exposes the attention each token RECEIVES, which feeds the bounding-box attention loss and
      the flip-consistency loss in losses.py (training-time regularisers, not part of inference).

INTERPRETATION notes for details the GvT text leaves ambiguous are marked as such below.
Nothing here uses pretrained weights.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

TOKENIZER_STRIDE = 16   # conv 8x8/8 followed by group-conv 5x5/2  ->  224 / 16 = 14 tokens per side


# ----------------------------------------------------------------------------------------------
# Generic building blocks
# ----------------------------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float, drop: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ----------------------------------------------------------------------------------------------
# Graph utilities
# ----------------------------------------------------------------------------------------------
def grid_adjacency(g: int) -> torch.Tensor:
    """A_I: 8-directional neighbour relations on a g x g token grid (no self loops)."""
    idx = torch.arange(g * g).view(g, g)
    A = torch.zeros(g * g, g * g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys = torch.arange(max(0, -dy), min(g, g - dy))
            xs = torch.arange(max(0, -dx), min(g, g - dx))
            src = idx[ys][:, xs].reshape(-1)
            dst = idx[ys + dy][:, xs + dx].reshape(-1)
            A[src, dst] = 1.0
    return A


def grid_scale_masks(g: int, scales: Sequence[int]) -> torch.Tensor:
    """(G, n, n) binary masks, one per head group. Scale s>0: tokens within s hops (Chebyshev radius s) of
    each other on the grid, self included. Scale 0: global (all ones)."""
    n = g * g
    A = grid_adjacency(g) + torch.eye(n)
    masks = []
    for s in scales:
        if s == 0:
            masks.append(torch.ones(n, n))
            continue
        P = A.clone()
        for _ in range(s - 1):
            P = ((P @ A) > 0).float()
        masks.append(P)
    return torch.stack(masks)


def soft_scale_bases(A: torch.Tensor, scales: Sequence[int]) -> torch.Tensor:
    """Pooled (soft) graph: scale s = s-th power of the row-normalised adjacency; 0 = global.  (B,n,n)->(B,G,n,n)"""
    A = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
    out = []
    for s in scales:
        if s == 0:
            out.append(torch.ones_like(A))
            continue
        P = A
        for _ in range(s - 1):
            P = P @ A
        out.append(P)
    return torch.stack(out, dim=1)


def sym_norm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} with D = diag(row sums). Works on (..., n, n)."""
    deg = A.sum(-1).clamp_min(eps)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2)


# ----------------------------------------------------------------------------------------------
# MSCViT: Local Feature Extraction
# ----------------------------------------------------------------------------------------------
class LocalFeatureExtraction(nn.Module):
    """Local(X) = DWConv2( GELU( BN( DWConv1(X) ) ) + X )   with DWConv1 = 3x3 depthwise, DWConv2 = 1x1 depthwise.
    Used as a residual branch on the token grid; it injects position / locality implicitly, so no absolute
    position embedding is needed."""

    def __init__(self, dim: int, grid: int):
        super().__init__()
        self.grid = grid
        self.dw1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.dw2 = nn.Conv2d(dim, dim, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, n, d) -> (B, n, d)
        B, n, d = x.shape
        t = x.transpose(1, 2).reshape(B, d, self.grid, self.grid)
        t = self.dw2(self.act(self.bn(self.dw1(t))) + t)
        return t.flatten(2).transpose(1, 2)


# ----------------------------------------------------------------------------------------------
# Multi-scale sparse graph attention (GvT attention + head groups at different graph scales)
# ----------------------------------------------------------------------------------------------
class MultiScaleGraphAttention(nn.Module):
    """
    1. Graph: neighbour mask (per head group, see grid_scale_masks) x feature-similarity E
              A_g = base_g * E          (INTERPRETATION: cosine SIMILARITY in a learnable-temperature softmax;
                                          the paper's text mixes "distance" and "similarity".)
    2. Queries / keys by graph convolution:  Q = Â_g (X Wq),  K = Â_g (X Wk),  Â = D^-1/2 A D^-1/2,
       separate (d/h x d/h) weights per head; heads of group g use Â_g.
    3. S = softmax(Q K^T / sqrt(d_h)).
    4. Sparse talking-heads (optional): token-wise shrinkage gate Z from the heads' descriptors,
       S_hat = diag(Z) S diag(Z); heads exchange information via a learnable h x h matrix Phi: R = Phi . S_hat.
    5. Values by graph convolution guided by R:  out = D_R^-1/2 R D_R^-1/2 (X Wv) + X.
    Attention is computed in fp32 even under mixed precision.

    By default the scale masks only shape the graph used for queries and keys: S and R still span all tokens, so
    every head can receive from anywhere. restrict_attention=True limits each head's attention to its own scale:
    the similarity graph is normalised inside the mask, S is masked (grid) or weighted (soft pooled graph), and
    Phi only mixes heads of the same group. The sparse gate, when on, still pools statistics over all tokens.
    """

    def __init__(self, dim: int, heads: int, n_tokens: int, scales: Sequence[int], attn_drop: float,
                 proj_drop: float, use_gate: bool = True, adj_mode: str = "hadamard",
                 restrict_attention: bool = False):
        super().__init__()
        assert dim % heads == 0
        G = len(scales)
        assert heads % G == 0, f"heads ({heads}) must be divisible by the number of scale groups ({G})"
        self.dim, self.heads, self.dh, self.n = dim, heads, dim // heads, n_tokens
        self.G, self.hg = G, heads // G
        self.use_gate, self.adj_mode = use_gate, adj_mode
        self.restrict_attention = restrict_attention
        if restrict_attention:
            group = torch.arange(heads) // self.hg
            self.register_buffer("phi_mask", (group[:, None] == group[None, :]).float(), persistent=False)
        dh = self.dh
        self.wq = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        self.wk = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        self.wv = nn.Parameter(torch.randn(heads, dh, dh) * dh ** -0.5)
        if use_gate:
            self.w_h = nn.Parameter(torch.randn(heads, 2 * dh) * (2 * dh) ** -0.5)   # token descriptor
            self.w_c = nn.Parameter(torch.randn(heads, n_tokens) * 0.5)             # gate scale (h x n)
            self.gate_slope = nn.Parameter(torch.ones(1, heads, 1))                 # trainable, init 1
        self.phi = nn.Parameter(torch.eye(heads))                                    # head mixing, init identity
        self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))                    # similarity temperature
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, bases: torch.Tensor):
        """
        x     : (B, n, d) layer-normalised tokens
        bases : (G, n, n) grid masks, or (B, G, n, n) soft bases after pooling
        returns out (B, n, d), A (B, n, n) mean row-normalised adjacency over groups, aux dict
        """
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, n, d = x.shape
            h, dh, G, hg = self.heads, self.dh, self.G, self.hg
            xh = x.view(B, n, h, dh).permute(0, 2, 1, 3)                              # (B,h,n,dh)

            # --- 1. graph: scale mask x feature similarity
            xn = F.normalize(x, dim=-1)
            E = torch.softmax((xn @ xn.transpose(1, 2)) / self.log_tau.exp().clamp_min(1e-3), dim=-1)  # (B,n,n)
            bases = bases.float()
            if bases.ndim == 3:
                bases = bases.unsqueeze(0)                                           # (1,G,n,n)
            Eg = E.unsqueeze(1)
            A = bases * Eg if self.adj_mode == "hadamard" else bases + Eg           # (B,G,n,n)
            if self.restrict_attention:                                              # cancels E's all-token softmax
                A = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
            A_hat = sym_norm(A).unsqueeze(2)                                         # (B,G,1,n,n)

            def graph_conv(w):                                                       # heads of a group share Â_g
                z = torch.einsum("bhnd,hde->bhne", xh, w).reshape(B, G, hg, n, dh)
                return (A_hat @ z).reshape(B, h, n, dh)

            # --- 2./3. graph-conv queries and keys, attention
            q, k = graph_conv(self.wq), graph_conv(self.wk)
            scores = (q @ k.transpose(-1, -2)) * dh ** -0.5                          # (B,h,n,n)
            if self.restrict_attention:
                head_bases = bases.repeat_interleave(hg, dim=1)                      # (1 or B,h,n,n)
                scores = (scores + head_bases.clamp_min(1e-12).log()).masked_fill(head_bases <= 0, float("-inf"))
            S = torch.softmax(scores, dim=-1)
            S = self.attn_drop(S)

            # --- 4. sparse talking-heads
            if self.use_gate:
                f = torch.einsum("bhnc,hc->bhn", torch.cat([q, k], dim=-1), self.w_h)
                # standardise each head's descriptor over tokens so the pooled head-correlation is O(1);
                # otherwise the gate input is ~1e-3, the gate stays frozen at 0.5 and no sparsity can emerge.
                f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + 1e-5)
                C = (f @ f.transpose(1, 2)) / n                                      # bilinear pooling (B,h,h)
                Z = torch.sigmoid((C @ self.w_c) * self.gate_slope)                  # (B,h,n)
                S_hat = Z.unsqueeze(-1) * S * Z.unsqueeze(-2)
                gate_mean, gate_sp = Z.detach().mean(), (Z.detach() < 0.1).float().mean()
            else:
                S_hat = S
                gate_mean, gate_sp = x.new_ones(()), x.new_zeros(())
            phi = self.phi * self.phi_mask if self.restrict_attention else self.phi
            R = torch.einsum("gh,bhnm->bgnm", phi, S_hat).clamp_min(0.0)            # (B,h,n,n)

            # --- 5. graph-conv values guided by R, residual of the head slice
            v = torch.einsum("bhnd,hde->bhne", xh, self.wv)
            out = (sym_norm(R) @ v + xh).permute(0, 2, 1, 3).reshape(B, n, d)

            # --- aux: attention RECEIVED by each token (sums to 1) -> used by the bbox / consistency losses
            Rn = R / R.sum(-1, keepdim=True).clamp_min(1e-6)
            received = Rn.mean(dim=(1, 2))                                           # (B,n)
            A_rs = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
            aux = {"received": received, "gate_mean": gate_mean, "gate_sparsity": gate_sp}
        out = self.proj_drop(self.proj(out.to(in_dtype)))
        return out, A_rs.mean(1), aux


class PXBlock(nn.Module):
    def __init__(self, dim, heads, n_tokens, scales, mlp_ratio, drop, attn_drop, drop_path, use_gate, adj_mode,
                 lfe_grid: Optional[int], restrict_attention: bool = False):
        super().__init__()
        self.lfe = LocalFeatureExtraction(dim, lfe_grid) if lfe_grid else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiScaleGraphAttention(dim, heads, n_tokens, scales, attn_drop, drop, use_gate, adj_mode,
                                             restrict_attention)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, bases):
        if self.lfe is not None:
            x = x + self.lfe(x)
        a, A, aux = self.attn(self.norm1(x), bases)
        x = x + self.drop_path(a)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, A, aux


class GraphPool(nn.Module):
    """Graph pooling n -> t tokens.
    INTERPRETATION: the paper's dimensions (X' = U X^T X, C = U X^T) do not match, so we implement the stated
    idea: t learnable queries U attend over the n tokens, C = softmax(U K^T) in R^{t x n};
        X' = C X          A' = C A C^T   (+ self loops)"""

    def __init__(self, dim: int, n_out: int):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_out, dim) * dim ** -0.5)
        self.key = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dim, self.n_out = dim, n_out

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            k = self.key(x32)
            C = torch.softmax((self.U.float() @ k.transpose(1, 2)) * self.dim ** -0.5, dim=-1)   # (B,t,n)
            x_new = C @ x32
            A_rn = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
            A_new = C @ A_rn @ C.transpose(1, 2)
            A_new = A_new / A_new.sum(-1, keepdim=True).clamp_min(1e-6)
            A_new = A_new + torch.eye(self.n_out, device=A_new.device).unsqueeze(0)
        return self.norm(x_new.to(in_dtype)), A_new, C


class ConvTokenizer(nn.Module):
    """224x224 -> (conv 8x8/8) 28x28 -> (group-conv 5x5/2) 14x14 = 196 tokens (an effective 16x16 patch)."""

    def __init__(self, in_ch: int, dim: int, groups: int):
        super().__init__()
        mid = dim // 2
        assert mid % groups == 0 and dim % groups == 0, "embed_dim/2 and embed_dim must be divisible by tok_groups"
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=8, stride=8)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, dim, kernel_size=5, stride=2, padding=2, groups=groups)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x.flatten(2).transpose(1, 2)                                          # (B, n, d)


# ----------------------------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------------------------
class PatchVisionX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.img_size % TOKENIZER_STRIDE == 0, "img_size must be a multiple of 16"
        self.cfg = cfg
        g = cfg.img_size // TOKENIZER_STRIDE
        self.grid, n0 = g, g * g
        self.scales = cfg.effective_scales()
        pre = cfg.resolved_pool_after()
        self.n_pre = pre if 0 < pre < cfg.depth else cfg.depth          # depth < 2 -> no pooling
        n_post = max(1, n0 // cfg.pool_ratio)

        self.tokenizer = ConvTokenizer(3, cfg.embed_dim, cfg.tok_groups)
        if (not cfg.use_lfe) or cfg.keep_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, n0, cfg.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)
        self.register_buffer("base_adj", grid_scale_masks(g, self.scales), persistent=False)

        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            pre_pool = i < self.n_pre
            self.blocks.append(PXBlock(
                cfg.embed_dim, cfg.heads, n0 if pre_pool else n_post, self.scales, cfg.mlp_ratio, cfg.drop,
                cfg.attn_drop, dpr[i], cfg.use_sparse_gate, cfg.adj_mode,
                lfe_grid=g if (cfg.use_lfe and pre_pool) else None, restrict_attention=cfg.restrict_attention))
        self.pool = GraphPool(cfg.embed_dim, n_post) if self.n_pre < cfg.depth else None
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.tokenizer(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        bases = self.base_adj
        attn_maps: List[torch.Tensor] = []
        gate_mean, gate_sp = [], []
        A, assign = None, None
        for i, blk in enumerate(self.blocks):
            if self.pool is not None and i == self.n_pre:
                x, A_pool, assign = self.pool(x, A)
                bases = soft_scale_bases(A_pool, self.scales)
            x, A, aux = blk(x, bases)
            if i < self.n_pre:                       # tokens still map 1:1 to image patches
                attn_maps.append(aux["received"])
            gate_mean.append(aux["gate_mean"])
            gate_sp.append(aux["gate_sparsity"])
        logits = self.head(self.norm(x).mean(dim=1))
        return logits, {"attn_maps": attn_maps, "pool_assign": assign,
                        "gate_mean": torch.stack(gate_mean).mean(),
                        "gate_sparsity": torch.stack(gate_sp).mean()}


def build_model(cfg: ModelConfig) -> nn.Module:
    return PatchVisionX(cfg)
