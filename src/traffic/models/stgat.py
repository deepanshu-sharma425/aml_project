"""Spatio-temporal Graph Attention Network (ST-GAT) and the XGBoost->GAT hybrid.

Spatial mixing uses graph attention (Velickovic et al., ICLR 2018) restricted to
the edges of the *fixed* road graph:

    e_ij      = LeakyReLU( a_dst^T W h_i + a_src^T W h_j )        for j in N(i) u {i}
    alpha_ij  = softmax_j( e_ij )
    h_i'      = ||_{k=1..K}  sum_j alpha^k_ij W^k h_j              (K heads, concatenated)

The road graph is directed (73 % of edges are one-way), so attention is computed
separately over *downstream* neighbours (edges i -> j) and *upstream* neighbours
(edges j -> i), each with its own heads, and the two results are concatenated.
That lets the model weigh, for example, a slowdown just downstream (a queue that will
spill back) differently from one upstream.

Temporal mixing uses gated (GLU) convolutions as in STGCN (Yu et al., 2018). Each
block is a "sandwich" TConv -> GAT -> TConv; valid convolutions shrink the
12-step window 12 -> 10 -> 8 -> 6 -> 4 and a final temporal convolution collapses
it to one representation per sensor, which a linear head maps to 12 horizons.

The hybrid variant (``base_dim > 0``) additionally receives, for every sensor,
the 12-horizon forecast of an independent per-segment XGBoost model. A fusion
GAT layer lets each sensor attend to its neighbours' forecasts and states, and
the network outputs a *correction* that is added to the XGBoost forecast
(``residual=True``). The final layer is zero-initialised, so training starts
exactly at the XGBoost forecast and only has to learn what the independent
segment models miss: how congestion propagates between road segments.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SPATIAL_MODES = ("attention", "mean", "none", "dense")


def neighbour_mask(adj: np.ndarray, direction: str) -> torch.Tensor:
    """Boolean (N, N) mask, ``mask[i, j]`` = target i may attend to source j.

    ``adj[i, j] > 0`` means a road path from sensor i to sensor j (j is downstream).
    ``'down'``: i attends to its downstream sensors (adj[i, j]) and itself.
    ``'up'``:   i attends to its upstream sensors (adj[j, i]) and itself.
    ``'all'``:  every pair (fully connected attention, no road prior; ablation).
    """
    n = adj.shape[0]
    if direction == "all":
        return torch.ones(n, n, dtype=torch.bool)
    a = adj > 0
    a = a if direction == "down" else a.T
    return torch.as_tensor(a | np.eye(n, dtype=bool))


class GraphAttention(nn.Module):
    """Multi-head graph attention layer over a fixed neighbourhood mask.

    Input ``x``: (B, T, N, d_in). Output: (B, T, N, heads * d_head).

    The coefficients follow GAT exactly, but are computed once per input window
    from the time-averaged projected states and shared by its T steps. ASTGCN
    (Guo et al., 2019) shares spatial attention across a window in the same way.
    This makes the layer one masked (N x N) matrix product per head, which is fast and
    memory-light. With T = 1 it is the standard GAT layer; a unit test checks it
    against PyTorch Geometric's ``GATConv``.

    ``mode='mean'`` replaces the learned coefficients with a uniform average over
    the same neighbours (GraphSAGE-mean / GCN-style), isolating the value of
    attention itself.
    """

    def __init__(self, d_in: int, d_head: int, heads: int, mask: torch.Tensor,
                 mode: str = "attention", attn_dropout: float = 0.0, negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.d_head, self.mode = heads, d_head, mode
        self.lin = nn.Linear(d_in, heads * d_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, 1, d_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, 1, d_head))
        self.bias = nn.Parameter(torch.zeros(heads * d_head))
        self.register_buffer("mask", mask.clone())
        self.attn_dropout, self.slope = attn_dropout, negative_slope
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def attention(self, wh: torch.Tensor) -> torch.Tensor:
        """Coefficients alpha, (B, heads, N, N); rows sum to 1 over each target's neighbours."""
        b, _, n, _, _ = wh.shape
        if self.mode == "mean":
            m = self.mask.float()
            return (m / m.sum(1, keepdim=True)).expand(b, self.heads, n, n)
        summary = wh.mean(1).transpose(1, 2)                       # (B, H, N, D)
        s_src = (summary * self.att_src).sum(-1)                   # (B, H, N)
        s_dst = (summary * self.att_dst).sum(-1)
        e = F.leaky_relu(s_dst[..., :, None] + s_src[..., None, :], self.slope)
        e = e.masked_fill(~self.mask, float("-inf"))
        return torch.softmax(e, -1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        b, t, n, _ = x.shape
        wh = self.lin(x).view(b, t, n, self.heads, self.d_head)
        alpha = self.attention(wh)
        a = F.dropout(alpha, self.attn_dropout, self.training)
        out = torch.einsum("bhij,btjhd->btihd", a, wh).reshape(b, t, n, -1) + self.bias
        return (out, alpha) if return_attention else out


class DirectionalGAT(nn.Module):
    """Attention over downstream and over upstream neighbours, concatenated and mixed.
    ``mode='dense'`` uses one attention over all sensor pairs instead (no road prior)."""

    def __init__(self, d: int, adj: np.ndarray, heads: int = 4, mode: str = "attention",
                 attn_dropout: float = 0.1):
        super().__init__()
        d_head = max(d // (2 * heads), 1)
        if mode == "dense":
            self.layers = nn.ModuleList([GraphAttention(d, 2 * d_head, heads, neighbour_mask(adj, "all"),
                                                        "attention", attn_dropout)])
        else:
            self.layers = nn.ModuleList([GraphAttention(d, d_head, heads, neighbour_mask(adj, direction),
                                                        mode, attn_dropout) for direction in ("down", "up")])
        self.proj = nn.Linear(2 * heads * d_head, d)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        outs = [layer(x, return_attention) for layer in self.layers]
        if return_attention:
            return self.proj(F.elu(torch.cat([o[0] for o in outs], -1))), [o[1] for o in outs]
        return self.proj(F.elu(torch.cat(outs, -1)))


class TemporalGLU(nn.Module):
    """Gated temporal convolution (valid padding): P * sigmoid(Q) + residual."""

    def __init__(self, c_in: int, c_out: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(c_in, 2 * c_out, (1, kernel))
        self.res = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, C, N, T)
        p, q = self.conv(x).chunk(2, 1)
        return (p + self.res(x)[..., self.kernel - 1:]) * torch.sigmoid(q)


class STBlock(nn.Module):
    def __init__(self, d: int, adj: np.ndarray, heads: int, spatial: str, kernel: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.t1 = TemporalGLU(d, d, kernel)
        self.spatial = spatial
        if spatial != "none":
            self.gat = DirectionalGAT(d, adj, heads, spatial, attn_dropout)
        self.t2 = TemporalGLU(d, d, kernel)
        self.norm = nn.LayerNorm(d)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, C, N, T)
        x = self.t1(x)
        if self.spatial != "none":
            h = x.permute(0, 3, 2, 1)                             # (B, T, N, C)
            h = h + self.gat(h)                                   # residual spatial update
            x = h.permute(0, 3, 2, 1)
        x = self.t2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return F.dropout(x, self.dropout, self.training)


class STGAT(nn.Module):
    """Input x: (B, T_in, N, C); optional base forecast (B, T_out, N) in normalised units.
    Output: (B, T_out, N) in normalised units.

    Parameters
    ----------
    spatial : 'attention' (directional GAT on the road graph), 'dense' (attention
        over all sensor pairs), 'mean' (uniform neighbour average) or 'none'
        (no information crosses sensors).
    base_dim : number of base-model horizons fed in (0 = pure deep model).
    residual : add the network output to the base forecast (hybrid only).
    """

    def __init__(self, num_nodes: int, adj: np.ndarray, in_dim: int = 6, seq_in: int = 12,
                 seq_out: int = 12, d: int = 64, heads: int = 4, blocks: int = 2, kernel: int = 3,
                 node_emb: int = 16, spatial: str = "attention", dropout: float = 0.2,
                 attn_dropout: float = 0.1, base_dim: int = 0, residual: bool = False):
        super().__init__()
        assert spatial in SPATIAL_MODES, spatial
        self.spatial, self.base_dim, self.residual = spatial, base_dim, residual and base_dim > 0
        self.node_emb = nn.Parameter(torch.randn(num_nodes, node_emb) * 0.1) if node_emb else None
        self.inp = nn.Conv2d(in_dim + (node_emb or 0), d, 1)
        self.blocks = nn.ModuleList([STBlock(d, adj, heads, spatial, kernel, dropout, attn_dropout)
                                     for _ in range(blocks)])
        t_left = seq_in - blocks * 2 * (kernel - 1)
        assert t_left >= 1, "window too short for this many blocks"
        self.t_out = TemporalGLU(d, d, t_left)                    # collapse remaining steps
        if base_dim:
            self.base_proj = nn.Linear(base_dim, d)
            self.fuse = nn.Linear(2 * d, d)
            self.fuse_gat = DirectionalGAT(d, adj, heads, spatial, attn_dropout) if spatial != "none" else None
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, seq_out))
        if self.residual:                                        # start exactly at the base forecast
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def encode(self, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        """Per-sensor representation (B, N, d) before the output head."""
        b, t, n, _ = x.shape
        x = x.permute(0, 3, 2, 1)                                 # (B, C, N, T)
        if self.node_emb is not None:
            emb = self.node_emb.T[None, :, :, None].expand(b, -1, -1, t)
            x = torch.cat([x, emb], 1)
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        h = self.t_out(x).squeeze(-1).transpose(1, 2)            # (B, N, d)
        if self.base_dim:
            z = self.base_proj(base.transpose(1, 2))             # (B, N, d)
            h = F.relu(self.fuse(torch.cat([h, z], -1)))
            if self.fuse_gat is not None:
                h = h + self.fuse_gat(h[:, None])[:, 0]
        return h

    def forward(self, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        out = self.head(self.encode(x, base)).transpose(1, 2)    # (B, T_out, N)
        return base + out if self.residual else out

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor, block: int = 0) -> list[torch.Tensor]:
        """GAT coefficients of one ST block: [downstream, upstream], each (B, heads, N, N)."""
        b, t, _, _ = x.shape
        h = x.permute(0, 3, 2, 1)
        if self.node_emb is not None:
            h = torch.cat([h, self.node_emb.T[None, :, :, None].expand(b, -1, -1, t)], 1)
        h = self.inp(h)
        for i, blk in enumerate(self.blocks):
            if i == block % len(self.blocks):
                return blk.gat(blk.t1(h).permute(0, 3, 2, 1), return_attention=True)[1]
            h = blk(h)
        raise IndexError(block)
