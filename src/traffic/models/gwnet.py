"""Graph WaveNet (Wu et al., IJCAI 2019), re-implemented in compact form.

Each layer stacks a gated dilated causal convolution over time (WaveNet) with
a diffusion graph convolution over sensors. The graph convolution uses the
forward / backward random-walk matrices of the road network plus a learned
*adaptive* adjacency softmax(ReLU(E1 E2)) that can discover dependencies the
distance graph misses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def graph_mix(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """x: (B, C, N, T), a: (N, N) -> sum_v x[..., v, :] * a[v, w]."""
    return torch.einsum("bcvt,vw->bcwt", x, a)


class DiffusionConv(nn.Module):
    """Z = sum_k sum_s  P_k^s X W_{k,s}   for s = 0..order, k over supports."""

    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(c_in * (n_supports * order + 1), c_out, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in supports:
            x1 = graph_mix(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = graph_mix(x1, a)
                out.append(x1)
        h = self.mlp(torch.cat(out, 1))
        return F.dropout(h, self.dropout, self.training)


class GraphWaveNet(nn.Module):
    """Input x: (B, T_in, N, C)   Output: (B, T_out, N).

    ``supports`` are fixed transition matrices (may be empty); ``adaptive``
    toggles the learned adjacency; ``use_graph=False`` drops graph convolution
    entirely (a temporal-only TCN, used in the ablation study).
    """

    def __init__(self, num_nodes: int, supports: list[torch.Tensor] | None = None,
                 in_dim: int = 2, seq_out: int = 12, adaptive: bool = True, use_graph: bool = True,
                 residual: int = 32, dilation: int = 32, skip: int = 256, end: int = 512,
                 blocks: int = 4, layers: int = 2, kernel: int = 2, emb_dim: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.supports = list(supports or [])
        for i, s in enumerate(self.supports):
            self.register_buffer(f"support_{i}", s)
        self.adaptive = adaptive and use_graph
        self.use_graph = use_graph and (self.adaptive or len(self.supports) > 0)
        if self.adaptive:
            self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
            self.e2 = nn.Parameter(torch.randn(emb_dim, num_nodes))
        n_supports = len(self.supports) + int(self.adaptive)

        self.start = nn.Conv2d(in_dim, residual, 1)
        self.filter, self.gate, self.skip, self.residual, self.gconv, self.bn = (
            nn.ModuleList() for _ in range(6))
        self.receptive_field = 1
        for _ in range(blocks):
            d, extra = 1, kernel - 1
            for _ in range(layers):
                self.filter.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.gate.append(nn.Conv2d(residual, dilation, (1, kernel), dilation=d))
                self.skip.append(nn.Conv2d(dilation, skip, 1))
                if self.use_graph:
                    self.gconv.append(DiffusionConv(dilation, residual, n_supports, dropout=dropout))
                else:
                    self.residual.append(nn.Conv2d(dilation, residual, 1))
                self.bn.append(nn.BatchNorm2d(residual))
                self.receptive_field += extra
                extra *= 2
                d *= 2
        self.end1 = nn.Conv2d(skip, end, 1)
        self.end2 = nn.Conv2d(end, seq_out, 1)

    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.e1 @ self.e2), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1)                               # (B, C, N, T)
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0))
        x = self.start(x)
        supports = [getattr(self, f"support_{i}") for i in range(len(self.supports))]
        if self.adaptive:
            supports = supports + [self.adaptive_adj()]

        skip = 0
        for i in range(len(self.filter)):
            res = x
            x = torch.tanh(self.filter[i](res)) * torch.sigmoid(self.gate[i](res))
            s = self.skip[i](x)
            skip = s + (skip[..., -s.size(3):] if torch.is_tensor(skip) else 0)
            x = self.gconv[i](x, supports) if self.use_graph else self.residual[i](x)
            x = self.bn[i](x + res[..., -x.size(3):])

        h = F.relu(self.end1(F.relu(skip)))
        return self.end2(h).squeeze(-1)                         # (B, T_out, N)
