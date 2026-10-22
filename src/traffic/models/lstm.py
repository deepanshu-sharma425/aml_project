"""Node-shared LSTM: a purely temporal deep baseline (no spatial information)."""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Encode each sensor's 12-step history with one shared LSTM, then map the
    final hidden state to all 12 future steps at once (direct strategy).

    Input  x: (B, T_in, N, C)    Output: (B, T_out, N)
    """

    def __init__(self, in_dim: int = 2, hidden: int = 64, layers: int = 2,
                 seq_out: int = 12, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, seq_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        _, (h, _) = self.lstm(seq)
        out = self.head(h[-1])                                  # (B*N, T_out)
        return out.view(b, n, -1).transpose(1, 2)
