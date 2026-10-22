"""Training loop shared by the deep models."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import METRLA, time_of_day


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (target > 0).float()
    mask = mask / mask.mean().clamp_min(1e-8)
    return (torch.abs(pred - target) * mask).mean()


class WindowSampler:
    """Builds (x, y) batches on the device straight from the full series.

    x channels: z-scored speed (missing zeros keep their raw value, as in the
    DCRNN pipeline) and time-of-day. y is raw mph so the loss is in mph.
    """

    def __init__(self, data: METRLA, device: torch.device):
        tod = np.broadcast_to(time_of_day(data.index)[:, None], data.speed.shape)
        feats = np.stack([data.scaler.transform(data.speed), tod], -1).astype(np.float32)
        self.x = torch.from_numpy(feats).to(device)                     # (T, N, 2)
        self.y = torch.from_numpy(data.speed.astype(np.float32)).to(device)
        self.x_off = torch.arange(-data.seq_in + 1, 1, device=device)
        self.y_off = torch.arange(1, data.seq_out + 1, device=device)
        self.device = device

    def batch(self, anchors: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.as_tensor(anchors, device=self.device)[:, None]
        return self.x[a + self.x_off], self.y[a + self.y_off]


class Trainer:
    def __init__(self, model: nn.Module, data: METRLA, device: torch.device | None = None,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64,
                 clip: float = 5.0, ckpt: Path | None = None):
        self.device = device or pick_device()
        self.model = model.to(self.device)
        self.data = data
        self.sampler = WindowSampler(data, self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size, self.clip = batch_size, clip
        self.scale = (data.scaler.mean, data.scaler.std)
        self.ckpt = ckpt
        self.history: list[dict] = []

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # Models predict in normalised units; convert back to mph.
        return self.model(x) * self.scale[1] + self.scale[0]

    def train_epoch(self, rng: np.random.Generator) -> float:
        self.model.train()
        anchors = rng.permutation(self.data.split.train)
        losses = []
        for i in range(0, len(anchors), self.batch_size):
            x, y = self.sampler.batch(anchors[i:i + self.batch_size])
            loss = masked_mae_loss(self._forward(x), y)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def predict(self, anchors: np.ndarray, batch_size: int = 128) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(anchors), batch_size):
            x, _ = self.sampler.batch(anchors[i:i + batch_size])
            out.append(self._forward(x).cpu().numpy())
        return np.concatenate(out)

    def val_mae(self) -> float:
        pred = self.predict(self.data.split.val)
        y = self.data.speed[self.data.split.val[:, None] + np.arange(1, self.data.seq_out + 1)]
        m = y > 0
        return float(np.abs(pred[m] - y[m]).mean())

    def fit(self, epochs: int = 50, patience: int = 10, seed: int = 0) -> list[dict]:
        rng = np.random.default_rng(seed)
        best, wait, best_state = np.inf, 0, None
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(rng)
            va = self.val_mae()
            self.history.append({"epoch": ep, "train_mae": tr, "val_mae": va, "sec": time.time() - t0})
            print(f"epoch {ep:3d}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.0f}s)", flush=True)
            if va < best - 1e-4:
                best, wait = va, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.ckpt:
                    self.ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, self.ckpt)
            else:
                wait += 1
                if wait >= patience:
                    print(f"early stop at epoch {ep}; best val MAE {best:.4f}")
                    break
        self.model.load_state_dict(best_state)
        return self.history

    def save_history(self, path: Path) -> None:
        path.write_text(json.dumps(self.history, indent=2))
