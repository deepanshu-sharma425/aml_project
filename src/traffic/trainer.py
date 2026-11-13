"""Training loop shared by the deep models."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import METRLA, day_of_week, time_of_day


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


INPUT_MODES = ("raw", "masked")


def input_features(data: METRLA, mode: str = "raw") -> np.ndarray:
    """Per-step model inputs, shape (T, N, C).

    ``raw`` (Phase 1, C=2): z-scored speed with missing zeros left in place, plus
    time of day in [0, 1). This is the DCRNN / Graph WaveNet pipeline, but it shows the
    network an outage as a real reading of 0 mph.

    ``masked`` (C=6): outages are *masked*, not treated as zero traffic. Speed is
    causally forward-filled and z-scored, an explicit 0/1 channel says whether the
    reading was actually observed, and time of day and day of week enter as
    sin/cos pairs so 23:55 is next to 00:00 and Sunday is next to Monday. Every
    channel at time t uses only data up to t, and the scaler was fitted on
    training timestamps only.
    """
    n = data.num_nodes
    if mode == "raw":
        tod = np.broadcast_to(time_of_day(data.index)[:, None], data.speed.shape)
        return np.stack([data.scaler.transform(data.speed), tod], -1).astype(np.float32)
    from .features import forward_fill
    filled = data.scaler.transform(forward_fill(data.speed))
    observed = (data.speed > 0).astype(np.float32)
    tod = 2 * np.pi * time_of_day(data.index)
    dow = 2 * np.pi * day_of_week(data.index) / 7
    cyc = np.stack([np.sin(tod), np.cos(tod), np.sin(dow), np.cos(dow)], -1).astype(np.float32)
    cyc = np.broadcast_to(cyc[:, None, :], (len(data.index), n, 4))
    return np.concatenate([filled[..., None], observed[..., None], cyc], -1).astype(np.float32)


class WindowSampler:
    """Builds (x, y[, base]) batches on the device straight from the full series.

    x: per-step inputs from :func:`input_features`. y is raw mph so the loss is in
    mph. ``base`` (hybrid models only) is an array (T, H, N) of base-model
    forecasts in mph indexed by anchor time; it is z-scored with the training
    scaler before it reaches the network.
    """

    def __init__(self, data: METRLA, device: torch.device, inputs: str = "raw",
                 base: np.ndarray | None = None):
        self.x = torch.from_numpy(input_features(data, inputs)).to(device)   # (T, N, C)
        self.y = torch.from_numpy(data.speed.astype(np.float32)).to(device)
        self.base = None
        if base is not None:
            b = data.scaler.transform(np.nan_to_num(base, nan=data.scaler.mean)).astype(np.float32)
            self.base = torch.from_numpy(b).to(device)
        self.x_off = torch.arange(-data.seq_in + 1, 1, device=device)
        self.y_off = torch.arange(1, data.seq_out + 1, device=device)
        self.device = device

    def batch(self, anchors: np.ndarray):
        a = torch.as_tensor(anchors, device=self.device)
        x, y = self.x[a[:, None] + self.x_off], self.y[a[:, None] + self.y_off]
        return (x, y, self.base[a]) if self.base is not None else (x, y)


class Trainer:
    def __init__(self, model: nn.Module, data: METRLA, device: torch.device | None = None,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 32,
                 clip: float = 5.0, ckpt: Path | None = None, inputs: str = "raw",
                 base: np.ndarray | None = None, optimizer: str = "adam",
                 lr_patience: int | None = None, epoch_fraction: float = 1.0):
        """``optimizer='adamw'`` uses decoupled weight decay (Loshchilov & Hutter,
        2019); ``lr_patience`` halves the learning rate after that many epochs
        without validation improvement; ``epoch_fraction`` < 1 trains each epoch on
        a fresh random subset of the training windows (adjacent windows share 11
        of 12 steps, so a random half loses little and halves the cost)."""
        self.device = device or pick_device()
        self.model = model.to(self.device)
        self.data = data
        self.sampler = WindowSampler(data, self.device, inputs, base)
        opt = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam
        self.opt = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=lr_patience)
                      if lr_patience else None)
        self.batch_size, self.clip = batch_size, clip
        self.scale = (data.scaler.mean, data.scaler.std)
        self.ckpt = ckpt
        self.epoch_fraction = epoch_fraction
        self.history: list[dict] = []

    def _forward(self, x: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        # Models predict in normalised units; convert back to mph.
        out = self.model(x) if base is None else self.model(x, base)
        return out * self.scale[1] + self.scale[0]

    def train_epoch(self, rng: np.random.Generator) -> float:
        self.model.train()
        anchors = rng.permutation(self.data.split.train)
        anchors = anchors[: max(1, int(round(len(anchors) * self.epoch_fraction)))]
        losses = []
        for i in range(0, len(anchors), self.batch_size):
            x, y, *base = self.sampler.batch(anchors[i:i + self.batch_size])
            loss = masked_mae_loss(self._forward(x, *base), y)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def predict(self, anchors: np.ndarray, batch_size: int = 32) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(anchors), batch_size):
            x, _, *base = self.sampler.batch(anchors[i:i + batch_size])
            out.append(self._forward(x, *base).cpu().numpy())
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
            lr = self.opt.param_groups[0]["lr"]
            if self.sched is not None:
                self.sched.step(va)
            self.history.append({"epoch": ep, "train_mae": tr, "val_mae": va, "lr": lr,
                                 "sec": time.time() - t0})
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
