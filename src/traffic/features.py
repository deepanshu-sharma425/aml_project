"""Hand-crafted features for the classical (non-deep) models.

One row is one (anchor time t, sensor i) pair and the label is the speed at
t + h. Every feature is *causal*: it only uses observations at or before t,
plus statistics fitted on the training period. Feature groups can be switched
on and off for the ablation study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import METRLA, STEPS_PER_DAY, STEPS_PER_WEEK, transition_matrices

N_LAGS = 12

FEATURE_GROUPS = ("lags", "window_stats", "calendar", "profile", "spatial", "sensor")


def forward_fill(speed: np.ndarray) -> np.ndarray:
    """Replace zeros (missing) with the last observed value of the same sensor.

    Leading gaps are back-filled with that sensor's first observation. This is
    causal except for that initial gap at the very start of the series.
    """
    df = pd.DataFrame(np.where(speed > 0, speed, np.nan))
    return df.ffill().bfill().to_numpy(dtype=np.float32)


def time_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Slot 0..2015 within the week (5-minute resolution)."""
    return (index.dayofweek * STEPS_PER_DAY + index.hour * 12 + index.minute // 5).to_numpy()


def historical_profile(speed: np.ndarray, tow: np.ndarray, end: int) -> np.ndarray:
    """Per-sensor mean speed for every time-of-week slot, from steps [0, end).

    Returns an array of shape (2016, N). Slots without observations fall back
    to the sensor's overall training mean.
    """
    s = speed[:end]
    obs = s > 0
    n = s.shape[1]
    total = np.zeros((STEPS_PER_WEEK, n))
    count = np.zeros((STEPS_PER_WEEK, n))
    np.add.at(total, tow[:end], np.where(obs, s, 0.0))
    np.add.at(count, tow[:end], obs)
    sensor_mean = np.where(obs, s, 0).sum(0) / np.maximum(obs.sum(0), 1)
    prof = np.where(count > 0, total / np.maximum(count, 1), sensor_mean)
    return prof.astype(np.float32)


class FeatureBuilder:
    def __init__(self, data: METRLA):
        self.data = data
        self.raw = data.speed
        self.filled = forward_fill(data.speed)
        self.tow = time_of_week(data.index)
        end = data.train_end
        self.profile = historical_profile(data.speed, self.tow, end)
        self.anomaly = self.filled - self.profile[self.tow]       # deviation from usual

        # Spatial operators without self-loops: forward = downstream sensors,
        # backward = upstream sensors (graph is directed along traffic flow).
        adj = data.adj.copy()
        np.fill_diagonal(adj, 0.0)
        self.p_fwd, self.p_bwd = transition_matrices(adj)

        train_obs = np.where(data.speed[:end] > 0, data.speed[:end], np.nan)
        self.sensor_mean = np.nanmean(train_obs, 0).astype(np.float32)
        self.sensor_std = np.nanstd(train_obs, 0).astype(np.float32)
        self.sensor_missing = np.isnan(train_obs).mean(0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def build(self, anchors: np.ndarray, h: int,
              groups: tuple[str, ...] = FEATURE_GROUPS) -> tuple[np.ndarray, list[str]]:
        """Feature matrix of shape (len(anchors) * N, F), anchor-major rows."""
        a = np.asarray(anchors)
        n = self.data.num_nodes
        lags = np.stack([self.filled[a - k] for k in range(N_LAGS)], -1)   # (A, N, 12)
        cols, names = [], []

        def add(name, value):
            value = np.broadcast_to(value, (len(a), n)).astype(np.float32)
            cols.append(value.reshape(-1))
            names.append(name)

        if "lags" in groups:
            for k in range(N_LAGS):
                add(f"lag_{k}", lags[..., k])
        if "window_stats" in groups:
            raw_win = np.stack([self.raw[a - k] for k in range(N_LAGS)], -1)
            add("diff_1", lags[..., 0] - lags[..., 1])
            add("diff_3", lags[..., 0] - lags[..., 3])
            add("roll_mean", lags.mean(-1))
            add("roll_std", lags.std(-1))
            add("roll_min", lags.min(-1))
            add("n_missing", (raw_win == 0).sum(-1))
            add("missing_now", raw_win[..., 0] == 0)
        if "calendar" in groups:
            ang = 2 * np.pi * (self.tow[a + h] % STEPS_PER_DAY) / STEPS_PER_DAY
            dow = self.data.index.dayofweek.to_numpy()[a + h]
            add("tod_sin", np.sin(ang)[:, None])
            add("tod_cos", np.cos(ang)[:, None])
            add("dow", dow[:, None])
            add("weekend", (dow >= 5)[:, None])
        if "profile" in groups:
            add("profile_target", self.profile[self.tow[a + h]])
            add("profile_now", self.profile[self.tow[a]])
            add("anomaly_now", self.anomaly[a])
        if "spatial" in groups:
            cur = self.filled[a]
            add("down_mean", cur @ self.p_fwd.T)
            add("up_mean", cur @ self.p_bwd.T)
            add("down_anomaly", self.anomaly[a] @ self.p_fwd.T)
            add("up_anomaly", self.anomaly[a] @ self.p_bwd.T)
            add("down_trend", (cur - self.filled[a - 3]) @ self.p_fwd.T)
        if "sensor" in groups:
            add("sensor_mean", self.sensor_mean[None])
            add("sensor_std", self.sensor_std[None])
            add("sensor_missing", self.sensor_missing[None])
        return np.stack(cols, 1), names

    def target(self, anchors: np.ndarray, h: int) -> np.ndarray:
        return self.raw[np.asarray(anchors) + h].reshape(-1)
