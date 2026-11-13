"""Independent per-segment forecasters: the AML baseline of the hybrid pipeline.

One multi-output XGBoost model per road segment (sensor) maps that segment's own
lagged speeds, rolling statistics, cyclic calendar and weekly-profile features to
all 12 horizons. No information from other segments enters, by design: this is
the "independent segment model" whose blind spot, congestion travelling between
links, the graph attention network is meant to fill.

Base forecasts for the training period are produced by blocked, purged
cross-fitting, so the hybrid's second stage never sees in-sample (overfitted)
base predictions. See :func:`blocked_folds`.
"""
from __future__ import annotations

import numpy as np

from ..data import STEPS_PER_DAY
from ..features import N_LAGS, FeatureBuilder


class SegmentFeatures:
    """Horizon-independent features for one sensor, so one model covers all horizons.

    Per anchor t (37 features): 12 forward-filled lags; 1-step and 3-step
    differences, rolling mean/std/min, missing count and missing-now flag over the
    last hour; sin/cos time of day, day of week and weekend flag at t; the weekly
    profile at each target time t+1..t+12 (known in advance) and at t; and the
    current anomaly from the profile. All inputs are causal; the profile is fitted
    on training timestamps only.
    """

    def __init__(self, fb: FeatureBuilder, seq_out: int = 12):
        self.fb, self.seq_out = fb, seq_out
        idx = fb.data.index
        self.tod = (fb.tow % STEPS_PER_DAY) / STEPS_PER_DAY
        self.dow = idx.day_of_week.to_numpy()
        self.names = ([f"lag_{k}" for k in range(N_LAGS)]
                      + ["diff_1", "diff_3", "roll_mean", "roll_std", "roll_min", "n_missing", "missing_now"]
                      + ["tod_sin", "tod_cos", "dow", "weekend"]
                      + [f"profile_t+{h}" for h in range(1, seq_out + 1)] + ["profile_now", "anomaly_now"])

    def build(self, anchors: np.ndarray, i: int) -> np.ndarray:
        a = np.asarray(anchors)
        fb = self.fb
        lags = fb.filled[a[:, None] - np.arange(N_LAGS), i]            # (A, 12), lag 0 first
        raw = fb.raw[a[:, None] - np.arange(N_LAGS), i]
        ang = 2 * np.pi * self.tod[a]
        prof = fb.profile[fb.tow[a[:, None] + np.arange(1, self.seq_out + 1)], i]
        cols = [lags,
                (lags[:, 0] - lags[:, 1])[:, None], (lags[:, 0] - lags[:, 3])[:, None],
                lags.mean(1, keepdims=True), lags.std(1, keepdims=True), lags.min(1, keepdims=True),
                (raw == 0).sum(1, keepdims=True), (raw[:, :1] == 0),
                np.sin(ang)[:, None], np.cos(ang)[:, None], self.dow[a][:, None], (self.dow[a] >= 5)[:, None],
                prof, fb.profile[fb.tow[a], i][:, None], fb.anomaly[a, i][:, None]]
        return np.concatenate([np.asarray(c, dtype=np.float32) for c in cols], 1)

    def target(self, anchors: np.ndarray, i: int) -> np.ndarray:
        a = np.asarray(anchors)
        return self.fb.raw[a[:, None] + np.arange(1, self.seq_out + 1), i]    # (A, 12), 0 = missing


def blocked_folds(anchors: np.ndarray, k: int = 5, purge: int = 23) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous K-fold split of (chronological) anchors with a purge gap.

    Each fold holds out one contiguous block. Training anchors within ``purge``
    steps of the block are dropped, so no training window (11 input + 12 target
    steps) overlaps a held-out window. Blocked rather than shuffled folds keep
    serial dependence from leaking across the split (Bergmeir & Benitez, 2012).
    """
    blocks = np.array_split(np.asarray(anchors), k)
    folds = []
    for b in blocks:
        lo, hi = b[0] - purge, b[-1] + purge
        train = anchors[(anchors < lo) | (anchors > hi)]
        folds.append((train, b))
    return folds


def make_xgb(**kw):
    from xgboost import XGBRegressor
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  objective="reg:absoluteerror", multi_strategy="one_output_per_tree",
                  n_jobs=1, random_state=0)
    params.update(kw)
    return XGBRegressor(**params)


def fit_segment(sf: SegmentFeatures, i: int, anchors: np.ndarray, params: dict | None = None,
                val_anchors: np.ndarray | None = None):
    """Fit sensor i's multi-output model on windows whose 12 targets are all observed."""
    x, y = sf.build(anchors, i), sf.target(anchors, i)
    keep = (y > 0).all(1)
    kw = {}
    model = make_xgb(**(params or {}))
    if val_anchors is not None:
        xv, yv = sf.build(val_anchors, i), sf.target(val_anchors, i)
        kv = (yv > 0).all(1)
        model.set_params(early_stopping_rounds=30)
        kw = {"eval_set": [(xv[kv], yv[kv])], "verbose": False}
    return model.fit(x[keep], y[keep], **kw)
