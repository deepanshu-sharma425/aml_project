"""Direct multi-horizon strategy: one tabular regressor per forecast step."""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import clone

from ..features import FEATURE_GROUPS, FeatureBuilder


class PerHorizonRegressor:
    """Fit an independent copy of ``estimator`` for each horizon h = 1..H.

    The direct strategy avoids the error accumulation of recursive forecasting
    and lets each model weight features differently (e.g. the historical
    profile matters more at 60 min than at 5 min).
    """

    def __init__(self, estimator, fb: FeatureBuilder, horizons=range(1, 13),
                 groups: tuple[str, ...] = FEATURE_GROUPS):
        self.estimator, self.fb = estimator, fb
        self.horizons, self.groups = list(horizons), groups
        self.models: dict[int, object] = {}
        self.feature_names: list[str] = []

    def fit(self, anchors: np.ndarray, val_anchors: np.ndarray | None = None,
            verbose: bool = True) -> "PerHorizonRegressor":
        """``val_anchors`` (chronologically after training) drive early stopping
        for estimators that accept ``X_val`` / ``y_val``."""
        for h in self.horizons:
            t0 = time.time()
            x, self.feature_names = self.fb.build(anchors, h, self.groups)
            y = self.fb.target(anchors, h)
            keep = y > 0                                       # train on observed labels only
            kwargs = {}
            if val_anchors is not None:
                xv, _ = self.fb.build(val_anchors, h, self.groups)
                yv = self.fb.target(val_anchors, h)
                kwargs = {"X_val": xv[yv > 0], "y_val": yv[yv > 0]}
            self.models[h] = clone(self.estimator).fit(x[keep], y[keep], **kwargs)
            if verbose:
                print(f"  h={h:2d}  rows={keep.sum():,}  {time.time() - t0:.1f}s", flush=True)
        return self

    def predict(self, anchors: np.ndarray) -> np.ndarray:
        n = self.fb.data.num_nodes
        out = np.empty((len(anchors), len(self.horizons), n), dtype=np.float32)
        for j, h in enumerate(self.horizons):
            x, _ = self.fb.build(anchors, h, self.groups)
            out[:, j] = self.models[h].predict(x).reshape(len(anchors), n)
        return out
