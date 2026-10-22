"""Non-learned reference forecasters."""
from __future__ import annotations

import numpy as np

from ..features import FeatureBuilder


class HistoricalAverage:
    """Predict the sensor's mean training speed for the target time-of-week slot."""

    def __init__(self, fb: FeatureBuilder):
        self.fb = fb

    def predict(self, anchors: np.ndarray, seq_out: int = 12) -> np.ndarray:
        steps = anchors[:, None] + np.arange(1, seq_out + 1)
        return self.fb.profile[self.fb.tow[steps]]            # (A, H, N)


class Persistence:
    """Carry the last observed speed forward over the whole horizon."""

    def __init__(self, fb: FeatureBuilder):
        self.fb = fb

    def predict(self, anchors: np.ndarray, seq_out: int = 12) -> np.ndarray:
        last = self.fb.filled[anchors]                          # (A, N)
        return np.repeat(last[:, None], seq_out, axis=1)
