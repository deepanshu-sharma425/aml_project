"""Data loading, graph construction, splitting and windowing for METR-LA.

METR-LA contains 5-minute average speeds (mph) from 207 loop detectors on Los
Angeles County highways, 1 Mar 2012 - 27 Jun 2012 (34,272 time steps).
A reading of exactly 0 denotes a missing / faulty observation, not a stopped
vehicle, and is masked out of every loss and metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Dataset"

STEPS_PER_DAY = 288  # 24 h * 12 five-minute intervals
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


def load_speed(path: Path | str = DATA_DIR / "METR-LA.h5") -> pd.DataFrame:
    """Read the speed matrix (time x sensor).

    The file was written by pandas 0.15 and cannot be read by recent
    ``pd.read_hdf``, so the underlying PyTables arrays are read with h5py.
    """
    with h5py.File(path, "r") as f:
        g = f["df"]
        cols = [c.decode() for c in g["axis0"][:]]
        index = pd.to_datetime(g["axis1"][:])
        values = g["block0_values"][:].astype(np.float32)
    return pd.DataFrame(values, index=index, columns=cols)


def load_sensor_ids(path: Path | str = DATA_DIR / "graph_sensor_ids.txt") -> list[str]:
    return Path(path).read_text().strip().split(",")


def load_locations(path: Path | str = DATA_DIR / "graph_sensor_locations.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df["sensor_id"] = df["sensor_id"].astype(str)
    return df


def distance_matrix(sensor_ids: list[str],
                    path: Path | str = DATA_DIR / "distances_la_2012.csv") -> np.ndarray:
    """Directed road-network distance (metres); ``inf`` where no path is listed."""
    dist = pd.read_csv(path, dtype={"from": str, "to": str})
    pos = {s: i for i, s in enumerate(sensor_ids)}
    dist = dist[dist["from"].isin(pos) & dist["to"].isin(pos)]
    n = len(sensor_ids)
    d = np.full((n, n), np.inf, dtype=np.float32)
    d[dist["from"].map(pos).to_numpy(), dist["to"].map(pos).to_numpy()] = dist["cost"].to_numpy()
    return d


def gaussian_adjacency(dist: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Thresholded Gaussian kernel, W_ij = exp(-d_ij^2 / sigma^2) (Li et al., 2018).

    sigma is the standard deviation of all finite distances; weights below
    ``threshold`` are zeroed to keep the graph sparse.
    """
    finite = dist[np.isfinite(dist)]
    sigma = finite.std()
    w = np.exp(-np.square(dist / sigma))
    w[w < threshold] = 0.0
    return w.astype(np.float32)


def load_adjacency(sensor_ids: list[str] | None = None, threshold: float = 0.1) -> np.ndarray:
    sensor_ids = sensor_ids or load_sensor_ids()
    return gaussian_adjacency(distance_matrix(sensor_ids), threshold)


def transition_matrices(adj: np.ndarray) -> list[np.ndarray]:
    """Forward and backward random-walk matrices D_O^{-1} W and D_I^{-1} W^T."""
    def rw(a):
        d = a.sum(1, keepdims=True)
        d[d == 0] = 1.0
        return (a / d).astype(np.float32)
    return [rw(adj), rw(adj.T)]


@dataclass
class Split:
    """Index ranges into the array of window *anchor* times (last input step)."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(num_steps: int, seq_in: int = 12, seq_out: int = 12,
               ratios: tuple[float, float] = (0.7, 0.1)) -> Split:
    """Chronological 70/10/20 split of window anchors, as in DCRNN / Graph WaveNet.

    An anchor ``t`` uses inputs ``t-seq_in+1 .. t`` and targets ``t+1 .. t+seq_out``.
    Splitting by time (never shuffling across the boundary) prevents look-ahead
    leakage.
    """
    anchors = np.arange(seq_in - 1, num_steps - seq_out)
    n = len(anchors)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    return Split(anchors[:n_train], anchors[n_train:n_train + n_val], anchors[n_train + n_val:])


class StandardScaler:
    """z-score normalisation fitted on observed (non-zero) training values only."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray) -> "StandardScaler":
        obs = x[x > 0]
        self.mean, self.std = float(obs.mean()), float(obs.std())
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Fraction of the day elapsed, in [0, 1)."""
    return ((index.hour * 60 + index.minute) / (24 * 60)).to_numpy(dtype=np.float32)


def day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek.to_numpy(dtype=np.int64)


def windows(values: np.ndarray, anchors: np.ndarray, seq_in: int = 12,
            seq_out: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Gather (inputs, targets) for the given anchors from a (T, ...) array."""
    x_off = np.arange(-seq_in + 1, 1)
    y_off = np.arange(1, seq_out + 1)
    return values[anchors[:, None] + x_off], values[anchors[:, None] + y_off]


@dataclass
class METRLA:
    """Everything needed by the experiments, loaded once."""
    speed: np.ndarray           # (T, N) raw mph, 0 = missing
    index: pd.DatetimeIndex
    sensor_ids: list[str]
    adj: np.ndarray             # (N, N)
    split: Split
    scaler: StandardScaler
    seq_in: int = 12
    seq_out: int = 12

    @property
    def num_nodes(self) -> int:
        return self.speed.shape[1]

    @property
    def train_end(self) -> int:
        """First time step not visible to training (inputs or targets)."""
        return int(self.split.train[-1] + self.seq_out + 1)


def load_metrla(seq_in: int = 12, seq_out: int = 12) -> METRLA:
    df = load_speed()
    ids = load_sensor_ids()
    assert list(df.columns) == ids, "speed columns must follow graph_sensor_ids order"
    split = make_split(len(df), seq_in, seq_out)
    speed = df.to_numpy()
    train_end = int(split.train[-1] + seq_out + 1)
    scaler = StandardScaler().fit(speed[:train_end])
    return METRLA(speed, df.index, ids, load_adjacency(ids), split, scaler, seq_in, seq_out)
