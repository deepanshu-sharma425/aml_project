"""Data validation, cleaning and imputation for METR-LA.

The cleaned series is an *analysis* artefact (data-quality report, figures,
the app's raw-vs-clean view). The forecasting models deliberately keep using
the raw speeds with a missing-value mask, as in DCRNN / Graph WaveNet:
interpolation looks at future values, and cleaning the test targets would make
our numbers incomparable with the literature.

Pipeline (each cell gets at most one issue code, assigned in this order):

1. ``out_of_range``  non-finite, negative or above the 70 mph sensor cap
2. ``outage``        every sensor reads 0 at that time step (network failure)
3. ``missing``       the sensor reads 0 (the dataset's missing-value code)
4. ``stuck``         the same non-zero value repeated for >= 1 h below the cap
                     (the first reading of the run is kept)
5. ``spike``         Hampel filter: far from the centred rolling median

Flagged cells are then imputed: gaps of <= 1 h with observations on both sides
by linear interpolation, longer gaps by the sensor's weekly profile fitted on
the training period only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import DATA_DIR
from .features import historical_profile, time_of_week

CLEAN_PATH = DATA_DIR / "cleaned" / "METR-LA_clean.h5"

# Issue codes (int8, one per cell); 0 = valid observation kept as is
OK, OUT_OF_RANGE, OUTAGE, MISSING, STUCK, SPIKE = range(6)
ISSUES = {OUT_OF_RANGE: "out_of_range", OUTAGE: "outage", MISSING: "missing", STUCK: "stuck", SPIKE: "spike"}
# Fill codes (int8, one per cell)
RAW, LINEAR, PROFILE = range(3)
FILLS = {RAW: "original", LINEAR: "linear", PROFILE: "weekly_profile"}


@dataclass
class CleanConfig:
    speed_cap: float = 70.0          # sensors clip speeds at 70 mph
    stuck_min_steps: int = 12        # 1 h of identical readings
    spike_window: int = 7            # centred, 35 min
    spike_k: float = 5.0             # robust z-score threshold
    spike_min_mph: float = 20.0      # and at least this far from the median
    max_interp_steps: int = 12       # longest gap filled by linear interpolation


@dataclass
class CleanResult:
    speed: np.ndarray                # (T, N) cleaned mph, no gaps
    issue: np.ndarray                # (T, N) int8 issue code
    filled: np.ndarray               # (T, N) int8 fill code
    config: CleanConfig = field(default_factory=CleanConfig)

    def counts(self) -> dict:
        total = self.issue.size
        issues = {name: int((self.issue == c).sum()) for c, name in ISSUES.items()}
        fills = {name: int((self.filled == c).sum()) for c, name in FILLS.items()}
        return {"cells": int(total), "valid": int((self.issue == OK).sum()),
                "issues": issues, "issues_pct": {k: 100 * v / total for k, v in issues.items()},
                "fills": fills, "fills_pct": {k: 100 * v / total for k, v in fills.items()}}


# ---------------------------------------------------------------- validation
def validate(df: pd.DataFrame, sensor_ids: list[str], locations: pd.DataFrame,
             distances: pd.DataFrame, speed_cap: float = 70.0) -> list[dict]:
    """Schema and integrity checks on the raw files. Returns one row per check."""
    v = df.to_numpy()
    step = np.diff(df.index.asi8)
    in_graph = distances["from"].isin(sensor_ids) & distances["to"].isin(sensor_ids)
    checks = [
        ("timestamps strictly increasing", bool(df.index.is_monotonic_increasing and df.index.is_unique),
         f"{df.index.duplicated().sum()} duplicates"),
        ("regular 5-minute grid", bool((step == 5 * 60 * 10**9).all()), f"{int((step != 5 * 60 * 10**9).sum())} irregular steps"),
        ("columns match graph sensor order", list(df.columns) == list(sensor_ids), f"{df.shape[1]} columns"),
        ("no duplicate sensors", not df.columns.duplicated().any(), f"{df.columns.duplicated().sum()} duplicates"),
        ("speeds finite", bool(np.isfinite(v).all()), f"{int((~np.isfinite(v)).sum())} NaN/inf"),
        (f"speeds within [0, {speed_cap:g}] mph", bool(((v >= 0) & (v <= speed_cap)).all()),
         f"{int(((v < 0) | (v > speed_cap)).sum())} out of range"),
        ("every sensor has a location", set(sensor_ids) <= set(locations["sensor_id"]),
         f"{len(set(sensor_ids) - set(locations['sensor_id']))} missing"),
        ("locations unique and complete", not locations["sensor_id"].duplicated().any()
         and not locations[["latitude", "longitude"]].isna().any().any(), f"{len(locations)} rows"),
        ("coordinates inside Los Angeles", bool(locations["latitude"].between(33.5, 34.5).all()
                                                and locations["longitude"].between(-119, -117.5).all()), ""),
        ("road distances non-negative", bool((distances["cost"] >= 0).all()), f"{int((distances['cost'] < 0).sum())} negative"),
        ("no duplicate road distances", not distances.duplicated(["from", "to"]).any(),
         f"{int(distances.duplicated(['from', 'to']).sum())} duplicates"),
        ("distances between graph sensors", bool(in_graph.any()), f"{int(in_graph.sum())} of {len(distances)} rows"),
    ]
    return [{"check": c, "passed": bool(p), "detail": d} for c, p, d in checks]


# ---------------------------------------------------------------- detectors
def detect_out_of_range(speed: np.ndarray, cap: float = 70.0) -> np.ndarray:
    return ~np.isfinite(speed) | (speed < 0) | (speed > cap)


def detect_outages(speed: np.ndarray) -> np.ndarray:
    """(T,) mask of time steps at which every sensor reads 0."""
    return (speed == 0).all(1)


def detect_stuck(speed: np.ndarray, min_steps: int = 12, cap: float = 70.0) -> np.ndarray:
    """Repeats of a constant non-zero reading lasting >= ``min_steps``.

    Runs at the cap are normal free-flow traffic (e.g. at night) and are kept;
    so is the first reading of every run.
    """
    out = np.zeros(speed.shape, dtype=bool)
    for i in range(speed.shape[1]):
        c = speed[:, i]
        start = np.r_[True, c[1:] != c[:-1]]
        run = np.cumsum(start) - 1
        length = np.bincount(run)[run]
        out[:, i] = (length >= min_steps) & ~start & (c > 0) & (c < cap)
    return out


def detect_spikes(speed: np.ndarray, window: int = 7, k: float = 5.0, min_mph: float = 20.0) -> np.ndarray:
    """Hampel filter on observed (non-NaN) values.

    A reading is a spike if it deviates from the centred rolling median by more
    than ``k`` robust standard deviations (1.4826 * MAD) *and* by ``min_mph``.
    A genuine onset of congestion persists, so the median follows it and it is
    not flagged; an isolated one-step drop is.
    """
    x = pd.DataFrame(speed)
    med = x.rolling(window, center=True, min_periods=3).median()
    dev = (x - med).abs()
    mad = dev.rolling(window, center=True, min_periods=3).median()
    return ((dev > k * 1.4826 * mad) & (dev > min_mph)).to_numpy()


def gap_lengths(missing: np.ndarray) -> np.ndarray:
    """Length of the run of ``True`` each cell belongs to (0 where False), per column."""
    m = np.asarray(missing, dtype=bool)
    out = np.zeros(m.shape, dtype=np.int64)
    for i in range(m.shape[1]):
        c = m[:, i]
        brk = np.r_[True, c[1:] != c[:-1]]
        run = np.cumsum(brk) - 1
        out[:, i] = np.where(c, np.bincount(run)[run], 0)
    return out


# ---------------------------------------------------------------- imputation
def interpolate_short(x: np.ndarray, max_steps: int = 12) -> np.ndarray:
    """Linear interpolation of NaN gaps of <= ``max_steps`` bounded on both sides."""
    full = pd.DataFrame(x).interpolate(limit_area="inside").to_numpy(dtype=np.float32)
    short = np.isnan(x) & (gap_lengths(np.isnan(x)) <= max_steps)
    return np.where(short, full, x)


def clean(speed: np.ndarray, index: pd.DatetimeIndex, train_end: int,
          config: CleanConfig | None = None) -> CleanResult:
    """Flag bad readings and impute them. ``speed`` is not modified."""
    cfg = config or CleanConfig()
    raw = np.asarray(speed, dtype=np.float32)
    issue = np.zeros(raw.shape, dtype=np.int8)

    def flag(mask, code):
        issue[(issue == OK) & mask] = code

    flag(detect_out_of_range(raw, cfg.speed_cap), OUT_OF_RANGE)
    flag(np.broadcast_to(detect_outages(raw)[:, None], raw.shape), OUTAGE)
    flag(raw == 0, MISSING)
    flag(detect_stuck(raw, cfg.stuck_min_steps, cfg.speed_cap), STUCK)
    flag(detect_spikes(np.where(issue == OK, raw, np.nan), cfg.spike_window, cfg.spike_k, cfg.spike_min_mph), SPIKE)

    x = np.where(issue == OK, raw, np.nan).astype(np.float32)
    filled = np.where(issue == OK, RAW, PROFILE).astype(np.int8)
    x = interpolate_short(x, cfg.max_interp_steps)
    filled[(issue != OK) & ~np.isnan(x)] = LINEAR

    tow = time_of_week(index)
    profile = historical_profile(np.where(issue == OK, raw, 0), tow, train_end)   # training period only
    x = np.where(np.isnan(x), profile[tow], x).astype(np.float32)
    return CleanResult(x, issue, filled, cfg)


def sensor_quality(result: CleanResult, sensor_ids: list[str]) -> pd.DataFrame:
    """Per-sensor share of flagged readings by issue type and longest gap."""
    t = result.issue.shape[0]
    df = pd.DataFrame({"sensor_id": sensor_ids})
    for code, name in ISSUES.items():
        df[f"{name}_pct"] = 100 * (result.issue == code).sum(0) / t
    df["flagged_pct"] = 100 * (result.issue != OK).mean(0)
    df["longest_gap_steps"] = gap_lengths(result.issue != OK).max(0)
    df["longest_gap_hours"] = df["longest_gap_steps"] / 12
    df["linear_fill_pct"] = 100 * (result.filled == LINEAR).mean(0)
    df["profile_fill_pct"] = 100 * (result.filled == PROFILE).mean(0)
    return df.sort_values("flagged_pct", ascending=False).reset_index(drop=True)


def imputation_benchmark(result: CleanResult, index: pd.DatetimeIndex, train_end: int,
                         gap_steps=(1, 3, 6, 12, 36, 144, 288), n_gaps: int = 2000,
                         seed: int = 0) -> pd.DataFrame:
    """Hide blocks of valid readings after the training period and score each fill method.

    The weekly profile is fitted on training data only, so it never sees the
    hidden values. Returns MAE (mph) per gap length and method.
    """
    rng = np.random.default_rng(seed)
    valid = result.issue == OK
    tow = time_of_week(index)
    profile = historical_profile(np.where(valid, result.speed, 0), tow, train_end)
    x, t_max = result.speed, len(index)
    rows = []
    for gap in gap_steps:
        err = {m: [] for m in ("forward_fill", "linear", "weekly_profile", "hybrid (ours)")}
        tries = 0
        while len(err["linear"]) < n_gaps and tries < 50 * n_gaps:
            tries += 1
            i = rng.integers(x.shape[1])
            t0 = rng.integers(max(train_end, 1), t_max - gap - 1)
            if not valid[t0 - 1:t0 + gap + 1, i].all():
                continue
            truth = x[t0:t0 + gap, i]
            before, after = x[t0 - 1, i], x[t0 + gap, i]
            w = np.arange(1, gap + 1) / (gap + 1)
            linear = before + (after - before) * w
            prof = profile[tow[t0:t0 + gap], i]
            err["forward_fill"].append(np.abs(before - truth).mean())
            err["linear"].append(np.abs(linear - truth).mean())
            err["weekly_profile"].append(np.abs(prof - truth).mean())
            err["hybrid (ours)"].append(err["linear" if gap <= result.config.max_interp_steps else "weekly_profile"][-1])
        for m, e in err.items():
            rows.append({"gap_steps": gap, "gap_minutes": gap * 5, "method": m, "MAE": float(np.mean(e)), "n": len(e)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- persistence
def save_clean(result: CleanResult, index: pd.DatetimeIndex, sensor_ids: list[str],
               path: Path | str = CLEAN_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("speed", data=result.speed, compression="gzip", compression_opts=4)
        f.create_dataset("issue", data=result.issue, compression="gzip", compression_opts=4)
        f.create_dataset("filled", data=result.filled, compression="gzip", compression_opts=4)
        f.create_dataset("timestamp", data=index.asi8)
        f.create_dataset("sensor_id", data=np.array(sensor_ids, dtype="S"))
        f.attrs["issue_codes"] = str(ISSUES)
        f.attrs["fill_codes"] = str(FILLS)
        f.attrs["description"] = "METR-LA speeds (mph) after validation, cleaning and imputation"
    return path


def load_clean(path: Path | str = CLEAN_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Cleaned speeds as a (time x sensor) DataFrame, plus the issue and fill code arrays."""
    with h5py.File(path, "r") as f:
        index = pd.to_datetime(f["timestamp"][:])
        cols = [c.decode() for c in f["sensor_id"][:]]
        df = pd.DataFrame(f["speed"][:], index=index, columns=cols)
        return df, f["issue"][:], f["filled"][:]
