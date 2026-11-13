"""Masked error metrics and a Diebold-Mariano significance test.

Targets equal to 0 are missing sensor readings, so they are excluded from
every metric (the convention of DCRNN, Graph WaveNet and follow-up work).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

HORIZONS = {"15 min": 3, "30 min": 6, "60 min": 12}


def _mask(y_true: np.ndarray) -> np.ndarray:
    return y_true > 0


def masked_mae(y_pred, y_true) -> float:
    m = _mask(y_true)
    return float(np.abs(y_pred[m] - y_true[m]).mean())


def masked_rmse(y_pred, y_true) -> float:
    m = _mask(y_true)
    return float(np.sqrt(np.square(y_pred[m] - y_true[m]).mean()))


def masked_mape(y_pred, y_true) -> float:
    """Mean absolute percentage error, in percent."""
    m = _mask(y_true)
    return float((np.abs(y_pred[m] - y_true[m]) / y_true[m]).mean() * 100)


def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, dict[str, float]]:
    """Metrics at the standard 15/30/60-minute horizons and averaged over all 12.

    Both arrays have shape (samples, horizon, nodes) in mph.
    """
    out = {}
    for name, h in HORIZONS.items():
        p, t = y_pred[:, h - 1], y_true[:, h - 1]
        out[name] = {"MAE": masked_mae(p, t), "RMSE": masked_rmse(p, t), "MAPE": masked_mape(p, t)}
    out["avg"] = {"MAE": masked_mae(y_pred, y_true), "RMSE": masked_rmse(y_pred, y_true),
                  "MAPE": masked_mape(y_pred, y_true)}
    return out


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, lag: int = 12) -> tuple[float, float]:
    """Two-sided Diebold-Mariano test on absolute-error loss.

    ``err_a`` and ``err_b`` are per-time-step losses (already averaged over
    sensors), ordered in time. The long-run variance uses a Newey-West
    estimator with ``lag`` autocovariances, since consecutive forecast
    windows overlap. A negative statistic means model A is more accurate.
    """
    d = np.asarray(err_a, dtype=np.float64) - np.asarray(err_b, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / np.sqrt(lrv / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def per_step_abs_error(y_pred: np.ndarray, y_true: np.ndarray, h: int) -> np.ndarray:
    """Masked MAE over sensors for each test window at horizon ``h`` (1-based).

    Windows with no observed target are NaN; the mask depends only on
    ``y_true`` so the series of two models stay aligned.
    """
    p, t = y_pred[:, h - 1], y_true[:, h - 1]
    m = _mask(t)
    err = np.where(m, np.abs(p - t), 0.0).sum(1)
    cnt = m.sum(1)
    return np.where(cnt > 0, err / np.maximum(cnt, 1), np.nan)


def subset_metrics(y_pred: np.ndarray, y_true: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / MAPE over observed targets where ``keep`` is True (broadcastable)."""
    m = _mask(y_true) & np.broadcast_to(keep, y_true.shape)
    e = y_pred[m] - y_true[m]
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt(np.square(e).mean())),
            "MAPE": float((np.abs(e) / y_true[m]).mean() * 100), "n": int(m.sum())}


def is_peak(hour: np.ndarray, dayofweek: np.ndarray) -> np.ndarray:
    """Weekday AM (06:00-10:00) or PM (15:00-19:00) peak, from target-time hour and weekday."""
    h = np.asarray(hour)
    return (np.asarray(dayofweek) < 5) & (((h >= 6) & (h < 10)) | ((h >= 15) & (h < 19)))


def block_bootstrap_ci(err_sum: np.ndarray, cnt: np.ndarray, block: int = 288, n_boot: int = 1000,
                       seed: int = 0) -> np.ndarray:
    """95% CI of an overall MAE by resampling whole days (keeps serial dependence).

    ``err_sum`` / ``cnt``: per-window sum of absolute errors and number of
    observed targets, in time order.
    """
    rng = np.random.default_rng(seed)
    nb = len(err_sum) // block
    es = err_sum[: nb * block].reshape(nb, block).sum(1)
    cs = cnt[: nb * block].reshape(nb, block).sum(1)
    idx = rng.integers(0, nb, (n_boot, nb))
    return np.percentile(es[idx].sum(1) / cs[idx].sum(1), [2.5, 97.5])
