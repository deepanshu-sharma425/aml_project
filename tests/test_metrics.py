import numpy as np
import pytest

from traffic.metrics import diebold_mariano, evaluate, masked_mae, masked_mape, masked_rmse


def test_masking_ignores_zero_targets():
    y = np.array([0.0, 50.0, 60.0])
    p = np.array([999.0, 55.0, 60.0])
    assert masked_mae(p, y) == pytest.approx(2.5)
    assert masked_rmse(p, y) == pytest.approx(np.sqrt(12.5))
    assert masked_mape(p, y) == pytest.approx(5.0)


def test_evaluate_horizons():
    y = np.full((4, 12, 3), 60.0)
    p = y.copy()
    p[:, 2] += 3                          # error only at step 3 (15 min)
    m = evaluate(p, y)
    assert m["15 min"]["MAE"] == pytest.approx(3)
    assert m["30 min"]["MAE"] == 0 and m["60 min"]["MAE"] == 0
    assert m["avg"]["MAE"] == pytest.approx(3 / 12)


def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    base = rng.random(2000)
    better = base * 0.8
    stat, p = diebold_mariano(better, base)
    assert stat < 0 and p < 1e-6
    stat, p = diebold_mariano(base, base + rng.normal(0, 1e-3, 2000))
    assert p > 0.01


def test_peak_definition():
    from traffic.metrics import is_peak
    hour = np.array([5, 6, 9, 10, 15, 18, 19, 17])
    dow = np.array([0, 0, 0, 0, 0, 0, 0, 6])          # last one is a Sunday
    assert is_peak(hour, dow).tolist() == [False, True, True, False, True, True, False, False]


def test_subset_metrics_masks_missing():
    from traffic.metrics import subset_metrics
    y = np.array([[0.0, 50.0, 40.0]])
    p = np.array([[9.0, 52.0, 30.0]])
    s = subset_metrics(p, y, np.array([[True, True, False]]))
    assert s["n"] == 1 and np.isclose(s["MAE"], 2.0)
