import numpy as np

from traffic.features import FEATURE_GROUPS, forward_fill, historical_profile


def test_forward_fill_is_causal():
    s = np.array([[50.0], [0.0], [0.0], [40.0]])
    assert forward_fill(s)[:, 0].tolist() == [50, 50, 50, 40]


def test_profile_ignores_missing():
    s = np.array([[10.0], [0.0], [30.0], [0.0]])
    tow = np.array([0, 0, 1, 1])
    prof = historical_profile(s, tow, end=4)
    assert prof[0, 0] == 10 and prof[1, 0] == 30


def test_profile_uses_training_period_only(data, fb):
    # Corrupt the test period: the profile must not change.
    from traffic.features import time_of_week
    s = data.speed.copy()
    s[data.train_end:] = 1.0
    prof = historical_profile(s, time_of_week(data.index), data.train_end)
    assert np.allclose(prof, fb.profile)


def test_build_shapes_no_nans(data, fb):
    a = data.split.test[:5]
    x, names = fb.build(a, 6)
    assert x.shape == (5 * data.num_nodes, len(names))
    assert len(set(names)) == len(names)
    assert not np.isnan(x).any()
    # anchor-major ordering: first block of rows is anchor 0
    assert np.allclose(x[: data.num_nodes, names.index("lag_0")], fb.filled[a[0]])


def test_no_future_leakage(data, fb):
    """Changing values after the anchor must not change any feature."""
    import copy
    a = data.split.val[:3]
    x_ref, _ = fb.build(a, 12)
    fb2 = copy.copy(fb)
    fb2.filled = fb.filled.copy()
    fb2.raw = fb.raw.copy()
    fb2.anomaly = fb.anomaly.copy()
    for arr in (fb2.filled, fb2.raw, fb2.anomaly):
        arr[a.max() + 1:] = -123.0
    x_new, _ = fb2.build(a, 12)
    assert np.allclose(x_ref, x_new)


def test_groups_subset(fb, data):
    x, names = fb.build(data.split.val[:2], 3, ("lags",))
    assert names == [f"lag_{k}" for k in range(12)]
    assert set(FEATURE_GROUPS) >= {"lags", "spatial"}
