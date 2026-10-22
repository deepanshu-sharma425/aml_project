import numpy as np

from traffic.data import STEPS_PER_DAY, gaussian_adjacency, make_split, transition_matrices, windows


def test_shapes_and_order(data):
    assert data.speed.shape == (34272, 207)
    assert data.index[0].isoformat() == "2012-03-01T00:00:00"
    assert (np.diff(data.index.asi8) == 5 * 60 * 10**9).all()
    assert data.adj.shape == (207, 207)


def test_split_is_chronological_and_disjoint(data):
    s = data.split
    assert len(s.train) + len(s.val) + len(s.test) == 34272 - 12 - 12 + 1
    assert s.train[-1] < s.val[0] and s.val[-1] < s.test[0]
    # training targets never reach into the validation inputs' future
    assert data.train_end <= s.val[0] + data.seq_out + 1


def test_split_sizes_match_literature():
    s = make_split(34272)
    assert (len(s.train), len(s.val), len(s.test)) == (23974, 3425, 6850)


def test_scaler_uses_train_only(data):
    obs = data.speed[: data.train_end]
    obs = obs[obs > 0]
    assert np.isclose(data.scaler.mean, obs.mean(), atol=1e-3)
    z = data.scaler.transform(np.array([data.scaler.mean]))
    assert np.allclose(z, 0)
    assert np.allclose(data.scaler.inverse_transform(data.scaler.transform(np.array([42.0]))), 42.0)


def test_adjacency_properties(data):
    a = data.adj
    assert (a >= 0).all() and (a <= 1).all()
    assert np.allclose(np.diag(a), 1.0)                 # d_ii = 0 -> weight 1
    assert int((a > 0).sum()) == 1722                   # matches DCRNN preprocessing
    for p in transition_matrices(a):
        assert np.allclose(p.sum(1), 1.0)


def test_gaussian_kernel_threshold():
    d = np.array([[0.0, 1.0], [np.inf, 0.0]])
    w = gaussian_adjacency(d, threshold=0.1)
    assert w[1, 0] == 0 and w[0, 0] == 1


def test_windows_alignment():
    v = np.arange(100)
    x, y = windows(v, np.array([11, 50]))
    assert x[0].tolist() == list(range(0, 12)) and y[0].tolist() == list(range(12, 24))
    assert x[1, -1] == 50 and y[1, 0] == 51


def test_steps_per_day(data):
    assert (data.index[STEPS_PER_DAY] - data.index[0]).days == 1
