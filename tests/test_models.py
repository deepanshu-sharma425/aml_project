import numpy as np
import torch
from sklearn.linear_model import Ridge

from traffic.data import transition_matrices
from traffic.models import GraphWaveNet, HistoricalAverage, LSTMForecaster, PerHorizonRegressor, Persistence
from traffic.trainer import masked_mae_loss


def test_baselines_shapes(data, fb):
    a = data.split.test[:4]
    assert HistoricalAverage(fb).predict(a).shape == (4, 12, 207)
    p = Persistence(fb).predict(a)
    assert np.allclose(p[:, 0], p[:, -1])


def test_lstm_shape():
    out = LSTMForecaster()(torch.randn(2, 12, 5, 2))
    assert out.shape == (2, 12, 5)


def test_gwnet_variants_shape(data):
    sup = [torch.tensor(s[:6, :6]) for s in transition_matrices(data.adj)]
    for kw in [dict(), dict(adaptive=False), dict(use_graph=False)]:
        m = GraphWaveNet(6, sup, **kw)
        assert m.receptive_field == 13
        assert m(torch.randn(3, 12, 6, 2)).shape == (3, 12, 6)
    adj = GraphWaveNet(6, [], adaptive=True).adaptive_adj()
    assert torch.allclose(adj.sum(1), torch.ones(6))


def test_no_graph_variant_keeps_nodes_independent():
    """Without graph convolution, perturbing sensor 0 cannot change sensor 1's forecast."""
    torch.manual_seed(0)
    m = GraphWaveNet(4, [], adaptive=False, use_graph=False).eval()
    x = torch.randn(1, 12, 4, 2)
    x2 = x.clone()
    x2[:, :, 0] += 5
    assert torch.allclose(m(x)[:, :, 1], m(x2)[:, :, 1], atol=1e-5)


def test_masked_loss_ignores_missing():
    t = torch.tensor([0.0, 10.0])
    p = torch.tensor([100.0, 12.0])
    assert torch.isclose(masked_mae_loss(p, t), torch.tensor(2.0))


def test_per_horizon_regressor(data, fb):
    m = PerHorizonRegressor(Ridge(), fb, horizons=[1, 2]).fit(data.split.train[:50], verbose=False)
    assert m.predict(data.split.test[:3]).shape == (3, 2, 207)
