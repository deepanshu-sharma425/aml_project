"""Tests for the Phase 2 models: graph attention, ST-GAT, the hybrid and the
per-segment baseline. They pin down the properties the report relies on."""
import copy
import dataclasses

import numpy as np
import pytest
import torch

from traffic.models.segment import SegmentFeatures, blocked_folds
from traffic.models.stgat import STGAT, GraphAttention, neighbour_mask
from traffic.trainer import WindowSampler, input_features


def small_adj(n=6):
    """Directed chain 0 -> 1 -> ... -> n-1."""
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        a[i, i + 1] = 1.0
    return a


# ---------------------------------------------------------------- attention
def test_gat_matches_pytorch_geometric(data):
    """With one time step, our layer is exactly GATConv (Velickovic et al., 2018)."""
    pyg = pytest.importorskip("torch_geometric.nn")
    torch.manual_seed(0)
    mask = neighbour_mask(data.adj, "down")
    ours = GraphAttention(8, 5, 3, mask).eval()
    ref = pyg.GATConv(8, 5, heads=3, add_self_loops=False).eval()
    with torch.no_grad():
        ref.lin.weight.copy_(ours.lin.weight)
        ref.att_src.copy_(ours.att_src.view(1, 3, 5))
        ref.att_dst.copy_(ours.att_dst.view(1, 3, 5))
        ref.bias.zero_()
    tgt, src = torch.nonzero(mask, as_tuple=True)
    x = torch.randn(data.num_nodes, 8)
    assert torch.allclose(ours(x[None, None])[0, 0], ref(x, torch.stack([src, tgt])), atol=1e-5)


def test_attention_respects_road_graph(data):
    torch.manual_seed(0)
    for direction in ("down", "up"):
        mask = neighbour_mask(data.adj, direction)
        layer = GraphAttention(4, 2, 2, mask).eval()
        _, alpha = layer(torch.randn(3, 5, data.num_nodes, 4), return_attention=True)
        assert torch.allclose(alpha.sum(-1), torch.ones(3, 2, data.num_nodes), atol=1e-5)
        assert (alpha[..., ~mask] == 0).all()
    # 'up' is the transpose of 'down' apart from the shared self-loops
    assert torch.equal(neighbour_mask(data.adj, "up"), neighbour_mask(data.adj, "down").T)


def test_mean_mode_is_uniform():
    mask = neighbour_mask(small_adj(), "down")
    layer = GraphAttention(4, 2, 2, mask, mode="mean")
    _, alpha = layer(torch.randn(1, 2, 6, 4), return_attention=True)
    assert torch.allclose(alpha[0, 0, 0], torch.tensor([0.5, 0.5, 0, 0, 0, 0]))


# ---------------------------------------------------------------- ST-GAT
@pytest.mark.parametrize("spatial", ["attention", "mean", "none", "dense"])
def test_stgat_shapes(spatial):
    m = STGAT(6, small_adj(), in_dim=6, spatial=spatial, d=16, heads=2)
    assert m(torch.randn(3, 12, 6, 6)).shape == (3, 12, 6)


def test_stgat_without_graph_keeps_sensors_independent():
    torch.manual_seed(0)
    m = STGAT(6, small_adj(), in_dim=6, spatial="none", d=16, heads=2).eval()
    x = torch.randn(1, 12, 6, 6)
    x2 = x.clone()
    x2[:, :, 0] += 5
    assert torch.allclose(m(x)[..., 1:], m(x2)[..., 1:], atol=1e-5)


def test_stgat_attention_propagates_upstream_and_downstream():
    """On the chain 0->1->...->5, sensor 2 must react to its neighbours 1 and 3
    but not to sensor 5, which is three hops away (2 blocks = 2 hops)."""
    torch.manual_seed(0)
    m = STGAT(6, small_adj(), in_dim=6, spatial="attention", d=16, heads=2).eval()
    x = torch.randn(1, 12, 6, 6)
    for j, should_change in ((1, True), (3, True), (5, False)):
        x2 = x.clone()
        x2[:, :, j] += 5
        changed = not torch.allclose(m(x)[..., 2], m(x2)[..., 2], atol=1e-6)
        assert changed == should_change, j


def test_hybrid_starts_at_base_forecast():
    m = STGAT(6, small_adj(), in_dim=6, d=16, heads=2, base_dim=12, residual=True).eval()
    base = torch.randn(2, 12, 6)
    assert torch.allclose(m(torch.randn(2, 12, 6, 6), base), base)


def test_attention_maps(data):
    m = STGAT(data.num_nodes, data.adj, in_dim=6, d=16, heads=2).eval()
    maps = m.attention_maps(torch.randn(2, 12, data.num_nodes, 6))
    assert len(maps) == 2 and maps[0].shape == (2, 2, data.num_nodes, data.num_nodes)


# ---------------------------------------------------------------- inputs
def test_masked_inputs(data):
    x = input_features(data, "masked")
    assert x.shape == (len(data.index), data.num_nodes, 6)
    observed = data.speed > 0
    assert np.array_equal(x[..., 1], observed.astype(np.float32))
    # an outage is not shown as 0 mph: it carries the last observed reading
    from traffic.features import forward_fill
    filled = data.scaler.transform(forward_fill(data.speed))
    assert np.allclose(x[..., 0], filled)
    assert not np.isclose(x[..., 0][~observed], data.scaler.transform(0.0)).any()
    assert np.abs(x[..., 2:]).max() <= 1.0 + 1e-6


def test_masked_inputs_are_causal(data):
    t = data.split.val[0]
    speed = data.speed.copy()
    speed[t + 1:] = 7.0
    d2 = dataclasses.replace(data, speed=speed)
    assert np.allclose(input_features(data, "masked")[: t + 1], input_features(d2, "masked")[: t + 1])


def test_sampler_returns_base(data):
    base = np.full((len(data.index), 12, data.num_nodes), data.scaler.mean, dtype=np.float32)
    s = WindowSampler(data, torch.device("cpu"), "masked", base)
    x, y, b = s.batch(data.split.test[:4])
    assert x.shape == (4, 12, data.num_nodes, 6) and y.shape == b.shape == (4, 12, data.num_nodes)
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-5)          # scaler mean -> 0


# ---------------------------------------------------------------- segment baseline
def test_segment_features_causal_and_independent(data, fb):
    sf = SegmentFeatures(fb)
    a = data.split.val[:4]
    x = sf.build(a, 10)
    assert x.shape == (4, len(sf.names)) and not np.isnan(x).any()
    fb2 = copy.copy(fb)
    fb2.filled, fb2.raw, fb2.anomaly = fb.filled.copy(), fb.raw.copy(), fb.anomaly.copy()
    for arr in (fb2.filled, fb2.raw, fb2.anomaly):
        arr[a.max() + 1:] = -123.0          # the future
        arr[:, 11] = -55.0                  # another sensor
    assert np.allclose(x, SegmentFeatures(fb2).build(a, 10))


def test_blocked_folds_purge_and_cover(data):
    anchors = data.split.train
    folds = blocked_folds(anchors, k=5, purge=23)
    held = np.concatenate([h for _, h in folds])
    assert np.array_equal(np.sort(held), anchors)
    for train, h in folds:
        gap = np.abs(train[:, None] - h[None, [0, -1]]).min()
        assert gap > 23 and not np.intersect1d(train, h).size


# ------------------------------------------------------------- road graph
def test_graph_stats_small_chain():
    from traffic.graph import graph_stats
    s = graph_stats(small_adj(6) + np.eye(6, dtype=np.float32))     # self-loops are ignored
    assert s["directed_edges"] == 5 and s["one_way_fraction"] == 1.0
    assert s["nodes_without_downstream"] == 1 and s["nodes_without_upstream"] == 1
    assert s["weakly_connected_components"] == 1 and s["strongly_connected_components"] == 6
    assert np.isclose(s["mean_2hop_neighbours"], (2 + 3 + 4 + 4 + 3 + 2) / 6)


def test_graph_stats_metrla_match_report(data):
    """Numbers quoted in the report (Sections on ST-GAT and the data)."""
    from traffic.data import distance_matrix
    from traffic.graph import graph_stats
    s = graph_stats(data.adj, distance_matrix(data.sensor_ids))
    assert s["nodes"] == 207 and s["directed_edges"] == 1515
    assert round(100 * s["one_way_fraction"]) == 73
    assert round(s["mean_2hop_neighbours"]) == 36
    assert round(s["median_edge_km"], 1) == 2.8
