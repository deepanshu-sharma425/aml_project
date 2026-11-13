"""Road-graph statistics (NetworkX) quoted in the report.

``adj[i, j] > 0`` is a directed road path from sensor i to sensor j (j downstream),
as built by :func:`traffic.data.gaussian_adjacency`. Self-loops are ignored.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def road_graph(adj: np.ndarray) -> nx.DiGraph:
    """Directed NetworkX graph of the thresholded adjacency, edge weight = W_ij."""
    a = np.array(adj, dtype=float)
    np.fill_diagonal(a, 0.0)
    return nx.from_numpy_array(a, create_using=nx.DiGraph)


def graph_stats(adj: np.ndarray, dist: np.ndarray | None = None, hops: int = 2) -> dict:
    """Size, direction, connectivity and receptive-field statistics of the road graph.

    ``dist`` (metres, optional) adds the median road length of an edge.
    ``hops`` sets the neighbourhood radius reported as ``mean_{hops}hop_neighbours``
    (undirected), i.e. how many sensors a ``hops``-layer GNN can see.
    """
    g = road_graph(adj)
    u = g.to_undirected()
    e = np.array(nx.to_numpy_array(g, weight=None) > 0)
    reach = [len(nx.single_source_shortest_path_length(u, i, cutoff=hops)) - 1 for i in u]
    out = {
        "nodes": g.number_of_nodes(),
        "directed_edges": g.number_of_edges(),
        "one_way_fraction": float((e & ~e.T).sum() / max(e.sum(), 1)),
        "mean_out_degree": float(e.sum(1).mean()),
        "nodes_without_downstream": int((e.sum(1) == 0).sum()),
        "nodes_without_upstream": int((e.sum(0) == 0).sum()),
        "isolated_nodes": int(sum(1 for n in u if u.degree(n) == 0)),
        "weakly_connected_components": nx.number_weakly_connected_components(g),
        "strongly_connected_components": nx.number_strongly_connected_components(g),
        f"mean_{hops}hop_neighbours": float(np.mean(reach)),
    }
    if dist is not None:
        out["median_edge_km"] = float(np.median(np.asarray(dist)[e]) / 1000)
    return out
