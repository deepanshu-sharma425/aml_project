# Viva preparation: Phase 2 (deep learning and hybrid)

Short answers to the questions examiners are most likely to ask, with pointers to
the code. Numbers come from `results/tables/p2_*` (see the results summary at the end).

## 1. The problem and the "gap"

**What exactly is the gap you are filling?**
An independent per-segment model `f_i` only sees sensor *i*'s own history. If the
error `r_i = y_i - f_i(own history)` can be predicted from *other* sensors at forecast
time, the independent model leaves accuracy on the table. By the law of total
variance, the extra MSE is `E[(E[y|network] - E[y|own])^2] >= 0`, which is zero only
when the network adds nothing (report Eq. "gap").

**How did you show the gap exists, without peeking at the test set?**
On *out-of-fold training* residuals of the XGBoost per-segment models
(`scripts/evaluate_phase2.py::gap_analysis`):
1. errors of neighbouring segments are correlated (and the correlation decays with road distance);
2. the error correlates with the *downstream / upstream anomaly at time t*, information
   the per-segment model never received, while it is essentially uncorrelated with the
   segment's own anomaly (which the model did use).

**Why speed only? The brief mentions flow and occupancy.**
METR-LA only publishes speed. We say so explicitly; PeMS-Bay has the same limitation,
while full PeMS (raw station files) would provide flow and occupancy for an extension.

## 2. Architecture logic (ST-GAT, `src/traffic/models/stgat.py`)

**Walk me through the architecture.**
Input (12 steps x 207 sensors x 6 channels) + 16-d node embedding -> 1x1 conv to 64
channels -> 2 x [gated temporal conv (k=3) -> directional GAT -> gated temporal conv ->
LayerNorm -> dropout] -> gated conv collapsing the remaining 4 steps -> MLP -> 12 horizons.

**Why GAT and not a GCN / diffusion convolution?**
GCN/diffusion use fixed weights from road distance; GAT computes weights from the
current traffic state, so a congested neighbour can receive more weight. With zero
attention vectors, GAT reduces to the mean aggregator, so it is strictly more general.
The "mean aggregation" ablation measures exactly this difference.

**Why two attention directions?**
The graph is directed (73 % of edges one-way). Queues spill *upstream*; free-flow
recovery travels *downstream*. Separate heads over downstream and upstream neighbours
let the model treat the two differently, the same reason DCRNN uses forward and
backward diffusion.

**Why is attention shared across the time steps of a window?**
Engineering and precedent. Window-level coefficients turn each head into one masked N x N
matmul and need T = 12 times fewer attention scores than per-step attention (H*N^2 instead
of T*H*N^2). The price: weights cannot change within a one-hour window.
ASTGCN shares spatial attention across the window in the same way. With T = 1 the layer
equals standard GAT, which `tests/test_phase2.py::test_gat_matches_pytorch_geometric`
checks against PyTorch Geometric's `GATConv`.

**Receptive field?** Temporal: all 12 inputs. Spatial: 2 hops per direction (2 blocks),
plus one more in the hybrid's fusion layer. Two undirected hops reach about 36 sensors on average.

**Why the fixed adjacency (and not a learned one like Graph WaveNet)?**
The brief asks for a fixed graph, and it is a strong, interpretable prior. The "dense"
ablation lets attention look at all 207 x 207 pairs (no road prior) to test whether
the prior helps.

## 3. Hybrid (`STGAT(base_dim=12, residual=True)`, `scripts/run_segment_xgb.py`)

**What is hybrid about it, and why is it justified?**
`y_hat = y_hat_XGB + g_theta(inputs, all sensors' XGB forecasts; graph)`. XGBoost does what
trees do well per segment (level, seasonality, recent trend, non-linear thresholds);
GAT does what XGBoost structurally cannot (cross-segment propagation). The gap
analysis shows the residual carries spatial structure, so a graph model is the right
tool for the residual. This is the residual-hybrid principle of Zhang (2003) and
stacking (Wolpert 1992).

**Why zero-initialise the last layer?**
Training then starts exactly at the XGBoost forecast (unit test
`test_hybrid_starts_at_base_forecast`), so the network can only improve on it from
step one and optimisation starts from a strong point (residual learning).

**What is cross-fitting and why do you need it?**
If the GAT trained on *in-sample* XGBoost forecasts it would see residuals that are
much smaller than at test time (compare `train_insample_MAE` with `train_oof_MAE` in
`results/metrics/xgb_seg.json`) and would learn to under-correct. We split training
anchors into 5 contiguous blocks, fit on 4 (after purging 23 steps around the held-out
block so no window overlaps it) and predict the 5th. Validation and test get forecasts
from the model fitted on all training data, so evaluation stays strictly causal. The
"in-sample base" ablation shows the cost of skipping this.

**Is there any leakage in the hybrid?**
- Base forecasts for test windows come from a model that never saw validation or test data;
  its early stopping uses every second validation window only.
- Every scaler, profile and statistic is fitted on training timestamps.
- Out-of-fold folds are blocked and purged.
- The deep model's early stopping uses validation; the test set is used once.

## 4. Dataset and regularisation

- **Outages:** forward-filled values plus a 0/1 mask channel instead of "0 mph" inputs (ablation:
  `stgat_rawinput_ablation`). Missing targets are masked out of the loss and metrics.
- **Cyclic time:** sin/cos of time of day and day of week, so 23:55 sits next to 00:00.
- **Scaling:** z-score fitted on observed training readings only.
- **Splits:** chronological 70/10/20, never shuffled across boundaries.
- **Regularisation:** dropout 0.2, attention dropout 0.1, LayerNorm, AdamW weight decay 1e-4,
  gradient clipping 5, LR halving on plateau (patience 3), early stopping (patience 8),
  a random half of the training windows per epoch, and node embeddings kept small (16-d).
- **Loss:** masked MAE, whose minimiser is the conditional median, which is robust to the heavy
  left tail of speeds (sudden congestion).

## 5. Ablations (`results/tables/p2_ablation_*.csv`)

All variants use the same seed and the same fixed budget (12 epochs, half the windows
per epoch). The full model is re-run under that budget, so rows are directly comparable.

| Question | Variant |
|---|---|
| Is spatial information needed? | `none`: no information crosses sensors |
| Is attention better than averaging? | `mean`: uniform weights over the same neighbours |
| Does the road-graph prior help? | `dense`: attention over all sensor pairs |
| Does outage masking help? | `rawinput`: Phase 1 zero-filled inputs |
| Does the hybrid need the graph? | hybrid with `none`: per-segment residual correction |
| Is a linear correction enough? | median (LAD) linear regression on neighbour anomalies (`hybrid_linear`) |
| Is the residual connection needed? | `--no-residual`: XGB forecast as input only |
| Does cross-fitting matter? | `xgb_insample`: in-sample base forecasts |

## 6. Technical validation

- MAE / RMSE / MAPE at all 12 horizons (`p2_horizon_metrics.csv`)
- peak (weekdays 06–10, 15–19) vs off-peak, and congested (< 40 mph) targets (`p2_regimes.csv`)
- Diebold–Mariano with Newey–West variance (`p2_diebold_mariano.csv`), because
  forecast windows overlap and errors are autocorrelated, so a plain t-test would be over-confident
- day-block bootstrap CIs (`p2_bootstrap_ci.csv`)
- 18 Phase 2 unit tests (`tests/test_phase2.py`; 52 in total): GAT equals PyG, masks
  respected, no-graph independence, multi-hop propagation, hybrid initialisation, causality
  of inputs and segment features, fold purging, road-graph statistics quoted in the report

## 7. Signal control: what we do and do not claim

An **offline replay**: at each 5-minute step, a controller per sensor switches to a
congestion plan if the expected speed 15 minutes ahead is below 40 mph. We compare
reactive (current speed), model forecasts, and an oracle on the test period. We
report the share of congestion onsets anticipated, the lead time, false activations and plan
switches per day. We claim nothing about live control or real delay savings; that
would need a calibrated micro-simulation (SUMO / CityFlow) with demand and signal
data that METR-LA does not have.

## 8. Limitations to volunteer

- One seed per configuration (compute budget: one laptop GPU). DM tests and bootstrap
  CIs quantify sampling uncertainty over test days but not seed variance.
- Ablations use a shorter budget than the main runs.
- Window-shared attention cannot change weights *within* a 1-hour window.
- GAT v1 attention is "static" (Brody et al., 2022); GATv2 is the natural next step.
- Speed only; no incidents, weather or events.

## 9. Hard questions about the results

**Your hybrid is not the best model. Why call it a success?**
It does what a hybrid should: it beats the per-segment XGBoost it builds on at every
horizon (3.20 to 3.06 mph, DM p < 0.001) and at 79 % of sensors, and it is the best of
our graph-attention models at 15 min (2.72, significantly better than ST-GAT, p = 0.001).
At 30 and 60 min the stand-alone ST-GAT is better. We report that honestly. A likely
reason is in the learning curve: the hybrid's validation MAE plateaus at about 2.87
after 25 epochs while its training MAE keeps falling (2.76 to 2.70), so the correction
network starts fitting noise in the residuals. A horizon-dependent gate between the
two forecasts is the natural fix (future work).

**Graph WaveNet beats you at 15 min and on congested targets. Why?**
Its learned dense adjacency and larger capacity (301k parameters) help most in rare,
severe events; ST-GAT is better in free flow and at 60 min overall. The differences
are small but significant both ways, and we report them in Table VI.

**The bootstrap intervals overlap. Are the gains real?**
The intervals overlap because some test days are much harder for every model. The
Diebold-Mariano test compares models on the same forecast times (paired), which
removes that shared day effect; it rejects equal accuracy with p < 0.001 against the
per-segment model at every horizon.

## Results summary (test set, 4-27 June 2012)

| Model | 15 min | 30 min | 60 min | Average MAE |
|---|---|---|---|---|
| XGBoost per segment | 2.89 | 3.26 | 3.64 | 3.20 |
| Gradient boosting (Phase 1 features) | 2.89 | 3.29 | 3.66 | 3.22 |
| Graph WaveNet | 2.72 | 3.10 | 3.59 | 3.07 |
| ST-GAT | 2.74 | 3.08 | 3.47 | 3.04 |
| Hybrid XGB to GAT | 2.72 | 3.09 | 3.52 | 3.06 |

- Gap: XGBoost error correlation between road neighbours 0.13 / 0.21 (15 / 60 min) vs
  0.02 / 0.04 for non-neighbours; |corr| with downstream anomaly 0.12 vs own anomaly 0.06.
- Peak, 60 min: XGBoost 5.19, ST-GAT 4.88 (-6 %), hybrid 4.94, Graph WaveNet 5.12.
- Latest reading missing, 60-min MAE change: Graph WaveNet +52 %, XGBoost +32 %, ST-GAT +28 %.
- Signal replay: onsets anticipated 0 % reactive, 20 % XGBoost, 30 % ST-GAT, 36 % hybrid,
  39 % Graph WaveNet; median lead 10 min.
- Road graph (NetworkX): 207 nodes, 1,515 directed edges, 73 % one-way, median edge 2.8 km,
  about 36 sensors within two hops.
