# Literature Review: Short-Term Traffic Speed Forecasting

This review places our METR-LA experiments in the context of four decades of
traffic forecasting research. It is organised chronologically by modelling
paradigm and ends with the gaps that motivate our design choices.

## 1. Problem setting

Short-term traffic forecasting predicts a traffic variable (speed, flow or
occupancy) a few minutes to an hour ahead from recent sensor history. It feeds
Intelligent Transportation Systems: route guidance, ramp metering, incident
detection and travel-time estimation. Vlahogianni et al. [1] survey the field
and identify its persistent challenges: non-linearity near congestion,
non-stationarity, spatial propagation of congestion along the network, and
missing or faulty data. All four are visible in METR-LA (see the EDA).

## 2. Statistical time-series models (1979 to about 2010)

| Work | Method | Key idea / finding |
|---|---|---|
| Ahmed & Cook, 1979 [2] | Box-Jenkins ARIMA | First application of ARIMA to freeway data; ARIMA(0,1,3) beat moving-average and double-exponential smoothing. |
| Williams & Hoel, 2003 [3] | Seasonal ARIMA | Gave a theoretical argument that weekly-differenced traffic is stationary, which justifies seasonal models. |
| Smith & Demetsky, 1997 [4] | k-NN (non-parametric regression) vs. ARIMA vs. neural nets | Non-parametric regression was competitive and more robust to unusual conditions. |

**Limitations.** These are univariate, per-sensor, mostly linear models. They
ignore the road network, react slowly to abrupt congestion and struggle beyond
about 15 minutes. In the DCRNN benchmark [5], ARIMA's METR-LA MAE grows from
3.99 (15 min) to 6.90 (60 min). The Historical Average baseline, which simply
predicts the usual speed for that time of week, is used as the seasonal
reference in every later paper and is our first baseline.

## 3. Classical machine learning (about 2000 to 2015)

Support vector regression, k-NN, random forests and gradient boosting
(Friedman [6]; histogram-based variants such as LightGBM [7]) model non-linear
relations between hand-crafted features and the future value. Their accuracy
depends on **feature engineering**: lags, rolling statistics, calendar
encodings and, less often, neighbouring-sensor readings. In the DCRNN benchmark,
SVR (MAE 6.72 at 60 min) is not much better than ARIMA because it used only
raw lags. Part of our project tests the converse hypothesis: with careful,
leakage-free features (weekly profile, anomaly and upstream/downstream
aggregates), a gradient-boosted tree model can close much of the gap to deep
models.

## 4. Deep sequence models (2015 to 2017)

* **Stacked autoencoders**: Lv et al. [8] were among the first to apply deep
  learning to large-scale traffic flow prediction (PeMS) and outperformed
  shallow models.
* **LSTM**: Ma et al. [9] used LSTM [10] for speed prediction from microwave
  detectors in Beijing and showed that its gating captures long temporal
  dependencies better than RNNs and statistical models.
* **Seq2seq / FC-LSTM** [11]: encoder-decoder LSTMs predict an entire
  multi-step horizon. In [5], FC-LSTM reaches MAE 3.44 / 3.77 / 4.37 at
  15/30/60 min on METR-LA.

**Limitation.** A fully connected LSTM over all sensors either ignores space
(shared per-sensor model) or flattens it into a long vector, which discards
road topology and does not scale with the number of sensors.

## 5. Spatio-temporal graph neural networks (2017 onwards)

Graph convolution generalises CNNs to irregular graphs: spectral ChebNet
(Defferrard et al. [12]) and its first-order simplification GCN (Kipf & Welling
[13]). A road network is a natural graph, which led to a rapid line of work:

| Model | Venue | Spatial module | Temporal module | METR-LA MAE @60 min |
|---|---|---|---|---|
| **DCRNN** [5] | ICLR 2018 | Bidirectional diffusion convolution (random walks on a directed distance graph) | GRU seq2seq with scheduled sampling | 3.60 |
| **STGCN** [14] | IJCAI 2018 | Chebyshev graph convolution | Gated 1-D CNN (fully convolutional) | 4.59* |
| **ASTGCN** [15] | AAAI 2019 | Spatial attention + Chebyshev GCN | Temporal attention + conv, with recent/daily/weekly branches | (evaluated on PeMS) |
| **Graph WaveNet** [16] | IJCAI 2019 | Diffusion convolution + **self-adaptive adjacency** learned from node embeddings | Stacked dilated causal gated convolutions (WaveNet [17]) | 3.53 |
| **GMAN** [18] | AAAI 2020 | Spatial attention | Temporal attention + transform attention | 3.44 |
| **MTGNN** [19] | KDD 2020 | Learned graph (no prior needed) + mix-hop propagation | Dilated inception convolution | 3.49 |
| **AGCRN** [20] | NeurIPS 2020 | Node-adaptive parameters + data-adaptive graph | GRU | (evaluated on PeMS) |
| **STID** [21] | CIKM 2022 | None (spatial / time-of-day / day-of-week *identity embeddings* + MLP) | MLP | 3.55 |
| **STAEformer** [22] | CIKM 2023 | Spatio-temporal adaptive embedding + vanilla Transformer | Transformer | 3.34 |

\*STGCN as re-evaluated in [16].

The main points:

1. **Graph structure matters.** DCRNN's diffusion convolution models traffic
   flowing in both directions of a directed road graph, and its MAE beats
   FC-LSTM's by about 0.8 mph at 60 min.
2. **The predefined graph is incomplete.** Graph WaveNet showed that learning
   an adaptive adjacency A = softmax(ReLU(E1 E2^T)) recovers dependencies the
   distance graph misses, and that the model still works with *no* prior graph.
   MTGNN and AGCRN built on this idea.
3. **Recurrence is not required.** Convolutional (STGCN, Graph WaveNet) and
   attention models (GMAN) train in parallel and avoid error accumulation.
4. **Simple can be strong.** STID showed that identity embeddings (which
   sensor, what time, which weekday) recover much of the gain. This supports
   our finding that calendar and weekly-profile features dominate long-horizon
   gradient-boosting importances.

Jiang & Luo [23] give a comprehensive survey of graph neural networks for
traffic forecasting.

## 6. Evaluation conventions we follow

Following [5], [16] and later work, we use:

* METR-LA: 207 sensors, 1 Mar to 30 Jun 2012 (the released file ends on 27 Jun), 5-minute speeds;
* a chronological 70 / 10 / 20 train / validation / test split;
* 12 input steps (1 h) predicting 12 output steps (1 h);
* MAE, RMSE and MAPE at 15, 30 and 60 min, **excluding zero (missing)
  targets**;
* a thresholded Gaussian-kernel adjacency built from road-network distances
  (threshold 0.1).

Following one shared protocol makes our numbers directly comparable with the
published ones.

## 7. Research gaps addressed by this project

1. **Fair classical baselines.** Published comparisons often pit deep models
   against feature-poor classical baselines. We build a leakage-tested feature
   set and report a strong gradient-boosting baseline.
2. **Statistical significance.** Few papers test whether differences between
   models are significant. We add Diebold-Mariano tests [24] with Newey-West
   variance and day-level block-bootstrap confidence intervals.
3. **Where the graph helps.** We ablate Graph WaveNet's spatial component
   (none / road graph / adaptive / both) under an equal training budget, and
   analyse errors by time of day, by sensor and on windows with missing input.

## References

[1] E. I. Vlahogianni, M. G. Karlaftis, and J. C. Golias, "Short-term traffic forecasting: Where we are and where we're going," *Transportation Research Part C*, vol. 43, 2014.
[2] M. S. Ahmed and A. R. Cook, "Analysis of freeway traffic time-series data by using Box-Jenkins techniques," *Transportation Research Record*, no. 722, 1979.
[3] B. M. Williams and L. A. Hoel, "Modeling and forecasting vehicular traffic flow as a seasonal ARIMA process: Theoretical basis and empirical results," *J. Transportation Engineering*, vol. 129, no. 6, 2003.
[4] B. L. Smith and M. J. Demetsky, "Traffic flow forecasting: Comparison of modeling approaches," *J. Transportation Engineering*, vol. 123, no. 4, 1997.
[5] Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in *Proc. ICLR*, 2018.
[6] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, 2001.
[7] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, 2017.
[8] Y. Lv, Y. Duan, W. Kang, Z. Li, and F.-Y. Wang, "Traffic flow prediction with big data: A deep learning approach," *IEEE Trans. Intelligent Transportation Systems*, vol. 16, no. 2, 2015.
[9] X. Ma, Z. Tao, Y. Wang, H. Yu, and Y. Wang, "Long short-term memory neural network for traffic speed prediction using remote microwave sensor data," *Transportation Research Part C*, vol. 54, 2015.
[10] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, 1997.
[11] I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to sequence learning with neural networks," in *Proc. NeurIPS*, 2014.
[12] M. Defferrard, X. Bresson, and P. Vandergheynst, "Convolutional neural networks on graphs with fast localized spectral filtering," in *Proc. NeurIPS*, 2016.
[13] T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, 2017.
[14] B. Yu, H. Yin, and Z. Zhu, "Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting," in *Proc. IJCAI*, 2018.
[15] S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, "Attention based spatial-temporal graph convolutional networks for traffic flow forecasting," in *Proc. AAAI*, 2019.
[16] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, "Graph WaveNet for deep spatial-temporal graph modeling," in *Proc. IJCAI*, 2019.
[17] A. van den Oord et al., "WaveNet: A generative model for raw audio," arXiv:1609.03499, 2016.
[18] C. Zheng, X. Fan, C. Wang, and J. Qi, "GMAN: A graph multi-attention network for traffic prediction," in *Proc. AAAI*, 2020.
[19] Z. Wu, S. Pan, G. Long, J. Jiang, X. Chang, and C. Zhang, "Connecting the dots: Multivariate time series forecasting with graph neural networks," in *Proc. KDD*, 2020.
[20] L. Bai, L. Yao, C. Li, X. Wang, and C. Wang, "Adaptive graph convolutional recurrent network for traffic forecasting," in *Proc. NeurIPS*, 2020.
[21] Z. Shao, Z. Zhang, F. Wang, W. Wei, and Y. Xu, "Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting," in *Proc. CIKM*, 2022.
[22] H. Liu et al., "Spatio-temporal adaptive embedding makes vanilla Transformer SOTA for spatio-temporal forecasting," in *Proc. CIKM*, 2023.
[23] W. Jiang and J. Luo, "Graph neural network for traffic forecasting: A survey," *Expert Systems with Applications*, vol. 207, 2022.
[24] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *J. Business & Economic Statistics*, vol. 13, no. 3, 1995.
