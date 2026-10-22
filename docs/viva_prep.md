# Viva Voce Preparation

Questions an examiner is likely to ask, grouped by rubric area. Each answer is
short enough to say aloud and points to where the evidence lives in the repo.
Numbers marked ⟨R⟩ come from `results/tables/` (re-check them after any re-run).

---

## A. Problem and dataset

**Q1. What problem are you solving?**
Multi-step traffic **speed forecasting**: given the last hour (12 × 5-min steps) of
speeds at 207 loop detectors in Los Angeles, predict the next hour (12 steps) at
every sensor. Inputs and outputs are both 12 × 207 matrices, plus a road graph.

**Q2. Why is this an ML problem and not just "use the average"?**
The Historical Average already gets MAE ≈ 4.19 mph. But traffic deviates from its
usual pattern (incidents, weather, demand spikes), and those deviations persist for
tens of minutes and spread along the road. Learning to forecast the *anomaly* is
where models add value.

**Q3. Describe METR-LA.**
34,272 time steps (1 Mar – 27 Jun 2012, 5-min resolution) × 207 sensors; speeds in
mph, capped at 70. It ships with sensor coordinates and pairwise **road-network
distances** (not straight-line). It was released with DCRNN (Li et al., ICLR 2018)
and is the standard benchmark for spatio-temporal forecasting.

**Q4. How are missing values represented and how did you handle them?**
As **0** (8.1 % of readings). A 0-mph reading on a freeway is a sensor dropout. We:
(1) **mask** zero targets out of every loss and metric; (2) for tabular features, use
a **causal forward-fill**: last observation carried forward, which never uses future
data; (3) for deep models, leave zeros in the input (the standard protocol), so the
network can learn to recognise them; (4) analyse robustness separately for windows
whose latest input is missing.

**Q5. Why not interpolate the missing values?**
Linear interpolation uses the *next* observation, which is future information at
prediction time, so it leaks. Forward-fill is the causal alternative.

**Q6. What did EDA tell you, and how did it change the design?**
See the table at the end of `notebooks/01_eda.ipynb`. In short: lag-1 ACF 0.86 →
lag features; weekly seasonality (ACF peaks every 24 h, highest at 168 h) →
calendar and weekly-profile features; ADF rejects a unit root → no differencing;
correlation decays with road distance, and graph neighbours have de-seasonalised
correlation 0.27 vs 0.07 for unconnected pairs → spatial features and a GNN;
left-skewed distribution → MAE loss; test split has 12 % missing vs 7 % → robustness
analysis and confidence intervals.

**Q7. Is the data stationary?**
ADF rejects a unit root at all 10 sampled sensors (p ≪ 0.01). Speed is bounded and
mean-reverting. The non-stationarity is **seasonal** (mean depends on time of week),
which we model explicitly through the weekly profile.

**Q8. Why a chronological split and not random K-fold?**
Adjacent windows share 11 of 12 input steps. A random split would put almost
identical samples in train and test, inflating scores (leakage). Real deployment
always predicts the future from the past. We use 70/10/20 by time, the same as the
literature, so numbers are comparable.

**Q9. How did you make sure there is no leakage?**
(a) Chronological split. (b) Scaler, weekly profile and sensor statistics are fit on
the training period only (`test_scaler_uses_train_only`,
`test_profile_uses_training_period_only`). (c) Every feature uses data ≤ t: a unit
test overwrites all values after the anchor and asserts the features don't change
(`test_no_future_leakage`). (d) Hyper-parameters are chosen on validation; test is used once.

---

## B. Feature engineering

**Q10. List your features and why each exists.**
34 features in 6 groups (`src/traffic/features.py`):
* **lags** (12): speed at t … t−11. Strong autocorrelation.
* **window stats** (7): first/third differences (trend), rolling mean/std/min,
  number of missing readings, missing-now flag. They capture volatility and data quality.
* **calendar** (4): sin/cos time-of-day *of the target time*, day-of-week, weekend
  flag. The target time is known in advance, so this is not leakage. Sin/cos keeps 23:55 next to 00:00.
* **weekly profile** (3): the sensor's mean training speed at the target's
  time-of-week slot, the profile now, and the **anomaly** = current speed − profile.
  This decomposes the series into "usual" + "deviation".
* **spatial** (5): weighted mean speed / anomaly of **downstream** and
  **upstream** neighbours (forward/backward random-walk matrices of the road graph)
  and the downstream 15-min trend. Congestion waves travel *upstream*, so the
  downstream state is an early warning.
* **sensor** (3): training mean, std and missing-rate per sensor. A cheap
  "which sensor is this" identity, similar to STID's spatial embedding.

**Q11. How do you know the features help?**
Two pieces of evidence: (1) a **cumulative ablation** that adds one group at a time
and measures validation MAE at 15/30/60 min (`fe_ablation.png`,
`tables/feature_ablation.csv`); (2) **permutation importance** on held-out
validation rows (`fe_permutation_importance.png`). Permutation importance measures
the increase in MAE when a feature is shuffled. Unlike impurity importance it is not
biased toward high-cardinality features.

**Q12. Which features matter most, and why does it change with horizon?**
`profile_target` (the sensor's usual speed at the target time-of-week) is the most
important feature at both 15 and 60 min. Shuffling it raises MAE by ≈ 5.3 and
≈ 6.7 mph. The horizon effect shows in the *recent-state* features: `lag_0` costs
≈ 2.4 mph when shuffled at 15 min but only ≈ 0.5 mph at 60 min, and `roll_mean`
drops from ≈ 1.3 to ≈ 0.15. The current anomaly decays (anomaly ACF ≈ 0.41 at
1 h), so long-horizon forecasts revert toward the seasonal mean, like a
mean-reverting AR process. Caveat: permutation importance *splits* credit among
correlated features (lag_0, lag_1 and roll_mean are near-duplicates), so their
individual bars understate the importance of "recent speed" as a group. That is
why we also run the group-level ablation.

**Q13. Why sin/cos for time of day?**
A raw 0–287 index puts 23:55 and 00:00 at opposite ends. Mapping to a circle
keeps them neighbours, so linear models can use it too.

---

## C. Theory

**Q14. Write down the forecasting objective.**
Learn f with Ŷ = f(X_{t−11:t}, G) minimising
L = Σ m_{t,i} |ŷ_{t,i} − y_{t,i}| / Σ m_{t,i}, with mask m = 1[y > 0].

**Q15. Why MAE as the training loss?**
MAE's minimiser is the conditional **median**. With heavy-tailed, left-skewed
speeds (sudden congestion), it is less dominated by rare huge errors than MSE, and it
matches the headline metric. The trade-off is that RMSE is not directly optimised.

**Q16. Explain the adjacency matrix.**
W_ij = exp(−d_ij² / σ²) if ≥ 0.1, else 0, with d_ij the *road* distance and σ the std of
distances. Nearby sensors get weight near 1, and the threshold keeps it sparse (1,515
directed edges plus self-loops = 1,722 non-zeros, identical to DCRNN). It is
asymmetric because road distance is directional (73 % of edges are one-way).

**Q17. What is diffusion convolution?**
It models traffic as a random walk on the graph. The forward transition matrix
P_f = D_O⁻¹W (row-normalised) moves information downstream, and P_b = D_I⁻¹Wᵀ moves it
upstream. A K-step diffusion convolution is
Z = Σ_{k=0}^{K} (P_f^k X Θ_{k,1} + P_b^k X Θ_{k,2}).
It aggregates information from up to K hops in both directions, and costs
O(K·|E|·C) with sparse matrices. Graph WaveNet uses K = 2.

**Q18. What is the adaptive adjacency?**
Ã = softmax(ReLU(E₁E₂)), with E₁ ∈ ℝ^{N×10} and E₂ ∈ ℝ^{10×N} learned end-to-end.
ReLU removes weak links and softmax row-normalises it like a transition matrix. It can
learn dependencies the distance graph misses (e.g. sensors on parallel freeways or
with no recorded edges). It costs O(N²) memory, which is fine for 207 nodes but not
for 10⁴.

**Q19. Gated temporal convolution and receptive field?**
h = tanh(Θ₁ ⋆ x) ⊙ σ(Θ₂ ⋆ x). The sigmoid gate controls information flow, as in
an LSTM but convolutional and parallel. Dilated **causal** convolutions with kernel 2
and dilations 1, 2 repeated over 4 blocks give a receptive field of
1 + 4·(1+2) = 13 ≥ 12 inputs, so every output sees the full hour. The input is
left-padded by one step.

**Q20. Why a CNN instead of an RNN (DCRNN)?**
There is no sequential dependence across time steps during training, so it
parallelises. There are no vanishing gradients through 12 recurrent steps, and no
exposure bias from feeding predictions back. Graph WaveNet beats DCRNN on the
literature benchmark, as Table 1 shows.

**Q21. LSTM equations?**
i,f,o = σ(W·[h_{t−1},x_t]+b), g = tanh(·), c_t = f⊙c_{t−1} + i⊙g, h_t = o⊙tanh(c_t).
The additive cell update lets gradients flow over long sequences. Our LSTM is shared
across sensors, so it is a purely temporal model and isolates the value of spatial
information.

**Q22. Ridge regression?**
β̂ = (XᵀX + λI)⁻¹Xᵀy. The lags are highly collinear (ACF 0.86), and the L2 penalty
stabilises the inverse, trading a little bias for less variance. λ was chosen on
validation. Validation MAE was nearly flat across λ = 0.01…1000, so the model is not
sensitive to it.

**Q23. Gradient boosting?**
An additive model F_M(x) = Σ ν·h_m(x). Each tree h_m fits the negative gradient of
the loss at the current prediction. For absolute error that gradient is sign(y − F),
and the leaf values are medians of residuals. The histogram variant bins each feature
into ≤ 255 bins, so split finding is O(bins) instead of O(n). We stop early on a
validation set that is chronologically after training.

**Q24. Direct vs recursive multi-step forecasting?**
Recursive feeds its own 1-step predictions back in, so errors compound. Direct
trains a separate model per horizon (our tabular models) or a single model with 12
outputs (our deep models). Direct avoids accumulation at the cost of more models
(Ben Taieb et al., 2012).

**Q25. Why does persistence beat HA at 15 min but lose at 60 min?**
Persistence is optimal for a random walk. Traffic is mean-reverting, so the last
value is a great 5–15 min guess but drifts away from the seasonal mean as the
horizon grows. HA ignores the present entirely, so its error is flat across horizons.
The learned models interpolate between the two.

---

## D. Model application and validation

**Q26. Which models did you train and why that set?**
A ladder of increasing complexity, so each step's gain is attributable:
HA (seasonality only) → Persistence (recency only) → Ridge (linear, engineered
features) → Gradient boosting (non-linear, engineered features) → LSTM (learned
temporal features, no space) → Graph WaveNet (learned spatio-temporal features).

**Q27. How did you tune hyper-parameters?**
On the validation split only. Ridge: α grid. Gradient boosting: small grid
(learning rate × leaves) with early stopping on a disjoint half of validation. Deep
models: literature defaults (Graph WaveNet paper), Adam lr 1e-3, weight decay 1e-4,
gradient clipping 5, early stopping on validation MAE. Batch size 32 instead of 64
because of 8 GB unified memory.

**Q28. How do you know the improvement is real and not noise?**
(1) **Diebold–Mariano tests** on per-window absolute errors with a Newey–West
variance (overlapping windows are autocorrelated) (`tables/dm_tests`). (2) 95 %
**block-bootstrap** CIs that resample whole days, which preserves within-day
dependence (`tables/bootstrap_ci.csv`).

**Q29. Your Graph WaveNet numbers differ slightly from the paper. Why?**
A shorter training budget (≤ 35 epochs vs 100), batch 32, a single seed, and minor
implementation differences (no scheduled lr decay). The trend matches the literature,
and our HA (4.19 vs 4.16) and graph statistics (1,722 non-zeros) match exactly, which
validates the pipeline.

**Q30. What does the ablation show?**
Same architecture and same 8-epoch budget, varying only the spatial module: none
(TCN only), road graph, adaptive graph, both. The gap between "none" and the graph
variants is the measured value of spatial information. See `tables/gwnet_ablation.csv` ⟨R⟩.

**Q31. When are your models worst?**
At rush hours (especially the evening peak) and at sensors with frequent congestion.
See `res_mae_by_hour.png` and `res_sensor_error_map.png`. Errors also rise when the
latest input is missing (`tables/robustness_missing.csv`).

**Q32. Limitations?**
A single city and season (4 months, 2012); speed only (no flow, weather or
incidents); a static distance graph; single-seed deep runs; the masked metric hides
performance during outages; and the test period has more missing data than training.

**Q33. What would you do next (Phase 2)?**
Multi-seed runs, a learning-rate schedule, and more epochs; add flow / PEMS-BAY for
generalisation; uncertainty (quantile loss or conformal prediction); recent models
(STAEformer, STID); and deployment as a real-time service.

---

## E. Code and reproducibility

**Q34. How is the repository organised?**
`src/traffic` is an installable package (data, features, metrics, models, trainer,
eda, plotting). `scripts/` holds one entry point per experiment, `tests/` holds
pytest unit tests (leakage, alignment, masking, shapes), `notebooks/` the executed
EDA, `results/` the metrics, tables and figures, `report/` the IEEE LaTeX paper, and
`docs/` this file and the literature review.

**Q35. How do I reproduce everything?**
`pip install -e .[dev]`, then `python scripts/run_eda.py`,
`python scripts/run_classical.py`, `bash scripts/run_deep_experiments.sh`,
`python scripts/evaluate.py`, and `pytest`. Seeds are fixed (numpy/torch seed 0).

**Q36. Why does the HDF5 file need h5py?**
It was written with pandas 0.15, and modern `pd.read_hdf` fails on it. We read the
PyTables arrays directly (`load_speed`), and a test checks shape, order and
timestamps.
