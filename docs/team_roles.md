# Phase 2: who owns what (5 members)

Each member owns one part of the Phase 2 rubric end to end: the code, the report
section, the slides and the viva questions for it. Owning a part means you can explain
every line of it and answer questions on it without help. Replace "Member 1" to
"Member 5" with your names.

Viva answers for every part are in `docs/viva_notes.md` (section numbers below).

| Member | Owns | Rubric items |
|---|---|---|
| 1 | Data, the per-segment XGBoost baseline and the "gap" | DL dataset, theoretical rigour |
| 2 | ST-GAT, the graph attention network | Architecture logic, regularisation |
| 3 | The XGBoost → GAT hybrid | Hybrid innovation |
| 4 | Ablations, statistics and tests | Ablation studies, technical validation |
| 5 | Literature, deliverables (maps, signal simulation, app) and the final package | DL literature review, GitHub repo, presentation |

---

## Member 1: data, baseline and the gap

**Story to tell:** a strong model per road segment still misses what its neighbours know.

- **Code:** `src/traffic/data.py` (chronological 70/10/20 split, scaler fitted on
  training only, road-graph adjacency), `src/traffic/features.py`, `src/traffic/graph.py`
  (NetworkX graph statistics), `src/traffic/models/segment.py`, `scripts/run_segment_xgb.py`,
  `scripts/compare_xgb_strategy.py`
- **Report:** Section VI-A (baseline and the gap equation, Eq. 8), Section IX-A (the gap exists)
- **Slides:** "Data and a fair protocol", "AML baseline: XGBoost per segment",
  "The gap, measured before any deep model"
- **Key numbers:** 207 sensors, 8.1 % missing, 1,515 directed edges (73 % one-way);
  XGBoost per segment 3.20 mph; neighbour error correlation 0.21 vs 0.04 (60 min)
- **Viva:** sections 1 and 4 of `viva_notes.md`. Be ready for: why speed only (no flow or
  occupancy), why a chronological split, how outages are masked, what the gap equation says.

## Member 2: ST-GAT architecture

**Story to tell:** attention over upstream and downstream neighbours on the fixed road graph.

- **Code:** `src/traffic/models/stgat.py` (`GraphAttention`, `DirectionalGAT`,
  `TemporalGLU`, `STBlock`, `STGAT`), `input_features()` and `Trainer` in
  `src/traffic/trainer.py`, `scripts/train_deep.py`
- **Report:** Section VI-B (ST-GAT), Section VI-D (DL dataset and regularisation),
  Section IX-F (attention analysis), Section IX-H (training behaviour)
- **Slides:** "ST-GAT: spatio-temporal graph attention network", "Graph attention: why,
  and how it differs", "Deep learning dataset and regularisation", "Where does ST-GAT look?"
- **Key numbers:** 163,516 parameters; 2 blocks, 4+4 heads, d = 64;
  ST-GAT 3.04 mph average, 3.47 at 60 min
- **Viva:** sections 2 and 4. Be ready for: the GAT equations, why two attention directions,
  why attention is shared across a window, why GAT generalises mean aggregation, every
  regulariser and why it is there.

## Member 3: the hybrid

**Story to tell:** XGBoost does each segment, graph attention corrects it with the network.

- **Code:** the `base_dim` / `residual` path of `STGAT` (zero-initialised head),
  the `base` handling in `WindowSampler` (`trainer.py`), the cross-fitting in
  `scripts/run_segment_xgb.py` and `blocked_folds()` in `segment.py`
- **Report:** Section VI-C (hybrid design and leakage-free stacking), the hybrid
  paragraphs of Section IX-B and IX-H
- **Slides:** "Hybrid: XGBoost per segment → graph attention", plus the hybrid rows of
  "Results: network-aware models win"
- **Key numbers:** hybrid 3.06 mph (−4.5 % vs XGBoost), best of our GAT models at
  15 min (2.72); better than XGBoost at 79 % of sensors; in-sample 2.69 vs out-of-fold
  2.99 mph (why cross-fitting is needed)
- **Viva:** sections 3 and 9. Be ready for the hardest question: *why is the hybrid not
  better than ST-GAT at 30 and 60 min?* With the same 12-epoch budget it *is* better at
  every horizon (3.05 vs 3.08); with long training it starts to overfit (validation
  plateaus while training keeps falling).

## Member 4: ablations, statistics and tests

**Story to tell:** every design choice is measured, and every claim is tested.

- **Code:** `scripts/run_phase2.sh` (resumable runner), `scripts/evaluate_phase2.py`
  (`ablations`, `significance`, `regime_tables`, `robustness`), `src/traffic/metrics.py`
  (Diebold-Mariano, block bootstrap), `tests/test_phase2.py` (18 tests; 52 in total)
- **Report:** Section IX-B (DM tests, bootstrap intervals), IX-C (peak / off-peak),
  IX-D (outage robustness), IX-E (ablations)
- **Slides:** "Accuracy at every horizon", "Peak, off-peak and congestion",
  "Validation and robustness", and the ablation slides (added when the runs finish)
- **Key numbers:** DM vs XGBoost p < 0.001 at all horizons; removing the graph raises
  ablation MAE from 3.08 to 3.20 mph; ST-GAT fed 0 mph for outages is 0.26 mph worse at 60 min
- **Viva:** sections 5, 6 and the bootstrap question in 9. Be ready for: why
  Newey-West, why day blocks, what each ablation removes, why ablations share one budget.
- **Ablation numbers to know:** ST-GAT full 3.08, mean aggregation 3.10, all-pairs attention
  3.14, no graph 3.20, outages as 0 mph 3.23; hybrid full 3.05, no residual 3.07, no
  cross-fitting 3.14, no graph 3.16 (`results/tables/p2_ablation_*.csv`).

## Member 5: literature, deliverables and the final package

**Story to tell:** where this sits in the literature, and what the forecasts are good for.

- **Code:** `congestion_maps()` and `signal_policy()` in `scripts/evaluate_phase2.py`,
  the app (`app/app.py`, "Graph attention (Phase 2)" tab), `README.md`
- **Report:** Section II (DL literature review, `refs.bib`), Section IX-F (congestion maps),
  IX-G (simulated signal-plan switching), the abstract and the conclusion
- **Slides:** "The question", "Congestion map", "Simulated signal-plan switching
  (offline only)", "Conclusions and limitations"; also assembles and rehearses the deck
- **Key numbers:** onsets anticipated: reactive 0 %, XGBoost 20 %, ST-GAT 30 %,
  hybrid 36 %; median lead 10 min. No live-control claim.
- **Viva:** sections 7 and 8. Be ready for: GAT (Veličković 2018), CityFlow (Zhang 2019),
  DCRNN, STGCN, Graph WaveNet, GMAN, and why we did not use SUMO.
- **Open tasks:** put everyone's names in `report/main.tex` and `presentation/slides.tex`;
  rebuild the PDFs (`tectonic`); demo the app (`streamlit run app/app.py`).

---

## Everyone

- Know the one-line answer: *graph attention over the road network cuts the per-segment
  error by 5 % (6 % at peak), significant at every horizon, and the ablation shows the
  gain disappears without the graph.*
- Read sections 8 (limitations) and 9 (hard questions) of `viva_notes.md`.
- Be able to run `pytest` and the app on your own machine.
- **Presentation order:** Member 5 opens (the question) → 1 (data, baseline, gap) →
  2 (ST-GAT) → 3 (hybrid) → 4 (results, ablations, validation) → 5 (maps, signal
  simulation, conclusions).
