"""Per-segment XGBoost baseline and leakage-free base forecasts for the hybrid.

For every one of the 207 sensors we fit an independent multi-output XGBoost model
(vector-leaf trees, absolute-error loss) on that sensor's own features
(``SegmentFeatures``), with early stopping on every second validation window.

Outputs (results/predictions, git-ignored):
  xgb_seg.npy            test forecasts (A_test, 12, N), scored like every other model
  xgb_seg_base.npy       (T, 12, N) base forecasts indexed by anchor time:
                         training anchors -> out-of-fold (5 blocked, purged folds),
                         validation / test anchors -> model fitted on all training data
  xgb_seg_insample.npy   same, but training anchors get *in-sample* forecasts
                         (used only by the leakage ablation of the hybrid)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
from traffic.data import load_metrla
from traffic.features import FeatureBuilder
from traffic.metrics import evaluate
from traffic.models.segment import SegmentFeatures, blocked_folds, fit_segment

OUT = ROOT / "results"
PARAMS = dict(learning_rate=0.1, n_estimators=400, max_depth=6, multi_strategy="multi_output_tree")
STRIDE = 2          # consecutive windows overlap in 11 of 12 steps


def one_sensor(sf, i, split):
    t0 = time.time()
    tr, va, te = split.train, split.val, split.test
    es = va[::2]                                          # every second validation window drives early stopping
    full = fit_segment(sf, i, tr[::STRIDE], PARAMS, val_anchors=es)
    out = {"insample": full.predict(sf.build(tr, i)), "val": full.predict(sf.build(va, i)),
           "test": full.predict(sf.build(te, i)), "best_it": int(full.best_iteration)}
    oof = np.empty((len(tr), 12), dtype=np.float32)
    pos = {a: k for k, a in enumerate(tr)}
    for fold_train, held in blocked_folds(tr):
        m = fit_segment(sf, i, fold_train[::STRIDE], PARAMS, val_anchors=es)
        oof[[pos[a] for a in held]] = m.predict(sf.build(held, i))
    out["oof"] = oof
    out["sec"] = time.time() - t0
    return i, out


def main():
    data = load_metrla()
    fb = FeatureBuilder(data)
    sf = SegmentFeatures(fb)
    sp = data.split
    n, t_all = data.num_nodes, len(data.index)
    t0 = time.time()
    results = Parallel(n_jobs=6, verbose=5)(
        delayed(one_sensor)(sf, i, sp) for i in range(n))
    print(f"fitted {n} sensors x 6 models in {time.time() - t0:.0f}s", flush=True)

    base = np.full((t_all, 12, n), np.nan, dtype=np.float32)
    insample = np.full((t_all, 12, n), np.nan, dtype=np.float32)
    test = np.empty((len(sp.test), 12, n), dtype=np.float32)
    its = {}
    for i, r in results:
        base[sp.train, :, i] = r["oof"]
        insample[sp.train, :, i] = r["insample"]
        for arr in (base, insample):
            arr[sp.val, :, i] = r["val"]
            arr[sp.test, :, i] = r["test"]
        test[:, :, i] = r["test"]
        its[data.sensor_ids[i]] = r["best_it"]

    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "predictions" / "xgb_seg.npy", test)
    np.save(OUT / "predictions" / "xgb_seg_base.npy", base)
    np.save(OUT / "predictions" / "xgb_seg_insample.npy", insample)

    y = lambda a: data.speed[a[:, None] + np.arange(1, 13)]
    oof_mae = evaluate(base[sp.train], y(sp.train))["avg"]["MAE"]
    ins_mae = evaluate(insample[sp.train], y(sp.train))["avg"]["MAE"]
    metrics = evaluate(test, y(sp.test))
    info = {"model": "xgb_seg", "params": PARAMS, "train_stride": STRIDE, "folds": 5, "purge": 23,
            "train_sec": time.time() - t0, "best_iteration": its,
            "train_oof_MAE": oof_mae, "train_insample_MAE": ins_mae,
            "val_MAE": evaluate(base[sp.val], y(sp.val))["avg"]["MAE"], "metrics": metrics}
    (OUT / "metrics" / "xgb_seg.json").write_text(json.dumps(info, indent=2))
    print(f"train MAE  out-of-fold {oof_mae:.3f}  in-sample {ins_mae:.3f}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
