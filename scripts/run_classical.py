"""Baselines and classical ML models on engineered features.

Runs Historical Average, Persistence, Ridge regression and histogram gradient
boosting (direct multi-horizon strategy), plus the feature-group ablation and
permutation importances used in the report. All model selection uses the
validation split; the test split is only touched for the final numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
from traffic.data import load_metrla
from traffic.features import FEATURE_GROUPS, FeatureBuilder
from traffic.metrics import evaluate
from traffic.models import HistoricalAverage, PerHorizonRegressor, Persistence

OUT = ROOT / "results"
SEED = 0


def hgb(**kw) -> HistGradientBoostingRegressor:
    params = dict(loss="absolute_error", learning_rate=0.1, max_iter=400, max_leaf_nodes=63,
                  min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                  n_iter_no_change=20, random_state=SEED)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def val_mae(model, fb, anchors, h, groups=FEATURE_GROUPS) -> float:
    x, _ = fb.build(anchors, h, groups)
    y = fb.target(anchors, h)
    m = y > 0
    return float(np.abs(model.predict(x[m]) - y[m]).mean())


def fit_one(est, fb, anchors, h, groups=FEATURE_GROUPS, val=None):
    x, _ = fb.build(anchors, h, groups)
    y = fb.target(anchors, h)
    m = y > 0
    kw = {}
    if val is not None:
        xv, _ = fb.build(val, h, groups)
        yv = fb.target(val, h)
        kw = {"X_val": xv[yv > 0], "y_val": yv[yv > 0]}
    return est.fit(x[m], y[m], **kw)


def save(name, pred, y, extra=None):
    metrics = evaluate(pred, y)
    np.save(OUT / "predictions" / f"{name}.npy", pred.astype(np.float32))
    (OUT / "metrics" / f"{name}.json").write_text(json.dumps({"model": name, "metrics": metrics, **(extra or {})}, indent=2))
    print(f"{name:10s}", {k: round(v["MAE"], 3) for k, v in metrics.items()}, flush=True)
    return metrics


def main():
    for d in ("predictions", "metrics", "tables"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    data = load_metrla()
    fb = FeatureBuilder(data)
    sp = data.split
    test = sp.test
    y_test = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]

    # Consecutive 5-min windows overlap in 11 of 12 steps, so every 4th training
    # anchor keeps almost all information at a quarter of the cost.
    train_sub = sp.train[::4]
    val_sel = sp.val[::2]      # hyper-parameter selection
    val_es = sp.val[1::2]      # early stopping (disjoint from selection)

    # 1. Non-learned baselines --------------------------------------------
    save("ha", HistoricalAverage(fb).predict(test), y_test)
    save("persistence", Persistence(fb).predict(test), y_test)

    # 2. Ridge: choose alpha on validation at h = 6 ------------------------
    alphas = [0.01, 1.0, 100.0, 1000.0]
    ridge_scores = {a: val_mae(fit_one(make_pipeline(StandardScaler(), Ridge(alpha=a)), fb, train_sub, 6),
                               fb, val_sel, 6) for a in alphas}
    best_alpha = min(ridge_scores, key=ridge_scores.get)
    print("ridge val MAE by alpha:", ridge_scores, "->", best_alpha, flush=True)
    t0 = time.time()
    ridge = PerHorizonRegressor(make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)), fb).fit(train_sub, verbose=False)
    save("ridge", ridge.predict(test), y_test,
         {"alpha": best_alpha, "val_mae_by_alpha": ridge_scores, "train_sec": time.time() - t0})

    # 3. Gradient boosting: small grid on validation at h = 6 --------------
    grid = [dict(learning_rate=0.1, max_leaf_nodes=31),
            dict(learning_rate=0.1, max_leaf_nodes=127),
            dict(learning_rate=0.05, max_leaf_nodes=63, max_iter=800)]
    grid_scores = []
    for g in grid:
        m = fit_one(hgb(**g), fb, train_sub, 6, val=val_es)
        grid_scores.append({**g, "val_mae": val_mae(m, fb, val_sel, 6), "n_iter": int(m.n_iter_)})
        print("  hgb", grid_scores[-1], flush=True)
    best = min(grid_scores, key=lambda r: r["val_mae"])
    best_params = {k: v for k, v in best.items() if k not in ("val_mae", "n_iter")}
    pd.DataFrame(grid_scores).to_csv(OUT / "tables" / "hgb_grid.csv", index=False)

    t0 = time.time()
    gbm = PerHorizonRegressor(hgb(**best_params), fb).fit(train_sub, val_anchors=val_es)
    train_sec = time.time() - t0
    t1 = time.time()
    pred = gbm.predict(test)
    save("hgb", pred, y_test, {"params": best_params, "train_sec": train_sec,
                                "infer_sec": time.time() - t1,
                                "n_iter": {h: int(m.n_iter_) for h, m in gbm.models.items()}})

    # 4. Permutation importance (validation rows, h = 3 and 12) ------------
    rng = np.random.default_rng(SEED)
    rows = []
    for h in (3, 12):
        x, names = fb.build(val_sel, h)
        y = fb.target(val_sel, h)
        m = y > 0
        idx = rng.choice(np.flatnonzero(m), 60_000, replace=False)
        r = permutation_importance(gbm.models[h], x[idx], y[idx], scoring="neg_mean_absolute_error",
                                   n_repeats=3, random_state=SEED, n_jobs=1)
        rows += [{"horizon": h, "feature": n, "importance": mu, "std": sd}
                 for n, mu, sd in zip(names, r.importances_mean, r.importances_std)]
    pd.DataFrame(rows).to_csv(OUT / "tables" / "permutation_importance.csv", index=False)

    # 5. Feature-group ablation (cumulative), scored on validation ---------
    abl = []
    for k in range(1, len(FEATURE_GROUPS) + 1):
        groups = FEATURE_GROUPS[:k]
        for h in (3, 6, 12):
            m = fit_one(hgb(**best_params), fb, train_sub, h, groups, val=val_es)
            abl.append({"groups": "+".join(groups), "n_groups": k, "horizon": h,
                        "val_mae": val_mae(m, fb, val_sel, h, groups)})
            print("  ablation", abl[-1], flush=True)
    pd.DataFrame(abl).to_csv(OUT / "tables" / "feature_ablation.csv", index=False)


if __name__ == "__main__":
    main()
