"""Design check for the per-segment XGBoost: vector-leaf trees vs one tree per horizon.

For a few randomly chosen sensors, fit both multi-output strategies with the
settings of run_segment_xgb.py (training every second window, early stopping on
every second validation window) and score them on the *other* validation
windows, which neither model has seen. Writes results/tables/xgb_strategy.csv.

    python scripts/compare_xgb_strategy.py [--sensors 6]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from traffic.data import load_metrla
from traffic.features import FeatureBuilder
from traffic.models.segment import SegmentFeatures, fit_segment

STRATEGIES = {"multi_output_tree": "vector-leaf trees", "one_output_per_tree": "one tree per horizon"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sensors", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    data = load_metrla()
    sf = SegmentFeatures(FeatureBuilder(data))
    tr, va = data.split.train, data.split.val
    es, held = va[::2], va[1::2]
    sensors = np.random.default_rng(args.seed).choice(data.num_nodes, args.sensors, replace=False)
    rows = []
    for i in sensors:
        y = sf.target(held, i)
        for strat, name in STRATEGIES.items():
            t0 = time.time()
            m = fit_segment(sf, i, tr[::2], dict(learning_rate=0.1, n_estimators=400, max_depth=6,
                                                multi_strategy=strat), val_anchors=es)
            sec = time.time() - t0
            pred = m.predict(sf.build(held, i))
            rows.append({"sensor": data.sensor_ids[i], "strategy": name, "val_MAE": np.abs(pred - y)[y > 0].mean(),
                         "fit_sec": sec, "trees": int(m.best_iteration) + 1})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "xgb_strategy.csv", index=False)
    print(df.groupby("strategy")[["val_MAE", "fit_sec"]].mean().round(4))


if __name__ == "__main__":
    main()
