"""Model-level details of the three Phase 2 models, for the project report.

  * XGBoost per segment: gain importance of the 37 features (refits a random subset
    of sensors with the settings of run_segment_xgb.py) and the number of trees kept
    by early stopping for all 207 sensors
  * ST-GAT and hybrid: tensor shape after every layer and parameter count per module
  * hybrid: size of the graph-attention correction applied to the XGBoost forecasts
  * all three: test MAE at each of the 12 horizons

Run after run_phase2.sh:  python scripts/model_details.py [--sensors 30]
Writes results/tables/p2_model_details.json, results/tables/p2_xgb_importance.csv,
results/figures/p2_xgb_importance.png, results/figures/p2_horizon_three.png and report/tables/p2_*detail*.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
from traffic import plotting as P
from traffic.data import load_metrla
from traffic.features import FeatureBuilder
from traffic.metrics import is_peak
from traffic.models import STGAT
from traffic.models.segment import SegmentFeatures, fit_segment

RES, TEX = ROOT / "results", ROOT / "report" / "tables"
PARAMS = dict(learning_rate=0.1, n_estimators=400, max_depth=6, multi_strategy="multi_output_tree")
NAMES = {"xgb_seg": "XGBoost per segment", "stgat_attention_main": "ST-GAT",
         "hybrid_attention_main": "Hybrid XGB$\\to$GAT"}


def xgb_importance(data, n_sensors, seed=0):
    sf = SegmentFeatures(FeatureBuilder(data))
    tr, va = data.split.train, data.split.val
    sensors = np.random.default_rng(seed).choice(data.num_nodes, n_sensors, replace=False)
    rows = []
    for i in sensors:
        m = fit_segment(sf, i, tr[::2], PARAMS, val_anchors=va[::2])
        gain = m.get_booster().get_score(importance_type="gain")
        total = sum(gain.values())
        rows.append({sf.names[int(k[1:])]: v / total for k, v in gain.items()})
    imp = pd.DataFrame(rows).fillna(0.0).mean().sort_values(ascending=False)
    return imp, [data.sensor_ids[i] for i in sensors]


def group_of(name):
    if name.startswith("lag_"):
        return "recent speeds (12 lags)"
    if name.startswith("profile_t+"):
        return "usual speed at target times"
    if name in ("diff_1", "diff_3", "roll_mean", "roll_std", "roll_min"):
        return "last-hour statistics"
    if name in ("n_missing", "missing_now"):
        return "missing-data flags"
    if name in ("tod_sin", "tod_cos", "dow", "weekend"):
        return "calendar"
    return "current level vs. usual (profile, anomaly)"


def layer_table(data, base_dim):
    model = STGAT(data.num_nodes, data.adj, in_dim=6, base_dim=base_dim, residual=bool(base_dim))
    shapes = []
    hooks = []
    def hook(name):
        def f(_, __, out):
            o = out[0] if isinstance(out, tuple) else out
            shapes.append((name, tuple(o.shape)))
        return f
    for name, mod in [("1x1 input convolution", model.inp),
                      ("block 1: gated temporal conv", model.blocks[0].t1),
                      ("block 1: directional GAT", model.blocks[0].gat),
                      ("block 1: gated temporal conv", model.blocks[0].t2),
                      ("block 2: gated temporal conv", model.blocks[1].t1),
                      ("block 2: directional GAT", model.blocks[1].gat),
                      ("block 2: gated temporal conv", model.blocks[1].t2),
                      ("output gated conv", model.t_out)] + \
                     ([("base-forecast embedding", model.base_proj), ("fusion layer", model.fuse),
                       ("fusion GAT", model.fuse_gat)] if base_dim else []) + [("MLP head", model.head)]:
        hooks.append(mod.register_forward_hook(hook(name)))
    x = torch.zeros(1, 12, data.num_nodes, 6)
    with torch.no_grad():
        model(x, torch.zeros(1, 12, data.num_nodes)) if base_dim else model(x)
    for h in hooks:
        h.remove()
    count = lambda m: sum(p.numel() for p in m.parameters())
    params = {"node embedding": model.node_emb.numel(), "input convolution": count(model.inp),
              "ST block 1": count(model.blocks[0]), "ST block 2": count(model.blocks[1]),
              "output gated conv": count(model.t_out), "MLP head": count(model.head)}
    gat_share = {f"block {k + 1} GAT": count(model.blocks[k].gat) for k in range(2)}
    if base_dim:
        params.update({"base-forecast embedding": count(model.base_proj), "fusion layer": count(model.fuse),
                       "fusion GAT": count(model.fuse_gat)})
    total = sum(p.numel() for p in model.parameters())
    assert total == sum(params.values()), (total, params)
    return shapes, params, gat_share, total


def tex_rows_table(rows, header, caption, label, colspec):
    return "\n".join([r"\begin{table}[H]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\small", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
                     + [r + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", type=int, default=30)
    args = ap.parse_args()
    P.setup()
    data = load_metrla()
    out = {}

    # ---- XGBoost per segment
    info = json.loads((RES / "metrics" / "xgb_seg.json").read_text())
    its = np.array(list(info["best_iteration"].values()))
    out["xgb_trees"] = {"median": float(np.median(its) + 1), "min": int(its.min() + 1), "max": int(its.max() + 1),
                        "q25": float(np.percentile(its, 25) + 1), "q75": float(np.percentile(its, 75) + 1),
                        "sensors": len(its)}
    out["xgb_mae"] = {"train_out_of_fold": info["train_oof_MAE"], "train_in_sample": info["train_insample_MAE"],
                      "validation": info["val_MAE"], "test": info["metrics"]["avg"]["MAE"]}
    imp, sensors = xgb_importance(data, args.sensors)
    imp.rename("gain_share").to_csv(RES / "tables" / "p2_xgb_importance.csv")
    groups = imp.groupby(group_of).sum().sort_values(ascending=False)
    out["xgb_importance_groups"] = groups.to_dict()
    out["xgb_importance_top10"] = imp.head(10).to_dict()
    out["xgb_importance_sensors"] = sensors

    fig, ax = P.plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    top = imp.head(12)[::-1]
    ax[0].barh(top.index, 100 * top.values, color=P.SERIES[1])
    ax[0].set(xlabel="share of total gain (%)", title=f"Top features ({args.sensors} random sensors)")
    ax[1].hist(its + 1, bins=25, color=P.SERIES[0])
    ax[1].set(xlabel="trees kept by early stopping", ylabel="sensors", title="Model size across the 207 sensors")
    for a in ax:
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "p2_xgb_importance.png")
    P.plt.close(fig)
    rows = [f"{g} & {100 * v:.1f}\\,\\%" for g, v in groups.items()]
    (TEX / "p2_xgb_importance_groups.tex").write_text(tex_rows_table(
        rows, r"Feature group & Share of gain", f"What the per-segment XGBoost models use: share of the total "
        f"split gain by feature group, averaged over {args.sensors} random sensors.", "tab:xgbgroups", "lr"))

    # ---- ST-GAT and hybrid architecture
    for key, base in (("stgat", 0), ("hybrid", 12)):
        shapes, params, gat_share, total = layer_table(data, base)
        out[f"{key}_shapes"] = shapes
        out[f"{key}_params"] = params
        out[f"{key}_gat_params"] = gat_share
        out[f"{key}_total_params"] = total
    for key, label in (("stgat", "tab:stgatshapes"), ("hybrid", "tab:hybridshapes")):
        rows = ["input window (6 channels) + node embedding (16) & $(1, 22, 207, 12)$"]
        rows += [f"{name} & $({', '.join(map(str, shp))})$" for name, shp in out[f"{key}_shapes"]]
        what = "ST-GAT" if key == "stgat" else "the hybrid"
        (TEX / f"p2_{key}_shapes.tex").write_text(tex_rows_table(
            rows, r"Layer & Output shape",
            f"Tensor shape after each layer of {what} for one window, measured with forward hooks. Convolutions "
            "use (batch, channels, sensors, steps), attention layers (batch, steps, sensors, channels), the "
            "base-forecast and fusion layers (batch, sensors, channels), and the head (batch, sensors, horizons).",
            label, "ll"))
    prow = lambda d: [f"{k} & {v:,}" for k, v in d.items()]
    (TEX / "p2_params.tex").write_text(tex_rows_table(
        [r"\multicolumn{2}{@{}l}{\emph{ST-GAT}}"] + prow(out["stgat_params"]) + [f"\\textbf{{total}} & \\textbf{{{out['stgat_total_params']:,}}}", r"\midrule",
         r"\multicolumn{2}{@{}l}{\emph{Hybrid: additional modules}}"]
        + [f"{k} & {out['hybrid_params'][k]:,}" for k in ("base-forecast embedding", "fusion layer", "fusion GAT")]
        + [f"\\textbf{{total}} & \\textbf{{{out['hybrid_total_params']:,}}}"],
        r"Module & Parameters", "Trainable parameters of ST-GAT and the hybrid by module.", "tab:params", "lr"))

    # ---- horizon-wise MAE of the three models and the hybrid's correction
    hm = pd.read_csv(RES / "tables" / "p2_horizon_metrics.csv")
    plain = {"xgb_seg": "XGBoost per segment", "stgat_attention_main": "ST-GAT", "hybrid_attention_main": "Hybrid XGB→GAT"}
    piv = hm[hm.model.isin(plain.values())].pivot(index="horizon_min", columns="model", values="MAE")[list(plain.values())]
    rows = []
    for h, r in piv.iterrows():
        best = r.min()
        cells = [rf"\textbf{{{v:.2f}}}" if np.isclose(v, best) else f"{v:.2f}" for v in r.values]
        gain = 100 * (1 - min(r.iloc[1], r.iloc[2]) / r.iloc[0])
        rows.append(f"{h} & " + " & ".join(cells) + f" & {gain:.1f}\\,\\%")
    (TEX / "p2_horizon_detail.tex").write_text(tex_rows_table(
        rows, r"Horizon (min) & XGBoost per segment & ST-GAT & Hybrid & Best gain",
        "Test MAE (mph) of the three models at every horizon. Best gain: improvement of the better network model "
        "over XGBoost per segment.", "tab:horizondetail", "rcccc"))
    out["horizon_mae"] = piv.round(4).to_dict()
    colors = [P.SERIES[1], P.SERIES[0], P.SERIES[2]]
    fig, ax = P.plt.subplots(1, 2, figsize=(9.5, 3.4))
    for met, a in zip(("MAE", "RMSE"), ax):
        tab = hm[hm.model.isin(plain.values())].pivot(index="horizon_min", columns="model", values=met)
        for name, c in zip(plain.values(), colors):
            a.plot(tab.index, tab[name], marker="o", ms=3, color=c, label=name)
        a.set(xlabel="forecast horizon (min)", ylabel=f"test {met} (mph)", title=f"{met} by horizon",
              xticks=range(10, 61, 10))
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "p2_horizon_three.png")
    P.plt.close(fig)

    test = data.split.test
    y = data.speed[test[:, None] + np.arange(1, 13)]
    hyb = np.load(RES / "predictions" / "hybrid_attention_main.npy")
    xgb = np.load(RES / "predictions" / "xgb_seg.npy")
    corr = np.abs(hyb - xgb)
    tgt = data.index[(test[:, None] + np.arange(1, 13)).ravel()]
    peak = is_peak(tgt.hour.to_numpy(), tgt.day_of_week.to_numpy()).reshape(y.shape[:2])[..., None]
    peak = np.broadcast_to(peak, corr.shape)
    congested = (y > 0) & (y < 40)
    out["hybrid_correction_mph"] = {
        "mean_abs_all": float(corr.mean()), "mean_abs_peak": float(corr[peak].mean()),
        "mean_abs_offpeak": float(corr[~peak].mean()), "mean_abs_congested_target": float(corr[congested].mean()),
        "mean_abs_by_horizon": [float(corr[:, h].mean()) for h in range(12)],
        "share_corrections_over_2mph": float((corr > 2).mean())}
    (RES / "tables" / "p2_model_details.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if "shapes" not in k}, indent=1, default=str))


if __name__ == "__main__":
    main()
