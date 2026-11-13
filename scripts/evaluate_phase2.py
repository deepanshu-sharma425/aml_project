"""Phase 2 evaluation: graph attention and the XGBoost -> GAT hybrid.

Run after run_classical.py, run_segment_xgb.py and run_phase2.sh. Every number
in the Phase 2 sections of the report is produced here:

  * horizon-wise MAE / RMSE / MAPE for every model (all 12 horizons)
  * peak vs off-peak and congested vs free-flow breakdowns
  * the "gap" diagnostic: are the per-segment model's errors predictable from
    neighbouring segments at forecast time?
  * Diebold-Mariano tests and day-block bootstrap intervals
  * ST-GAT and hybrid ablation tables
  * congestion maps, per-sensor improvement map and attention analysis
  * a simulated, offline signal/ramp-meter plan-switching experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
from traffic import plotting as P
from traffic.data import distance_matrix, load_locations, load_metrla
from traffic.features import FeatureBuilder
from traffic.graph import graph_stats
from traffic.metrics import (HORIZONS, block_bootstrap_ci, diebold_mariano, evaluate, is_peak,
                             per_step_abs_error, subset_metrics)
from traffic.models import STGAT
from traffic.trainer import WindowSampler, pick_device

RES = ROOT / "results"
FIG, TAB, TEX = RES / "figures", RES / "tables", ROOT / "report" / "tables"
plt = P.plt

MAIN = {  # file stem -> display name; order = table order and colour slot
    "ha": "Historical Average",
    "persistence": "Persistence",
    "ridge": "Ridge (features)",
    "hgb": "Gradient Boosting (features)",
    "xgb_seg": "XGBoost per segment",
    "lstm_main": "LSTM",
    "gwnet_full_main": "Graph WaveNet",
    "stgat_attention_main": "ST-GAT",
    "hybrid_attention_main": "Hybrid XGB$\\to$GAT",
}
PLAIN = {k: v.replace("$\\to$", "→") for k, v in MAIN.items()}
COLOR = {"ha": "#898781", "persistence": "#c3c2b7", "ridge": "#52514e", "hgb": P.SERIES[3],
         "xgb_seg": P.SERIES[1], "lstm_main": P.SERIES[4], "gwnet_full_main": P.SERIES[6],
         "stgat_attention_main": P.SERIES[0], "hybrid_attention_main": P.SERIES[2]}
KEY = ["xgb_seg", "gwnet_full_main", "stgat_attention_main", "hybrid_attention_main"]
ABL_STGAT = {
    "stgat_attention_ablation": "Road-graph attention (full)",
    "stgat_mean_ablation": "Mean aggregation (no attention)",
    "stgat_dense_ablation": "Attention over all pairs (no road prior)",
    "stgat_none_ablation": "No spatial module",
    "stgat_rawinput_ablation": "Outages fed as 0 mph (no mask)",
}
ABL_HYBRID = {
    "hybrid_attention_ablation": "Full hybrid (OOF base, residual, GAT)",
    "hybrid_none_ablation": "Correction without graph",
    "hybrid_linear": "Linear neighbour correction (LAD)",
    "hybrid_noresid_ablation": "Base as input only (no residual)",
    "hybrid_insample_ablation": "In-sample base (no cross-fitting)",
    "xgb_seg": "XGBoost per segment alone",
}
LITERATURE = [
    ("DCRNN [Li et al. 2018]", [2.77, 5.38, 7.30, 3.15, 6.45, 8.80, 3.60, 7.60, 10.50]),
    ("STGCN [Yu et al. 2018]$^\\dagger$", [2.88, 5.74, 7.62, 3.47, 7.24, 9.57, 4.59, 9.40, 12.70]),
    ("Graph WaveNet [Wu et al. 2019]", [2.69, 5.15, 6.90, 3.07, 6.22, 8.37, 3.53, 7.37, 10.01]),
    ("GMAN [Zheng et al. 2020]", [2.80, 5.55, 7.41, 3.12, 6.49, 8.73, 3.44, 7.35, 10.07]),
]
BANDS = [("free flow ($\\geq$55)", 55, "#0ca30c"), ("slowing (40–55)", 40, "#eda100"),
         ("congested (25–40)", 25, "#eb6834"), ("severe ($<$25)", 0, "#d03b3b")]


def load_preds(names):
    out = {}
    for k in names:
        f = RES / "predictions" / f"{k}.npy"
        if f.exists():
            out[k] = np.load(f)
        else:
            print(f"  (missing predictions: {k})")
    return out


def tex_table(rows, header, caption, label, colspec, wide=False):
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3.5pt}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             header + r" \\", r"\midrule"] + [r if r.lstrip().startswith((r"\midrule", r"\cmidrule")) else r + r" \\"
                                          for r in rows] + [r"\bottomrule", r"\end{tabular}",
                                                                           rf"\end{{{env}}}"]
    return "\n".join(lines)


def bold_min(values, fmt="{:.2f}"):
    best = np.nanmin(values)
    return [rf"\textbf{{{fmt.format(v)}}}" if np.isclose(v, best) else fmt.format(v) for v in values]


# ============================================================== main tables
def main_tables(preds, y):
    rows, long = [], []
    for k, p in preds.items():
        m = evaluate(p, y)
        rows.append({"model": PLAIN[k], **{f"{h} {met}": m[h][met] for h in list(HORIZONS) + ["avg"]
                                          for met in ("MAE", "RMSE", "MAPE")}})
        for h in range(12):
            s = subset_metrics(p[:, h], y[:, h], True)
            long.append({"model": PLAIN[k], "horizon_min": 5 * (h + 1), **{x: s[x] for x in ("MAE", "RMSE", "MAPE")}})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p2_main_results.csv", index=False)
    pd.DataFrame(long).to_csv(TAB / "p2_horizon_metrics.csv", index=False)

    cols = [f"{h} {m}" for h in HORIZONS for m in ("MAE", "RMSE", "MAPE")]
    bolds = {c: bold_min(df[c].to_numpy()) for c in cols}
    body = [f"{MAIN[k]} & " + " & ".join(bolds[c][i] for c in cols) for i, k in enumerate(preds)]
    body += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\emph{{Published results (same data and protocol)}}}}"]
    body += [f"{n} & " + " & ".join(f"{v:.2f}" for v in vals) for n, vals in LITERATURE]
    header = (" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in HORIZONS) + r" \\ "
              + " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3)) + "\n"
              + "Model & " + " & ".join(["MAE", "RMSE", "MAPE\\%"] * 3))
    (TEX / "p2_main_results.tex").write_text(tex_table(
        body, header, "Test-set accuracy on METR-LA (12-step history for every model). Best of our models in "
        "bold. $^\\dagger$As reported by Wu et al. (2019).", "tab:p2main", "l" + "ccc" * 3, wide=True))
    print(df[["model", "15 min MAE", "30 min MAE", "60 min MAE", "avg MAE"]].round(3).to_string(index=False))

    # horizon-wise figure: all 12 steps, MAE and RMSE
    hm = pd.DataFrame(long)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, k in enumerate(preds):
        s = hm[hm.model == PLAIN[k]]
        for a, met in zip(ax, ("MAE", "RMSE")):
            a.plot(s.horizon_min, s[met], marker="o", ms=3, color=COLOR[k],
                   ls="--" if k in ("ha", "persistence", "ridge") else "-", label=PLAIN[k])
    for a, met in zip(ax, ("MAE", "RMSE")):
        a.set(xlabel="forecast horizon (min)", ylabel=f"test {met} (mph)", xticks=range(10, 61, 10))
    ax[0].set_title("MAE by horizon")
    ax[1].set_title("RMSE by horizon")
    ax[1].legend(fontsize=6.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "p2_horizon_metrics.png")
    plt.close(fig)
    return df


# ======================================================= peak / congestion
def regime_tables(preds, y, peak):
    """``peak``: (A, 12) bool, whether each target time falls in a peak period."""
    congested = (y > 0) & (y < 40)
    rows = []
    for k, p in preds.items():
        for name, h in HORIZONS.items():
            for regime, sel in (("peak", peak[:, h - 1]), ("off-peak", ~peak[:, h - 1])):
                s = subset_metrics(p[:, h - 1], y[:, h - 1], sel[:, None])
                rows.append({"model": PLAIN[k], "horizon": name, "regime": regime, **s})
            for regime, sel in (("congested (<40 mph)", congested[:, h - 1]),
                                ("free flow (>=40 mph)", ~congested[:, h - 1])):
                s = subset_metrics(p[:, h - 1], y[:, h - 1], sel)
                rows.append({"model": PLAIN[k], "horizon": name, "regime": regime, **s})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p2_regimes.csv", index=False)

    body = []
    for k in preds:
        cells = []
        for regime in ("peak", "off-peak"):
            for name in HORIZONS:
                r = df[(df.model == PLAIN[k]) & (df.regime == regime) & (df.horizon == name)].iloc[0]
                cells.append(f"{r.MAE:.2f}")
        for name in ("60 min",):
            r = df[(df.model == PLAIN[k]) & (df.regime == "congested (<40 mph)") & (df.horizon == name)].iloc[0]
            cells.append(f"{r.MAE:.2f}")
        body.append([MAIN[k]] + cells)
    arr = np.array([[float(c) for c in r[1:]] for r in body])
    for j in range(arr.shape[1]):
        col = bold_min(arr[:, j])
        for i in range(len(body)):
            body[i][1 + j] = col[i]
    header = (r" & \multicolumn{3}{c}{Peak} & \multicolumn{3}{c}{Off-peak} & Congested \\ "
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}" "\n"
              r"Model & 15 & 30 & 60 & 15 & 30 & 60 & 60 min")
    n_peak = int(peak[:, 11].sum())
    (TEX / "p2_regimes.tex").write_text(tex_table(
        [" & ".join(r) for r in body], header,
        f"Test MAE (mph) in peak periods (weekdays 06--10 and 15--19, by target time; "
        f"{100 * n_peak / len(peak):.0f}\\% of windows) and off-peak, and on congested targets ($<$40\\,mph).",
        "tab:p2regime", "lccccccc", wide=True))

    keys = [k for k in KEY if k in preds]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    w = 0.8 / max(len(keys), 1)
    for a, (name, h) in zip(ax, HORIZONS.items()):
        for i, k in enumerate(keys):
            vals = [df[(df.model == PLAIN[k]) & (df.regime == r) & (df.horizon == name)].MAE.iloc[0]
                    for r in ("off-peak", "peak", "congested (<40 mph)")]
            a.bar(np.arange(3) + (i - (len(keys) - 1) / 2) * w, vals, w, color=COLOR[k],
                  label=PLAIN[k])
        a.set_xticks(range(3), ["off-peak", "peak", "congested"])
        a.set_title(f"{name} ahead")
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("test MAE (mph)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "p2_peak_offpeak.png")
    plt.close(fig)
    return df


def robustness(data, preds, y):
    """60-minute MAE when the sensor's latest input reading is observed vs missing.
    Phase 1 fed outages to the networks as 0 mph; Phase 2 masks them."""
    last_missing = data.speed[data.split.test] == 0                 # (A, N)
    rows = []
    for k, p in preds.items():
        r = {"model": PLAIN[k]}
        for lab, sel in (("observed", ~last_missing), ("missing", last_missing)):
            r[f"MAE60_latest_{lab}"] = subset_metrics(p[:, 11], y[:, 11], sel)["MAE"]
        r["degradation_%"] = 100 * (r["MAE60_latest_missing"] / r["MAE60_latest_observed"] - 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p2_robustness_missing.csv", index=False)
    body = [f"{MAIN[k]} & {r.MAE60_latest_observed:.2f} & {r.MAE60_latest_missing:.2f} & "
            f"{r['degradation_%']:+.0f}\\%" for k, (_, r) in zip(preds, df.iterrows())]
    (TEX / "p2_robustness.tex").write_text(tex_table(
        body, r"Model & Latest observed & Latest missing & Change",
        f"Robustness to sensor outages: test MAE at 60\\,min when the sensor's most recent input "
        f"reading is observed or missing ({100 * last_missing.mean():.1f}\\% of cases).", "tab:p2robust", "lccc"))
    print(df.round(3).to_string(index=False))
    return df


# ============================================================ gap diagnostic
def gap_analysis(data, fb):
    """Are the independent model's errors explained by *neighbours* at forecast time?

    Uses out-of-fold training residuals only, so the test set is not involved.
    """
    base = np.load(RES / "predictions" / "xgb_seg_base.npy", mmap_mode="r")
    a = data.split.train
    out = {}
    adj = data.adj.copy()
    np.fill_diagonal(adj, 0)
    nbr = (adj > 0) | (adj.T > 0)
    from traffic.data import distance_matrix
    dist = distance_matrix(data.sensor_ids)
    dist = np.minimum(dist, dist.T)
    rows = []
    for h in (3, 12):
        y = data.speed[a + h]
        r = np.where(y > 0, np.asarray(base[a, h - 1]) - y, np.nan)          # (A, N)
        c = pd.DataFrame(r).corr(min_periods=500).to_numpy()
        off = ~np.eye(len(c), dtype=bool)
        out[f"h{h}_resid_corr_neighbours"] = float(np.nanmean(c[nbr]))
        out[f"h{h}_resid_corr_non_neighbours"] = float(np.nanmean(c[off & ~nbr]))
        # information available at time t that the per-segment model never saw
        own = fb.anomaly[a]
        down = own @ fb.p_fwd.T
        up = own @ fb.p_bwd.T
        trend = (fb.filled[a] - fb.filled[a - 3]) @ fb.p_fwd.T

        def mean_corr(u):
            vals = []
            for i in range(r.shape[1]):
                ok = ~np.isnan(r[:, i])
                if ok.sum() > 500 and u[ok, i].std() > 0:
                    vals.append(np.corrcoef(r[ok, i], u[ok, i])[0, 1])
            return float(np.mean(np.abs(vals)))
        out[f"h{h}_abs_corr_resid_own_anomaly"] = mean_corr(own)
        out[f"h{h}_abs_corr_resid_downstream_anomaly"] = mean_corr(down)
        out[f"h{h}_abs_corr_resid_upstream_anomaly"] = mean_corr(up)
        out[f"h{h}_abs_corr_resid_downstream_trend"] = mean_corr(trend)
        finite = np.isfinite(dist) & off
        bins = [0, 1000, 2000, 4000, 8000, 16000, np.inf]
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = finite & (dist >= lo) & (dist < hi)
            rows.append({"horizon": h, "dist_bin_km": f"{lo / 1000:g}-{hi / 1000:g}", "lo": lo / 1000,
                         "resid_corr": float(np.nanmean(c[sel])), "pairs": int(sel.sum())})
        rows.append({"horizon": h, "dist_bin_km": "no road path", "lo": np.nan,
                     "resid_corr": float(np.nanmean(c[off & ~np.isfinite(dist)])),
                     "pairs": int((off & ~np.isfinite(dist)).sum())})
    (TAB / "p2_gap_analysis.json").write_text(json.dumps(out, indent=2))
    bydist = pd.DataFrame(rows)
    bydist.to_csv(TAB / "p2_gap_resid_corr_by_distance.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for i, h in enumerate((3, 12)):
        s = bydist[(bydist.horizon == h) & (bydist.pairs > 0)]
        ax[0].plot(range(len(s)), s.resid_corr, marker="o", color=P.SERIES[i], label=f"{h * 5} min ahead")
    ax[0].set_xticks(range(len(s)), s.dist_bin_km, rotation=25, ha="right")
    ax[0].set(xlabel="road distance between sensors (km)", ylabel="corr. of XGBoost errors",
              title="Independent-model errors are spatially correlated")
    ax[0].legend()
    labels = ["own anomaly\n(model input)", "downstream\nanomaly", "upstream\nanomaly", "downstream\ntrend"]
    keys = ["own_anomaly", "downstream_anomaly", "upstream_anomaly", "downstream_trend"]
    w = 0.38
    for i, h in enumerate((3, 12)):
        ax[1].bar(np.arange(4) + (i - 0.5) * w, [out[f"h{h}_abs_corr_resid_{k}"] for k in keys], w,
                  color=P.SERIES[i], label=f"{h * 5} min ahead")
    ax[1].set_xticks(range(4), labels, fontsize=7.5)
    ax[1].set(ylabel="mean |corr| with XGBoost error", title="Neighbour signals the model never saw")
    ax[1].grid(axis="x", visible=False)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "p2_gap_analysis.png")
    plt.close(fig)
    print("gap:", json.dumps(out, indent=1))
    return out


def linear_neighbour_correction(data, fb, rows: int = 50_000):
    """Cheapest possible hybrid: per-horizon linear model on network features that
    corrects the XGBoost residual (pooled over sensors, fit on out-of-fold residuals).

    Fitted with absolute loss (median regression), like every other model here.
    XGBoost minimises absolute error, so its residuals have median ~0 but a
    negative mean (congestion makes large negative errors). A squared-loss (ridge)
    correction learns that mean shift and *raises* MAE: on validation it went from
    2.75 to 2.83 mph at 15 min, whereas the median fit lowers it to 2.70.
    The exact LAD fit uses a random subsample of ``rows`` training rows per horizon.
    """
    from sklearn.linear_model import QuantileRegressor
    base = np.load(RES / "predictions" / "xgb_seg_base.npy", mmap_mode="r")
    tr, te = data.split.train[::4], data.split.test
    n = data.num_nodes
    rng = np.random.default_rng(0)

    def feats(a, h):
        anom = fb.anomaly[a]
        b = np.asarray(base[a, h - 1])
        cols = [anom, anom @ fb.p_fwd.T, anom @ fb.p_bwd.T, (fb.filled[a] - fb.filled[a - 3]) @ fb.p_fwd.T,
                (fb.filled[a] - fb.filled[a - 3]) @ fb.p_bwd.T, b - fb.filled[a], b @ fb.p_fwd.T - b,
                b @ fb.p_bwd.T - b]
        return np.stack([c.reshape(-1) for c in cols], 1)

    pred = np.empty((len(te), 12, n), dtype=np.float32)
    for h in range(1, 13):
        y = data.speed[tr + h].reshape(-1)
        r = y - np.asarray(base[tr, h - 1]).reshape(-1)
        ok = np.flatnonzero(y > 0)
        idx = rng.choice(ok, min(rows, len(ok)), replace=False)
        m = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(feats(tr, h)[idx], r[idx])
        pred[:, h - 1] = np.asarray(base[te, h - 1]) + m.predict(feats(te, h)).reshape(len(te), n)
    np.save(RES / "predictions" / "hybrid_linear.npy", pred)
    return pred


# =============================================================== significance
def significance(preds, y):
    pairs = [("hybrid_attention_main", "xgb_seg"), ("stgat_attention_main", "xgb_seg"),
             ("hybrid_attention_main", "stgat_attention_main"), ("hybrid_attention_main", "gwnet_full_main"),
             ("stgat_attention_main", "gwnet_full_main"), ("hybrid_attention_main", "hgb"),
             ("xgb_seg", "hgb")]
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        for name, h in HORIZONS.items():
            ea, eb = per_step_abs_error(preds[a], y, h), per_step_abs_error(preds[b], y, h)
            stat, pv = diebold_mariano(ea, eb, lag=h)
            rows.append({"model_a": PLAIN[a], "model_b": PLAIN[b], "horizon": name, "mae_a": np.nanmean(ea),
                         "mae_b": np.nanmean(eb), "dm_stat": stat, "p_value": pv, "a": a, "b": b})
    dm = pd.DataFrame(rows)
    dm.to_csv(TAB / "p2_diebold_mariano.csv", index=False)
    if dm.empty:
        return dm
    body = []
    for (a, b), g in dm.groupby(["a", "b"], sort=False):
        cells = []
        for _, r in g.iterrows():
            pv = "$<$.001" if r.p_value < 1e-3 else f"{r.p_value:.3f}"
            cells.append(f"{r.dm_stat:.1f} ({pv})")
        body.append(f"{MAIN[a]} vs.\\ {MAIN[b]} & " + " & ".join(cells))
    (TEX / "p2_dm.tex").write_text(tex_table(
        body, r"Comparison (A vs.\ B) & 15 min & 30 min & 60 min",
        r"Diebold--Mariano statistic ($p$-value) on absolute error with Newey--West variance. "
        r"Negative: A is more accurate.", "tab:p2dm", "lccc", wide=True))

    mask = y > 0
    ci = []
    for k, p in preds.items():
        e = np.where(mask, np.abs(p - y), 0).sum((1, 2))
        c = mask.sum((1, 2))
        lo, hi = block_bootstrap_ci(e, c)
        ci.append({"model": PLAIN[k], "avg_MAE": e.sum() / c.sum(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ci).to_csv(TAB / "p2_bootstrap_ci.csv", index=False)
    return dm


# ================================================================= ablations
def ablations(y):
    out = {}
    for name, spec, cap, label in (
            ("stgat", ABL_STGAT, "ST-GAT ablation: what each design choice is worth. Identical training "
                                 "budget and seed; test MAE (mph).", "tab:p2abl_stgat"),
            ("hybrid", ABL_HYBRID, "Hybrid ablation. Every variant corrects the same per-segment XGBoost "
                                   "forecasts; test MAE (mph).", "tab:p2abl_hybrid")):
        preds = load_preds(spec)
        if not preds:
            continue
        rows = []
        for k, p in preds.items():
            m = evaluate(p, y)
            f = RES / "metrics" / f"{k}.json"
            info = json.loads(f.read_text()) if f.exists() else {}
            n_par = info.get("params")                     # xgb_seg stores its hyper-parameters here
            rows.append({"variant": spec[k], "key": k,
                         "params": float(n_par) if isinstance(n_par, (int, float)) else np.nan,
                         "epochs": info.get("epochs_run", np.nan),
                         **{f"{h} MAE": m[h]["MAE"] for h in list(HORIZONS) + ["avg"]},
                         "60 min RMSE": m["60 min"]["RMSE"]})
        df = pd.DataFrame(rows)
        df.to_csv(TAB / f"p2_ablation_{name}.csv", index=False)
        cols = ["15 min MAE", "30 min MAE", "60 min MAE", "avg MAE"]
        b = {c: bold_min(df[c].to_numpy()) for c in cols}
        body = [f"{r.variant} & " + ("--" if np.isnan(r.params) else f"{int(r.params) // 1000}k") + " & "
                + " & ".join(b[c][i] for c in cols) for i, r in df.iterrows()]
        (TEX / f"p2_ablation_{name}.tex").write_text(tex_table(
            body, r"Variant & Params & 15 min & 30 min & 60 min & Avg", cap, label, "lrcccc", wide=True))
        print(df[["variant"] + cols].round(3).to_string(index=False))
        out[name] = df
    return out


# =========================================================== congestion maps
def congestion_maps(data, preds, y):
    test = data.split.test
    keys = [k for k in ("xgb_seg", "stgat_attention_main", "hybrid_attention_main") if k in preds]
    if not keys:
        return
    times = data.index[test + 12]
    obs = y[:, 11]
    # the weekday window (target time 16:00-19:00) with the most congested observed sensors
    cand = np.flatnonzero((times.day_of_week < 5) & (times.hour >= 16) & (times.hour < 19))
    j = cand[np.argmax(((obs[cand] > 0) & (obs[cand] < 40)).sum(1))]
    loc = load_locations().set_index("sensor_id").loc[data.sensor_ids]
    panels = [("Observed", obs[j])] + [(f"{PLAIN[k]}\n(made 60 min earlier)", preds[k][j, 11]) for k in keys]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 2.9), sharex=True, sharey=True)
    for a, (title, v) in zip(ax, panels):
        col = np.full(len(v), "#c3c2b7", dtype=object)
        for _, lo, c in BANDS[::-1]:
            col[v >= lo] = c
        col[v <= 0] = "#c3c2b7"
        a.scatter(loc.longitude, loc.latitude, c=list(col), s=16, edgecolor="white", linewidth=0.3)
        m = obs[j] > 0
        err = np.abs(v[m] - obs[j][m]).mean() if title != "Observed" else None
        a.set_title(title + (f"\nMAE {err:.2f} mph" if err is not None else f"\n{times[j]:%a %d %b %H:%M}"),
                    fontsize=8.5)
        a.set_aspect(1 / np.cos(np.deg2rad(loc.latitude.mean())))
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    from matplotlib.patches import Patch
    fig.tight_layout()
    ax[len(ax) // 2].legend(handles=[Patch(color=c, label=l) for l, _, c in BANDS]
                            + [Patch(color="#c3c2b7", label="no reading")],
                            loc="upper center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.savefig(FIG / "p2_congestion_map.png", bbox_inches="tight")
    plt.close(fig)

    # where does the hybrid help? per-sensor MAE change at 60 min vs XGBoost per segment
    if {"xgb_seg", "hybrid_attention_main"} <= preds.keys():
        m = y[:, 11] > 0
        def sensor_mae(p):
            return np.nanmean(np.where(m, np.abs(p[:, 11] - y[:, 11]), np.nan), 0)
        delta = sensor_mae(preds["hybrid_attention_main"]) - sensor_mae(preds["xgb_seg"])
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        lim = np.nanpercentile(np.abs(delta), 98)
        sc = ax.scatter(loc.longitude, loc.latitude, c=delta, cmap="RdBu_r", vmin=-lim, vmax=lim, s=24,
                        edgecolor="#52514e", linewidth=0.3)
        fig.colorbar(sc, ax=ax, label="MAE change at 60 min (mph)", fraction=0.04)
        ax.set(xlabel="longitude", ylabel="latitude",
               title=f"Hybrid vs. per-segment XGBoost: better at {np.mean(delta < 0):.0%} of sensors")
        ax.set_aspect(1 / np.cos(np.deg2rad(loc.latitude.mean())))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        fig.savefig(FIG / "p2_improvement_map.png")
        plt.close(fig)
        pd.DataFrame({"sensor": data.sensor_ids, "delta_mae_60": delta}).to_csv(TAB / "p2_sensor_delta.csv", index=False)


# ============================================================ attention study
def load_stgat(data, run):
    f = RES / "metrics" / f"{run}.json"
    ck = ROOT / "checkpoints" / f"{run}.pt"
    if not (f.exists() and ck.exists()):
        return None, None
    a = json.loads(f.read_text())["args"]
    m = STGAT(data.num_nodes, data.adj, in_dim=6, d=int(a["hidden"]), heads=int(a["heads"]),
              spatial=a["spatial"], dropout=float(a["dropout"]), attn_dropout=float(a["attn_dropout"]),
              node_emb=int(a["node_emb"]))
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    return m.eval(), a


def attention_study(data):
    model, _ = load_stgat(data, "stgat_attention_main")
    if model is None:
        return
    dev = pick_device()
    model.to(dev)
    sampler = WindowSampler(data, dev, "masked")
    test = data.split.test
    self_w = {"down": [], "up": []}
    for i in range(0, len(test), 64):
        x, _ = sampler.batch(test[i:i + 64])
        maps = model.attention_maps(x, block=0)
        for d, A in zip(("down", "up"), maps):
            self_w[d].append(torch.diagonal(A.mean(1), dim1=-2, dim2=-1).cpu().numpy())   # (B, N)
    self_w = {d: np.concatenate(v) for d, v in self_w.items()}
    cur = data.speed[test]                                          # speed at forecast time
    cong, free = (cur > 0) & (cur < 40), cur >= 55
    deg_down = ((data.adj > 0) & ~np.eye(data.num_nodes, dtype=bool)).sum(1)
    has_nbr = {"down": deg_down > 0, "up": ((data.adj.T > 0) & ~np.eye(data.num_nodes, dtype=bool)).sum(1) > 0}
    stats = {}
    for d in ("down", "up"):
        nb = 1 - self_w[d]                                          # attention mass on neighbours
        ok = has_nbr[d][None, :]
        stats[f"{d}_neighbour_mass_congested"] = float(nb[cong & ok].mean())
        stats[f"{d}_neighbour_mass_freeflow"] = float(nb[free & ok].mean())
        stats[f"{d}_neighbour_mass_all"] = float(nb[np.broadcast_to(ok, nb.shape)].mean())
    (TAB / "p2_attention_stats.json").write_text(json.dumps(stats, indent=2))
    print("attention:", stats)

    # map: one congested sensor at the evening peak and where it looks
    times = data.index[test]
    cand = np.flatnonzero((times.day_of_week < 5) & (times.hour == 17))
    j = cand[np.argmax(((cur[cand] > 0) & (cur[cand] < 40)).sum(1))]
    x, _ = sampler.batch(test[j:j + 1])
    maps = [A[0].mean(0).cpu().numpy() for A in model.attention_maps(x, block=0)]
    nbr_count = (data.adj > 0).sum(1) + (data.adj > 0).sum(0)
    congested_now = np.flatnonzero((cur[j] > 0) & (cur[j] < 40))
    s = congested_now[np.argmax(nbr_count[congested_now])] if len(congested_now) else int(np.argmax(nbr_count))
    loc = load_locations().set_index("sensor_id").loc[data.sensor_ids]
    lon, lat = loc.longitude.to_numpy(), loc.latitude.to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for a, A, ttl, c in zip(ax[:2], maps, ("downstream heads", "upstream heads"), (P.SERIES[0], P.SERIES[1])):
        a.scatter(lon, lat, s=6, color="#c3c2b7")
        nb = np.flatnonzero(A[s] > 1e-4)
        for k in nb:
            if k != s:
                a.plot([lon[s], lon[k]], [lat[s], lat[k]], color=c, lw=0.5 + 6 * A[s, k], alpha=0.8)
        a.scatter(lon[nb], lat[nb], s=12 + 300 * A[s, nb], color=c, edgecolor="white", linewidth=0.4, zorder=3)
        a.scatter([lon[s]], [lat[s]], s=60, marker="*", color=P.TEXT, zorder=4)
        pad = 0.04
        a.set(xlim=(lon[nb].min() - pad, lon[nb].max() + pad), ylim=(lat[nb].min() - pad, lat[nb].max() + pad),
              title=f"{ttl}: self weight {A[s, s]:.2f}")
        a.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    lab = ["downstream", "upstream"]
    w = 0.35
    for i, d in enumerate(("down", "up")):
        ax[2].bar(np.arange(2) + (i - 0.5) * w, [stats[f"{d}_neighbour_mass_freeflow"], stats[f"{d}_neighbour_mass_congested"]],
                  w, color=P.SERIES[i], label=lab[i])
    ax[2].set_xticks(range(2), ["free flow now\n($\\geq$55 mph)", "congested now\n($<$40 mph)"])
    ax[2].set(ylabel="attention on neighbours (1 - self)", title="Attention on neighbours by current state", ylim=(0, 1.15))
    ax[2].grid(axis="x", visible=False)
    ax[2].legend(loc="upper center", ncol=2, fontsize=7.5)
    fig.suptitle(f"ST-GAT attention of sensor {data.sensor_ids[s]} at {times[j]:%a %d %b %H:%M} "
                 f"(speed {cur[j, s]:.0f} mph)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "p2_attention.png")
    plt.close(fig)


# ========================================================== signal policy sim
def signal_policy(data, preds, y):
    """Offline simulation of a congestion-responsive plan switch (no live control).

    At every 5-minute decision time t, a controller at each sensor's upstream
    signal / ramp meter switches to a congestion plan if it expects the speed at
    t+15 min to be below 40 mph. Policies differ only in that expectation:
    reactive (speed now, i.e. persistence), a model's 15-minute forecast, or an
    oracle (the true future). We score how often a congestion onset is
    *anticipated*: the plan is on at some point in the 30 minutes before the speed
    first drops below 40 mph after 30 free-flowing minutes. We also count false
    activations (no congestion within the next 30 minutes) and plan switches.
    """
    test = data.split.test
    th, h = 40.0, 3
    speed = data.speed
    t_idx = test                                                 # decision times
    obs_future = y[:, h - 1]
    policies = {"Reactive (speed now)": None, **{PLAIN[k]: preds[k][:, h - 1] for k in KEY if k in preds},
                "Oracle (true future)": obs_future}
    ffill = FeatureBuilder(data).filled
    rows = []
    # congestion onsets at time tau (on the test anchors' own time axis)
    s = speed[t_idx]                                             # (A, N)
    below = (s > 0) & (s < th)
    prior_free = np.zeros_like(below)
    for k in range(1, 7):
        prev = np.roll(s, k, 0)
        ok = (prev >= th)
        prior_free = ok if k == 1 else prior_free & ok
    prior_free[:6] = False
    onset = below & prior_free
    # congestion within the next 30 minutes (for false-alarm accounting)
    fut = np.zeros_like(below)
    for k in range(1, 7):
        fut |= np.roll(below, -k, 0)
    fut[-6:] = True
    for name, p in policies.items():
        expect = ffill[t_idx] if p is None else np.where(p > 0, p, np.inf)
        on = expect < th
        if name.startswith("Oracle"):
            on = (p > 0) & (p < th)
        ant = np.zeros_like(onset)
        for k in range(1, 7):                                     # plan on 5-30 min before onset
            ant |= np.roll(on, k, 0)
        lead = np.zeros(onset.shape, dtype=float)
        for k in range(6, 0, -1):                                 # earliest activation in the window
            lead = np.where(np.roll(on, k, 0) & (lead == 0), 5 * k, lead)
        hits = ant[onset]
        days = len(test) / 288
        rows.append({"policy": name, "onsets": int(onset.sum()), "anticipated_%": 100 * hits.mean(),
                     "median_lead_min": float(np.median(lead[onset][hits])) if hits.any() else 0.0,
                     "false_activation_%": 100 * float((on & ~fut & ~below).sum() / max(on.sum(), 1)),
                     "switches_per_sensor_day": float(np.abs(np.diff(on.astype(int), axis=0)).sum() / data.num_nodes / days),
                     "plan_on_%": 100 * float(on.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p2_signal_policy.csv", index=False)
    body = [f"{r.policy.replace('→', '$\\to$')} & {r['anticipated_%']:.1f} & {r.median_lead_min:.0f} & "
            f"{r['false_activation_%']:.1f} & {r.switches_per_sensor_day:.1f}" for _, r in df.iterrows()]
    (TEX / "p2_signal.tex").write_text(tex_table(
        body, r"Policy & Antic.\,\% & Lead (min) & False\,\% & Switch/day",
        f"Simulated congestion-plan switching on the test period ({int(df.onsets.iloc[0]):,} congestion "
        r"onsets). Offline replay only; no live control.", "tab:p2signal", "lcccc"))
    print(df.round(2).to_string(index=False))


# =========================================================== learning curves
def learning_curves():
    runs = [k for k in ("stgat_attention_main", "hybrid_attention_main") if (RES / "history" / f"{k}.json").exists()]
    if not runs:
        return
    fig, ax = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3), squeeze=False)
    for a, k in zip(ax[0], runs):
        h = pd.DataFrame(json.loads((RES / "history" / f"{k}.json").read_text()))
        a.plot(h.epoch, h.train_mae, color=P.SERIES[0], label="train (masked MAE, dropout on)")
        a.plot(h.epoch, h.val_mae, color=P.SERIES[1], label="validation")
        b = h.val_mae.idxmin()
        a.scatter([h.epoch[b]], [h.val_mae[b]], s=30, color=P.SERIES[1], zorder=3)
        if "lr" in h:
            for e in h.epoch[h.lr.diff() < 0]:
                a.axvline(e, color=P.TEXT_2, lw=0.6, ls=":")
        a.set(title=PLAIN[k], xlabel="epoch", ylabel="MAE (mph)")
    ax[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "p2_learning_curves.png")
    plt.close(fig)


def main():
    P.setup()
    for d in (FIG, TAB, TEX):
        d.mkdir(parents=True, exist_ok=True)
    data = load_metrla()
    fb = FeatureBuilder(data)
    test = data.split.test
    y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
    tgt = data.index[(test[:, None] + np.arange(1, 13)).ravel()]
    peak = is_peak(tgt.hour.to_numpy(), tgt.day_of_week.to_numpy()).reshape(y.shape[:2])   # (A, 12)

    gs = graph_stats(data.adj, distance_matrix(data.sensor_ids))
    (TAB / "p2_graph_stats.json").write_text(json.dumps(gs, indent=2))
    print("road graph:", gs)
    if (RES / "predictions" / "xgb_seg_base.npy").exists():
        gap_analysis(data, fb)
        linear_neighbour_correction(data, fb)
    preds = load_preds(MAIN)
    main_tables(preds, y)
    regime_tables(preds, y, peak)
    significance(preds, y)
    robustness(data, preds, y)
    ablations(y)
    congestion_maps(data, preds, y)
    attention_study(data)
    signal_policy(data, preds, y)
    learning_curves()


if __name__ == "__main__":
    main()
