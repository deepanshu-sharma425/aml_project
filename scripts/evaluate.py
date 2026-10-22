"""Aggregate saved predictions into tables, significance tests and figures.

Run after ``run_classical.py`` and ``run_deep_experiments.sh``. Writes CSV
tables to results/tables, LaTeX tables to report/tables (so the report always
quotes the numbers produced here) and figures to results/figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from traffic import plotting as P
from traffic.data import load_locations, load_metrla
from traffic.metrics import HORIZONS, diebold_mariano, evaluate, per_step_abs_error

RES = ROOT / "results"
FIG, TAB, TEX = RES / "figures", RES / "tables", ROOT / "report" / "tables"
plt = P.plt

MAIN = {  # file stem -> display name (order = table order, also fixes colour slots)
    "ha": "Historical Average", "persistence": "Persistence", "ridge": "Ridge (features)",
    "hgb": "Gradient Boosting (features)", "lstm_main": "LSTM", "gwnet_full_main": "Graph WaveNet",
}
ABLATION = {
    "gwnet_no_graph_ablation": "No graph (TCN only)",
    "gwnet_no_adaptive_ablation": "Road graph only",
    "gwnet_adaptive_only_ablation": "Adaptive graph only",
    "gwnet_full_ablation": "Road + adaptive (full)",
}
# Published METR-LA numbers (same split / masking protocol) for context.
LITERATURE = [
    ("HA [Li et al. 2018]", [4.16, 7.80, 13.0, 4.16, 7.80, 13.0, 4.16, 7.80, 13.0]),
    ("FC-LSTM [Li et al. 2018]", [3.44, 6.30, 9.60, 3.77, 7.23, 10.90, 4.37, 8.69, 13.20]),
    ("DCRNN [Li et al. 2018]", [2.77, 5.38, 7.30, 3.15, 6.45, 8.80, 3.60, 7.60, 10.50]),
    ("STGCN [Yu et al. 2018]$^\\dagger$", [2.88, 5.74, 7.62, 3.47, 7.24, 9.57, 4.59, 9.40, 12.70]),
    ("Graph WaveNet [Wu et al. 2019]", [2.69, 5.15, 6.90, 3.07, 6.22, 8.37, 3.53, 7.37, 10.01]),
]


def load_preds(names):
    out = {}
    for k in names:
        f = RES / "predictions" / f"{k}.npy"
        if f.exists():
            out[k] = np.load(f)
        else:
            print(f"missing predictions: {k}")
    return out


def metrics_frame(preds, y, labels) -> pd.DataFrame:
    rows = []
    for k, p in preds.items():
        m = evaluate(p, y)
        row = {"model": labels[k]}
        for hz in list(HORIZONS) + ["avg"]:
            for met in ("MAE", "RMSE", "MAPE"):
                row[f"{hz} {met}"] = m[hz][met]
        rows.append(row)
    return pd.DataFrame(rows)


def latex_table(df: pd.DataFrame, caption: str, label: str, literature=None) -> str:
    hz = list(HORIZONS)
    best = {c: df[c].min() for c in df.columns if c != "model"}
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "ccc" * len(hz) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{h}}}" for h in hz) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(hz))),
             "Model & " + " & ".join(["MAE", "RMSE", "MAPE (\\%)"] * len(hz)) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for h in hz:
            for met in ("MAE", "RMSE", "MAPE"):
                v = r[f"{h} {met}"]
                s = f"{v:.2f}"
                cells.append(rf"\textbf{{{s}}}" if np.isclose(v, best[f"{h} {met}"]) else s)
        lines.append(f"{r['model']} & " + " & ".join(cells) + r" \\")
    if literature:
        lines += [r"\midrule", rf"\multicolumn{{{1 + 3 * len(hz)}}}{{l}}{{\emph{{Published results (same dataset and protocol)}}}} \\"]
        for name, vals in literature:
            lines.append(f"{name} & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def block_bootstrap_ci(err_sum, cnt, block=288, n_boot=1000, seed=0):
    """95% CI of overall MAE resampling whole days (keeps serial dependence)."""
    rng = np.random.default_rng(seed)
    nb = len(err_sum) // block
    es = err_sum[: nb * block].reshape(nb, block).sum(1)
    cs = cnt[: nb * block].reshape(nb, block).sum(1)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, nb, nb)
        stats.append(es[idx].sum() / cs[idx].sum())
    return np.percentile(stats, [2.5, 97.5])


def main():
    P.setup()
    for d in (FIG, TAB, TEX):
        d.mkdir(parents=True, exist_ok=True)
    data = load_metrla()
    test = data.split.test
    y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
    mask = y > 0
    preds = load_preds(MAIN)
    colors = {k: P.SERIES[i] for i, k in enumerate(MAIN)}

    # --- main table ------------------------------------------------------
    df = metrics_frame(preds, y, MAIN)
    df.to_csv(TAB / "main_results.csv", index=False)
    (TEX / "main_results.tex").write_text(latex_table(
        df, "Test-set performance on METR-LA (our runs, top) and published results (bottom). "
            "Best of our models in bold. $^\\dagger$As reported by Wu et al. (2019).",
        "tab:main", LITERATURE))
    print(df.round(3).to_string(index=False))

    # --- bootstrap CIs on average MAE -----------------------------------
    ci_rows = []
    for k, p in preds.items():
        e = np.where(mask, np.abs(p - y), 0).sum((1, 2))
        c = mask.sum((1, 2))
        lo, hi = block_bootstrap_ci(e, c)
        ci_rows.append({"model": MAIN[k], "avg_MAE": e.sum() / c.sum(), "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ci_rows).to_csv(TAB / "bootstrap_ci.csv", index=False)

    # --- Diebold-Mariano tests --------------------------------------------
    dm_rows = []
    ref = "gwnet_full_main" if "gwnet_full_main" in preds else None
    pairs = [(ref, k) for k in preds if ref and k != ref]
    if {"hgb", "ridge"} <= preds.keys():
        pairs.append(("hgb", "ridge"))
    if {"lstm_main", "hgb"} <= preds.keys():
        pairs.append(("lstm_main", "hgb"))
    for a, b in pairs:
        for name, h in HORIZONS.items():
            ea, eb = per_step_abs_error(preds[a], y, h), per_step_abs_error(preds[b], y, h)
            stat, pv = diebold_mariano(ea, eb, lag=h)
            dm_rows.append({"model_a": MAIN[a], "model_b": MAIN[b], "horizon": name,
                            "mae_a": np.nanmean(ea), "mae_b": np.nanmean(eb), "dm_stat": stat, "p_value": pv})
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(TAB / "diebold_mariano.csv", index=False)
    if len(dm):
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Diebold--Mariano tests (absolute-error loss, Newey--West variance). "
                 r"Negative statistic: model A is more accurate.}", r"\label{tab:dm}",
                 r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrr}", r"\toprule",
                 r"Model A vs.\ Model B & Horizon & DM & $p$ \\", r"\midrule"]
        for _, r in dm.iterrows():
            pv = "$<10^{-4}$" if r.p_value < 1e-4 else f"{r.p_value:.3f}"
            lines.append(f"{r.model_a} vs.\\ {r.model_b} & {r.horizon} & {r.dm_stat:.2f} & {pv} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TEX / "dm_tests.tex").write_text("\n".join(lines))

    # --- MAE vs horizon ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    steps = np.arange(1, 13)
    for k, p in preds.items():
        mae = [np.abs(p[:, h][mask[:, h]] - y[:, h][mask[:, h]]).mean() for h in range(12)]
        ax.plot(steps * 5, mae, marker="o", ms=3.5, color=colors[k], label=MAIN[k])
    ax.set(xlabel="forecast horizon (minutes)", ylabel="test MAE (mph)", title="Error growth with horizon",
           xticks=steps[1::2] * 5)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "res_mae_vs_horizon.png")
    plt.close(fig)

    # --- MAE by hour of day (60-min horizon) ------------------------------
    hours = data.index[test + 12].hour.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k in ("ha", "hgb", "lstm_main", "gwnet_full_main"):
        if k not in preds:
            continue
        e = np.where(mask[:, 11], np.abs(preds[k][:, 11] - y[:, 11]), np.nan)
        by_h = pd.DataFrame(e).groupby(hours).mean().mean(axis=1)
        ax.plot(by_h.index, by_h.values, color=colors[k], label=MAIN[k])
    ax.set(xlabel="hour of day (target time)", ylabel="MAE at 60 min (mph)",
           title="When are forecasts hardest?", xticks=range(0, 24, 3))
    ax.legend(fontsize=7)
    fig.savefig(FIG / "res_mae_by_hour.png")
    plt.close(fig)

    # --- example forecasts on one congested sensor ------------------------
    day = 2 * 288                                     # third test day
    win = slice(day, day + 288)
    sens = int(np.argmax(((y[win, 11] > 0) & (y[win, 11] < 40)).sum(0)))
    t = data.index[test[win] + 12]
    fig, ax = plt.subplots(figsize=(8, 3))
    truth = np.where(y[win, 11, sens] > 0, y[win, 11, sens], np.nan)
    ax.plot(t, truth, color=P.TEXT, lw=1.8, label="observed")
    for k in ("ha", "hgb", "gwnet_full_main"):
        if k in preds:
            ax.plot(t, preds[k][win, 11, sens], color=colors[k], lw=1.3, label=f"{MAIN[k]} (60 min ahead)")
    ax.set(ylabel="speed (mph)", title=f"60-minute-ahead forecasts, sensor {data.sensor_ids[sens]}, "
                                       f"{t[0]:%a %d %b %Y}")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(FIG / "res_example_forecast.png")
    plt.close(fig)

    # --- robustness: windows whose latest input is missing ----------------
    rob = []
    last_missing = data.speed[test] == 0                         # (A, N)
    for k, p in preds.items():
        for lab, sel in (("latest input observed", ~last_missing), ("latest input missing", last_missing)):
            m = mask[:, 11] & sel
            rob.append({"model": MAIN[k], "condition": lab,
                        "MAE_60min": float(np.abs(p[:, 11][m] - y[:, 11][m]).mean()), "n": int(m.sum())})
    pd.DataFrame(rob).to_csv(TAB / "robustness_missing.csv", index=False)

    # --- per-sensor error map ---------------------------------------------
    if "gwnet_full_main" in preds:
        p = preds["gwnet_full_main"]
        e = np.where(mask[:, 11], np.abs(p[:, 11] - y[:, 11]), np.nan)
        sensor_mae = np.nanmean(e, 0)
        loc = load_locations().set_index("sensor_id").loc[data.sensor_ids]
        fig, ax = plt.subplots(figsize=(5.5, 4.3))
        sc = ax.scatter(loc.longitude, loc.latitude, c=sensor_mae, cmap=P.BLUES, s=22,
                        edgecolor="white", linewidth=0.5)
        fig.colorbar(sc, ax=ax, label="MAE at 60 min (mph)", fraction=0.04)
        ax.set(xlabel="longitude", ylabel="latitude", title="Graph WaveNet error per sensor")
        ax.set_aspect(1 / np.cos(np.deg2rad(loc.latitude.mean())))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        fig.savefig(FIG / "res_sensor_error_map.png")
        plt.close(fig)
        pd.DataFrame({"sensor": data.sensor_ids, "mae_60": sensor_mae}).to_csv(TAB / "sensor_mae_gwnet.csv", index=False)

    # --- learning curves ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for a, k in zip(ax, ("gwnet_full_main", "lstm_main")):
        f = RES / "history" / f"{k}.json"
        if not f.exists():
            continue
        h = pd.DataFrame(json.loads(f.read_text()))
        a.plot(h.epoch, h.train_mae, color=P.SERIES[0], label="train (masked MAE)")
        a.plot(h.epoch, h.val_mae, color=P.SERIES[1], label="validation")
        best = h.val_mae.idxmin()
        a.scatter([h.epoch[best]], [h.val_mae[best]], s=30, color=P.SERIES[1], zorder=3)
        a.set(title=MAIN[k], xlabel="epoch")
    ax[0].set_ylabel("MAE (mph)")
    if ax[0].lines:
        ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "res_learning_curves.png")
    plt.close(fig)

    # --- Graph WaveNet ablation -------------------------------------------
    abl = load_preds(ABLATION)
    if abl:
        da = metrics_frame(abl, y, ABLATION)
        info = {k: json.loads((RES / "metrics" / f"{k}.json").read_text()) for k in abl}
        da["params"] = [info[k]["params"] for k in abl]
        da.to_csv(TAB / "gwnet_ablation.csv", index=False)
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Graph WaveNet ablation (identical 8-epoch budget). Test MAE (mph).}",
                 r"\label{tab:ablation}", r"\begin{tabular}{lrccc}", r"\toprule",
                 r"Spatial component & Params & 15 min & 30 min & 60 min \\", r"\midrule"]
        for _, r in da.iterrows():
            lines.append(f"{r.model} & {r.params:,} & {r['15 min MAE']:.2f} & {r['30 min MAE']:.2f} & {r['60 min MAE']:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TEX / "ablation.tex").write_text("\n".join(lines))
        print(da.round(3).to_string(index=False))

    # --- learned adaptive adjacency vs road graph -------------------------
    f = RES / "predictions" / "gwnet_full_main_adaptive_adj.npy"
    if f.exists():
        A = np.load(f)
        road = (data.adj > 0) & ~np.eye(data.num_nodes, dtype=bool)
        k = 5
        top = np.argsort(-A, 1)[:, :k]
        hit = np.mean([road[i, top[i]].mean() for i in range(len(A))])
        dens = road.mean()
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        for a, (M, ttl) in zip(ax, ((data.adj, "Road-distance adjacency"), (A, "Learned adaptive adjacency"))):
            im = a.imshow(M, cmap=P.BLUES, interpolation="nearest",
                          vmax=np.percentile(M[M > 0], 99) if (M > 0).any() else None)
            a.set(title=ttl, xlabel="target sensor", ylabel="source sensor")
            a.grid(False)
            fig.colorbar(im, ax=a, fraction=0.045)
        fig.tight_layout()
        fig.savefig(FIG / "res_adjacency.png")
        plt.close(fig)
        (TAB / "adaptive_adj_stats.json").write_text(json.dumps(
            {"top5_overlap_with_road_graph": hit, "road_graph_density": dens,
             "lift_over_random": hit / dens,
             "mean_entropy_rows": float(-(A * np.log(A + 1e-12)).sum(1).mean()),
             "max_entropy": float(np.log(len(A)))}, indent=2))

    # --- feature engineering figures --------------------------------------
    f = TAB / "permutation_importance.csv"
    if f.exists():
        imp = pd.read_csv(f)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=False)
        for a, h, c in zip(ax, (3, 12), (P.SERIES[0], P.SERIES[1])):
            s = imp[imp.horizon == h].sort_values("importance").tail(12)
            a.barh(s.feature, s.importance, xerr=s["std"], color=c, height=0.6)
            a.set(title=f"Permutation importance, {h * 5} min ahead", xlabel="increase in MAE (mph)")
            a.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(FIG / "fe_permutation_importance.png")
        plt.close(fig)
    f = TAB / "feature_ablation.csv"
    if f.exists():
        ab = pd.read_csv(f)
        labels = ["lags", "+window stats", "+calendar", "+weekly profile", "+spatial", "+sensor stats"]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i, h in enumerate((3, 6, 12)):
            s = ab[ab.horizon == h].sort_values("n_groups")
            ax.plot(s.n_groups, s.val_mae, marker="o", ms=4, color=P.SERIES[i], label=f"{h * 5} min")
        ax.set_xticks(range(1, 7), labels, rotation=20, ha="right")
        ax.set(ylabel="validation MAE (mph)", title="Cumulative feature-group ablation (gradient boosting)")
        ax.legend(title="horizon")
        fig.savefig(FIG / "fe_ablation.png")
        plt.close(fig)
        piv = ab.pivot(index="n_groups", columns="horizon", values="val_mae")
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Cumulative feature-group ablation for gradient boosting (validation MAE, mph).}",
                 r"\label{tab:feat_ablation}", r"\begin{tabular}{lccc}", r"\toprule",
                 r"Feature groups & 15 min & 30 min & 60 min \\", r"\midrule"]
        for n, lab in zip(piv.index, labels):
            lines.append(f"{lab} & " + " & ".join(f"{piv.loc[n, h]:.3f}" for h in (3, 6, 12)) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TEX / "feature_ablation.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
