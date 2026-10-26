"""Validate, clean and impute METR-LA; write the cleaned file, a quality report and figures.

Outputs
    Dataset/cleaned/METR-LA_clean.h5          cleaned speeds + per-cell issue/fill codes (git-ignored)
    results/tables/cleaning_report.json       validation checks, counts per step, before/after stats
    results/tables/cleaning_validation.csv    one row per integrity check
    results/tables/sensor_quality.csv         per-sensor flagged share and longest gap
    results/tables/imputation_benchmark.csv   MAE of each fill method on artificially hidden blocks
    results/figures/clean_*.png

The models are unaffected: they read the raw file through ``load_metrla``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from traffic import cleaning as C
from traffic import plotting as P
from traffic.data import DATA_DIR, load_locations, load_metrla

plt = P.plt


def describe(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)),
            "median": float(np.median(x)), "max": float(np.max(x))}


def fig_issues(counts: dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [3, 2]})
    names = ["outage", "missing", "spike", "stuck", "out_of_range"]
    labels = ["network outage", "sensor missing (0 mph)", "spike (Hampel)", "stuck sensor", "out of range"]
    pct = [counts["issues_pct"][n] for n in names]
    ax[0].barh(labels[::-1], pct[::-1], color=P.SERIES[0])
    for y, (p, n) in enumerate(zip(pct[::-1], names[::-1])):
        share = "<0.01" if 0 < p < 0.01 else f"{p:.2f}"
        ax[0].text(p + 0.05, y, f"{counts['issues'][n]:,} ({share}%)", va="center", color=P.TEXT_2)
    ax[0].set(xlabel="share of all readings (%)", title="Readings flagged, by issue", xlim=(0, max(pct) * 1.45))
    ax[0].grid(axis="y", visible=False)
    fills = ["original", "linear", "weekly_profile"]
    fp = [counts["fills_pct"][f] for f in fills]
    ax[1].bar(["kept as\nmeasured", "linear\n(gap ≤ 1 h)", "weekly profile\n(gap > 1 h)"], fp,
              color=[P.TEXT_2, P.SERIES[0], P.SERIES[1]])
    for x, p in enumerate(fp):
        ax[1].text(x, p + 1.5, f"{p:.1f}%", ha="center", color=P.TEXT)
    ax[1].set(ylabel="share of readings (%)", title="Source of each cleaned value", ylim=(0, 108))
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_benchmark(bench: pd.DataFrame, max_interp: int):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    order = ["forward_fill", "linear", "weekly_profile", "hybrid (ours)"]
    for j, m in enumerate(order):
        b = bench[bench.method == m]
        style = dict(lw=2.6, color=P.TEXT, marker="o", ms=4, zorder=3) if m.startswith("hybrid") else \
            dict(lw=1.4, color=P.SERIES[j], marker="o", ms=3)
        ax.plot(b.gap_minutes, b.MAE, label=m.replace("_", " "), **style)
    ax.axvline(max_interp * 5, color=P.GRID, lw=1, ls="--", zorder=0)
    ax.text(max_interp * 5 * 1.08, ax.get_ylim()[1] * 0.95, "switch to profile", color=P.TEXT_2, va="top")
    ax.set_xscale("log")
    ticks = bench.gap_minutes.unique()
    ax.set_xticks(ticks, [f"{t // 60} h" if t >= 60 else f"{t} min" for t in ticks])
    ax.minorticks_off()
    ax.set(xlabel="length of hidden gap", ylabel="imputation MAE (mph)",
           title="Imputation error on hidden blocks (val + test period)")
    ax.legend()
    return fig


def fig_examples(raw: np.ndarray, res: C.CleanResult, index: pd.DatetimeIndex, ids: list[str]):
    """Raw vs cleaned for the worst stuck run, the largest spike, and a long sensor gap."""
    issue = res.issue
    def centre_of_longest(mask):
        runs = C.gap_lengths(mask)
        t, i = np.unravel_index(np.argmax(runs), runs.shape)        # first cell of the longest run
        return t + runs[t, i] // 2, i

    t_stuck, i_stuck = centre_of_longest(issue == C.STUCK)
    # the largest spike whose +-3 neighbours are all valid, so the panel shows an isolated glitch
    ok = issue == C.OK
    clear = np.ones_like(ok)
    for k in (1, 2, 3):
        clear[k:-k] &= ok[2 * k:] & ok[:-2 * k]
    dev = np.where((issue == C.SPIKE) & clear, np.abs(raw - res.speed), 0)
    t_spike, i_spike = np.unravel_index(np.argmax(dev), dev.shape)
    gaps = C.gap_lengths(issue == C.MISSING)
    t_gap, i_gap = centre_of_longest((gaps >= 36) & (gaps <= 96))   # a 3-8 h gap fits in one panel
    cases = [("stuck sensor", t_stuck, i_stuck, 72), ("spike", t_spike, i_spike, 36), ("long gap", t_gap, i_gap, 84)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, (title, t, i, half) in zip(axes, cases):
        sl = slice(max(t - half, 0), min(t + half, len(index)))
        tt, r, c, iss = index[sl], raw[sl, i], res.speed[sl, i], issue[sl, i]
        ax.plot(tt, c, color=P.SERIES[0], lw=2, label="cleaned")
        ax.plot(tt, np.where(iss == C.OK, r, np.nan), color=P.TEXT_2, lw=1.1, label="raw (kept)")
        bad = iss != C.OK
        ax.scatter(tt[bad], r[bad], color=P.SERIES[1], s=10, zorder=3, label="raw (flagged)")
        ax.set(title=f"{title} · sensor {ids[i]}", ylim=(-3, 74))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
    axes[0].set_ylabel("speed (mph)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return fig


def main():
    P.setup()
    fig_dir, tab_dir = ROOT / "results" / "figures", ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    d = load_metrla()
    df = pd.DataFrame(d.speed, index=d.index, columns=d.sensor_ids)
    dist = pd.read_csv(DATA_DIR / "distances_la_2012.csv", dtype={"from": str, "to": str})
    checks = C.validate(df, d.sensor_ids, load_locations(), dist)
    pd.DataFrame(checks).to_csv(tab_dir / "cleaning_validation.csv", index=False)
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}  {c['detail']}")
    if not all(c["passed"] for c in checks):
        raise SystemExit("validation failed; fix the raw data before cleaning")

    res = C.clean(d.speed, d.index, d.train_end)
    counts = res.counts()
    assert not np.isnan(res.speed).any(), "imputation left gaps"
    assert np.array_equal(res.speed[res.issue == C.OK], d.speed[res.issue == C.OK]), "valid readings changed"
    path = C.save_clean(res, d.index, d.sensor_ids)

    quality = C.sensor_quality(res, d.sensor_ids)
    quality.round(4).to_csv(tab_dir / "sensor_quality.csv", index=False)
    bench = C.imputation_benchmark(res, d.index, d.train_end)
    bench.round(4).to_csv(tab_dir / "imputation_benchmark.csv", index=False)

    raw_obs = d.speed[d.speed > 0]
    report = {
        "config": vars(res.config),
        "validation": checks,
        "counts": counts,
        "complete_time_steps": {"raw": int((d.speed > 0).all(1).sum()), "clean": int(len(d.index))},
        "raw_observed": describe(raw_obs),
        "clean_all": describe(res.speed),
        "clean_valid_only": describe(res.speed[res.issue == C.OK]),
        "sensors_over_15pct_flagged": quality.loc[quality.flagged_pct > 15, "sensor_id"].tolist(),
        "imputation_mae": bench.pivot(index="gap_minutes", columns="method", values="MAE").round(3)
                               .to_dict(orient="index"),
        "output_file": str(path.relative_to(ROOT)),
    }
    (tab_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    for name, fig in (("clean_issues", fig_issues(counts)),
                      ("clean_imputation_benchmark", fig_benchmark(bench, res.config.max_interp_steps)),
                      ("clean_examples", fig_examples(d.speed, res, d.index, d.sensor_ids))):
        fig.savefig(fig_dir / f"{name}.png")
        plt.close(fig)
        print("saved", name)

    print(f"\nflagged {100 - 100 * counts['valid'] / counts['cells']:.2f}% of readings:",
          ", ".join(f"{k} {v:.2f}%" for k, v in counts["issues_pct"].items()))
    print("filled:", ", ".join(f"{k} {v:.2f}%" for k, v in counts["fills_pct"].items()))
    print("wrote", path.relative_to(ROOT), f"({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
