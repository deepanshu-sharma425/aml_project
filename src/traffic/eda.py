"""Exploratory data analysis: each function returns (figure, statistics dict)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller

from . import plotting as P
from .data import STEPS_PER_DAY, STEPS_PER_WEEK, METRLA, load_locations
from .features import forward_fill, historical_profile, time_of_week

plt = P.plt


def summary(d: METRLA) -> dict:
    s = d.speed
    obs = s[s > 0]
    sp = d.split
    return {
        "time_steps": int(s.shape[0]), "sensors": int(s.shape[1]),
        "start": str(d.index[0]), "end": str(d.index[-1]), "days": int(s.shape[0] // STEPS_PER_DAY),
        "interval_min": 5, "missing_pct": float((s == 0).mean() * 100),
        "all_sensor_outage_steps": int((s == 0).all(1).sum()),
        "speed_mean": float(obs.mean()), "speed_std": float(obs.std()),
        "speed_min": float(obs.min()), "speed_median": float(np.median(obs)), "speed_max": float(obs.max()),
        "pct_at_cap_70": float((obs == 70).mean() * 100),
        "pct_below_40": float((obs < 40).mean() * 100),
        "graph_edges_incl_self": int((d.adj > 0).sum()),
        "graph_edges_excl_self": int((d.adj > 0).sum() - np.count_nonzero(np.diag(d.adj))),
        "split_windows": {"train": len(sp.train), "val": len(sp.val), "test": len(sp.test)},
        "split_dates": {k: [str(d.index[v[0]]), str(d.index[v[-1]])] for k, v in
                        (("train", sp.train), ("val", sp.val), ("test", sp.test))},
    }


def missingness(d: METRLA):
    miss = pd.DataFrame(d.speed == 0, index=d.index)
    daily = miss.resample("D").mean()                                  # days x sensors
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 1]})
    im = ax[0].imshow(daily.T.to_numpy(), aspect="auto", cmap=P.BLUES, vmin=0, vmax=1, interpolation="nearest")
    ticks = np.arange(0, len(daily), 14)
    ax[0].set_xticks(ticks, [daily.index[i].strftime("%d %b") for i in ticks])
    ax[0].set(xlabel="day", ylabel="sensor index", title="Fraction of missing readings per sensor-day")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.03, pad=0.01)
    per_sensor = miss.mean().to_numpy() * 100
    ax[1].hist(per_sensor, bins=30, color=P.SERIES[0], edgecolor="white", linewidth=0.5)
    ax[1].set(xlabel="missing readings (%)", ylabel="sensors", title="Per-sensor missing rate")
    fig.tight_layout()
    day_frac = miss.mean(1).resample("D").mean()
    stats = {
        "per_sensor_missing_pct": {"min": float(per_sensor.min()), "median": float(np.median(per_sensor)),
                                   "max": float(per_sensor.max())},
        "days_over_50pct_missing": [str(x.date()) for x in day_frac[day_frac > 0.5].index],
        "longest_gap_steps_median": int(np.median([_longest_run(c) for c in (d.speed == 0).T])),
    }
    return fig, stats


def _date_axis(ax) -> None:
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def distribution(d: METRLA):
    obs = d.speed[d.speed > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(obs, bins=70, range=(0, 70), color=P.SERIES[0], edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(obs), color=P.TEXT_2, ls="--", lw=1)
    ax.text(np.median(obs) - 1, ax.get_ylim()[1] * 0.9, f"median {np.median(obs):.1f}", ha="right", color=P.TEXT_2)
    ax.set(xlabel="speed (mph)", ylabel="observations", title="Distribution of observed speeds")
    q = np.percentile(obs, [1, 5, 25, 50, 75, 95, 99])
    skew = float(((obs - obs.mean()) ** 3).mean() / obs.std() ** 3)
    return fig, {"percentiles_1_5_25_50_75_95_99": q.round(2).tolist(), "skewness": skew}


def daily_profile(d: METRLA):
    s = pd.DataFrame(np.where(d.speed > 0, d.speed, np.nan), index=d.index)
    city = s.mean(1)
    tod = (d.index.hour * 60 + d.index.minute) / 60
    weekend = d.index.dayofweek >= 5
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    out = {}
    for i, (lab, m) in enumerate((("weekday", ~weekend), ("weekend", weekend))):
        g = city[m].groupby(tod[m])
        mu, lo, hi = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax[0].plot(mu.index, mu.values, color=P.SERIES[i], label=lab)
        ax[0].fill_between(mu.index, lo.values, hi.values, color=P.SERIES[i], alpha=0.15, lw=0)
        out[f"{lab}_min_speed"] = float(mu.min())
        out[f"{lab}_min_at_hour"] = float(mu.idxmin())
    ax[0].set(xlabel="hour of day", ylabel="network mean speed (mph)", title="Daily profile (mean, IQR band)",
              xticks=range(0, 25, 3))
    ax[0].legend()
    hw = city.groupby([d.index.dayofweek, d.index.hour]).mean().unstack()
    im = ax[1].imshow(hw.to_numpy(), aspect="auto", cmap=P.BLUES.reversed(), interpolation="nearest")
    ax[1].set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax[1].set_xticks(range(0, 24, 3))
    ax[1].set(xlabel="hour of day", title="Mean speed by weekday and hour (darker = slower)")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.01, label="mph")
    fig.tight_layout()
    congested = pd.Series(d.speed[(d.speed > 0)] < 40).mean()
    out["pct_below_40mph"] = float(congested * 100)
    return fig, out


def autocorrelation(d: METRLA, nlags: int = STEPS_PER_WEEK):
    end = d.train_end
    filled = forward_fill(d.speed)[:end]
    tow = time_of_week(d.index)
    anomaly = filled - historical_profile(d.speed, tow, end)[tow[:end]]
    raw_acf = np.mean([acf(filled[:, i], nlags=nlags, fft=True) for i in range(d.num_nodes)], 0)
    an_acf = np.mean([acf(anomaly[:, i], nlags=nlags, fft=True) for i in range(d.num_nodes)], 0)
    lags_h = np.arange(nlags + 1) / 12
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lags_h, raw_acf, color=P.SERIES[0], label="raw speed")
    ax.plot(lags_h, an_acf, color=P.SERIES[1], label="de-seasonalised (speed - weekly profile)")
    for k in range(1, 8):
        ax.axvline(24 * k, color=P.GRID, lw=0.8, zorder=0)
    ax.set(xlabel="lag (hours)", ylabel="mean ACF over sensors", title="Autocorrelation (training period)",
           xticks=range(0, 169, 24))
    ax.legend()
    return fig, {"acf_raw_lag_5min": float(raw_acf[1]), "acf_raw_lag_1h": float(raw_acf[12]),
                 "acf_raw_lag_1day": float(raw_acf[STEPS_PER_DAY]), "acf_raw_lag_1week": float(raw_acf[-1]),
                 "acf_anomaly_lag_1h": float(an_acf[12]), "acf_anomaly_lag_1day": float(an_acf[STEPS_PER_DAY])}


def stationarity(d: METRLA, n_sensors: int = 10, seed: int = 0) -> dict:
    """Augmented Dickey-Fuller test on a random subset of sensors (training period)."""
    rng = np.random.default_rng(seed)
    filled = forward_fill(d.speed)[: d.train_end]
    rows = []
    for i in sorted(rng.choice(d.num_nodes, n_sensors, replace=False)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)    # tuple-vs-object return notice
            stat, p, *_ = adfuller(filled[:, i], maxlag=STEPS_PER_DAY // 12, autolag=None)
        rows.append({"sensor": d.sensor_ids[i], "adf_stat": float(stat), "p_value": float(p)})
    return {"adf": rows, "share_rejecting_unit_root_5pct": float(np.mean([r["p_value"] < 0.05 for r in rows]))}


def spatial_correlation(d: METRLA, dist: np.ndarray):
    end = d.train_end
    filled = forward_fill(d.speed)[:end]
    tow = time_of_week(d.index)
    anomaly = filled - historical_profile(d.speed, tow, end)[tow[:end]]
    c_raw, c_an = np.corrcoef(filled.T), np.corrcoef(anomaly.T)
    iu = np.isfinite(dist) & ~np.eye(d.num_nodes, dtype=bool)
    km = dist[iu] / 1000
    bins = np.arange(0, 22, 1.0)
    which = np.digitize(km, bins)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for j, (lab, c) in enumerate((("raw speed", c_raw), ("de-seasonalised", c_an))):
        v = c[iu]
        mids = [bins[k - 1] + 0.5 for k in range(1, len(bins)) if (which == k).sum() > 20]
        mu = [v[which == k].mean() for k in range(1, len(bins)) if (which == k).sum() > 20]
        ax.plot(mids, mu, marker="o", ms=4, color=P.SERIES[j], label=lab)
    ax.set(xlabel="road-network distance (km)", ylabel="mean Pearson correlation",
           title="Correlation between sensor pairs vs. road distance")
    ax.legend()
    nb = (d.adj > 0) & ~np.eye(d.num_nodes, dtype=bool)
    far = ~np.isfinite(dist)
    stats = {"corr_anomaly_graph_neighbours": float(c_an[nb].mean()),
             "corr_anomaly_unconnected_pairs": float(c_an[far & ~np.eye(d.num_nodes, dtype=bool)].mean()),
             "corr_raw_graph_neighbours": float(c_raw[nb].mean()),
             "spearman_distance_vs_anomaly_corr": float(pd.Series(km).corr(pd.Series(c_an[iu]), method="spearman"))}
    return fig, stats


def sensor_map(d: METRLA):
    loc = load_locations().set_index("sensor_id").loc[d.sensor_ids]
    mean_speed = np.nanmean(np.where(d.speed > 0, d.speed, np.nan), 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ii, jj = np.nonzero((d.adj > 0) & ~np.eye(d.num_nodes, dtype=bool))
    for i, j in zip(ii, jj):
        ax.plot(loc.longitude.iloc[[i, j]], loc.latitude.iloc[[i, j]], color=P.GRID, lw=0.6, zorder=1)
    sc = ax.scatter(loc.longitude, loc.latitude, c=mean_speed, cmap=P.BLUES.reversed(), s=18,
                    edgecolor="white", linewidth=0.5, zorder=2)
    fig.colorbar(sc, ax=ax, label="mean speed (mph)", fraction=0.04)
    ax.set(xlabel="longitude", ylabel="latitude", title="207 METR-LA sensors and graph edges")
    ax.set_aspect(1 / np.cos(np.deg2rad(loc.latitude.mean())))
    ax.xaxis.set_major_locator(P.plt.MaxNLocator(4))
    edges = (d.adj > 0) & ~np.eye(d.num_nodes, dtype=bool)
    deg_out = ((d.adj > 0).sum(1) - 1)
    deg_in = ((d.adj > 0).sum(0) - 1)
    return fig, {"out_degree_mean": float(deg_out.mean()), "out_degree_max": int(deg_out.max()),
                 "isolated_sensors_no_out_edges": int((deg_out == 0).sum()),
                 "isolated_sensors_no_in_edges": int((deg_in == 0).sum()),
                 "one_way_edge_fraction": float((edges & ~edges.T).sum() / edges.sum())}


def split_timeline(d: METRLA):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-sensor outage steps
        s = pd.Series(np.nanmean(np.where(d.speed > 0, d.speed, np.nan), 1), index=d.index)
    daily = s.resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(daily.index, daily.values, color=P.TEXT_2, lw=1.2)
    sp = d.split
    stats = {}
    for i, (lab, a) in enumerate((("train", sp.train), ("val", sp.val), ("test", sp.test))):
        lo, hi = d.index[a[0]], d.index[a[-1]]
        ax.axvspan(lo, hi, color=P.SERIES[i], alpha=0.12, lw=0)
        ax.text(lo + (hi - lo) / 2, daily.max() + 0.3, lab, ha="center", color=P.TEXT)
        v = d.speed[a[0]:a[-1] + 1]
        v = v[v > 0]
        stats[lab] = {"mean": float(v.mean()), "std": float(v.std()), "missing_pct":
                      float((d.speed[a[0]:a[-1] + 1] == 0).mean() * 100)}
    ax.set(ylabel="daily mean speed (mph)", title="Chronological 70/10/20 split")
    _date_axis(ax)
    ax.set_ylim(daily.min() - 1, daily.max() + 1.2)
    return fig, stats


def example_week(d: METRLA, sensor: int = 0):
    s = d.speed[: STEPS_PER_WEEK, sensor]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    t = d.index[:STEPS_PER_WEEK]
    ax.plot(t, np.where(s > 0, s, np.nan), color=P.SERIES[0], lw=1)
    miss = s == 0
    ax.scatter(t[miss], np.zeros(miss.sum()) + 2, color=P.SERIES[1], s=8, label="missing (recorded as 0)")
    ax.set(ylabel="speed (mph)", title=f"First week at sensor {d.sensor_ids[sensor]}")
    _date_axis(ax)
    ax.legend(loc="lower right")
    return fig, {}
