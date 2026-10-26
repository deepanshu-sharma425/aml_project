"""Graph WaveNet speed forecasts on the METR-LA test period.

Run from the repository root (so .streamlit/config.toml is picked up):
    streamlit run app/app.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch

# The traffic package lives in src/. Hosted deployments (e.g. Streamlit Community
# Cloud) install requirements.txt but not the package itself, so import it from source.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic import cleaning as C
from traffic.data import DATA_DIR, load_metrla, transition_matrices
from traffic.metrics import evaluate
from traffic.models import GraphWaveNet
from traffic.trainer import Trainer, pick_device

CKPT = ROOT / "checkpoints" / "gwnet_full_main.pt"
# Fixed legend order = fixed categorical slot per model
MODELS = {"gwnet": "Graph WaveNet", "hgb": "Gradient Boosting", "ridge": "Ridge",
          "persistence": "Persistence", "ha": "Historical Average"}
HORIZONS = {"15 min": 3, "30 min": 6, "60 min": 12}

# Congestion bands (mph) use the status palette; each always ships with its label
BANDS = [("Free-flow", 55, "#0ca30c"), ("Slowing", 40, "#fab219"),
         ("Congested", 25, "#ec835a"), ("Severe", 0, "#d03b3b")]

PALETTE = {
    "light": dict(actual="#52514e", forecast="#2a78d6", muted="#898781", grid="#e1e0d9",
                  surface="#fcfcfb", ring="#0b0b0b",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark": dict(actual="#c3c2b7", forecast="#3987e5", muted="#898781", grid="#2c2c2a",
                 surface="#1a1a19", ring="#f0efec",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}

st.set_page_config(page_title="LA Traffic Forecast", page_icon="🚦", layout="wide")


# ---------------------------------------------------------------- data & model
@st.cache_resource(show_spinner="Loading data and model…")
def load():
    data = load_metrla()
    supports = [torch.tensor(s) for s in transition_matrices(data.adj)]
    model = GraphWaveNet(data.num_nodes, supports, adaptive=True, use_graph=True)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    trainer = Trainer(model, data, device=pick_device(), batch_size=64)
    locs = pd.read_csv(DATA_DIR / "graph_sensor_locations.csv", dtype={"sensor_id": str})
    locs = locs.set_index("sensor_id").loc[data.sensor_ids]
    return data, trainer, locs


@st.cache_data(show_spinner="Loading test-set accuracy…")
def test_set_metrics() -> dict[str, dict]:
    """Saved test-set metrics per model; Graph WaveNet is re-scored only if its file is missing."""
    out = {}
    for key, name in MODELS.items():
        f = ROOT / "results" / "metrics" / f"{'gwnet_full_main' if key == 'gwnet' else key}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())["metrics"]
    if "Graph WaveNet" not in out:
        data, trainer, _ = load()
        test = data.split.test
        y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
        out = {"Graph WaveNet": evaluate(trainer.predict(test, batch_size=64), y), **out}
    return out


@st.cache_resource(show_spinner="Checking and cleaning the data…")
def data_quality() -> tuple[C.CleanResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cleaned copy of the speeds (for display only; the model reads the raw data with zeros masked)."""
    data, _, locs = load()
    res = C.clean(data.speed, data.index, data.train_end)
    checks = C.validate(pd.DataFrame(data.speed, index=data.index, columns=data.sensor_ids), data.sensor_ids,
                        locs.reset_index(), pd.read_csv(DATA_DIR / "distances_la_2012.csv",
                                                        dtype={"from": str, "to": str}))
    f = ROOT / "results" / "tables" / "imputation_benchmark.csv"
    bench = pd.read_csv(f) if f.exists() else C.imputation_benchmark(res, data.index, data.train_end)
    return res, pd.DataFrame(checks), C.sensor_quality(res, data.sensor_ids), bench


def band(speed: float) -> tuple[str, str]:
    for name, lo, color in BANDS:
        if speed >= lo:
            return name, color
    return BANDS[-1][0], BANDS[-1][2]


def hex_rgb(h: str) -> list[int]:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def nan_missing(x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    x[x == 0] = np.nan
    return x


def utc(t):
    """Tag naive timestamps as UTC; otherwise Streamlit shifts them by the server's timezone."""
    return t.tz_localize("UTC")


def clock_axis() -> alt.X:
    # Times are tagged UTC (see utc()) and drawn on a UTC scale, so the browser never shifts them
    return alt.X("time:T", title=None, scale=alt.Scale(type="utc"),
                 axis=alt.Axis(format="%H:%M", labelColor=pal["muted"], tickColor=pal["grid"],
                               domainColor=pal["grid"], grid=False))


def speed_axis(lo: float, hi: float) -> alt.Y:
    return alt.Y("speed:Q", title="Speed (mph)", scale=alt.Scale(domain=[lo, hi], nice=False),
                 axis=alt.Axis(labelColor=pal["muted"], titleColor=pal["muted"], gridColor=pal["grid"],
                               domain=False, ticks=False, tickCount=5))


def series_chart(wide: pd.DataFrame, now: pd.Timestamp, height: int, shade_until=None) -> alt.LayerChart:
    """Actual vs. forecast lines with a 'now' marker and a crosshair tooltip."""
    wide = wide.assign(time=utc(pd.DatetimeIndex(wide.time)))
    now, shade_until = utc(now), (utc(shade_until) if shade_until is not None else None)
    long = wide.melt("time", ["Actual", "Forecast"], "series", "speed").dropna()
    vals = long.speed
    lo, hi = max(0.0, np.floor(vals.min() / 5) * 5 - 5), min(80.0, np.ceil(vals.max() / 5) * 5 + 5)
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top", direction="horizontal", labelColor=pal["muted"]),
                      scale=alt.Scale(domain=["Actual", "Forecast"], range=[pal["actual"], pal["forecast"]]))
    base = alt.Chart(long).encode(x=clock_axis(), y=speed_axis(lo, hi), color=color)
    layers = []
    if shade_until is not None:
        layers.append(alt.Chart(pd.DataFrame({"a": [now], "b": [shade_until]})).mark_rect(
            color=pal["forecast"], opacity=0.07).encode(x=alt.X("a:T", scale=alt.Scale(type="utc")), x2="b:T"))
    layers.append(base.mark_line(strokeWidth=2))
    nowdf = pd.DataFrame({"time": [now], "label": ["now"]})
    layers.append(alt.Chart(nowdf).mark_rule(color=pal["muted"], strokeDash=[3, 3]).encode(
        x=alt.X("time:T", scale=alt.Scale(type="utc"))))
    layers.append(alt.Chart(nowdf).mark_text(color=pal["muted"], align="left", dx=4, dy=-6, baseline="top").encode(
        x=alt.X("time:T", scale=alt.Scale(type="utc")), y=alt.value(0), text="label:N"))

    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False, clear="pointerout")
    tip = wide.assign(clock=wide.time.dt.strftime("%a %H:%M"))
    layers.append(alt.Chart(tip).mark_rule(strokeWidth=24, opacity=0).encode(
        x=alt.X("time:T", scale=alt.Scale(type="utc")),
        tooltip=[alt.Tooltip("clock:N", title="Time"),
                 alt.Tooltip("Actual:Q", format=".1f", title="Actual (mph)"),
                 alt.Tooltip("Forecast:Q", format=".1f", title="Forecast (mph)")]).add_params(hover))
    layers.append(alt.Chart(tip).mark_rule(color=pal["muted"], strokeWidth=1).encode(
        x=alt.X("time:T", scale=alt.Scale(type="utc"))).transform_filter(hover))
    layers.append(base.mark_point(filled=True, size=70, stroke=pal["surface"], strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))))
    return alt.layer(*layers).properties(height=height).configure_view(stroke=None)


data, trainer, locs = load()
pal = PALETTE["dark" if st.context.theme.type == "dark" else "light"]
test = data.split.test
test_times = data.index[test]


@st.cache_data(show_spinner=False)
def day_forecasts(day: dt.date) -> tuple[np.ndarray, np.ndarray]:
    anchors = test[test_times.date == day]
    return anchors, trainer.predict(anchors)


def nearest_pos(day: dt.date, tod: dt.time) -> int:
    """Index into the test anchors of the forecast time closest to day + tod."""
    return int(test_times.get_indexer([pd.Timestamp.combine(day, tod)], method="nearest")[0])


def go_to(ts: pd.Timestamp) -> None:
    st.session_state.day, st.session_state.tod = ts.date(), ts.time()


def step(k: int) -> None:
    go_to(test_times[int(np.clip(st.session_state.pos + k, 0, len(test_times) - 1))])


def preset(tod: dt.time) -> None:
    go_to(test_times[nearest_pos(st.session_state.day, tod)])


def pick_sensor(i: int) -> None:
    st.session_state.sensor = i


# ---------------------------------------------------------------- state (defaults, play, map click)
if "day" not in st.session_state:
    st.session_state.update(day=dt.date(2012, 6, 4), tod=dt.time(16, 30), horizon=12,
                            sensor=int(np.argmax(data.speed[test].std(axis=0))))
if "jump" in st.session_state:                      # set by the player at the end of the last run
    go_to(st.session_state.pop("jump"))
clicked = (st.session_state.get("map") or {}).get("selection", {}).get("objects", {}).get("sensors", [])
if clicked and clicked[0]["idx"] != st.session_state.get("last_click"):
    st.session_state.sensor = st.session_state.last_click = clicked[0]["idx"]

# ---------------------------------------------------------------- sidebar: settings form + quick actions
first, last = test_times[0].date(), test_times[-1].date()
with st.sidebar:
    st.markdown("### 🚦 Forecast settings")
    with st.form("settings", border=True):
        st.date_input("Date", min_value=first, max_value=last, key="day", format="DD/MM/YYYY",
                      help=f"Test period: {first:%d %b} – {last:%d %b %Y}")
        st.slider("Time of day", min_value=dt.time(0, 0), max_value=dt.time(23, 55),
                  step=dt.timedelta(minutes=5), format="HH:mm", key="tod")
        st.selectbox("Sensor", range(data.num_nodes), key="sensor",
                     format_func=lambda i: f"{data.sensor_ids[i]}", help="Or click a sensor on the map.")
        st.select_slider("Network map shows", options=list(range(1, 13)), key="horizon",
                         format_func=lambda h: f"+{h * 5} min")
        submitted = st.form_submit_button("Run forecast", type="primary", icon=":material/play_arrow:",
                                          width="stretch")

    pos = nearest_pos(st.session_state.day, st.session_state.tod)
    st.session_state.pos = pos
    t_now = test_times[pos]
    sensor, horizon = st.session_state.sensor, st.session_state.horizon
    if submitted:
        st.toast(f"Forecast updated · {t_now:%a %d %b, %H:%M} · sensor {data.sensor_ids[sensor]}", icon="✅")
    if t_now != pd.Timestamp.combine(st.session_state.day, st.session_state.tod):
        st.caption(f"No data at that time, so showing the nearest: **{t_now:%d %b %H:%M}**")

    st.markdown("**Step through time**")
    b = st.columns(2, gap="small")
    for i, (label, icon, k) in enumerate([("5 min", ":material/chevron_left:", -1),
                                          ("5 min", ":material/chevron_right:", 1),
                                          ("1 hour", ":material/keyboard_double_arrow_left:", -12),
                                          ("1 hour", ":material/keyboard_double_arrow_right:", 12)]):
        b[i % 2].button(label, icon=icon, on_click=step, args=(k,), width="stretch", key=f"step{k}",
                        help=f"{'Back' if k < 0 else 'Forward'} {label}")
    st.markdown("**Jump to**")
    j = st.columns(2, gap="small")
    for i, (label, icon, tod, what) in enumerate([
            ("Morning", ":material/wb_twilight:", dt.time(8, 0), "morning rush"),
            ("Midday", ":material/light_mode:", dt.time(12, 30), "midday"),
            ("Evening", ":material/traffic:", dt.time(17, 30), "evening rush"),
            ("Night", ":material/dark_mode:", dt.time(23, 0), "night")]):
        j[i % 2].button(label, icon=icon, on_click=preset, args=(tod,), width="stretch",
                        help=f"{tod:%H:%M}, {what}, same date")
    st.toggle("Play through the day", key="playing", help="Advance the forecast 5 minutes every second.")

# ---------------------------------------------------------------- live prediction
anchor = int(test[pos])
pred = trainer.predict(np.array([anchor]))[0]                      # (12, N)
past = data.speed[anchor - data.seq_in + 1: anchor + 1]             # (12, N)
future = data.speed[anchor + 1: anchor + data.seq_out + 1]          # (12, N)
past_idx = data.index[anchor - data.seq_in + 1: anchor + 1]
fut_idx = data.index[anchor + 1: anchor + data.seq_out + 1]

now_speed = float(past[-1, sensor])
ms = future[:, sensor] > 0
mae_s = float(np.abs(pred[ms, sensor] - future[ms, sensor]).mean()) if ms.any() else None
m = future > 0
mae_all = float(np.abs(pred[m] - future[m]).mean())
congested = int((pred[horizon - 1] < 40).sum())

with st.sidebar:
    st.markdown("**Find a sensor**")
    observed = past[-1] > 0
    slowest = int(np.argmin(np.where(observed, past[-1], np.inf)))
    drop = int(np.argmax(np.where(observed, past[-1] - pred[-1], -np.inf)))
    err = np.where(m, np.abs(pred - future), np.nan)
    worst = int(np.nanargmax(np.nanmean(np.where(m.any(0), err, 0), 0)))
    rnd = int(np.random.default_rng(pos).integers(data.num_nodes))
    s = st.columns(2, gap="small")
    s[0].button("Slowest", icon=":material/speed:", on_click=pick_sensor, args=(slowest,), width="stretch",
                help=f"Sensor {data.sensor_ids[slowest]}: {past[-1, slowest]:.0f} mph right now")
    s[1].button("Slowing", icon=":material/trending_down:", on_click=pick_sensor, args=(drop,), width="stretch",
                help=f"Sensor {data.sensor_ids[drop]}: forecast to slow by {past[-1, drop] - pred[-1, drop]:.0f} mph")
    s[0].button("Worst", icon=":material/error:", on_click=pick_sensor, args=(worst,), width="stretch",
                help=f"Sensor {data.sensor_ids[worst]}: largest error over the next hour")
    s[1].button("Random", icon=":material/shuffle:", on_click=pick_sensor, args=(rnd,), width="stretch")

# ---------------------------------------------------------------- header
st.markdown("## Los Angeles freeway speed forecast")
st.markdown(
    f"<span style='color:{pal['muted']}'>Graph WaveNet forecast made at "
    f"**{t_now:%A %d %B %Y, %H:%M}** · next 60 minutes · {data.num_nodes} sensors</span>",
    unsafe_allow_html=True)

forecast_tab, day_tab, acc_tab, dq_tab = st.tabs(["Live forecast", "Whole day", "Model accuracy", "Data quality"])

with forecast_tab:
    k = st.columns(5)
    k[0].metric("Speed now", f"{now_speed:.1f} mph" if now_speed > 0 else "—", border=True, height=150,
                chart_data=list(nan_missing(past[:, sensor])), chart_type="area",
                help=None if now_speed > 0 else "No reading at this sensor right now.")
    for col, (label, h) in zip(k[1:4], HORIZONS.items()):
        v = float(pred[h - 1, sensor])
        col.metric(f"Forecast +{label}", f"{v:.1f} mph", border=True, height=150,
                   delta=f"{v - now_speed:+.1f} mph" if now_speed > 0 else None,
                   delta_description="vs. now" if now_speed > 0 else None)
    k[4].metric("Error, next hour", f"{mae_s:.2f} mph" if mae_s is not None else "n/a", border=True, height=150,
                help="Mean absolute error of this sensor's 12 forecasts against what was then measured. "
                     f"Across all sensors for this hour: {mae_all:.2f} mph.")
    if mae_s is None:
        st.info("This sensor reported no readings in the next hour (offline), so the forecast "
                "can't be checked. Pick another sensor, or click one on the map.", icon="ℹ️")

    left, right = st.columns([7, 5], gap="medium")
    with left, st.container(border=True):
        st.markdown(f"**Sensor {data.sensor_ids[sensor]}** · last hour and next hour")
        wide = pd.DataFrame({
            "time": past_idx.append(fut_idx),
            "Actual": np.concatenate([nan_missing(past[:, sensor]), nan_missing(future[:, sensor])]),
            "Forecast": np.concatenate([[np.nan] * (data.seq_in - 1), [now_speed or np.nan], pred[:, sensor]]),
        })
        st.altair_chart(series_chart(wide, t_now, 360, shade_until=fut_idx[-1]), width="stretch")

    with right, st.container(border=True):
        st.markdown(f"**Network in {horizon * 5} min** · {congested} of {data.num_nodes} sensors below 40 mph")
        speeds = pred[horizon - 1]
        bands = [band(v) for v in speeds]
        pts = pd.DataFrame({
            "idx": np.arange(data.num_nodes), "sensor": data.sensor_ids,
            "lat": locs.latitude.to_numpy(), "lon": locs.longitude.to_numpy(),
            "speed": [f"{v:.1f} mph" for v in speeds], "status": [b[0] for b in bands],
            "color": [hex_rgb(b[1]) + [230] for b in bands],
        })
        layer = pdk.Layer("ScatterplotLayer", pts, id="sensors", get_position=["lon", "lat"],
                          get_fill_color="color", get_radius=260, radius_min_pixels=4, radius_max_pixels=12,
                          stroked=True, get_line_color=hex_rgb(pal["surface"]), line_width_min_pixels=1,
                          pickable=True, auto_highlight=True)
        ring = pdk.Layer("ScatterplotLayer", pts.iloc[[sensor]], id="selected", get_position=["lon", "lat"],
                         filled=False, stroked=True, get_line_color=hex_rgb(pal["ring"]),
                         line_width_min_pixels=2.5, get_radius=700, radius_min_pixels=10)
        view = pdk.ViewState(latitude=float(locs.latitude.mean()), longitude=float(locs.longitude.mean()), zoom=10.2)
        deck = pdk.Deck(layers=[layer, ring], initial_view_state=view, map_style=None,
                        tooltip={"html": "<b>Sensor {sensor}</b><br/>{speed} · {status}"})
        st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="map", height=330)
        counts = pd.Series([b[0] for b in bands]).value_counts()
        st.markdown(" &nbsp; ".join(
            f"<span style='white-space:nowrap'><span style='display:inline-block;width:10px;height:10px;"
            f"border-radius:50%;background:{c};margin-right:5px'></span>{name} ≥{lo} "
            f"<span style='color:{pal['muted']}'>({counts.get(name, 0)})</span></span>".replace(" ≥0", " <25")
            for name, lo, c in BANDS), unsafe_allow_html=True)

with day_tab:
    day = t_now.date()
    day_anchors, day_pred = day_forecasts(day)
    st.markdown(f"**Sensor {data.sensor_ids[sensor]}** · {day:%A %d %B} · each point is a forecast made "
                f"15 or 60 minutes earlier, against what was measured")
    for h in (2, 11):
        with st.container(border=True):
            st.markdown(f"**{(h + 1) * 5} minutes ahead**")
            wide = pd.DataFrame({"time": data.index[day_anchors + h + 1],
                                 "Actual": nan_missing(data.speed[day_anchors + h + 1, sensor]),
                                 "Forecast": day_pred[:, h, sensor]})
            st.altair_chart(series_chart(wide, t_now, 230), width="stretch")

with acc_tab:
    metrics = test_set_metrics()
    best_base = min((v["avg"]["MAE"], k) for k, v in metrics.items() if k != "Graph WaveNet")
    gain = 100 * (1 - metrics["Graph WaveNet"]["avg"]["MAE"] / best_base[0])
    a, b = st.columns([1, 2], gap="medium")
    with a:
        st.metric("Graph WaveNet average error", f"{metrics['Graph WaveNet']['avg']['MAE']:.2f} mph", border=True,
                  delta=f"{gain:.1f}% lower than {best_base[1]}", delta_color="off", delta_arrow="off")
        st.caption("Whole test period: 6,850 forecast times × 207 sensors × 12 steps. "
                   "Missing readings are ignored. Lower is better.")
    with b, st.container(border=True):
        head, pick = st.columns([3, 2], vertical_alignment="center")
        unit = {"MAE": "mph", "RMSE": "mph", "MAPE": "%"}
        metric = pick.segmented_control("Metric", list(unit), default="MAE", label_visibility="collapsed") or "MAE"
        head.markdown(f"**{metric} by forecast horizon**")
        names = [n for n in MODELS.values() if n in metrics]
        long = pd.DataFrame([{"model": n, "horizon": h, "minutes": int(h.split()[0]), "value": metrics[n][h][metric]}
                             for n in names for h in HORIZONS])
        hover = alt.selection_point(fields=["model"], on="pointerover", clear="pointerout")
        enc = dict(x=alt.X("minutes:Q", title="Minutes ahead", scale=alt.Scale(domain=[12, 63]),
                           axis=alt.Axis(values=[15, 30, 60], labelColor=pal["muted"], titleColor=pal["muted"],
                                         grid=False, domainColor=pal["grid"], tickColor=pal["grid"])),
                   y=alt.Y("value:Q", title=f"{metric} ({unit[metric]})", scale=alt.Scale(zero=False),
                           axis=alt.Axis(labelColor=pal["muted"], titleColor=pal["muted"],
                                         gridColor=pal["grid"], domain=False, ticks=False)),
                   color=alt.Color("model:N", title=None, sort=names,
                                   scale=alt.Scale(domain=names, range=pal["series"][:len(names)]),
                                   legend=alt.Legend(orient="right", labelColor=pal["muted"])),
                   opacity=alt.condition(hover, alt.value(1), alt.value(0.85)))
        tips = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("horizon:N", title="Horizon"),
                alt.Tooltip("value:Q", format=".3f", title=f"{metric} ({unit[metric]})")]
        chart = (alt.Chart(long).mark_line(strokeWidth=2).encode(**enc)
                 + alt.Chart(long).mark_point(filled=True, size=80, stroke=pal["surface"], strokeWidth=2)
                 .encode(**enc, tooltip=tips).add_params(hover))
        st.altair_chart(chart.properties(height=260).configure_view(stroke=None), width="stretch")

    table = pd.DataFrame({n: {("average" if h == "avg" else h): d[metric] for h, d in metrics[n].items()}
                          for n in names}).T
    st.markdown(f"**{metric} ({unit[metric]})** · best per column in bold")
    st.dataframe(table.style.format("{:.3f}").highlight_min(axis=0, props="font-weight:700"),
                 width="stretch")

with dq_tab:
    clean, checks, quality, bench = data_quality()
    counts = clean.counts()
    flagged = 100 * (1 - counts["valid"] / counts["cells"])
    st.caption("A validated, cleaned copy of the data for analysis. The model itself reads the raw readings "
               "with missing values masked, the standard METR-LA protocol, so its scores stay comparable "
               "with published results.")
    k = st.columns(4)
    k[0].metric("Integrity checks passed", f"{int(checks.passed.sum())} / {len(checks)}", border=True,
                help="Timestamps, 5-minute grid, sensor order, value range, locations and road distances.")
    k[1].metric("Readings flagged", f"{flagged:.2f}%", border=True,
                help=f"{counts['cells'] - counts['valid']:,} of {counts['cells']:,} readings")
    k[2].metric("Network-wide outages", f"{int(C.detect_outages(data.speed).sum()):,} steps", border=True,
                help="Time steps at which every sensor reported 0 mph.")
    k[3].metric("Glitches removed", f"{counts['issues']['spike'] + counts['issues']['stuck']:,}", border=True,
                help=f"{counts['issues']['spike']:,} one-step spikes (Hampel filter) and "
                     f"{counts['issues']['stuck']:,} stuck-sensor repeats.")

    with st.container(border=True):
        day = t_now.date()
        rows = np.flatnonzero(data.index.date == day)
        st.markdown(f"**Sensor {data.sensor_ids[sensor]}** · {day:%A %d %B} · raw readings vs. cleaned series")
        iss = clean.issue[rows, sensor]
        wide = pd.DataFrame({"time": utc(data.index[rows]), "Raw": np.where(iss == C.OK, data.speed[rows, sensor], np.nan),
                             "Cleaned": clean.speed[rows, sensor],
                             "issue": [C.ISSUES.get(int(c), "valid") for c in iss],
                             "fill": [C.FILLS[int(c)] for c in clean.filled[rows, sensor]]})
        long = wide.melt(["time", "issue", "fill"], ["Cleaned", "Raw"], "series", "speed").dropna()
        color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top", labelColor=pal["muted"]),
                          scale=alt.Scale(domain=["Cleaned", "Raw"], range=[pal["forecast"], pal["actual"]]))
        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=clock_axis(), y=speed_axis(0, 75), color=color)
        bad = wide[wide.issue != "valid"].assign(speed=data.speed[rows, sensor][iss != C.OK])
        marks = alt.Chart(bad).mark_point(shape="cross", size=40, color=pal["series"][1], filled=True).encode(
            x=clock_axis(), y=speed_axis(0, 75),
            tooltip=[alt.Tooltip("time:T", format="%H:%M", title="Time"), alt.Tooltip("speed:Q", title="Raw (mph)"),
                     alt.Tooltip("issue:N", title="Issue"), alt.Tooltip("fill:N", title="Filled by")])
        st.altair_chart(alt.layer(lines, marks).properties(height=260).configure_view(stroke=None), width="stretch")
        n_bad = int((iss != C.OK).sum())
        st.caption(f"{n_bad} of {len(rows)} readings flagged this day (orange crosses). "
                   "Gaps up to 1 hour are filled by linear interpolation, longer ones by the sensor's "
                   "usual speed for that time of week, learned from the training period only.")

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.markdown("**Readings flagged, by issue**")
        names = {"outage": "Network outage", "missing": "Sensor missing", "spike": "Spike",
                 "stuck": "Stuck sensor", "out_of_range": "Out of range"}
        bars = pd.DataFrame({"issue": list(names.values()),
                             "count": [counts["issues"][n] for n in names],
                             "pct": [counts["issues_pct"][n] for n in names]})
        st.altair_chart(alt.Chart(bars).mark_bar(color=pal["forecast"], cornerRadiusEnd=3).encode(
            x=alt.X("pct:Q", title="Share of all readings (%)",
                    axis=alt.Axis(labelColor=pal["muted"], titleColor=pal["muted"], gridColor=pal["grid"])),
            y=alt.Y("issue:N", sort=None, title=None, axis=alt.Axis(labelColor=pal["muted"], ticks=False, domain=False)),
            tooltip=[alt.Tooltip("issue:N", title="Issue"), alt.Tooltip("count:Q", format=",", title="Readings"),
                     alt.Tooltip("pct:Q", format=".3f", title="Share (%)")])
            .properties(height=220).configure_view(stroke=None), width="stretch")
    with b, st.container(border=True):
        st.markdown("**Imputation error on hidden gaps** · validation + test period")
        methods = ["hybrid (ours)", "linear", "weekly_profile", "forward_fill"]
        bl = bench[bench.method.isin(methods)]
        st.altair_chart(alt.Chart(bl).mark_line(point=True).encode(
            x=alt.X("gap_minutes:Q", title="Gap length (minutes, log scale)", scale=alt.Scale(type="log"),
                    axis=alt.Axis(values=sorted(bl.gap_minutes.unique()), labelColor=pal["muted"],
                                  titleColor=pal["muted"], grid=False)),
            y=alt.Y("MAE:Q", title="MAE (mph)", axis=alt.Axis(labelColor=pal["muted"], titleColor=pal["muted"],
                                                               gridColor=pal["grid"])),
            color=alt.Color("method:N", title=None, sort=methods, legend=alt.Legend(orient="top", labelColor=pal["muted"]),
                            scale=alt.Scale(domain=methods, range=pal["series"][:4])),
            strokeWidth=alt.condition(alt.datum.method == "hybrid (ours)", alt.value(3), alt.value(1.5)),
            tooltip=[alt.Tooltip("method:N", title="Method"), alt.Tooltip("gap_minutes:Q", title="Gap (min)"),
                     alt.Tooltip("MAE:Q", format=".2f", title="MAE (mph)")])
            .properties(height=220).configure_view(stroke=None), width="stretch")

    c, q = st.columns([2, 3], gap="medium")
    with c:
        st.markdown("**Integrity checks**")
        st.dataframe(checks.assign(passed=checks.passed.map({True: "✅", False: "❌"})),
                     hide_index=True, width="stretch")
    with q:
        st.markdown("**Least reliable sensors** · share of readings flagged")
        st.dataframe(quality[["sensor_id", "flagged_pct", "missing_pct", "spike_pct", "stuck_pct",
                              "longest_gap_hours"]].head(15).style.format(precision=2),
                     hide_index=True, width="stretch")

if st.session_state.playing and pos + 1 < len(test_times):
    time.sleep(1.0)
    st.session_state.jump = test_times[pos + 1]
    st.rerun()
