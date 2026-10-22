"""Local demo: Graph WaveNet speed forecasts on the METR-LA test period.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from traffic.data import DATA_DIR, load_metrla, transition_matrices
from traffic.metrics import evaluate
from traffic.models import GraphWaveNet
from traffic.trainer import Trainer, pick_device

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "gwnet_full_main.pt"
BASELINES = {"ha": "Historical Average", "persistence": "Persistence",
             "ridge": "Ridge (features)", "hgb": "Gradient Boosting (features)"}

st.set_page_config(page_title="METR-LA Forecasting", layout="wide")


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


@st.cache_data(show_spinner="Scoring the full test set (≈15 s, once)…")
def test_set_scores() -> pd.DataFrame:
    data, trainer, _ = load()
    test = data.split.test
    pred = trainer.predict(test, batch_size=64)
    y = data.speed[test[:, None] + np.arange(1, data.seq_out + 1)]
    rows = {"Graph WaveNet": evaluate(pred, y)}
    for key, name in BASELINES.items():
        f = ROOT / "results" / "metrics" / f"{key}.json"
        if f.exists():
            rows[name] = json.loads(f.read_text())["metrics"]
    table = {name: {f"{h} {k}": v for h, d in m.items() for k, v in d.items()} for name, m in rows.items()}
    return pd.DataFrame(table).T.sort_values("avg MAE")


def nan_missing(x: np.ndarray) -> np.ndarray:
    x = x.astype(float).copy()
    x[x == 0] = np.nan
    return x


data, trainer, locs = load()
test = data.split.test
test_times = data.index[test]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Forecast settings")
st.sidebar.caption(f"Model: Graph WaveNet · checkpoint `{CKPT.name}` · device `{trainer.device}`")
days = sorted(set(test_times.date))
day = st.sidebar.selectbox("Date (test period)", days,
                           format_func=lambda d: d.strftime("%a %d %b %Y"))
day_times = test_times[test_times.date == day]
# Default to 16:30, the start of the evening slowdown, when the day has it
default = next((t for t in day_times if (t.hour, t.minute) == (16, 30)), day_times[len(day_times) // 2])
t_now = st.sidebar.select_slider("Current time (forecast made at)", options=list(day_times),
                                 value=default, format_func=lambda t: t.strftime("%H:%M"))
busiest = int(np.argmax(data.speed[test].std(axis=0)))
sensor = st.sidebar.selectbox("Sensor", range(data.num_nodes), index=busiest,
                              format_func=lambda i: data.sensor_ids[i])
horizon = st.sidebar.select_slider("Map horizon", options=list(range(1, 13)), value=3,
                                   format_func=lambda h: f"{h * 5} min ahead")

# ---------------------------------------------------------------- live prediction
anchor = int(test[test_times.get_loc(t_now)])
pred = trainer.predict(np.array([anchor]))[0]                      # (12, N)
past = data.speed[anchor - data.seq_in + 1: anchor + 1]             # (12, N)
future = data.speed[anchor + 1: anchor + data.seq_out + 1]          # (12, N)
past_idx = data.index[anchor - data.seq_in + 1: anchor + 1]
fut_idx = data.index[anchor + 1: anchor + data.seq_out + 1]

st.title("Traffic speed forecasting — METR-LA")
st.write(f"Forecast made at **{t_now:%a %d %b %Y, %H:%M}** for the next hour, from the last hour of "
         f"readings at all {data.num_nodes} sensors.")

m = future > 0
mae_all = float(np.abs(pred[m] - future[m]).mean()) if m.any() else float("nan")
ms = future[:, sensor] > 0
mae_s = float(np.abs(pred[ms, sensor] - future[ms, sensor]).mean()) if ms.any() else float("nan")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current speed", f"{past[-1, sensor]:.1f} mph" if past[-1, sensor] > 0 else "missing")
c2.metric("Forecast in 60 min", f"{pred[-1, sensor]:.1f} mph")
c3.metric("MAE, this sensor (next hour)", f"{mae_s:.2f} mph" if ms.any() else "n/a (no readings)")
c4.metric("MAE, all sensors (next hour)", f"{mae_all:.2f} mph" if m.any() else "n/a")
if not ms.any():
    st.info("This sensor has no readings in the next hour (offline), so its forecast can't be scored. "
            "Pick another sensor or date.")

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Sensor {data.sensor_ids[sensor]}")
    chart = pd.DataFrame(index=past_idx.append(fut_idx))
    chart["observed (input)"] = pd.Series(nan_missing(past[:, sensor]), index=past_idx)
    chart["actual (future)"] = pd.Series(nan_missing(future[:, sensor]), index=fut_idx)
    chart["forecast"] = pd.Series(pred[:, sensor], index=fut_idx)
    chart.loc[past_idx[-1], ["actual (future)", "forecast"]] = chart.loc[past_idx[-1], "observed (input)"]
    st.line_chart(chart, y_label="speed (mph)", color=["#555555", "#2ca02c", "#1f77b4"])

with right:
    st.subheader(f"Network, {horizon * 5} min ahead")
    fig, ax = plt.subplots(figsize=(5, 4.4))
    sc = ax.scatter(locs.longitude, locs.latitude, c=pred[horizon - 1], cmap="RdYlGn",
                    vmin=20, vmax=70, s=18, edgecolors="none")
    ax.scatter(locs.longitude.iloc[sensor], locs.latitude.iloc[sensor], s=140,
               facecolors="none", edgecolors="black", linewidths=1.5)
    fig.colorbar(sc, ax=ax, label="forecast speed (mph)")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- whole day + test metrics
with st.expander("Whole day at this sensor (15 and 60 min ahead)"):
    day_anchors = test[test_times.date == day]
    day_pred = trainer.predict(day_anchors)
    for h in (2, 11):
        idx = data.index[day_anchors + h + 1]
        df = pd.DataFrame({"actual": nan_missing(data.speed[day_anchors + h + 1, sensor]),
                           "forecast": day_pred[:, h, sensor]}, index=idx)
        st.caption(f"{(h + 1) * 5} min ahead")
        st.line_chart(df, y_label="speed (mph)", color=["#555555", "#1f77b4"])

with st.expander("Test-set accuracy vs. baselines (6,850 windows × 207 sensors)", expanded=True):
    st.dataframe(test_set_scores().style.format("{:.3f}").highlight_min(axis=0, color="#c8e6c9"))
    st.caption("Masked metrics: missing readings (0 mph) are ignored. Lower is better.")
