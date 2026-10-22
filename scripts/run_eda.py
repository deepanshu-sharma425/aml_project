"""Generate every EDA figure and a JSON summary of the dataset statistics."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from traffic import eda
from traffic import plotting as P
from traffic.data import distance_matrix, load_metrla


def main():
    P.setup()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = load_metrla()
    stats = {"summary": eda.summary(d)}
    steps = {
        "eda_missingness": lambda: eda.missingness(d),
        "eda_distribution": lambda: eda.distribution(d),
        "eda_daily_profile": lambda: eda.daily_profile(d),
        "eda_acf": lambda: eda.autocorrelation(d),
        "eda_spatial_corr": lambda: eda.spatial_correlation(d, distance_matrix(d.sensor_ids)),
        "eda_sensor_map": lambda: eda.sensor_map(d),
        "eda_split": lambda: eda.split_timeline(d),
        "eda_example_week": lambda: eda.example_week(d),
    }
    for name, fn in steps.items():
        fig, s = fn()
        fig.savefig(fig_dir / f"{name}.png")
        P.plt.close(fig)
        stats[name] = s
        print("saved", name, flush=True)
    stats["stationarity"] = eda.stationarity(d)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "eda_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
