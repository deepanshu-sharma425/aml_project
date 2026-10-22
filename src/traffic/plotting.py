"""Shared matplotlib style so every figure in the report looks the same."""
from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:      # scripts render off-screen; notebooks inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical slots in fixed order (CVD-validated order; never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "lines.linewidth": 1.6, "legend.frameon": False,
    })
