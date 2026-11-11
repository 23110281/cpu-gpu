"""Shared chart style for the dashboard: sharp black background with a yellow accent.

Colour roles (validated with the dataviz skill's validator on a #000000 surface, dark mode,
adjacent pairs: lightness band, chroma, CVD >= 9.4, normal-vision >= 19.7, contrast >= 3:1):
  SERIES = amber-yellow, blue, magenta, violet, orange, aqua -- fixed order, never cycled.
  In CPU-vs-GPU charts the GPU is slot 1 (yellow) and the CPU slot 2 (blue).
  Limits are neutral white dashed lines, never a series colour.
  Heatmaps use YELLOW_RAMP: one hue, near-black -> yellow (lightness increases monotonically).
Every chart: one y-axis, units on the axes, a legend whenever there are >= 2 series.
"""
from __future__ import annotations

SURFACE = "#000000"
INK = "#f5f5f5"
INK_MUTED = "#a3a3a3"
GRID = "#262626"
LIMIT = "#f5f5f5"
ACCENT = "#FFD60A"            # UI accent (not a data colour)
SERIES = ["#c98500", "#3987e5", "#d55181", "#9085e9", "#d95926", "#199e70"]
RAMP = ["#5c4b00", "#7a6400", "#9a7e00", "#bb9900", "#ddb600", "#FFD60A"]   # >= 2:1 on black at the dark end

LIB_NAMES = {
    "torch": "PyTorch", "cudagraph": "PyTorch + CUDA graph", "native": "our own kernel",
    "graph": "CUDA graph launch", "coop": "cooperative launch", "custom": "our own CUDA kernel",
    "fast": "our own kernel, fast exp", "histc": "PyTorch histc", "bincount": "PyTorch bincount",
    "openblas": "OpenBLAS", "blis": "BLIS", "opencv": "OpenCV", "openssl": "OpenSSL",
    "gnu_parallel": "GNU parallel sort", "tbb": "Intel oneTBB",
}
HIDDEN_KEYS = ("precision", "variant", "lib", "_env")


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#3a3a3a", "axes.labelcolor": INK_MUTED, "axes.titlecolor": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.fontsize": 8,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6, "font.family": "DejaVu Sans",
    })


def ramp_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("yellow_ramp", RAMP)


def fmt_num(v) -> str:
    """Compact binary-ish size: 1048576 -> 1M, 16384 -> 16K, 65 -> 65."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    v = int(v) if float(v).is_integer() else v
    if isinstance(v, int):
        for unit, s in ((1 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")):
            if v >= unit and v % unit == 0:
                return f"{v // unit}{s}"
    return f"{v:g}"


