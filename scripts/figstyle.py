"""Shared matplotlib style for all figures (validated CVD-safe palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
GRAY = "#8a8984"

# Categorical slots (fixed order; first four validated all-pairs for scatter use)
BLUE, ORANGE, AQUA, VIOLET, YELLOW, MAGENTA, GREEN, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4", "#008300", "#e34948")

METHOD_STYLE = {
    "DP-PreCo-GRL": dict(color=BLUE, marker="o", ls="-"),
    "LS-GRL": dict(color=ORANGE, marker="s", ls="-"),
    "NSGA-II": dict(color=AQUA, marker="^", ls="-"),
    "NSGA-III": dict(color=VIOLET, marker="v", ls="-"),
    "PWG": dict(color=GRAY, marker="+", ls="-"),
    "Rules": dict(color=INK, marker="x", ls="-"),
}


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "pdf.fonttype": 42,
    })


COL_W = 3.5     # IEEE single column (in)
PAGE_W = 7.16   # IEEE double column (in)
