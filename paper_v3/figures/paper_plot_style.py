"""Shared matplotlib style for all paper figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300
FIG_DIR = "paper_v3/figures"

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

# colorblind-safe (Okabe-Ito subset)
C_OURS = "#0072B2"      # blue
C_BASE = "#D55E00"      # vermillion
C_NEUTRAL = "#7f7f7f"   # gray
C_ACCENT = "#009E73"    # green


def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.pdf"
    fig.savefig(path)
    print(f"Saved: {path}")
