#!/usr/bin/env python3
"""Fig 2: deliberation under homogeneous input collapses into herding;
information asymmetry preserves independent signal.

Two panels: (a) mean per-question std of agent forecasts (disagreement),
(b) mean pairwise inter-agent error correlation; both round 1 -> round 2.
Run from repo root.
"""
import json
import sys

sys.path.insert(0, "paper_v3/figures")
from paper_plot_style import plt, save_fig, C_OURS, C_BASE

R = json.load(open("paper_v3/figures/paper_numbers.json"))
A = R["agent_analysis"]

DATASETS = [("polymarket_250", "PolyGym-250"), ("futurex", "FutureX-231")]
METHODS = [("infodelphi", "InfoDelphi (partitioned)", C_OURS, "o"),
           ("standard_debate", "Homogeneous input", C_BASE, "s")]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))

for ax, key, ylabel in [
    (axes[0], "std", "Forecast disagreement (std)"),
    (axes[1], "corr", "Inter-agent error correlation"),
]:
    for di, (ds, dlabel) in enumerate(DATASETS):
        for m, mlabel, color, marker in METHODS:
            vals = A[f"{ds}_{m}_{key}"]
            y = [vals["0"], vals["1"]]
            ls = "-" if di == 0 else "--"
            label = f"{mlabel}" if di == 0 else None
            ax.plot([1, 2], y, ls, color=color, marker=marker, markersize=4.5,
                    label=label, linewidth=1.4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Round 1", "Round 2"])
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.8, 2.2)

axes[0].legend(frameon=False, loc="upper right", fontsize=8)
# dataset linestyle key on second panel
from matplotlib.lines import Line2D
axes[1].legend(handles=[
    Line2D([0], [0], color="k", ls="-", label="PolyGym-250"),
    Line2D([0], [0], color="k", ls="--", label="FutureX-231"),
], frameon=False, loc="lower right", fontsize=8)

fig.tight_layout()
save_fig(fig, "fig2_herding")

for ds, _ in DATASETS:
    for m, *_ in METHODS:
        s = A[f"{ds}_{m}_std"]; c = A[f"{ds}_{m}_corr"]
        print(f"{ds} {m}: std {s['0']:.4f}->{s['1']:.4f}  corr {c['0']:.3f}->{c['1']:.3f}")
