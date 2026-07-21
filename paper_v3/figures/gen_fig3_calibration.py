#!/usr/bin/env python3
"""Fig 3: calibration study — prompt-level skepticism (v2/v3) vs
aggregation-layer shrinkage. Two panels:
(a) Brier of prompt v1/v2/v3, raw vs +shrink, with constant base-rate line;
(b) mean forecast vs empirical base rate (distribution depression).
Run from repo root.
"""
import json
import sys

sys.path.insert(0, "paper_v3/figures")
from paper_plot_style import plt, save_fig, C_OURS, C_BASE, C_NEUTRAL, C_ACCENT

R = json.load(open("paper_v3/figures/paper_numbers.json"))
cal = R["calibration_study"]
p = R["shrink_params"]
P0, W_LO, W_HI = p["p0"], p["w_lo"], p["w_hi"]


def unshrink(q):
    w = W_HI if q > P0 else W_LO
    return P0 + (q - P0) / w


def shrink(q):
    w = W_HI if q > P0 else W_LO
    return P0 + w * (q - P0)


# reconstruct raw and shrunk Brier for each variant from the stored files
import pathlib
RV2 = pathlib.Path("data/results_v2")
FILES = {
    "v1": ("polymarket_250_s1s2_arguments_pooling_calibrated_5.4mini.jsonl", False),
    "v2": ("polymarket_250_s1s2_argpool_calv2_da_shrink_5.4mini.jsonl", True),
    "v3": ("polymarket_250_s1s2_argpool_calv3_shrink_5.4mini.jsonl", True),
}
stats = {}
for name, (fname, stored_shrunk) in FILES.items():
    raws, shr, ys = [], [], []
    for line in open(RV2 / fname):
        r = json.loads(line)
        if r.get("scenario") != "s2":
            continue
        q = float(r["p_yes"])
        p_raw = unshrink(q) if stored_shrunk else q
        raws.append(p_raw)
        shr.append(shrink(p_raw))
        ys.append(1.0 if r["outcome"] else 0.0)
    n = len(ys)
    stats[name] = {
        "raw_brier": sum((a - y) ** 2 for a, y in zip(raws, ys)) / n,
        "shrunk_brier": sum((a - y) ** 2 for a, y in zip(shr, ys)) / n,
        "raw_mean_p": sum(raws) / n,
    }
base_rate = cal["base_rate"]
const_brier = cal["const_baserate_brier"]

# sanity vs MANIFEST
assert abs(stats["v1"]["raw_brier"] - 0.1647) < 5e-4
assert abs(stats["v2"]["raw_brier"] - 0.1896) < 2e-3, stats["v2"]["raw_brier"]
assert abs(stats["v3"]["raw_brier"] - 0.1780) < 2e-3, stats["v3"]["raw_brier"]

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))

labels = ["v1\n(calibrated)", "v2 (+hard cap,\ndevil's advocate)", "v3 (soft\ncriteria check)"]
x = range(3)
w = 0.35
ax = axes[0]
ax.bar([i - w / 2 for i in x], [stats[v]["raw_brier"] for v in ("v1", "v2", "v3")],
       width=w, color=C_NEUTRAL, label="raw (mean agg.)")
ax.bar([i + w / 2 for i in x], [stats[v]["shrunk_brier"] for v in ("v1", "v2", "v3")],
       width=w, color=C_OURS, label="+ calibrated shrink")
ax.axhline(const_brier, color=C_BASE, ls=":", lw=1.2)
ax.text(2.4, const_brier + 0.001, "constant\nbase rate", color=C_BASE, fontsize=7.5,
        ha="right", va="bottom")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Brier score")
ax.set_ylim(0.14, 0.20)
ax.legend(frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
means = [stats[v]["raw_mean_p"] for v in ("v1", "v2", "v3")]
ax.bar(list(x), means, width=0.5, color=[C_OURS, C_ACCENT, C_ACCENT])
ax.axhline(base_rate, color=C_BASE, ls=":", lw=1.2)
ax.text(2.4, base_rate + 0.004, f"base rate {base_rate:.3f}", color=C_BASE,
        fontsize=7.5, ha="right", va="bottom")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Mean forecast $\\bar{p}$")
ax.set_ylim(0.0, 0.35)

fig.tight_layout()
save_fig(fig, "fig3_calibration")
for v in ("v1", "v2", "v3"):
    print(v, {k: round(val, 4) for k, val in stats[v].items()})
