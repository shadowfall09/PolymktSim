"""
Generate motivation figure for Introduction:
Real case: "Over $1.8B committed to the MegaETH public sale?" (Answer: NO)

Left: Standard Multi-Agent Debate (same input → herding → wrong)
Right: InfoDelphi (information asymmetry → diverse reasoning → correct)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5.5))

# ============================================
# Color scheme
# ============================================
C_PUBLIC = "#4393C3"
C_PRIVATE = ["#E6550D", "#31A354", "#756BB1"]
C_AGENT = "#525252"
C_HERDING = "#D6604D"
C_GOOD = "#1A9850"
C_BG_LEFT = "#FFF5F5"
C_BG_RIGHT = "#F0FFF0"

# ============================================
# LEFT PANEL: Standard Debate (All-Info)
# ============================================
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 11)
ax_left.set_aspect('equal')
ax_left.axis('off')
ax_left.set_facecolor(C_BG_LEFT)

# Title
ax_left.text(5, 10.5, "Standard Multi-Agent Debate", fontsize=11, fontweight='bold',
             ha='center', va='center', color=C_HERDING)

# Question
ax_left.text(5, 9.7, '"Over \\$1.8B committed to MegaETH public sale?"',
             fontsize=8, ha='center', va='center', color='#333333', fontstyle='italic')

# Evidence block (single, same for all)
evidence_box = FancyBboxPatch((1.5, 8.0), 7, 1.0, boxstyle="round,pad=0.1",
                               facecolor="#FEE08B", edgecolor="#999999", linewidth=1.5)
ax_left.add_patch(evidence_box)
ax_left.text(5, 8.5, "Same Evidence $\\mathcal{E}$ (all docs)", fontsize=9, ha='center', va='center')

# Arrows from evidence to agents
for x in [2.5, 5.0, 7.5]:
    ax_left.annotate("", xy=(x, 6.8), xytext=(x, 8.0),
                     arrowprops=dict(arrowstyle="->, head_width=0.2", color="#666666", lw=1.2))

# Agent boxes
for i, x in enumerate([2.5, 5.0, 7.5]):
    agent_box = FancyBboxPatch((x-0.9, 5.8), 1.8, 1.0, boxstyle="round,pad=0.08",
                                facecolor="#D9D9D9", edgecolor=C_AGENT, linewidth=1.2)
    ax_left.add_patch(agent_box)
    ax_left.text(x, 6.3, f"Agent {i+1}", fontsize=9, ha='center', va='center')

# Round 0 predictions
ax_left.text(5, 5.3, "Round 1 (independent):", fontsize=8, ha='center', color='#555555')
r0_preds = [0.99, 0.94, 0.97]
for i, (x, p) in enumerate(zip([2.5, 5.0, 7.5], r0_preds)):
    ax_left.text(x, 4.7, f"p={p:.2f}", fontsize=9, ha='center', va='center',
                 color=C_HERDING, fontfamily='monospace')

# Deliberation arrows (dashed = echo)
ax_left.annotate("", xy=(3.5, 6.3), xytext=(4.1, 6.3),
                 arrowprops=dict(arrowstyle="<->", color=C_HERDING, lw=1, linestyle='dashed'))
ax_left.annotate("", xy=(5.9, 6.3), xytext=(6.5, 6.3),
                 arrowprops=dict(arrowstyle="<->", color=C_HERDING, lw=1, linestyle='dashed'))

# Round 1 predictions
ax_left.text(5, 4.0, "Round 2 (after deliberation):", fontsize=8, ha='center', color='#555555')
r1_preds = [0.24, 0.92, 0.86]
for i, (x, p) in enumerate(zip([2.5, 5.0, 7.5], r1_preds)):
    ax_left.text(x, 3.4, f"p={p:.2f}", fontsize=9, ha='center', va='center',
                 color=C_HERDING, fontfamily='monospace')

# Final output
ax_left.annotate("", xy=(5, 2.3), xytext=(5, 3.0),
                 arrowprops=dict(arrowstyle="->, head_width=0.3", color="#666666", lw=1.5))
output_box = FancyBboxPatch((3.0, 1.3), 4.0, 0.9, boxstyle="round,pad=0.1",
                             facecolor="#FFCCCC", edgecolor=C_HERDING, linewidth=1.5)
ax_left.add_patch(output_box)
ax_left.text(5, 1.75, "$\\hat{p}$ = 0.67  (Brier = 0.45)", fontsize=9, ha='center', va='center',
             fontweight='bold', color=C_HERDING)

# Bottom label
ax_left.text(5, 0.5, "Herding: all agents misled by same evidence", fontsize=8,
             ha='center', va='center', color="#666666", fontstyle='italic')

# ============================================
# RIGHT PANEL: InfoDelphi (BM25 routing)
# ============================================
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 11)
ax_right.set_aspect('equal')
ax_right.axis('off')
ax_right.set_facecolor(C_BG_RIGHT)

# Title
ax_right.text(5, 10.5, "InfoDelphi (Ours)", fontsize=11, fontweight='bold',
              ha='center', va='center', color=C_GOOD)

# Question
ax_right.text(5, 9.7, 'Ground Truth: NO',
             fontsize=8, ha='center', va='center', color='#333333', fontweight='bold')

# Evidence blocks: public + private
pub_box = FancyBboxPatch((2.0, 8.2), 6, 0.7, boxstyle="round,pad=0.08",
                          facecolor=C_PUBLIC, edgecolor="#2166AC", linewidth=1.2, alpha=0.3)
ax_right.add_patch(pub_box)
ax_right.text(5, 8.55, "Public $\\mathcal{E}^{pub}$ (shared context)", fontsize=8,
              ha='center', va='center', color="#2166AC")

# Private subsets
for i, (x, c, label) in enumerate(zip([2.0, 5.0, 8.0], C_PRIVATE,
                                        ["$\\mathcal{E}_1^{priv}$", "$\\mathcal{E}_2^{priv}$", "$\\mathcal{E}_3^{priv}$"])):
    priv_box = FancyBboxPatch((x-0.7, 7.2), 1.4, 0.7, boxstyle="round,pad=0.05",
                               facecolor=c, edgecolor=c, linewidth=1, alpha=0.25)
    ax_right.add_patch(priv_box)
    ax_right.text(x, 7.55, label, fontsize=8, ha='center', va='center', color=c)

# Arrows to agents
for i, (x, c) in enumerate(zip([2.0, 5.0, 8.0], C_PRIVATE)):
    ax_right.annotate("", xy=(x, 6.4), xytext=(x, 7.2),
                      arrowprops=dict(arrowstyle="->, head_width=0.15", color=c, lw=1.2))

# Agent boxes (colored)
for i, (x, c) in enumerate(zip([2.0, 5.0, 8.0], C_PRIVATE)):
    agent_box = FancyBboxPatch((x-0.9, 5.4), 1.8, 1.0, boxstyle="round,pad=0.08",
                                facecolor=c, edgecolor=c, linewidth=1.2, alpha=0.2)
    ax_right.add_patch(agent_box)
    ax_right.text(x, 5.9, f"Agent {i+1}", fontsize=9, ha='center', va='center', color=c)

# Round 0 predictions
ax_right.text(5, 4.9, "Round 1 (independent):", fontsize=8, ha='center', color='#555555')
r0_preds_bm25 = [0.95, 0.01, 0.02]
for i, (x, c, p) in enumerate(zip([2.0, 5.0, 8.0], C_PRIVATE, r0_preds_bm25)):
    ax_right.text(x, 4.3, f"p={p:.2f}", fontsize=9, ha='center', va='center',
                  color=c, fontfamily='monospace', fontweight='bold' if p < 0.1 else 'normal')

# Deliberation arrows (solid = meaningful)
ax_right.annotate("", xy=(3.1, 5.9), xytext=(3.9, 5.9),
                  arrowprops=dict(arrowstyle="<->", color=C_GOOD, lw=1.5))
ax_right.annotate("", xy=(6.1, 5.9), xytext=(6.9, 5.9),
                  arrowprops=dict(arrowstyle="<->", color=C_GOOD, lw=1.5))

# Annotation: key insight
ax_right.text(5, 3.7, "Agents 2&3 share evidence → Agent 1 corrects",
              fontsize=7.5, ha='center', color=C_GOOD, fontstyle='italic')

# Round 1 predictions
ax_right.text(5, 3.2, "Round 2 (after rationale sharing):", fontsize=8, ha='center', color='#555555')
r1_preds_bm25 = [0.08, 0.08, 0.03]
for i, (x, c, p) in enumerate(zip([2.0, 5.0, 8.0], C_PRIVATE, r1_preds_bm25)):
    ax_right.text(x, 2.6, f"p={p:.2f}", fontsize=9, ha='center', va='center',
                  color=c, fontfamily='monospace')

# Final output
ax_right.annotate("", xy=(5, 1.7), xytext=(5, 2.3),
                  arrowprops=dict(arrowstyle="->, head_width=0.3", color="#666666", lw=1.5))
output_box = FancyBboxPatch((3.0, 0.7), 4.0, 0.9, boxstyle="round,pad=0.1",
                             facecolor="#CCFFCC", edgecolor=C_GOOD, linewidth=1.5)
ax_right.add_patch(output_box)
ax_right.text(5, 1.15, "$\\hat{p}$ = 0.06  (Brier = 0.004)", fontsize=9, ha='center', va='center',
              fontweight='bold', color=C_GOOD)

# Bottom label
ax_right.text(5, 0.1, "Private evidence enables self-correction", fontsize=8,
              ha='center', va='center', color="#666666", fontstyle='italic')

# ============================================
# Final layout
# ============================================
plt.tight_layout(w_pad=1.5)
plt.savefig("paper/latex/fig_motivation.pdf", bbox_inches='tight', dpi=300)
plt.savefig("paper/latex/fig_motivation.png", bbox_inches='tight', dpi=200)
plt.close()
print("Saved: paper/latex/fig_motivation.pdf and .png")
