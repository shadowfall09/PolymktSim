"""
Generate paper figures from 375-question experiment results.

Fig 1: Prediction extremity vs accuracy (quartile analysis)
Fig 2: Agent variance before/after deliberation (3 routing strategies)

Usage:
    python paper/generate_figures.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ---- Config ----
RESULTS_DIR = Path("data/results")
OUTPUT_DIR = Path("paper/latex")

# Main run files
BM25_DETAIL = RESULTS_DIR / "20260413_215746_detail.jsonl"      # CW + BM25
RANDOM_DETAIL = RESULTS_DIR / "20260413_215747_detail.jsonl"    # CW + Random
ALLINFO_DETAIL = RESULTS_DIR / "20260414_183430_detail.jsonl"   # All Info (public_ratio=1.0)

# Temporal filter: only keep questions with endDate >= 2025-09-01
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")


def load_valid_qids():
    """Load qids that pass the temporal filter (375 questions).

    qid format in detail files: row_{NNNN}_{market_id}
    CSV rows are 1-indexed, market_id is the 'id' column.
    """
    import csv
    valid_qids = set()
    with open(MARKETS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            end_date = row.get("endDateIso", row.get("endDate", ""))
            if end_date >= "2025-09-01":
                market_id = row["id"]
                qid = f"row_{i:04d}_{market_id}"
                valid_qids.add(qid)
    return valid_qids


def load_detail_jsonl(path, valid_qids=None):
    """Load detail JSONL file, optionally filtering by valid qids."""
    records = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line.strip())
            if valid_qids and rec["qid"] not in valid_qids:
                continue
            records.append(rec)
    return records


def try_load_valid_qids():
    """Try loading from CSV, fall back to using all qids in the detail file."""
    try:
        return load_valid_qids()
    except Exception:
        # Fall back: load all qids from main detail file and trust they are filtered
        records = load_detail_jsonl(BM25_DETAIL)
        return set(r["qid"] for r in records)


# =========================================================
# Figure 1: Extremity vs Accuracy
# =========================================================
def generate_fig1(valid_qids):
    """Stratify individual agent predictions by extremity quartiles."""
    records = load_detail_jsonl(BM25_DETAIL, valid_qids)

    # Use S2 (scenario=s2) final round predictions for the main analysis
    # Filter to s2 round_id=1 (the final round after deliberation)
    s2_final = [r for r in records if r["scenario"] == "s2" and r["round_id"] == 1]

    # If no s2 data with round_id=1, try s1 round_id=0
    if not s2_final:
        s2_final = [r for r in records if r["scenario"] == "s1" and r["round_id"] == 0]

    if not s2_final:
        # Use all available predictions
        s2_final = records

    print(f"Fig 1: {len(s2_final)} individual agent predictions")

    # Compute extremity and accuracy for each prediction
    extremities = []
    correct = []
    briers = []
    for r in s2_final:
        p = r["p_yes"]
        outcome = 1.0 if r["outcome"] else 0.0
        ext = abs(p - 0.5)
        is_correct = (p >= 0.5) == r["outcome"]
        brier = (p - outcome) ** 2

        extremities.append(ext)
        correct.append(is_correct)
        briers.append(brier)

    extremities = np.array(extremities)
    correct = np.array(correct)
    briers = np.array(briers)

    # Split into quartiles
    quartile_edges = np.percentile(extremities, [25, 50, 75])
    q_labels = ["Q1\n(least extreme)", "Q2", "Q3", "Q4\n(most extreme)"]

    q_indices = np.digitize(extremities, quartile_edges)  # 0,1,2,3

    accs = []
    brier_means = []
    counts = []
    for qi in range(4):
        mask = q_indices == qi
        accs.append(correct[mask].mean() * 100)
        brier_means.append(briers[mask].mean())
        counts.append(mask.sum())

    print(f"  Quartile counts: {counts}")
    print(f"  Accuracies: {[f'{a:.1f}%' for a in accs]}")
    print(f"  Brier scores: {[f'{b:.3f}' for b in brier_means]}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    x = np.arange(4)
    width = 0.35

    color_acc = "#c6e1ee"
    color_brier = "#f1d2d0"

    bars1 = ax1.bar(x - width/2, accs, width, label="Accuracy (%)",
                    color=color_acc, edgecolor="#4393C3", linewidth=1.5, hatch="..")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(30, 100)
    ax1.tick_params(axis="y", labelsize=9)
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, brier_means, width, label="Brier Score",
                    color=color_brier, edgecolor="#D6604D", linewidth=1.5, hatch="//")
    ax2.set_ylabel("Brier Score", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 0.35)
    ax2.tick_params(axis="y", labelsize=9)
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Prediction Extremity Quartile ($|p - 0.5|$)", fontsize=11, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#4393C3", fontweight="bold")
    for bar, val in zip(bars2, brier_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#D6604D", fontweight="bold")

    fig.legend(loc="upper left", ncol=2, framealpha=0.9,
               bbox_to_anchor=(0.138, 0.95), prop={"weight": "bold", "size": 8.5})
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_extremity_accuracy.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("  Saved: fig1_extremity_accuracy.pdf")


# =========================================================
# Figure 2: Agent Variance Analysis
# =========================================================
def generate_fig2(valid_qids):
    """Compare inter-agent variance before/after deliberation across routing strategies."""

    strategies = {
        "BM25 Split": BM25_DETAIL,
        "Random Split": RANDOM_DETAIL,
        "All Info": ALLINFO_DETAIL,
    }

    results = {}
    for name, path in strategies.items():
        if not path.exists():
            print(f"  Warning: {path} not found, skipping {name}")
            continue

        records = load_detail_jsonl(path, valid_qids)
        # Filter to s2 scenario (deliberation)
        s2_recs = [r for r in records if r["scenario"] == "s2"]
        if not s2_recs:
            s2_recs = [r for r in records if r["scenario"] in ("s1", "s2")]

        # Group by (qid, round_id) -> list of p_yes
        grouped = defaultdict(list)
        for r in s2_recs:
            grouped[(r["qid"], r["round_id"])].append(r["p_yes"])

        # Compute variance per (qid, round_id)
        var_by_round = defaultdict(list)
        for (qid, round_id), preds in grouped.items():
            if len(preds) >= 2:
                var_by_round[round_id].append(np.var(preds))

        # Get round 0 (before) and round 1 (after)
        rounds_available = sorted(var_by_round.keys())
        if len(rounds_available) >= 2:
            r0, r1 = rounds_available[0], rounds_available[1]
        else:
            r0 = rounds_available[0]
            r1 = r0

        mean_var_before = np.mean(var_by_round[r0])
        mean_var_after = np.mean(var_by_round[r1])
        reduction = (mean_var_before - mean_var_after) / mean_var_before * 100 if mean_var_before > 0 else 0

        results[name] = {
            "before": mean_var_before,
            "after": mean_var_after,
            "reduction": reduction,
            "n_questions": len(var_by_round[r0]),
        }
        print(f"  {name}: before={mean_var_before:.4f}, after={mean_var_after:.4f}, "
              f"reduction={reduction:.1f}%, n={results[name]['n_questions']}")

    if not results:
        print("  ERROR: No data available for Fig 2")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))

    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35

    before_vals = [results[n]["before"] for n in names]
    after_vals = [results[n]["after"] for n in names]
    reductions = [results[n]["reduction"] for n in names]

    color_before = "#c6e1ee"
    color_after = "#f1d2d0"

    bars1 = ax.bar(x - width/2, before_vals, width, label="Round 1 (before)",
                   color=color_before, edgecolor="#4393C3", linewidth=1.5, hatch="..")
    bars2 = ax.bar(x + width/2, after_vals, width, label="Round 2 (after)",
                   color=color_after, edgecolor="#D6604D", linewidth=1.5, hatch="//")

    ax.set_ylabel("Mean Inter-Agent Variance", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Add reduction annotations
    for i, (b1, b2, red) in enumerate(zip(bars1, bars2, reductions)):
        y_max = max(b1.get_height(), b2.get_height())
        ax.annotate(f"$-${red:.0f}%",
                    xy=(x[i], y_max + 0.001),
                    ha="center", fontsize=9, color="#333333", fontweight="bold")

    ax.legend(loc="upper right", framealpha=0.9,
              prop={"weight": "bold", "size": 8.5})
    ax.set_ylim(0, max(before_vals) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_agent_variance.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("  Saved: fig2_agent_variance.pdf")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Loading valid qids (375-question filtered set)...")
    valid_qids = try_load_valid_qids()
    print(f"  {len(valid_qids)} valid questions")

    print("\n--- Generating Figure 1: Extremity vs Accuracy ---")
    generate_fig1(valid_qids)

    print("\n--- Generating Figure 2: Agent Variance ---")
    generate_fig2(valid_qids)

    print("\nDone! Figures saved to paper/latex/")
