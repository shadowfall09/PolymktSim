"""
Generate analysis figures for the paper:
1. Topic-wise breakdown (grouped bar chart)
2. Calibration curve (reliability diagram)
3. Belief revision visualization (R1 → R2 shift)
4. Difficulty-stratified analysis
5. Cost-performance trade-off

All figures use the 375-question filtered dataset.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path
import csv
from datetime import date

# ---- Config ----
RESULTS_DIR = Path("data/results")
OUTPUT_DIR = Path("paper/latex")
MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
BM25_DETAIL = RESULTS_DIR / "20260413_215746_detail.jsonl"
ALLINFO_DETAIL = RESULTS_DIR / "20260414_183430_detail.jsonl"
S0_FILE = RESULTS_DIR / "s0_fair_temp07.jsonl"

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

COLORS = {
    'infodelphi': '#1A9850',
    'single_agent': '#4393C3',
    'standard_debate': '#F46D43',
    'independent': '#74ADD1',
    'highlight': '#D73027',
    'neutral': '#878787',
}


def load_valid_qids():
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


def load_detail_dedup(path, valid_qids, scenario='s2', round_id=None):
    """Load detail file, deduplicated by (qid, agent_id, round_id)."""
    grouped = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['qid'] not in valid_qids:
                continue
            if r['scenario'] != scenario:
                continue
            if round_id is not None and r['round_id'] != round_id:
                continue
            key = (r['qid'], r['agent_id'], r['round_id'])
            grouped[key] = r
    return list(grouped.values())


def load_aggregated(path, valid_qids):
    """Load aggregated JSONL results."""
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['qid'] in valid_qids:
                records.append(r)
    return records


# =========================================================
# Figure: Topic-wise Breakdown
# =========================================================
def generate_topic_breakdown(valid_qids):
    """Grouped bar chart: Brier by topic for Single-Agent vs InfoDelphi vs Standard Debate."""
    print("--- Generating Topic Breakdown ---")

    # Load S0 (single agent)
    s0_by_topic = defaultdict(list)
    for rec in load_aggregated(S0_FILE, valid_qids):
        outcome = 1.0 if rec['outcome'] else 0.0
        brier = (rec['p_yes'] - outcome) ** 2
        s0_by_topic[rec.get('topic', 'Other')].append(brier)

    # Load InfoDelphi S2 final (BM25, round 1)
    bm25_s2 = load_detail_dedup(BM25_DETAIL, valid_qids, 's2', round_id=1)
    infodelphi_by_qid = defaultdict(list)
    topic_map = {}
    for r in bm25_s2:
        infodelphi_by_qid[r['qid']].append(r['p_yes'])
        topic_map[r['qid']] = r.get('topic', 'Other')

    infodelphi_by_topic = defaultdict(list)
    for qid, preds in infodelphi_by_qid.items():
        if len(preds) >= 3:
            agg = np.mean(preds[:3])
            outcome = 1.0 if bm25_s2[0]['outcome'] else 0.0
            # need to find outcome for this qid
            for r in bm25_s2:
                if r['qid'] == qid:
                    outcome = 1.0 if r['outcome'] else 0.0
                    break
            brier = (agg - outcome) ** 2
            infodelphi_by_topic[topic_map.get(qid, 'Other')].append(brier)

    # Load Standard Debate (All-Info, S2 round 1)
    allinfo_s2 = load_detail_dedup(ALLINFO_DETAIL, valid_qids, 's2', round_id=1)
    debate_by_qid = defaultdict(list)
    for r in allinfo_s2:
        debate_by_qid[r['qid']].append(r)

    debate_by_topic = defaultdict(list)
    for qid, agents in debate_by_qid.items():
        if len(agents) >= 3:
            preds = [a['p_yes'] for a in agents[:3]]
            outcome = 1.0 if agents[0]['outcome'] else 0.0
            agg = np.mean(preds)
            brier = (agg - outcome) ** 2
            topic = agents[0].get('topic', 'Other')
            debate_by_topic[topic].append(brier)

    # Get common topics (sorted by frequency)
    all_topics = sorted(s0_by_topic.keys(), key=lambda t: -len(s0_by_topic[t]))
    # Only show topics with >= 10 questions
    topics = [t for t in all_topics if len(s0_by_topic[t]) >= 10]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(topics))
    width = 0.25

    s0_means = [np.mean(s0_by_topic[t]) for t in topics]
    debate_means = [np.mean(debate_by_topic[t]) if debate_by_topic[t] else 0 for t in topics]
    info_means = [np.mean(infodelphi_by_topic[t]) if infodelphi_by_topic[t] else 0 for t in topics]

    ax.bar(x - width, s0_means, width, label='Single-Agent', color=COLORS['single_agent'], alpha=0.85)
    ax.bar(x, debate_means, width, label='Standard Debate', color=COLORS['standard_debate'], alpha=0.85)
    ax.bar(x + width, info_means, width, label='InfoDelphi', color=COLORS['infodelphi'], alpha=0.85)

    ax.set_ylabel('Brier Score (lower is better)', fontsize=10)
    ax.set_xticks(x)
    # Add count to labels
    ax.set_xticklabels([f"{t}\n(n={len(s0_by_topic[t])})" for t in topics], fontsize=8)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, max(s0_means) * 1.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_topic_breakdown.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "fig_topic_breakdown.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved: fig_topic_breakdown.pdf")

    # Print numbers
    for t in topics:
        print(f"  {t}: S0={np.mean(s0_by_topic[t]):.3f}, Debate={np.mean(debate_by_topic[t]):.3f}, InfoDelphi={np.mean(infodelphi_by_topic[t]):.3f}")


# =========================================================
# Figure: Calibration Curve
# =========================================================
def generate_calibration_curve(valid_qids):
    """Reliability diagram comparing Single-Agent vs InfoDelphi."""
    print("--- Generating Calibration Curve ---")

    # Load S0
    s0_preds, s0_outcomes = [], []
    for rec in load_aggregated(S0_FILE, valid_qids):
        s0_preds.append(rec['p_yes'])
        s0_outcomes.append(1.0 if rec['outcome'] else 0.0)

    # Load InfoDelphi S2
    bm25_s2 = load_detail_dedup(BM25_DETAIL, valid_qids, 's2', round_id=1)
    infodelphi_by_qid = defaultdict(lambda: {'preds': [], 'outcome': None})
    for r in bm25_s2:
        infodelphi_by_qid[r['qid']]['preds'].append(r['p_yes'])
        infodelphi_by_qid[r['qid']]['outcome'] = 1.0 if r['outcome'] else 0.0

    info_preds, info_outcomes = [], []
    for qid, data in infodelphi_by_qid.items():
        if len(data['preds']) >= 3:
            info_preds.append(np.mean(data['preds'][:3]))
            info_outcomes.append(data['outcome'])

    def compute_calibration(preds, outcomes, n_bins=10):
        preds, outcomes = np.array(preds), np.array(outcomes)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []
        bin_counts = []
        for i in range(n_bins):
            mask = (preds >= bin_edges[i]) & (preds < bin_edges[i+1])
            if mask.sum() > 0:
                bin_centers.append(preds[mask].mean())
                bin_freqs.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())
        return bin_centers, bin_freqs, bin_counts

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration', linewidth=1)

    # S0
    centers_s0, freqs_s0, counts_s0 = compute_calibration(s0_preds, s0_outcomes)
    ax.plot(centers_s0, freqs_s0, 'o-', color=COLORS['single_agent'], label='Single-Agent',
            markersize=6, linewidth=1.5, alpha=0.85)

    # InfoDelphi
    centers_info, freqs_info, counts_info = compute_calibration(info_preds, info_outcomes)
    ax.plot(centers_info, freqs_info, 's-', color=COLORS['infodelphi'], label='InfoDelphi',
            markersize=6, linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Observed Frequency', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_calibration.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "fig_calibration.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved: fig_calibration.pdf")


# =========================================================
# Figure: Belief Revision (R1 → R2)
# =========================================================
def generate_belief_revision(valid_qids):
    """Visualize how predictions shift from Round 1 to Round 2."""
    print("--- Generating Belief Revision ---")

    # Load BM25 S2 both rounds
    r0_data = load_detail_dedup(BM25_DETAIL, valid_qids, 's2', round_id=0)
    r1_data = load_detail_dedup(BM25_DETAIL, valid_qids, 's2', round_id=1)

    # Match by (qid, agent_id)
    r0_map = {(r['qid'], r['agent_id']): r for r in r0_data}
    r1_map = {(r['qid'], r['agent_id']): r for r in r1_data}

    # Compute shifts
    shifts_toward_correct = []
    shifts_away = []
    all_shifts = []

    for key in r0_map:
        if key not in r1_map:
            continue
        p0 = r0_map[key]['p_yes']
        p1 = r1_map[key]['p_yes']
        outcome = 1.0 if r0_map[key]['outcome'] else 0.0

        shift = p1 - p0
        # Distance to truth
        dist_before = abs(p0 - outcome)
        dist_after = abs(p1 - outcome)

        if dist_after < dist_before:
            shifts_toward_correct.append(shift)
        else:
            shifts_away.append(shift)
        all_shifts.append((shift, dist_after < dist_before))

    toward_pct = len(shifts_toward_correct) / len(all_shifts) * 100
    print(f"  Shifts toward correct: {len(shifts_toward_correct)}/{len(all_shifts)} ({toward_pct:.1f}%)")
    print(f"  Mean shift magnitude: {np.mean(np.abs([s for s, _ in all_shifts])):.3f}")

    # Plot: histogram of shifts colored by direction
    fig, ax = plt.subplots(figsize=(6, 3.5))

    toward_shifts = [s for s, correct in all_shifts if correct]
    away_shifts = [s for s, correct in all_shifts if not correct]

    bins = np.linspace(-0.6, 0.6, 40)
    ax.hist(toward_shifts, bins=bins, alpha=0.75, color=COLORS['infodelphi'],
            label=f'Toward correct ({toward_pct:.0f}%)', edgecolor='white', linewidth=0.5)
    ax.hist(away_shifts, bins=bins, alpha=0.75, color=COLORS['highlight'],
            label=f'Away from correct ({100-toward_pct:.0f}%)', edgecolor='white', linewidth=0.5)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel('Prediction Shift (Round 2 $-$ Round 1)', fontsize=10)
    ax.set_ylabel('Number of Agent Predictions', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_belief_revision.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "fig_belief_revision.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved: fig_belief_revision.pdf")


# =========================================================
# Figure: Difficulty-Stratified Analysis
# =========================================================
def generate_difficulty_analysis(valid_qids):
    """Stratify by question difficulty, show improvement pattern."""
    print("--- Generating Difficulty Analysis ---")

    # Load S0 as difficulty proxy
    s0_by_qid = {}
    for rec in load_aggregated(S0_FILE, valid_qids):
        outcome = 1.0 if rec['outcome'] else 0.0
        s0_by_qid[rec['qid']] = {
            'brier': (rec['p_yes'] - outcome) ** 2,
            'outcome': outcome,
        }

    # Load InfoDelphi
    bm25_s2 = load_detail_dedup(BM25_DETAIL, valid_qids, 's2', round_id=1)
    info_by_qid = defaultdict(list)
    outcome_by_qid = {}
    for r in bm25_s2:
        info_by_qid[r['qid']].append(r['p_yes'])
        outcome_by_qid[r['qid']] = 1.0 if r['outcome'] else 0.0

    # Compute InfoDelphi Brier per question
    info_brier_by_qid = {}
    for qid, preds in info_by_qid.items():
        if len(preds) >= 3 and qid in outcome_by_qid:
            agg = np.mean(preds[:3])
            info_brier_by_qid[qid] = (agg - outcome_by_qid[qid]) ** 2

    # Common qids
    common_qids = sorted(set(s0_by_qid.keys()) & set(info_brier_by_qid.keys()))

    # Sort by S0 difficulty (Brier)
    s0_briers = [s0_by_qid[q]['brier'] for q in common_qids]
    info_briers = [info_brier_by_qid[q] for q in common_qids]

    # Split into terciles by difficulty
    tercile_edges = np.percentile(s0_briers, [33.3, 66.7])
    labels = ['Easy\n(S0 Brier < {:.2f})'.format(tercile_edges[0]),
              'Medium\n({:.2f}–{:.2f})'.format(tercile_edges[0], tercile_edges[1]),
              'Hard\n(S0 Brier > {:.2f})'.format(tercile_edges[1])]

    indices = np.digitize(s0_briers, tercile_edges)  # 0, 1, 2

    s0_tercile_means = []
    info_tercile_means = []
    counts = []
    for t in range(3):
        mask = np.array(indices) == t
        s0_tercile_means.append(np.mean(np.array(s0_briers)[mask]))
        info_tercile_means.append(np.mean(np.array(info_briers)[mask]))
        counts.append(mask.sum())

    print(f"  Tercile counts: {counts}")
    for i, label in enumerate(labels):
        improvement = (s0_tercile_means[i] - info_tercile_means[i]) / s0_tercile_means[i] * 100
        print(f"  {label.split(chr(10))[0]}: S0={s0_tercile_means[i]:.3f}, InfoDelphi={info_tercile_means[i]:.3f}, improvement={improvement:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(x - width/2, s0_tercile_means, width, label='Single-Agent',
                   color=COLORS['single_agent'], alpha=0.85)
    bars2 = ax.bar(x + width/2, info_tercile_means, width, label='InfoDelphi',
                   color=COLORS['infodelphi'], alpha=0.85)

    # Add improvement annotations
    for i in range(3):
        improvement = (s0_tercile_means[i] - info_tercile_means[i]) / s0_tercile_means[i] * 100
        y_pos = max(s0_tercile_means[i], info_tercile_means[i]) + 0.01
        ax.annotate(f'$-${improvement:.0f}%', xy=(x[i], y_pos),
                    ha='center', fontsize=9, color=COLORS['infodelphi'], fontweight='bold')

    ax.set_ylabel('Brier Score (lower is better)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Easy\n(n={counts[0]})", f"Medium\n(n={counts[1]})", f"Hard\n(n={counts[2]})"],
                       fontsize=9)
    ax.set_xlabel('Question Difficulty (by Single-Agent Brier)', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, max(s0_tercile_means) * 1.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_difficulty.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "fig_difficulty.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved: fig_difficulty.pdf")


# =========================================================
# Figure: Cost-Performance Trade-off
# =========================================================
def generate_cost_performance(valid_qids):
    """Scatter plot: Brier vs API cost for all methods."""
    print("--- Generating Cost-Performance Trade-off ---")

    # Approximate costs per 375 questions (gpt-5.4-mini ~ $0.003/1K input, $0.012/1K output)
    # Rough estimate: ~$0.004 per API call for this task
    cost_per_call = 0.004

    methods = [
        ('Zero-shot', 0.192, 375 * cost_per_call, 'single'),
        ('Direct', 0.227, 375 * cost_per_call, 'single'),
        ('Single-Agent', 0.216, 375 * cost_per_call, 'single'),
        ('Self-Consistency', 0.221, 375 * 5 * cost_per_call, 'single'),
        ('Superforecaster', 0.242, 375 * cost_per_call, 'single'),
        ('Halawi et al.', 0.239, 375 * 3 * cost_per_call, 'single'),
        ('Crowd Ensemble', 0.245, 375 * 5 * cost_per_call, 'multi'),
        ('MoA', 0.220, 375 * 4 * cost_per_call, 'multi'),
        ('Standard Debate', 0.202, 375 * 6 * cost_per_call, 'multi'),
        ('AIA Forecaster', 0.247, 375 * 11 * cost_per_call, 'multi'),
        ('InfoDelphi', 0.178, 375 * 6 * cost_per_call, 'ours'),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for name, brier, cost, category in methods:
        if category == 'single':
            color, marker = COLORS['single_agent'], 'o'
        elif category == 'multi':
            color, marker = COLORS['standard_debate'], 's'
        else:
            color, marker = COLORS['infodelphi'], '*'

        size = 120 if category == 'ours' else 60
        ax.scatter(cost, brier, c=color, marker=marker, s=size, zorder=5, alpha=0.85,
                   edgecolors='white', linewidth=0.5)
        # Label
        offset_x = 0.1
        offset_y = 0.003
        if name == 'InfoDelphi':
            offset_y = -0.008
        ax.annotate(name, (cost, brier), xytext=(cost + offset_x, brier + offset_y),
                    fontsize=7, alpha=0.8)

    # Legend
    legend_elements = [
        plt.scatter([], [], c=COLORS['single_agent'], marker='o', s=60, label='Single-Agent'),
        plt.scatter([], [], c=COLORS['standard_debate'], marker='s', s=60, label='Multi-Agent (homogeneous)'),
        plt.scatter([], [], c=COLORS['infodelphi'], marker='*', s=120, label='InfoDelphi (ours)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    ax.set_xlabel('Approximate Cost (USD) for 375 questions', fontsize=10)
    ax.set_ylabel('Brier Score (lower is better)', fontsize=10)
    ax.set_xlim(0, max(c for _, _, c, _ in methods) * 1.3)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cost_performance.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "fig_cost_performance.png", bbox_inches='tight', dpi=200)
    plt.close()
    print("  Saved: fig_cost_performance.pdf")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Loading valid qids...")
    valid_qids = load_valid_qids()
    print(f"  {len(valid_qids)} valid questions\n")

    generate_topic_breakdown(valid_qids)
    print()
    generate_calibration_curve(valid_qids)
    print()
    generate_belief_revision(valid_qids)
    print()
    generate_difficulty_analysis(valid_qids)
    print()
    generate_cost_performance(valid_qids)

    print("\nAll analysis figures generated!")
