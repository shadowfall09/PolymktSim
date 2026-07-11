#!/usr/bin/env python3
"""
Analyze PolymktSim experiment results.
Usage:
    python analyze_results.py                          # latest run
    python analyze_results.py --run 20260407_160500   # specific run
    python analyze_results.py --results-dir data/results
"""
import argparse
import json
import math
import os
import glob
from collections import defaultdict

# ── helpers ────────────────────────────────────────────────────────────────────

def load_main(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def brier(p, outcome):
    return (p - float(outcome)) ** 2

def logloss(p, outcome):
    p = max(1e-6, min(1 - 1e-6, p))
    return -(math.log(p) if outcome else math.log(1 - p))

def accuracy(p, outcome):
    pred = p >= 0.5
    return int(pred == bool(outcome))

def calibration_bins(records, n_bins=10):
    """Return (bin_center, mean_pred, mean_outcome, count) per bin."""
    bins = defaultdict(lambda: {"preds": [], "outcomes": []})
    for r in records:
        b = min(int(r["p_yes"] * n_bins), n_bins - 1)
        bins[b]["preds"].append(r["p_yes"])
        bins[b]["outcomes"].append(float(r["outcome"]))
    result = []
    for i in range(n_bins):
        if bins[i]["preds"]:
            center = (i + 0.5) / n_bins
            result.append((
                center,
                sum(bins[i]["preds"]) / len(bins[i]["preds"]),
                sum(bins[i]["outcomes"]) / len(bins[i]["outcomes"]),
                len(bins[i]["preds"]),
            ))
    return result

def stats(values):
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, median=None, std=None)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
    return dict(n=n, mean=mean, median=median, std=std)

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "  N/A "

# ── analysis functions ──────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def analyze_overall(records):
    section("OVERALL SUMMARY")
    valid = [r for r in records if r["outcome"] is not None]
    null_count = len(records) - len(valid)
    print(f"  Total records : {len(records)}")
    print(f"  Resolved      : {len(valid)}  (skipped {null_count} with outcome=None)")

    scenarios = sorted(set(r["scenario"] for r in valid))
    print(f"  Scenarios     : {scenarios}")
    print(f"  Model         : {records[0].get('model','?')}")
    print(f"  num_agents    : {records[0].get('num_agents','?')}")
    print(f"  num_rounds    : {records[0].get('num_rounds','?')}")

def analyze_scenario(records):
    section("BY SCENARIO")
    valid = [r for r in records if r["outcome"] is not None]

    header = f"  {'Scenario':<10} {'N':>5}  {'Brier':>8}  {'LogLoss':>8}  {'Accuracy':>9}  {'Calib ECE':>10}"
    print(header)
    print("  " + "-" * 58)

    for scenario in sorted(set(r["scenario"] for r in valid)):
        sub = [r for r in valid if r["scenario"] == scenario]
        briers   = [r["brier"] for r in sub]
        loglosses = [logloss(r["p_yes"], r["outcome"]) for r in sub]
        accs     = [accuracy(r["p_yes"], r["outcome"]) for r in sub]

        # ECE (expected calibration error)
        cal = calibration_bins(sub)
        ece = sum(abs(mp - mo) * cnt for _, mp, mo, cnt in cal) / len(sub) if cal else None

        n = len(sub)
        print(f"  {scenario:<10} {n:>5}  {sum(briers)/n:>8.4f}  "
              f"{sum(loglosses)/n:>8.4f}  {sum(accs)/n:>9.3f}  "
              f"{ece:>10.4f}" if ece else
              f"  {scenario:<10} {n:>5}  {sum(briers)/n:>8.4f}  "
              f"{sum(loglosses)/n:>8.4f}  {sum(accs)/n:>9.3f}  {'N/A':>10}")

def analyze_by_topic(records):
    section("BY TOPIC × SCENARIO")
    valid = [r for r in records if r["outcome"] is not None]
    topics = sorted(set(r["topic"] for r in valid))
    scenarios = sorted(set(r["scenario"] for r in valid))

    # header
    col = 12
    header = f"  {'Topic':<18}" + "".join(f"  {s:>{col}}" for s in scenarios)
    print(header + "  (Mean Brier)")
    print("  " + "-" * (18 + (col + 2) * len(scenarios)))

    for topic in topics:
        row = f"  {topic:<18}"
        for scenario in scenarios:
            sub = [r for r in valid if r["topic"] == topic and r["scenario"] == scenario]
            if sub:
                b = sum(r["brier"] for r in sub) / len(sub)
                row += f"  {b:>{col}.4f}"
            else:
                row += f"  {'—':>{col}}"
        print(row)

def analyze_difficulty(records):
    section("OUTCOME DISTRIBUTION & DIFFICULTY")
    valid = [r for r in records if r["outcome"] is not None]
    n = len(valid)
    yes_count = sum(1 for r in valid if r["outcome"])
    no_count = n - yes_count
    print(f"  YES outcomes : {yes_count} ({100*yes_count/n:.1f}%)")
    print(f"  NO  outcomes : {no_count} ({100*no_count/n:.1f}%)")

    # confidence analysis per scenario
    print()
    print(f"  {'Scenario':<10}  {'High-conf correct':>18}  {'High-conf wrong':>16}  {'Low-conf (0.4-0.6)':>19}")
    print("  " + "-" * 68)
    for scenario in sorted(set(r["scenario"] for r in valid)):
        sub = [r for r in valid if r["scenario"] == scenario]
        hc_correct = sum(1 for r in sub if abs(r["p_yes"] - 0.5) > 0.4 and accuracy(r["p_yes"], r["outcome"]))
        hc_wrong   = sum(1 for r in sub if abs(r["p_yes"] - 0.5) > 0.4 and not accuracy(r["p_yes"], r["outcome"]))
        low_conf   = sum(1 for r in sub if 0.4 <= r["p_yes"] <= 0.6)
        print(f"  {scenario:<10}  {hc_correct:>10} ({100*hc_correct/len(sub):5.1f}%)  "
              f"{hc_wrong:>8} ({100*hc_wrong/len(sub):5.1f}%)  "
              f"{low_conf:>8} ({100*low_conf/len(sub):5.1f}%)")

def analyze_calibration(records):
    section("CALIBRATION (prediction vs actual frequency)")
    valid = [r for r in records if r["outcome"] is not None]

    for scenario in sorted(set(r["scenario"] for r in valid)):
        sub = [r for r in valid if r["scenario"] == scenario]
        cal = calibration_bins(sub, n_bins=5)
        print(f"\n  {scenario} — bin center | mean pred | mean outcome | count")
        print(f"  {'Bin':>6}  {'Pred':>8}  {'Actual':>8}  {'N':>6}  {'Gap':>8}")
        print("  " + "-" * 44)
        for center, mp, mo, cnt in cal:
            gap = mp - mo
            print(f"  {center:>6.2f}  {mp:>8.3f}  {mo:>8.3f}  {cnt:>6}  {gap:>+8.3f}")

def analyze_worst(records, n=10):
    section(f"TOP {n} WORST PREDICTIONS (highest Brier per scenario)")
    valid = [r for r in records if r["outcome"] is not None and r["brier"] is not None]

    for scenario in sorted(set(r["scenario"] for r in valid)):
        sub = sorted([r for r in valid if r["scenario"] == scenario],
                     key=lambda x: x["brier"], reverse=True)[:n]
        print(f"\n  [{scenario}]")
        print(f"  {'Brier':>6}  {'p_yes':>6}  {'label':>5}  Question")
        print("  " + "-" * 70)
        for r in sub:
            q = r["question"][:55] + "…" if len(r["question"]) > 55 else r["question"]
            print(f"  {r['brier']:>6.4f}  {r['p_yes']:>6.2f}  {r['label']:>5}  {q}")

# ── main ────────────────────────────────────────────────────────────────────────

def find_latest(results_dir):
    files = glob.glob(os.path.join(results_dir, "*.jsonl"))
    # exclude detail files
    main_files = [f for f in files if "_detail" not in os.path.basename(f)]
    if not main_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    return sorted(main_files)[-1]

def main():
    ap = argparse.ArgumentParser(description="Analyze PolymktSim results")
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--run", default=None, help="run timestamp, e.g. 20260407_160500")
    ap.add_argument("--no-worst", action="store_true", help="skip worst-predictions table")
    args = ap.parse_args()

    if args.run:
        path = os.path.join(args.results_dir, f"{args.run}.jsonl")
    else:
        path = find_latest(args.results_dir)

    print(f"\nLoading: {path}")
    records = load_main(path)

    analyze_overall(records)
    analyze_scenario(records)
    analyze_by_topic(records)
    analyze_difficulty(records)
    analyze_calibration(records)
    if not args.no_worst:
        analyze_worst(records)

    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")

if __name__ == "__main__":
    main()
