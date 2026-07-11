#!/usr/bin/env python3
"""Analyze all full-scale runs on the 375-question filtered subset.

Applies the temporal filter (endDate >= 2025-09-01) to all results from
rows_1-500 / full_500 runs, and produces a unified comparison table for paper use.

Usage:
    python analyze_375.py
    python analyze_375.py --by-topic
    python analyze_375.py --calibration
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

MARKETS_CSV = Path("data/outputs/final_markets_500.csv")
RESULTS_DIR = Path("data/results")
CUTOFF = date(2025, 9, 1)

# Model name normalization (OpenRouter -> standard)
MODEL_ALIASES = {
    "openai/gpt-5.4-mini": "gpt-5.4-mini",
    "google/gemini-3-flash-preview": "gemini-3-flash-preview",
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
}


def get_valid_row_indices() -> set[int]:
    """Return row indices (1-based) that pass the temporal filter."""
    valid = set()
    with open(MARKETS_CSV) as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            endiso = row.get("endDateIso") or row.get("endDate") or ""
            if not endiso:
                continue
            try:
                ed = date.fromisoformat(str(endiso)[:10])
            except (ValueError, TypeError):
                continue
            if ed >= CUTOFF:
                valid.add(row_idx)
    return valid


def extract_row_num(qid: str) -> int | None:
    try:
        return int(qid.split("_")[1])
    except (IndexError, ValueError):
        return None


def load_filtered(filepath: Path, valid_rows: set[int]) -> list[dict]:
    """Load JSONL and filter to valid rows."""
    records = []
    with open(filepath) as f:
        for line in f:
            r = json.loads(line)
            row_num = extract_row_num(r.get("qid", ""))
            if row_num and row_num in valid_rows and r.get("outcome") is not None:
                records.append(r)
    return records


def compute_metrics(records: list[dict]) -> dict:
    """Compute Brier, LogLoss, Accuracy, ECE for a list of records."""
    if not records:
        return {"n": 0, "brier": None, "logloss": None, "acc": None, "ece": None}
    briers, accs, loglosses = [], [], []
    for r in records:
        p = r["p_yes"]
        truth = 1.0 if r["outcome"] else 0.0
        briers.append((p - truth) ** 2)
        accs.append(int((p >= 0.5) == bool(r["outcome"])))
        p_clipped = max(1e-6, min(1 - 1e-6, p))
        loglosses.append(-(math.log(p_clipped) if r["outcome"] else math.log(1 - p_clipped)))
    # ECE (5 bins)
    bins = defaultdict(lambda: {"preds": [], "outcomes": []})
    for r in records:
        b = min(int(r["p_yes"] * 5), 4)
        bins[b]["preds"].append(r["p_yes"])
        bins[b]["outcomes"].append(float(r["outcome"]))
    ece = 0.0
    for b_data in bins.values():
        if b_data["preds"]:
            mp = sum(b_data["preds"]) / len(b_data["preds"])
            mo = sum(b_data["outcomes"]) / len(b_data["outcomes"])
            ece += abs(mp - mo) * len(b_data["preds"])
    ece /= len(records)
    n = len(records)
    return {
        "n": n,
        "brier": sum(briers) / n,
        "logloss": sum(loglosses) / n,
        "acc": sum(accs) / n,
        "ece": ece,
    }


def get_run_metadata() -> dict[str, dict]:
    """Load run metadata from runs.jsonl."""
    meta = {}
    runs_path = RESULTS_DIR / "runs.jsonl"
    if not runs_path.exists():
        return meta
    with open(runs_path) as f:
        for line in f:
            r = json.loads(line)
            ts = r["run_ts"]
            if ts not in meta:
                meta[ts] = r
    return meta


def find_full_runs() -> list[Path]:
    """Find result files from full-scale runs (rows_1-500 or full_500)."""
    runs_meta = get_run_metadata()
    full_run_ts = set()
    for ts, m in runs_meta.items():
        ds = m.get("dataset", "")
        if "500" in ds or "rows_1-500" in ds:
            full_run_ts.add(ts)
    # Also include s0_fair_temp07 if it exists
    s0_fair = RESULTS_DIR / "s0_fair_temp07.jsonl"
    paths = []
    for ts in sorted(full_run_ts):
        p = RESULTS_DIR / f"{ts}.jsonl"
        if p.exists():
            paths.append(p)
    if s0_fair.exists():
        paths.append(s0_fair)
    return paths


def deduplicate_by_scenario(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by scenario, keeping first occurrence per (scenario, qid)."""
    by_scenario = defaultdict(dict)
    for r in records:
        sc = r["scenario"]
        qid = r["qid"]
        if qid not in by_scenario[sc]:
            by_scenario[sc][qid] = r
    return {sc: list(qids.values()) for sc, qids in by_scenario.items()}


def main():
    ap = argparse.ArgumentParser(description="Analyze 375-question filtered results")
    ap.add_argument("--by-topic", action="store_true", help="Show per-topic breakdown")
    ap.add_argument("--calibration", action="store_true", help="Show calibration details")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-run file info")
    args = ap.parse_args()

    valid_rows = get_valid_row_indices()
    print(f"Temporal filter: endDate >= {CUTOFF} → {len(valid_rows)} valid questions\n")

    runs_meta = get_run_metadata()
    full_paths = find_full_runs()

    if not full_paths:
        print("No full-scale run files found.")
        return

    # Collect all results
    all_results = []  # (label, scenario, metrics_dict)

    print("=" * 110)
    print(f"{'Setting':<55} {'Scen':<5} {'n':<5} {'Brier':<8} {'LogLoss':<8} {'Acc':<7} {'ECE':<7}")
    print("=" * 110)

    for path in full_paths:
        ts = path.stem.replace("_detail", "")
        records = load_filtered(path, valid_rows)
        if not records:
            continue

        by_scenario = deduplicate_by_scenario(records)

        # Get metadata
        meta = runs_meta.get(ts, {})
        model = MODEL_ALIASES.get(meta.get("model", ""), meta.get("model", ts))
        agg = meta.get("aggregator", "?")
        bm25 = "BM25" if meta.get("bm25") else "rand"
        pr = meta.get("public_ratio", "?")
        temp = meta.get("temperature", "?")

        # Special case for s0_fair_temp07
        if "s0_fair" in ts:
            model = "gpt-5.4-mini"
            agg = "—"
            bm25 = "—"
            pr = "all"
            temp = 0.7

        for sc in sorted(by_scenario.keys()):
            recs = by_scenario[sc]
            m = compute_metrics(recs)
            if sc == "s0":
                label = f"{model} | {sc} | temp={temp}"
            else:
                label = f"{model} | {agg} | {bm25} | PR={pr} | temp={temp}"

            print(f"{label:<55} {sc:<5} {m['n']:<5} {m['brier']:<8.4f} {m['logloss']:<8.4f} {m['acc']:<7.3f} {m['ece']:<7.4f}")
            all_results.append((label, sc, m))

    print("=" * 110)
    print(f"\nTotal configurations evaluated: {len(all_results)}")

    # Best results summary
    print("\n" + "=" * 60)
    print("  BEST RESULTS (sorted by Brier)")
    print("=" * 60)
    sorted_results = sorted(all_results, key=lambda x: x[2]["brier"] or 999)
    for label, sc, m in sorted_results[:10]:
        print(f"  {m['brier']:.4f}  {m['acc']:.3f}  {sc:<3}  {label}")

    # Topic breakdown
    if args.by_topic:
        print("\n" + "=" * 60)
        print("  BY TOPIC (best config per topic)")
        print("=" * 60)
        # Reload and analyze by topic for the best overall run
        best_path = full_paths[0]  # TODO: identify best run
        records = load_filtered(best_path, valid_rows)
        topics = sorted(set(r.get("topic", "?") for r in records))
        by_scenario = deduplicate_by_scenario(records)
        for sc in sorted(by_scenario.keys()):
            print(f"\n  [{sc}]")
            print(f"  {'Topic':<18} {'n':>4} {'Brier':>8} {'Acc':>7}")
            print(f"  {'-'*40}")
            for topic in topics:
                sub = [r for r in by_scenario[sc] if r.get("topic") == topic]
                if sub:
                    m = compute_metrics(sub)
                    print(f"  {topic:<18} {m['n']:>4} {m['brier']:>8.4f} {m['acc']:>7.3f}")


if __name__ == "__main__":
    main()
