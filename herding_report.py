#!/usr/bin/env python3
"""Round-by-round disagreement report from *_detail.jsonl files.

The core diagnostic for the InfoDelphi claim: if between-agent std collapses
after one deliberation round (e.g. 0.07 → 0.015, as in the 2026-07-11 runs),
the agents are herding and evidence routing is not preserving diversity.
A protocol fix should keep "retention" (round-k std / round-0 std) well above
the ~0.2 observed with the legacy full share mode.

Usage:
    python herding_report.py data/results/polymarket_375_infodelphi_detail.jsonl \
                             data/results/polymarket_375_standard_debate_detail.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def brier(p: float, outcome: bool) -> float:
    return (p - (1.0 if outcome else 0.0)) ** 2


def load_detail(path: str | Path) -> dict:
    """Group per-agent records: {scenario: {qid: {round_id: [(p_yes, agent_id)]}}} + outcomes."""
    dedup: dict[tuple, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("scenario"), r["qid"], r.get("agent_id"), r.get("round_id"))
            dedup[key] = r
    grouped: dict = defaultdict(lambda: defaultdict(dict))
    outcomes: dict = {}
    for r in dedup.values():
        grouped[r.get("scenario")][r["qid"]].setdefault(r.get("round_id", 0), []).append(r["p_yes"])
        if r.get("outcome") is not None:
            outcomes[r["qid"]] = r["outcome"]
    return {"grouped": grouped, "outcomes": outcomes}


def report_file(path: str | Path) -> None:
    data = load_detail(path)
    print(f"== {path}")
    for scenario, by_q in sorted(data["grouped"].items(), key=lambda kv: str(kv[0])):
        rounds = sorted({rd for q in by_q.values() for rd in q})
        base_std = None
        for rd in rounds:
            stds, ind_briers, agg_briers = [], [], []
            for qid, per_round in by_q.items():
                ps = per_round.get(rd)
                if not ps:
                    continue
                stds.append(float(np.std(ps)))
                outcome = data["outcomes"].get(qid)
                if outcome is not None:
                    ind_briers += [brier(p, outcome) for p in ps]
                    agg_briers.append(brier(float(np.mean(ps)), outcome))
            if not stds:
                continue
            mean_std = float(np.mean(stds))
            if base_std is None:
                base_std = mean_std
            retention = mean_std / base_std if base_std > 1e-9 else float("nan")
            print(f"  [{scenario}] round {rd}: n={len(stds):4d}  agent_std={mean_std:.4f}  "
                  f"retention={retention:.2f}  agent_brier={np.mean(ind_briers):.4f}  "
                  f"agg(mean)_brier={np.mean(agg_briers):.4f}")
        if base_std is not None and len(rounds) > 1:
            print(f"  [{scenario}] retention < ~0.3 means the deliberation is herding; "
                  f"diversity should survive the final round.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("detail_files", nargs="+", help="*_detail.jsonl files to analyze")
    args = ap.parse_args()
    for path in args.detail_files:
        report_file(path)


if __name__ == "__main__":
    main()
