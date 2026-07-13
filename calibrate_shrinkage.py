#!/usr/bin/env python3
"""Offline probability shrinkage ablation for JSONL forecast results."""

import argparse
import csv
import json
import math
from collections import defaultdict


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("outcome") is None:
                continue
            records.append(record)
    return records


def shrink(p_yes, alpha):
    return 0.5 + alpha * (p_yes - 0.5)


def logloss(p_yes, outcome):
    p_yes = max(1e-6, min(1 - 1e-6, p_yes))
    return -(math.log(p_yes) if outcome else math.log(1 - p_yes))


def confidence_ece(preds, outcomes, n_bins=10):
    bins = defaultdict(list)
    for p_yes, outcome in zip(preds, outcomes):
        confidence = max(p_yes, 1 - p_yes)
        idx = min(int((confidence - 0.5) / 0.05), n_bins - 1)
        bins[idx].append((p_yes, outcome, confidence))

    ece = 0.0
    n = len(preds)
    for values in bins.values():
        avg_conf = sum(v[2] for v in values) / len(values)
        acc = sum((v[0] >= 0.5) == bool(v[1]) for v in values) / len(values)
        ece += abs(avg_conf - acc) * len(values) / n
    return ece


def metrics(records, alpha):
    preds = [shrink(float(r["p_yes"]), alpha) for r in records]
    outcomes = [bool(r["outcome"]) for r in records]
    n = len(records)
    brier = sum((p - float(y)) ** 2 for p, y in zip(preds, outcomes)) / n
    acc = sum((p >= 0.5) == y for p, y in zip(preds, outcomes)) / n
    avg_conf = sum(max(p, 1 - p) for p in preds) / n
    return {
        "n": n,
        "alpha": alpha,
        "brier": brier,
        "accuracy": acc,
        "ece": confidence_ece(preds, outcomes),
        "logloss": sum(logloss(p, y) for p, y in zip(preds, outcomes)) / n,
        "avg_confidence": avg_conf,
    }


def analytic_alpha(records):
    numerator = 0.0
    denominator = 0.0
    for r in records:
        x = float(r["p_yes"]) - 0.5
        target = float(bool(r["outcome"])) - 0.5
        numerator += x * target
        denominator += x * x
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def parse_alphas(value):
    return [float(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSONL result file")
    parser.add_argument("--scenario", default=None, help="Optional scenario filter, e.g. s2")
    parser.add_argument(
        "--alphas",
        default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0",
        help="Comma-separated shrinkage alphas",
    )
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    records = load_records(args.input)
    if args.scenario is not None:
        records = [r for r in records if r.get("scenario") == args.scenario]
    if not records:
        raise SystemExit("No records matched the requested input/scenario.")

    rows = []
    base_brier = metrics(records, 1.0)["brier"]
    for alpha in parse_alphas(args.alphas):
        row = metrics(records, alpha)
        row["delta_brier"] = row["brier"] - base_brier
        row["kind"] = "grid"
        rows.append(row)

    alpha_star = analytic_alpha(records)
    star = metrics(records, alpha_star)
    star["delta_brier"] = star["brier"] - base_brier
    star["kind"] = "analytic"
    rows.append(star)

    fieldnames = [
        "kind",
        "n",
        "alpha",
        "brier",
        "delta_brier",
        "accuracy",
        "ece",
        "logloss",
        "avg_confidence",
    ]
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[k]) for k in fieldnames))

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
