#!/usr/bin/env python3
"""Post-hoc Platt recalibration of forecast result files.

Fits p' = sigmoid(a * logit(p) + b) by minimizing log loss on the --fit files,
then reports raw vs recalibrated Brier/accuracy/ECE on the --apply file.
Always fit on datasets disjoint from the one you report (no leakage).

Background: the 2026-07-11 benchmark shows every method shares a systematic
overconfident-toward-YES bias (cross-dataset fits give a ≈ 0.4-0.6, b ≈ -0.6);
correcting it is worth ~0.03 Brier on PolyMarket/FutureX — several times the
gap between any two protocols.

Usage:
    python recalibrate.py \
        --fit data/results/futurex_infodelphi.jsonl data/results/forecastbench_infodelphi.jsonl \
        --apply data/results/polymarket_375_infodelphi.jsonl \
        --scenario s2
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path: str | Path, scenario: str | None = None) -> list[tuple[float, float]]:
    """Return [(p_yes, outcome)] with resolved outcomes, deduped by (qid, scenario) keeping the last record."""
    dedup: dict[tuple, tuple[float, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("outcome") is None:
                continue
            if scenario and r.get("scenario") != scenario:
                continue
            key = (r["qid"], r.get("scenario"))
            dedup[key] = (float(r["p_yes"]), 1.0 if r["outcome"] else 0.0)
    return list(dedup.values())


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def fit_platt(data: list[tuple[float, float]], lr: float = 0.5, steps: int = 2000) -> tuple[float, float]:
    """Gradient descent on log loss over (a, b)."""
    ps = np.array([p for p, _ in data])
    ys = np.array([y for _, y in data])
    x = logit(ps)
    a, b = 1.0, 0.0
    for _ in range(steps):
        q = sigmoid(a * x + b)
        a -= lr * float(((q - ys) * x).mean())
        b -= lr * float((q - ys).mean())
    return a, b


def ece(ps: np.ndarray, ys: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        hi_incl = ps <= edges[i + 1] if i == bins - 1 else ps < edges[i + 1]
        m = (ps >= edges[i]) & hi_incl
        if m.sum():
            total += m.sum() / len(ps) * abs(ps[m].mean() - ys[m].mean())
    return total


def report(name: str, data: list[tuple[float, float]], a: float = 1.0, b: float = 0.0) -> None:
    ps = sigmoid(a * logit(np.array([p for p, _ in data])) + b)
    ys = np.array([y for _, y in data])
    brier = float(((ps - ys) ** 2).mean())
    acc = float(((ps >= 0.5) == ys).mean())
    print(f"  {name:14s} n={len(ps)}  Brier={brier:.4f}  Acc={acc:.3f}  ECE={ece(ps, ys):.4f}  mean_p={ps.mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True, help="Result JSONL file(s) to fit (a, b) on")
    ap.add_argument("--apply", required=True, help="Held-out result JSONL file to evaluate")
    ap.add_argument("--scenario", default=None, help="Filter records by scenario (e.g. s1, s2)")
    args = ap.parse_args()

    train = [rec for path in args.fit for rec in load_records(path, args.scenario)]
    test = load_records(args.apply, args.scenario)
    if not train or not test:
        raise SystemExit(f"no resolved records (train={len(train)}, test={len(test)}); check paths/--scenario")

    a, b = fit_platt(train)
    print(f"Fitted on {len(train)} records from {len(args.fit)} file(s): a={a:.3f}, b={b:+.3f}")
    print(f"  (a < 1 → predictions are overconfident; b < 0 → biased toward YES)")
    print(f"Held-out: {args.apply}" + (f"  [scenario={args.scenario}]" if args.scenario else ""))
    report("raw", test)
    report("recalibrated", test, a, b)


if __name__ == "__main__":
    main()
