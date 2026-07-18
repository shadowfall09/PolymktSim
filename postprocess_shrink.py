#!/usr/bin/env python3
"""Apply the calibrated-shrink correction to an existing result JSONL offline.

Same transform as CalibratedShrinkAggregator (src/aggregation/calibrated_shrink.py),
so old runs can be re-scored without new API calls:

    p' = p0 + w * (p - p0),  w = w_hi if p > p0 else w_lo

Default params (p0=0.30, w_lo=0.8, w_hi=0.5) were chosen for cross-dataset
robustness on the 2026-07 gpt-5.4-mini s2 runs:
    polymarket_250: Brier 0.1647 -> 0.1575
    futurex 231:    Brier 0.2212 -> 0.2006

Usage:
    python postprocess_shrink.py data/results_v2/xxx.jsonl --scenario s2 \
        --out data/results_v2/xxx_shrunk.jsonl
"""
import argparse
import json
import statistics
from pathlib import Path


def shrink(p: float, p0: float, w_lo: float, w_hi: float) -> float:
    w = w_hi if p > p0 else w_lo
    return p0 + w * (p - p0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Result JSONL to post-process")
    ap.add_argument("--out", default=None, help="Output JSONL (default: <input>_shrunk.jsonl)")
    ap.add_argument("--scenario", default=None, help="Only transform this scenario (others pass through)")
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--w-lo", type=float, default=0.8)
    ap.add_argument("--w-hi", type=float, default=0.5)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_shrunk.jsonl")

    raw_briers, new_briers = [], []
    raw_hits = new_hits = resolved = 0
    n_transformed = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.scenario is None or r.get("scenario") == args.scenario:
                p_raw = float(r["p_yes"])
                p_new = shrink(p_raw, args.p0, args.w_lo, args.w_hi)
                r["p_yes_raw"] = p_raw
                r["p_yes"] = p_new
                r["label"] = "YES" if p_new >= 0.5 else "NO"
                r["postprocess"] = f"calibrated_shrink(p0={args.p0}, w_lo={args.w_lo}, w_hi={args.w_hi})"
                n_transformed += 1
                if r.get("outcome") is not None:
                    y = 1.0 if r["outcome"] else 0.0
                    r["brier"] = (p_new - y) ** 2
                    raw_briers.append((p_raw - y) ** 2)
                    new_briers.append(r["brier"])
                    raw_hits += (p_raw >= 0.5) == bool(r["outcome"])
                    new_hits += (p_new >= 0.5) == bool(r["outcome"])
                    resolved += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({n_transformed} records transformed)")
    if resolved:
        print(f"  raw:    Brier={statistics.mean(raw_briers):.4f}  median={statistics.median(raw_briers):.4f}  Acc={raw_hits / resolved:.4f}")
        print(f"  shrunk: Brier={statistics.mean(new_briers):.4f}  median={statistics.median(new_briers):.4f}  Acc={new_hits / resolved:.4f}")


if __name__ == "__main__":
    main()
