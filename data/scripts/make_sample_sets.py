#!/usr/bin/env python3
"""Create reusable sample subsets for faster experiment iteration."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MARKETS_CSV = ROOT / "data/outputs/final_markets_500.csv"
EVIDENCES_DIR = ROOT / "data/evidences"
RESULTS_DIR = ROOT / "data/results"
SAMPLES_DIR = ROOT / "data/samples"


def parse_args():
    ap = argparse.ArgumentParser(description="Build sample subsets from a finished run")
    ap.add_argument("--run", required=True, help="Run timestamp, e.g. 20260407_160500")
    ap.add_argument("--dev-small-size", type=int, default=48)
    ap.add_argument("--targeted-size", type=int, default=80)
    ap.add_argument("--min-evidence", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=str(SAMPLES_DIR))
    return ap.parse_args()


def load_run_records(path: Path):
    records = [json.loads(line) for line in path.open() if line.strip()]
    by_qid = defaultdict(dict)
    for rec in records:
        by_qid[rec["qid"]][rec["scenario"]] = rec
    return by_qid


def iter_market_rows():
    with MARKETS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            yield row_idx, row


def evidence_count_for(row_idx: int, market_id: str) -> int:
    path = EVIDENCES_DIR / f"row_{row_idx:04d}_{market_id}.json"
    if not path.exists():
        return 0
    with path.open() as f:
        data = json.load(f)
    return len(data.get("evidences", []))


def build_metadata(run_records: dict[str, dict], min_evidence: int):
    rows = []
    for row_idx, row in iter_market_rows():
        qid = f"row_{row_idx:04d}_{row['id']}"
        run_row = run_records.get(qid)
        if not run_row:
            continue
        s0_record = run_row.get("s0") or run_row.get("s1") or run_row.get("s2") or {}
        outcome = s0_record.get("outcome")
        evidence_count = evidence_count_for(row_idx, row["id"])
        meta = {
            "qid": qid,
            "row_idx": row_idx,
            "market_id": row["id"],
            "question": row["question"],
            "topic": row.get("topic", ""),
            "outcome": outcome,
            "evidence_count": evidence_count,
            "eligible_for_targeted": outcome is not None and evidence_count >= min_evidence and "s0" in run_row,
        }
        for scenario in ("s0", "s1", "s2"):
            rec = run_row.get(scenario)
            meta[f"{scenario}_p_yes"] = rec.get("p_yes") if rec else None
            meta[f"{scenario}_brier"] = rec.get("brier") if rec else None
            meta[f"{scenario}_label"] = rec.get("label") if rec else None
        if meta["s0_p_yes"] is not None:
            meta["s0_confidence"] = abs(meta["s0_p_yes"] - 0.5)
            meta["s0_high_conf_wrong"] = (
                outcome is not None
                and meta["s0_confidence"] > 0.4
                and ((meta["s0_p_yes"] >= 0.5) != outcome)
            )
        else:
            meta["s0_confidence"] = None
            meta["s0_high_conf_wrong"] = False
        if meta["s0_brier"] is not None and meta["s1_brier"] is not None:
            meta["s1_gain_vs_s0"] = meta["s0_brier"] - meta["s1_brier"]
        else:
            meta["s1_gain_vs_s0"] = None
        rows.append(meta)
    return rows


def choose_dev_small(metadata: list[dict], size: int, seed: int):
    rng = random.Random(seed)
    eligible = [m for m in metadata if m["outcome"] is not None and m["evidence_count"] > 0]
    by_topic = defaultdict(list)
    for row in eligible:
        by_topic[row["topic"]].append(row)
    for rows in by_topic.values():
        rng.shuffle(rows)

    selected = []
    used = set()
    topics = sorted(by_topic, key=lambda t: (-len(by_topic[t]), t))

    # Round-robin by topic to keep the dev set broad.
    while len(selected) < size:
        progressed = False
        for topic in topics:
            while by_topic[topic]:
                candidate = by_topic[topic].pop()
                if candidate["qid"] in used:
                    continue
                selected.append(candidate)
                used.add(candidate["qid"])
                progressed = True
                break
            if len(selected) >= size:
                break
        if not progressed:
            break

    # Reorder for easier inspection: topic, then row index.
    return sorted(selected, key=lambda r: (r["topic"], r["row_idx"]))


def targeted_score(row: dict) -> float:
    gain = row["s1_gain_vs_s0"] or 0.0
    confidence_bonus = row["s0_confidence"] or 0.0
    evidence_bonus = min(row["evidence_count"], 20) / 100.0
    score = (row["s0_brier"] or 0.0) + 0.35 * gain + 0.15 * confidence_bonus + evidence_bonus
    if row["s0_high_conf_wrong"]:
        score += 0.5
    return score


def choose_targeted(metadata: list[dict], size: int, seed: int):
    rng = random.Random(seed)
    eligible = [m for m in metadata if m["eligible_for_targeted"]]
    eligible.sort(key=targeted_score, reverse=True)

    selected = []
    used = set()
    by_topic_count = defaultdict(int)

    # First pass: cap each topic so Crypto does not swallow the whole set.
    for row in eligible:
        if len(selected) >= size:
            break
        if row["qid"] in used:
            continue
        if by_topic_count[row["topic"]] >= max(3, size // 4):
            continue
        selected.append(row)
        used.add(row["qid"])
        by_topic_count[row["topic"]] += 1

    if len(selected) < size:
        remainder = [row for row in eligible if row["qid"] not in used]
        rng.shuffle(remainder)
        remainder.sort(key=targeted_score, reverse=True)
        selected.extend(remainder[: size - len(selected)])

    return sorted(selected[:size], key=lambda r: (-targeted_score(r), r["row_idx"]))


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_qids(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(row["qid"] + "\n")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_path = RESULTS_DIR / f"{args.run}.jsonl"
    records = load_run_records(run_path)
    metadata = build_metadata(records, min_evidence=args.min_evidence)

    metadata_path = out_dir / f"{args.run}_metadata.jsonl"
    write_jsonl(metadata_path, metadata)

    dev_small = choose_dev_small(metadata, args.dev_small_size, args.seed)
    targeted = choose_targeted(metadata, args.targeted_size, args.seed)

    dev_small_jsonl = out_dir / f"{args.run}_dev_small.jsonl"
    dev_small_qids = out_dir / f"{args.run}_dev_small.qids.txt"
    targeted_jsonl = out_dir / f"{args.run}_dev_targeted.jsonl"
    targeted_qids = out_dir / f"{args.run}_dev_targeted.qids.txt"

    write_jsonl(dev_small_jsonl, dev_small)
    write_qids(dev_small_qids, dev_small)
    write_jsonl(targeted_jsonl, targeted)
    write_qids(targeted_qids, targeted)

    print(f"metadata      : {metadata_path}")
    print(f"dev_small     : {dev_small_jsonl} ({len(dev_small)} rows)")
    print(f"dev_small qids: {dev_small_qids}")
    print(f"targeted      : {targeted_jsonl} ({len(targeted)} rows)")
    print(f"targeted qids : {targeted_qids}")


if __name__ == "__main__":
    main()
