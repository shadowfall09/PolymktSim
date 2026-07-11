#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from topic_classify_and_allocate import infer_topic_from_text


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_dt(value: str):
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("+00"):
        text = text + ":00"
    if text.endswith("-00"):
        text = text + ":00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def parse_outcome_prices(value: str):
    text = (value or "").strip()
    if not text:
        return None
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and len(arr) >= 2:
            return float(arr[0]), float(arr[1])
        return None
    except Exception:
        return None


def is_yes_no_market(row: Dict[str, str]) -> bool:
    outcomes = (row.get("outcomes", "") or "").replace(" ", "")
    return outcomes in ['["Yes","No"]', "['Yes','No']"]


def is_closed_before_cutoff(row: Dict[str, str], cutoff: datetime) -> bool:
    if not parse_bool(row.get("closed", "")):
        return False
    closed_time = parse_dt(row.get("closedTime", ""))
    if closed_time is None:
        return False
    return closed_time < cutoff


def has_probability_bias(row: Dict[str, str], min_bias: float):
    parsed = parse_outcome_prices(row.get("outcomePrices", ""))
    if parsed is None:
        return False, None, None
    p_yes, p_no = parsed
    bias = abs(p_yes - 0.5)
    return bias >= min_bias, p_yes, p_no


def market_quality_score(row: Dict[str, str]) -> float:
    score = 0.0
    score += 1.2 * min(1.0, parse_float(row.get("volume", "")) / 200000.0)
    score += 1.0 * min(1.0, parse_float(row.get("liquidity", "")) / 50000.0)
    score += 0.6 if parse_bool(row.get("active", "")) else 0.0
    score += 0.4 if parse_bool(row.get("acceptingOrders", "")) else 0.0
    score += 0.25 if not parse_bool(row.get("closed", "")) else 0.0

    question = (row.get("question", "") or "").strip()
    if len(question) >= 20:
        score += 0.2

    outcomes = (row.get("outcomes", "") or "").replace(" ", "")
    if outcomes in ['["Yes","No"]', "['Yes','No']"]:
        score += 0.2

    spread = parse_float(row.get("spread", ""))
    if spread > 0:
        score += max(0.0, 0.15 - min(0.15, spread * 10.0))

    return score


def read_quota(summary_path: str, target_total: int) -> Dict[str, int]:
    quota: Dict[str, int] = {}
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            topic = row.get("topic", "")
            quota[topic] = int(float(row.get("recommended_market_quota", "0") or 0))

    current_total = sum(quota.values())
    if current_total == target_total:
        return quota

    if current_total < target_total:
        max_topic = max(quota.items(), key=lambda x: x[1])[0]
        quota[max_topic] += target_total - current_total
    else:
        overflow = current_total - target_total
        topics = sorted(quota.keys(), key=lambda t: quota[t], reverse=True)
        for topic in topics:
            cut = min(overflow, quota[topic])
            quota[topic] -= cut
            overflow -= cut
            if overflow == 0:
                break
    return quota


def main():
    parser = argparse.ArgumentParser(description="Build final 500-market dataset by topic quotas.")
    parser.add_argument("--data-dir", default="data/polymarket-prediction-markets")
    parser.add_argument("--summary", default="outputs/topic_market_summary.csv")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/final_markets_500.csv")
    parser.add_argument("--stats", default="outputs/final_markets_500_stats.csv")
    parser.add_argument("--cutoff-date", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--min-bias", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)

    quota = read_quota(args.summary, args.target)

    markets_path = os.path.join(args.data_dir, "polymarket_markets.csv")
    with open(markets_path, newline="", encoding="utf-8") as f:
        markets = list(csv.DictReader(f))

    cutoff = parse_dt(args.cutoff_date)
    if cutoff is None:
        raise ValueError(f"Invalid cutoff date: {args.cutoff_date}")

    buckets: Dict[str, List[Tuple[float, Dict[str, str]]]] = defaultdict(list)
    filtered_in = 0
    for row in markets:
        if not is_yes_no_market(row):
            continue
        if not is_closed_before_cutoff(row, cutoff):
            continue
        has_bias, p_yes, p_no = has_probability_bias(row, args.min_bias)
        if not has_bias:
            continue

        topic = infer_topic_from_text((row.get("description", "") or ""))
        score = market_quality_score(row)
        row = dict(row)
        row["p_yes"] = f"{p_yes:.6f}"
        row["p_no"] = f"{p_no:.6f}"
        row["probability_bias"] = f"{abs(p_yes - 0.5):.6f}"
        buckets[topic].append((score, row))
        filtered_in += 1

    for topic in buckets:
        random.shuffle(buckets[topic])
        buckets[topic].sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict[str, str]] = []
    selected_ids = set()
    topic_selected_counter = Counter()

    for topic, need in quota.items():
        rows = buckets.get(topic, [])
        for score, row in rows:
            if topic_selected_counter[topic] >= need:
                break
            market_id = row.get("id")
            if market_id in selected_ids:
                continue
            row = dict(row)
            row["topic"] = topic
            row["quality_score"] = f"{score:.6f}"
            selected.append(row)
            selected_ids.add(market_id)
            topic_selected_counter[topic] += 1

    if len(selected) < args.target:
        remainder: List[Tuple[float, Dict[str, str], str]] = []
        for topic, rows in buckets.items():
            for score, row in rows:
                market_id = row.get("id")
                if market_id in selected_ids:
                    continue
                remainder.append((score, row, topic))
        remainder.sort(key=lambda x: x[0], reverse=True)

        for score, row, topic in remainder:
            if len(selected) >= args.target:
                break
            row = dict(row)
            row["topic"] = topic
            row["quality_score"] = f"{score:.6f}"
            selected.append(row)
            selected_ids.add(row.get("id"))
            topic_selected_counter[topic] += 1

    selected = selected[: args.target]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(selected[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    with open(args.stats, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["topic", "quota_target", "selected_count"],
        )
        writer.writeheader()
        for topic in sorted(set(list(quota.keys()) + list(topic_selected_counter.keys()))):
            writer.writerow(
                {
                    "topic": topic,
                    "quota_target": quota.get(topic, 0),
                    "selected_count": topic_selected_counter.get(topic, 0),
                }
            )

    print("output_file=", args.out)
    print("stats_file=", args.stats)
    print("selected_total=", len(selected))
    print("filtered_candidates=", filtered_in)
    print("constraints=", {
        "yes_no_outcomes": True,
        "closed_before": args.cutoff_date,
        "min_abs_p_yes_minus_0_5": args.min_bias,
    })
    print("topic_counts=", dict(topic_selected_counter))


if __name__ == "__main__":
    main()
