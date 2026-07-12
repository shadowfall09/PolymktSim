#!/usr/bin/env python3
"""Build a uniformly sampled, high-confidence resolved non-crypto dataset.

Selection constraints:
* binary Yes/No markets only;
* closedTime strictly after the requested cutoff;
* final Yes outcome price >= the requested threshold OR <= its complement;
* Crypto excluded using the repository's existing topic classifier.

The eligible pool is sampled uniformly with a fixed random seed. It does not
impose topic or label quotas: the resulting topic and Yes/No distributions are
the natural composition of the eligible pool.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_final_market_dataset import parse_dt, parse_outcome_prices


def infer_balanced_topic(question: str, description: str) -> str:
    """Classify with word boundaries; avoid the legacy bare-``ai`` substring bug."""
    # Resolution criteria often contain generic words such as "AI-generated",
    # "research", or "tariff" that do not describe the market topic. The
    # question plus the opening context is a much cleaner topical signal.
    text = f"{question} {description[:500]}".lower()
    def has(pattern: str) -> bool:
        return bool(re.search(pattern, text, flags=re.IGNORECASE))

    if has(r"\b(bitcoin|btc|ethereum|eth|solana|crypto|token|airdrop|defi|nft|xrp|usdt|binance)\b"):
        return "Crypto"
    # Match sport-specific resolution language as well as common sports terms.
    if has(r"\b(nba|nfl|nhl|mlb|ufc|fifa|uefa|serie a|premier league|world cup|olympic|tennis|soccer|football|grand prix|formula ?1|first 90 minutes|stoppage time|upcoming game|match)\b"):
        return "Sports"
    if has(r"\b(ukraine|russia|china|taiwan|israel|gaza|ceasefire|nato|military engagement|war)\b"):
        return "Geopolitics"
    if has(r"\b(election|president|senate|congress|governor|minister|white house|democrat|republican|mayor|parliament|vote)\b"):
        return "Politics"
    if has(r"\b(fed|interest rate|inflation|cpi|gdp|recession|employment|tariff|treasury|fomc|national debt)\b"):
        return "Macro"
    if has(r"\b(earnings|revenue|guidance|ipo|stock|shares|nasdaq|s&p|dow|nvidia|tesla|apple|microsoft|netflix|market cap)\b"):
        return "Business"
    if has(r"\b(oscar|grammy|emmy|box office|movie|film|album|netflix|celebrity|mrbeast|youtube)\b"):
        return "Entertainment"
    if has(r"\b(spacex|nasa|openai|gemini|qwen|artificial intelligence|\bai\b|robot|vaccine|clinical|science|research|model release)\b"):
        return "ScienceTech"
    return "Other"


def is_yes_no_market(row: dict[str, str]) -> bool:
    return (row.get("outcomes") or "").replace(" ", "") == '["Yes","No"]'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/polymarket-prediction-markets/polymarket_markets.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/outputs/final_markets_250_noncrypto_confident.csv"))
    parser.add_argument("--stats", type=Path, default=Path("data/outputs/final_markets_250_noncrypto_confident_stats.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/outputs/final_markets_250_noncrypto_confident_metadata.json"))
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--closed-after", default="2025-09-01T00:00:00+00:00")
    parser.add_argument("--min-confidence", type=float, default=0.9, help="Keep p_yes >= C or p_yes <= 1-C (default: 0.9).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target < 1 or not 0.5 <= args.min_confidence <= 1:
        raise ValueError("target must be positive and min-confidence must be in [0.5, 1]")
    cutoff = parse_dt(args.closed_after)
    if cutoff is None:
        raise ValueError(f"invalid --closed-after: {args.closed_after}")

    eligible: list[dict[str, str]] = []
    filter_counts: Counter[str] = Counter()
    with args.input.open(newline="", encoding="utf-8") as f:
        for source_row in csv.DictReader(f):
            filter_counts["input"] += 1
            if not is_yes_no_market(source_row):
                continue
            filter_counts["yes_no"] += 1
            if (source_row.get("closed") or "").strip().lower() != "true":
                continue
            closed_time = parse_dt(source_row.get("closedTime", ""))
            if closed_time is None or closed_time <= cutoff:
                continue
            filter_counts["closed_after"] += 1
            prices = parse_outcome_prices(source_row.get("outcomePrices", ""))
            if prices is None or not (prices[0] >= args.min_confidence or prices[0] <= 1 - args.min_confidence):
                continue
            filter_counts["p_yes_threshold"] += 1
            topic = infer_balanced_topic(
                source_row.get("question", "") or "", source_row.get("description", "") or ""
            )
            if topic == "Crypto":
                continue
            row = dict(source_row)
            row["topic"] = topic
            row["p_yes"] = f"{prices[0]:.6f}"
            row["p_no"] = f"{prices[1]:.6f}"
            row["resolved_label"] = "YES" if prices[0] >= args.min_confidence else "NO"
            eligible.append(row)

    if len(eligible) < args.target:
        raise ValueError(f"only {len(eligible)} eligible markets, below target={args.target}")
    # Uniform sampling gives every eligible market the same probability of
    # inclusion. No topic, event, volume, or outcome balancing is imposed.
    selected = random.Random(args.seed).sample(eligible, args.target)
    if len(selected) != args.target or any(row["topic"] == "Crypto" for row in selected):
        raise AssertionError("invalid final sample")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    available_by_topic = Counter(row["topic"] for row in eligible)
    selected_by_topic = Counter(row["topic"] for row in selected)
    stats_rows: list[dict[str, Any]] = []
    for topic in sorted(available_by_topic):
        topic_rows = [row for row in selected if row["topic"] == topic]
        stats_rows.append({
            "topic": topic,
            "available_candidates": available_by_topic[topic],
            "available_unique_events": len({row.get("event_id") or row.get("id") for row in eligible if row["topic"] == topic}),
            "selected_count": len(topic_rows),
            "selected_unique_events": len({row.get("event_id") or row.get("id") for row in topic_rows}),
        })
    with args.stats.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)

    metadata = {
        "input": str(args.input), "output": str(args.output), "target": args.target,
        "closed_after_exclusive": args.closed_after, "confidence_rule": f"p_yes >= {args.min_confidence} OR p_yes <= {1 - args.min_confidence}",
        "seed": args.seed, "filter_counts": dict(filter_counts),
        "eligible_total": len(eligible), "available_by_topic": dict(available_by_topic),
        "selected_by_topic": dict(selected_by_topic),
        "selected_by_label": dict(Counter(row["resolved_label"] for row in selected)),
        "sampling": "uniform without topic/event/label balancing",
        "note": "p_yes comes from final resolved outcomePrices.",
    }
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
