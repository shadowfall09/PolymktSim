#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


TOPIC_KEYWORDS = {
    "Sports": [" vs ", "nba", "nfl", "nhl", "mlb", "ufc", "world cup", "olympic", "tennis", "soccer", "playoff", "championship"],
    "Politics": ["election", "president", "senate", "congress", "governor", "minister", "white house", "poll", "vote", "democrat", "republican"],
    "Crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "token", "airdrop", "defi", "nft", "xrp"],
    "Macro": ["fed", "interest rate", "inflation", "cpi", "gdp", "recession", "employment", "tariff", "treasury", "fomc"],
    "Business": ["earnings", "revenue", "guidance", "ipo", "stock", "shares", "nasdaq", "s&p", "dow", "tesla", "apple", "microsoft"],
    "Entertainment": ["oscar", "grammy", "emmy", "box office", "movie", "album", "tv show", "celebrity", "pop culture"],
    "ScienceTech": ["spacex", "nasa", "launch", "ai", "openai", "robot", "vaccine", "clinical", "science", "research"],
    "Geopolitics": ["ukraine", "russia", "china", "taiwan", "israel", "gaza", "war", "ceasefire", "nato", "geopolitics"],
}

TOPIC_ORDER = [
    "Sports",
    "Politics",
    "Crypto",
    "Macro",
    "Business",
    "Entertainment",
    "ScienceTech",
    "Geopolitics",
    "Other",
]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def infer_topic_from_text(text: str) -> str:
    lowered = (text or "").lower()
    for topic in TOPIC_ORDER:
        if topic == "Other":
            continue
        if any(keyword in lowered for keyword in TOPIC_KEYWORDS[topic]):
            return topic
    return "Other"


def infer_event_topic(event_row: Dict[str, str]) -> str:
    text = " ".join(
        [
            event_row.get("title", "") or "",
            event_row.get("slug", "") or "",
            event_row.get("seriesSlug", "") or "",
            event_row.get("tags", "") or "",
            event_row.get("description", "") or "",
        ]
    )
    return infer_topic_from_text(text)


def event_popularity_score(event_row: Dict[str, str]) -> float:
    volume = parse_float(event_row.get("volume", ""))
    liquidity = parse_float(event_row.get("liquidity", ""))
    comment_count = parse_float(event_row.get("commentCount", ""))
    market_count = parse_float(event_row.get("market_count", ""))

    score = (
        1.00 * math.log1p(max(0.0, volume))
        + 0.70 * math.log1p(max(0.0, liquidity))
        + 0.35 * math.log1p(max(0.0, comment_count))
        + 0.15 * math.log1p(max(0.0, market_count))
    )
    return score


def allocate_by_share(
    popularity_by_topic: Dict[str, float],
    available_by_topic: Dict[str, int],
    total_target: int,
) -> Dict[str, int]:
    topics = list(popularity_by_topic.keys())
    total_popularity = sum(popularity_by_topic.values()) or 1.0

    raw = {topic: total_target * popularity_by_topic[topic] / total_popularity for topic in topics}
    base = {topic: min(int(raw[topic]), available_by_topic.get(topic, 0)) for topic in topics}

    assigned = sum(base.values())
    remain = max(0, total_target - assigned)

    candidates: List[Tuple[float, str]] = []
    for topic in topics:
        cap_left = available_by_topic.get(topic, 0) - base[topic]
        if cap_left <= 0:
            continue
        frac = raw[topic] - int(raw[topic])
        candidates.append((frac, topic))

    candidates.sort(reverse=True)

    idx = 0
    while remain > 0 and candidates:
        _, topic = candidates[idx % len(candidates)]
        if base[topic] < available_by_topic.get(topic, 0):
            base[topic] += 1
            remain -= 1
        idx += 1
        if idx > 100000:
            break

    return base


def main():
    parser = argparse.ArgumentParser(description="Classify topics from events, assign markets by description, and allocate sample counts by popularity.")
    parser.add_argument("--data-dir", default="data/polymarket-prediction-markets")
    parser.add_argument("--target", type=int, default=500, help="Total market sample target for allocation recommendation")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    events_path = os.path.join(args.data_dir, "polymarket_events.csv")
    markets_path = os.path.join(args.data_dir, "polymarket_markets.csv")

    with open(events_path, newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    with open(markets_path, newline="", encoding="utf-8") as f:
        markets = list(csv.DictReader(f))

    event_topic_by_id: Dict[str, str] = {}
    event_count_by_topic: Counter = Counter()
    popularity_by_topic: Dict[str, float] = defaultdict(float)

    for event in events:
        event_id = event.get("id", "")
        topic = infer_event_topic(event)
        event_topic_by_id[event_id] = topic
        event_count_by_topic[topic] += 1
        popularity_by_topic[topic] += event_popularity_score(event)

    market_count_by_topic: Counter = Counter()
    market_topic_rows: List[Dict[str, str]] = []

    for market in markets:
        topic = infer_topic_from_text((market.get("description", "") or ""))
        market_count_by_topic[topic] += 1
        market_topic_rows.append(
            {
                "market_id": market.get("id", ""),
                "event_id": market.get("event_id", ""),
                "topic_from_market_description": topic,
                "event_topic": event_topic_by_id.get(market.get("event_id", ""), "Other"),
                "question": market.get("question", ""),
            }
        )

    all_topics = set(TOPIC_ORDER)
    all_topics.update(event_count_by_topic.keys())
    all_topics.update(market_count_by_topic.keys())

    popularity_complete = {topic: popularity_by_topic.get(topic, 0.0) for topic in all_topics}
    availability_complete = {topic: market_count_by_topic.get(topic, 0) for topic in all_topics}
    allocation = allocate_by_share(popularity_complete, availability_complete, args.target)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "topic_market_summary.csv")
    mapping_path = os.path.join(args.out_dir, "market_topic_mapping.csv")

    topics_sorted = sorted(all_topics, key=lambda topic: (market_count_by_topic.get(topic, 0), topic), reverse=True)
    total_markets = len(markets)
    total_popularity = sum(popularity_complete.values()) or 1.0

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topic",
                "event_count",
                "market_count_by_description",
                "market_share",
                "event_popularity_score",
                "event_popularity_share",
                "recommended_market_quota",
            ],
        )
        writer.writeheader()
        for topic in topics_sorted:
            writer.writerow(
                {
                    "topic": topic,
                    "event_count": event_count_by_topic.get(topic, 0),
                    "market_count_by_description": market_count_by_topic.get(topic, 0),
                    "market_share": f"{market_count_by_topic.get(topic, 0) / total_markets:.6f}",
                    "event_popularity_score": f"{popularity_complete.get(topic, 0.0):.6f}",
                    "event_popularity_share": f"{popularity_complete.get(topic, 0.0) / total_popularity:.6f}",
                    "recommended_market_quota": allocation.get(topic, 0),
                }
            )

    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["market_id", "event_id", "topic_from_market_description", "event_topic", "question"],
        )
        writer.writeheader()
        writer.writerows(market_topic_rows)

    print("summary_file=", summary_path)
    print("mapping_file=", mapping_path)
    print("top_topics_by_market_count=")
    for topic in topics_sorted[:10]:
        print(
            topic,
            "markets=", market_count_by_topic.get(topic, 0),
            "events=", event_count_by_topic.get(topic, 0),
            "quota=", allocation.get(topic, 0),
        )


if __name__ == "__main__":
    main()
