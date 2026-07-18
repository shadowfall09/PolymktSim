#!/usr/bin/env python3
"""Inject fixture metadata parsed from Polymarket URL slugs into Sports questions.

Motivation (2026-07 diagnosis): Sports is 42% of polymarket_250 and nearly
skill-free — agent rationales repeatedly say the evidence never identifies the
opponent. But the market URL slug encodes the fixture, e.g.

    .../market/mls-mia-nyc-2025-11-29-nyc
        league=mls, pairing=mia vs nyc, date, market subject=nyc

This script parses that slug, uses one cheap LLM call per match to decode the
opponent team code into a full name (parametric knowledge of team codes only —
no web access, so no outcome leakage beyond what the model already has), and
prepends a "market metadata" evidence item to the question's evidence list.
The input file is never modified.

Usage:
    python data/scripts/augment_sports_slug_metadata.py \
        data/processed/polymarket_250_with_evidence.jsonl \
        --out data/processed/polymarket_250_with_evidence_slugmeta.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

# slug shapes: {league}-{codeA}-{codeB}-{YYYY-MM-DD}-{subject}
# subject is a team code or the literal "draw"/"tie".
_SLUG_RE = re.compile(
    r"^(?P<league>[a-z0-9]{2,6})-(?P<a>[a-z0-9]{2,6})-(?P<b>[a-z0-9]{2,6})-"
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<subject>[a-z0-9]{2,6}|draw|tie)$"
)

_DECODE_PROMPT = """You are given metadata parsed from a Polymarket sports market URL slug.

League code: {league}
Team codes in the slug (in order): {a}, {b}
Scheduled date: {date}
Market question: {title}

The market question names one of the two teams. Using your knowledge of sports team abbreviations in this league, identify the OTHER team (the opponent). If you are not confident, say so.

Respond with exactly one JSON object, no other text:
{{"opponent_name": "<full team name or 'unknown'>", "league_name": "<full league/competition name or 'unknown'>", "confidence": "high"|"medium"|"low"}}"""


def parse_slug(url: str):
    slug = (url or "").rstrip("/").rsplit("/", 1)[-1]
    m = _SLUG_RE.match(slug)
    return (slug, m.groupdict()) if m else (slug, None)


def decode_opponent(client, model: str, info: dict, title: str) -> dict:
    prompt = _DECODE_PROMPT.format(league=info["league"], a=info["a"], b=info["b"],
                                   date=info["date"], title=title)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=200,
        )
        txt = r.choices[0].message.content.strip()
        txt = re.sub(r"^```(json)?|```$", "", txt, flags=re.M).strip()
        d = json.loads(txt)
        return {
            "opponent_name": str(d.get("opponent_name", "unknown")),
            "league_name": str(d.get("league_name", "unknown")),
            "confidence": str(d.get("confidence", "low")),
        }
    except Exception as e:
        print(f"  decode failed ({e}); keeping codes only", file=sys.stderr)
        return {"opponent_name": "unknown", "league_name": "unknown", "confidence": "low"}


def build_meta_item(rec: dict, slug: str, info: dict, decoded: dict) -> dict:
    title_line = rec["question"].split("\n")[0].strip()
    subject = info["subject"]
    is_draw = subject in ("draw", "tie")
    lines = [
        f"Market pairing metadata derived from the Polymarket market URL slug '{slug}' "
        f"(this metadata existed when the market was created; it contains no outcome information).",
        f"Fixture: team code '{info['a']}' vs team code '{info['b']}' on {info['date']}, "
        f"league code '{info['league']}'"
        + (f" ({decoded['league_name']})" if decoded["league_name"] != "unknown" else "") + ".",
    ]
    if is_draw:
        lines.append("This market resolves YES only if the game ends in a draw.")
    else:
        lines.append(f"The market subject ('{title_line}') corresponds to team code '{subject}'.")
    if decoded["opponent_name"] != "unknown":
        lines.append(
            f"Opponent decoded from the other team code: {decoded['opponent_name']} "
            f"(decoding confidence: {decoded['confidence']}; cross-check against other evidence)."
        )
    lines.append("Slug order does not reliably indicate home/away.")
    return {
        "doc_id": f"{rec['qid']}_slugmeta",
        "source": "polymarket_market_slug",
        "title": f"Fixture metadata: {title_line}",
        "url": rec.get("url"),
        "content": " ".join(lines),
        "retrieval_score": 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="Sports", help="Only augment this topic (default Sports)")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM opponent decoding (codes only)")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input) if l.strip()]

    client = model = None
    if not args.no_llm:
        from src.agents.llm_agent import LLMAgent
        agent = LLMAgent(temperature=0.0, max_tokens=200)
        client, model = agent._client, agent.model_name

    targets = []
    for rec in records:
        if rec.get("topic") != args.topic:
            continue
        slug, info = parse_slug(rec.get("url", ""))
        if info:
            targets.append((rec, slug, info))
    print(f"{len(records)} records; {len(targets)} {args.topic} records with parseable match slug")

    def _one(t):
        rec, slug, info = t
        decoded = ({"opponent_name": "unknown", "league_name": "unknown", "confidence": "low"}
                   if args.no_llm else
                   decode_opponent(client, model, info, rec["question"].split("\n")[0]))
        return rec["qid"], build_meta_item(rec, slug, info, decoded), decoded

    meta_by_qid = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for qid, item, decoded in ex.map(_one, targets):
            meta_by_qid[qid] = item
            print(f"  {qid}: opponent={decoded['opponent_name']!r} ({decoded['confidence']})")

    n_aug = 0
    with open(args.out, "w") as f:
        for rec in records:
            if rec["qid"] in meta_by_qid:
                rec = dict(rec)
                rec["evidence"] = [meta_by_qid[rec["qid"]]] + list(rec.get("evidence", []))
                n_aug += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}: {n_aug} records augmented")


if __name__ == "__main__":
    main()
