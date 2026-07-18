#!/usr/bin/env python3
"""Targeted pre-cutoff Tavily retrieval for sports match questions, using the
opponent name decoded from the market URL slug.

Follow-up to augment_sports_slug_metadata.py: injecting the opponent's
identity alone measurably moved predictions but not accuracy — the missing
signal is quantitative (odds, form, head-to-head). This script fetches exactly
that, with the same no-leak discipline as the main collector:

  1. Tavily `end_date` restricted to the evidence cutoff (day before resolution).
  2. quarantine_late_dates_for_rewrite: any item whose text mentions a
     post-cutoff date is marked; marked items are DROPPED here (no rewrite
     salvage — strictest policy).
  3. llm_post_filter_leaks: every remaining item must be explicitly judged
     non-leaky by the LLM; rewrite candidates are dropped too.

Input must be the slugmeta-augmented dataset (evidence[0].source ==
'polymarket_market_slug' carries the decoded opponent). The input file is
never modified; new docs are inserted right after the slugmeta doc.

Usage:
    python data/scripts/augment_sports_opponent_evidence.py \
        data/processed/polymarket_250_with_evidence_slugmeta.jsonl \
        --out data/processed/polymarket_250_with_evidence_slugodds.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from collect_evidences_tavily import (  # noqa: E402
    SearchItem,
    clamp_query,
    compact_text,
    evidence_cutoff_date,
    llm_post_filter_leaks,
    normalize_ws,
    quarantine_late_dates_for_rewrite,
    tavily_search,
)

_OPP_RE = re.compile(r"Opponent decoded from the other team code: (.+?) \(decoding confidence: (high|medium|low)")
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def month_year(iso: str) -> str:
    try:
        d = date.fromisoformat(iso[:10])
        return f"{_MONTHS[d.month - 1]} {d.year}"
    except (ValueError, IndexError):
        return iso[:7]


def build_queries(subject: str, opponent: str, match_date: str) -> list[str]:
    my = month_year(match_date)
    return [
        f"{subject} vs {opponent} {my} match preview betting odds prediction",
        f"{subject} vs {opponent} head to head record recent meetings",
        f"{subject} recent form results last 5 matches {my}",
        f"{opponent} recent form results last 5 matches {my}",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="slugmeta-augmented dataset JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-results", type=int, default=6, help="Tavily results per query (default 6)")
    ap.add_argument("--keep-top", type=int, default=8, help="Max new docs kept per question (default 8)")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="Only augment first N eligible questions (for testing)")
    args = ap.parse_args()

    import os
    from tavily import TavilyClient
    from src.agents.llm_agent import LLMAgent

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        raise SystemExit("TAVILY_API_KEY not set")
    client = TavilyClient(api_key=tavily_key)
    agent = LLMAgent(temperature=0.0)
    llm_client, llm_model = agent._client, agent.model_name

    records = [json.loads(l) for l in open(args.input) if l.strip()]

    targets = []
    for rec in records:
        evs = rec.get("evidence") or []
        if not evs or evs[0].get("source") != "polymarket_market_slug":
            continue
        m = _OPP_RE.search(evs[0].get("content", ""))
        if not m:
            continue
        opponent, conf = m.group(1), m.group(2)
        if conf == "low":
            continue
        subject = rec["question"].split("\n")[0].strip().rstrip("?")
        subject = re.sub(r"^Will\s+", "", subject)
        subject = re.sub(r"\s+(win on|win|vs\.?|end in a draw).*$", "", subject, flags=re.I).strip()
        resolution_date = str(rec.get("resolution_date", ""))[:10]
        if not subject or not resolution_date:
            continue
        targets.append((rec["qid"], subject, opponent, resolution_date, rec["question"].split("\n")[0].strip()))
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(records)} records; {len(targets)} match questions eligible for opponent retrieval")

    def _one(t):
        qid, subject, opponent, resolution_date, question_title = t
        cutoff = evidence_cutoff_date(resolution_date)
        items: list[SearchItem] = []
        for i, q in enumerate(build_queries(subject, opponent, resolution_date), start=1):
            sq = clamp_query(q, 400)
            kwargs = {"query": sq, "search_depth": "advanced", "max_results": args.max_results}
            if cutoff:
                kwargs["end_date"] = cutoff
            try:
                resp = tavily_search(client, kwargs)
            except Exception as e:
                print(f"  {qid} query#{i} failed: {e}", file=sys.stderr)
                continue
            for r in resp.get("results", []) or []:
                items.append(SearchItem(
                    title=normalize_ws(str(r.get("title", ""))),
                    url=normalize_ws(str(r.get("url", ""))),
                    content=compact_text(str(r.get("content", "")), 1200),
                    score=float(r.get("score")) if r.get("score") is not None else None,
                    query=sq,
                    search_index=i,
                ))
        seen, deduped = set(), []
        for it in items:
            k = it.dedupe_key()
            if k not in seen:
                seen.add(k)
                deduped.append(it)
        quarantine_late_dates_for_rewrite(deduped, cutoff)
        deduped = [it for it in deduped if not it.requires_temporal_rewrite]
        if deduped:
            kept, stats, _audit = llm_post_filter_leaks(
                deduped, question_title, resolution_date, cutoff, llm_client, llm_model)
        else:
            kept, stats = [], {}
        kept.sort(key=lambda it: -(it.score or 0.0))
        kept = kept[: args.keep_top]
        docs = [{
            "doc_id": f"{qid}_opp_{j:02d}",
            "source": f"tavily_opponent_search",
            "title": it.title,
            "url": it.url,
            "content": it.content,
            "retrieval_score": it.score,
        } for j, it in enumerate(kept, start=1)]
        print(f"  {qid} ({subject} vs {opponent}): fetched={len(items)} kept={len(docs)}")
        return qid, docs

    new_docs = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for qid, docs in ex.map(_one, targets):
            new_docs[qid] = docs

    n_aug = 0
    with open(args.out, "w") as f:
        for rec in records:
            docs = new_docs.get(rec["qid"])
            if docs:
                rec = dict(rec)
                evs = list(rec["evidence"])
                rec["evidence"] = evs[:1] + docs + evs[1:]
                n_aug += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total = sum(len(d) for d in new_docs.values())
    print(f"Wrote {args.out}: {n_aug} records augmented with {total} leak-filtered docs")


if __name__ == "__main__":
    main()
