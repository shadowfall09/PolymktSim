#!/usr/bin/env python3
"""Add point-in-time, pre-cutoff evidence to an already processed JSONL file.

This is intentionally stricter than the broad evidence collector. For each
row it generates exactly five queries tied to five discrete safe calendar days
before the cutoff (by default 1, 3, 7, 10, and 15 days earlier), rather than
a month or other time range. Search results are candidates only: results
containing later material are rewritten into a standalone, URL-free
pre-cutoff fact when possible, independently verified, and otherwise
discarded.

The script never modifies the input.  New evidence is placed before retained
old evidence in the output so it is not hidden by a downstream top-N limit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from collect_evidences_tavily import (
    SearchItem,
    TavilySearchError,
    clamp_query,
    compact_text,
    evidence_cutoff_date,
    extract_explicit_dates,
    llm_post_filter_leaks,
    normalize_end_date,
    normalize_ws,
    quarantine_late_dates_for_rewrite,
    slugify,
    tavily_search,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower() or "web"
    except ValueError:
        return "web"


def parse_query_response(raw: str) -> list[str]:
    """Accept only a complete JSON object with exactly five usable queries."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("queries"), list):
        raise ValueError("point-query response is not {queries: [...]}")
    queries: list[str] = []
    for value in parsed["queries"]:
        query = clamp_query(str(value), 400)
        if query and query not in queries:
            queries.append(query)
    if len(queries) != 5:
        raise ValueError(f"expected exactly five point queries, got {len(queries)}")
    return queries


def build_point_queries(
    question: str,
    cutoff: str,
    anchor_offsets: list[int],
    client: Any,
    model: str,
) -> list[tuple[str, str, int]]:
    """Generate one exact-date query for each requested pre-cutoff offset."""
    cutoff_day = date.fromisoformat(cutoff)
    anchors = [((cutoff_day - timedelta(days=offset)).isoformat(), offset) for offset in anchor_offsets]
    schedule = [
        {"query_index": index, "anchor_date": anchor, "days_before_cutoff": offset}
        for index, (anchor, offset) in enumerate(anchors, start=1)
    ]
    prompt = (
        "Return JSON only: {\"queries\":[\"...\",\"...\",\"...\",\"...\",\"...\"]}.\n"
        "Generate exactly five concise, diverse search queries for a forecasting dataset. Query i MUST use "
        "the corresponding schedule item i below: it MUST contain that item's exact anchor date and seek a "
        "fact, data point, filing, quote, odds snapshot, "
        "schedule, official statement, or contemporaneous report specifically ON that date.\n"
        "This is point-in-time retrieval: do NOT request a date range, month, week, 'recent', 'latest', "
        "'today', a recap, an outcome, a final result, or whether the market resolved. Do not ask for facts "
        "after the anchor date. The five queries must use distinct useful lenses: official/source data, "
        "a directly measured indicator, a named-entity update, a market/odds/constraint signal when relevant, "
        "and independent contemporaneous reporting. Prefer a source that can expose a dated value over generic "
        "background. A query must not be a paraphrase of the forecasting question.\n"
        f"Forecasting question: {question}\n"
        f"Hard evidence cutoff: {cutoff}\n"
        "Discrete query schedule (do not merge these into a range): "
        f"{json.dumps(schedule, ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        raw = "\n".join(str(part.get("text", "")) for part in raw if isinstance(part, dict))
    queries = parse_query_response(str(raw))
    output: list[tuple[str, str, int]] = []
    for query, (anchor, offset) in zip(queries, anchors, strict=True):
        if anchor not in query:
            raise ValueError(f"point-query response omitted required anchor {anchor}")
        output.append((query, anchor, offset))
    return output


def search_queries(
    queries: list[tuple[str, str, int]], cutoff: str, client: Any, sleep_seconds: float, row_index: int
) -> tuple[list[SearchItem], list[dict[str, Any]]]:
    items: list[SearchItem] = []
    metadata: list[dict[str, Any]] = []
    for search_index, (query, anchor, offset) in enumerate(queries, start=1):
        # end_date is only candidate recall control; it is not treated as a
        # proof of a result's publication time.
        kwargs = {"query": query, "search_depth": "advanced", "max_results": 10, "end_date": cutoff}
        print(
            f"[{row_index}] point query {search_index}/5 anchor={anchor} "
            f"({offset}d before cutoff={cutoff}): {query}"
        )
        try:
            response = tavily_search(client, kwargs)
        except TavilySearchError as exc:
            print(f"[{row_index}] point query {search_index}/5 failed: {exc}")
            metadata.append({
                "query": query, "anchor_date": anchor, "days_before_cutoff": offset,
                "request": kwargs, "error": str(exc), "results_count": 0,
            })
            continue
        results = response.get("results", []) or []
        metadata.append({
            "query": query, "anchor_date": anchor, "days_before_cutoff": offset,
            "request": kwargs, "results_count": len(results),
        })
        for result in results:
            items.append(SearchItem(
                title=normalize_ws(str(result.get("title") or "")),
                url=normalize_ws(str(result.get("url") or "")),
                content=compact_text(str(result.get("content") or ""), 1200),
                score=float(result["score"]) if result.get("score") is not None else None,
                query=query,
                search_index=search_index,
            ))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return items, metadata


def dedupe(items: list[SearchItem]) -> list[SearchItem]:
    best: dict[str, SearchItem] = {}
    for item in items:
        key = item.dedupe_key()
        old = best.get(key)
        if old is None or (item.score or -1.0) > (old.score or -1.0):
            best[key] = item
    return sorted(best.values(), key=lambda x: x.score if x.score is not None else -1.0, reverse=True)


def point_safe_after_verification(item: SearchItem, cutoff: str) -> bool:
    """Deterministic final gate for this point-data-only collector.

    This collector does not keep prospective plans, so any later full date in
    a retained rewrite is unnecessary and rejected.  The LLM verification in
    ``llm_post_filter_leaks`` handles natural-language hindsight; this gate
    prevents a verifier mistake from retaining an explicit later date.
    """
    cutoff_day = date.fromisoformat(cutoff)
    text = " ".join((item.title, item.content))
    if any(found > cutoff_day for found in extract_explicit_dates(text)):
        return False
    forbidden = ("current price", "latest", "today", "at close", "after hours", "update time")
    lower = text.lower()
    return not any(token in lower for token in forbidden)


def evidence_record(qid: str, index: int, item: SearchItem, anchor: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "doc_id": f"{qid}_point_doc_{index:03d}",
        "source": source_from_url(item.url),
        "title": item.title,
        "url": item.url or None,
        "content": item.content,
        "retrieval_score": item.score,
        "point_anchor_date": anchor,
        "point_in_time_augmentation": True,
    }
    if item.rewritten_for_leakage:
        record["rewritten_for_leakage"] = True
        record["url"] = None
        record["source"] = "sanitized_search_snippet"
    return record


def old_key(evidence: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_ws(str(evidence.get("url") or "")).lower(),
        normalize_ws(str(evidence.get("title") or "")).lower(),
        compact_text(str(evidence.get("content") or ""), 220).lower(),
    )


def cache_file(cache_dir: Path, index: int, qid: str) -> Path:
    return cache_dir / f"row_{index:04d}_{slugify(qid, max_len=120)}_point_augmentation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--api-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("CMU_API_KEY", ""))
    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", "https://ai-gateway.andrew.cmu.edu"))
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--cutoff-days", type=int, default=1)
    parser.add_argument(
        "--anchor-offset-days",
        default="1,3,7,10,15",
        help="Exactly five distinct positive day offsets before the cutoff, in query order (default: 1,3,7,10,15).",
    )
    parser.add_argument("--max-new-evidence", type=int, default=10)
    parser.add_argument("--max-evidence", type=int, default=20)
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if not args.api_key or not args.llm_api_key:
        raise ValueError("TAVILY_API_KEY and CMU_API_KEY (or corresponding flags) are required.")
    if args.cutoff_days < 1 or args.max_new_evidence < 1 or args.max_evidence < 1:
        raise ValueError("cutoff-days, max-new-evidence, and max-evidence must be positive")
    if args.start_row < 1 or args.limit < 0 or args.workers < 1:
        raise ValueError("start-row/workers must be >= 1 and limit must be >= 0")
    try:
        anchor_offsets = [int(value.strip()) for value in args.anchor_offset_days.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("anchor-offset-days must be comma-separated integers") from exc
    if len(anchor_offsets) != 5 or len(set(anchor_offsets)) != 5 or any(value < 1 for value in anchor_offsets):
        raise ValueError("anchor-offset-days must contain exactly five distinct positive integers")

    from tavily import TavilyClient
    openai = importlib.import_module("openai")
    rows = read_jsonl(args.input)
    start = args.start_row - 1
    end = len(rows) if args.limit == 0 else min(len(rows), start + args.limit)
    selected = rows[start:end]
    print(f"processing rows {start + 1}..{end} of {len(rows)}")

    def process(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
        qid = normalize_ws(str(row.get("qid") or ""))
        question = normalize_ws(str(row.get("question") or ""))
        resolution = normalize_end_date(str(row.get("resolution_date") or ""))
        cutoff = evidence_cutoff_date(resolution, args.cutoff_days)
        if not qid or not question or not cutoff:
            raise ValueError(f"row {index} requires qid, question, and valid resolution_date")
        tavily = TavilyClient(args.api_key)
        llm = openai.OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
        try:
            queries = build_point_queries(question, cutoff, anchor_offsets, llm, args.llm_model)
        except Exception as exc:
            queries = []
            query_error = str(exc)
            print(f"[{index}] point query generation failed; no new evidence: {exc}")
        else:
            query_error = ""

        candidates, request_meta = search_queries(queries, cutoff, tavily, args.sleep_seconds, index)
        candidates = dedupe(candidates)
        quarantined = quarantine_late_dates_for_rewrite(candidates, cutoff)
        safe, filter_stats, leakage_audit = llm_post_filter_leaks(
            candidates, question, resolution, cutoff, llm, args.llm_model
        ) if candidates else ([], {"llm_filtered_leak_count": 0, "llm_rewritten_kept_count": 0}, [])
        final_new = [item for item in safe if point_safe_after_verification(item, cutoff)]
        hard_rejected = len(safe) - len(final_new)
        final_new = final_new[:min(args.max_new_evidence, args.max_evidence)]

        anchors_by_query = {query: anchor for query, anchor, _ in queries}
        new_records = [
            evidence_record(qid, i, item, anchors_by_query.get(item.query, cutoff))
            for i, item in enumerate(final_new, start=1)
        ]
        prior = [item for item in row.get("evidence", []) if isinstance(item, dict)]
        seen = {old_key(item) for item in new_records}
        merged = list(new_records)
        for item in prior:
            key = old_key(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)
            if len(merged) == args.max_evidence:
                break
        enriched = dict(row)
        enriched["evidence"] = merged
        enriched["point_augmentation"] = {
            "cutoff_date": cutoff,
            "anchor_offsets_days": anchor_offsets,
            "queries": [
                {"query": query, "anchor_date": anchor, "days_before_cutoff": offset}
                for query, anchor, offset in queries
            ],
            "new_evidence_kept": len(new_records),
            "safe_after_final_gate": len(final_new),
            "hard_rejected_after_rewrite_verification": hard_rejected,
        }
        payload = {
            "row_index": index,
            "qid": qid,
            "question": question,
            "resolution_date": resolution,
            "evidence_cutoff_date": cutoff,
            "anchor_offsets_days": anchor_offsets,
            "queries": [
                {"query": query, "anchor_date": anchor, "days_before_cutoff": offset}
                for query, anchor, offset in queries
            ],
            "query_generation_error": query_error,
            "search_requests": request_meta,
            "leakage_audit": leakage_audit,
            "stats": {
                "raw_candidate_count": len(candidates),
                "hard_quarantined_late_date_count": quarantined,
                "safe_before_final_gate": len(safe),
                "hard_rejected_after_rewrite_verification": hard_rejected,
                "final_new_evidence_count": len(new_records),
                **filter_stats,
            },
            "evidences": new_records,
        }
        path = cache_file(args.cache_dir, index, qid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return index, enriched, (
            f"[{index}] anchor={cutoff} candidates={len(candidates)} "
            f"new_safe={len(new_records)} total_evidence={len(merged)}"
        )

    indexed = list(enumerate(selected, start=start + 1))
    completed: dict[int, dict[str, Any]] = {}
    if args.workers == 1:
        for index, row in indexed:
            result_index, enriched, status = process(index, row)
            completed[result_index] = enriched
            print(status)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, index, row) for index, row in indexed]
            for future in as_completed(futures):
                result_index, enriched, status = future.result()
                completed[result_index] = enriched
                print(status)
    write_jsonl(args.output, [completed[index] for index, _ in indexed])


if __name__ == "__main__":
    main()
