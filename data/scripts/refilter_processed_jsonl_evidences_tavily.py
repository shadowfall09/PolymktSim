#!/usr/bin/env python3
"""Strictly re-filter cached evidence and fill gaps with time-anchored Tavily searches.

Unlike the initial collector, this tool never rewrites a potentially leaky
snippet: a direct answer, hindsight, or concrete post-cutoff fact is always
dropped.  Unknown publication time alone is allowed, provided the supplied
title, URL, and snippet do not reveal the answer or a later state.
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
    compact_text,
    evidence_cutoff_date,
    normalize_end_date,
    normalize_ws,
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


def cache_path(cache_dir: Path, row_index: int, qid: str) -> Path:
    return cache_dir / f"row_{row_index:04d}_{slugify(qid, max_len=120)}.json"


def source_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower() or "web"
    except ValueError:
        return "web"


def items_from_cache(payload: dict[str, Any]) -> list[SearchItem]:
    items = []
    for index, evidence in enumerate(payload.get("evidences") or [], start=1):
        if not isinstance(evidence, dict):
            continue
        items.append(SearchItem(
            title=normalize_ws(str(evidence.get("title") or "")),
            url=normalize_ws(str(evidence.get("url") or "")),
            content=compact_text(str(evidence.get("content") or ""), 1200),
            score=float(evidence["score"]) if evidence.get("score") is not None else None,
            query=normalize_ws(str(evidence.get("query") or "cached evidence")),
            search_index=int(evidence.get("search_index") or index),
        ))
    return items


def dedupe(items: list[SearchItem]) -> list[SearchItem]:
    seen: set[str] = set()
    output = []
    for item in items:
        key = item.dedupe_key()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def strict_filter_prompt(question: str, cutoff: str, items: list[SearchItem]) -> str:
    rows = [
        {"index": i, "title": item.title, "url": item.url, "snippet": compact_text(item.content, 900)}
        for i, item in enumerate(items)
    ]
    return (
        "You are the final, fail-closed anti-leakage gate for a forecasting dataset. "
        "Return JSON only: {\"decisions\":[{\"index\":0,\"keep\":false,\"relevance\":\"low\",\"reason\":\"...\"}]}.\n"
        "The forecaster is at the historical cutoff shown below. Keep useful background and contemporaneous "
        "context, including genuinely pre-cutoff prospective evidence: announcements, schedules, planned "
        "airings, filings, odds, forecasts, intentions, previews, and statements that an event WILL happen. "
        "Such a plan is valid even if its planned event date is after the cutoff.\n"
        "However, drop an item whenever its title, URL, or snippet explicitly says it was published, released, "
        "updated, posted, broadcast, streamed, or reported AFTER the cutoff, or gives a concrete post-cutoff "
        "page state. Also drop past-tense or completed-result language: 'aired', 'was broadcast', 'scaled', "
        "'won', 'lost', 'ended', 'withdrew', 'was revealed', 'has happened', final score, final result, or "
        "current status. This is true even if the item also contains useful background.\n"
        "Distinguish carefully: 'will air on January 18' in an earlier-dated preview can be kept; a page marked "
        "'Release 01/18/2026' or saying it 'airs 7 PM' when the cutoff is 01/17/2026 must be dropped. A future "
        "date alone is not leakage, but a post-cutoff publication/release date is. Publication time may be "
        "unknown; unknown time alone is not a reason to drop, but never use it to excuse an explicit completed "
        "outcome. Treat title and URL as evidence too. When uncertain whether an item is prospective or "
        "retrospective, drop it.\n"
        "Also rate relevance to forecasting this exact question: high = directly informs likelihood, timing, "
        "constraints, primary data, or a leading indicator; medium = useful supporting context; low = generic "
        "background, a tangential entity mention, or material that would not help a forecast. Low relevance "
        "is not leakage: retain it if safe, but rate it low so better supplemental evidence can be sought.\n"
        f"Question: {question}\nEvidence cutoff: {cutoff}\nItems: {json.dumps(rows, ensure_ascii=False)}"
    )


def strict_filter(
    items: list[SearchItem], question: str, cutoff: str, client: Any, model: str, stage: str
) -> tuple[list[SearchItem], int, dict[str, int], list[dict[str, Any]]]:
    kept: list[SearchItem] = []
    dropped = 0
    relevance_counts = {"high": 0, "medium": 0, "low": 0}
    audit: list[dict[str, Any]] = []
    for start in range(0, len(items), 10):
        batch = items[start : start + 10]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": strict_filter_prompt(question, cutoff, batch)}],
                temperature=0,
            )
            raw = response.choices[0].message.content
            if isinstance(raw, list):
                raw = "\n".join(str(part.get("text", "")) for part in raw if isinstance(part, dict))
            parsed = json.loads(str(raw))
            decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else []
            by_index = {
                int(d["index"]): (
                    bool(d.get("keep")),
                    normalize_ws(str(d.get("relevance") or "low")).lower(),
                    normalize_ws(str(d.get("reason") or "no reason supplied")),
                ) for d in decisions
                if isinstance(d, dict) and isinstance(d.get("index"), int)
            }
            if (
                set(by_index) != set(range(len(batch)))
                or any(value[1] not in relevance_counts for value in by_index.values())
            ):
                raise ValueError("incomplete decisions")
        except Exception as exc:
            print(f"strict filter failed; dropped {len(batch)} items: {exc}")
            dropped += len(batch)
            for item in batch:
                audit.append({
                    "stage": stage, "title": item.title, "url": item.url, "query": item.query,
                    "llm_keep": None, "relevance": None, "disposition": "dropped",
                    "reason": f"filter_error: {exc}",
                })
            continue
        for i, item in enumerate(batch):
            keep, relevance, reason = by_index[i]
            relevance_counts[relevance] += 1
            if keep:
                kept.append(item)
                disposition = "kept"
            else:
                dropped += 1
                disposition = "dropped"
            audit.append({
                "stage": stage, "title": item.title, "url": item.url, "query": item.query,
                "llm_keep": keep, "relevance": relevance, "disposition": disposition,
                "reason": reason,
            })
    return kept, dropped, relevance_counts, audit


def parse_anchored_query_response(raw: str) -> tuple[dict[str, Any], bool]:
    """Parse an anchored-query response, repairing only a missing final array close.

    Some responses contain a complete ``queries`` array but end in ``..."}``
    rather than ``..."]}``.  Do not try to generically repair model output:
    that could silently change a search query.  This narrowly handles the
    observed, unambiguous missing-``]`` case and still relies on json.loads to
    validate the repaired value.
    """
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("anchored query response is not a JSON object")
        return parsed, False
    except json.JSONDecodeError as original_error:
        candidate = raw.rstrip()
        if not candidate.endswith("}"):
            raise original_error
        try:
            repaired = json.loads(f"{candidate[:-1]}]}}")
        except json.JSONDecodeError:
            raise original_error
        if not isinstance(repaired, dict) or not isinstance(repaired.get("queries"), list):
            raise original_error
        return repaired, True


def anchored_queries(
    question: str,
    cutoff: str,
    anchor: str,
    client: Any,
    model: str,
    failure_path: Path | None = None,
) -> list[str]:
    prompt = (
        "Return JSON only: {\"queries\":[\"...\",\"...\",\"...\"]}. Generate exactly three diverse web "
        "queries for evidence available before a forecasting cutoff. Each query must explicitly include the "
        "historical anchor date and seek information available on or before that date. Queries MAY directly "
        "seek pre-cutoff announcements, schedules, plans, filings, odds, previews, or contemporary reporting "
        "about the predicted event. They must NOT seek a later result, resolution, current status, recap, "
        "outcome, winner, confirmation that the event happened, or a post-event update. Cover distinct lenses: "
        "(1) primary source/rules or data, (2) recent relevant "
        "developments and leading indicators near the anchor, (3) independent domain analysis or historical "
        "base rate. Prefer concrete entities and a recent time window over generic background.\n"
        f"Question: {question}\nHard evidence cutoff: {cutoff}\nSearch anchor (roughly two weeks earlier): {anchor}"
    )
    raw_text = ""
    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        raw = response.choices[0].message.content
        if isinstance(raw, list):
            raw = "\n".join(str(part.get("text", "")) for part in raw if isinstance(part, dict))
        raw_text = str(raw)
        parsed, repaired = parse_anchored_query_response(raw_text)
        if repaired:
            print("anchored query response was missing a final ']' and was repaired")
        queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
    except Exception as exc:
        if failure_path is not None:
            try:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(json.dumps({
                    "error": str(exc),
                    "question": question,
                    "cutoff": cutoff,
                    "anchor": anchor,
                    "raw": raw_text,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as save_exc:
                print(f"anchored query failure could not be saved: {save_exc}")
            else:
                print(f"anchored query generation failed: {exc}; raw saved to {failure_path}")
                return []
        print(f"anchored query generation failed: {exc}")
        return []
    clean = []
    for query in queries:
        query = normalize_ws(str(query))
        if query and anchor in query and len(query) <= 400 and query not in clean:
            clean.append(query)
        if len(clean) == 3:
            break
    return clean


def search_supplemental(queries: list[str], cutoff: str, client: Any, sleep_seconds: float) -> list[SearchItem]:
    found = []
    for search_index, query in enumerate(queries, start=1):
        response = tavily_search(client, {"query": query, "search_depth": "advanced", "max_results": 10, "end_date": cutoff})
        for result in response.get("results", []) or []:
            found.append(SearchItem(
                title=normalize_ws(str(result.get("title") or "")), url=normalize_ws(str(result.get("url") or "")),
                content=compact_text(str(result.get("content") or ""), 1200),
                score=float(result["score"]) if result.get("score") is not None else None,
                query=query, search_index=search_index,
            ))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return found


def as_evidence(qid: str, items: list[SearchItem], limit: int) -> list[dict[str, Any]]:
    return [{"doc_id": f"{qid}_doc_{i:03d}", "source": source_from_url(item.url), "title": item.title,
             "url": item.url or None, "content": item.content, "retrieval_score": item.score}
            for i, item in enumerate(items[:limit], start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--start-row", type=int, default=1, help="1-based first row to process (default: 1).")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Number of rows to process; 0 means all. Limited runs write only the selected rows.",
    )
    parser.add_argument("--api-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--llm-api-key", default=os.getenv("CMU_API_KEY", ""))
    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", "https://ai-gateway.andrew.cmu.edu"))
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--cutoff-days", type=int, default=1)
    parser.add_argument("--anchor-days-before-cutoff", type=int, default=16)
    parser.add_argument("--minimum-evidence", type=int, default=15)
    parser.add_argument(
        "--minimum-high-relevance",
        type=int,
        default=4,
        help="Supplement when fewer than this many retained items are highly relevant (default: 4).",
    )
    parser.add_argument("--max-evidence", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Rows to process concurrently; 1 keeps serial processing (default: 1).",
    )
    parser.add_argument("--write-cache", action="store_true", help="Replace cache payloads with refiltered evidence.")
    args = parser.parse_args()
    if not args.api_key or not args.llm_api_key:
        raise ValueError("TAVILY_API_KEY and CMU_API_KEY (or corresponding flags) are required.")
    if args.cutoff_days < 1 or args.anchor_days_before_cutoff < 1:
        raise ValueError("cutoff and anchor offsets must be at least 1 day")
    if args.start_row < 1 or args.limit < 0 or args.workers < 1:
        raise ValueError("--start-row and --workers must be at least 1, and --limit cannot be negative")
    from tavily import TavilyClient
    openai = importlib.import_module("openai")
    input_rows = read_jsonl(args.input)
    start_index = args.start_row - 1
    end_index = len(input_rows) if args.limit == 0 else min(len(input_rows), start_index + args.limit)
    selected_rows = input_rows[start_index:end_index]
    print(f"processing rows {start_index + 1}..{end_index} of {len(input_rows)}")
    def process_row(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
        # Use separate clients per worker: the SDKs' HTTP/session state is not
        # assumed to be thread-safe, while each task writes a distinct cache file.
        tavily = TavilyClient(args.api_key)
        llm = openai.OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
        qid, question = str(row.get("qid") or ""), normalize_ws(str(row.get("question") or ""))
        resolution = normalize_end_date(str(row.get("resolution_date") or ""))
        cutoff = evidence_cutoff_date(resolution, args.cutoff_days)
        if not qid or not question or not cutoff:
            raise ValueError(f"row {index} requires qid, question, and a valid resolution_date")
        payload_path = cache_path(args.cache_dir, index, qid)
        payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.exists() else {}
        retained, dropped, relevance, audit = strict_filter(
            dedupe(items_from_cache(payload)), question, cutoff, llm, args.llm_model, stage="cached"
        )
        # A previous --write-cache may already have replaced a too-strictly
        # filtered set. Re-run the original broad-but-pre-cutoff queries so
        # relaxing the policy can actually recover useful candidates.
        recovered_queries = [
            normalize_ws(str(query)) for query in payload.get("expanded_queries", [])
            if normalize_ws(str(query))
        ][:5]
        if len(retained) < args.minimum_evidence and recovered_queries:
            recovered = search_supplemental(recovered_queries, cutoff, tavily, args.sleep_seconds)
            safe_recovered, recovered_dropped, recovered_relevance, recovered_audit = strict_filter(
                dedupe(recovered), question, cutoff, llm, args.llm_model, stage="recovered_search"
            )
            retained = dedupe(retained + safe_recovered)
            dropped += recovered_dropped
            audit.extend(recovered_audit)
            for key, value in recovered_relevance.items():
                relevance[key] += value
        queries: list[str] = []
        needs_supplemental = (
            len(retained) < args.minimum_evidence
            or relevance["high"] < args.minimum_high_relevance
        )
        if needs_supplemental:
            anchor = (date.fromisoformat(cutoff) - timedelta(days=args.anchor_days_before_cutoff)).isoformat()
            failure_path = args.cache_dir / "llm_failures" / f"row_{index:04d}_{slugify(qid, max_len=120)}_anchored_queries.json"
            queries = anchored_queries(question, cutoff, anchor, llm, args.llm_model, failure_path)
            supplemental = search_supplemental(queries, cutoff, tavily, args.sleep_seconds)
            safe_supplemental, supplemental_dropped, supplemental_relevance, supplemental_audit = strict_filter(
                dedupe(supplemental), question, cutoff, llm, args.llm_model, stage="supplemental_search"
            )
            retained = dedupe(retained + safe_supplemental)
            dropped += supplemental_dropped
            audit.extend(supplemental_audit)
            for key, value in supplemental_relevance.items():
                relevance[key] += value
        retained.sort(key=lambda item: item.score if item.score is not None else -1, reverse=True)
        enriched = dict(row)
        enriched["evidence"] = as_evidence(qid, retained, args.max_evidence)
        if args.write_cache:
            payload.update({"question": question, "resolution_date": resolution, "evidence_cutoff_date": cutoff,
                            "refiltered": True, "evidences": [item.to_dict() for item in retained],
                            "recovered_queries": recovered_queries, "refilter_queries": queries,
                            "refilter_audit": audit,
                            "refilter_stats": {"kept": len(retained), "dropped": dropped,
                            "relevance": relevance, "supplemental_for_relevance": needs_supplemental}})
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        status = (
            f"[{index}] kept={len(retained)} high_relevance={relevance['high']} "
            f"dropped={dropped} supplemental_queries={len(queries)}"
        )
        return index, enriched, status

    indexed_rows = list(enumerate(selected_rows, start=start_index + 1))
    completed: dict[int, dict[str, Any]] = {}
    if args.workers == 1:
        for index, row in indexed_rows:
            result_index, enriched, status = process_row(index, row)
            completed[result_index] = enriched
            print(status)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, index, row) for index, row in indexed_rows]
            for future in as_completed(futures):
                result_index, enriched, status = future.result()
                completed[result_index] = enriched
                print(status)
    output_rows = [completed[index] for index, _ in indexed_rows]
    write_jsonl(args.output, output_rows)


if __name__ == "__main__":
    main()
