#!/usr/bin/env python3
"""Collect Tavily evidence for a processed QuestionExample JSONL dataset.

This is useful for datasets such as FutureX that are already converted to the
PolymktSim JSONL schema:

  {"qid": "...", "question": "...", "evidence": [], "outcome": true}

The script writes one cache JSON per question and a new JSONL with evidence
embedded, so it can be passed directly to run_workflow.py --dataset-jsonl.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from collect_evidences_tavily import TavilySearchError, collect_for_row, normalize_ws, slugify


def _source_from_url(url: str) -> str:
    if not url:
        return "web"
    try:
        return urlparse(url).netloc.lower() or "web"
    except ValueError:
        return "web"


def _cache_filename(row_index: int, qid: str) -> str:
    return f"row_{row_index:04d}_{slugify(qid, max_len=120)}.json"


def _row_for_collection(item: dict[str, Any]) -> dict[str, str]:
    qid = str(item.get("qid") or "").strip()
    question = str(item.get("question") or "").strip()
    resolution_date = str(item.get("resolution_date") or "").strip()
    return {
        "id": qid,
        "question": question,
        "description": question,
        "event_title": question,
        "endDate": resolution_date,
        "endDateIso": resolution_date,
    }


def _convert_evidence(qid: str, payload: dict[str, Any], max_evidence: int) -> list[dict[str, Any]]:
    evidences = payload.get("evidences") or []
    if max_evidence > 0:
        evidences = evidences[:max_evidence]

    converted = []
    for i, ev in enumerate(evidences, start=1):
        if not isinstance(ev, dict):
            continue
        url = str(ev.get("url") or "")
        converted.append(
            {
                "doc_id": f"{qid}_doc_{i:03d}",
                "source": _source_from_url(url),
                "title": str(ev.get("title") or ""),
                "url": url or None,
                "content": str(ev.get("content") or ""),
                "retrieval_score": ev.get("score"),
            }
        )
    return converted


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/futurex_past_yes_no.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/futurex_past_yes_no_with_evidence.jsonl"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/evidences_futurex"))
    parser.add_argument("--api-key", default=os.getenv("TAVILY_API_KEY", ""))
    parser.add_argument("--start-row", type=int, default=1, help="1-based start row for collection")
    parser.add_argument("--limit", type=int, default=0, help="Rows to collect; 0 means all")
    parser.add_argument("--max-evidence-per-question", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent questions to collect (default: 4; use 1 for serial collection)",
    )
    parser.add_argument(
        "--evidence-cutoff-days",
        type=int,
        default=1,
        help="Search this many calendar days before resolution (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-api-key", default=os.getenv("CMU_API_KEY", ""))
    parser.add_argument("--llm-base-url", default=os.getenv("OPENAI_BASE_URL", "https://ai-gateway.andrew.cmu.edu"))
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing Tavily API key. Pass --api-key or set TAVILY_API_KEY.")

    try:
        from tavily import TavilyClient
    except Exception as exc:
        raise RuntimeError("tavily-python is not installed. Install via: pip install tavily-python") from exc

    if args.llm_api_key:
        openai_module = importlib.import_module("openai")
    else:
        raise ValueError("Missing --llm-api-key: LLM post-filter is required to prevent leakage.")

    if args.evidence_cutoff_days < 1:
        raise ValueError("--evidence-cutoff-days must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    rows = _read_jsonl(args.input)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    start_idx = max(args.start_row - 1, 0)
    end_idx = len(rows) if args.limit <= 0 else min(len(rows), start_idx + args.limit)
    print(f"total_rows={len(rows)} collect_start={start_idx + 1} collect_end={end_idx}")

    payloads: dict[int, dict[str, Any]] = {}
    work_items: list[tuple[int, dict[str, Any], Path]] = []
    skipped = 0
    for idx, item in enumerate(rows, start=1):
        qid = normalize_ws(str(item.get("qid") or f"row_{idx:04d}"))
        cache_path = args.cache_dir / _cache_filename(idx, qid)

        should_collect = start_idx < idx <= end_idx
        payload = None
        if cache_path.exists() and not args.overwrite:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            skipped += 1 if should_collect else 0
            payloads[idx] = payload
        elif should_collect:
            work_items.append((idx, item, cache_path))

    def collect_one(idx: int, item: dict[str, Any]) -> dict[str, Any]:
        task_tavily_client = TavilyClient(args.api_key)
        task_llm_client = openai_module.OpenAI(api_key=args.llm_api_key, base_url=args.llm_base_url)
        return collect_for_row(
            client=task_tavily_client,
            row_index=idx,
            row=_row_for_collection(item),
            sleep_seconds=args.sleep_seconds,
            llm_client=task_llm_client,
            llm_model=args.llm_model,
            evidence_cutoff_days=args.evidence_cutoff_days,
        )

    pending = {}
    processed = failed = 0
    fatal_tavily_error: tuple[int, Exception] | None = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, item, cache_path in work_items:
            future = executor.submit(collect_one, idx, item)
            pending[future] = (idx, item, cache_path)

        for future in as_completed(pending):
            idx, item, cache_path = pending[future]
            qid = normalize_ws(str(item.get("qid") or f"row_{idx:04d}"))
            try:
                payload = future.result()
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                payloads[idx] = payload
                processed += 1
                print(
                    f"[{idx}] ok: {cache_path.name} "
                    f"raw={payload['stats']['raw_result_count']} "
                    f"deduped={payload['stats']['deduped_result_count']} "
                    f"quarantined_hard_time={payload['stats']['hard_quarantined_late_date_count']} "
                    f"kept_leak_no={payload['stats']['llm_kept_leak_no_count']} "
                    f"rewritten_kept={payload['stats']['llm_rewritten_kept_count']} "
                    f"dropped_leak_yes={payload['stats']['llm_filtered_leak_count']}"
                )
            except TavilySearchError as exc:
                fatal_tavily_error = (idx, exc)
                for pending_future in pending:
                    pending_future.cancel()
                print(f"[{idx}] fatal Tavily failure; current row discarded: {exc}")
                break
            except Exception as exc:
                failed += 1
                err_path = args.cache_dir / f"row_{idx:04d}_ERROR.json"
                with err_path.open("w", encoding="utf-8") as f:
                    json.dump({"qid": qid, "error": str(exc)}, f, ensure_ascii=False, indent=2)
                print(f"[{idx}] failed: {exc}")

    if fatal_tavily_error is not None:
        idx, exc = fatal_tavily_error
        raise RuntimeError(f"Aborted after Tavily failure at row {idx}; no cache or output JSONL was written for that row.") from exc

    output_rows = []
    for idx, item in enumerate(rows, start=1):
        qid = normalize_ws(str(item.get("qid") or f"row_{idx:04d}"))
        payload = payloads.get(idx)
        enriched = dict(item)
        if payload is not None:
            enriched["evidence"] = _convert_evidence(qid, payload, args.max_evidence_per_question)
        output_rows.append(enriched)

    _write_jsonl(args.output, output_rows)
    print(
        "done "
        f"processed={processed} skipped={skipped} failed={failed} "
        f"output={args.output} cache_dir={args.cache_dir}"
    )


if __name__ == "__main__":
    main()
