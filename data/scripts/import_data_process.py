#!/usr/bin/env python3
"""Import data_process dataset into PolymktSim JSONL format.

Usage:
  python scripts/import_data_process.py \
    --data-process-root ../data_process \
    --output data/processed/markets_500.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse


def _source_from_url(url: str) -> str:
    if not url:
        return "web"
    try:
        host = urlparse(url).netloc.lower()
        return host or "web"
    except ValueError:
        return "web"


def _load_market_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    market_map: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            market_id = (row.get("id") or "").strip()
            if market_id:
                market_map[market_id] = row
    return market_map


def _convert_one_question(payload: dict, row: dict[str, str] | None, max_evidence: int) -> dict:
    market_id = str(payload.get("market_id") or (row or {}).get("id") or "unknown")
    question = str(payload.get("question") or (row or {}).get("question") or "").strip()

    evidences = payload.get("evidences") or []
    if max_evidence > 0:
        evidences = evidences[:max_evidence]

    converted_evidence: list[dict] = []
    for i, ev in enumerate(evidences, 1):
        if not isinstance(ev, dict):
            continue
        url = str(ev.get("url") or "")
        converted_evidence.append(
            {
                "doc_id": f"{market_id}_doc_{i:03d}",
                "source": _source_from_url(url),
                "title": str(ev.get("title") or ""),
                "url": url or None,
                "content": str(ev.get("content") or ""),
                "retrieval_score": ev.get("score"),
            }
        )

    return {
        "qid": f"mkt_{market_id}",
        "question": question,
        "evidence": converted_evidence,
        "outcome": None,
        "resolution_date": payload.get("resolution_date") or (row or {}).get("endDateIso") or (row or {}).get("endDate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert data_process outputs into PolymktSim JSONL")
    parser.add_argument(
        "--data-process-root",
        type=Path,
        default=Path("../data_process"),
        help="Path to data_process root containing outputs/ and evidences/",
    )
    parser.add_argument(
        "--market-csv",
        type=str,
        default="outputs/final_markets_500.csv",
        help="CSV path relative to data-process-root",
    )
    parser.add_argument(
        "--evidence-dir",
        type=str,
        default="evidences",
        help="Evidence directory path relative to data-process-root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/markets_500.jsonl"),
        help="Output JSONL path in PolymktSim",
    )
    parser.add_argument(
        "--max-evidence-per-question",
        type=int,
        default=20,
        help="Keep at most N evidence items per question; <=0 means keep all",
    )
    args = parser.parse_args()

    data_root = args.data_process_root.resolve()
    market_csv = (data_root / args.market_csv).resolve()
    evidence_dir = (data_root / args.evidence_dir).resolve()

    if not market_csv.exists():
        raise FileNotFoundError(f"Market CSV not found: {market_csv}")
    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")

    market_rows = _load_market_rows(market_csv)
    evidence_files = sorted(evidence_dir.glob("row_*.json"))
    if not evidence_files:
        raise FileNotFoundError(f"No evidence files found under: {evidence_dir}")

    converted: list[dict] = []
    skipped = 0
    for path in evidence_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        market_id = str(payload.get("market_id") or "").strip()
        row = market_rows.get(market_id)

        q = _convert_one_question(payload, row, args.max_evidence_per_question)
        if not q["question"]:
            skipped += 1
            continue
        converted.append(q)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(converted)} questions to {args.output}")
    print(f"Skipped {skipped} rows with empty question")


if __name__ == "__main__":
    main()
